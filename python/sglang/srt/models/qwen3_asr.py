import logging
import math
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

from python.sglang.srt.configs.qwen3_asr import Qwen3ASRAudioEncoderConfig, Qwen3ASRConfig, Qwen3ASRThinkerConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def _get_feat_extract_output_lengths(
    input_lengths: torch.Tensor, n_window: int = 50
) -> torch.Tensor:
    """Compute the output length of the audio encoder from input mel lengths."""
    chunk_size = n_window * 2
    input_lengths_leave = input_lengths % chunk_size
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        (feat_lengths - 1) // 2 + 1 - 1
    ) // 2 + 1 + (input_lengths // chunk_size) * 13
    return output_lengths


class SinusoidsPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for audio encoder.
    Copy from [qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/vllm_backend/qwen3_asr.py#L127C1-L154C53)
    
    """

    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale

        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")

        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        positional_embedding = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
        )
        self.register_buffer(
            "positional_embedding", positional_embedding, persistent=False
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :] # type: ignore


class Qwen3ASRAudioAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got embed_dim={self.embed_dim}, num_heads={self.num_heads})"
            )

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.size(0)

        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        if cu_seqlens is not None and len(cu_seqlens) > 2:
            attention_mask = torch.full(
                (seq_length, seq_length),
                torch.finfo(q.dtype).min,
                device=q.device,
                dtype=q.dtype,
            )
            for i in range(1, len(cu_seqlens)):
                s, e = cu_seqlens[i - 1].item(), cu_seqlens[i].item()
                attention_mask[s:e, s:e] = 0
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = None

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, scale=self.scaling
        )
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class Qwen3ASRAudioEncoderLayer(nn.Module):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        return hidden_states


class Qwen3ASRAudioEncoder(nn.Module):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

        downsample_hidden_size = config.downsample_hidden_size
        output_dim = config.output_dim

        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )

        self.layers = nn.ModuleList(
            [Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.ln_post = nn.LayerNorm(embed_dim)

        # Conv2D downsampling
        self.conv2d1 = nn.Conv2d(1, downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            downsample_hidden_size, downsample_hidden_size, 3, 2, padding=1
        )
        self.conv2d3 = nn.Conv2d(
            downsample_hidden_size, downsample_hidden_size, 3, 2, padding=1
        )

        mel_after_conv = (((self.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(
            downsample_hidden_size * mel_after_conv, embed_dim, bias=False
        )

        # Projection
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(embed_dim, output_dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv2d1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.conv2d1.weight.device

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Encode audio mel spectrogram features.

        Args:
            input_features: Mel spectrogram of shape (num_mel_bins, total_frames)
            feature_lens: Tensor of shape (batch_size,) with actual frame lengths

        Returns:
            Audio embeddings of shape (total_output_tokens, output_dim)
        """
        chunk_size = self.n_window * 2
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens, self.n_window)
        chunk_num = torch.ceil(feature_lens / chunk_size).long()

        chunk_lengths = torch.tensor(
            [chunk_size] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % chunk_size
        chunk_lengths[chunk_lengths == 0] = chunk_size

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(
            chunk_lengths, self.n_window
        )
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [
                torch.ones(length, dtype=torch.bool, device=padded_feature.device)
                for length in feature_lens_after_cnn
            ],
            batch_first=True,
        )

        padded_feature = padded_feature.unsqueeze(1)

        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding

        hidden_states = padded_embed[padded_mask_after_cnn]

        # Build cu_seqlens for windowed attention
        cu_chunk_lens: list[int] = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (
            self.n_window_infer // (self.n_window * 2)
        )
        for cnn_len in aftercnn_lens:
            cnn_len_val = cnn_len.item()
            cu_chunk_lens += [window_aftercnn] * (cnn_len_val // window_aftercnn)
            remainder = cnn_len_val % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(
            cu_chunk_lens, device=aftercnn_lens.device
        ).cumsum(-1, dtype=torch.int32)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


class Qwen3ASRForConditionalGeneration(nn.Module):

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3ASRConfig, # type: ignore
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        thinker_config: Qwen3ASRThinkerConfig = config.thinker_config
        audio_config = thinker_config.audio_config
        text_config = thinker_config.text_config

        if audio_config is None:
            raise ValueError("Qwen3-ASR config must have thinker_config.audio_config")
        if text_config is None:
            raise ValueError("Qwen3-ASR config must have thinker_config.text_config")

        self.audio_token_id = thinker_config.audio_token_id

        self.audio_tower = Qwen3ASRAudioEncoder(audio_config)
        self.language_model = Qwen3ForCausalLM(
            text_config, quant_config, prefix=add_prefix("language_model", prefix)
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        if mm_inputs and mm_inputs.mm_items:
            audio_token_id = mm_inputs.audio_token_id
            if audio_token_id is not None:
                audio_items = [
                    item for item in mm_inputs.mm_items if item.is_audio()
                ]
                runs: list[tuple[int, int]] = []
                i = 0
                while i < len(input_ids):
                    if input_ids[i] == audio_token_id:
                        start = i
                        while i < len(input_ids) and input_ids[i] == audio_token_id:
                            i += 1
                        runs.append((start, i - 1))
                    else:
                        i += 1
                for idx, item in enumerate(audio_items):
                    if idx < len(runs):
                        item.offsets = [runs[idx]]
                    else:
                        item.offsets = []

        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        all_features = []
        for item in items:
            input_features = item.feature.to(
                device=self.audio_tower.device, dtype=self.audio_tower.dtype
            )
            feature_lens = item.audio_feature_lens.to(device=self.audio_tower.device)

            # Slice to actual audio length (input may be padded to max_length)
            actual_len = feature_lens[0].item()
            if input_features.ndim == 2 and input_features.shape[-1] > actual_len:
                input_features = input_features[:, :actual_len]

            audio_embeds = self.audio_tower(input_features, feature_lens=feature_lens)
            all_features.append(audio_embeds)

        return torch.cat(all_features, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            # Strip the outer "thinker." prefix from all weights
            if name.startswith("thinker."):
                name = name[len("thinker.") :]

            # Map thinker weights to sglang model structure:
            #   thinker.audio_tower.* -> audio_tower.*
            #   thinker.model.* -> language_model.model.*
            #   thinker.lm_head.* -> language_model.lm_head.*
            if name.startswith("model."):
                name = "language_model." + name
            elif name.startswith("lm_head."):
                name = "language_model." + name

            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            text_config = getattr(
                getattr(self.config, "thinker_config", self.config),
                "text_config",
                self.config,
            )
            if getattr(text_config, "tie_word_embeddings", False):
                if "lm_head.weight" in name:
                    continue

            # Skip audio tower weights for stacked params
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "audio_tower" in name:
                    continue
                name_tmp = name.replace(weight_name, param_name)

                if name_tmp.endswith(".bias") and name_tmp not in params_dict:
                    continue
                if name_tmp not in params_dict:
                    break
                param = params_dict[name_tmp]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = Qwen3ASRForConditionalGeneration
