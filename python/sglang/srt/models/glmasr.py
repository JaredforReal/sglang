# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modeling from:
# ./llama.py and
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glmasr/modular_glmasr.py
"""Inference-only GLM-ASR-HF model compatible with HuggingFace weights."""

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GlmAsrConfig, GlmAsrEncoderConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.glmasr.modeling_glmasr import GlmAsrMultiModalProjector

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
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
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to partial dimensions of q and k.

    GLM-ASR uses partial rotary embedding where only the first `rotary_dim`
    dimensions are rotated.

    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        cos: Cosine tensor of shape [seq_len, rotary_dim]
        sin: Sine tensor of shape [seq_len, rotary_dim]

    Returns:
        Rotated query and key tensors.
    """
    # cos/sin shape: [seq_len, rotary_dim] -> [1, seq_len, 1, rotary_dim]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first rotary_dim dimensions
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class GlmAsrRotaryEmbedding(nn.Module):
    """Rotary position embedding for GLM-ASR encoder."""

    def __init__(self, config: GlmAsrEncoderConfig):
        super().__init__()
        self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.head_dim = config.hidden_size // config.num_attention_heads
        # GLM-ASR uses partial rotary factor
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        self.rotary_ndims = int(self.head_dim * partial_rotary_factor)

        # Get rope_theta from config - try multiple attribute names for compatibility
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = getattr(config, "default_theta", None)
        if (
            rope_theta is None
            and hasattr(config, "rope_scaling")
            and config.rope_scaling
        ):
            rope_theta = config.rope_scaling.get("rope_theta", 10000.0)
        if rope_theta is None:
            rope_theta = 10000.0

        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, self.rotary_ndims, 2, dtype=torch.float32)
                / self.rotary_ndims
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary position embeddings.

        Args:
            x: Input tensor (used for device and dtype)
            position_ids: Position indices [batch_size, seq_len]

        Returns:
            cos, sin tensors of shape [seq_len, rotary_dim]
        """
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().squeeze(0)  # [seq_len, rotary_dim]
        sin = emb.sin().squeeze(0)  # [seq_len, rotary_dim]

        return cos.to(x.dtype), sin.to(x.dtype)


class GlmAsrMLP(nn.Module):
    """MLP layer for GLM-ASR encoder with tensor parallelism support."""

    def __init__(
        self,
        config: GlmAsrEncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act

        self.fc1 = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.act_fn = get_act_fn(self.hidden_act)
        self.fc2 = RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class GlmAsrAttention(nn.Module):
    """Multi-head attention for GLM-ASR encoder with tensor parallelism.

    Key differences from standard attention:
    - Uses partial rotary position embeddings
    - Non-causal attention (encoder)
    - q_proj has bias, k_proj has no bias, v_proj has bias
    """

    def __init__(
        self,
        config: GlmAsrEncoderConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = self.num_heads // self.tp_size
        self.num_kv_heads_per_partition = max(1, self.num_kv_heads // self.tp_size)

        # QKV projection with tensor parallelism
        # Note: In GLM-ASR, q has bias, k has no bias, v has bias
        # We use QKVParallelLinear with bias=True for simplicity,
        # the k bias will be zeros and handled by weight loading
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for attention.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_embeddings: (cos, sin) tuple for rotary embeddings
            attention_mask: [batch_size, 1, seq_len, seq_len] or similar

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)

        # Split into q, k, v
        q_size = self.num_heads_per_partition * self.head_dim
        kv_size = self.num_kv_heads_per_partition * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape to [batch, seq, heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads_per_partition, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads_per_partition, self.head_dim)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        q, k = apply_partial_rotary_pos_emb(q, k, cos, sin)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (non-causal for encoder)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # Reshape back: [batch, seq, heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class GlmAsrEncoderLayer(nn.Module):
    """Single encoder layer for GLM-ASR with pre-norm architecture."""

    def __init__(
        self,
        config: GlmAsrEncoderConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GlmAsrAttention(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = GlmAsrMLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-norm and residual connections.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_embeddings: (cos, sin) tuple for rotary embeddings
            attention_mask: Attention mask

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GlmAsrEncoder(nn.Module):
    """GLM-ASR audio encoder with Whisper-style convolutions and transformer layers.

    This encoder processes mel-spectrogram features through:
    1. Two 1D convolutions for initial feature extraction
    2. Multiple transformer encoder layers with rotary position embeddings
    3. Final layer normalization
    """

    def __init__(
        self,
        config: GlmAsrEncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        # Convolutional layers for mel-spectrogram processing
        self.conv1 = nn.Conv1d(
            in_channels=config.num_mel_bins,
            out_channels=config.hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                GlmAsrEncoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size)

        # Rotary position embeddings
        self.rotary_emb = GlmAsrRotaryEmbedding(config=config)

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the encoder parameters."""
        return self.conv1.weight.dtype

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutput:
        """Forward pass through the audio encoder.

        Args:
            input_features: Mel-spectrogram features [batch, num_mel_bins, time]
            attention_mask: Attention mask [batch, 1, seq_len, seq_len]

        Returns:
            BaseModelOutput with last_hidden_state of shape [batch, seq_len, hidden_size]
        """
        # Convolutional feature extraction
        # input_features: [batch, num_mel_bins, time]
        hidden_states = F.gelu(self.conv1(input_features))
        hidden_states = F.gelu(self.conv2(hidden_states))

        # Transpose for transformer: [batch, hidden_size, time] -> [batch, time, hidden_size]
        hidden_states = hidden_states.transpose(1, 2)

        # Compute position embeddings
        batch_size, seq_len, _ = hidden_states.shape
        position_ids = torch.arange(
            seq_len, device=hidden_states.device, dtype=torch.long
        ).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Apply transformer encoder layers
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        # Final layer normalization
        hidden_states = self.norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class GlmAsrForConditionalGeneration(nn.Module):
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
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: GlmAsrConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if getattr(self.config, "audio_config", None) is None:
            self.config.audio_config = GlmAsrEncoderConfig(self.config._name_or_path)

        self.audio_tower = GlmAsrEncoder(
            config.audio_config,
        )
        self.multi_modal_projector = GlmAsrMultiModalProjector(config)
        self.language_model = LlamaForCausalLM(
            config.text_config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # Extract audio features from input items
        features = [item.feature for item in items]

        # Check if all have same shape
        shapes = [f.shape for f in features]
        max_len = max(s[-1] for s in shapes)
        device = self.audio_tower.conv1.weight.device
        dtype = self.audio_tower.dtype

        if all(s[-1] == max_len for s in shapes):
            input_features = torch.cat(features, dim=0).to(device=device, dtype=dtype)
            attention_mask = None
            seq_lens = None
        else:
            # Pad
            batch_size = len(features)
            num_mel_bins = shapes[0][
                1
            ]  # [1, num_mel_bins, time] or [num_mel_bins, time]?
            # Existing code: torch.cat([item.feature], dim=0)
            # If item.feature is [num_mel_bins, time], cat(dim=0) -> [N*num_mel_bins, time] -> WRONG for conv1d
            # conv1d expects [batch, channels, time].
            # So item.feature MUST be [1, num_mel_bins, time].
            # Let's assume item.feature is [1, num_mel_bins, time].

            num_mel_bins = shapes[0][1]
            input_features = torch.zeros(
                (batch_size, num_mel_bins, max_len), dtype=dtype, device=device
            )

            seq_lens = []
            for i, f in enumerate(features):
                # f is [1, bins, time]
                l = f.shape[-1]
                input_features[i, :, :l] = f.to(device=device, dtype=dtype)
                # Calculate seq len after convs
                # conv1: l
                # conv2: floor((l+1)/2)
                seq_len = (l + 1) // 2
                seq_lens.append(seq_len)

            max_seq_len = (max_len + 1) // 2

            # Create float mask: 0.0 for valid, -inf for padding
            attention_mask = torch.full(
                (batch_size, 1, max_seq_len, max_seq_len),
                torch.finfo(dtype).min,
                dtype=dtype,
                device=device,
            )
            for i, l in enumerate(seq_lens):
                attention_mask[i, :, :l, :l] = 0.0

        audio_embeds = self.audio_tower(
            input_features, attention_mask=attention_mask
        ).last_hidden_state

        if seq_lens is not None:
            # Extract valid tokens
            valid_embeds = []
            for i, l in enumerate(seq_lens):
                valid_embeds.append(audio_embeds[i, :l, :])
            audio_embeds = torch.cat(valid_embeds, dim=0)
        else:
            audio_embeds = audio_embeds.reshape(
                -1, self.config.audio_config.intermediate_size
            )

        audio_embeds = self.multi_modal_projector(audio_embeds)

        return audio_embeds

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Stacked params mapping for language model (LLM)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Stacked params mapping for audio encoder
        audio_stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            # Handle audio_tower weights with separate mapping
            if "audio_tower" in name:
                # Handle stacked params for audio encoder
                handled = False
                for param_name, weight_name, shard_id in audio_stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name_tmp = name.replace(weight_name, param_name)

                    # Skip loading extra bias for GPTQ models.
                    if name_tmp.endswith(".bias") and name_tmp not in params_dict:
                        continue
                    if name_tmp not in params_dict:
                        continue
                    param = params_dict[name_tmp]
                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader is not None:
                        weight_loader(param, loaded_weight, shard_id)
                    handled = True
                    break

                if not handled:
                    # Direct loading for non-stacked params
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                continue

            # Handle language model weights
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_tmp = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name_tmp.endswith(".bias") and name_tmp not in params_dict:
                    continue
                param = params_dict[name_tmp]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = GlmAsrForConditionalGeneration
