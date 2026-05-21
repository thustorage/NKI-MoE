# coding=utf-8
"""Qwen3-MoE entrypoint with a Qwen-aware mega token-generation path."""

from typing import Optional, Tuple
import warnings

import torch
import torch.nn.functional as F

from . import qwen_with_nki_original as base
from .qwen_with_nki_original import (
    Qwen3MoeInferenceConfig,
    convert_qwen3_moe_hf_to_neuron_state_dict,
)
from transformers import Qwen3MoeForCausalLM
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM

from neuronx_distributed_inference.modules.attention.utils import manual_softmax
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from .qwen_moe_tkg_mega import QwenMoeTKGMegaRunner

__all__ = ["NeuronQwen3MoeForCausalLM"]


class NeuronQwen3MoEFastTokenGenAttention(base.NeuronQwen3MoEAttention):
    """Qwen3-MoE attention with a narrow hook for batch-1 decode attention."""

    def apply_rotary_embedding(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        position_ids: torch.Tensor,
        cos_cache: Optional[torch.Tensor],
        sin_cache: Optional[torch.Tensor],
        use_polar_compatible_rope: bool,
    ):
        if use_polar_compatible_rope or self.rotary_emb is None:
            return super().apply_rotary_embedding(
                Q,
                K,
                V,
                position_ids,
                cos_cache,
                sin_cache,
                use_polar_compatible_rope,
            )

        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)

        if Q.shape[0] != 1 or Q.shape[2] != 1 or K.shape[0] != 1 or K.shape[2] != 1:
            return super().apply_rotary_embedding(
                Q,
                K,
                V,
                position_ids,
                cos_cache,
                sin_cache,
                use_polar_compatible_rope,
            )

        q_heads = Q.shape[1]
        qk = torch.cat((Q, K), dim=1)
        half_dim = qk.shape[-1] // 2
        qk_rotated_half = torch.cat((-qk[..., half_dim:], qk[..., :half_dim]), dim=-1)
        cos = cos_cache.unsqueeze(1)
        sin = sin_cache.unsqueeze(1)
        qk = (qk * cos) + (qk_rotated_half * sin)
        Q, K = torch.split(qk, [q_heads, K.shape[1]], dim=1)
        return Q, K, cos_cache, sin_cache

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        skip_rope=False,
        residual=None,
        use_polar_compatible_rope=False,
    ):
        return super().prep_qkv_tensors(
            position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            skip_rope=skip_rope,
            residual=residual,
            use_polar_compatible_rope=use_polar_compatible_rope,
        )

    def _can_use_fast_qwen_tkg_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor,
        active_mask: Optional[torch.Tensor],
        is_prefix_caching: bool,
    ) -> bool:
        # return False
        if is_prefix_caching:
            return False
        if past_key_value is None or position_ids is None:
            return False
        if self.get_learned_sinks() is not None:
            return False
        if Q.shape[0] != 1 or Q.shape[2] != 1:
            return False
        if K.shape[0] != 1 or K.shape[1] != 1 or K.shape[2] != 1:
            return False
        if V.shape[0] != 1 or V.shape[1] != 1 or V.shape[2] != 1:
            return False
        if past_key_value[0].shape[0] != 1 or past_key_value[0].shape[1] != 1:
            return False
        if past_key_value[1].shape[0] != 1 or past_key_value[1].shape[1] != 1:
            return False
        return Q.shape[0] == 1 and Q.shape[2] == 1

    def _flatten_attention_mask_for_fast_qwen_tkg(
        self,
        attention_mask: torch.Tensor,
        num_heads: int,
        prior_len: int,
    ) -> torch.Tensor:
        mask = attention_mask[:, :, :, :prior_len]
        if (
            prior_len > attention_mask.shape[-1]
            and self.neuron_config.apply_seq_ids_mask
        ):
            mask = F.pad(mask, (0, prior_len - attention_mask.shape[-1]), "constant", 0)

        if mask.shape[1] == 1:
            return mask[0, 0, 0, :].reshape(1, prior_len).expand(num_heads, prior_len)
        return mask[0, :, 0, :]

    def _compute_for_token_gen_fast(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        active_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_heads = Q.shape[1]
        head_dim = Q.shape[-1]

        q_2d = Q[0, :, 0, :]
        if not self.k_cache_transposed:
            k_prior_2d = past_key_value[0][0, 0, :, :]
        else:
            k_prior_2d = past_key_value[0][0, 0, :, :].transpose(0, 1)
        v_prior_2d = past_key_value[1][0, 0, :, :]
        k_active_2d = K[0, 0, :, :]
        v_active_2d = V[0, 0, :, :]

        prior_scores = (
            torch.matmul(q_2d, k_prior_2d.transpose(0, 1)) / self.softmax_scale
        )
        prior_mask = self._flatten_attention_mask_for_fast_qwen_tkg(
            attention_mask,
            num_heads,
            prior_scores.shape[-1],
        )
        prior_scores = torch.where(
            prior_mask,
            prior_scores,
            torch.finfo(prior_scores.dtype).min,
        )

        active_scores = (
            torch.matmul(q_2d, k_active_2d.transpose(0, 1)) / self.softmax_scale
        )

        prior_scores = prior_scores.to(torch.float32)
        active_scores = active_scores.to(torch.float32)
        softmax_prior, softmax_active = manual_softmax(
            prior_scores, active_scores, False
        )
        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(
            Q.dtype
        )
        attn_prior = torch.matmul(softmax_prior, v_prior_2d)
        attn_active = torch.matmul(softmax_active, v_active_2d)
        attn_output = attn_prior + attn_active
        return attn_output.reshape(1, num_heads, 1, head_dim)

    def compute_for_token_gen(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        active_mask: Optional[torch.Tensor],
        is_prefix_caching: bool = False,
    ) -> torch.Tensor:
        if self._can_use_fast_qwen_tkg_attention(
            Q,
            K,
            V,
            position_ids,
            past_key_value,
            attention_mask,
            active_mask,
            is_prefix_caching,
        ):
            return self._compute_for_token_gen_fast(
                Q,
                K,
                V,
                position_ids,
                past_key_value,
                attention_mask,
                active_mask,
            )

        return super().compute_for_token_gen(
            Q,
            K,
            V,
            position_ids,
            past_key_value,
            attention_mask,
            active_mask,
            is_prefix_caching=is_prefix_caching,
        )


class NeuronQwen3MoeDecoderLayer(base.NeuronQwen3MoeDecoderLayer):
    """Decoder layer that routes token generation through the mega runner."""

    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = NeuronQwen3MoEFastTokenGenAttention(config=config)
        self.config = config
        self.layer_idx = layer_idx
        self._mega_runner = QwenMoeTKGMegaRunner(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        if self._mega_runner.can_run(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        ):
            return self._mega_runner.forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                padding_mask=padding_mask,
                **kwargs,
            )

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            padding_mask=padding_mask,
            **kwargs,
        )


class NeuronQwen3MoeModel(base.NeuronQwen3MoeModel):
    """Model that swaps decoder layers to the mega decode path."""

    def init_model(self, config: Qwen3MoeInferenceConfig):
        print("[qwen_with_nki_mega] Qwen3-MoE mega decode path enabled")
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = base.ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = torch.nn.ModuleList(
            [
                NeuronQwen3MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = base.get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = base.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


class NeuronQwen3MoeForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronQwen3MoeModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Qwen3MoeForCausalLM.from_pretrained(model_path, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return Qwen3MoeInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: Qwen3MoeInferenceConfig
    ) -> dict:
        return convert_qwen3_moe_hf_to_neuron_state_dict(state_dict, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self):
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = (
                "-O3" if self.neuron_config.moe_ep_degree > 1 else "-O1"
            )
        compiler_args = f"--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer {optimization_level}"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --disable-prefetch-block-tensors '"
        compiler_args += " --auto-cast=none"
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        if self.neuron_config.scratchpad_page_size:
            compiler_args += f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size} "
        if self.neuron_config.attn_block_tkg_nki_kernel_enabled:
            assert (
                self.neuron_config.attn_block_tkg_nki_kernel_cascaded_attention
            ), "If using attn_block_tkg_nki_kernel_enabled for Qwen3MoE you must also use attn_block_tkg_nki_kernel_cascaded_attention"
            self.neuron_config.pre_rope_rmsnorm = True
            compiler_args += " --internal-max-instruction-limit=15000000"
        return compiler_args
