"""Qwen3-MoE token-generation with MoE megakernel"""

from typing import Optional, Tuple

import torch

from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    EPDispatchOption,
)

from .moe.moe_selective import moe_selective_v3
from .utils.common_types import ActFnType, RouterActFnType
from neuronx_distributed.parallel_layers import mappings, parallel_state

_ROUTER_ACT_FN_MAP = {
    "sigmoid": RouterActFnType.SIGMOID,
    "softmax": RouterActFnType.SOFTMAX,
}


class QwenMoeTKGMegaRunner:
    """Runs the decode block with full mega plus three baseline swap variants."""

    def __init__(self, layer):
        self.layer = layer
        self._input_layernorm = layer.input_layernorm
        self._post_attention_layernorm = layer.post_attention_layernorm
        self._self_attn = layer.self_attn
        self._mlp = layer.mlp
        self._moe_fused_nki_kernel_enabled = bool(
            getattr(layer, "moe_fused_nki_kernel_enabled", False)
        )
        neuron_config = getattr(getattr(layer, "config", None), "neuron_config", None)
        self._attention_core = self._get_attention_core()
        self._moe_core = self._get_moe_core()
        self._tp_degree = int(getattr(neuron_config, "tp_degree", 1) or 1)
        self._replica_groups = (
            (tuple(range(self._tp_degree)),) if self._tp_degree > 1 else None
        )
        self._kernel_dtype = self._get_kernel_dtype(neuron_config)
        self._mega_reason = self._get_mega_disable_reason()
        self._mega_supported = self._mega_reason is None
        self._start_marker = ModuleMarkerStartWrapper()
        self._end_marker = ModuleMarkerEndWrapper()

    def _get_attention_core(self):
        return (
            self._self_attn.attn
            if hasattr(self._self_attn, "attn")
            else self._self_attn
        )

    def _get_moe_core(self):
        candidates = [
            self._mlp,
            getattr(self._mlp, "moe", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if hasattr(candidate, "router") and hasattr(candidate, "expert_mlps"):
                return candidate
        return None

    def _get_kernel_dtype(self, neuron_config):
        if self._attention_core is not None and hasattr(
            self._attention_core, "torch_dtype"
        ):
            dtype = self._attention_core.torch_dtype
            if dtype is not None:
                return dtype
        if neuron_config is not None and hasattr(neuron_config, "torch_dtype"):
            dtype = neuron_config.torch_dtype
            if dtype is not None:
                return dtype
        return torch.bfloat16

    def _get_mega_disable_reason(self) -> Optional[str]:
        attn = self._attention_core
        if attn is None:
            return "attention core is unavailable"
        # if not hasattr(attn, "get_qkv_proj") or not hasattr(attn.get_qkv_proj(), "Wqkv"):
        #     return "attention core does not expose fused Wqkv weights"
        if not hasattr(attn, "get_o_proj") or not hasattr(attn.get_o_proj(), "o_proj"):
            return "attention core does not expose output projection weights"
        if not hasattr(attn, "q_layernorm") or not hasattr(attn.q_layernorm, "weight"):
            return "attention core does not expose q_layernorm weights"
        if not hasattr(attn, "k_layernorm") or not hasattr(attn.k_layernorm, "weight"):
            return "attention core does not expose k_layernorm weights"
        if not hasattr(attn, "rotary_emb"):
            return "attention core does not expose rotary_emb"
        if bool(getattr(attn, "sequence_parallel_enabled", False)):
            return "mega attention currently requires sequence_parallel_enabled=False"
        if (
            getattr(attn, "ep_dispatch_cc_option", EPDispatchOption.AR_AG)
            != EPDispatchOption.AR_AG
        ):
            return (
                "mega attention currently only supports EPDispatchOption.AR_AG, "
                f"got {getattr(attn, 'ep_dispatch_cc_option', None)}"
            )
        if int(getattr(attn, "num_key_value_heads", 0) or 0) != 1:
            return (
                "attention_block_tkg_v2 currently requires per-rank num_key_value_heads == 1, "
                f"got {getattr(attn, 'num_key_value_heads', None)}"
            )

        moe = self._moe_core
        if moe is None:
            return "MoE core does not expose router/expert_mlps"

        router = getattr(moe, "router", None)
        expert_mlps = getattr(moe, "expert_mlps", None)
        if router is None or expert_mlps is None:
            return "MoE core is missing router or expert_mlps"
        if getattr(router, "act_fn", None) not in _ROUTER_ACT_FN_MAP:
            return f"unsupported router activation {getattr(router, 'act_fn', None)}"
        top_k = int(getattr(router, "top_k", 0) or 0)
        if top_k <= 0:
            return f"invalid router top_k={top_k}"
        if top_k % 2 != 0:
            return f"top_k={top_k} must be divisible by 2 for current expert sharding"
        if bool(getattr(router, "apply_act_fn_over_topk", False)):
            return "moe_selective_v3 megakernel requires apply_act_fn_over_topk=False"
        if not hasattr(router, "weight_T") and not (
            hasattr(router, "linear_router") and hasattr(router.linear_router, "weight")
        ):
            return "router does not expose transposed or raw projection weights"

        mlp_op = getattr(expert_mlps, "mlp_op", None)
        if mlp_op is None:
            return "expert_mlps does not expose mlp_op"
        if not hasattr(mlp_op, "gate_up_proj") or not hasattr(
            mlp_op.gate_up_proj, "weight"
        ):
            return "expert_mlps.mlp_op does not expose gate_up_proj weights"
        if not hasattr(mlp_op, "down_proj") or not hasattr(mlp_op.down_proj, "weight"):
            return "expert_mlps.mlp_op does not expose down_proj weights"

        return None

    def can_run(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_value: Optional[Tuple[torch.Tensor]],
    ) -> bool:
        if not self._mega_supported:
            return False
        if past_key_value is None or len(past_key_value) != 2:
            return False
        if attention_mask is None or position_ids is None:
            return False
        if hidden_states.shape[1] != 1:
            return False
        return hidden_states.shape[0] * hidden_states.shape[1] <= 128

    def _ensure_mega_supported(self, caller: str):
        if not self._mega_supported:
            raise RuntimeError(
                f"{caller} reached without mega-kernel support: {self._mega_reason}"
            )

    def _run_baseline_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ):
        qkv_fused_rmsnorm = None
        if self._input_layernorm:
            if bool(getattr(self.layer, "qkv_kernel_enabled", False)) and bool(
                getattr(self.layer, "qkv_kernel_fused_rmsnorm", False)
            ):
                qkv_fused_rmsnorm = self._input_layernorm
            else:
                hidden_states = self._input_layernorm(hidden_states)

        return self._self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )

    def _build_moe_kernel_inputs(self):
        moe = self._moe_core
        router = moe.router
        expert_mlps = moe.expert_mlps
        mlp_op = expert_mlps.mlp_op

        router_weight_t = (
            router.weight_T.data
            if hasattr(router, "weight_T")
            else router.linear_router.weight.data.transpose(0, 1).contiguous()
        )
        # router_weight_t = router.linear_router.weight.data # note we don't transpose the router weights here since the kernel expects the non-transposed layout, and we will transpose inside the kernel if needed. This is to avoid an extra transpose on the host side when the router weights are already in the correct layout.
        # if router_weight_t.dtype != torch.float32:
        #     router_weight_t = router_weight_t.float()

        router_bias = None
        if hasattr(router, "linear_router") and router.linear_router.bias is not None:
            router_bias = router.linear_router.bias.data.float().unsqueeze(0)

        gate_up_weight = mlp_op.gate_up_proj.weight.data
        num_experts, hidden_size, two_i = gate_up_weight.shape
        gate_up_weight_4d = gate_up_weight.view(num_experts, hidden_size, 2, two_i // 2)

        return {
            "gamma_mlp": self._post_attention_layernorm.weight.data.unsqueeze(0),
            "router_weight_t": router_weight_t,
            "router_bias": router_bias,
            "gate_up_weight_4d": gate_up_weight_4d,
            "down_weight": mlp_op.down_proj.weight.data,
            "top_k": int(router.top_k),
            "norm_topk_prob": bool(
                getattr(
                    expert_mlps.routed_experts_mlp_config,
                    "normalize_top_k_affinities",
                    False,
                )
            ),
            "router_act_fn": _ROUTER_ACT_FN_MAP[router.act_fn],
        }

    def _launch_moe_only_mega_kernelv2(self, kernel_input, moe_inputs):
        """Launch MoE kernel using moe_selective_v3 directly (mirroring moe_fused_tkg.py pattern)."""
        # moe_selective_v3 expects [T, H] 2D input
        hidden_input_2d = kernel_input.reshape(-1, kernel_input.shape[-1])
        out = moe_selective_v3[2](
            hidden_input=hidden_input_2d,
            gamma=moe_inputs["gamma_mlp"],
            router_weights=moe_inputs["router_weight_t"],
            expert_gate_up_weights=moe_inputs["gate_up_weight_4d"],
            expert_down_weights=moe_inputs["down_weight"],
            router_bias=None,
            eps=float(self.layer.config.rms_norm_eps),
            top_k=8,
            router_act_fn=moe_inputs["router_act_fn"],
            norm_topk_prob=True,
            activation_fn=ActFnType.SiLU,
        )
        return out, kernel_input

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self._ensure_mega_supported("QwenMoeTKGMegaRunner.forward")

        residual = hidden_states
        moe_inputs = self._build_moe_kernel_inputs()
        attn_input = self._start_marker(hidden_states)
        attn_hidden, present_key_value, cos_cache, sin_cache = (
            self._run_baseline_attention(
                hidden_states=attn_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )
        )
        attn_residual = residual + attn_hidden
        moe_local, residual = self._launch_moe_only_mega_kernelv2(
            attn_residual, moe_inputs
        )
        output = mappings.reduce_from_tensor_model_parallel_region(
            moe_local, process_group=parallel_state.get_tensor_model_parallel_group()
        )
        output = output.unsqueeze(0)
        output = output + residual
        layer_output = output
        assert layer_output.dtype == torch.bfloat16
        layer_output = self._end_marker(layer_output)
        return (
            layer_output,
            present_key_value,
            cos_cache,
            sin_cache,
            None,
        )
