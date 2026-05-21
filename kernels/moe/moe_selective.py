# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Selective-expert MoE token generation kernel v3.

Supports two modes:
  1. Pre-computed routing: expert_index and expert_affinities passed directly.
  2. Fused routing: router_weights provided; RMSNorm + Router Matmul + Softmax
     + TopK + L1 Norm computed inline, then MoE MLP.

Uses inline GEMV specializations."""

from typing import Optional

import nki
import nki.language as nl

from .moe_parameters import MLPExpertParameters, MLPParameters
from ..utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    NormType,
    RouterActFnType,
)

from .selective_expert_impl import _selective_expert_moe_tkg

P_MAX = 128


@nki.jit(grid=[2])
def moe_selective_v3(
    hidden_input: nl.ndarray,
    expert_gate_up_weights: nl.ndarray,
    expert_down_weights: nl.ndarray,
    expert_index: Optional[nl.ndarray] = None,
    expert_affinities: Optional[nl.ndarray] = None,
    expert_gate_up_bias: Optional[nl.ndarray] = None,
    expert_down_bias: Optional[nl.ndarray] = None,
    activation_fn: ActFnType = ActFnType.SiLU,
    output_dtype=None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    output_in_sbuf: bool = False,
    # --- Optional inline RMSNorm + Router TopK parameters ---
    gamma: Optional[nl.ndarray] = None,
    router_weights: Optional[nl.ndarray] = None,
    router_bias: Optional[nl.ndarray] = None,
    router_act_fn: RouterActFnType = RouterActFnType.SOFTMAX,
    router_pre_norm=False,
    router_mm_dtype=None,
    norm_topk_prob: bool = False,
    top_k: int = 1,
    eps: float = 1e-6,
    layer_idx: int = 0,
) -> nl.ndarray:
    """
    Selective-expert MoE MLP token generation kernel v3.

    Supports two modes:
      Mode 1 (pre-computed routing): Pass expert_index and expert_affinities.
      Mode 2 (fused routing): Pass router_weights (and optionally gamma for RMSNorm).
        Router matmul + activation + topk + L1 norm computed inline.

    Dimensions:
        T: Number of tokens (batch_size * seq_len, <= 128)
        H: Hidden dimension
        I: Intermediate dimension
        E_L: Number of local experts
        K: Top-k experts per token

    Args:
        hidden_input: [T, H] Input hidden states in HBM.
        expert_gate_up_weights: [E_L, H, 2, I] Fused gate and up projection weights.
        expert_down_weights: [E_L, H, I] Down projection weights (transposed layout).
        expert_index: [T, K] Top-K expert indices per token. Required for mode 1.
        expert_affinities: [T, E_L] Expert affinity weights per token. Required for mode 1.
        expert_gate_up_bias: [E_L, 2, I] Optional bias for gate/up projections.
        expert_down_bias: [E_L, H] Optional bias for down projection.
        activation_fn: Activation function type. (default: SiLU)
        output_dtype: Output tensor data type. Defaults to hidden_input.dtype.
        gamma: [1, H] RMSNorm weight. When provided with router_weights, enables
            fused RMSNorm before router matmul.
        router_weights: [H, E] Router weights for matmul. Enables fused routing mode.
        router_bias: [1, E] Optional router bias.
        router_act_fn: Router activation function (default: SOFTMAX).
        norm_topk_prob: L1 normalize top-K probabilities (default: False).
        top_k: Number of experts per token (default: 1). Used in fused routing mode.
        eps: RMSNorm epsilon (default: 1e-6).

    Returns:
        output: [T, H] bf16 — both cores reduce internally via sendrecv,
            core 0 writes the final result.
    """
    assert hidden_input is not None, "hidden_input must not be None"
    assert expert_gate_up_weights is not None, "expert_gate_up_weights must not be None"
    assert expert_down_weights is not None, "expert_down_weights must not be None"

    assert (
        router_weights is not None
    ), "router_weights must be provided for fused routing mode"
    assert (
        top_k is not None and top_k > 0
    ), f"top_k must be positive in fused routing mode, got {top_k}"
    # Fused routing mode: expert_index/affinities computed internally
    expert_params = MLPExpertParameters(
        expert_index=None,
        expert_affinities=None,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )

    output_dtype = hidden_input.dtype if output_dtype is None else output_dtype
    name_prefix = f"L{layer_idx}_"

    T, H = hidden_input.shape
    assert T <= 128, "currently only batch size * seq len <= 128 is supported"
    assert (
        expert_gate_up_weights.shape[1] == H
    ), f"expert_gate_up_weights hidden dim mismatch: expected {H}, got {expert_gate_up_weights.shape[1]}"
    assert (
        len(expert_down_weights.shape) == 3
    ), f"expert_down_weights must be rank-3 [E, I, H], got shape {expert_down_weights.shape}"
    assert (
        expert_down_weights.shape[2] == H
    ), f"expert_down_weights hidden dim mismatch: expected {H}, got {expert_down_weights.shape[2]}"
    if gamma is not None:
        assert (
            gamma.shape[-1] == H
        ), f"gamma hidden dim mismatch: expected {H}, got {gamma.shape[-1]}"
    if router_weights is not None:
        assert (
            router_weights.shape[0] == H
        ), f"router_weights hidden dim mismatch: expected {H}, got {router_weights.shape[0]}"

    normed_hbm = None
    mlp_hidden = hidden_input
    mlp_gamma = gamma

    mlp_params = MLPParameters(
        hidden_tensor=mlp_hidden,
        gate_proj_weights_tensor=expert_gate_up_weights,
        up_proj_weights_tensor=expert_gate_up_weights,
        down_proj_weights_tensor=expert_down_weights,
        activation_fn=activation_fn,
        normalization_type=NormType.NO_NORM,
        gate_proj_bias_tensor=expert_gate_up_bias,
        up_proj_bias_tensor=expert_gate_up_bias,
        down_proj_bias_tensor=expert_down_bias,
        output_dtype=output_dtype,
        store_output_in_sbuf=False,
        use_tkg_gate_up_proj_column_tiling=False,
        use_tkg_down_proj_column_tiling=False,
        down_weight_transposed=False,
        expert_params=expert_params,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        # Router fusion parameters (only used when router_weights is not None)
        router_matmul_weights=router_weights,
        router_bias=router_bias,
        router_act_fn=router_act_fn,
        norm_topk_prob=norm_topk_prob,
        gamma=mlp_gamma,
        eps=eps,
        top_k=top_k,
        normed_hbm=normed_hbm,
        name_prefix=name_prefix,
    )

    # [T, H] bf16 — cores reduce internally via sendrecv, core 0 writes final result
    output = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    _selective_expert_moe_tkg(
        mlp_params,
        output,
    )

    return output
