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

"""Simplified MLP parameters for MoE operations (BF16 only, no quantization)."""

from dataclasses import dataclass
from typing import Optional

import nki
import nki.language as nl
from nki.language import NKIObject

from ..utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    NormType,
    RouterActFnType,
)
from ..utils.kernel_assert import kernel_assert

SUPPORTED_DTYPES = [nl.bfloat16, nl.float16]


@dataclass
class MLPTKGConstantsDimensionSizes(nl.NKIObject):
    """
    Dimension sizes for MLP TKG computation.

    Contains all dimension constants computed from input parameters including
    partition sizes, sharding info, and tiling parameters.
    """

    _pmax: int
    _psum_fmax: int
    _psum_bmax: int
    T: int
    H: int
    I: int
    H0: int
    H1: int
    I0: int
    num_shards: int
    shard_id: int
    H_shard: int
    H1_shard: int
    H1_offset: int
    H_per_shard: int
    num_total_128_tiles_per_I: int
    num_128_tiles_per_I: int
    remainderI: int
    remainderIFused: int
    column_tiling_dim: int
    column_tiling_factor: int
    num_shards_per_I: int
    max_I_shard_size: int
    do_norm_batch_sharding: int
    K: Optional[int] = None
    E: Optional[int] = None


@dataclass
class MLPExpertParameters(NKIObject):
    """Expert parameters for MoE (simplified)."""

    expert_affinities: nl.ndarray
    expert_index: nl.ndarray
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = (
        ExpertAffinityScaleMode.POST_SCALE
    )


@dataclass
class MLPBiasParameters(NKIObject):
    """Bias parameters for MLP projections."""

    gate_proj_bias_tensor: Optional[nl.ndarray]
    up_proj_bias_tensor: Optional[nl.ndarray]
    down_proj_bias_tensor: Optional[nl.ndarray]

    def __init__(
        self,
        gate_proj_bias_tensor: Optional[nl.ndarray],
        up_proj_bias_tensor: Optional[nl.ndarray],
        down_proj_bias_tensor: Optional[nl.ndarray],
    ):
        self.gate_proj_bias_tensor = gate_proj_bias_tensor
        self.up_proj_bias_tensor = up_proj_bias_tensor
        self.down_proj_bias_tensor = down_proj_bias_tensor


@dataclass
class MLPNormalizationParameters(NKIObject):
    """Normalization parameters (simplified)."""

    normalization_type: NormType
    normalization_weights_tensor: Optional[nl.ndarray]
    normalization_bias_tensor: Optional[nl.ndarray]

    def __init__(
        self,
        normalization_type: NormType,
        normalization_weights_tensor: Optional[nl.ndarray],
        normalization_bias_tensor: Optional[nl.ndarray],
    ):
        if normalization_type == NormType.NO_NORM:
            self.normalization_type = NormType.NO_NORM
            self.normalization_weights_tensor = None
            self.normalization_bias_tensor = None
        else:
            self.normalization_type = normalization_type
            self.normalization_weights_tensor = normalization_weights_tensor
            self.normalization_bias_tensor = normalization_bias_tensor


@dataclass
class MoEGateParameters(NKIObject):
    """Gate parameters for optional RMSNorm + Router TopK in MoE.

    When gamma is provided: RMSNorm + Router TopK + Selective MLP (full fuse).
    When gamma is None: Router TopK + Selective MLP (RMSNorm done externally,
        hidden_input is already the normed output).
    """

    gamma: Optional[nl.ndarray]  # [1, H] RMSNorm weights (None = skip RMSNorm)
    router_weights: nl.ndarray  # [H, E] Router weights
    router_bias: Optional[nl.ndarray]  # [1, E] Optional router bias
    router_act_fn: RouterActFnType  # SIGMOID or SOFTMAX
    router_pre_norm: bool  # Apply activation before top-K
    norm_topk_prob: bool  # Normalize top-K probabilities
    router_mm_dtype: nki.dtype  # dtype for router matmul
    hidden_actual: Optional[int]  # actual H for RMSNorm mean calc
    top_k: int  # number of top-K experts per token

    def __init__(
        self,
        router_weights: nl.ndarray,
        top_k: int,
        gamma: Optional[nl.ndarray] = None,
        router_bias: Optional[nl.ndarray] = None,
        router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
        router_pre_norm: bool = True,
        norm_topk_prob: bool = False,
        router_mm_dtype: nki.dtype = nl.bfloat16,
        hidden_actual: Optional[int] = None,
    ):
        self.gamma = gamma
        self.router_weights = router_weights
        self.top_k = top_k
        self.router_bias = router_bias
        self.router_act_fn = router_act_fn
        self.router_pre_norm = router_pre_norm
        self.norm_topk_prob = norm_topk_prob
        self.router_mm_dtype = router_mm_dtype
        self.hidden_actual = hidden_actual


@dataclass
class MLPParameters(NKIObject):
    """Simplified MLP parameters for BF16 inference only."""

    hidden_tensor: nl.ndarray
    gate_proj_weights_tensor: nl.ndarray
    up_proj_weights_tensor: nl.ndarray
    down_proj_weights_tensor: nl.ndarray
    activation_fn: ActFnType
    output_dtype: nki.dtype
    norm_params: Optional[MLPNormalizationParameters]
    bias_params: Optional[MLPBiasParameters]
    expert_params: Optional[MLPExpertParameters]
    gate_params: Optional[MoEGateParameters]
    router_probs: Optional[
        nl.ndarray
    ]  # [T, E] pre-computed softmax probs (for topk-only path)
    router_matmul_weights: Optional[
        nl.ndarray
    ]  # [H, E] router weights for matmul (for router_matmul path)
    router_bias: Optional[nl.ndarray]  # [1, E] router bias (for router_matmul path)
    router_act_fn: Optional[
        RouterActFnType
    ]  # SIGMOID or SOFTMAX (for router_matmul path)
    norm_topk_prob: bool  # L1 normalize top-K (for router_matmul path)
    gamma: Optional[
        nl.ndarray
    ]  # [1, H] RMSNorm weights (for router_matmul path, None=skip)
    hidden_actual: Optional[
        int
    ]  # actual H for RMSNorm mean calc (for router_matmul path)
    normed_hbm: Optional[
        nl.ndarray
    ]  # [T, H] scratch HBM for RMSNorm output (for router_matmul + gamma path)
    top_k: Optional[int]  # top-K for router_probs / router_matmul path
    name_prefix: str  # optional prefix for generated op names
    eps: float
    batch_size: int
    sequence_len: int
    hidden_size: int
    intermediate_size: int
    input_in_sbuf: bool
    store_output_in_sbuf: bool
    use_tkg_gate_up_proj_column_tiling: bool
    use_tkg_down_proj_column_tiling: bool
    gate_clamp_lower_limit: Optional[float]
    gate_clamp_upper_limit: Optional[float]
    up_clamp_lower_limit: Optional[float]
    up_clamp_upper_limit: Optional[float]
    down_weight_transposed: bool

    def __init__(
        self,
        hidden_tensor: nl.ndarray,
        gate_proj_weights_tensor: nl.ndarray,
        up_proj_weights_tensor: nl.ndarray,
        down_proj_weights_tensor: nl.ndarray,
        normalization_weights_tensor: Optional[nl.ndarray] = None,
        gate_proj_bias_tensor: Optional[nl.ndarray] = None,
        up_proj_bias_tensor: Optional[nl.ndarray] = None,
        down_proj_bias_tensor: Optional[nl.ndarray] = None,
        normalization_bias_tensor: Optional[nl.ndarray] = None,
        activation_fn: ActFnType = ActFnType.SiLU,
        normalization_type: NormType = NormType.NO_NORM,
        output_dtype: nki.dtype = nl.bfloat16,
        store_output_in_sbuf: bool = False,
        eps: float = 1e-6,
        use_tkg_gate_up_proj_column_tiling: bool = False,
        use_tkg_down_proj_column_tiling: bool = False,
        gate_clamp_lower_limit: Optional[float] = None,
        gate_clamp_upper_limit: Optional[float] = None,
        up_clamp_lower_limit: Optional[float] = None,
        up_clamp_upper_limit: Optional[float] = None,
        down_weight_transposed: bool = False,
        expert_params: Optional[MLPExpertParameters] = None,
        gate_params: Optional[MoEGateParameters] = None,
        router_probs: Optional[nl.ndarray] = None,
        router_matmul_weights: Optional[nl.ndarray] = None,
        router_bias: Optional[nl.ndarray] = None,
        router_act_fn: Optional[RouterActFnType] = None,
        norm_topk_prob: bool = False,
        gamma: Optional[nl.ndarray] = None,
        hidden_actual: Optional[int] = None,
        normed_hbm: Optional[nl.ndarray] = None,
        top_k: Optional[int] = None,
        name_prefix: str = "",
    ):
        self.input_in_sbuf = hidden_tensor.buffer == nl.sbuf
        if self.input_in_sbuf:
            # SBUF input shape: [H0, T, H1]
            kernel_assert(
                len(hidden_tensor.shape) == 3,
                "SBUF input must have 3D shape [H0, T, H1]",
            )
            _, T, _ = hidden_tensor.shape
            self.batch_size = 1
            self.sequence_len = T
            self.hidden_size = down_proj_weights_tensor.shape[-1]
        elif len(hidden_tensor.shape) == 3:  # B, S, H
            self.batch_size = hidden_tensor.shape[0]
            self.sequence_len = hidden_tensor.shape[1]
            self.hidden_size = hidden_tensor.shape[2]
        else:  # T, H
            self.batch_size = 1
            self.sequence_len = hidden_tensor.shape[0]
            self.hidden_size = hidden_tensor.shape[1]

        self.down_weight_transposed = down_weight_transposed
        if len(down_proj_weights_tensor.shape) == 3:
            if down_weight_transposed:
                # [E, H, I] layout
                self.intermediate_size = down_proj_weights_tensor.shape[2]
                kernel_assert(
                    down_proj_weights_tensor.shape[1] == self.hidden_size,
                    "unexpected down project weight shape {down_proj_weights_tensor.shape}",
                )
            else:
                # [E, I, H] layout
                self.intermediate_size = down_proj_weights_tensor.shape[1]
                kernel_assert(
                    down_proj_weights_tensor.shape[2] == self.hidden_size,
                    "unexpected down project weight shape {down_proj_weights_tensor.shape}",
                )
        elif len(down_proj_weights_tensor.shape) == 2:  # I, H
            self.intermediate_size = down_proj_weights_tensor.shape[0]
            kernel_assert(
                down_proj_weights_tensor.shape[1] == self.hidden_size,
                "unexpected down project weight shape {down_proj_weights_tensor.shape}",
            )

        self.hidden_tensor = hidden_tensor
        self.gate_proj_weights_tensor = gate_proj_weights_tensor
        self.up_proj_weights_tensor = up_proj_weights_tensor
        self.down_proj_weights_tensor = down_proj_weights_tensor
        self.activation_fn = activation_fn
        self.output_dtype = output_dtype
        self.eps = eps
        self.store_output_in_sbuf = store_output_in_sbuf
        self.use_tkg_gate_up_proj_column_tiling = use_tkg_gate_up_proj_column_tiling
        self.use_tkg_down_proj_column_tiling = use_tkg_down_proj_column_tiling
        self.gate_clamp_lower_limit = gate_clamp_lower_limit
        self.gate_clamp_upper_limit = gate_clamp_upper_limit
        self.up_clamp_lower_limit = up_clamp_lower_limit
        self.up_clamp_upper_limit = up_clamp_upper_limit

        self.norm_params = MLPNormalizationParameters(
            normalization_type, normalization_weights_tensor, normalization_bias_tensor
        )
        self.bias_params = MLPBiasParameters(
            gate_proj_bias_tensor, up_proj_bias_tensor, down_proj_bias_tensor
        )
        self.expert_params = expert_params
        self.gate_params = gate_params
        self.router_probs = router_probs
        self.router_matmul_weights = router_matmul_weights
        self.router_bias = router_bias
        self.router_act_fn = router_act_fn
        self.norm_topk_prob = norm_topk_prob
        self.gamma = gamma
        self.hidden_actual = hidden_actual
        self.normed_hbm = normed_hbm
        self.top_k = top_k
        self.name_prefix = name_prefix
