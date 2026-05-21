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

"""Selective-expert MoE token generation implementation v3.

Implements the fused routing path used by the live kernels: RMSNorm + Router
Matmul + Softmax + TopK + L1 Norm computed inline, then MoE MLP.

Uses inline GEMV specializations instead of the shared
process_gate_up_projection / process_down_projection helpers.
"""

import nki.isa as nisa
import nki.language as nl


from .moe_parameters import MLPParameters, MLPTKGConstantsDimensionSizes

# Subkernels for router fusion path
from .rmsnorm_tkg import rmsnorm_tkg
from .router_topk import router_topk_decode, XSBLayout_tp2013__1

# common utils
from ..utils.allocator import SbufManager
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info
from ..utils.logging import Logger
from ..utils.tensor_view import TensorView


def stream_shuffle_broadcast(src: nl.ndarray, dst: nl.ndarray) -> None:
    """
    Broadcasts the first partition of src onto the partition dim of dst.

    All inputs and outputs to this function are assumed to be in sbuf.
    This requires 2D src and dst, and the final dim of src matching the final dim of dst.
    """
    dst_npar = dst.shape[0]
    kernel_assert(
        len(src.shape) == 2 and len(dst.shape) == 2, "src and dst must be 2D tensors"
    )
    kernel_assert(
        src.shape[1] == dst.shape[1], "src and dst must have matching final dimension"
    )

    shuffle_mask = [0] * 32
    for i in range((dst_npar + 31) // 32):
        cur_npar = min(32, dst_npar - i * 32)
        nisa.nc_stream_shuffle(
            src=src[0:1, :],
            dst=dst[i * 32 : i * 32 + cur_npar, 0 : dst.shape[1]],
            shuffle_mask=shuffle_mask,
        )


def _selective_fused_gate_up_gemv_t1(
    hidden: nl.ndarray,
    output: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
    gate_up_weight: TensorView,
    warmup: bool = False,
) -> None:
    """
    T=1 fused gate/up GEMV path for selective MoE.

    Computes gate and up into FP32 scratch, applies the same clamp/activation
    semantics as process_gate_up_projection, then writes the multiplied result
    into the existing bf16 intermediate buffer.

    Requires gate_up_weight as the combined [H, 2, I] TensorView and always
    loads gate+up weights through a single fused DMA.
    """
    H0 = dims.H0
    I0 = dims.I0
    T = 1
    H = dims.H_per_shard
    I = dims.I
    num_h_tiles = H // H0
    num_h_tiles_by_2 = num_h_tiles // 2
    num_i_tiles = dims.num_total_128_tiles_per_I
    kernel_assert(
        num_h_tiles % 2 == 0,
        f"tp2013 layout requires an even H/128 tile count, got {num_h_tiles}",
    )
    kernel_assert(
        len(gate_up_weight.shape) == 3
        and gate_up_weight.shape[0] == H
        and gate_up_weight.shape[1] == 2
        and gate_up_weight.shape[2] == I,
        f"fused gate/up weights must have shape [H, 2, I] = [{H}, 2, {I}], got {gate_up_weight.shape}",
    )

    gate_sb = sbm.alloc_stack(
        (dims.I0, dims.num_total_128_tiles_per_I, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
        name=f"{sbm.get_name_prefix()}gate_gemv",
        align=4,
    )
    gate_silued = sbm.alloc_stack(
        gate_sb.shape,
        dtype=nl.float32,
        buffer=nl.sbuf,
        name=f"{sbm.get_name_prefix()}gate_silued",
    )
    up_sb = sbm.alloc_stack(
        (dims.I0, dims.num_total_128_tiles_per_I, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
        name=f"{sbm.get_name_prefix()}up_gemv",
        align=4,
    )

    # Fused DMA in tp2013 layout:
    # [H, 2, I] -> [2, H0, H/256, 2, I] -> [H0, 2, H/256, 2, I] -> [H0, H/128, 2, I].
    w_fused_5d = sbm.alloc_heap(
        (H0, 2, num_h_tiles_by_2, 2, I),
        dtype=gate_up_weight.dtype,
        buffer=nl.sbuf,
        name=f"{sbm.get_name_prefix()}gate_up_gemv_w_fused",
    )
    fused_weight_view = gate_up_weight.reshape_dim(
        dim=0, shape=(2, H0, num_h_tiles_by_2)
    ).permute(dims=[1, 0, 2, 3, 4])
    if warmup:
        nisa.dma_copy(
            dst=w_fused_5d,
            src=fused_weight_view.get_view(),
            dge_mode=nisa.dge_mode.hwdge,  # use standard DGE mode for warmup to avoid skewing perf stats with SWDGE overhead when counting DMAs
            engine=nisa.scalar_engine,  # use scalar engine for warmup to avoid skewing perf stats with GPSIMD overhead when counting DMAs
        )
    else:
        nisa.dma_copy(
            dst=w_fused_5d,
            src=fused_weight_view.get_view(),
            dge_mode=nisa.dge_mode.swdge,
        )
    w_fused = w_fused_5d.reshape((H0, num_h_tiles, 2, I))
    # Gate GEMV using w_fused[:, :, 0, :]
    for i_tile in range(num_i_tiles):
        tile_i_start = i_tile * I0
        tile_i_size = min(I0, I - tile_i_start)

        psum_acc = nl.ndarray(
            (I0, T),
            dtype=nl.float32,
            buffer=nl.psum,
            name=f"gate_{sbm.get_name_prefix()}gemv_psum_first_i{i_tile}",
            address=None if sbm.is_auto_alloc() else (0, 0),
        )
        psum_acc_second = nl.ndarray(
            (I0, T),
            dtype=nl.float32,
            buffer=nl.psum,
            name=f"gate_{sbm.get_name_prefix()}gemv_psum_second_i{i_tile}",
            address=None if sbm.is_auto_alloc() else (0, dims._psum_fmax * 4),
        )

        for h_idx in range(num_h_tiles_by_2):
            nisa.nc_matmul(
                dst=psum_acc[0:tile_i_size, 0:T],
                stationary=w_fused[
                    0:H0, h_idx, 0, tile_i_start : tile_i_start + tile_i_size
                ],
                moving=hidden[0:H0, 0:T, h_idx],
            )
        for h_idx in range(num_h_tiles_by_2, num_h_tiles):
            nisa.nc_matmul(
                dst=psum_acc_second[0:tile_i_size, 0:T],
                stationary=w_fused[
                    0:H0, h_idx, 0, tile_i_start : tile_i_start + tile_i_size
                ],
                moving=hidden[0:H0, 0:T, h_idx],
            )

        out_dst = (
            TensorView(gate_sb)
            .slice(dim=0, start=0, end=tile_i_size)
            .slice(dim=1, start=i_tile, end=i_tile + 1)
            .squeeze_dim(dim=1)
            .get_view()
        )
        nisa.tensor_copy(dst=out_dst, src=psum_acc[0:tile_i_size, 0:T])
        psum_acc_second_sb = sbm.alloc_stack(
            (I0, T),
            dtype=nl.float32,
            buffer=nl.sbuf,
            name=f"gate_{sbm.get_name_prefix()}gemv_psum_second_sbuf_i{i_tile}",
            align=4,
        )
        nisa.tensor_copy(
            dst=psum_acc_second_sb[0:tile_i_size, 0:T],
            src=psum_acc_second[0:tile_i_size, 0:T],
        )
        nisa.tensor_tensor(
            dst=out_dst,
            data1=out_dst,
            data2=psum_acc_second_sb[0:tile_i_size, 0:T],
            op=nl.add,
        )

    # Up GEMV using w_fused[:, :, 1, :]
    for i_tile in range(num_i_tiles):
        tile_i_start = i_tile * I0
        tile_i_size = min(I0, I - tile_i_start)

        psum_acc = nl.ndarray(
            (I0, T),
            dtype=nl.float32,
            buffer=nl.psum,
            name=f"up_{sbm.get_name_prefix()}gemv_psum_first_i{i_tile}",
            address=None if sbm.is_auto_alloc() else (0, 0),
        )
        psum_acc_second = nl.ndarray(
            (I0, T),
            dtype=nl.float32,
            buffer=nl.psum,
            name=f"up_{sbm.get_name_prefix()}gemv_psum_second_i{i_tile}",
            address=None if sbm.is_auto_alloc() else (0, dims._psum_fmax * 4),
        )

        for h_idx in range(num_h_tiles_by_2):
            nisa.nc_matmul(
                dst=psum_acc[0:tile_i_size, 0:T],
                stationary=w_fused[
                    0:H0, h_idx, 1, tile_i_start : tile_i_start + tile_i_size
                ],
                moving=hidden[0:H0, 0:T, h_idx],
            )
        for h_idx in range(num_h_tiles_by_2, num_h_tiles):
            nisa.nc_matmul(
                dst=psum_acc_second[0:tile_i_size, 0:T],
                stationary=w_fused[
                    0:H0, h_idx, 1, tile_i_start : tile_i_start + tile_i_size
                ],
                moving=hidden[0:H0, 0:T, h_idx],
            )

        out_dst = (
            TensorView(up_sb)
            .slice(dim=0, start=0, end=tile_i_size)
            .slice(dim=1, start=i_tile, end=i_tile + 1)
            .squeeze_dim(dim=1)
            .get_view()
        )
        nisa.tensor_copy(dst=out_dst, src=psum_acc[0:tile_i_size, 0:T])
        psum_acc_second_sb = sbm.alloc_stack(
            (I0, T),
            dtype=nl.float32,
            buffer=nl.sbuf,
            name=f"up_{sbm.get_name_prefix()}gemv_psum_second_sbuf_i{i_tile}",
            align=4,
        )
        nisa.tensor_copy(
            dst=psum_acc_second_sb[0:tile_i_size, 0:T],
            src=psum_acc_second[0:tile_i_size, 0:T],
        )
        nisa.tensor_tensor(
            dst=out_dst,
            data1=out_dst,
            data2=psum_acc_second_sb[0:tile_i_size, 0:T],
            op=nl.add,
        )

    nisa.activation(
        dst=gate_silued,
        op=nl.silu,
        data=gate_sb,
        scale=1.0,
    )
    nisa.tensor_tensor(dst=output, data1=gate_silued, data2=up_sb, op=nl.multiply)


def _selective_down_gemv_t1(
    hidden: nl.ndarray,
    weight: TensorView,
    output: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
    affinity_scale: nl.ndarray = None,
) -> None:
    """
    T=1 GEMV specialization for selective down projection.

    Weight layout: [H, I] (transposed / row-major for H).
    Computes output[H0, H1] = weight[H, I] @ hidden[I].

    Loads full weight [H, I] as [H0, H1, I] in a single DMA.
    For each (h_tile, i_tile) block, nc_transpose [H0, I0] -> [I0, H0],
    then nc_matmul: dst[H0, T] = stationary[I0, H0]^T @ moving[I0, T].

    Uses H-tile batching in PSUM banks, I-outer reduction loop.
    """

    H0 = dims.H0  # 128
    I0 = dims.I0  # 128
    T = 1
    I = dims.I

    num_i_tiles = dims.num_total_128_tiles_per_I
    num_h_tiles = dims.H1_shard  # H // H0
    MAX_PSUM_BANKS = 8
    num_h_batches = (num_h_tiles + MAX_PSUM_BANKS - 1) // MAX_PSUM_BANKS

    # Load full weight per I-tile: each w_tiles[i] has shape [tile_size, num_h_tiles, H0]
    w_tiles = []
    for i_tile in range(num_i_tiles):
        tile_i_start = i_tile * I0
        tile_i_size = min(I0, I - tile_i_start)
        w_tile = sbm.alloc_heap(
            (tile_i_size, num_h_tiles, H0),
            dtype=weight.dtype,
            buffer=nl.sbuf,
            name=f"down_{sbm.get_name_prefix()}gemv_w_tile{i_tile}",
        )
        weight_view = weight.slice(
            dim=0, start=tile_i_start, end=tile_i_start + tile_i_size
        ).reshape_dim(dim=1, shape=(num_h_tiles, H0))
        nisa.dma_copy(
            dst=w_tile,
            src=weight_view.get_view(),
            dge_mode=nisa.dge_mode.swdge,
        )
        w_tiles.append((w_tile, tile_i_size))

    # Allocate PSUM once for all h_batches so that interleaved results
    # from different batches coexist in the same PSUM tensor.
    # Layout: psum_all[:, b, h_batch] = result for h_tile = h_batch + b*num_h_batches
    psum_all = nl.ndarray(
        (H0, MAX_PSUM_BANKS, 512),
        dtype=nl.float32,
        buffer=nl.psum,
        name=f"down_{sbm.get_name_prefix()}psum_all",
        address=None if sbm.is_auto_alloc() else (0, 0),
    )
    # Per-bank views for matmul accumulation
    psum_tiles = []
    for b in range(MAX_PSUM_BANKS):
        psum_tiles.append(psum_all[:, b : b + 1, :])

    # Process H-tiles in batches of MAX_PSUM_BANKS (output dimension)
    for h_batch in range(num_h_batches):
        # Inner loop: accumulate across all I-tiles for each H-tile in this batch
        for i_tile in range(num_i_tiles):
            w_tile, tile_i_size = w_tiles[i_tile]
            for b in range(MAX_PSUM_BANKS):
                h_tile = h_batch + b * num_h_batches
                nisa.nc_matmul(
                    dst=psum_tiles[b][0:H0, 0, h_batch : h_batch + T],
                    stationary=w_tile[0:tile_i_size, h_tile, 0:H0],
                    moving=hidden[0:tile_i_size, i_tile, 0:T],
                )

    # PSUM -> output SBUF: single activation after all h_batches complete.
    # Element mapping: output[p, b*num_h_batches + h_batch] = psum_all[p, b, h_batch]
    # which correctly interleaves h_tiles when num_h_batches > 1.
    nisa.activation(
        dst=output[0:H0, 0:num_h_tiles],
        op=nl.copy,
        data=psum_all[0:H0, 0:MAX_PSUM_BANKS, 0:num_h_batches],
        scale=affinity_scale if affinity_scale is not None else 1.0,
    )


# ---------------------------------------------------------------------------
# Main kernel entry
# ---------------------------------------------------------------------------


def _selective_expert_moe_tkg(
    params: MLPParameters,
    output: nl.ndarray,
) -> nl.ndarray:
    """
    Selective-expert Mixture of Experts (MoE) kernel for token generation (TKG) — v3.

    Fused-routing entry point for the live MoE path. Processes only the top-K
    selected experts for each token, computing MLP projections via inline GEMV
    specializations and accumulating results weighted by expert affinities.

    Requires LNC2 (grid=[2]).

    Args:
        params: MLPParameters with router_matmul_weights and gamma set.
        output: Output tensor [2*T, H] — each core writes its partial sum.

    Returns:
        output with accumulated expert results.
    """

    # Expert-sharding v3 requires LNC2 (grid=[2])
    _grid_ndim, _n_prgs, _prg_id = get_verified_program_sharding_info(
        "selective_expert_moe_tkg_v3", (0, 1)
    )
    kernel_assert(
        _n_prgs == 2,
        f"selective_expert_implv3 requires LNC2 (grid=[2]), got num_shards={_n_prgs}",
    )

    H = params.hidden_tensor.shape[-1]
    # router_topk_decode uses bare nl.ndarray(buffer=nl.sbuf) allocations;
    # compiler requires all SBUF tensors to use the same allocation mode (auto or fixed address).
    # Force auto_alloc when router fusion is used to avoid NCC ICE.
    need_auto_alloc = (
        H >= 16 * 1024
        or params.input_in_sbuf
        or (params.router_matmul_weights is not None)
    )
    sbm = SbufManager(
        0,
        200 * 1024,
        Logger("selective_expert_moe_tkg_v3"),
        use_auto_alloc=need_auto_alloc,
    )
    sbm.open_scope()
    sbm.set_name_prefix(params.name_prefix)

    io_dtype = params.hidden_tensor.dtype
    kernel_assert(
        params.router_matmul_weights is not None,
        "selective_expert_implv3 only supports fused routing with router_matmul_weights",
    )
    kernel_assert(
        params.gamma is not None,
        "selective_expert_implv3 requires gamma for fused routing",
    )

    # Manually compute dims to avoid calculate_constants sharding assert (H1 % num_shards == 0).
    # v3 uses expert-sharding (not H-sharding), so each core computes full H.
    _pmax = nl.tile_size.pmax
    _psum_fmax = nl.tile_size.psum_fmax
    _psum_bmax = 8
    if _pmax <= 0:
        _pmax = 128
    if _psum_fmax <= 0:
        _psum_fmax = 512

    T = params.batch_size * params.sequence_len
    H = params.hidden_size

    weight_rank = len(params.gate_proj_weights_tensor.shape)
    kernel_assert(
        weight_rank == 4,
        f"selective_expert_implv3 requires combined gate/up weights with rank 4 [E, H, 2, I], got rank {weight_rank}",
    )
    local_E, _, gate_up_dim, I = params.gate_proj_weights_tensor.shape
    kernel_assert(
        gate_up_dim == 2,
        f"selective_expert_implv3 requires combined gate/up weights with shape [E, H, 2, I], got projection dim {gate_up_dim}",
    )

    H0 = _pmax
    I0 = _pmax
    H1 = H // H0

    K = params.top_k
    _, router_E = params.router_matmul_weights.shape
    local_E = router_E

    num_128_tiles_per_I, remainderI = divmod(I, I0)
    num_total_128_tiles_per_I = num_128_tiles_per_I + int(remainderI != 0)

    max_I_shard_size = 128 * 8
    num_shards_per_I = (I + max_I_shard_size - 1) // max_I_shard_size

    real_num_shards = _n_prgs
    real_shard_id = _prg_id

    dims = MLPTKGConstantsDimensionSizes(
        _pmax=_pmax,
        _psum_fmax=_psum_fmax,
        _psum_bmax=_psum_bmax,
        T=T,
        H=H,
        I=I,
        H0=H0,
        H1=H1,
        I0=I0,
        num_shards=1,
        shard_id=0,
        H_shard=H,
        H1_shard=H1,
        H1_offset=0,
        H_per_shard=H,
        num_total_128_tiles_per_I=num_total_128_tiles_per_I,
        num_128_tiles_per_I=num_128_tiles_per_I,
        remainderI=remainderI,
        remainderIFused=remainderI,
        column_tiling_dim=64,
        column_tiling_factor=128 // 64,
        num_shards_per_I=num_shards_per_I,
        max_I_shard_size=max_I_shard_size,
        do_norm_batch_sharding=False,
        K=K,
        E=local_E,
    )

    H_full = params.hidden_size
    H_free = H_full // _pmax

    rmsnorm_input = params.hidden_tensor
    if len(params.hidden_tensor.shape) == 2:
        rmsnorm_input = params.hidden_tensor.reshape((1, T, H_full))
    rmsnorm_out = sbm.alloc_stack(
        [_pmax, T, H_free],
        dtype=io_dtype,
        buffer=nl.sbuf,
        name="rmsnorm_out",
    )
    rmsnorm_out = rmsnorm_tkg(
        input=rmsnorm_input,
        gamma=params.gamma,
        output=rmsnorm_out,
        eps=params.eps,
        hidden_actual=params.hidden_actual,
        hidden_dim_tp=False,
        no_h_interleave=False,
        sbm=sbm,
        name_prefix=f"{params.name_prefix}router_",
    )

    input_sb = rmsnorm_out

    router_in = sbm.alloc_stack(
        [_pmax, T, H_free],
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name="router_in_f32_gamma",
    )
    nisa.tensor_copy(dst=router_in, src=rmsnorm_out)

    expert_index_sb, expert_affinities_eager = router_topk_decode(
        x=router_in,
        w=params.router_matmul_weights,
        w_bias=params.router_bias,
        act_fn=params.router_act_fn,
        k=K,
        norm_topk_prob=params.norm_topk_prob,
        x_sb_layout=XSBLayout_tp2013__1,
    )

    # Expert-sharding: split K experts across cores
    experts_per_core = dims.K // real_num_shards
    expert_start = real_shard_id * experts_per_core
    kernel_assert(
        dims.K % real_num_shards == 0,
        f"K={dims.K} must be divisible by num_shards={real_num_shards} for expert-sharding",
    )

    # Allocate SBUF location to store per-expert down results for cross-core reduce.
    # Shape: [H0, experts_per_core, H1] — one bf16 slice per local expert.
    # NOT accumulated here; accumulated after sendrecv with the other core's experts.
    down_per_expert = sbm.alloc_stack(
        (dims.H0, experts_per_core, dims.H1_shard),
        dtype=nl.bfloat16,
        name="down_per_expert",
        buffer=nl.sbuf,
    )

    # Allocate SBUF locations for gate/up projection result (only experts_per_core slots needed)
    gate_up_output = sbm.alloc_stack(
        (dims.I0, dims.num_total_128_tiles_per_I, experts_per_core),
        dtype=nl.bfloat16,
        name="intermediate_state_sbuf",
        buffer=nl.sbuf,
    )
    # silu(fp32gate)->bf16 siluted * fp32 up -> bf16 output

    # Allocate SBUF locations for down result (only experts_per_core slots needed)
    # Use bf16 to match nkilib precision (down PSUM -> bf16, then affinity scale bf16*fp32 -> bf16)
    down_output_list = []
    for local_k_idx in range(experts_per_core):
        down_sb = sbm.alloc_stack(
            (dims.H0, dims.H1_shard),
            dtype=nl.bfloat16,
            name=f"down_sbuf_{local_k_idx}",
            buffer=nl.sbuf,
        )
        down_output_list.append(down_sb)

    expert_idx = expert_index_sb

    params.use_tkg_gate_up_proj_column_tiling = False
    params.use_tkg_down_proj_column_tiling = False

    initial_gate_proj_weights_tensor = params.gate_proj_weights_tensor
    initial_down_proj_weights_tensor = params.down_proj_weights_tensor

    actual_dims_t = dims.T

    affinities_padded_sb = sbm.alloc_stack(
        (dims._pmax, dims.K),
        dtype=nl.float32,
        buffer=nl.sbuf,
        name="affinities_padded_sb",
    )
    nisa.tensor_copy(
        dst=affinities_padded_sb[0:actual_dims_t, 0 : dims.K],
        src=expert_affinities_eager,
    )

    # convert dims.T to 1 to compute output by each token
    dims.T = 1
    for token_idx in range(actual_dims_t):
        sbm.set_name_prefix(f"{params.name_prefix}T{token_idx}_")

        expert_affinity_sb = sbm.alloc_stack(
            (dims._pmax, dims.K),
            dtype=nl.float32,
            buffer=nl.sbuf,
            name="expert_affinity_sb",
        )
        stream_shuffle_broadcast(
            src=affinities_padded_sb[token_idx : token_idx + 1, 0 : dims.K],
            dst=expert_affinity_sb,
        )

        # sbm.open_scope(interleave_degree=)
        for local_k_idx in range(experts_per_core):
            expert_k_idx = expert_start + local_k_idx
            sbm.set_name_prefix(f"{params.name_prefix}T{token_idx}_K{expert_k_idx}_")

            expert_id_scalar_offset = expert_idx.ap(
                pattern=[[dims.K, 1], [1, 1]], offset=token_idx * dims.K + expert_k_idx
            )
            # Combined gate+up weight view [H, 2, I] for fused DMA
            gate_up_weight_combined = TensorView(
                initial_gate_proj_weights_tensor
            ).select(dim=0, index=expert_id_scalar_offset)
            kernel_assert(
                len(gate_up_weight_combined.shape) == 3
                and gate_up_weight_combined.shape[1] == 2,
                f"per-expert fused gate/up weights must have shape [H, 2, I], got {gate_up_weight_combined.shape}",
            )
            params.down_proj_weights_tensor = TensorView(
                initial_down_proj_weights_tensor
            ).select(dim=0, index=expert_id_scalar_offset)

            # Gate-Up projection: act_fn(gate(x)) * up(x)
            _selective_fused_gate_up_gemv_t1(
                hidden=input_sb[:, token_idx : token_idx + 1, :],
                output=gate_up_output[:, :, local_k_idx : local_k_idx + 1],
                dims=dims,
                sbm=sbm,
                gate_up_weight=gate_up_weight_combined,
                warmup=local_k_idx
                == 0,  # only count for the first expert to avoid redundant DMA stats in fused path
            )

            # Down projection: output bf16 (no affinity fused), matching nkilib precision.
            down_sb = down_output_list[local_k_idx]
            _selective_down_gemv_t1(
                hidden=gate_up_output[:, :, local_k_idx : local_k_idx + 1],
                weight=params.down_proj_weights_tensor,
                output=down_sb,
                dims=dims,
                sbm=sbm,
                affinity_scale=None,
            )

            # Apply affinity scaling separately: bf16 down_sb * fp32 affinity -> bf16 down_sb
            # This matches nkilib's tensor_scalar(dst=down_sb, data=down_sb, multiply, affinity)
            nisa.tensor_scalar(
                dst=down_sb,
                data=down_sb,
                op0=nl.multiply,
                operand0=expert_affinity_sb[:, expert_k_idx : expert_k_idx + 1],
            )

            # Store bf16 down result (with affinity applied) into per-expert buffer.
            # Will be reduced across cores after the expert loop.
            nisa.tensor_copy(
                dst=down_per_expert[0 : dims.H0, local_k_idx, 0 : dims.H1_shard],
                src=down_sb,
            )

            # sbm.increment_section()
        # sbm.close_scope()

    # revert dims.T
    dims.T = actual_dims_t

    # Save output result
    sbm.set_name_prefix(params.name_prefix)

    # --- Cross-core bf16 reduce with H-sharded accumulation ---
    #
    # Each core has down_per_expert [H0, experts_per_core, H1] with K/2 bf16 results.
    # Exchange via sendrecv so each core has all K experts' bf16 down results.
    # Then each core reduces K experts over its own H/2 slice (H-sharding).
    #
    # Layout after exchange:
    #   all_experts [H0, K, H1] where:
    #     core 0's local experts at [:, 0:K/2, :]
    #     core 1's local experts at [:, K/2:K, :]
    #
    # Core 0 reduces [:, :, 0:H1/2], core 1 reduces [:, :, H1/2:H1].

    H1_half = dims.H1_shard // 2

    # Receive the other core's per-expert results
    down_per_expert_recv = sbm.alloc_stack(
        (dims.H0, experts_per_core, dims.H1_shard),
        dtype=nl.bfloat16,
        name="down_per_expert_recv",
        buffer=nl.sbuf,
    )
    nisa.sendrecv(
        dst=down_per_expert_recv,
        src=down_per_expert,
        send_to_rank=1 - real_shard_id,
        recv_from_rank=1 - real_shard_id,
        pipe_id=0,
    )

    # Determine which half of H1 this core reduces.
    # Core 0: h1_start=0 (first half), Core 1: h1_start=H1_half (second half)
    h1_start = real_shard_id * H1_half

    # Assemble: local experts first, then received experts (ordered by core 0 first)
    # Core 0 local = expert_start=0 (experts 0..K/2-1)
    # Core 1 local = expert_start=K/2 (experts K/2..K-1)
    # So: core 0 sees [local(0..K/2-1), recv(K/2..K-1)] — correct order
    #     core 1 sees [local(K/2..K-1), recv(0..K/2-1)] — need to swap
    if real_shard_id == 0:
        first_half = down_per_expert  # experts 0..K/2-1
        second_half = down_per_expert_recv  # experts K/2..K-1
    else:
        first_half = down_per_expert_recv  # experts 0..K/2-1 (from core 0)
        second_half = down_per_expert  # experts K/2..K-1 (local)

    # Reduce K experts into bf16 output for this core's H1 half.
    # Accumulate in bf16 to match nkilib (bf16 + bf16 → bf16).
    output_reduced = sbm.alloc_stack(
        (dims.H0, H1_half),
        dtype=nl.bfloat16,
        name="output_reduced",
        buffer=nl.sbuf,
    )

    # Start with first expert
    nisa.tensor_copy(
        dst=output_reduced[0 : dims.H0, 0:H1_half],
        src=first_half[0 : dims.H0, 0, h1_start : h1_start + H1_half],
    )
    # Add remaining experts from first_half
    for k_idx in range(1, experts_per_core):
        nisa.tensor_tensor(
            dst=output_reduced[0 : dims.H0, 0:H1_half],
            data1=output_reduced[0 : dims.H0, 0:H1_half],
            data2=first_half[0 : dims.H0, k_idx, h1_start : h1_start + H1_half],
            op=nl.add,
        )
    # Add all experts from second_half
    for k_idx in range(experts_per_core):
        nisa.tensor_tensor(
            dst=output_reduced[0 : dims.H0, 0:H1_half],
            data1=output_reduced[0 : dims.H0, 0:H1_half],
            data2=second_half[0 : dims.H0, k_idx, h1_start : h1_start + H1_half],
            op=nl.add,
        )

    # Transpose output_reduced [H0, H1_half] -> [T=1, H_half] and DMA store.
    # Each core writes its H/2 portion to the correct columns in output [T, H].
    h_offset = real_shard_id * H1_half * dims.H0  # byte offset in H dimension

    output_sb = sbm.alloc_stack(
        (dims.T, H1_half * dims.H0),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name="tkg_moe_output_sb",
    )
    output_reduced_fp32 = sbm.alloc_stack(
        (dims.H0, H1_half),
        dtype=nl.float32,
        buffer=nl.sbuf,
        name="output_reduced_fp32",
    )
    nisa.tensor_copy(dst=output_reduced_fp32, src=output_reduced)
    for h1_idx in range(H1_half):
        psum_idx = h1_idx % dims._psum_bmax
        tp_psum = nl.ndarray(
            (dims.T, dims.H0),
            dtype=nl.float32,
            buffer=nl.psum,
            name=f"{params.name_prefix}transpose_output_{h1_idx}",
            address=(
                None if sbm.is_auto_alloc() else (0, psum_idx * dims._psum_fmax * 4)
            ),
        )
        nisa.nc_transpose(
            dst=tp_psum[0 : dims.T, 0 : dims.H0],
            data=output_reduced_fp32[0 : dims.H0, h1_idx : h1_idx + 1],
        )
        nisa.tensor_copy(
            dst=output_sb[0 : dims.T, h1_idx * dims.H0 : (h1_idx + 1) * dims.H0],
            src=tp_psum[0 : dims.T, 0 : dims.H0],
        )

    # Each core writes its H/2 columns to output HBM
    output_slice = output[0 : dims.T, h_offset : h_offset + H1_half * dims.H0]
    nisa.dma_copy(
        dst=output_slice,
        src=output_sb[0 : dims.T, 0 : H1_half * dims.H0],
    )

    sbm.close_scope()
    return output
