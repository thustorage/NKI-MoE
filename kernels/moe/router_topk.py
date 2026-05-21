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

"""
This file implements router input layout helpers and the slim router decode top-K path
for Mixture of Experts (MoE) models.
"""

import nki.isa as nisa
import nki.language as nl

from ..utils import tensor_view
from ..utils.kernel_assert import kernel_assert

P_MAX = 128
PE_COLUMN_TILE_32 = 32  # PE array column tile size for T <= 32
PE_COLUMN_TILE_64 = 64  # PE array column tile size for 32 < T <= 64
PE_COLUMN_TILE_128 = 128  # PE array column tile size (full width, disables tiling)

# TODO: the new FE is having issue with using Enum in megakernel setting
# thus we work around it by using explicit constants
XHBMLayout_H_T__0 = 0
XHBMLayout_T_H__1 = 1

XSBLayout_tp102__0 = 0
XSBLayout_tp2013__1 = 1
XSBLayout_tp201__2 = 2
XSBLayout__128_Hdiv128_T__3 = 3


def router_topk_input_w_load(w: nl.ndarray, x_sb_layout, name=""):
    """
    Load weight tensor w from HBM to SBUF with layout matching x tensor.

    Performs DMA transfer from HBM to SBUF with layout conversion that matches
    the H-dimension stride pattern of the x tensor in SBUF, enabling efficient
    matmul operations.

    Args:
        w (nl.ndarray): Weight tensor [H, E] in HBM
        x_sb_layout (int): Layout of x in SBUF (0-3), determines w layout
        name (str): Optional tensor annotation name for debugging

    Returns:
        w_sb (nl.ndarray): Weight tensor in SBUF [128, H/128, E] with layout
                          matching x_sb_layout H-dimension stride pattern

    Notes:
        - H must be a multiple of 128
        - Layout must match x tensor layout for correct matmul contraction
        - Layout 0: H-tiles arranged horizontally with stride H/128
        - Layout 1: H-tiles with interleaved halves, stride H/256
        - Layouts 2-3: H-tiles with consecutive H elements
    """
    H, E = w.shape
    num_h_tiles = H // P_MAX
    num_h_tiles_by_2 = num_h_tiles // 2

    w_sb = None

    if x_sb_layout not in [0, 1, 2, 3]:
        kernel_assert(
            False,
            f"router_topk_input_w_load only x_sb_layout=0,1,2,3 are supported. Specified layout value: {x_sb_layout}",
        )

    if x_sb_layout == XSBLayout_tp102__0:
        # HBM tensor shape [H,E] reshaped (new view) as [128, num_h_tiles, E] where num_h_tiles = H/128.
        # A note about the terminology. In SBUF we aim to create num_h_tiles H-Tiles along the free-dim.
        # This is done by taking num_h_tiles rows from HBM and arranging them horizontally in the SB free-dim.
        # And we must do this for each SB partition-dimension row. We want to use common terminology between the
        # HBM and SB diagrams to show how HBM data is rearranged in SB.
        # Therefore we say that, in HBM, we have an H-tile-count of 128 and an H-tile-size of num_h_tiles=H/128.
        #
        # H-dim ↓     E columns →
        # ┌─────────────────────┐
        # │ H0                  │ ← H-Tile 0 (num_h_tiles rows)
        # │ H1                  │
        # │ H2                  │
        # │ ..                  │
        # │ H_num_h_tiles-1     │
        # ├─────────────────────┤
        # │ H_num_h_tiles       │ ← H-Tile 1 (num_h_tiles rows)
        # │ H_num_h_tiles+1     │
        # │ H_num_h_tiles+2     │
        # │ ..                  │
        # │ H_2*num_h_tiles-1   │
        # ├─────────────────────┤
        # │        ...          │ ← ... (more H-Tiles)
        # ├─────────────────────┤
        # │ H_127*num_h_tiles   │ ← H-Tile 127 (num_h_tiles rows)
        # │ H_127*num_h_tiles+1 │
        # │ H_127*num_h_tiles+2 │
        # │ ..                  │
        # │ H-1                 │
        # └─────────────────────┘

        # SBUF layout [128, num_h_tiles, E]
        #
        #        ←─E cols─→         ←─E cols─→         ←─E cols─→               ←─E cols─→
        # ┌───────────────────┬───────────────────┬───────────────────┬───┬───────────────────┐
        # │        H0         │        H1         │        H2         │...│  H_num_h_tiles-1  │ ← P-dim row 0 (H-Tile 0)
        # ├───────────────────┼───────────────────┼───────────────────┼───┼───────────────────┤
        # │   H_num_h_tiles   │  H_num_h_tiles+1  │  H_num_h_tiles+2  │...│ H_2*num_h_tiles-1 │ ← P-dim row 1 (H-Tile 1)
        # ├───────────────────┼───────────────────┼───────────────────┼───┼───────────────────┤
        # │        ...        │        ...        │        ...        │...│        ...        │ ← ... (more P-dim rows)
        # ├───────────────────┼───────────────────┼───────────────────┼───┼───────────────────┤
        # │ H_127*num_h_tiles │H_127*num_h_tiles+1│H_127*num_h_tiles+2│...│       H-1         │ ← P-dim row 127 (H-Tile 127)
        # └───────────────────┴───────────────────┴───────────────────┴───┴───────────────────┘
        #
        # Each P-dim row contains one H-Tile spread horizontally.
        # As we move down one column of SBUF we get H data with a stride of num_h_tiles, which matches the
        # the stride of the corresponding 'x' layout.

        w_reshape = w.reshape((P_MAX, num_h_tiles, E))
        w_sb = nl.ndarray(
            (P_MAX, num_h_tiles, E), dtype=w.dtype, buffer=nl.sbuf, name=name
        )
        nisa.dma_copy(src=w_reshape, dst=w_sb)

    elif x_sb_layout == XSBLayout_tp2013__1:
        # First, understand the above diagram for x_sb_layout==0.
        # x_sb_layout==1 shrinks the H-tile size by half (num_h_tiles_by_2) and introduces an additional
        # dimension of 2 on the H dimension.
        # num_h_tiles = nht = H/128
        # num_h_tiles_by_2 = nht2 = nht/2 = H/256

        # HBM tensor shape [H,E] reshaped (new view) as [2, 128, num_h_tiles_by_2, E]
        # In other words, this is similar to x_sb_layout=0 except the H dimension is further divided into 2 halves.
        # Within each half, we take the same view as x_sb_layout=0 but each H-Tile contains half as many rows.

        # Half-a (upper half - 128 H-Tiles):
        # ┌─────────────────┐
        # │  H0             │ ← H-Tile 0 (nht2 rows)
        # │  H1             │
        # │  H2             │
        # │  ..             │
        # │  H_nht2-1       │
        # ├─────────────────┤
        # │  H_nht2         │ ← H-Tile 1 (nht2 rows)
        # │  H_nht2+1       │
        # │  H_nht2+2       │
        # │  ..             │
        # │  H_2*nht2-1     │
        # ├─────────────────┤
        # │        ...      │ ← ... (more H-Tiles)
        # ├─────────────────┤
        # │ H_127*nht2      │ ← H-Tile 127 (nht2 rows)
        # │ H_127*nht2+1    │
        # │ H_127*nht2+2    │
        # │  ..             │
        # │ H_128*nht2-1    │
        # ├─────────────────┤
        # Half-b (lower half - 128 H-Tiles):
        # ├─────────────────┤
        # │ H_128*nht2      │ ← H-Tile 0 (nht2 rows)
        # │ H_128*nht2+1    │
        # │ H_128*nht2+2    │
        # │  ..             │
        # │ H_129*nht2-1    │
        # ├─────────────────┤
        # │ H_129*nht2      │ ← H-Tile 1 (nht2 rows)
        # │ H_129*nht2+1    │
        # │ H_129*nht2+2    │
        # │  ..             │
        # │ H_130*nht2-1    │
        # ├─────────────────┤
        # │        ...      │ ← ... (more H-Tiles)
        # ├─────────────────┤
        # │ H_255*nht2      │ ← H-Tile 127 (nht2 rows)
        # │ H_255*nht2+1    │
        # │ H_255*nht2+2    │
        # │  ..             │
        # │        H-1      │
        # └─────────────────┘
        #
        # Total: 2 halves × 128 H-Tiles, each H-Tile containing num_h_tiles_by_2×E elements

        # SBUF layout [128, 2, nht2, E] where halves are in separate dim-1 slices.
        # But returned as [128, H/128, E] (see below).
        #
        #     Half-a (dim-1=0)                                          Half-b (dim-1=1)
        #      ←─E─→        ←─E─→        ←─E─→                   ←─E─→           ←─E─→            ←─E─→
        # ┌─────────────┬─────────────┬─────────────┬───┐ ┌─────────────────┬─────────────────┬─────────────────┬───┐
        # │     H0      │     H1      │     H2      │...│ │   H_128*nht2    │  H_128*nht2+1   │  H_128*nht2+2   │...│ P-dim row 0 (H-Tile 0)
        # ├─────────────┼─────────────┼─────────────┼───┤ ├─────────────────┼─────────────────┼─────────────────┼───┤
        # │   H_nht2    │  H_nht2+1   │  H_nht2+2   │...│ │   H_129*nht2    │  H_129*nht2+1   │  H_129*nht2+2   │...│ P-dim row 1 (H-Tile 1)
        # ├─────────────┼─────────────┼─────────────┼───┤ ├─────────────────┼─────────────────┼─────────────────┼───┤
        # │     ...     │     ...     │     ...     │...│ │       ...       │       ...       │       ...       │...│ ... (more H-Tiles)
        # ├─────────────┼─────────────┼─────────────┼───┤ ├─────────────────┼─────────────────┼─────────────────┼───┤
        # │ H_127*nht2  │H_127*nht2+1 │H_127*nht2+2 │...│ │   H_255*nht2    │  H_255*nht2+1   │  H_255*nht2+2   │...│ P-dim row 127 (H-Tile 127)
        # └─────────────┴─────────────┴─────────────┴───┘ └─────────────────┴─────────────────┴─────────────────┴───┘
        #
        # Each P-dim row contains one HBM H-Tile from Half-a followed by one HBM H-Tile from Half-b.
        # As we move down one column of SBUF we get H data with a stride of num_h_tiles_by_2, which matches the
        #   the stride of the corresponding 'x' layout.

        w_reshape = w.reshape((2, P_MAX, num_h_tiles_by_2, E))
        w_sb = nl.ndarray(
            (P_MAX, 2, num_h_tiles_by_2, E), dtype=w.dtype, buffer=nl.sbuf, name=name
        )
        nisa.dma_copy(
            src=tensor_view.TensorView(w_reshape).permute([1, 0, 2, 3]).get_view(),
            dst=tensor_view.TensorView(w_sb).get_view(),
        )
        # We always return a 3D shape. The extra H-dimension of [0:2], while necessary for loading from HBM->SBUF above,
        # is not necessary when an eventual matmul reads from SBUF. It will simply read the entire free-dim in order.
        w_sb = w_sb.reshape((P_MAX, num_h_tiles, E))

    else:  # x_sb_layout = 2,3
        # HBM tensor shape [H,E] reshaped (new view) as [num_h_tiles, 128, E].
        # Divide the H dimension into num_h_tiles groups of 128 rows.
        #
        # H-dim ↓     E columns →
        # ┌─────────────────────────────────┐
        # │ H0          H-tile 0            │
        # │                                 │
        # │                                 │
        # │ H127                            │
        # ├─────────────────────────────────┤
        # │ H128        H-tile 1            │
        # │                                 │
        # │                                 │
        # │ H255                            │
        # ├─────────────────────────────────┤
        # │ H256        H-tile 2            │
        # │                                 │
        # │                                 │
        # │ H383                            │
        # ├─────────────────────────────────┤
        # │              ...                │
        # ├─────────────────────────────────┤
        # │ H_(num_h_tiles-1)*128           │
        # │         H-tile num_h_tiles-1    │
        # │                                 │
        # │ H-1                             │
        # └─────────────────────────────────┘
        #
        # SBUF layout [128, num_h_tiles, E]
        #
        #   ←─────E─────→     ←─────E─────→     ←─────E─────→             ←─────E─────→
        # ┌─────────────────┬─────────────────┬─────────────────┬───┬─────────────────────────┐
        # │ H0              │ H128            │ H256            │   │ H_(num_h_tiles-1)*128   │
        # │                 │                 │                 │   │                         │
        # │   H-tile 0      │   H-tile 1      │   H-tile 2      │...│ H-tile num_h_tiles-1    │ ← 128 rows
        # │                 │                 │                 │   │                         │   (P-dim)
        # │ H127            │ H255            │ H383            │   │ H-1                     │
        # └─────────────────┴─────────────────┴─────────────────┴───┴─────────────────────────┘
        #
        # H-tiles arranged left to right.
        # Therefore each column of SBUF contains H data with a stride of 1 (i.e. consecutive H data).

        w_sb = nl.ndarray(
            (P_MAX, num_h_tiles, E), dtype=w.dtype, buffer=nl.sbuf, name=name
        )

        nisa.dma_copy(
            src=w.ap(pattern=[[E, 128], [128 * E, num_h_tiles], [1, E]], offset=0),
            dst=tensor_view.TensorView(w_sb).get_view(),
        )

    return w_sb


def router_topk_decode(
    x: nl.ndarray,
    w: nl.ndarray,
    w_bias: nl.ndarray,
    act_fn,
    k: int,
    norm_topk_prob: bool = False,
    x_sb_layout: int = XSBLayout_tp2013__1,
):
    """
    Slim router top-K for decode (T<=128, SBUF in/out).

    Only implements the Qwen3 decode path: ACT1 -> TopK -> optional Norm.
    Returns [T, K] top-K affinities directly — no scatter to [T, E] and no roundtrip.

    Args:
        x: [128, T, H/128] in SBUF (layout determined by x_sb_layout)
        w: [H, E] in HBM
        w_bias: [1, E] or None
        act_fn: RouterActFnType (SOFTMAX or SIGMOID)
        k: top-K
        norm_topk_prob: L1 normalize top-K values
        x_sb_layout: SBUF layout of x. 0=tp102, 1=tp2013 (interleaved, default), 2=tp201

    Returns:
        (expert_index [T, K] uint32 SBUF, expert_affinities_topk [T, K] f32 SBUF)
    """
    # Derive dims
    H, E = w.shape
    _, T, H_free = x.shape  # [128, T, H/128]

    kernel_assert(T <= P_MAX, f"router_topk_decode requires T <= {P_MAX}, got {T}")
    kernel_assert(H % P_MAX == 0, f"H ({H}) must be a multiple of {P_MAX}")
    kernel_assert(k <= 8, f"K ({k}) must be <= 8")
    kernel_assert(x.dtype == nl.bfloat16, f"x dtype ({x.dtype}) must be bfloat16")
    kernel_assert(w.dtype == nl.bfloat16, f"w dtype ({w.dtype}) must be bfloat16")
    num_h_tiles = H // P_MAX
    has_bias = w_bias is not None

    # ---- Load weight w to SBUF (layout must match x) ----
    w_sb = router_topk_input_w_load(
        w, x_sb_layout=x_sb_layout, name="router_decode_w_sb"
    )

    # Ensure w_sb matches x dtype (both must be the same for nc_matmul).
    # Always cast to x.dtype (f32 when router_mm_dtype=float32).

    # ---- Load and broadcast bias ----
    router_logits_bias_broadcasted_sb = None
    if has_bias:
        assert False

    # ---- Column-tiled matmul: x @ w -> router_logits_sb [T, 1, E] ----
    # Single t_tile since T <= 128
    if T <= PE_COLUMN_TILE_32:
        pe_array_column_tiling_size = PE_COLUMN_TILE_32
    elif T <= PE_COLUMN_TILE_64:
        pe_array_column_tiling_size = PE_COLUMN_TILE_64
    else:
        pe_array_column_tiling_size = PE_COLUMN_TILE_128

    num_pe_array_column_tiles = PE_COLUMN_TILE_128 // pe_array_column_tiling_size
    tile_size = (P_MAX, pe_array_column_tiling_size)

    router_logits_psum = nl.ndarray((P_MAX, E), nl.float32, buffer=nl.psum)
    nisa.memset(dst=router_logits_psum, value=0)
    for h_tile_idx in range(num_h_tiles):
        w_tile_sb = w_sb[:, h_tile_idx, :]
        x_tile_sb = x.ap(
            pattern=[[T * num_h_tiles, P_MAX], [num_h_tiles, T]],
            offset=h_tile_idx,
        )

        current_column_tile_index = h_tile_idx % num_pe_array_column_tiles
        current_column_tile_column_offset = (
            current_column_tile_index * pe_array_column_tiling_size
        )
        tile_position = (0, current_column_tile_column_offset)

        nisa.nc_matmul(
            dst=router_logits_psum[nl.ds(current_column_tile_column_offset, T), :],
            stationary=x_tile_sb,
            moving=w_tile_sb,
            tile_position=tile_position,
            tile_size=tile_size,
        )

    # Copy PSUM -> SBUF with optional bias add
    router_logits_sb1 = nl.ndarray((T, 1, E), dtype=nl.float32, buffer=nl.sbuf)
    if has_bias:
        nisa.tensor_tensor(
            dst=router_logits_sb1[:T, 0, :],
            data1=router_logits_psum[:T, :E],
            data2=router_logits_bias_broadcasted_sb[:T, :E],
            op=nl.add,
        )
    else:
        nisa.tensor_copy(
            dst=router_logits_sb1[:T, 0, :],
            src=router_logits_psum[:T, :E],
        )

    # Merge remaining column tiles
    for column_tile_idx in range(1, num_pe_array_column_tiles):
        current_column_tile_column_offset = (
            column_tile_idx * pe_array_column_tiling_size
        )
        nisa.tensor_tensor(
            dst=router_logits_sb1[:T, 0, :],
            data1=router_logits_sb1[:T, 0, :],
            data2=router_logits_psum[nl.ds(current_column_tile_column_offset, T), :],
            op=nl.add,
        )

    # Work around: removed redundant fp32->bf16 cast that was truncating router logits
    # before softmax. Keep fp32 throughout to match nkilib precision.
    # Original: router_logits_sb = nl.ndarray((T, 1, E), dtype=nl.bfloat16, ...)
    router_logits_sb = router_logits_sb1
    # ---- ACT1 + TopK path ----
    router_logits_2d = router_logits_sb.reshape((T, E))

    # --- Original path: full activation on [T, E], then topK ---
    expert_affinities_full_2d = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    negmax_sb = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(
        dst=negmax_sb,
        op=nl.maximum,
        data=router_logits_2d,
        axis=1,
        negate=True,
        keepdims=True,
    )
    exp_sum_sb = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    result_exp = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=result_exp,
        op=nl.exp,
        data=router_logits_2d,
        bias=negmax_sb,
        reduce_op=nl.add,
        reduce_res=exp_sum_sb,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    nisa.reciprocal(dst=exp_sum_sb, data=exp_sum_sb)
    nisa.tensor_scalar(
        dst=expert_affinities_full_2d,
        data=result_exp,
        op0=nl.multiply,
        operand0=exp_sum_sb,
    )

    # Reshape back to 3D for topK (consistent indexing)
    expert_affinities_full_sb = expert_affinities_full_2d.reshape((T, 1, E))

    # ---- TopK: max8 + nc_find_index8 ----
    topk_input_sb = expert_affinities_full_sb

    router_logits_topk_sb = nl.ndarray((T, 1, k), dtype=nl.float32, buffer=nl.sbuf)
    router_indexes_topk_sb = nl.ndarray((T, 1, k), dtype=nl.uint32, buffer=nl.sbuf)

    router_logits_top8_sb = nl.ndarray((T, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(
        dst=router_logits_top8_sb[:T, :],
        src=topk_input_sb[:T, 0, :],
    )
    nisa.tensor_copy(
        dst=router_logits_topk_sb[:T, 0, :],
        src=router_logits_top8_sb[:T, :k],
    )

    tmp_buffer = nl.ndarray((T, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(
        dst=tmp_buffer[:T, :],
        data=topk_input_sb[:T, 0, :],
        vals=router_logits_top8_sb,
    )
    nisa.tensor_copy(
        dst=router_indexes_topk_sb[:T, 0, :k],
        src=tmp_buffer[:T, :k],
    )

    # ---- Optional Norm: L1 normalize top-K values ----
    # Work around: was bf16, truncating L1-normed affinities. Keep fp32 to match nkilib.
    # Original: expert_affinities_topk_sb = nl.ndarray((T, 1, k), dtype=nl.bfloat16, ...)
    expert_affinities_topk_sb = nl.ndarray((T, 1, k), dtype=nl.float32, buffer=nl.sbuf)
    sum_of_max_sb = nl.ndarray((T, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(
        dst=sum_of_max_sb[:T, 0, :],
        op=nl.add,
        data=router_logits_topk_sb[:T, 0, :],
        axis=1,
        keepdims=True,
    )
    nisa.reciprocal(
        dst=sum_of_max_sb[:T, 0, :],
        data=sum_of_max_sb[:T, 0, :],
    )
    nisa.tensor_scalar(
        dst=expert_affinities_topk_sb[:T, 0, :],
        data=router_logits_topk_sb[:T, 0, :],
        op0=nl.multiply,
        operand0=sum_of_max_sb[:T, 0, :],
    )

    # Return [T, K] shaped results directly (squeeze the num_t_tiles=1 dim)
    expert_index_out = router_indexes_topk_sb.reshape((T, k))
    expert_affinities_topk_out = expert_affinities_topk_sb.reshape((T, k))

    return expert_index_out, expert_affinities_topk_out
