"""Output projection v2 specialized for the Qwen3 TKG hot path."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl
from nki.language import static_range

from ..utils.kernel_assert import kernel_assert


def _program_info():
    grid_ndim = nl.program_ndim()
    if grid_ndim is None:
        grid_ndim = 0
    if grid_ndim != 0:
        return nl.num_programs(axes=0), nl.program_id(axis=0)
    return 1, 0


def o_projection_v2(
    attention: nl.ndarray,
    weight: nl.ndarray,
    TRANSPOSE_OUT: bool = False,
    OUT_IN_SB: bool = False,
    name_prefix: str = "",
    sbm: Optional[object] = None,
):
    """Compute output projection for the Qwen3 TKG hot path.

    Supported shape:
      attention: [D=128, B=1, N=8, S=1]
      weight:    [N * D = 1024, H = 2048]
      output:    [B * S = 1, H]

    This follows the selective down GEMV shape: output H tiles are processed
    in PSUM-bank batches, and each batch accumulates across all input tiles.
    """
    d_size, b_size, n_size, s_size = attention.shape
    input_dim, output_dim = weight.shape
    n_prgs, prg_id = _program_info()

    kernel_assert(not TRANSPOSE_OUT, "o_projection_v2 does not support TRANSPOSE_OUT")
    kernel_assert(b_size == 1 and s_size == 1, "o_projection_v2 only supports B=1, S=1")
    kernel_assert(d_size == nl.tile_size.pmax, "o_projection_v2 requires D == pmax")
    kernel_assert(input_dim == n_size * d_size, f"o_projection_v2 weight input dim must match N * D ({input_dim} != {n_size} * {d_size})")
    kernel_assert(input_dim % nl.tile_size.pmax == 0, "o_projection_v2 input dim must be divisible by pmax")
    kernel_assert(output_dim % nl.tile_size.pmax == 0, "o_projection_v2 output dim must be divisible by pmax")
    kernel_assert((output_dim // nl.tile_size.pmax) % n_prgs == 0, "o_projection_v2 output tiles must divide LNC")
    kernel_assert(n_prgs == 1 or not OUT_IN_SB, "o_projection_v2 LNC2 only supports HBM output")

    H0 = nl.tile_size.pmax
    I0 = nl.tile_size.pmax
    T = b_size * s_size
    MAX_PSUM_BANKS = 4
    PSUM_FMAX = nl.tile_size.psum_fmax

    hidden_h1 = input_dim // nl.tile_size.pmax
    output_tiles = output_dim // nl.tile_size.pmax
    output_tiles_per_prg = output_tiles // n_prgs
    prg_h_start_tile = prg_id * output_tiles_per_prg
    local_output_dim = output_tiles_per_prg * H0
    num_h_batches = (output_tiles_per_prg + MAX_PSUM_BANKS - 1) // MAX_PSUM_BANKS

    if sbm is not None:
        output_sb = sbm.alloc_stack(
            (T, local_output_dim if not OUT_IN_SB else output_dim),
            dtype=attention.dtype,
            buffer=nl.sbuf,
            name=f"{name_prefix}o_projection_v2_out_sb",
        )
    else:
        output_sb = nl.ndarray(
            (T, local_output_dim if not OUT_IN_SB else output_dim),
            dtype=attention.dtype,
            buffer=nl.sbuf,
            name=f"{name_prefix}o_projection_v2_out_sb",
        )

    for batch_idx in static_range(num_h_batches):
        h_start_tile = batch_idx * MAX_PSUM_BANKS
        h_end_tile = min(h_start_tile + MAX_PSUM_BANKS, output_tiles_per_prg)
        batch_h_tiles = h_end_tile - h_start_tile
        batch_h_size = batch_h_tiles * H0
        global_h_start = (prg_h_start_tile + h_start_tile) * H0
        w_all_free = hidden_h1 * batch_h_size

        if sbm is not None:
            w_all_sb = sbm.alloc_stack(
                (I0, w_all_free),
                dtype=weight.dtype,
                buffer=nl.sbuf,
                name=f"{name_prefix}o_projection_v2_w_batch{batch_idx}",
            )
            out_batch_sb = sbm.alloc_stack(
                (H0, batch_h_tiles, T),
                dtype=attention.dtype,
                buffer=nl.sbuf,
                name=f"{name_prefix}o_projection_v2_out_batch{batch_idx}",
            )
        else:
            w_all_sb = nl.ndarray(
                (I0, w_all_free),
                dtype=weight.dtype,
                buffer=nl.sbuf,
                name=f"{name_prefix}o_projection_v2_w_batch{batch_idx}",
            )
            out_batch_sb = nl.ndarray(
                (H0, batch_h_tiles, T),
                dtype=attention.dtype,
                buffer=nl.sbuf,
                name=f"{name_prefix}o_projection_v2_out_batch{batch_idx}",
            )

        psum_all = nl.ndarray(
            (H0, MAX_PSUM_BANKS, PSUM_FMAX),
            dtype=nl.float32,
            buffer=nl.psum,
            name=f"{name_prefix}o_projection_v2_psum_batch{batch_idx}",
        )
        psum_tiles = []
        for b in static_range(batch_h_tiles):
            psum_tiles.append(psum_all[:, b : b + 1, 0:PSUM_FMAX])


        for i_idx in static_range(hidden_h1):
            tile_i_start = i_idx * I0
            nisa.dma_copy(
                dst=w_all_sb.ap(
                    pattern=[[w_all_free, I0], [1, batch_h_size]],
                    offset=i_idx * batch_h_size,
                ),
                src=weight.ap(
                    pattern=[[output_dim, I0], [1, batch_h_size]],
                    offset=tile_i_start * output_dim + global_h_start,
                ),
                dge_mode=nisa.dge_mode.swdge,
            )

        for i_idx in static_range(hidden_h1):
            for b in static_range(batch_h_tiles):
                nisa.nc_matmul(
                    dst=psum_tiles[b][0:H0, 0, 0:T],
                    stationary=w_all_sb.ap(
                        pattern=[[w_all_free, I0], [1, H0]],
                        offset=i_idx * batch_h_size + b * H0,
                    ),
                    moving=attention[0:I0, 0, i_idx, 0:T],
                )

        nisa.activation(
            dst=out_batch_sb[0:H0, 0:batch_h_tiles, 0:T],
            op=nl.copy,
            data=psum_all[0:H0, 0:batch_h_tiles, 0:T],
        )

        # T is fixed to 1 in this hot path, so keep the output in H-tile layout
        # and write each tile directly to HBM. This avoids a PSUM->PSUM
        # nc_transpose pattern that can make neuronx-cc's walrus pass crash.
        for b in static_range(batch_h_tiles):
            transpose_psum = nl.ndarray(
                (T, H0),
                dtype=attention.dtype,
                buffer=nl.psum,
                name=f"{name_prefix}o_projection_v2_transpose_b{batch_idx}_h{b}",
            )
            nisa.nc_transpose(
                dst=transpose_psum[0:T, 0:H0],
                data=out_batch_sb[0:H0, b, 0:T],
            )
            nisa.tensor_copy(
                dst=output_sb[0:T, nl.ds(h_start_tile * H0 + b * H0, H0)],
                src=transpose_psum[0:T, 0:H0],
            )

    if OUT_IN_SB:
        return output_sb

    output = nl.ndarray(
        (T, output_dim),
        dtype=attention.dtype,
        buffer=nl.shared_hbm,
        name=f"{name_prefix}o_projection_v2_out",
    )
    nisa.dma_copy(
        dst=output.ap(
            pattern=[[output_dim, T], [1, local_output_dim]],
            offset=prg_h_start_tile * H0,
        ),
        src=output_sb[0:T, 0:local_output_dim],
    )
    return output
