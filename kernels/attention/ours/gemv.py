"""
Standalone GEMV kernel extracted from qkv_tkgv3.

Computes: output = input @ weight  (i.e. [1, 1, H] @ [H, I] -> [1, I])

Specialized for BxS=1 (token generation / decoding).
No RMSNorm, no residual add -- pure matrix-vector multiply.

The kernel uses the "flipped operand" strategy from v3:
  - weight (128x128) as stationary operand (fills full systolic PE array)
  - hidden (128x1) as moving operand
  - nc_matmul: dst = stationary.T @ moving = weight.T @ hidden

Input is loaded into SBUF with interleaved layout (128, 1, H1).
Weight is loaded with matching interleaved access pattern.
Output is written to HBM in (I, 1) layout, or to SBUF in (1, I) layout
when the caller provides an SBUF output tensor.

Usage:
  python ops/ours/gemv.py                          # default dims from config
  python ops/ours/gemv.py --tp 4                   # simulate TP=4 sharding
  python ops/ours/gemv.py --H 512 --I 1280         # custom dimensions
"""

import argparse
import json
import math
import os

import nki
import nki.isa as nisa
import nki.language as nl

import torch
import torch_xla.core.xla_model as xm


# ---------------------------------------------------------------------------
# NKI Kernel
# ---------------------------------------------------------------------------

def gemv_kernel(
    hidden_sb: nl.ndarray,
    weight_hbm: nl.ndarray,
    output: nl.ndarray,
    h_start_tile: int = 0,
    hidden_start_tile: int = 0,
    hidden_num_tiles: int = 0,
    weight_num_h_tiles: int = 0,
    output_offset: int = 0,
    i_start_tile: int = 0,
    i_num_tiles: int = 0,
    weight_h_tiles_contiguous: bool = False,
    name_prefix: str = "",
):
    """
    GEMV kernel: output = hidden @ weight, for BxS=1.

    H-outer I-inner tiling with per-I-tile PSUM banks:
    - Each I tile gets its own PSUM bank (independent, no data dependency)
    - Outer loop over H tiles: same H slice reused across all I tiles
    - Inner loop over I tiles: nc_matmul accumulates into bank[i] (H auto-sums)
    - I tiles processed in batches of ≤8, one PSUM bank per I tile
    - All tensors use manual address= for consistent allocation

    Args:
        hidden_sb:  SBUF tensor, shape (H0=128, 1, H1) where H = H0 * H1.
                    Partition dim = 128, free dim = 1, H1 tiles interleaved.
        weight_hbm: HBM tensor, shape (H, I).
        output: HBM tensor with shape (I, 1), or SBUF tensor with shape
                (1, I). HBM output uses (I, BxS) layout; SBUF output uses
                (BxS, I) layout for direct downstream consumption.
        h_start_tile: starting H tile in weight_hbm for H-sharded execution.
        hidden_start_tile: starting H tile in hidden_sb. Used when hidden_sb
                contains the full H dimension but this call computes one
                H-sharded partial projection.
        hidden_num_tiles: number of H tiles to read from hidden_sb. Defaults to
                all tiles from hidden_start_tile.
        weight_num_h_tiles: full H tile count in weight_hbm. Defaults to hidden_sb H tile count.
        output_offset: flat element offset in output_hbm for H-sharded execution.
        i_start_tile: starting I tile in weight_hbm/output for I-sharded execution.
        i_num_tiles: number of I tiles to compute. Defaults to the full remaining I range.
        weight_h_tiles_contiguous: when True, weight rows are grouped as
                row = h_tile * 128 + p. The default QKV path uses
                row = p * H_tiles + h_tile.
        name_prefix: prefix for temporary tensor names when multiple GEMV
                calls are traced inside the same NKI kernel.
    """
    H0 = 128  # nl.tile_size.pmax
    BxS = 1
    N_INNER = 4  # H-tile inner group size for Qwen3 TKG shape

    _, _, H1 = hidden_sb.shape
    H, I = weight_hbm.shape
    I_TILE = 128

    NUM_H_TILES = H1 - hidden_start_tile
    if hidden_num_tiles != 0:
        NUM_H_TILES = hidden_num_tiles
    FULL_H_TILES = H // H0
    if weight_num_h_tiles != 0:
        FULL_H_TILES = weight_num_h_tiles
    FULL_NUM_I_TILES = I // I_TILE
    I_TILE_BASE = i_start_tile
    NUM_I_TILES = FULL_NUM_I_TILES - I_TILE_BASE
    if i_num_tiles != 0:
        NUM_I_TILES = i_num_tiles
    MAX_PSUM_BANKS = 8  # 8 hardware PSUM banks
    assert hidden_start_tile + NUM_H_TILES <= H1, "Hidden H tile range exceeds hidden_sb H dimension"
    assert NUM_H_TILES % N_INNER == 0, f"H tile count must be divisible by {N_INNER}, got {NUM_H_TILES}"
    assert I_TILE_BASE + NUM_I_TILES <= FULL_NUM_I_TILES, "I tile range exceeds weight/output I dimension"
    NUM_H_OUTER = NUM_H_TILES // N_INNER

    # ---- Manual SBUF allocation ----
    # Weight buffer: one H-group per batch. Shape is (128, N_INNER, batch_I_size)
    # so one DMA covers N_INNER H tiles instead of launching one DMA per H tile.
    SBUF_W_GROUP_SIZE = N_INNER * MAX_PSUM_BANKS * I_TILE * 2  # max bytes per H-group buffer
    SBUF_W_OFFSET = 0
    num_batches = (NUM_I_TILES + MAX_PSUM_BANKS - 1) // MAX_PSUM_BANKS
    # Total: num_batches * SBUF_W_GROUP_SIZE

    # out_sb: single buffer to collect ALL I tiles' results, then one DMA write
    # Shape: (I_TILE=128, NUM_I_TILES) bf16, each column is one I tile's output
    SBUF_OUT_OFFSET = SBUF_W_OFFSET + num_batches * SBUF_W_GROUP_SIZE

    # ---- Manual PSUM allocation ----
    PSUM_FMAX = nl.tile_size.psum_fmax
    if PSUM_FMAX <= 0:
        PSUM_FMAX = 512

    # Single output buffer: activation writes to column i_idx via ap pattern
    out_dtype = output.dtype
    out_sb = nl.ndarray(
        (I_TILE, NUM_I_TILES), dtype=out_dtype, buffer=nl.sbuf,
        name=f"{name_prefix}out_sb",
        address=(0, SBUF_OUT_OFFSET),
    )

    # Process I tiles in batches of MAX_PSUM_BANKS
    for batch_idx in range(num_batches):
        batch_i_start_tile = batch_idx * MAX_PSUM_BANKS
        batch_i_end_tile = min(batch_i_start_tile + MAX_PSUM_BANKS, NUM_I_TILES)
        batch_size = batch_i_end_tile - batch_i_start_tile
        batch_I_size = batch_size * I_TILE

        # One grouped weight buffer per batch. Each h_outer iteration loads
        # N_INNER H tiles in a single DMA, then AP slices feed nc_matmul.
        w_group_sb = nl.ndarray(
            (H0, N_INNER, batch_I_size), dtype=weight_hbm.dtype, buffer=nl.sbuf,
            name=f"{name_prefix}w_group_sb_batch{batch_idx}",
            address=(0, SBUF_W_OFFSET + batch_idx * SBUF_W_GROUP_SIZE),
        )

        # Pack all I-tile accumulators for this batch into one PSUM tensor.
        # The second dim selects the PSUM bank, so the final copy can move the
        # whole batch instead of issuing one activation per I tile.
        # psum_all = nl.ndarray(
        #     (I_TILE, MAX_PSUM_BANKS, PSUM_FMAX),
        #     dtype=nl.float32,
        #     buffer=nl.psum,
        #     name=f"{name_prefix}psum_batch{batch_idx}",
        #     address=(0, 0),
        # )
        psum_tiles = []
        for b in range(batch_size):
            # psum_tiles.append(psum_all[:, b : b + 1, 0:PSUM_FMAX])
            psum_tiles.append(nl.ndarray(
                (I_TILE, 1, PSUM_FMAX),
                dtype=nl.float32,
                buffer=nl.psum,
                name=f"{name_prefix}psum_batch{batch_idx}_tile{b}",
                address=(0, b*2048),  # TODO: manual PSUM address management if we want to reuse PSUM across batches
            ))
            nisa.memset(psum_tiles[-1], value=0)

        # ---- H-outer loop: process N_INNER H-tiles per iteration ----
        for h_outer in nl.sequential_range(NUM_H_OUTER):
            h_base = h_outer * N_INNER

            # DMA: load N_INNER H-tiles into SBUF as one larger transfer.
            global_i_start_tile = I_TILE_BASE + batch_i_start_tile
            if weight_h_tiles_contiguous:
                # Source element: weight[(h_start_tile + h_base + hi) * 128 + p, i + col]
                w_hbm_pattern = [[I, H0], [I * H0, N_INNER], [1, batch_I_size]]
                w_hbm_offset = (h_start_tile + h_base) * H0 * I + global_i_start_tile * I_TILE
            else:
                # Source element: weight[(h_start_tile + h_base + hi) + p * FULL_H_TILES, i + col]
                w_hbm_pattern = [[I * FULL_H_TILES, H0], [I, N_INNER], [1, batch_I_size]]
                w_hbm_offset = (h_start_tile + h_base) * I + global_i_start_tile * I_TILE

            w_dst_pattern = [[N_INNER * batch_I_size, H0], [batch_I_size, N_INNER], [1, batch_I_size]]
            nisa.dma_copy(
                w_group_sb.ap(pattern=w_dst_pattern, offset=0),
                weight_hbm.ap(pattern=w_hbm_pattern, offset=w_hbm_offset),
            )

            # Compute: N_INNER H-tiles × batch_size I-tiles
            for hi in nl.sequential_range(N_INNER):
                h_idx = h_base + hi
                hidden_pattern = [[BxS * H1, H0], [H1, BxS]]
                hidden_offset = hidden_start_tile + h_idx

                for b in nl.sequential_range(batch_size):
                    w_sb_pattern = [[N_INNER * batch_I_size, H0], [1, I_TILE]]
                    w_sb_offset = hi * batch_I_size + b * I_TILE

                    nisa.nc_matmul(
                        psum_tiles[b][0:I_TILE, 0, 0:BxS],
                        w_group_sb.ap(pattern=w_sb_pattern, offset=w_sb_offset),
                        hidden_sb.ap(pattern=hidden_pattern, offset=hidden_offset),
                        tile_position=(0, 0),
                        tile_size=(H0, I_TILE),
                    )
        for b in range(batch_size):
            nisa.tensor_copy(
                dst=out_sb[0:I_TILE, batch_i_start_tile + b],
                src=psum_tiles[b][:, 0, 0:BxS],
                engine=nisa.scalar_engine
            )
            
        # ---- PSUM → SBUF: copy the whole I-tile batch (no HBM DMA yet) ----
        # psum_batch_view = psum_all[0:I_TILE, 0:batch_size, 0:BxS]
        # nisa.activation(
        #     dst=out_sb[0:I_TILE, batch_i_start_tile:batch_i_start_tile + batch_size],
        #     op=nl.copy,
        #     data=psum_batch_view,
        # )

    # SBUF layout: out_sb[p, i] = result of I_tile i, partition p (p=0..127)
    # SBUF ap: stride NUM_I_TILES per partition row, count I_TILE; stride 1, count NUM_I_TILES
    out_sb_pattern  = [[NUM_I_TILES, I_TILE], [1, NUM_I_TILES]]

    if output.buffer == nl.sbuf:
        # SBUF output must be (BxS, I). Transpose one I tile at a time from
        # the GEMV-native (I_TILE, BxS) layout into the downstream QKV layout.
        for i_idx in range(NUM_I_TILES):
            global_i_idx = I_TILE_BASE + i_idx
            output_tile = nl.ndarray(
                (BxS, I_TILE),
                dtype=out_dtype,
                buffer=nl.psum,
                name=f"output_sbuf_transpose_i{global_i_idx}",
            )
            out_sb_tile_pattern = [[NUM_I_TILES, I_TILE], [1, BxS]]
            nisa.nc_transpose(
                dst=output_tile,
                data=out_sb.ap(pattern=out_sb_tile_pattern, offset=i_idx),
            )
            nisa.tensor_copy(dst=output[:, nl.ds(global_i_idx * I_TILE, I_TILE)], src=output_tile)
    else:
        assert False

    return output


# ---------------------------------------------------------------------------
# nki.jit entry point
# ---------------------------------------------------------------------------

@nki.jit(mode="auto", debug_kernel=True, show_compiler_tb=True)
def gemv_nki(input_tensor: nl.ndarray, weight: nl.ndarray):
    """
    Top-level NKI entry point for GEMV.

    Args:
        input_tensor: HBM tensor, shape (1, H) or (1, 1, H) bf16 -- BxS=1
        weight:       HBM tensor, shape (H, I) bf16

    Returns:
        output: HBM tensor, shape (I, 1) bf16. Caller reshapes to (1, I) or (1, 1, I).
    """
    input_shape = input_tensor.shape
    if len(input_shape) == 2:
        BxS, H = input_shape
    else:
        B, S, H = input_shape
        BxS = B * S
    _H, I = weight.shape
    assert H == _H, f"Hidden dim mismatch: input H={H} vs weight H={_H}"
    H0 = 128  # nl.tile_size.pmax
    H1 = H // H0
    grid_ndim = nl.program_ndim()
    if grid_ndim is None:
        grid_ndim = 0
    if grid_ndim != 0:
        num_shards = nl.num_programs(axes=0)
        shard_id = nl.program_id(axis=0)
    else:
        num_shards = 1
        shard_id = 0
    # Load input from HBM to SBUF with interleaved layout:
    # (1, 1, H) -> reshape (1, 128, H1) -> SBUF (128, 1, H1)
    # hidden_sb[p, 0, h1] = input[p * H1 + h1]
    # Manual SBUF address: placed after gemv_kernel's buffers
    #   w_group_sb: num_batches grouped buffers, each max N_INNER*8*128*2 bytes
    #   out_sb: (128, NUM_I_TILES) = NUM_I_TILES * 128 * 2 bytes
    #   MAX_PSUM_BANKS = 8
    NUM_I_TILES = I // 128
    N_INNER = 4
    num_batches_est = (NUM_I_TILES + 7) // 8  # ceil(NUM_I_TILES / 8)
    w_total = num_batches_est * N_INNER * 8 * 128 * 2
    out_total = NUM_I_TILES * 128 * 2
    SBUF_HIDDEN_OFFSET = w_total + out_total + 2
    hidden_sb = nl.ndarray(
        (H0, BxS, H1), dtype=input_tensor.dtype, buffer=nl.sbuf, name="hidden_sb",
        address=(0, SBUF_HIDDEN_OFFSET),
    )
    hidden_hbm_reshaped = input_tensor.reshape((BxS, H0, H1))  # works for both 2D and 3D
    hbm_pattern = [[H1, H0], [H, BxS], [1, H1]]
    nisa.dma_copy(hidden_sb, hidden_hbm_reshaped.ap(pattern=hbm_pattern, offset=0))

    # Allocate output in HBM: (I, 1) layout
    output_hbm = nl.ndarray((I, BxS), dtype=nl.float32, buffer=nl.shared_hbm, name="gemv_output")

    i_start_tile = (NUM_I_TILES // num_shards) * shard_id
    i_num_tiles = NUM_I_TILES // num_shards
    gemv_kernel(hidden_sb, weight, output_hbm, i_start_tile=i_start_tile, i_num_tiles=i_num_tiles)

    return output_hbm


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def load_model_dims(config_path=None):
    if config_path is None:
        config_path = os.path.join(_project_root(), "qwen3_config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    H = cfg["hidden_size"]
    n_q = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    d_head = cfg["head_dim"]
    I = (n_q + 2 * n_kv) * d_head
    return H, I, n_q, n_kv, d_head


def compute_metrics(ref, test, name):
    diff = torch.abs(ref.float() - test.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_abs = torch.abs(ref.float())
    nonzero_mask = ref_abs > 1e-8
    if nonzero_mask.any():
        rel_err = diff[nonzero_mask] / ref_abs[nonzero_mask]
        max_rel = rel_err.max().item()
        mean_rel = rel_err.mean().item()
    else:
        max_rel = float('nan')
        mean_rel = float('nan')
    print(f"  [{name}]")
    print(f"    abs: max={max_diff:.6e}  mean={mean_diff:.6e}")
    print(f"    rel: max={max_rel:.6e}  mean={mean_rel:.6e}")


def main():
    parser = argparse.ArgumentParser(description="Standalone GEMV kernel test")
    parser.add_argument("--H", type=int, default=None, help="Hidden dim (default: from config / tp)")
    parser.add_argument("--I", type=int, default=None, dest="I_dim", help="Output dim (default: from config / tp)")
    parser.add_argument("--tp", type=int, default=1, help="Simulate TP sharding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.H is not None and args.I_dim is not None:
        H, I = args.H, args.I_dim
    else:
        H_cfg, I_cfg, n_q, n_kv, d_head = load_model_dims()
        H = args.H if args.H is not None else H_cfg
        I = args.I_dim if args.I_dim is not None else I_cfg
        if args.tp > 1:
            n_q = n_q // args.tp
            n_kv = n_kv // args.tp
            I = (n_q + 2 * n_kv) * d_head
            H = H // args.tp

    assert H % 128 == 0, f"H={H} not divisible by 128"
    assert I % 128 == 0, f"I={I} not divisible by 128"

    print("=" * 60)
    print(f"GEMV Kernel Test: (1, 1, {H}) @ ({H}, {I}) = (1, {I})")
    print(f"  H tiles={H // 128}, I tiles={I // 128}, tp={args.tp}")
    print("=" * 60)

    device = xm.xla_device()
    torch.manual_seed(args.seed)
    h_scale = 100

    hidden_cpu = torch.randn(1, 1, H, dtype=torch.bfloat16) * h_scale
    w_cpu = torch.randn(H, I, dtype=torch.bfloat16) * h_scale

    ref_native = torch.matmul(hidden_cpu, w_cpu)
    ref_fp32 = torch.matmul(hidden_cpu.float(), w_cpu.float()).to(torch.bfloat16)

    hidden_dev = hidden_cpu.to(device)
    w_dev = w_cpu.to(device)
    xm.mark_step()

    print("\nRunning NKI kernel...")
    nki_out = gemv_nki(input_tensor=hidden_dev, weight=w_dev)
    xm.mark_step()

    nki_result = nki_out.cpu().reshape(1, 1, I)

    # Per-tile diagnostics
    print("\n  [Per I-tile diagnostics]")
    for t in range(I // 128):
        s, e = t * 128, (t + 1) * 128
        diff = torch.abs(ref_native[0, 0, s:e].float() - nki_result[0, 0, s:e].float())
        print(f"    tile {t} [{s}:{e}]: max_abs={diff.max().item():.6e}")

    print()
    compute_metrics(ref_native, nki_result, "NKI vs native_gt (bf16)")
    compute_metrics(ref_fp32, nki_result, "NKI vs fp32_gt")

    print("\nDONE")


if __name__ == "__main__":
    main()
