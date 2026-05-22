"""
QKV Projection TKG Kernel v5.

Hot-path-only QKV projection for token generation.

Supported configuration:
  - B=1, S=1
  - output_layout == BSD
  - HBM or SBUF output
  - norm_type in {NO_NORM, RMS_NORM}
  - fused_add == False
  - LNC1 and LNC2 for matmul-only and RMSNorm hot paths

Projection reuses the GEMV kernel from ops/ours/gemv.py.
"""

from typing import Optional, Tuple

import nki.isa as nisa
import nki.language as nl

from ..ours.gemv import gemv_kernel
from ..subkernels.rmsnorm_tkg import rmsnorm_tkg
from ..utils.allocator import SbufManager
from ..utils.common_types import NormType, QKVOutputLayout


I_TILE = 128
MAX_P = 128
MAX_PSUM_BANKS = 8
N_INNER = 4


def _dtype_size_bytes(dtype) -> int:
    dtype_str = str(dtype)
    if dtype_str == str(nl.float32):
        return 4
    if dtype_str in (str(nl.bfloat16), str(nl.float16), str(nl.uint16)):
        return 2
    if dtype_str in (str(nl.int8), str(nl.uint8), "float8e4", "float8_e4m3"):
        return 1
    if dtype_str in (str(nl.int32), str(nl.uint32)):
        return 4
    assert False, f"Unsupported dtype for v5 hot path: {dtype}"


def _gemv_hidden_sbuf_offset(i_dim: int, dtype) -> int:
    num_i_tiles = i_dim // I_TILE
    num_batches = (num_i_tiles + MAX_PSUM_BANKS - 1) // MAX_PSUM_BANKS
    elem_bytes = _dtype_size_bytes(dtype)
    w_total = num_batches * N_INNER * MAX_PSUM_BANKS * I_TILE * elem_bytes
    out_total = num_i_tiles * I_TILE * elem_bytes
    return w_total + out_total + 2


def _tile_size_bytes(shape, dtype) -> int:
    size = _dtype_size_bytes(dtype)
    for dim in shape[1:]:
        size *= dim
    return size


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _load_hidden_to_gemv_sbuf(
    hidden_hbm: nl.ndarray,
    hidden_sb: nl.ndarray,
    h1_shard_offset: int = 0,
) -> nl.ndarray:
    bsz, seqlen, hidden_dim = hidden_hbm.shape
    bxs = bsz * seqlen
    h0 = nl.tile_size.pmax
    h1 = hidden_dim // h0

    hidden_hbm_r = hidden_hbm.reshape((bxs, h0, h1))
    _, _, h1_shard = hidden_sb.shape
    hidden_hbm_pattern = [[h1, h0], [hidden_dim, bxs], [1, h1_shard]]
    nisa.dma_copy(
        hidden_sb,
        hidden_hbm_r.ap(pattern=hidden_hbm_pattern, offset=h1_shard_offset),
    )
    return hidden_sb


def _reduce_lnc2_sbuf_output(output_sub: nl.ndarray, gemv_sbuf: nl.ndarray, shard_id: int) -> None:
    """Reduce H-sharded partial output across the two LNC cores in-place."""
    output_recv_sb = nl.ndarray(
        gemv_sbuf.shape,
        dtype=gemv_sbuf.dtype,
        buffer=nl.sbuf,
        name="qkv_output_reduce_recv",
    )
    other_core = 1 - shard_id
    nisa.sendrecv(
        src=gemv_sbuf,
        dst=output_recv_sb,
        send_to_rank=other_core,
        recv_from_rank=other_core,
        pipe_id=0,
    )
    nisa.tensor_tensor(dst=output_sub, data1=gemv_sbuf, data2=output_recv_sb, op=nl.add)


def _store_sbuf_to_bsd_hbm(output_hbm: nl.ndarray, output_sbuf: nl.ndarray, bsz: int, seqlen: int) -> None:
    """Store a contiguous (B*S, I) SBUF QKV result into a (B, S, I) HBM tensor."""
    _, i_dim = output_sbuf.shape
    nisa.dma_copy(dst=output_hbm.reshape((bsz * seqlen, i_dim)), src=output_sbuf)


def _store_i_shard_sbuf_to_bsd_hbm(
    output_hbm: nl.ndarray,
    output_sbuf: nl.ndarray,
    bsz: int,
    seqlen: int,
    i_start_tile: int,
    i_num_tiles: int,
) -> None:
    """Store the current LNC core's I-sharded slice into a shared (B, S, I) HBM tensor."""
    i_start = i_start_tile * I_TILE
    i_size = i_num_tiles * I_TILE
    output_hbm_flat = output_hbm.reshape((bsz * seqlen, output_sbuf.shape[1]))
    nisa.dma_copy(
        dst=output_hbm_flat[:, nl.ds(i_start, i_size)],
        src=output_sbuf[:, nl.ds(i_start, i_size)],
    )


def qkv_tkg(
    hidden: nl.ndarray,
    qkv_w: nl.ndarray,
    norm_w: Optional[nl.ndarray] = None,
    fused_add: bool = False,
    mlp_prev: Optional[nl.ndarray] = None,
    attn_prev: Optional[nl.ndarray] = None,
    d_head: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.RMS_NORM,
    hidden_actual: Optional[int] = None,
    output_in_sbuf: bool = False,
    shard_output_i: bool = False,
    sbm=None,
) -> nl.ndarray | Tuple[nl.ndarray, nl.ndarray]:
    assert not fused_add, "qkv_tkgv5 hot path does not support fused_add"
    assert mlp_prev is None, "qkv_tkgv5 hot path does not use mlp_prev"
    assert attn_prev is None, "qkv_tkgv5 hot path does not use attn_prev"
    assert d_head is None, "qkv_tkgv5 hot path does not use d_head"
    assert num_kv_heads is None, "qkv_tkgv5 hot path does not use num_kv_heads"
    assert num_q_heads is None, "qkv_tkgv5 hot path does not use num_q_heads"
    if output_in_sbuf:
        assert sbm is not None, "qkv_tkgv5 output_in_sbuf requires an SBUF manager"
    if shard_output_i:
        assert not output_in_sbuf, "qkv_tkgv5 shard_output_i writes QKV to shared HBM"
    assert output_layout == QKVOutputLayout.BSD, "qkv_tkgv5 hot path only supports BSD output"
    if norm_type != NormType.NO_NORM and norm_type != NormType.RMS_NORM:
        assert False, "qkv_tkgv5 hot path only supports NO_NORM and RMS_NORM"

    grid_ndim = nl.program_ndim()
    if grid_ndim is None:
        grid_ndim = 0
    if grid_ndim != 0:
        num_shards = nl.num_programs(axes=0)
        shard_id = nl.program_id(axis=0)
    else:
        num_shards = 1
        shard_id = 0
    bsz, seqlen, hidden_dim = hidden.shape
    _, i_dim = qkv_w.shape
    h0 = nl.tile_size.pmax
    h1 = hidden_dim // h0
    io_dtype = hidden.dtype

    assert bsz == 1 and seqlen == 1, f"qkv_tkgv5 hot path only supports B=1,S=1, got B={bsz}, S={seqlen}"
    assert hidden_dim % h0 == 0, f"H must be divisible by {h0}, got {hidden_dim}"
    assert qkv_w.shape[0] == hidden_dim, f"Weight H must match hidden H, got {qkv_w.shape[0]} vs {hidden_dim}"
    assert i_dim % I_TILE == 0, f"I must be divisible by {I_TILE}, got {i_dim}"
    assert h1 % num_shards == 0, f"H tile count must be divisible by num_shards, got H1={h1}, num_shards={num_shards}"

    if hidden_actual is None:
        hidden_actual = hidden_dim

    h1_shard = h1 if shard_output_i else h1 // num_shards
    h1_shard_offset = 0 if shard_output_i else shard_id * h1_shard
    assert h1_shard % N_INNER == 0, f"H tile count per shard must be divisible by {N_INNER}, got {h1_shard}"

    rmsnorm_gemv_layout = norm_type == NormType.RMS_NORM and num_shards > 1
    rmsnorm_h_shard_layout = rmsnorm_gemv_layout and not shard_output_i
    hidden_sb_h1 = h1 if norm_type == NormType.RMS_NORM else h1_shard
    hidden_offset = _gemv_hidden_sbuf_offset(i_dim, io_dtype)
    hidden_sb = nl.ndarray(
        (h0, 1, hidden_sb_h1),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name="hidden_sb",
        address=(0, hidden_offset),
    )

    if norm_type == NormType.NO_NORM:
        _load_hidden_to_gemv_sbuf(
            hidden_hbm=hidden,
            hidden_sb=hidden_sb,
            h1_shard_offset=h1_shard_offset,
        )
    else:
        assert norm_w is not None, "norm_w must be provided for RMS_NORM"
        # Work around compiler allocation bugs when combining the v4 RMSNorm temp-buffer
        # route with the manual-address GEMV hot path. Use the existing RMSNorm subkernel
        # to write directly into the GEMV input layout.
        rmsnorm_sbm = SbufManager(
            sb_lower_bound=_align_up(hidden_offset + _tile_size_bytes((h0, 1, hidden_sb_h1), io_dtype), 4),
            sb_upper_bound=nl.tile_size.total_available_sbuf_size,
            use_auto_alloc=False,
        )
        rmsnorm_tkg(
            input=hidden,
            gamma=norm_w,
            output=hidden_sb,
            eps=eps,
            hidden_actual=hidden_actual,
            output_num_h_shards=1 if rmsnorm_gemv_layout else None,
            sbm=rmsnorm_sbm,
        )

    output = nl.ndarray((bsz * seqlen, i_dim), dtype=nl.float32, buffer=nl.sbuf, name="qkv_output_sb")
    output_offset = 0
    if shard_output_i:
        i_tiles_per_shard = (i_dim // I_TILE) // num_shards
        i_start_tile = shard_id * i_tiles_per_shard
        gemv_kernel(
            hidden_sb,
            qkv_w,
            output,
            h_start_tile=0,
            weight_num_h_tiles=h1,
            output_offset=output_offset,
            i_start_tile=i_start_tile,
            i_num_tiles=i_tiles_per_shard,
        )
    else:
        gemv_kernel(
            hidden_sb,
            qkv_w,
            output,
            h_start_tile=h1_shard_offset,
            hidden_start_tile=h1_shard_offset if rmsnorm_h_shard_layout else 0,
            hidden_num_tiles=h1_shard if rmsnorm_h_shard_layout else 0,
            weight_num_h_tiles=h1,
            output_offset=output_offset,
        )
        if num_shards > 1:
            assert num_shards == 2, f"qkv_tkgv5 SBUF output reduce only supports LNC2, got {num_shards}"
            final_output = nl.ndarray(
                (bsz * seqlen, i_dim),
                dtype=nl.float32,
                buffer=nl.sbuf,
                name="qkv_output_sb_bf16",
            )
            _reduce_lnc2_sbuf_output(final_output, output, shard_id)
            output = final_output

    if output_in_sbuf:
        return output

    assert False