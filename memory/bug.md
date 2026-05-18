# Bug Record

## Bug 1: SBUF interleave layout mismatch

**Principle**: When a dimension exceeds 128 (SBUF partition max), it must be reshaped into `[128, N]` and placed across the free dimension. This reshape causes an interleaved layout — the free dim index changes fastest. All tensors sharing that dimension must use the **same reshape pattern** before slicing, otherwise the index correspondence breaks.

**Example**: H=768 → reshape to `[128, 6]`. Selecting tile `h_idx=0` gives elements at indices `[0, 6, 12, ..., 762]` (stride=6), NOT `[0, 1, 2, ..., 127]` (contiguous). If tensor A uses reshape+slice but tensor B uses contiguous slice, their H indices don't match in matmul.

**Checklist**:
- When two tensors participate in the same matmul along a shared dimension, verify both use the same reshape/slice pattern
- Contiguous slice `[h_idx*128 : (h_idx+1)*128]` is WRONG when the other operand uses interleaved reshape
- Correct: `reshape_dim(dim, shape=(128, N)).slice(tile_dim, h_idx)` on both operands

---

## Bug 2: nc_transpose direction determines max tile size

**Principle**: `nisa.nc_transpose` has different hardware limits depending on direction:
- **SBUF -> PSUM** (Tensor Engine): max `[128, 128]`
- **PSUM -> SBUF** (Vector Engine): max `[32, 32]`

If the free dimension of the source exceeds the limit, only partial data gets written (rest is zeros), with **no error or warning**.

**Checklist**:
- PSUM->SBUF transpose: if free dim > 32, must loop in chunks of 32
- Prefer designing matmul dst layout to avoid transpose entirely (e.g. choose stationary/moving operands so dst is already in the target layout)
- Single-tile (128x128) tests won't catch this — always test with multi-tile shapes
