---
name: access-pattern
description: how to vist/operate on tensor with different access pattern
---
# NKI Access Pattern (AP) Quick Reference

## Semantics

`tensor.ap(pattern, offset=0)` flattens the tensor to 1D, then:

```
pattern = [[step0, num0], [step1, num1], ...]
for d0 in range(num0):
  for d1 in range(num1):
    ...
      idx = offset + d0*step0 + d1*step1 + ...
```

AP shape = (num0, num1, ...). `.ap()` only describes access, no computation.

## Rules

1. **step and num are independent.** step = stride, num = iteration count.
   Only constraint: accessed indices must be in-bounds.

2. **SBUF dim0 = partition dim.**
   step0 = free_size (total free elements per partition), num0 = partition_count.
   Partition access must be contiguous.

3. **[1, 1] placeholder:** step=1, num=1, executes once = no-op.
   OK at: dim1, dim2 (middle), dim0 for HBM, dim3.
   NOT OK: dim0 for SBUF (must be partition dim).

4. **dma_transpose permutations:**
   - 2D: `[1, 0]`
   - 3D: `[2, 1, 0]`
   - 4D: `[3, 1, 2, 0]`

   4D means: `dst[d3, d1, d2, d0] = src[d0, d1, d2, d3]`
   - src dim0 ↔ dst dim3 (num must match)
   - src dim1 ↔ dst dim1 (num must match, "axis 1 unchanged")
   - src dim2 ↔ dst dim2 (num must match, "axis 2 unchanged")
   - src dim3 ↔ dst dim0 (num must match)

   "Axes 1,2 don't move" = NUM must match, STEP and OFFSET are independent.

5. **AP cannot be nested:** `tensor.ap(...).ap(...)` is illegal.

## Examples

### Example 1: dma_transpose HBM[T, H] → SBUF[128, T, H//128]

Constraint: T in [1..8], H % 128 == 0, bfloat16.
Split H into H0=128, H1=H//128, free_size=T*H1.

**src** HBM[T, H]:
```
pattern = [[1, 1], [H, T], [H0, H1], [1, H0]]
index = d1*H + d2*128 + d3 = t*H + h1*128 + h0
```

**dst** SBUF[128, T, H1]:
```
pattern = [[free_size, H0], [H1, T], [1, H1], [1, 1]]
index = partition*free_size + d1*H1 + d2 = partition*free_size + t*H1 + h1
```

Permutation [3,1,2,0] check:
src dim0(1)→dst dim3(1), dim1(T)→dim1(T), dim2(H1)→dim2(H1), dim3(H0)→dim0(H0) ✓

```python
T, H = input_tensor.shape
H0, H1 = 128, H // 128
free_size = T * H1

tp_sb = nl.ndarray((H0, T, H1), dtype=input_tensor.dtype, buffer=nl.sbuf)

nisa.dma_transpose(
    dst=tp_sb.ap([[free_size, H0], [H1, T], [1, H1], [1, 1]]),
    src=input_tensor.ap([[1, 1], [H, T], [H0, H1], [1, H0]]),
)

# then dma_copy sbuf -> hbm as needed
```
