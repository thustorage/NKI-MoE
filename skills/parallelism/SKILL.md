---
name: parallelism
description: NeuronCore instruction-level parallelism and loop iterator optimization guide
---
# NeuronCore Instruction-Level Parallelism & Loop Optimization

## Hardware Execution Model

A single NeuronCore has **independent execution engines** that can run in parallel:

```
┌─────────────────────────────────────────────────────┐
│                    NeuronCore                        │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │DMA Engine│  │ Tensor   │  │  Vector  │           │
│  │          │  │ Engine   │  │  Engine  │           │
│  │ dma_copy │  │ nc_matmul│  │tensor_   │           │
│  │ dma_trans│  │          │  │ scalar   │           │
│  │ HBM<->SB│  │ SBUF->   │  │tensor_   │           │
│  │          │  │   PSUM   │  │ tensor   │           │
│  │          │  │          │  │activation│           │
│  └──────────┘  └──────────┘  └──────────┘           │
│       │              │              │                │
│       └──────────────┼──────────────┘                │
│              Shared SBUF / PSUM                      │
└─────────────────────────────────────────────────────┘
```

Key point: **DMA, Tensor Engine, and Vector Engine can execute simultaneously** as long as there are no data hazards (read-after-write, write-after-write on the same SBUF/PSUM addresses).

---

## Loop Iterator Types and Their Impact on Parallelism

### `range()` -> `sequential_range`

```python
for i in range(N):
    body(i)
```

Compiler replaces Python `range()` with `sequential_range()`. The compiler:
- **Respects all inter-iteration dependencies** (conservative)
- **Likely inserts barriers at iteration boundaries** — even when no true data dependency exists between engines across iterations
- Intra-iteration DMA/compute overlap still happens (instruction scheduling within one iteration)
- **Cross-iteration overlap is unlikely** — the compiler does not aggressively analyze which cross-iteration operations are independent

### `affine_range` (preferred default)

```python
for i in nl.affine_range(N):
    body(i)
```

The compiler:
- **Assumes no loop-carried dependency** (except associative reductions like matmul accumulation into the same PSUM buffer)
- **Can pipeline/parallelize iterations across different engines** — e.g., iteration i's VPE work overlaps with iteration i+1's DMA
- Enables loop vectorization and other loop-level optimizations
- **Unsafe if there is a true loop-carried dependency** (will cause numerical errors)

### `static_range` (debug fallback only)

```python
for i in nl.static_range(N):
    body(i)
```

Fully unrolls the loop at trace time. Compilation time and SBUF usage blow up. No loop-level optimization. Only use for debugging.

---

## The Cross-Iteration Overlap Problem

### Typical MoE Kernel Pattern (current)

```python
for token_idx in range(T):           # sequential_range
    for expert_k in range(K):        # sequential_range
        gate_w_load()                 # DMA engine
        gate_matmul()                 # Tensor engine
        up_w_load()                   # DMA engine
        up_matmul()                   # Tensor engine
        activation_multiply()         # Vector engine
        down_w_load()                 # DMA engine
        down_matmul()                 # Tensor engine
        affinity_scale()              # Vector engine
        accumulate_output()           # Vector engine  <-- writes to shared buffer
```

### What the compiler likely does with `sequential_range`

```
Expert 0: |--DMA--|--TE--|--DMA--|--TE--|--VPE--|--DMA--|--TE--|--VPE--|--VPE--|
          [barrier]
Expert 1:                                                                       |--DMA--|--TE--|...
                                                                                ^
                                                          DMA engine sits idle here, waiting for barrier
```

The barrier at iteration boundaries means **Expert 1's weight DMA cannot start while Expert 0's VPE accumulation is still running**, even though DMA and VPE are independent engines with no address conflict.

### What could happen with proper `affine_range`

```
Expert 0: |--DMA--|--TE--|--DMA--|--TE--|--VPE--|--DMA--|--TE--|--VPE+ACC--|
Expert 1:                                              |--DMA--|--TE--|--DMA--|--TE--|--VPE--|...
                                                        ^
                                          DMA starts as soon as engine is free, no barrier
```

The DMA engine for Expert 1 begins loading weights while Expert 0 is still doing VPE work. This **hides DMA latency behind compute**.

---

## When Can You Use `affine_range`?

### Safe: No loop-carried dependency

```python
# Each iteration writes to independent buffers
for i in nl.affine_range(K):
    result[i] = compute(input, weights[i])
```

### Safe: Associative reduction (matmul accumulation)

```python
psum_buf = nl.zeros((...), buffer=nl.psum)
for i in nl.affine_range(N):
    psum_buf += nl.matmul(a[i], b[i])  # accumulate into same PSUM
```

### UNSAFE: Read-modify-write on SBUF

```python
accum = nl.zeros((...), buffer=nl.sbuf)
for i in nl.affine_range(K):  # BUG: loop-carried dependency
    partial = compute(input, weights[i])
    accum = accum + partial  # reads accum from previous iteration
```

This is the classic problem in the MoE accumulation pattern:
```python
if expert_k_idx == 0:
    tensor_copy(dst=output_temp[token], src=down_sb)
else:
    tensor_tensor(dst=output_temp[token], data1=output_temp[token], data2=down_sb, op=add)
```

`output_temp[token]` is read in iteration k and written in iteration k-1. This is a **true loop-carried dependency**.

---

## Strategy: Separate Independent Compute from Dependent Accumulation

To get `affine_range` benefits while keeping correctness, split the loop into two phases:

### Phase 1: Independent compute (affine_range safe)

```python
# All K experts computed independently, results stored in per-expert buffers
for k in nl.affine_range(K):
    gate_up_out[k] = gate_up_projection(input[token], weights[expert[k]])
    down_out[k] = down_projection(gate_up_out[k], down_weights[expert[k]])
    down_out[k] *= affinity[k]
```

### Phase 2: Reduction (sequential or tree-reduce)

```python
# Accumulate results — has loop-carried dependency
output = down_out[0]
for k in nl.sequential_range(1, K):
    output += down_out[k]
```

### Trade-off

- **Pro**: Phase 1 gets full cross-iteration DMA/compute overlap
- **Con**: Need K separate output buffers in SBUF instead of accumulating in-place
- For typical K=2 (top-2 MoE), this only doubles the SBUF buffer cost for down output — acceptable
- For K=8+, SBUF pressure may become a problem

---

## SBUF Address Management for Cross-Iteration Overlap

For the compiler to overlap iterations, it must prove no address conflict. Two approaches:

### Approach 1: `interleave_degree` in SbufManager

```python
sbm.open_scope(interleave_degree=2)
for k in range(K):
    buf = sbm.alloc_stack(...)  # alternates between 2 SBUF sections
    compute(buf)
    sbm.increment_section()     # move to next section
sbm.close_scope()
```

This allocates **separate SBUF regions** for adjacent iterations, eliminating address conflicts. But with `sequential_range`, the compiler may still not exploit this.

### Approach 2: Explicit double-buffering

```python
bufs = [sbm.alloc_stack(...) for _ in range(2)]
for k in nl.affine_range(K):
    buf = bufs[k % 2]  # compiler sees independent addresses
    compute(buf)
```

With `affine_range`, the compiler can now freely overlap iteration k's VPE with iteration k+1's DMA because the buffers are provably non-conflicting.

---

## Weight DMA Prefetch Pattern

The most impactful overlap in DMA-bound kernels (like TKG with small T):

```python
# Conceptual pattern — interleave weight loading with compute
weight_buf = [alloc() for _ in range(2)]  # double buffer

# Prefetch first weight
dma_load(weight_buf[0], expert_weights[0])

for k in nl.affine_range(K):
    cur = k % 2
    nxt = (k + 1) % 2

    # Prefetch next weight while computing with current
    if k < K - 1:
        dma_load(weight_buf[nxt], expert_weights[k + 1])

    # Compute with current weight (tensor engine + VPE)
    result[k] = matmul(input, weight_buf[cur])
    result[k] = activate(result[k])
```

Note: The `if k < K-1` guard requires `static_range` or manual unrolling. With `affine_range`, the compiler may handle prefetching automatically if it can prove the pattern is safe.

---

## Summary: Decision Guide

```
Want cross-iteration engine overlap?
│
├── Does the loop body have loop-carried dependency?
│   │
│   ├── NO → Use affine_range directly
│   │        Compiler handles DMA/compute overlap across iterations
│   │
│   └── YES → Can you separate independent work from the dependency?
│       │
│       ├── YES → Split into two loops:
│       │         Loop 1 (affine_range): independent compute per iteration
│       │         Loop 2 (sequential_range or tree-reduce): accumulation
│       │         Trade-off: extra SBUF buffers for intermediate results
│       │
│       └── NO (tight dependency throughout) →
│               Use sequential_range + interleave_degree hint
│               Rely on compiler for intra-iteration overlap only
│               Consider manual unrolling if K is small (2-4)
│
└── Not sure → Profile with trace to check engine idle gaps
              See skills/debug_dump for --dump instructions
```

---

## Verification: How to Confirm Overlap is Happening

1. Compile with `--dump`:
   ```bash
   NEURON_CC_FLAGS="--dump /tmp/neff_dump" python test.py
   ```

2. Check `instruction_stats.txt` — look at total instruction counts per engine

3. Use `save_trace_name` in `nki.benchmark` to get execution trace:
   ```python
   bench = nki.benchmark(warmup=5, iters=10, save_trace_name="trace.ntff")
   ```

4. Look for **idle gaps between engines** in the trace timeline:
   - If DMA is idle while VPE is busy at iteration boundaries → barrier is blocking overlap
   - If DMA and VPE overlap across iteration boundaries → compiler is doing its job

---

## Applicability to MoE Selective Expert Kernel

Current implementation (`selective_expert_impl.py`):
- Both loops use Python `range()` → `sequential_range`
- `interleave_degree=2` is set for the inner expert loop
- The inner loop has a loop-carried dependency on `output_temp` accumulation

Optimization opportunity:
1. Separate MLP compute from accumulation in the inner loop
2. Use `affine_range` for the compute phase
3. Keep accumulation as a separate `sequential_range` reduction
4. This enables cross-expert DMA/compute overlap without breaking the subkernel interface — just restructure the caller loop

Whether breaking the MLP subkernel boundary adds further benefit depends on whether the compiler can overlap the **last instructions of `process_down_projection`** with the **first DMA instructions of `process_gate_up_projection`** in the next iteration. With `affine_range` and no address conflicts, it should — but this needs trace verification.
