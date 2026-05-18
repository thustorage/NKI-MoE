
### analyze_profile.py

Transforms raw profiler JSON into structured analysis.

```bash
python skills/profiler_analysis/analyze_profile.py <report_dir> [-o output.json]
```

## Analysis Output Dimensions

### Summary (`summary`)

| Field | Meaning |
|---|---|
| `total_time_ns` | Total kernel execution time |
| `bound_type` | `"compute-bound"` or `"memory-bound"` |
| `engines.{name}.active_time_pct` | Per-engine utilization |
| `mfu_pct` / `mbu_pct` / `hfu_pct` | Model/Memory/HBM FLOPs utilization |
| `hbm_read_bytes` / `hbm_write_bytes` | HBM I/O volume |

Interpretation:
- `memory-bound` + low `mbu_pct` => optimize data layout or access pattern
- `compute-bound` + low `mfu_pct` => check for stalls
- High `spill_bytes` / `reload_bytes` => SBUF pressure

### Time Windows + Phase Detection (`time_windows`)

Three scales: `scale_100` (fine), `scale_50` (medium), `scale_10` (coarse).

| Phase | Meaning |
|---|---|
| `IDLE` | No engine active |
| `COMPUTE` | Compute only, no DMA |
| `LOAD` / `STORE` | DMA only |
| `MIXED` | DMA + compute overlap (good) |
| `STALL` | Pipeline bubble |

Interpretation:
- `LOAD -> COMPUTE -> STORE` without `MIXED` => no prefetching, sequential execution
- `MIXED` phases indicate good DMA-compute overlap

### Cross-Engine Blocking (`blocking`)

| Pattern | Meaning | Action |
|---|---|---|
| `DATA_DEPENDENCY` | Waiting for DMA | Prefetch earlier, double-buffer |
| `ENGINE_DEPENDENCY` | Waiting for compute | Reorder ops, reduce dependency chain |
| `SEMAPHORE_STALL` | Semaphore wait, no concurrent activity | Possible over-synchronization |

### DMA-Compute Overlap (`dma_compute_overlap`)

| Field | Meaning |
|---|---|
| `overlap_ratio` | `overlapped / total` — higher is better (1.0 = fully hidden) |
| `non_overlapped_dma_ns` | DMA time exposed on critical path |

### Wait Chains (`wait_chains`)

```
waiter (engine/opcode) --[wait_ns]--> semaphore --> blocker (engine/opcode)
```

Longest wait chains are the primary optimization targets.

### Critical Path (`critical_path`)

| Field | Meaning |
|---|---|
| `efficiency_pct` | `compute / (compute + wait) * 100` — low = mostly waiting |
| `critical_path_ns` | Minimum possible execution time |

### Anomalies (`anomalies`)

Instructions with duration > 3x engine mean. Filter out `EVENT_SEMAPHORE` (wait-induced), focus on computation anomalies.

### Repetition Patterns (`repetition_patterns`)

Repeated opcode subsequences per engine. Large gap between `avg_duration` and `avg_interval` => idle time between iterations.

### Engine Handoffs (`engine_handoffs`)

Directed graph of engine scheduling. Large `avg_gap_ns` on frequent edges => scheduling inefficiency.

## Analysis Workflow

1. **Summary**: Check `bound_type` and engine utilization
2. **DMA Overlap**: Low overlap + memory-bound => pipeline issue
3. **Blocking**: Sort by `wait_ns` to find biggest stalls
4. **Wait Chains**: Trace blocker for top stalls
5. **Critical Path**: `efficiency_pct` shows room for improvement
6. **Phase Patterns**: `scale_10` for overview, `scale_100` for detail
7. **Anomalies**: Focus on non-semaphore anomalies
