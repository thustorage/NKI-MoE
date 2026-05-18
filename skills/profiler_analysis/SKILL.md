---
name: profiler-analysis
description: Neuron profiler session management, analysis, and best practices
---

# Neuron Profiler Analysis

## Quick Start

### Profile + View in one command

`profile_kernel.py` supports `--view` to auto-select the best NEFF/NTFF pair and launch the viewer:

```bash
# Profile and auto-launch viewer on port 3001
./remote_test.sh --push "python skills/profiler_analysis/profile_kernel.py --wrap 'python ops/ours/profile_v3_matmul.py' --view"

# Custom port
./remote_test.sh --push "python skills/profiler_analysis/profile_kernel.py --wrap 'python ops/moe/profile_moe.py' --view 3002"
```

What `--view` does automatically:
1. Kills any existing `neuron-profile view` process
2. Auto-selects NEFF/NTFF pair (largest for single-kernel, median for e2e — see selection strategy below)
3. Starts `neuron-profile view -n <neff> -s <ntff> -p <port> --force` in background
4. Prints the URL to open in browser

After the command finishes, open the URL shown in the output (e.g. `http://localhost:3001/profile/<hash>`).

**IMPORTANT**: Do NOT kill local ports or create SSH tunnels (`ssh -f -N -L ...`).
The user has VSCode permanent port forwarding configured. Only manage the remote
`neuron-profile view` process — kill old ones before starting new ones.

### Profile only (no viewer)

```bash
./remote_test.sh --push "python skills/profiler_analysis/profile_kernel.py --wrap 'python ops/ours/profile_v3_matmul.py'"
```

This still extracts summary reports (JSON, text) and prints the recommended NEFF/NTFF pair, but does not launch the viewer.

## Writing a Profile Script (Critical)

**Never profile test frameworks directly.** Test scripts have multiple `xm.mark_step()` calls (ground truth, metrics, etc.) that produce many NEFFs, making it impossible to identify the target kernel.

Write a **minimal, single-kernel profile script** instead:

```python
"""Example: ops/ours/profile_my_kernel.py"""
import torch
import torch_xla.core.xla_model as xm

# Import the @nki.jit WRAPPER, not the inner kernel function.
# Inner functions (e.g. qkv_tkg) expect NKI tensors with .buffer attribute.
# The @nki.jit wrapper handles torch tensor -> NKI tensor conversion.
from ops.qkv.test_qkv_precision import QKV_VERSIONS

# Use kernel[2] for LNC=2 (2 logical neuron cores).
# Use kernel (no subscript) for single core.
kernel = QKV_VERSIONS["v3"][2]

device = xm.xla_device()

# CRITICAL: Create tensors on CPU, then .to(device).
# Do NOT use torch.randn(..., device=device) — it compiles randn (sine/cosine/log
# from Box-Muller transform) into the SAME NEFF as the kernel, polluting the profile.
hidden = (torch.randn(1, 1, H, dtype=torch.bfloat16) * scale).to(device)
weight = (torch.randn(H, I, dtype=torch.bfloat16) * scale).to(device)
xm.mark_step()        # materialize H2D transfers in their own NEFF
xm.wait_device_ops()

# Step 1: Warmup (triggers compilation, profiler captures but easy to filter)
out = kernel(input=hidden, ...)
xm.mark_step()
xm.wait_device_ops()

# Step 2: Profiled run (same graph, cache hit, pure execution)
out = kernel(input=hidden, ...)
xm.mark_step()
xm.wait_device_ops()

_ = out.cpu()  # force execution
```

### Clean profile checklist

1. **CPU randn + .to(device)** — never `torch.randn(device=device)`
2. **mark_step after .to(device)** — isolate H2D transfer into its own NEFF
3. **kernel[2] for LNC=2** — match production config (2 logical neuron cores)
4. **Verify NEFF if results look off** — use the pollution check in Tools section below
   - Clean: only `custom-call` (the NKI kernel)
   - Polluted: `sine`, `cosine`, `log`, `rng-bit-generator` (Box-Muller from randn)
5. **NEURON_RT_VISIBLE_CORES** — set to match LNC needs (1 core = 2 NC for LNC=2)

### Key rules

| Rule | Why |
|---|---|
| One kernel per script | Fewer NEFFs, easy to identify target |
| Use `@nki.jit` wrapper | Inner kernel functions expect NKI `nl.ndarray` with `.buffer`, not torch tensors |
| **Create tensors on CPU, .to(device)** | `torch.randn(device=device)` compiles Box-Muller (sine/cosine/log) into the same NEFF as kernel |
| **mark_step after .to(device)** | Separates H2D transfer NEFF from kernel NEFF |
| warmup + profiled run | First `mark_step` compiles; second is the real perf measurement |
| No ground truth / metrics | `torch.matmul` etc. generate their own NEFFs and pollute profile data |
| `--view` auto-selects NEFF/NTFF | Single-kernel → largest NEFF; e2e → median by size; always matches by ID |
| Put script in `ops/` not `tmp/` | `tmp/` is in `.gitignore`, won't be pushed to remote via `remote_test.sh` |

## Tools

### profile_kernel.py

Profiling infrastructure decoupled from kernel code. Sets env vars, runs command, extracts reports.

```bash
python skills/profiler_analysis/profile_kernel.py --wrap "python my_script.py"
```

Options:
- `--wrap`: Command to run with profiling env vars injected (required)
- `--workdir`: Working directory (default: `tmp/profiler_workspace`)
- `--view [PORT]`: Auto-launch neuron-profile viewer after profiling (default port: 3001)

Auto-selection logic for `--view`:

| Mode | NEFFs generated | Selection rule |
|---|---|---|
| **Single kernel** (minimal profile script) | 1-2 NEFFs (H2D + kernel) | Pick the **largest** NEFF — the target kernel is almost always the biggest |
| **End-to-end** (main.py / full model) | 3+ NEFFs | Pick the **median by file size** — extremes are H2D transfers or oversized fused graphs |

NEFF and NTFF files share numeric IDs that **must match** when pairing:
- NEFF naming: `neff_{ID}.neff`
- NTFF naming: `{ID}_vnc_{core}.ntff` (one per neuron core, e.g. vnc_0 through vnc_3)

The auto-selection always pairs a NEFF with an NTFF of the **same ID** (picks the largest matching NTFF for max data).

### NEFF pollution check

If you need to manually verify a NEFF is clean (not polluted by Box-Muller or ground-truth ops):

```bash
ssh neuron "neuron-profile view --output-format json --output-file /tmp/neff_check.json \
  -n <neff_path> -s <ntff_path> 2>/dev/null && python3 -c \"
import json
with open('/tmp/neff_check.json') as f:
    d = json.load(f)
instrs = d.get('instruction', [])
hlo_names = sorted(set(i.get('hlo_name', '') for i in instrs if i.get('hlo_name', '')))
print(f'Total instructions: {len(instrs)}')
print(f'Unique HLO ops: {len(hlo_names)}')
for n in hlo_names:
    print(f'  {n}')
\""
```

**Red flags** (NEFF is polluted):
- `sine`, `cosine`, `log`, `sqrt` → Box-Muller transform from `torch.randn(device=device)`
- `dot`, `reduce` without `custom-call` → torch matmul ground truth compiled into same graph
- `rng-bit-generator` → random number generation on device

**Expected** (clean NKI kernel NEFF):
- `custom-call` → the NKI kernel itself
- `multiply`, `broadcast`, `reshape` → input/output data movement wrapper
- Few simple ops only

Python context manager mode:

```python
from skills.profiler_analysis.profile_kernel import profiler_session

with profiler_session("my_kernel") as session:
    # ... run kernel ...
    xm.mark_step()
    xm.wait_device_ops()

# session.report_dir
```
