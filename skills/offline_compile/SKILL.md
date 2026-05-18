---
name: offline-compile
description: NKI kernel offline compilation template for validating whether a kernel compiles on machines without a Neuron device
---
# NKI Offline Compile Without Device

## Purpose

Validate whether an NKI kernel can pass full compilation from HLO to NEFF on a machine without a Neuron device, such as trn2.
Successful compilation means the NKI IR is valid. It does not validate numerical correctness.

## Template

```python
"""
NKI offline compilation template.
Usage: python offline_compile.py [--target trn2]
"""
import os
import argparse
import torch
import torch_neuronx
import nki
import nki.language as nl


# ---- 1. Define the NKI kernel ----
@nki.jit
def my_kernel(a_hbm, b_hbm):
    """Replace with your kernel."""
    return out_hbm


# ---- 2. Offline compile ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="trn2", help="Compilation target platform")
    args = parser.parse_args()

    # CPU dummy inputs. Shapes are fixed into the NEFF.
    a_cpu = torch.randn(128, 512, dtype=torch.bfloat16)
    b_cpu = torch.randn(128, 512, dtype=torch.bfloat16)

    # Offline compile: set target platform and trace.
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = args.target
    workdir = f"/tmp/my-compile-{args.target}"

    torch_neuronx.trace(
        lambda a, b: my_kernel(a, b),
        (a_cpu, b_cpu),
        compiler_workdir=workdir,
        compiler_args=[f"--target={args.target}"],
    )
    print(f"✅ COMPILE succeeded")
    print(f"   NEFF: {workdir}/graph.neff")
    print(f"   HLO:  {workdir}/model/graph.hlo")


if __name__ == "__main__":
    main()
```

## Three Key Elements

| Element | Description |
|---|---|
| `os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]` | Tells the compiler the target platform, such as `trn2`. |
| `torch_neuronx.trace(fn, cpu_inputs)` | Uses CPU tensors for symbolic trace and triggers full compilation. |
| `compiler_args=["--target=trn2"]` | Compilation arguments passed to `neuronx-cc`. |

## Artifacts

```
/tmp/my-compile-trn2/
├── graph.neff       # Binary that can be deployed to the device.
└── model/
    └── graph.hlo    # HLO IR for offline analysis.
```

## Notes

1. **Shape freezing**: `trace` performs static compilation, so input shapes are fixed into the NEFF.
2. **Lambda wrapping**: `trace` needs a callable, so wrap the kernel with `lambda`.
4. **Compilation success does not imply numerical correctness**: numerical validation must run on a device.
