# NKI Kernel Debug Skills Guide

## Overview

This guide summarizes methods and toolchains for debugging NKI kernels on the AWS Neuron (trn1) platform.

---

## 1. Compiler Intermediate Artifact Dump (Recommended First Choice)

### Principle

Set the `NEURON_CC_FLAGS="--dump <dir>"` environment variable so the Neuron compiler writes all intermediate compilation artifacts to the specified directory.

### Usage

```python
import os
os.environ.setdefault("NEURON_CC_FLAGS", "--dump /tmp/neff_dump")
```

Or from the command line:
```bash
NEURON_CC_FLAGS="--dump /tmp/neff_dump" python my_test.py
```

### Artifact Directory Structure

After compilation, the following files are generated under `/tmp/ubuntu/neuroncc_compile_workdir/<uuid>/`:

| File | Description | Readability |
|------|------|--------|
| `*.hlo_module.pb` | HLO protobuf, which is XLA high-level IR | Requires parsing |
| `sg00/bir.json` | Backend IR, including backend instruction sequences | JSON |
| `sg00/def.json` | Subgraph definition, including input and output tensor mappings | JSON |
| `sg00/instruction_stats.txt` | Instruction type statistics | Text |
| `sg00/dma_stats.txt` | DMA descriptor details | Text |
| `sg00/SP.json` | Scheduling parameters | JSON |
| `log-neuron-cc.txt` | Full compilation log, including HBM usage | Text |
| `neuronx_cc_metadata.json` | Compiler metadata and full compilation command | JSON |
| `*.neff` | Final executable binary | Binary |

Additional `<kernel_name>*.klir` files are generated under `/tmp/`. These are NKI KLR intermediate representations.

### Parse HLO Protobuf

```python
import sys
sys.path.insert(0, "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages")

from libneuronxla.proto import hlo_pb2
from google.protobuf import text_format

m = hlo_pb2.HloModuleProto()
with open("path/to/model.*.hlo_module.pb", "rb") as f:
    m.ParseFromString(f.read())

# Get a human-readable representation of the entry computation graph.
for comp in m.computations:
    if comp.name == m.entry_computation_name:
        for inst in comp.instructions:
            dims = list(inst.shape.dimensions) if inst.shape.dimensions else []
            etype = hlo_pb2.PrimitiveType.Name(inst.shape.element_type)
            ops = list(inst.operand_ids)
            print(f"%{inst.name} = {inst.opcode}({etype}{dims}, operands={ops})")
```

### Key BIR Fields

Each instruction in `bir.json` contains:
- `instruction`: instruction type, such as TensorScalar, TensorTensor, Matmult, and Save.
- `debug`: mapping back to the source file and line number.
- Tensor memory layout and access pattern.

---

## 2. XLA IR/HLO Debug Environment Variables

### Usage

```bash
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
```

> Already integrated into `remote_test.sh`; no manual setup is required.

### Effect

- `XLA_IR_DEBUG=1`: outputs XLA IR debug information at runtime.
- `XLA_HLO_DEBUG=1`: outputs HLO debug information at runtime.

This is mainly used to observe how PyTorch/XLA translates Python-level operations into XLA graphs.

---

## 3. device_print (NxD Inference Only)

### Principle

Use `nl.device_print()` to print tile tensor values inside the kernel.

### Usage

```python
import nki.language as nl

def my_kernel(input_tensor):
    tile = nl.ndarray(...)
    # ... load data into tile ...
    nl.device_print("my_tile", tile)
```

### Output

Set an environment variable to specify the output directory:
```bash
export NEURON_RT_DEBUG_OUTPUT_DIR=/tmp/debug_output
```

Output directory structure: `<print_prefix>/core_<logical_core_id>/<iteration>/...`

### Limitations

⚠️ **Only available in the NxD Inference library**. It is not available in standalone `nki.jit` kernel tests.

---

## 4. Compilation Error Localization Tips

### Common Compilation Errors

#### `TensorScalarPtr arith immediate dtype must be fp32`

**Cause**: the `operand0` argument of `nisa.tensor_scalar` in pointer mode requires the data type to be `float32`.

**Fix**: ensure the tensor slice passed to `operand0` has type `nl.float32`. Cast when needed.

#### `ISA check failed`

**Cause**: instruction argument types or shapes usually violate hardware ISA constraints.

**Investigation**:
1. Check the line number reported by the compiler. Note that the line number may not exactly match the Python source.
2. Check `log-neuron-cc.txt` for more detailed error context.
3. Check the `debug` field of the corresponding instruction in `bir.json`.

#### `GPSIMD timeout`

**Cause**: the kernel timed out on the device, usually due to an infinite loop or excessive computation.

**Investigation**:
1. Reduce the input size to build a minimal reproduction.
2. Check loop boundary conditions.
3. Check whether DMA operations are correct.

---

## 5. Remote Testing Workflow

### Architecture

```
Local development machine (macOS)        Remote test machine (neuron/trn1)
┌─────────────────┐        ┌─────────────────────────┐
│ Edit code       │  git   │ /home/ubuntu/nki-moe    │
│ AI agent        │───────>│ neuron venv             │
│ analysis        │  ssh   │ Run tests               │
│ Analyze output <│────────│ Return results          │
└─────────────────┘        └─────────────────────────┘
```

### Environment Variables Preset by remote_test.sh

```bash
PYTHONPATH=$PYTHONPATH:~/nki-moe
NEURON_PLATFORM_TARGET_OVERRIDE=trn1
XLA_IR_DEBUG=1
XLA_HLO_DEBUG=1
```

### Common Commands

```bash
# Run tests.
./remote_test.sh "python ops/moe/test_moe_jit.py"

# Run with extra environment variables.
./remote_test.sh "NEURON_CC_FLAGS='--dump /tmp/neff_dump' python my_test.py"

# View compile cache.
./remote_test.sh "find /var/tmp/neuron-compile-cache -name '*.hlo*' | head -20"

# View compiler intermediate artifacts.
./remote_test.sh "find /tmp/ubuntu/neuroncc_compile_workdir -type f | head -30"

# Parse KLIR files.
./remote_test.sh "cat /tmp/*.klir"

# View instruction statistics.
./remote_test.sh "cat /tmp/ubuntu/neuroncc_compile_workdir/*/sg00/instruction_stats.txt"

# Clear compile cache to force recompilation.
./remote_test.sh "rm -rf /var/tmp/neuron-compile-cache/*"
```

---

## 6. Minimal Reproduction Script Template

During debugging, create a minimal reproduction script under `/tmp`:

```python
"""Minimal reproduction script for debugging."""
import os
import torch
import torch_xla.core.xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

os.environ.setdefault("NEURON_CC_FLAGS", "--dump /tmp/neff_dump")

@nki.jit
def minimal_kernel(input_tensor, output_tensor):
    """Stripped-down kernel with only the problematic operation."""
    tile = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=input_tensor[0:128, 0:128])
    # ... insert the operation to debug here ...
    nisa.dma_copy(dst=output_tensor[0:128, 0:128], src=tile)

def main():
    device = xm.xla_device()
    inp = torch.randn(128, 128, dtype=torch.bfloat16, device=device)
    out = torch.zeros(128, 128, dtype=torch.bfloat16, device=device)

    minimal_kernel(inp, out)
    xm.mark_step()

    print(f"Input:  {inp[:4, :4].cpu()}")
    print(f"Output: {out[:4, :4].cpu()}")

if __name__ == "__main__":
    main()
```

---

## 7. Debug Method Selection Decision Tree

```
Compilation failed?
├── Yes -> Check the line number and instruction type in the compilation error.
│       ├── Set --dump to obtain bir.json and log-neuron-cc.txt.
│       └── Create a minimal reproduction script, then add operations step by step to localize the issue.
│
└── No (runtime issue)
    ├── Incorrect result?
    │   ├── Parse HLO protobuf to confirm computation graph correctness.
    │   ├── Check whether tensor dtype and shape match expectations.
    │   └── Compare against the CPU reference implementation.
    │
    ├── GPSIMD timeout?
    │   ├── Reduce input size.
    │   ├── Check loop boundaries.
    │   └── Check DMA operations.
    │
    └── Performance issue?
        ├── Check instruction distribution in instruction_stats.txt.
        ├── Check DMA operation counts in dma_stats.txt.
        └── Check HBM usage in log-neuron-cc.txt.
```
