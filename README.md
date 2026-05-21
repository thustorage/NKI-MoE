# AWS Trainium2/3 MoE Kernel Challenge

**This repository contains the source code for our winning entry (1st Place) in the MLSys 2026 Programming Competition.**

Our work significantly enhances the performance of the baseline MoE inference system provided for the competition.

For more details about the competition, please refer to the [competition repository](https://github.com/aws-neuron/nki-moe).

## Overview

The MLSys 2026 competition track presented a unique challenge: optimizing MoE model inference on specialized AI hardware using low-level programming interfaces. Specifically, teams were tasked with implementing the Qwen3-30B-A3B model targeting a single AWS Trainium2/3 chip.

## Repository Structure

### Optimized MoE Kernels
`kernels/`

This directory contains the optimized inference path used by our submission. The main files are:

- `qwen_with_nki.py`: The exported model entry point for the optimized implementation. It adds the fast batch-1 token-generation attention path and the MoE megakernel path.
- `qwen_with_nki_original.py`: The baseline NKI-aware Qwen3-MoE model definition.
- `qwen_moe_tkg_mega.py`: The decode-time MoE megakernel runner. It checks whether it is doing token generation, gathers router and expert weights into the layouts expected by the kernel, and dispatches the optimized token-generation execution when the fast path is legal.
- `moe/moe_parameters.py`: Shared parameter and metadata definitions for the MoE kernels. It packages routing inputs, normalization options, bias tensors, expert parameters, tiling constants, and execution flags into structured objects that the NKI kernels consume.
- `moe/moe_selective.py`: The exported NKI selective-expert MoE kernel. Its `moe_selective_v3` entry point drives the fused decode path, combining routing inputs, RMSNorm, expert selection, and expert MLP execution into a single kernel-facing interface.
- `moe/selective_expert_impl.py`: The low-level implementation behind the selective-expert kernel. It contains the fused gate/up GEMV path and down-projection path.
- `moe/rmsnorm_tkg.py`: The token-generation RMSNorm implementation used by the fused MoE path.
- `moe/router_topk.py`: The router Top-K decode helper implementation. It handles router weight layout conversion, SBUF-friendly loading, and the slim decode-time top-k routing path that selects which experts each token should visit.
- `utils/`: Shared kernel support code such as allocators, tensor views, assertions, tiling helpers, and logging.

### AI-assistting System
`night-optimizer/`

This directory contains an optimization harness used to manage iterative kernel improvement. It includes CLI, workflow, execution, state tracking, repository handling, policy, and result parsing modules so repeated optimization runs can be organized and evaluated systematically instead of being managed manually.

`skills/`

This directory stores reusable optimization playbooks. The current skills focus on access-pattern analysis, debug-dump inspection, offline compilation, parallelism strategy, and profiler-based analysis, which makes the folder a compact knowledge base for coding agent to dignose and optimize kernel performance.

`memory/`

This directory stores project notes such as bug writeups and progress logs. It serves as lightweight working memory for the repository, keeping implementation observations and experiment notes close to the code.

### Engineering Utilities
`generate_submission.py`

This script is used to package the repository into the contest submission format. Its role is to collect the code that should be shipped to the evaluation environment from `kernels/` and ensure the submission artifact `qwen_with_nki.py` is assembled consistently.

`remote_test.sh`

This shell script is a helper for remote validation. It is intended for running checks on the remote server with Trainium3, which is useful when verifying behavior or performance outside the local development machine.

## Benchmarking and Performance

The checked-in file `qwen3-30b-a3b-trn3_score_records.csv` contains five Trn3 benchmark runs.

| Run | Accuracy | Latency (ms) | Throughput (tokens/s) | Reduced Latency | Increased Throughput | NKI FLOP Ratio | Final Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | True | 4078.66 | 160.43 | 4.20x | 4.24x | 0.9929 | 35.48 |
| 2 | True | 3056.97 | 215.68 | 4.07x | 4.12x | 0.9929 | 33.38 |
| 3 | True | 3821.14 | 168.93 | 4.24x | 4.20x | 0.9929 | 35.48 |
| 4 | True | 4095.84 | 157.64 | 4.19x | 4.16x | 0.9929 | 34.74 |
| 5 | True | 1012.88 | 640.34 | 3.82x | 3.80x | 0.9929 | 28.97 |


## Reproducing Results
The results can be reproduced on an AWS Trainium3 instance with AWS Neuron SDK 2.28.

Run the following command at the root of the repository:

```bash
python3 main.py --mode evaluate_all --enable-nki --model-path ~/data/model/ --compiled-model-path ~/data/traced_model --benchmark --platform-target trn3
```

Model path should point to the directory containing the Qwen3-30B-A3B model files. Compiled model path must have 60GB of free space for storing the compiled model.

## Acknowledgements
We thank the competition organizers and AWS for their generous sponsorship of computational resources, which enabled us to perform optimization on the NKI framework.
We also acknowledge the [AWS Neuron nki-library](https://github.com/aws-neuron/nki-library), whose kernel implementations provided the starting point for several of our optimizations.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Cite Us
If you find our work useful, please cite us:

```bibtex
@misc{moe-kernel-challenge,
  author={Shiwei Gao, Ruwen Fan, Tingxu Ren, Yibin Luo},
  title={Optimizing MoE Inference on AWS Trainium: A Winning Entry in the MLSys 2026 Programming Competition},
  year={2026},
  url={https://github.com/thustorage/NKI-MoE}
}
```
