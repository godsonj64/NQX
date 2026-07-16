# NanoQuant-X

NanoQuant-X is a production-oriented extension of Samsung Research's
[NanoQuant](https://github.com/SamsungLabs/NanoQuant), the ICML 2026
post-training method for binary and sub-1-bit large language model weights.
It keeps the official ADMM/SVID initializer and CUDA GEMV/GEMM kernels, then
fixes the gap between the continuous factorization objective and the signed,
scaled representation that is actually deployed.

The repository is a complete source bundle. It contains the Hugging Face model
pipeline, CUDA kernels, a dependency-light NumPy reference implementation,
packed artifact support, global rank allocation, tests, examples, and
reproducible matrix-level benchmarks.

## What is improved

| Area | Released path | NanoQuant-X path |
| --- | --- | --- |
| Reconstruction metric | Continuous factors | Exact signed runtime matrix |
| Scale initialization | Mean factor magnitudes | K-FAC-weighted alternating least squares |
| Rank amplitudes | Two boundary scale vectors | Strict two-scale or balanced per-rank scale |
| Packed rank | Padded lanes can remain unused | Reclaims every rank lane already paid for in each uint32 word |
| Bit allocation | Independent rank formula per layer | Global allocation using exact packing and optional measured rate-distortion curves |
| Candidate selection | Final continuous iterate | Selects final/best iterates by the exact deployed objective |
| Stored scales | Optimized in FP32, stored at lower precision | Refines and measures the FP16/BF16 values actually stored |
| KD cache | Full vocabulary logits on CPU | Top-k probabilities plus a probability-preserving tail bucket |
| Portable runtime | Recasts and rescales factors every call | Reusable scale-fused factors for two-BLAS inference |
| Portable artifact | PyTorch checkpoint | Deterministic, checksummed, non-pickle `.nqx` format |
| Validation | End metrics only | Packing, storage, determinism, numerical, and non-regression tests |
| Loading safety | Legacy pickle fallback | Weights-only PyTorch or SafeTensors only |

Two profiles are provided:

- `strict`: the paper's two boundary scales and no rank metadata;
- `balanced`: one FP16 coefficient per rank, optimized jointly with the two
  boundary scales. This is the recommended profile because its metadata cost is
  small for LLM-sized matrices and it consistently improved the included
  equal-work matrix benchmark.

## Installation

The full LLM pipeline targets Python 3.12, PyTorch 2.6 or newer, CUDA 12.4 or
newer, and an NVIDIA GPU.

```bash
conda create -n nqx python=3.12 -y
conda activate nqx
pip install -e ".[llm]"

cd src/nanoquant/kernel
bash compile_kernel.sh
cd ../../..
```

An installable wheel is also included at
`dist/nanoquant_x-0.4.0-py3-none-any.whl`.

The portable reference path needs only NumPy and can run on CPU:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "numpy>=1.26"
PYTHONPATH=src python -m nanoquant.reference.cli demo demo.nqx
```

## Quantize a Hugging Face model

The default `nqx` mode uses the balanced profile, packed-bit-aware global rank
allocation, exact scale refitting, compact teacher targets, and the official
block/model reconstruction stages.

The bundled configuration can be run directly:

```bash
python -m nanoquant.main configs/qwen3_0_6b_balanced.json
```

```bash
python -m nanoquant.main \
  --model_id Qwen/Qwen3-0.6B-Base \
  --qmodel_path outputs/qwen3-0.6b-nqx-1bpw.pt \
  --bits 1.0 \
  --admm_type nqx \
  --nqx_rank_scale true \
  --nqx_adaptive_rank true \
  --nqx_scale_iters 4 \
  --nqx_scale_ridge 1e-6 \
  --nqx_storage_aware true \
  --nqx_kd_topk 128 \
  --num_calib_samples 128 \
  --seqlen 2048 \
  --nonfact_epochs 8 \
  --fact_epochs 8 \
  --model_kd_epochs 8 \
  --admm_outer_iters 400 \
  --ppl_task wikitext2 \
  --zeroshot_task "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge" \
  --device_map cuda
```

For the strict representation, use `--nqx_rank_scale false`. To reproduce the
released algorithm, use `--admm_type nanoquant`. The `dbf` compatibility path
is also retained.

## Quantize and validate an individual matrix

```bash
PYTHONPATH=src python -m nanoquant.reference.cli quantize weight.npy weight.nqx \
  --bpw 0.75 \
  --profile balanced \
  --input-hessian input_diag.npy \
  --output-hessian output_diag.npy

PYTHONPATH=src python -m nanoquant.reference.cli inspect weight.nqx
PYTHONPATH=src python -m nanoquant.reference.cli validate weight.nqx weight.npy
PYTHONPATH=src python -m nanoquant.reference.cli benchmark weight.nqx --batch 16
```

The `.nqx` format stores row-major uint32 binary factors, FP16 scales, tensor
shapes, configuration, diagnostics, and SHA-256 checksums. Loading never invokes
pickle.

## Verification

### Real Qwen3 benchmarks

Version 0.4 includes a real-model runner with built-in selections for
`Qwen/Qwen3-0.6B-Base` and `Qwen/Qwen3-4B-Base`. It measures matched-token
WikiText-2 perplexity, compact teacher KL, top-1 and generation agreement,
prefill/generation throughput, load time, checkpoint bytes, and CUDA memory.
Runs are fingerprinted, atomically saved, and resumable.

```bash
pip install -e ".[benchmark]"

./scripts/preflight_qwen.sh 0.6b
./scripts/benchmark_qwen.sh 0.6b
./scripts/benchmark_qwen.sh 4b
```

Inspect a run without PyTorch or a model download:

```bash
nqx-bench list-models
nqx-bench run --model "qwen .6" --variants nqx-balanced --quick --dry-run
```

Use `./scripts/benchmark_qwen_full_sweep.sh` for released NanoQuant versus NQX
strict versus NQX balanced, or `NQX_BACKEND=gemv
./scripts/benchmark_qwen_kernel.sh 0.6b` after compiling the CUDA extension.
See [`docs/REAL_MODEL_BENCHMARKS.md`](docs/REAL_MODEL_BENCHMARKS.md) for metric
definitions, fairness controls, output schema, and recovery behavior.

### Portable verification

Run the CPU test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Run the deterministic 0.3-versus-0.2 equal-storage comparison and the portable
runtime-cache benchmark:

```bash
PYTHONPATH=src python benchmarks/compare_v03.py \
  --output benchmarks/results_v03_equal_storage.json

PYTHONPATH=src python benchmarks/runtime_cache.py \
  --output benchmarks/results_runtime_cache.json
```

Across 12 deterministic matrix cases, version 0.3 strict reduced mean exact
serialized reconstruction error by 20.744% versus the 0.2 path and won or tied
all 12 cases at the same 1.10 BPW. The main gain comes from expanding requested
rank 24 to rank 32, which occupies the same single uint32 word per factor row.
Balanced improved the mean by 21.458%, but used 1.1667 BPW because eight extra
FP16 rank coefficients are stored; it is not a matched-BPW claim.

On the bundled 2048-by-2048 NumPy runtime case, prepared factors were 4.08x
faster than the one-shot factorized path and used a cache equal to 6.25% of a
dense FP32 matrix. This is a portable-reference measurement, not a CUDA-kernel
claim. These results are matrix-level tests, not perplexity or downstream-task
claims.
This source environment did not contain PyTorch, CUDA, or model weights, so the
release does not claim Qwen perplexity or GPU speed numbers. The 0.4 runner is
the executable path for collecting them on target hardware; see
[`docs/REAL_MODEL_BENCHMARKS.md`](docs/REAL_MODEL_BENCHMARKS.md).

## Design and limitations

The effective bits-per-weight calculation includes uint32 padding and every
stored FP16 scale. Padding reclamation increases rank only inside factor words
that are already allocated. The global allocator never silently exceeds its
target budget. Exact refitting is chunked by output rows, so it does not need a
second full dense reconstruction buffer during optimization.

The CUDA extension cannot be runtime-tested on a machine without PyTorch, CUDA,
and an NVIDIA GPU. The bundle therefore distinguishes three verification
levels:

1. CPU reference and artifact tests, which are run in any NumPy environment;
2. PyTorch integration and source compilation checks;
3. GPU kernel, perplexity, task, throughput, VRAM, and energy measurements,
   which require the target CUDA system.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the mathematical formulation and
[`docs/VALIDATION.md`](docs/VALIDATION.md) for the exact test boundaries.

## License and attribution

NanoQuant-X is distributed under Apache License 2.0. The upstream NanoQuant
copyright, license, and NOTICE are retained. Modified upstream files carry a
modification notice.

If this work is used in research, cite the original NanoQuant paper:

```bibtex
@article{chong2026nanoquant,
  title={NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models},
  author={Chong, Hyochan and Kim, Dongkyu and Kim, Changdong and Choi, Minseop},
  journal={arXiv preprint arXiv:2602.06694},
  year={2026}
}
```
