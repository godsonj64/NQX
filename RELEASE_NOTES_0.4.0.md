# NanoQuant-X 0.4.0 — Real Small-Model Benchmarks

NanoQuant-X 0.4.0 turns the previous manual full-model runbook into an
executable benchmark release. It supports direct selection of Qwen3 0.6B Base
and Qwen3 4B Base, compares the original model with one or more NanoQuant
representations, and saves enough provenance to resume or audit a long run.

## Start here

```bash
pip install -e ".[benchmark]"

./scripts/preflight_qwen.sh 0.6b
./scripts/benchmark_qwen.sh 0.6b
./scripts/benchmark_qwen.sh 4b
```

The portable planner can be checked without ML dependencies:

```bash
nqx-bench list-models
nqx-bench run --model "qwen .6" --variants nqx-balanced --quick --dry-run
```

## New in this release

- Built-in Qwen3 0.6B and 4B profiles and aliases.
- Baseline, released NanoQuant, NQX strict, and NQX balanced variants.
- Fixed-token WikiText-2 perplexity and compact top-k-plus-tail fidelity.
- Top-1 and deterministic generation agreement.
- Prefill and end-to-end generation throughput distributions.
- Load/quantization time, checkpoint size, registered tensor size, and CUDA
  current/peak allocation.
- `torch`, `gemv`, `gemm`, and `gemlite` compressed-model backends.
- Atomic per-variant results, checksummed teacher cache, resume, and manual
  comparison commands.
- Separate experiment and quantization fingerprints with checkpoint sidecars.
- Preflight checks for packages, Transformers compatibility, CUDA, device,
  kernel imports, RAM guidance, disk space, and checkpoint provenance.
- Hugging Face revision propagation and resolved-commit recording.
- Five shell launchers and three ready-to-run JSON configurations.

## Validation boundary

The release environment passed all 28 dependency-light tests, compiled every
Python source, validated every shell launcher, built the 0.4.0 wheel, and
installed the wheel into an isolated target where the `nqx-bench` entry point
and dry-run planner passed.

The release environment did not have PyTorch, model weights, CUDA, or an NVIDIA
GPU. No Qwen perplexity, throughput, VRAM, or speedup number is claimed in the
bundle. Those values must come from `nqx-bench` on the target machine and will
be written under `benchmark-results/`.

See [`docs/REAL_MODEL_BENCHMARKS.md`](docs/REAL_MODEL_BENCHMARKS.md) for metric
definitions and [`docs/FULL_MODEL_RUNBOOK.md`](docs/FULL_MODEL_RUNBOOK.md) for
additional acceptance criteria.

