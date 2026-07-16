# Real-Model Benchmark Harness

NanoQuant-X 0.4 adds a reproducible, resumable benchmark runner for real
Hugging Face causal language models. The built-in targets are
[`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) and
[`Qwen/Qwen3-4B-Base`](https://huggingface.co/Qwen/Qwen3-4B-Base). Custom model
IDs remain possible, but only the built-in profiles have resource guidance and
known context metadata.

The runner does not contain prerecorded model numbers. It downloads or reads
the selected model, evaluates the baseline, loads or creates each compressed
checkpoint, and writes the measurements produced on the current machine.

## 1. Install and inspect

For baseline and reference-backend benchmarks:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[benchmark]"

nqx-bench list-models
nqx-bench run --model qwen3-0.6b --variants nqx-balanced --dry-run
```

For the custom CUDA backends, install the full environment and compile the
extension:

```bash
pip install -e ".[llm]"
cd src/nanoquant/kernel
bash compile_kernel.sh
cd ../../..
```

Qwen3 needs Transformers 4.51 or newer. The release pins 4.51.3 so all compared
runs use the same Transformers implementation.

## 2. One-command runs

Run the preflight before reserving a long GPU session:

```bash
./scripts/preflight_qwen.sh 0.6b
./scripts/preflight_qwen.sh 4b
```

Run the recommended baseline-versus-balanced comparison:

```bash
./scripts/benchmark_qwen.sh 0.6b
./scripts/benchmark_qwen.sh 4b
```

Accepted 0.6B aliases include `0.6b`, `.6`, `qwen .6`, and `qwen3-0.6b` at the
CLI level. Shell targets should be passed as `0.6b` or `.6`.

A short evaluation pass is useful for validating a newly created checkpoint:

```bash
./scripts/benchmark_qwen_quick.sh 0.6b
```

`--quick` changes only evaluation token counts, prompt count, warmups, and timed
repetitions. It does not weaken the calibration, ADMM, reconstruction, or
distillation settings. The quick and full runs therefore reuse the same
fingerprinted checkpoint while writing separate result directories.

Run all released representations on 0.6B:

```bash
./scripts/benchmark_qwen_full_sweep.sh
```

This creates three independent compressed checkpoints and can take much longer
than the recommended two-model comparison.

## 3. Variants

| Variant | ADMM path | Per-rank scale | Packed-aware rank allocation |
| --- | --- | --- | --- |
| `baseline` | None; BF16/FP16 Hugging Face model | N/A | N/A |
| `nanoquant` | Released NanoQuant-compatible path | No | No |
| `nqx-strict` | Exact deployed-objective NQX path | No | Yes |
| `nqx-balanced` | Exact deployed-objective NQX path | Yes | Yes |

If a compressed variant is requested without `baseline`, the runner prepends
the baseline automatically. This is necessary for a fixed teacher cache and a
directly comparable result.

## 4. Backends

| Backend | Purpose | Setup | Storage interpretation |
| --- | --- | --- | --- |
| `torch` | Portable correctness/reference run | PyTorch only | Packed checkpoint is unpacked to BF16 factors in memory |
| `gemv` | Batch-1/decode-oriented custom CUDA kernel | Compile `binary_kernels` | Packed runtime tensors |
| `gemm` | Prefill/batched custom CUDA kernel | Compile `binary_kernels` | Packed runtime tensors |
| `gemlite` | GemLite packed runtime | Install/configure GemLite | Packed runtime tensors |

The baseline always uses its ordinary PyTorch path. Select a compressed backend
with:

```bash
NQX_BACKEND=gemv ./scripts/benchmark_qwen_kernel.sh 0.6b
```

Do not compare throughput from different machines, power states, software
versions, or backends as though it were a controlled speedup. Every result JSON
records the selected backend, package versions, CUDA runtime, GPU model, and
compute capability.

## 5. Direct CLI examples

Evaluate an existing checkpoint but refuse to quantize a missing one:

```bash
nqx-bench run \
  --model qwen3-0.6b \
  --variants baseline,nqx-balanced \
  --no-quantize-if-missing \
  --backend torch
```

Create missing checkpoints and run a three-way comparison:

```bash
nqx-bench run \
  --model qwen3-0.6b \
  --variants nanoquant,nqx-strict,nqx-balanced \
  --quantize-if-missing \
  --bits 1.0 \
  --device cuda:0
```

Override only evaluation work from a bundled config:

```bash
nqx-bench run \
  --config configs/bench_qwen3_4b.json \
  --max-eval-tokens 16384 \
  --repeat-runs 5 \
  --prefill-lengths 128,512,1024,2048
```

Use `--revision COMMIT` to pin a Hugging Face revision. The resolved commit, if
provided by Transformers, is recorded in result and checkpoint metadata.

## 6. Measurements

Each variant records:

- WikiText-2 sliding-window perplexity with exact scored-token accounting;
- top-1 agreement and teacher-to-candidate KL over 128 teacher categories plus
  one probability-preserving tail bucket;
- deterministic greedy generations and token-level agreement to the baseline;
- prefill tokens/second at fixed token lengths;
- end-to-end generation tokens/second, explicitly including prompt prefill;
- warmup-excluded mean, median, standard deviation, minimum, maximum, and p90;
- model/download or checkpoint/quantization load time;
- current and peak CUDA allocated/reserved bytes;
- registered parameter and buffer bytes;
- exact compressed checkpoint bytes; and
- Python, package, CUDA, GPU, model-revision, and configuration provenance.

The fidelity metric is a coarsened KL. All vocabulary items outside the
teacher's top-k are grouped into one tail category, so the JSON calls it
`topk_tail_kl_nats` rather than implying full-vocabulary KL.

The output does not conflate disk and memory. A `torch`-backend NQX checkpoint
is packed on disk but unpacked for portable inference. CUDA allocator memory is
the correct live-memory measurement for that run. Packed-kernel runs report
their own allocator state after kernel preparation.

## 7. Fairness controls

The runner enforces or records the following controls:

1. identical token IDs, sequence length, stride, token budget, and prompts;
2. deterministic fidelity sample positions derived from the experiment seed;
3. a compact baseline cache saved before unloading the baseline model;
4. device synchronization around every timed call;
5. warmups excluded from timing distributions;
6. fixed-length greedy generation for timing and ordinary greedy generation
   for sample agreement;
7. exact checkpoint bytes instead of a theoretical rank-only size; and
8. separate experiment and quantization fingerprints.

The built-in resource guidance is intentionally conservative. A 4B
quantization run holds more state than ordinary 4B inference because the
current pipeline loads both working and teacher models on the host and performs
GPU block reconstruction. Preflight warns below 48 GiB host RAM and 24 GiB GPU
memory, but actual requirements depend on sequence length, backend, and system.

## 8. Crash recovery and provenance

Results are written through fsynced temporary files and atomic rename. The
runner saves one result per variant, then creates comparisons and `summary.json`.
Re-running the same command resumes only complete files with the same
experiment fingerprint.

Checkpoint names contain the first 12 characters of a separate quantization
fingerprint. Every newly created checkpoint gets a sidecar containing its full
quantization settings. A present checkpoint with a mismatched sidecar is always
rejected. A legacy checkpoint without a sidecar is rejected unless
`--allow-unverified-checkpoint` is explicit.

Typical result tree:

```text
benchmark-results/qwen3-0.6b/4f2c.../
├── preflight.json
├── resolved_config.json
├── manifest.json
├── teacher_cache.json
├── teacher_cache.npz
├── result-baseline.json
├── result-nqx-balanced.json
├── comparison-nqx-balanced.json
└── summary.json
```

Compare two completed result files manually:

```bash
nqx-bench compare \
  benchmark-results/.../result-baseline.json \
  benchmark-results/.../result-nqx-balanced.json \
  --output comparison.json
```

The command refuses results with different experiment fingerprints.

## 9. Current release boundary

The 0.4 source bundle validates the registry, configurations, dry-run planner,
atomic recovery, comparison calculations, and existing portable NQX path in a
CPU-only environment. It does not ship invented Qwen perplexity or throughput
numbers. Run the scripts on the target CUDA machine and cite the resulting JSON
alongside the exact hardware and backend.

