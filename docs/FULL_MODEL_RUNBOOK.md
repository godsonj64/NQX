# Full-Model Validation Runbook

> NanoQuant-X 0.4 automates the baseline/candidate portions of this runbook.
> Start with `./scripts/preflight_qwen.sh 0.6b` and
> `./scripts/benchmark_qwen.sh 0.6b`; see
> [`REAL_MODEL_BENCHMARKS.md`](REAL_MODEL_BENCHMARKS.md). The manual commands
> below remain useful for phase-level debugging and additional lm-eval tasks.

## 1. Environment

Use a clean Python 3.12 environment and record the GPU, driver, CUDA, PyTorch,
Transformers, and Git commit before testing.

```bash
python - <<'PY'
import platform
import torch
import transformers
print("python", platform.python_version())
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
PY
```

Compile and smoke-test the kernels:

```bash
cd src/nanoquant/kernel
bash compile_kernel.sh
cd ../../..
```

## 2. First target

Start with `Qwen/Qwen3-0.6B-Base`. Run three separately saved configurations:

1. upstream-compatible: `--admm_type nanoquant`;
2. NQX strict: `--admm_type nqx --nqx_rank_scale false`;
3. NQX balanced: `--admm_type nqx --nqx_rank_scale true`.

Keep the calibration sample indices, seed, sequence length, epochs, and target
BPW identical. Do not compare checkpoints whose actual packed BPW differs
without reporting the difference.

## 3. Compression command

```bash
python -m nanoquant.main \
  --model_id Qwen/Qwen3-0.6B-Base \
  --qmodel_path outputs/qwen3-0.6b-nqx-balanced.pt \
  --bits 1.0 \
  --seed 0 \
  --num_calib_samples 128 \
  --seqlen 2048 \
  --calib_dataset wikitext2 \
  --calib_strategy online \
  --calib_shrinkage 0.2 \
  --admm_type nqx \
  --admm_outer_iters 400 \
  --admm_inner_iters 5 \
  --admm_reg 0.03 \
  --admm_penalty_scheduler linear \
  --nqx_rank_scale true \
  --nqx_adaptive_rank true \
  --nqx_scale_iters 4 \
  --nqx_scale_ridge 1e-6 \
  --nqx_chunk_rows 256 \
  --nqx_storage_aware true \
  --nqx_kd_topk 128 \
  --nonfact_epochs 8 \
  --fact_epochs 8 \
  --model_kd_epochs 8 \
  --model_kd_lr 1e-6 \
  --ppl_task wikitext2 \
  --zeroshot_task "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge" \
  --device_map cuda
```

## 4. Required metrics

Record for every run:

- wall-clock compression time by phase;
- maximum allocated and reserved VRAM;
- peak host memory;
- exact packed checkpoint bytes and effective BPW;
- per-layer exact signed reconstruction error before and after refit;
- requested versus effective rank and reclaimed packed lanes per layer;
- BF16-stored versus FP32-preprojection scale error;
- WikiText-2 perplexity under the same stride and context length;
- individual zero-shot task scores and their mean;
- greedy generation agreement on a fixed prompt suite; and
- three-run mean and standard deviation for timing metrics.

## 5. Kernel benchmark

Place the checkpoint where `bench_decode.sh` expects it or pass the explicit
arguments supported by `test_decode.py`.

```bash
cd src/nanoquant/kernel
bash bench_decode.sh
```

Measure GEMV at batch 1 and GEMM at representative prefill/batched shapes.
Always include FP16/BF16 PyTorch, the upstream NanoQuant kernel path, and the NQX
checkpoint on the same GPU and power state.

## 6. Acceptance criteria

Treat NQX as successful only if all conditions hold:

1. no layer or full-model NaN/Inf;
2. packed round-trip and kernel/reference output agreement pass tolerance;
3. actual model BPW does not exceed the requested budget;
4. perplexity and task accuracy improve at matched BPW, or BPW decreases at
   matched quality;
5. compact KD does not regress full-logit KD beyond a predeclared tolerance;
6. decode throughput is not reduced by the optional rank scale; and
7. reported speedups exclude model loading and one-time kernel compilation
   unless explicitly labeled.
