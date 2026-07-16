# Audit and Results

## Audited baseline

- Paper source: arXiv:2602.06694v1.
- Official code: SamsungLabs/NanoQuant commit
  `a9e0a430881ff80d83b622c3129e330dc33c04f5`.
- Audit date: 2026-07-16.

## Material issues found

1. The layer reconstruction log compared the dense weight with the continuous
   factor product. Inference signs the factors and applies extracted scales, so
   the reported value was not the deployed reconstruction error.
2. The CLI exposed `admm_reg`, but the NanoQuant dispatcher did not pass it to
   `factorize_admm_nanoquant`.
3. Per-layer rank calculation used a continuous bit formula and did not count
   uint32 padding or optimize a global model budget.
4. Model-level distillation cached full vocabulary logits for every calibration
   token, which can require many gigabytes of host memory.
5. `get_shrunk_stats(..., shrinkage=1.0)` returned unshrunk statistics because
   the boundary value followed the early-return path.
6. Dataset-path detection compared `type(dataset_path)` with the string
   `"str"`, so the intended path branch could not execute.
7. Checkpoint loading automatically retried with unrestricted pickle when
   weights-only loading failed.
8. Hub loading could pass a directory to the single-file checkpoint loader,
   and SafeTensors was not resolved there.
9. The command-line dataclass omitted `model_kd_epochs`, although downstream
   code required the field.
10. Non-word-aligned ranks could pay for uint32 padding without using those
    already-stored binary lanes.
11. Scale optimization and diagnostics occurred in FP32 even though portable
    artifacts store FP16 scales and the PyTorch module stores BF16 scales.
12. Portable artifact ZIP metadata was not byte-deterministic, and the loader
    did not reject all unexpected members or oversized declared payloads.

## Implemented changes

- Added `factorize_admm_nqx`, which preserves the official initializer and
  optimizes the exact signed runtime representation.
- Added weighted alternating channel-scale refitting, an optional rank-scale
  solve, chunked reconstruction, and a non-regression guard.
- Replaced layer-local approximate rank calculation in NQX mode with a global,
  allocator using exact packed storage costs. It can consume measured
  per-layer rate-distortion points when they are available.
- Added packed-padding reclamation: a requested rank is expanded to every lane
  in its already-paid uint32 word without adding strict-profile factor bits.
- Added exact deployment-candidate selection and FP16/BF16 storage-aware scale
  projection with monotone global-gain placement.
- Added a prepared NumPy runtime that caches scale-fused factors, eliminating
  repeated casting and scale broadcasts during repeated inference.
- Replaced full-logit KD caching with top-k probabilities and an explicit tail
  bucket; added real KD batching and moved the retained teacher back to CPU.
  With vocabulary 151,936 and `k=128`, the tensor payload falls from 303,872
  bytes to 772 bytes per token, a theoretical 393.6-fold reduction before
  container overhead.
- Added a deterministic, checksummed, non-pickle `.nqx` format, aggregate size
  limits, duplicate/extra-member rejection, and a portable NumPy path.
- Added safe directory/SafeTensors checkpoint resolution and removed automatic
  unrestricted-pickle fallback.
- Fixed configuration propagation, shrinkage boundary behavior, dataset-path
  detection, and device-map propagation.
- Added a wheel, tests, configuration, runbook, and equal-work benchmark.

## Verified results

- 28 portable and benchmark-planning tests passed.
- Every Python source compiled successfully.
- Every bundled shell launcher passed `bash -n` validation.
- Every bundled Qwen benchmark JSON passed strict schema validation and a CLI
  dry run.
- The source built into `nanoquant_x-0.4.0-py3-none-any.whl`.
- The wheel installed into an isolated target and passed version/model-registry
  imports plus the installed `nqx-bench` entry point and dry-run planner.
- In the 12-case version comparison, 0.3 strict improved mean exact serialized
  reconstruction error by 20.744% versus 0.2 and won or tied 12/12 cases at
  exactly 1.10 BPW. Requested rank 24 reclaimed its paid padding as rank 32.
- Balanced improved the mean by 21.458%, but used 1.1667 BPW and is not
  presented as a matched-BPW result.
- In the bundled 2048-feature NumPy benchmark, prepared factorized inference
  was 4.08x faster than the uncached path; its 1 MiB cache was 6.25% of the
  equivalent dense FP32 matrix. This is not a CUDA claim.

## Not verified in this environment

PyTorch, CUDA, Triton, Transformers, model weights, and an NVIDIA GPU were not
available in the execution environment. Consequently, no end-to-end perplexity,
task accuracy, GPU throughput, VRAM, energy, or kernel claims are made. The
0.4 release includes an executable, resumable runner for Qwen3 0.6B and 4B; its
commands, metric definitions, and acceptance controls are provided in
`docs/REAL_MODEL_BENCHMARKS.md` and `docs/FULL_MODEL_RUNBOOK.md`.
