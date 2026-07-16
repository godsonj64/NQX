# Validation Status

## Verified in the portable environment

The included `unittest` suite verifies:

1. exact uint32 pack/unpack round trips at aligned and unaligned ranks;
2. rejection of non-binary factors;
3. exact packed-bit budget accounting;
4. global rank allocation honoring both sensitivity proxies and measured
   rate-distortion points;
5. duplicate layer-budget rejection;
6. prepared and uncached factorized multiplication matching dense reconstruction;
7. deterministic factorization under a fixed seed;
8. exact-objective scale refitting not regressing the paper-style deployed
   candidate in the fixed-work test;
9. balanced-profile rank-scale validity;
10. free packed-padding rank reclamation;
11. monotone deployment-candidate selection;
12. FP16-exact storage-aware scales and diagnostics;
13. `.nqx` FP16 artifact round trips;
14. byte-identical deterministic artifact creation;
15. SHA-256 detection of payload tampering; and
16. rejection of unexpected artifact members;
17. Qwen 0.6B/4B alias resolution to official model IDs;
18. custom model-ID preservation;
19. automatic baseline ordering;
20. quick/full quantization-fingerprint equivalence;
21. distinct released/strict/balanced quantization presets;
22. invalid and unknown benchmark configuration rejection;
23. all bundled Qwen benchmark configurations;
24. atomic JSON recovery after a failed non-finite write;
25. key-order-independent fingerprints;
26. timing statistics and interpolated percentiles;
27. baseline/candidate ratio and generation-agreement calculations; and
28. a dependency-light CLI dry-run.

Run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Matrix-level benchmark

`benchmarks/compare_v03.py` evaluates Gaussian, structured low-rank,
heavy-tailed, and sparse-outlier matrices across three deterministic seeds. It
compares version 0.3 with the exact FP16-serialized 0.2 representation, not the
continuous ADMM proxy.

The bundled result uses 400 initializer iterations for each path. Version 0.3
strict reclaimed requested rank 24 as rank 32 and reduced mean relative
reconstruction error by 20.744% versus 0.2, winning or tying all 12 cases at an
identical 1.10 BPW. Both ranks occupy one uint32 factor word per row. Balanced
improved the mean by 21.458% but used 1.1667 BPW, so it is not a matched-BPW
claim.

`benchmarks/runtime_cache.py` measures the portable repeated-inference path.
On the bundled 2048-feature, rank-64, batch-16 case, prepared factors were 4.08x
faster than uncached factorized inference. The prepared cache was 1 MiB, or
6.25% of the equivalent 16 MiB dense FP32 matrix. Timing is environment-specific
and does not describe the CUDA kernels.

These results establish a narrow matrix-level non-regression property. They do
not establish language-model perplexity, task accuracy, throughput, or energy
claims.

## Source-level checks

All Python sources are compiled with `compileall`. The CUDA sources are retained
from the official Apache-2.0 repository. The enhanced PyTorch integration is
source-compatible with the existing `NanoQuantLinear` representation, including
the optional middle scale already supported by the CUDA and GemLite paths.

The `nqx-bench` planner, schema, checkpoint provenance, shell syntax, and
bundled Qwen configurations are also checked without importing PyTorch. A real
run performs an additional preflight before downloading data or weights.

## Requires target hardware

The following checks require PyTorch, the model weights, calibration data, and
an NVIDIA CUDA system:

- extension compilation and kernel correctness;
- FP16/BF16 agreement for GEMV and GEMM;
- WikiText-2 perplexity and zero-shot tasks;
- generation agreement and stability;
- checkpoint load time and peak host/GPU memory;
- prefill and decode tokens per second;
- energy per token; and
- comparisons at exactly matched model-level BPW.

Use `REAL_MODEL_BENCHMARKS.md` and `FULL_MODEL_RUNBOOK.md` to perform those
measurements. Do not convert the portable matrix results into end-to-end LLM
claims.
