# Benchmarks

- `compare_v03.py` is the primary release comparison. It measures the exact
  serialized 0.2 strict path against 0.3 strict at identical packed BPW, plus
  the 0.3 balanced profile.
- `results_v03_equal_storage.json` is its deterministic 12-case result.
- `runtime_cache.py` compares prepared and uncached portable NumPy inference.
- `results_runtime_cache.json` records the release-environment result; timing
  is hardware and BLAS dependent.
- `compare_core.py` retains the original paper-style comparison harness.
- `results_v02_equal_work.json` is the historical 0.2 release result.

All reconstruction results are matrix-level. They do not establish language
model perplexity, downstream-task quality, CUDA throughput, VRAM, or energy.

Real-model experiments are run through the installed `nqx-bench` command or
the launchers in `scripts/`. They write machine-specific outputs under
`benchmark-results/` rather than bundling unverified numbers here. See
`docs/REAL_MODEL_BENCHMARKS.md`.
