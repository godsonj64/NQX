# Changelog

## 0.4.0

- Adds `nqx-bench`, a real-model baseline-versus-compressed benchmark runner.
- Adds official Qwen3 0.6B Base and 4B Base profiles plus flexible aliases and
  custom Hugging Face model IDs.
- Measures fixed-token WikiText-2 perplexity, compact top-k-plus-tail KL,
  top-1/generation agreement, prefill and generation throughput, load time,
  checkpoint size, registered tensor size, and CUDA allocator memory.
- Adds crash-safe per-variant JSON, a checksummed non-pickle teacher cache,
  experiment fingerprints, quantization fingerprints, checkpoint provenance,
  resume, preflight, quick-evaluation, and dry-run modes.
- Adds `torch`, `gemv`, `gemm`, and `gemlite` compressed inference backends with
  explicit storage/memory interpretation.
- Adds normal, quick, kernel, preflight, and full-sweep shell launchers and
  research-grade 0.6B/4B JSON configurations.
- Propagates optional Hugging Face revisions through quantization, tokenizer,
  and compressed-checkpoint loading.
- Makes lm-eval and cut-cross-entropy imports lazy outside the operations that
  actually need them.
- Expands the dependency-light suite from 16 to 28 tests.

## 0.3.0

- Reclaims unused lanes in the final uint32 rank word without adding strict
  factor storage.
- Selects final and best-continuous solver states by exact deployed error.
- Projects portable scales to FP16 and production scales to BF16 before final
  selection and diagnostics.
- Adds monotone storage-aware global-gain placement.
- Adds optional measured rate-distortion curves to global rank allocation.
- Adds a lazy scale-fused NumPy runtime cache for repeated inference.
- Makes `.nqx` output byte-deterministic and strengthens ZIP/payload validation.
- Adds centralized production configuration validation and explicit argument
  precedence for model/device selection.
- Expands the portable suite from 10 to 16 tests and adds version-comparison
  and runtime-cache benchmarks.

## 0.2.0

- Introduced exact signed-objective scale refitting and optional rank scales.
- Added exact packed-bit allocation, compact top-k KD targets, safe checkpoint
  loading, the portable NumPy reference path, and checksummed `.nqx` artifacts.
