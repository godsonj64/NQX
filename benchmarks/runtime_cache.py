"""Benchmark prepared versus one-shot NumPy factorized inference."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from nanoquant.reference import QuantizedMatrix


def median_time(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    signs = lambda shape: np.where(rng.integers(0, 2, size=shape), 1, -1).astype(np.int8)
    matrix = QuantizedMatrix(
        u=signs((args.features, args.rank)),
        v=signs((args.features, args.rank)),
        scale_out=rng.uniform(0.01, 0.2, args.features),
        scale_in=rng.uniform(0.01, 0.2, args.features),
        rank_scale=rng.uniform(0.5, 1.5, args.rank),
    )
    inputs = rng.standard_normal((args.batch, args.features)).astype(np.float32)
    uncached = median_time(
        lambda: matrix.matmul(inputs, prepared=False), args.warmup, args.repeats
    )
    matrix.prepare_runtime(np.float32)
    prepared = median_time(lambda: matrix.matmul(inputs), args.warmup, args.repeats)
    report = {
        "configuration": vars(args),
        "uncached_median_seconds": uncached,
        "prepared_median_seconds": prepared,
        "prepared_over_uncached": prepared / max(uncached, 1e-30),
        "speedup": uncached / max(prepared, 1e-30),
        "prepared_cache_bytes": matrix.runtime_cache_bytes,
        "dense_fp32_weight_bytes": args.features * args.features * 4,
        "cache_over_dense": matrix.runtime_cache_bytes / (args.features * args.features * 4),
        "scope": "portable NumPy path only; not a CUDA-kernel benchmark",
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
