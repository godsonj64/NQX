"""Deterministic matrix-level comparison of the paper-style and enhanced cores.

This benchmark deliberately does not claim end-to-end LLM accuracy.  It tests
the exact deployed weight representation on several controlled matrix families.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from nanoquant.reference import NQXConfig, paper_style_baseline, quantize_matrix


@dataclass
class Result:
    case: str
    seed: int
    baseline_error: float
    strict_error: float
    balanced_error: float
    baseline_iterations: int
    enhanced_iterations: int
    strict_bpw: float
    balanced_bpw: float
    baseline_seconds: float
    strict_seconds: float
    balanced_seconds: float


def make_matrix(case: str, seed: int, m: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if case == "gaussian":
        matrix = rng.standard_normal((m, n))
    elif case == "low_rank":
        latent = max(4, min(m, n) // 8)
        matrix = rng.standard_normal((m, latent)) @ rng.standard_normal((latent, n)) / np.sqrt(latent)
        matrix += 0.08 * rng.standard_normal((m, n))
    elif case == "heavy_tail":
        matrix = rng.standard_t(3.0, size=(m, n))
    elif case == "outliers":
        matrix = rng.standard_normal((m, n))
        indices = rng.choice(matrix.size, size=max(1, matrix.size // 100), replace=False)
        matrix.flat[indices] *= 12.0
    else:
        raise ValueError(case)
    return matrix.astype(np.float32)


def timed(function):
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def run(args: argparse.Namespace) -> dict:
    results: list[Result] = []
    for case in ("gaussian", "low_rank", "heavy_tail", "outliers"):
        for seed in range(args.seeds):
            weight = make_matrix(case, seed, args.out_features, args.in_features)
            baseline, baseline_time = timed(
                lambda: paper_style_baseline(weight, args.rank, iterations=args.baseline_iters, seed=0)
            )
            common = dict(
                rank=args.rank,
                max_iters=args.enhanced_iters,
                min_iters=args.enhanced_iters,
                patience=10_000,
                scale_iters=args.scale_iters,
                polish_iters=args.polish_iters,
                seed=0,
            )
            strict, strict_time = timed(lambda: quantize_matrix(weight, NQXConfig(**common)))
            balanced, balanced_time = timed(
                lambda: quantize_matrix(weight, NQXConfig(**common, rank_scale=True))
            )
            results.append(
                Result(
                    case=case,
                    seed=seed,
                    baseline_error=baseline.diagnostics.deployed_error,
                    strict_error=strict.diagnostics.deployed_error,
                    balanced_error=balanced.diagnostics.deployed_error,
                    baseline_iterations=args.baseline_iters,
                    enhanced_iterations=args.enhanced_iters,
                    strict_bpw=strict.effective_bpw(),
                    balanced_bpw=balanced.effective_bpw(),
                    baseline_seconds=baseline_time,
                    strict_seconds=strict_time,
                    balanced_seconds=balanced_time,
                )
            )

    baseline_errors = np.array([item.baseline_error for item in results])
    strict_errors = np.array([item.strict_error for item in results])
    balanced_errors = np.array([item.balanced_error for item in results])
    return {
        "scope": "matrix-level exact deployed representation; no STE, KD, perplexity, or task evaluation",
        "configuration": vars(args),
        "summary": {
            "cases": len(results),
            "mean_baseline_error": float(np.mean(baseline_errors)),
            "mean_strict_error": float(np.mean(strict_errors)),
            "mean_balanced_error": float(np.mean(balanced_errors)),
            "strict_relative_improvement_percent": float(100 * (1 - np.mean(strict_errors) / np.mean(baseline_errors))),
            "balanced_relative_improvement_percent": float(
                100 * (1 - np.mean(balanced_errors) / np.mean(baseline_errors))
            ),
            "strict_wins_or_ties": int(np.sum(strict_errors <= baseline_errors + 1e-12)),
            "balanced_wins_or_ties": int(np.sum(balanced_errors <= baseline_errors + 1e-12)),
        },
        "results": [asdict(item) for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-features", type=int, default=96)
    parser.add_argument("--in-features", type=int, default=80)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--baseline-iters", type=int, default=400)
    parser.add_argument("--enhanced-iters", type=int, default=400)
    parser.add_argument("--scale-iters", type=int, default=8)
    parser.add_argument("--polish-iters", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run(args)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
