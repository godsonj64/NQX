"""Compare NanoQuant-X 0.3 with the serialized 0.2 reference path.

The strict comparison reclaims unused lanes inside the same uint32 factor
words, so its packed storage is exactly matched.  This remains a matrix-level
benchmark and does not stand in for model perplexity or task evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from compare_core import make_matrix
from nanoquant.reference import NQXConfig, paper_style_baseline, quantize_matrix


@dataclass
class Result:
    case: str
    seed: int
    requested_rank: int
    v02_rank: int
    v03_strict_rank: int
    v03_balanced_rank: int
    paper_error: float
    v02_serialized_error: float
    v03_strict_error: float
    v03_balanced_error: float
    v02_bpw: float
    v03_strict_bpw: float
    v03_balanced_bpw: float
    paper_seconds: float
    v02_seconds: float
    v03_strict_seconds: float
    v03_balanced_seconds: float


def timed(function):
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def run(args: argparse.Namespace) -> dict:
    results: list[Result] = []
    for case in ("gaussian", "low_rank", "heavy_tail", "outliers"):
        for seed in range(args.seeds):
            weight = make_matrix(case, seed, args.out_features, args.in_features)
            paper, paper_time = timed(
                lambda: paper_style_baseline(weight, args.rank, iterations=args.iters, seed=0)
            )
            legacy_config = NQXConfig(
                rank=args.rank,
                max_iters=args.iters,
                min_iters=args.iters,
                patience=10_000,
                scale_iters=args.scale_iters,
                polish_iters=1,
                reclaim_packed_padding=False,
                candidate_selection=False,
                storage_aware=False,
                rank_scale=False,
                seed=0,
            )
            current_common = dict(
                rank=args.rank,
                max_iters=args.iters,
                min_iters=args.iters,
                patience=10_000,
                scale_iters=args.scale_iters,
                polish_iters=args.polish_iters,
                reclaim_packed_padding=True,
                candidate_selection=True,
                storage_aware=True,
                storage_refine_iters=args.storage_refine_iters,
                seed=0,
            )
            legacy, legacy_time = timed(lambda: quantize_matrix(weight, legacy_config))
            strict, strict_time = timed(
                lambda: quantize_matrix(weight, NQXConfig(**current_common, rank_scale=False))
            )
            balanced, balanced_time = timed(
                lambda: quantize_matrix(weight, NQXConfig(**current_common, rank_scale=True))
            )
            results.append(
                Result(
                    case=case,
                    seed=seed,
                    requested_rank=args.rank,
                    v02_rank=legacy.rank,
                    v03_strict_rank=strict.rank,
                    v03_balanced_rank=balanced.rank,
                    paper_error=paper.diagnostics.deployed_error,
                    v02_serialized_error=legacy.diagnostics.serialized_deployed_error,
                    v03_strict_error=strict.diagnostics.deployed_error,
                    v03_balanced_error=balanced.diagnostics.deployed_error,
                    v02_bpw=legacy.effective_bpw(),
                    v03_strict_bpw=strict.effective_bpw(),
                    v03_balanced_bpw=balanced.effective_bpw(),
                    paper_seconds=paper_time,
                    v02_seconds=legacy_time,
                    v03_strict_seconds=strict_time,
                    v03_balanced_seconds=balanced_time,
                )
            )

    paper_errors = np.array([item.paper_error for item in results])
    legacy_errors = np.array([item.v02_serialized_error for item in results])
    strict_errors = np.array([item.v03_strict_error for item in results])
    balanced_errors = np.array([item.v03_balanced_error for item in results])
    return {
        "scope": "matrix-level exact serialized/deployed representation; no STE, KD, perplexity, or tasks",
        "configuration": vars(args),
        "summary": {
            "cases": len(results),
            "mean_paper_error": float(np.mean(paper_errors)),
            "mean_v02_serialized_error": float(np.mean(legacy_errors)),
            "mean_v03_strict_error": float(np.mean(strict_errors)),
            "mean_v03_balanced_error": float(np.mean(balanced_errors)),
            "v03_strict_vs_v02_improvement_percent": float(
                100 * (1 - np.mean(strict_errors) / np.mean(legacy_errors))
            ),
            "v03_balanced_vs_v02_improvement_percent": float(
                100 * (1 - np.mean(balanced_errors) / np.mean(legacy_errors))
            ),
            "v03_strict_equal_storage_wins_or_ties": int(
                np.sum(strict_errors <= legacy_errors + 1e-12)
            ),
            "v03_strict_bpw": float(np.mean([item.v03_strict_bpw for item in results])),
            "v02_bpw": float(np.mean([item.v02_bpw for item in results])),
            "v03_balanced_bpw": float(np.mean([item.v03_balanced_bpw for item in results])),
        },
        "results": [asdict(item) for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-features", type=int, default=96)
    parser.add_argument("--in-features", type=int, default=80)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--scale-iters", type=int, default=8)
    parser.add_argument("--polish-iters", type=int, default=2)
    parser.add_argument("--storage-refine-iters", type=int, default=2)
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
