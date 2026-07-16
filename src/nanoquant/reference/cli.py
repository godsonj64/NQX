"""Command-line interface for the portable NanoQuant-X reference path.

Copyright 2026 Godson Johnson
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .core import NQXConfig, quantize_matrix, rank_for_budget
from .format import inspect_nqx, load_nqx, save_nqx


def _json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))


def _compact_diagnostics(diagnostics: Any) -> dict[str, Any] | None:
    if diagnostics is None:
        return None
    data = diagnostics.to_dict()
    history = data.pop("history", [])
    data["history_length"] = len(history)
    data["last_history_entry"] = history[-1] if history else None
    return data


def _load_array(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    return np.load(path, allow_pickle=False)


def _rank_from_args(args: argparse.Namespace, shape: tuple[int, int], rank_scale: bool) -> int:
    if args.rank is not None:
        return int(args.rank)
    return rank_for_budget(
        shape[0],
        shape[1],
        float(args.bpw),
        rank_scale=rank_scale,
        alignment=args.rank_alignment,
    )


def command_quantize(args: argparse.Namespace) -> None:
    weight = np.load(args.input, allow_pickle=False)
    rank_scale = args.profile == "balanced"
    rank = _rank_from_args(args, tuple(weight.shape), rank_scale)
    config = NQXConfig(
        rank=rank,
        max_iters=args.max_iters,
        min_iters=min(args.min_iters, args.max_iters),
        tolerance=args.tolerance,
        patience=args.patience,
        init=args.init,
        adaptive_rho=args.adaptive_rho,
        scale_iters=args.scale_iters,
        polish_iters=args.polish_iters,
        reclaim_packed_padding=not args.no_reclaim_padding,
        scale_ridge=args.scale_ridge,
        candidate_selection=not args.no_candidate_selection,
        storage_aware=not args.no_storage_aware,
        storage_refine_iters=args.storage_refine_iters,
        rank_scale=rank_scale,
        precondition_shrinkage=args.precondition_shrinkage,
        precondition_clip=args.precondition_clip,
        seed=args.seed,
    )
    started = time.perf_counter()
    quantized = quantize_matrix(
        weight,
        config,
        input_hessian=_load_array(args.input_hessian),
        output_hessian=_load_array(args.output_hessian),
    )
    output = save_nqx(quantized, args.output)
    _json(
        {
            "output": str(output),
            "shape": list(weight.shape),
            "rank": quantized.rank,
            "profile": args.profile,
            "effective_bpw": quantized.effective_bpw(),
            "elapsed_seconds": time.perf_counter() - started,
            "diagnostics": _compact_diagnostics(quantized.diagnostics),
        }
    )


def command_inspect(args: argparse.Namespace) -> None:
    _json(inspect_nqx(args.artifact))


def command_validate(args: argparse.Namespace) -> None:
    quantized = load_nqx(args.artifact)
    weight = np.load(args.reference, allow_pickle=False)
    if tuple(weight.shape) != (quantized.out_features, quantized.in_features):
        raise ValueError(
            f"Reference shape {weight.shape} does not match artifact shape "
            f"{(quantized.out_features, quantized.in_features)}."
        )
    reconstructed = quantized.reconstruct(dtype=np.float64)
    residual = weight.astype(np.float64) - reconstructed
    relative_mse = float(np.sum(residual * residual) / max(float(np.sum(weight.astype(np.float64) ** 2)), 1e-30))
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.samples, quantized.in_features)).astype(np.float32)
    dense = x @ reconstructed.astype(np.float32).T
    factorized = quantized.matmul(x)
    _json(
        {
            "relative_weight_mse": relative_mse,
            "max_matmul_absolute_error": float(np.max(np.abs(dense - factorized))),
            "mean_matmul_absolute_error": float(np.mean(np.abs(dense - factorized))),
            "effective_bpw": quantized.effective_bpw(),
        }
    )


def _median_time(function: Any, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples))


def command_benchmark(args: argparse.Namespace) -> None:
    quantized = load_nqx(args.artifact)
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.batch, quantized.in_features)).astype(np.float32)
    dense_weight = quantized.reconstruct()
    dense_time = _median_time(lambda: x @ dense_weight.T, args.warmup, args.repeats)
    uncached_time = _median_time(
        lambda: quantized.matmul(x, prepared=False),
        args.warmup,
        args.repeats,
    )
    quantized.prepare_runtime(np.float32)
    factor_time = _median_time(lambda: quantized.matmul(x), args.warmup, args.repeats)
    _json(
        {
            "backend": "numpy-reference",
            "batch": args.batch,
            "dense_median_seconds": dense_time,
            "factorized_uncached_median_seconds": uncached_time,
            "factorized_prepared_median_seconds": factor_time,
            "prepared_over_uncached": factor_time / max(uncached_time, 1e-30),
            "factorized_median_seconds": factor_time,
            "factorized_over_dense": factor_time / max(dense_time, 1e-30),
            "prepared_cache_bytes": quantized.runtime_cache_bytes,
            "note": "This measures the portable reference path, not the CUDA packed kernels.",
        }
    )


def command_demo(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    left = rng.standard_normal((args.out_features, args.latent_rank))
    right = rng.standard_normal((args.latent_rank, args.in_features))
    weight = (left @ right / np.sqrt(args.latent_rank) + args.noise * rng.standard_normal(
        (args.out_features, args.in_features)
    )).astype(np.float32)
    config = NQXConfig(
        rank=args.rank,
        max_iters=args.max_iters,
        min_iters=min(args.min_iters, args.max_iters),
        rank_scale=args.profile == "balanced",
        seed=args.seed,
    )
    quantized = quantize_matrix(weight, config)
    output = save_nqx(quantized, args.output)
    _json(
        {
            "output": str(output),
            "effective_bpw": quantized.effective_bpw(),
            "diagnostics": _compact_diagnostics(quantized.diagnostics),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nqx-ref", description="Portable NanoQuant-X tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quantize = subparsers.add_parser("quantize", help="Quantize a floating-point .npy matrix")
    quantize.add_argument("input")
    quantize.add_argument("output")
    rank_group = quantize.add_mutually_exclusive_group(required=True)
    rank_group.add_argument("--rank", type=int)
    rank_group.add_argument("--bpw", type=float)
    quantize.add_argument("--profile", choices=("strict", "balanced"), default="balanced")
    quantize.add_argument("--rank-alignment", type=int, default=32)
    quantize.add_argument("--input-hessian")
    quantize.add_argument("--output-hessian")
    quantize.add_argument("--max-iters", type=int, default=400)
    quantize.add_argument("--min-iters", type=int, default=64)
    quantize.add_argument("--tolerance", type=float, default=1e-4)
    quantize.add_argument("--patience", type=int, default=10_000)
    quantize.add_argument("--init", choices=("random", "spectral"), default="random")
    quantize.add_argument("--adaptive-rho", action="store_true")
    quantize.add_argument("--scale-iters", type=int, default=8)
    quantize.add_argument("--scale-ridge", type=float, default=1e-6)
    quantize.add_argument("--polish-iters", type=int, default=1)
    quantize.add_argument("--no-reclaim-padding", action="store_true")
    quantize.add_argument("--no-candidate-selection", action="store_true")
    quantize.add_argument("--no-storage-aware", action="store_true")
    quantize.add_argument("--storage-refine-iters", type=int, default=2)
    quantize.add_argument("--precondition-shrinkage", type=float, default=0.2)
    quantize.add_argument("--precondition-clip", type=float, default=8.0)
    quantize.add_argument("--seed", type=int, default=0)
    quantize.set_defaults(function=command_quantize)

    inspect = subparsers.add_parser("inspect", help="Inspect and verify an .nqx artifact")
    inspect.add_argument("artifact")
    inspect.set_defaults(function=command_inspect)

    validate = subparsers.add_parser("validate", help="Compare an artifact with its original .npy matrix")
    validate.add_argument("artifact")
    validate.add_argument("reference")
    validate.add_argument("--samples", type=int, default=8)
    validate.add_argument("--seed", type=int, default=0)
    validate.set_defaults(function=command_validate)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark the portable factorized path")
    benchmark.add_argument("artifact")
    benchmark.add_argument("--batch", type=int, default=16)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--repeats", type=int, default=10)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.set_defaults(function=command_benchmark)

    demo = subparsers.add_parser("demo", help="Generate and quantize a deterministic synthetic matrix")
    demo.add_argument("output")
    demo.add_argument("--out-features", type=int, default=128)
    demo.add_argument("--in-features", type=int, default=128)
    demo.add_argument("--latent-rank", type=int, default=12)
    demo.add_argument("--rank", type=int, default=32)
    demo.add_argument("--noise", type=float, default=0.08)
    demo.add_argument("--profile", choices=("strict", "balanced"), default="balanced")
    demo.add_argument("--max-iters", type=int, default=96)
    demo.add_argument("--min-iters", type=int, default=32)
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(function=command_demo)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
