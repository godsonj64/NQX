"""Command-line entry point for real Qwen/NanoQuant-X benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig, SUPPORTED_BACKENDS, SUPPORTED_DTYPES, SUPPORTED_VARIANTS
from .metrics import compare_results
from .models import list_model_profiles
from .preflight import collect_preflight, format_preflight
from .storage import atomic_write_json, read_json


def _boolean(parser: argparse.ArgumentParser, flag: str, destination: str, help_text: str) -> None:
    parser.add_argument(
        flag,
        dest=destination,
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_text,
    )


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="JSON configuration; CLI values override it")
    parser.add_argument("--model", help="Built-in alias or Hugging Face model ID")
    parser.add_argument("--variants", help="Comma list: baseline,nanoquant,nqx-strict,nqx-balanced")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--device", help="Torch device, for example cuda:0 or cpu")
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES)
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, help="Compressed-model inference backend")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--revision")
    parser.add_argument("--attn-implementation", dest="attn_implementation")
    _boolean(parser, "--trust-remote-code", "trust_remote_code", "Allow model repository Python code")
    _boolean(parser, "--local-files-only", "local_files_only", "Disable Hugging Face downloads")
    _boolean(parser, "--resume", "resume", "Reuse fingerprint-matched completed result stages")
    _boolean(parser, "--quantize-if-missing", "quantize_if_missing", "Create missing NanoQuant checkpoints")
    _boolean(
        parser,
        "--allow-unverified-checkpoint",
        "allow_unverified_checkpoint",
        "Allow a checkpoint without a provenance sidecar (mismatched sidecars are always rejected)",
    )
    _boolean(parser, "--hash-checkpoint", "hash_checkpoint", "SHA-256 hash large checkpoints")

    parser.add_argument("--dataset-split", choices=("train", "validation", "test"))
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-eval-tokens", type=int)
    parser.add_argument("--fidelity-samples", type=int)
    parser.add_argument("--fidelity-length", type=int)
    parser.add_argument("--teacher-topk", type=int)
    parser.add_argument("--prefill-lengths", help="Comma-separated token lengths")
    parser.add_argument("--generation-max-new-tokens", type=int)
    parser.add_argument("--warmup-runs", type=int)
    parser.add_argument("--repeat-runs", type=int)
    parser.add_argument("--prompts", help="JSON file containing a list of prompt strings")

    parser.add_argument("--bits", type=float)
    parser.add_argument("--num-calib-samples", type=int)
    parser.add_argument("--calibration-sequence-length", type=int)
    parser.add_argument("--calib-dataset")
    parser.add_argument("--calib-shrinkage", type=float)
    parser.add_argument("--calib-strategy", choices=("online", "two_phase", "dbf", "none"))
    _boolean(parser, "--tune-nonfact", "tune_nonfact", "Tune non-factorized weights")
    parser.add_argument("--nonfact-lr", type=float)
    parser.add_argument("--nonfact-batch-size", type=int)
    parser.add_argument("--nonfact-epochs", type=int)
    parser.add_argument("--admm-outer-iters", type=int)
    parser.add_argument("--admm-inner-iters", type=int)
    parser.add_argument("--admm-reg", type=float)
    parser.add_argument("--admm-penalty-scheduler")
    parser.add_argument("--nqx-scale-iters", type=int)
    parser.add_argument("--nqx-scale-ridge", type=float)
    parser.add_argument("--nqx-chunk-rows", type=int)
    _boolean(parser, "--tune-fact", "tune_fact", "Tune factorized weights")
    parser.add_argument("--fact-binary-lr", type=float)
    parser.add_argument("--fact-scale-lr", type=float)
    parser.add_argument("--fact-bias-lr", type=float)
    parser.add_argument("--fact-batch-size", type=int)
    parser.add_argument("--fact-epochs", type=int)
    _boolean(parser, "--tune-model", "tune_model", "Run model-level scale distillation")
    parser.add_argument("--model-kd-lr", type=float)
    parser.add_argument("--model-kd-batch-size", type=int)
    parser.add_argument("--model-kd-epochs", type=int)
    parser.add_argument("--nqx-kd-topk", type=int)
    parser.add_argument("--nqx-kd-temperature", type=float)


def _read_prompts(path: str | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    value = read_json(path)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("--prompts must point to a JSON list of non-empty strings")
    return tuple(value)


def _configuration(args: argparse.Namespace) -> BenchmarkConfig:
    config = BenchmarkConfig.from_json(args.config) if args.config else BenchmarkConfig().validate()
    names = {field.name for field in __import__("dataclasses").fields(BenchmarkConfig)}
    control = {"command", "config", "quick", "dry_run", "json", "debug"}
    values = vars(args)
    overrides = {name: values[name] for name in names if name in values and values[name] is not None and name not in control}
    if getattr(args, "prompts", None) is not None:
        overrides["prompts"] = _read_prompts(args.prompts)
    config = config.with_overrides(**overrides)
    if getattr(args, "quick", False):
        config = config.quick()
    return config.validate()


def _plan(config: BenchmarkConfig) -> dict[str, Any]:
    checkpoints = []
    for variant in config.resolved_variants:
        if variant == "baseline":
            continue
        path = config.checkpoint_path(variant)
        checkpoints.append({
            "variant": variant,
            "path": str(path),
            "present": path.is_file(),
            "will_quantize_if_missing": config.quantize_if_missing,
            "quantization_fingerprint": config.quantization_fingerprint(variant),
        })
    return {
        "schema_version": "nqx-real-model-plan/v1",
        "experiment_fingerprint": config.fingerprint,
        "model": config.profile.to_dict(),
        "variants": list(config.resolved_variants),
        "run_directory": str(config.run_directory),
        "checkpoints": checkpoints,
        "evaluation": {
            "perplexity_tokens": config.max_eval_tokens,
            "sequence_length": config.sequence_length,
            "stride": config.stride,
            "fidelity_samples": config.fidelity_samples,
            "fidelity_length": config.fidelity_length,
            "teacher_topk": config.teacher_topk,
            "prefill_lengths": list(config.prefill_lengths),
            "generation_max_new_tokens": config.generation_max_new_tokens,
            "warmup_runs": config.warmup_runs,
            "repeat_runs": config.repeat_runs,
        },
        "backend": config.backend,
        "configuration": config.to_dict(resolved=True),
        "phases": [
            "preflight",
            "fixed tokenization",
            "baseline perplexity/fidelity/throughput",
            "compact teacher-cache save",
            "candidate load or quantization",
            "candidate perplexity/fidelity/throughput",
            "atomic comparison and summary",
        ],
    }


def _print_models(as_json: bool) -> int:
    profiles = [profile.to_dict() for profile in list_model_profiles()]
    if as_json:
        print(json.dumps(profiles, indent=2, sort_keys=True))
        return 0
    print("KEY          OFFICIAL MODEL ID             PARAMS  CONTEXT  GPU GUIDE  HOST GUIDE")
    for item in profiles:
        print(
            f"{item['key']:<12} {item['model_id']:<29} "
            f"{item['parameters'] / 1e9:>4.1f}B  {item['context_length']:>7}  "
            f"{item['recommended_gpu_gib']:>5.0f} GiB   {item['recommended_host_gib']:>5.0f} GiB"
        )
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    config = _configuration(args)
    report = collect_preflight(config)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_preflight(report))
    return 0 if report["ok"] else 2


def _cmd_run(args: argparse.Namespace) -> int:
    config = _configuration(args)
    plan = _plan(config)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    report = collect_preflight(config)
    print(format_preflight(report), flush=True)
    config.run_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config.run_directory / "preflight.json", report)
    if not report["ok"]:
        print("Preflight failed; no model or dataset was downloaded.", file=sys.stderr)
        return 2
    from .runtime import run_benchmarks

    summary = run_benchmarks(config, progress=lambda message: print(message, flush=True))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Benchmark complete: {summary['run_directory']}")
        print(f"Summary: {Path(summary['run_directory']) / 'summary.json'}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    baseline = read_json(args.baseline)
    candidate = read_json(args.candidate)
    if baseline.get("status") != "complete" or candidate.get("status") != "complete":
        raise ValueError("Both inputs must be complete benchmark result JSON files")
    if baseline.get("variant") != "baseline":
        raise ValueError("The first input must be a baseline result")
    if baseline.get("experiment_fingerprint") != candidate.get("experiment_fingerprint"):
        raise ValueError("Results have different experiment fingerprints and are not directly comparable")
    comparison = compare_results(baseline, candidate)
    if args.output:
        atomic_write_json(args.output, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nqx-bench",
        description="Reproducible baseline-vs-NanoQuant-X benchmarks on real Hugging Face models",
    )
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("list-models", help="List built-in small-model aliases")
    models.add_argument("--json", action="store_true")

    preflight = subparsers.add_parser("preflight", help="Check dependencies, GPU, disk, and checkpoints")
    _add_config_arguments(preflight)
    preflight.add_argument("--quick", action="store_true", help="Apply low-cost evaluation settings")
    preflight.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run", help="Run or resume a real-model benchmark")
    _add_config_arguments(run)
    run.add_argument(
        "--quick",
        action="store_true",
        help="Reduce evaluation tokens/repeats only; quantization settings remain research-grade",
    )
    run.add_argument("--dry-run", action="store_true", help="Resolve and print the plan without heavy imports")
    run.add_argument("--json", action="store_true", help="Print final summary JSON")

    compare = subparsers.add_parser("compare", help="Compare two already-complete result files")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-models":
            return _print_models(args.json)
        if args.command == "preflight":
            return _cmd_preflight(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "compare":
            return _cmd_compare(args)
        parser.error(f"Unknown command: {args.command}")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

