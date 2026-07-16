"""Pure result aggregation used by the real-model runner and tests."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(seconds: Iterable[float], work_per_run: int | float | None = None) -> dict[str, float | int]:
    samples = [float(value) for value in seconds]
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timing samples must be finite and positive")
    result: dict[str, float | int] = {
        "runs": len(samples),
        "mean_seconds": statistics.fmean(samples),
        "median_seconds": statistics.median(samples),
        "stdev_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "p90_seconds": percentile(samples, 0.9),
    }
    if work_per_run is not None:
        work = float(work_per_run)
        if not math.isfinite(work) or work <= 0:
            raise ValueError("work_per_run must be finite and positive")
        result["work_per_run"] = work_per_run
        result["units_per_second"] = work * len(samples) / sum(samples)
    return result


def _nested(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if denominator == 0 or not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return None
    return float(numerator) / float(denominator)


def _agreement(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_samples = {item["prompt"]: item for item in baseline.get("samples", [])}
    candidate_samples = {item["prompt"]: item for item in candidate.get("samples", [])}
    prompts = sorted(set(baseline_samples) & set(candidate_samples))
    exact = 0
    matching_positions = 0
    compared_positions = 0
    common_prefix = 0
    total_reference_positions = 0
    per_prompt: list[dict[str, Any]] = []
    for prompt in prompts:
        left = list(baseline_samples[prompt].get("new_token_ids", []))
        right = list(candidate_samples[prompt].get("new_token_ids", []))
        is_exact = left == right
        exact += int(is_exact)
        width = max(len(left), len(right))
        position_matches = sum(a == b for a, b in zip(left, right))
        prefix = 0
        for a, b in zip(left, right):
            if a != b:
                break
            prefix += 1
        matching_positions += position_matches
        compared_positions += width
        common_prefix += prefix
        total_reference_positions += len(left)
        per_prompt.append({
            "prompt": prompt,
            "exact_match": is_exact,
            "matching_positions": position_matches,
            "compared_positions": width,
            "common_prefix_tokens": prefix,
            "baseline_tokens": len(left),
            "candidate_tokens": len(right),
        })
    return {
        "prompts": len(prompts),
        "exact_match_rate": exact / len(prompts) if prompts else None,
        "position_agreement": matching_positions / compared_positions if compared_positions else None,
        "common_prefix_fraction_of_baseline": common_prefix / total_reference_positions
        if total_reference_positions else None,
        "per_prompt": per_prompt,
    }


def compare_results(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Build an honest baseline/candidate comparison from complete result JSON."""

    baseline_ppl = _nested(baseline, "metrics", "quality", "perplexity", "value")
    candidate_ppl = _nested(candidate, "metrics", "quality", "perplexity", "value")
    baseline_load = _nested(baseline, "metrics", "load", "seconds")
    candidate_load = _nested(candidate, "metrics", "load", "seconds")
    baseline_resident = _nested(baseline, "metrics", "model", "resident_parameter_bytes")
    candidate_checkpoint = _nested(candidate, "metrics", "checkpoint", "bytes")
    baseline_generation = _nested(
        baseline, "metrics", "throughput", "generation", "tokens_per_second"
    )
    candidate_generation = _nested(
        candidate, "metrics", "throughput", "generation", "tokens_per_second"
    )

    baseline_prefill = {
        int(item["length"]): item.get("tokens_per_second")
        for item in _nested(baseline, "metrics", "throughput", "prefill", default=[])
    }
    candidate_prefill = {
        int(item["length"]): item.get("tokens_per_second")
        for item in _nested(candidate, "metrics", "throughput", "prefill", default=[])
    }
    prefill = []
    for length in sorted(set(baseline_prefill) & set(candidate_prefill)):
        prefill.append({
            "length": length,
            "baseline_tokens_per_second": baseline_prefill[length],
            "candidate_tokens_per_second": candidate_prefill[length],
            "speedup": _safe_ratio(candidate_prefill[length], baseline_prefill[length]),
        })

    quality_delta = None
    if isinstance(baseline_ppl, (int, float)) and isinstance(candidate_ppl, (int, float)):
        quality_delta = float(candidate_ppl) - float(baseline_ppl)

    return {
        "schema_version": "nqx-real-model-comparison/v1",
        "baseline_variant": baseline.get("variant"),
        "candidate_variant": candidate.get("variant"),
        "model": candidate.get("model", baseline.get("model")),
        "quality": {
            "baseline_perplexity": baseline_ppl,
            "candidate_perplexity": candidate_ppl,
            "absolute_perplexity_delta": quality_delta,
            "perplexity_ratio": _safe_ratio(candidate_ppl, baseline_ppl),
            "fidelity_to_baseline": _nested(candidate, "metrics", "quality", "fidelity"),
            "generation_agreement": _agreement(
                _nested(baseline, "metrics", "generation", default={}),
                _nested(candidate, "metrics", "generation", default={}),
            ),
        },
        "storage": {
            "baseline_resident_parameter_bytes": baseline_resident,
            "candidate_checkpoint_bytes": candidate_checkpoint,
            "checkpoint_to_baseline_parameter_ratio": _safe_ratio(candidate_checkpoint, baseline_resident),
        },
        "load": {
            "baseline_seconds": baseline_load,
            "candidate_seconds": candidate_load,
            "candidate_to_baseline_ratio": _safe_ratio(candidate_load, baseline_load),
        },
        "throughput": {
            "prefill": prefill,
            "baseline_generation_tokens_per_second": baseline_generation,
            "candidate_generation_tokens_per_second": candidate_generation,
            "generation_speedup": _safe_ratio(candidate_generation, baseline_generation),
        },
        "notes": [
            "Perplexity uses identical token IDs, context length, stride, and token budget.",
            "Fidelity KL is computed over teacher top-k categories plus one probability-preserving tail bucket.",
            "Checkpoint bytes are compared with baseline in-memory parameter bytes; these are different but named units.",
            "Generation throughput is end-to-end and includes prefill; prefill throughput is reported separately.",
        ],
    }

