"""Dependency-light tests for the real-model benchmark planning layer."""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from nanoquant.bench.config import BenchmarkConfig
from nanoquant.bench.metrics import compare_results, percentile, timing_summary
from nanoquant.bench.models import resolve_model
from nanoquant.bench.runner import main
from nanoquant.bench.storage import atomic_write_json, read_json, stable_fingerprint


class ModelRegistryTests(unittest.TestCase):
    def test_qwen_aliases_resolve_to_official_ids(self) -> None:
        for alias in ("qwen3-0.6b", "0.6b", ".6", "qwen .6", "Qwen/Qwen3-0.6B-Base"):
            self.assertEqual(resolve_model(alias).model_id, "Qwen/Qwen3-0.6B-Base")
        for alias in ("qwen3-4b", "4b", "Qwen/Qwen3-4B-Base"):
            self.assertEqual(resolve_model(alias).model_id, "Qwen/Qwen3-4B-Base")

    def test_custom_hub_id_is_preserved(self) -> None:
        profile = resolve_model("organization/custom-small-model")
        self.assertEqual(profile.model_id, "organization/custom-small-model")
        self.assertIsNone(profile.parameters)


class BenchmarkConfigTests(unittest.TestCase):
    def test_baseline_is_first_and_appears_once(self) -> None:
        config = BenchmarkConfig(variants=("nqx-balanced", "baseline", "nqx-strict")).validate()
        self.assertEqual(config.resolved_variants, ("baseline", "nqx-balanced", "nqx-strict"))

    def test_quick_does_not_change_quantization_fingerprint(self) -> None:
        config = BenchmarkConfig(variants=("nqx-balanced",)).validate()
        quick = config.quick()
        self.assertLess(quick.max_eval_tokens, config.max_eval_tokens)
        self.assertEqual(
            config.quantization_fingerprint("nqx-balanced"),
            quick.quantization_fingerprint("nqx-balanced"),
        )
        self.assertNotEqual(config.fingerprint, quick.fingerprint)

    def test_variant_presets_are_distinct(self) -> None:
        config = BenchmarkConfig().validate()
        released = config.quantization_config("nanoquant")
        strict = config.quantization_config("nqx-strict")
        balanced = config.quantization_config("nqx-balanced")
        self.assertEqual(released["admm_type"], "nanoquant")
        self.assertFalse(strict["nqx_rank_scale"])
        self.assertTrue(balanced["nqx_rank_scale"])
        self.assertNotEqual(
            config.quantization_fingerprint("nqx-strict"),
            config.quantization_fingerprint("nqx-balanced"),
        )

    def test_rejects_unknown_and_invalid_fields(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkConfig.from_mapping({"model": "0.6b", "typo_field": 1})
        with self.assertRaises(ValueError):
            BenchmarkConfig(sequence_length=128, stride=129).validate()
        with self.assertRaises(ValueError):
            BenchmarkConfig(variants=("not-a-variant",)).validate()

    def test_bundled_configs_are_valid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = {
            "bench_qwen3_0_6b.json": "Qwen/Qwen3-0.6B-Base",
            "bench_qwen3_4b.json": "Qwen/Qwen3-4B-Base",
            "bench_qwen3_0_6b_full_sweep.json": "Qwen/Qwen3-0.6B-Base",
        }
        for name, model_id in expected.items():
            config = BenchmarkConfig.from_json(root / "configs" / name)
            self.assertEqual(config.profile.model_id, model_id)
            self.assertTrue(config.quantize_if_missing)


class StorageTests(unittest.TestCase):
    def test_atomic_json_preserves_previous_file_on_nonfinite_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_write_json(path, {"status": "old"})
            with self.assertRaises(ValueError):
                atomic_write_json(path, {"bad": math.nan})
            self.assertEqual(read_json(path), {"status": "old"})

    def test_fingerprint_is_key_order_independent(self) -> None:
        self.assertEqual(stable_fingerprint({"a": 1, "b": 2}), stable_fingerprint({"b": 2, "a": 1}))


class MetricTests(unittest.TestCase):
    def test_timing_summary_and_percentile(self) -> None:
        summary = timing_summary([1.0, 2.0, 3.0], work_per_run=12)
        self.assertEqual(summary["runs"], 3)
        self.assertAlmostEqual(summary["units_per_second"], 6.0)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0], 0.9), 2.8)

    @staticmethod
    def _result(variant: str, ppl: float, prefill: float, generation: float, tokens: list[int]) -> dict:
        return {
            "status": "complete",
            "variant": variant,
            "experiment_fingerprint": "same",
            "model": {"model_id": "Qwen/Qwen3-0.6B-Base"},
            "metrics": {
                "load": {"seconds": 2.0 if variant == "baseline" else 1.0},
                "model": {"resident_parameter_bytes": 1200},
                "checkpoint": {"bytes": None if variant == "baseline" else 150},
                "quality": {
                    "perplexity": {"value": ppl},
                    "fidelity": {"top1_agreement": 0.75} if variant != "baseline" else None,
                },
                "throughput": {
                    "prefill": [{"length": 128, "tokens_per_second": prefill}],
                    "generation": {"tokens_per_second": generation},
                },
                "generation": {
                    "samples": [{"prompt": "p", "new_token_ids": tokens}],
                },
            },
        }

    def test_comparison_uses_named_units_and_agreement(self) -> None:
        baseline = self._result("baseline", 10.0, 100.0, 20.0, [1, 2, 3])
        candidate = self._result("nqx-balanced", 11.0, 125.0, 30.0, [1, 2, 4])
        comparison = compare_results(baseline, candidate)
        self.assertAlmostEqual(comparison["quality"]["perplexity_ratio"], 1.1)
        self.assertAlmostEqual(comparison["throughput"]["prefill"][0]["speedup"], 1.25)
        self.assertAlmostEqual(comparison["throughput"]["generation_speedup"], 1.5)
        self.assertAlmostEqual(comparison["storage"]["checkpoint_to_baseline_parameter_ratio"], 0.125)
        self.assertAlmostEqual(comparison["quality"]["generation_agreement"]["position_agreement"], 2 / 3)


class CLITests(unittest.TestCase):
    def test_dry_run_requires_no_heavy_runtime(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["run", "--model", "qwen .6", "--variants", "nqx-balanced", "--quick", "--dry-run"])
        self.assertEqual(code, 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["model"]["model_id"], "Qwen/Qwen3-0.6B-Base")
        self.assertEqual(plan["variants"], ["baseline", "nqx-balanced"])
        self.assertEqual(plan["evaluation"]["max_eval_tokens"] if "max_eval_tokens" in plan["evaluation"] else
                         plan["evaluation"]["perplexity_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()

