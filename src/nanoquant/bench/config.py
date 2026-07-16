"""Validated benchmark configuration and quantization presets."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from .models import ModelProfile, resolve_model
from .storage import stable_fingerprint


SCHEMA_VERSION = "nqx-real-model-benchmark/v1"
SUPPORTED_VARIANTS = ("baseline", "nanoquant", "nqx-strict", "nqx-balanced")
SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")
SUPPORTED_BACKENDS = ("torch", "gemv", "gemm", "gemlite")

DEFAULT_PROMPTS = (
    "The capital of France is",
    "Explain why the sky appears blue in two sentences.",
    "A fast sorting algorithm can be described as follows:",
    "In a distant future, humanity discovered",
)


def _tuple_of_strings(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        result = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, (list, tuple)):
        result = tuple(str(part).strip() for part in value if str(part).strip())
    else:
        raise TypeError(f"{field_name} must be a comma-separated string or list")
    return result


def _tuple_of_ints(value: Any, field_name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise TypeError(f"{field_name} must be a comma-separated string or list")
    try:
        return tuple(int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain integers") from exc


@dataclass(frozen=True)
class BenchmarkConfig:
    """Complete, serializable settings for one baseline/candidate experiment."""

    model: str = "qwen3-0.6b"
    variants: tuple[str, ...] = ("baseline", "nqx-balanced")
    output_dir: str = "benchmark-results"
    checkpoint_dir: str = "outputs/bench-checkpoints"
    run_name: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    backend: str = "torch"
    seed: int = 0
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    attn_implementation: str = "sdpa"
    resume: bool = True
    quantize_if_missing: bool = False
    allow_unverified_checkpoint: bool = False
    hash_checkpoint: bool = False

    # Evaluation and fidelity.
    dataset: str = "wikitext2"
    dataset_split: str = "test"
    sequence_length: int = 1024
    stride: int = 512
    max_eval_tokens: int = 16_384
    fidelity_samples: int = 4
    fidelity_length: int = 256
    teacher_topk: int = 128
    prefill_lengths: tuple[int, ...] = (128, 512, 1024)
    generation_max_new_tokens: int = 64
    warmup_runs: int = 2
    repeat_runs: int = 5
    prompts: tuple[str, ...] = DEFAULT_PROMPTS

    # Common quantization settings. Variant-specific representation choices are
    # applied by quantization_config().
    bits: float = 1.0
    num_calib_samples: int = 128
    calibration_sequence_length: int = 2048
    calib_dataset: str = "wikitext2"
    calib_shrinkage: float = 0.2
    calib_strategy: str = "online"
    tune_nonfact: bool = True
    nonfact_lr: float = 1e-4
    nonfact_batch_size: int = 4
    nonfact_epochs: int = 8
    admm_outer_iters: int = 400
    admm_inner_iters: int = 5
    admm_reg: float = 3e-2
    admm_penalty_scheduler: str = "linear"
    nqx_scale_iters: int = 4
    nqx_scale_ridge: float = 1e-6
    nqx_chunk_rows: int = 256
    tune_fact: bool = True
    fact_binary_lr: float = 1e-5
    fact_scale_lr: float = 1e-5
    fact_bias_lr: float = 1e-5
    fact_batch_size: int = 1
    fact_epochs: int = 8
    tune_model: bool = True
    model_kd_lr: float = 1e-6
    model_kd_batch_size: int = 1
    model_kd_epochs: int = 8
    nqx_kd_topk: int = 128
    nqx_kd_temperature: float = 1.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BenchmarkConfig":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown benchmark configuration field(s): {', '.join(unknown)}")
        values = dict(data)
        for name in ("variants", "prompts"):
            if name in values:
                values[name] = _tuple_of_strings(values[name], name)
        if "prefill_lengths" in values:
            values["prefill_lengths"] = _tuple_of_ints(values["prefill_lengths"], "prefill_lengths")
        return cls(**values).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TypeError("Benchmark config JSON must contain an object")
        return cls.from_mapping(value)

    @property
    def profile(self) -> ModelProfile:
        return resolve_model(self.model)

    @property
    def resolved_variants(self) -> tuple[str, ...]:
        # Every compressed run gets a baseline reference and teacher cache. The
        # insertion is explicit in the resolved configuration and result files.
        candidates = tuple(variant for variant in self.variants if variant != "baseline")
        if candidates:
            return ("baseline", *candidates)
        return ("baseline",)

    def validate(self) -> "BenchmarkConfig":
        errors: list[str] = []
        if not self.variants:
            errors.append("variants must not be empty")
        invalid_variants = sorted(set(self.variants) - set(SUPPORTED_VARIANTS))
        if invalid_variants:
            errors.append(f"unsupported variants: {', '.join(invalid_variants)}")
        if len(set(self.variants)) != len(self.variants):
            errors.append("variants must not contain duplicates")
        if self.dtype not in SUPPORTED_DTYPES:
            errors.append(f"dtype must be one of {', '.join(SUPPORTED_DTYPES)}")
        if self.backend not in SUPPORTED_BACKENDS:
            errors.append(f"backend must be one of {', '.join(SUPPORTED_BACKENDS)}")
        if self.backend != "torch" and self.dtype == "float32":
            errors.append("CUDA/GemLite compressed backends require float16 or bfloat16")
        if self.backend != "torch" and not self.device.startswith("cuda"):
            errors.append("CUDA/GemLite compressed backends require a cuda device")
        if self.dataset != "wikitext2":
            errors.append("this release currently supports dataset='wikitext2'")
        if self.dataset_split not in {"train", "validation", "test"}:
            errors.append("dataset_split must be train, validation, or test")
        if self.sequence_length < 2:
            errors.append("sequence_length must be at least 2")
        if self.stride <= 0 or self.stride > self.sequence_length:
            errors.append("stride must be in [1, sequence_length]")
        if self.max_eval_tokens < 2:
            errors.append("max_eval_tokens must be at least 2")
        for name in ("fidelity_samples", "fidelity_length", "teacher_topk", "generation_max_new_tokens",
                     "repeat_runs", "num_calib_samples", "calibration_sequence_length", "admm_outer_iters",
                     "admm_inner_iters", "nqx_chunk_rows", "nonfact_batch_size", "fact_batch_size",
                     "model_kd_batch_size"):
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be positive")
        if self.fidelity_length < 2:
            errors.append("fidelity_length must be at least 2")
        if self.warmup_runs < 0:
            errors.append("warmup_runs must be non-negative")
        if not self.prefill_lengths or any(length <= 0 for length in self.prefill_lengths):
            errors.append("prefill_lengths must contain positive lengths")
        if not self.prompts or any(not prompt for prompt in self.prompts):
            errors.append("prompts must contain non-empty text")
        if not math.isfinite(self.bits) or self.bits <= 0:
            errors.append("bits must be finite and positive")
        if not 0.0 <= self.calib_shrinkage <= 1.0:
            errors.append("calib_shrinkage must lie in [0, 1]")
        for name in ("nonfact_lr", "admm_reg", "nqx_scale_ridge", "fact_binary_lr", "fact_scale_lr",
                     "fact_bias_lr", "model_kd_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                errors.append(f"{name} must be finite and non-negative")
        if not math.isfinite(self.nqx_kd_temperature) or self.nqx_kd_temperature <= 0:
            errors.append("nqx_kd_temperature must be finite and positive")
        for name in ("nonfact_epochs", "fact_epochs", "model_kd_epochs", "nqx_scale_iters"):
            if getattr(self, name) < 0:
                errors.append(f"{name} must be non-negative")
        profile = self.profile
        requested_lengths = (self.sequence_length, self.fidelity_length, self.calibration_sequence_length,
                             *self.prefill_lengths)
        if profile.context_length and max(requested_lengths) > profile.context_length:
            errors.append(
                f"requested length {max(requested_lengths)} exceeds {profile.display_name}'s "
                f"{profile.context_length}-token context"
            )
        if self.run_name is not None and (not self.run_name.strip() or "/" in self.run_name or "\\" in self.run_name):
            errors.append("run_name must be a non-empty single path component")
        if errors:
            raise ValueError("Invalid benchmark configuration: " + "; ".join(errors))
        return self

    def with_overrides(self, **overrides: Any) -> "BenchmarkConfig":
        values = asdict(self)
        values.update({key: value for key, value in overrides.items() if value is not None})
        return self.from_mapping(values)

    def quick(self) -> "BenchmarkConfig":
        """Reduce evaluation cost without changing quantization quality."""

        return replace(
            self,
            max_eval_tokens=min(self.max_eval_tokens, 2048),
            fidelity_samples=min(self.fidelity_samples, 1),
            fidelity_length=min(self.fidelity_length, 128),
            prefill_lengths=tuple(length for length in self.prefill_lengths if length <= 512) or (128,),
            generation_max_new_tokens=min(self.generation_max_new_tokens, 16),
            warmup_runs=min(self.warmup_runs, 1),
            repeat_runs=min(self.repeat_runs, 2),
            prompts=self.prompts[:2],
        ).validate()

    def to_dict(self, resolved: bool = False) -> dict:
        value = asdict(self)
        if resolved:
            value["model"] = self.profile.model_id
            value["model_profile"] = self.profile.to_dict()
            value["variants"] = list(self.resolved_variants)
            value["schema_version"] = SCHEMA_VERSION
        return value

    def experiment_payload(self) -> dict:
        value = self.to_dict(resolved=True)
        # Location and execution-control fields do not alter measured work.
        for key in (
            "output_dir",
            "checkpoint_dir",
            "run_name",
            "resume",
            "quantize_if_missing",
            "allow_unverified_checkpoint",
            "hash_checkpoint",
        ):
            value.pop(key, None)
        return value

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.experiment_payload())

    @property
    def run_directory(self) -> Path:
        name = self.run_name or self.fingerprint[:12]
        return Path(self.output_dir) / self.profile.slug / name

    def checkpoint_path(self, variant: str) -> Path:
        if variant == "baseline":
            raise ValueError("baseline does not have a NanoQuant checkpoint")
        bits = str(self.bits).replace(".", "p")
        quantization_id = self.quantization_fingerprint(variant)[:12]
        return Path(self.checkpoint_dir) / self.profile.slug / f"{variant}-{bits}bpw-{quantization_id}.pt"

    def quantization_config(self, variant: str) -> dict:
        """Build the exact NanoQuantConfigDataclass payload for a variant."""

        if variant not in SUPPORTED_VARIANTS or variant == "baseline":
            raise ValueError(f"{variant!r} is not a compressed variant")
        if variant == "nanoquant":
            admm_type, rank_scale, storage_aware, adaptive_rank = "nanoquant", False, False, False
        elif variant == "nqx-strict":
            admm_type, rank_scale, storage_aware, adaptive_rank = "nqx", False, True, True
        else:
            admm_type, rank_scale, storage_aware, adaptive_rank = "nqx", True, True, True
        return {
            "model_id": self.profile.model_id,
            "revision": self.revision,
            "bits": self.bits,
            "seed": self.seed,
            "num_calib_samples": self.num_calib_samples,
            "calib_dataset": self.calib_dataset,
            "calib_shrinkage": self.calib_shrinkage,
            "calib_strategy": self.calib_strategy,
            "seqlen": self.calibration_sequence_length,
            "device_map": self.device,
            "tune_nonfact": self.tune_nonfact,
            "nonfact_lr": self.nonfact_lr,
            "nonfact_batch_size": self.nonfact_batch_size,
            "nonfact_epochs": self.nonfact_epochs,
            "admm_type": admm_type,
            "admm_outer_iters": self.admm_outer_iters,
            "admm_inner_iters": self.admm_inner_iters,
            "admm_reg": self.admm_reg,
            "admm_penalty_scheduler": self.admm_penalty_scheduler,
            "admm_print_steps": False,
            "nqx_scale_iters": self.nqx_scale_iters,
            "nqx_scale_ridge": self.nqx_scale_ridge,
            "nqx_rank_scale": rank_scale,
            "nqx_chunk_rows": self.nqx_chunk_rows,
            "nqx_storage_aware": storage_aware,
            "nqx_adaptive_rank": adaptive_rank,
            "tune_fact": self.tune_fact,
            "fact_binary_lr": self.fact_binary_lr,
            "fact_scale_lr": self.fact_scale_lr,
            "fact_bias_lr": self.fact_bias_lr,
            "fact_batch_size": self.fact_batch_size,
            "fact_epochs": self.fact_epochs,
            "tune_model": self.tune_model,
            "model_kd_lr": self.model_kd_lr,
            "model_kd_batch_size": self.model_kd_batch_size,
            "model_kd_epochs": self.model_kd_epochs,
            "nqx_kd_topk": self.nqx_kd_topk,
            "nqx_kd_temperature": self.nqx_kd_temperature,
        }

    def quantization_fingerprint(self, variant: str) -> str:
        return stable_fingerprint(self.quantization_config(variant))
