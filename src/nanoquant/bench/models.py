"""Small-model registry used by the reproducible benchmark runner."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelProfile:
    """Metadata and resource guidance for one benchmark target."""

    key: str
    model_id: str
    display_name: str
    parameters: Optional[int]
    non_embedding_parameters: Optional[int]
    layers: Optional[int]
    context_length: Optional[int]
    recommended_gpu_gib: float
    recommended_host_gib: float
    recommended_free_disk_gib: float
    aliases: tuple[str, ...] = ()

    @property
    def slug(self) -> str:
        return _slugify(self.key)

    def to_dict(self) -> dict:
        return asdict(self)


_PROFILES = (
    ModelProfile(
        key="qwen3-0.6b",
        model_id="Qwen/Qwen3-0.6B-Base",
        display_name="Qwen3 0.6B Base",
        parameters=600_000_000,
        non_embedding_parameters=440_000_000,
        layers=28,
        context_length=32_768,
        recommended_gpu_gib=12.0,
        recommended_host_gib=24.0,
        recommended_free_disk_gib=12.0,
        aliases=(
            "qwen-0.6b",
            "qwen3-.6b",
            "qwen-.6",
            "qwen-0.6",
            "qwen3-0.6b-base",
            "0.6b",
            ".6b",
            ".6",
        ),
    ),
    ModelProfile(
        key="qwen3-4b",
        model_id="Qwen/Qwen3-4B-Base",
        display_name="Qwen3 4B Base",
        parameters=4_000_000_000,
        non_embedding_parameters=3_600_000_000,
        layers=36,
        context_length=32_768,
        recommended_gpu_gib=24.0,
        recommended_host_gib=48.0,
        recommended_free_disk_gib=32.0,
        aliases=("qwen-4b", "qwen3-4b-base", "4b"),
    ),
)


def _normalize(value: str) -> str:
    value = re.sub(r"[\s_]+", "-", value.strip().lower())
    return re.sub(r"-+", "-", value)


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    return value.lower() or "model"


def list_model_profiles() -> tuple[ModelProfile, ...]:
    """Return the built-in profiles in stable display order."""

    return _PROFILES


def resolve_model(value: str) -> ModelProfile:
    """Resolve a friendly alias, official Hub ID, or custom model identifier."""

    if not value or not value.strip():
        raise ValueError("model must not be empty")
    normalized = _normalize(value)
    for profile in _PROFILES:
        names = (profile.key, profile.model_id, *profile.aliases)
        if normalized in {_normalize(name) for name in names}:
            return profile

    # Custom Hub IDs and local paths remain usable. Unknown resource fields are
    # deliberately null instead of guessed.
    return ModelProfile(
        key=_slugify(value),
        model_id=value.strip(),
        display_name=value.strip(),
        parameters=None,
        non_embedding_parameters=None,
        layers=None,
        context_length=None,
        recommended_gpu_gib=0.0,
        recommended_host_gib=0.0,
        recommended_free_disk_gib=0.0,
        aliases=(),
    )
