"""Read-only environment checks for long real-model benchmark runs."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import sys
import json
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig


def _version_tuple(value: str) -> tuple[int, ...]:
    pieces: list[int] = []
    for piece in value.split("."):
        digits = "".join(character for character in piece if character.isdigit())
        if not digits:
            break
        pieces.append(int(digits))
    return tuple(pieces)


def _existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _host_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def collect_preflight(config: BenchmarkConfig) -> dict[str, Any]:
    """Collect machine/dependency facts without downloading models or data."""

    config.validate()
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail})

    add("python", sys.version_info >= (3, 10), platform.python_version() + " (requires >=3.10)")

    versions: dict[str, str | None] = {}
    has_candidates = any(item != "baseline" for item in config.resolved_variants)
    needs_quantization = config.quantize_if_missing and any(
        not config.checkpoint_path(item).is_file()
        for item in config.resolved_variants
        if item != "baseline"
    )
    required_packages = ("numpy", "torch", "transformers", "datasets", "accelerate", "safetensors")
    if has_candidates:
        required_packages += ("tqdm", "loguru")
    if needs_quantization:
        required_packages += ("cut-cross-entropy",)
    for package in required_packages:
        try:
            versions[package] = importlib.metadata.version(package)
            add(f"package:{package}", True, versions[package] or "installed")
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            add(f"package:{package}", False, "not installed")

    transformers_version = versions.get("transformers")
    if transformers_version:
        add(
            "qwen3-transformers-version",
            _version_tuple(transformers_version) >= (4, 51),
            f"{transformers_version} (Qwen3 requires >=4.51)",
        )

    torch_info: dict[str, Any] = {
        "version": versions.get("torch"),
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
    }
    if versions.get("torch"):
        try:
            torch = importlib.import_module("torch")
            torch_info["cuda_available"] = bool(torch.cuda.is_available())
            torch_info["cuda_version"] = getattr(torch.version, "cuda", None)
            torch_info["device_count"] = int(torch.cuda.device_count()) if torch_info["cuda_available"] else 0
            for index in range(torch_info["device_count"]):
                props = torch.cuda.get_device_properties(index)
                torch_info["devices"].append({
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                })
        except Exception as exc:  # pragma: no cover - depends on native install
            add("torch-import", False, f"{type(exc).__name__}: {exc}")

    wants_cuda = config.device.startswith("cuda")
    add(
        "cuda",
        (not wants_cuda) or torch_info["cuda_available"],
        f"requested={config.device}; available={torch_info['cuda_available']}; runtime={torch_info['cuda_version']}",
    )
    if wants_cuda and torch_info["devices"]:
        try:
            index = int(config.device.split(":", 1)[1]) if ":" in config.device else 0
        except ValueError:
            index = -1
        valid_index = 0 <= index < len(torch_info["devices"])
        add("cuda-device-index", valid_index, f"requested index={index}; count={len(torch_info['devices'])}")
        if valid_index and config.profile.recommended_gpu_gib:
            actual = torch_info["devices"][index]["total_memory_bytes"] / (1024**3)
            if actual < config.profile.recommended_gpu_gib:
                warnings.append(
                    f"GPU has {actual:.1f} GiB; the {config.profile.display_name} profile recommends "
                    f"{config.profile.recommended_gpu_gib:.1f} GiB for this full pipeline."
                )

    if has_candidates and config.backend in {"gemv", "gemm"}:
        try:
            importlib.import_module("binary_kernels")
            add("binary-kernels", True, "CUDA extension import succeeded")
        except Exception as exc:
            add("binary-kernels", False, f"{type(exc).__name__}: {exc}")
    elif has_candidates and config.backend == "gemlite":
        try:
            importlib.import_module("gemlite")
            add("gemlite", True, "GemLite import succeeded")
        except Exception as exc:
            add("gemlite", False, f"{type(exc).__name__}: {exc}")

    output_parent = _existing_parent(Path(config.output_dir))
    checkpoint_parent = _existing_parent(Path(config.checkpoint_dir))
    output_disk = shutil.disk_usage(output_parent)
    checkpoint_disk = shutil.disk_usage(checkpoint_parent)
    disk_free = min(output_disk.free, checkpoint_disk.free)
    add("writable-output-parent", os.access(output_parent, os.W_OK), str(output_parent))
    add("writable-checkpoint-parent", os.access(checkpoint_parent, os.W_OK), str(checkpoint_parent))
    if config.profile.recommended_free_disk_gib:
        required = int(config.profile.recommended_free_disk_gib * (1024**3))
        if disk_free < required:
            warnings.append(
                f"Only {disk_free / (1024**3):.1f} GiB is free; the profile recommends "
                f"{config.profile.recommended_free_disk_gib:.1f} GiB."
            )

    host_memory = _host_memory_bytes()
    if host_memory and config.profile.recommended_host_gib and needs_quantization:
        actual = host_memory / (1024**3)
        if actual < config.profile.recommended_host_gib:
            warnings.append(
                f"Host has {actual:.1f} GiB RAM; the quantizer may need about "
                f"{config.profile.recommended_host_gib:.1f} GiB for {config.profile.display_name}."
            )

    checkpoints: list[dict[str, Any]] = []
    missing_for_quantization = False
    for variant in config.resolved_variants:
        if variant == "baseline":
            continue
        path = config.checkpoint_path(variant)
        present = path.is_file()
        missing_for_quantization = missing_for_quantization or not present
        checkpoints.append({"variant": variant, "path": str(path), "present": present})
        add(
            f"checkpoint:{variant}",
            present or config.quantize_if_missing,
            "present" if present else (
                "missing; will be created" if config.quantize_if_missing else "missing; use --quantize-if-missing"
            ),
        )
        if present:
            metadata_path = path.with_suffix(path.suffix + ".metadata.json")
            if not metadata_path.is_file():
                add(
                    f"checkpoint-provenance:{variant}",
                    config.allow_unverified_checkpoint,
                    "sidecar missing; explicitly allowed" if config.allow_unverified_checkpoint else
                    f"missing {metadata_path}",
                )
            else:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    actual = metadata.get("quantization_fingerprint") if isinstance(metadata, dict) else None
                    expected = config.quantization_fingerprint(variant)
                    add(
                        f"checkpoint-provenance:{variant}",
                        actual == expected,
                        f"fingerprint={actual}; expected={expected}",
                    )
                except (OSError, ValueError, TypeError) as exc:
                    add(f"checkpoint-provenance:{variant}", False, f"invalid sidecar: {exc}")

    if missing_for_quantization and config.quantize_if_missing:
        add(
            "quantization-device",
            wants_cuda and torch_info["cuda_available"],
            "on-the-fly quantization requires CUDA in the current NanoQuant pipeline",
        )

    required_failures = [item for item in checks if item["required"] and not item["ok"]]
    return {
        "ok": not required_failures,
        "model": config.profile.to_dict(),
        "device": config.device,
        "backend": config.backend,
        "checks": checks,
        "warnings": warnings,
        "versions": versions,
        "torch": torch_info,
        "host_memory_bytes": host_memory,
        "free_disk_bytes": disk_free,
        "checkpoints": checkpoints,
    }


def format_preflight(report: dict[str, Any]) -> str:
    lines = [f"Preflight: {'PASS' if report['ok'] else 'FAIL'}"]
    for item in report["checks"]:
        mark = "OK" if item["ok"] else "FAIL"
        lines.append(f"  [{mark:4}] {item['name']}: {item['detail']}")
    for warning in report.get("warnings", []):
        lines.append(f"  [WARN] {warning}")
    return "\n".join(lines)
