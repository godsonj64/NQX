"""Safe, deterministic ``.nqx`` artifact format.

The format is a ZIP container with a JSON manifest and non-pickle NumPy tensor
payloads.  Every payload is SHA-256 verified before decoding.

Copyright 2026 Godson Johnson
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .core import QuantizationDiagnostics, QuantizedMatrix
from .packing import pack_signs, unpack_signs


FORMAT_NAME = "nanoquant-x"
FORMAT_VERSION = 1
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_TENSOR_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_PAYLOAD_BYTES = 8 * 1024 * 1024 * 1024


def _canonical_zip_info(path: str) -> zipfile.ZipInfo:
    """Return deterministic, non-executable ZIP metadata."""
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_record(path: str, array: np.ndarray) -> tuple[dict[str, Any], bytes]:
    payload = _npy_bytes(array)
    return (
        {
            "path": path,
            "dtype": np.asarray(array).dtype.str,
            "shape": list(np.asarray(array).shape),
            "sha256": _sha256(payload),
            "size": len(payload),
        },
        payload,
    )


def save_nqx(matrix: QuantizedMatrix, path: str | os.PathLike[str]) -> Path:
    """Atomically save a quantized matrix."""
    destination = Path(path)
    if destination.suffix.lower() != ".nqx":
        destination = destination.with_suffix(destination.suffix + ".nqx" if destination.suffix else ".nqx")
    destination.parent.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    source_arrays = {
        "u_packed": pack_signs(matrix.u),
        "v_packed": pack_signs(matrix.v),
        "scale_out": matrix.scale_out.astype(np.float16),
        "scale_in": matrix.scale_in.astype(np.float16),
    }
    if matrix.rank_scale is not None:
        source_arrays["rank_scale"] = matrix.rank_scale.astype(np.float16)
    for name, array in source_arrays.items():
        record, payload = _tensor_record(f"tensors/{name}.npy", array)
        tensors[name] = record
        payloads[record["path"]] = payload

    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "word_bits": 32,
        "factor_shapes": {"u": list(matrix.u.shape), "v": list(matrix.v.shape)},
        "tensors": tensors,
        "config": matrix.config,
        "diagnostics": matrix.diagnostics.to_dict() if matrix.diagnostics is not None else None,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr(_canonical_zip_info("manifest.json"), manifest_bytes)
            for name in sorted(payloads):
                archive.writestr(_canonical_zip_info(name), payloads[name])
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _read_payload(archive: zipfile.ZipFile, record: dict[str, Any]) -> np.ndarray:
    if not isinstance(record, dict):
        raise ValueError("Invalid tensor record")
    required_fields = {"path", "dtype", "shape", "sha256", "size"}
    if set(record) != required_fields:
        raise ValueError("Tensor records must contain exactly path, dtype, shape, sha256, and size")
    path = str(record["path"])
    if path.startswith("/") or ".." in Path(path).parts:
        raise ValueError(f"Unsafe tensor path in manifest: {path}")
    info = archive.getinfo(path)
    expected_size = int(record["size"])
    if expected_size < 0 or expected_size > MAX_TENSOR_BYTES:
        raise ValueError(f"Tensor payload size is outside the safety limit for {path}")
    if info.flag_bits & 0x1:
        raise ValueError(f"Encrypted tensor entries are not supported: {path}")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"Compressed tensor entries are not supported: {path}")
    if info.file_size != expected_size:
        raise ValueError(f"Size mismatch for {path}")
    payload = archive.read(path)
    if _sha256(payload) != record["sha256"]:
        raise ValueError(f"Checksum mismatch for {path}")
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if array.dtype.hasobject or array.nbytes > MAX_TENSOR_BYTES:
        raise ValueError(f"Unsafe tensor dtype or size for {path}")
    if list(array.shape) != list(record["shape"]):
        raise ValueError(f"Shape mismatch for {path}")
    if array.dtype.str != record["dtype"]:
        raise ValueError(f"Dtype mismatch for {path}")
    return array


def load_nqx(path: str | os.PathLike[str]) -> QuantizedMatrix:
    """Load and verify a ``.nqx`` artifact without executing pickle code."""
    source = Path(path)
    with zipfile.ZipFile(source, "r") as archive:
        listed_names = archive.namelist()
        names = set(listed_names)
        if len(names) != len(listed_names):
            raise ValueError("The artifact contains duplicate ZIP entries")
        if "manifest.json" not in names:
            raise ValueError("The artifact does not contain manifest.json")
        manifest_info = archive.getinfo("manifest.json")
        if manifest_info.flag_bits & 0x1 or manifest_info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("manifest.json must be an unencrypted stored entry")
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest is unexpectedly large")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != FORMAT_NAME or manifest.get("version") != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported artifact format/version: {manifest.get('format')} {manifest.get('version')}"
            )
        tensors = manifest.get("tensors")
        if not isinstance(tensors, dict):
            raise ValueError("Invalid tensor table")
        required = {"u_packed", "v_packed", "scale_out", "scale_in"}
        if not required.issubset(tensors):
            raise ValueError(f"Missing required tensors: {sorted(required - set(tensors))}")
        allowed_tensors = required | {"rank_scale"}
        if not set(tensors).issubset(allowed_tensors):
            raise ValueError(f"Unknown tensors: {sorted(set(tensors) - allowed_tensors)}")
        tensor_paths = [str(record.get("path", "")) for record in tensors.values() if isinstance(record, dict)]
        if len(tensor_paths) != len(tensors) or len(set(tensor_paths)) != len(tensor_paths):
            raise ValueError("Tensor paths must be present and unique")
        expected_names = {"manifest.json", *tensor_paths}
        if names != expected_names:
            raise ValueError(f"Unexpected or missing ZIP members: {sorted(names ^ expected_names)}")
        declared_total = sum(int(record.get("size", -1)) for record in tensors.values())
        if declared_total < 0 or declared_total > MAX_TOTAL_PAYLOAD_BYTES:
            raise ValueError("The artifact exceeds the aggregate payload safety limit")
        arrays = {name: _read_payload(archive, record) for name, record in tensors.items()}

    factor_shapes = manifest["factor_shapes"]
    if not isinstance(factor_shapes, dict) or set(factor_shapes) != {"u", "v"}:
        raise ValueError("Invalid factor shape table")
    u_shape = tuple(int(value) for value in factor_shapes["u"])
    v_shape = tuple(int(value) for value in factor_shapes["v"])
    if len(u_shape) != 2 or len(v_shape) != 2 or min(*u_shape, *v_shape) <= 0:
        raise ValueError("Factor shapes must describe positive two-dimensional matrices")
    if u_shape[1] != v_shape[1]:
        raise ValueError("Factor ranks do not match")
    expected_dtypes = {
        "u_packed": np.dtype("<u4"),
        "v_packed": np.dtype("<u4"),
        "scale_out": np.dtype("<f2"),
        "scale_in": np.dtype("<f2"),
        "rank_scale": np.dtype("<f2"),
    }
    for name, array in arrays.items():
        if array.dtype != expected_dtypes[name]:
            raise ValueError(f"Unexpected dtype for {name}: {array.dtype}")
    u = unpack_signs(arrays["u_packed"], u_shape, word_bits=int(manifest["word_bits"]))
    v = unpack_signs(arrays["v_packed"], v_shape, word_bits=int(manifest["word_bits"]))
    diagnostic_data = manifest.get("diagnostics")
    diagnostics = QuantizationDiagnostics(**diagnostic_data) if diagnostic_data is not None else None
    return QuantizedMatrix(
        u=u,
        v=v,
        scale_out=arrays["scale_out"].astype(np.float32),
        scale_in=arrays["scale_in"].astype(np.float32),
        rank_scale=arrays.get("rank_scale", None),
        diagnostics=diagnostics,
        config=manifest.get("config"),
    )


def inspect_nqx(path: str | os.PathLike[str]) -> dict[str, Any]:
    matrix = load_nqx(path)
    diagnostics = matrix.diagnostics.to_dict() if matrix.diagnostics is not None else None
    if diagnostics is not None:
        history = diagnostics.pop("history", [])
        diagnostics["history_length"] = len(history)
        diagnostics["last_history_entry"] = history[-1] if history else None
    return {
        "out_features": matrix.out_features,
        "in_features": matrix.in_features,
        "rank": matrix.rank,
        "rank_scale": matrix.rank_scale is not None,
        "storage_bits": matrix.storage_bits(),
        "effective_bpw": matrix.effective_bpw(),
        "diagnostics": diagnostics,
        "config": matrix.config,
    }
