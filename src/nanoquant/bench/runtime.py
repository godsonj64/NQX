"""Heavy real-model benchmark runtime.

This module is imported only after configuration validation and preflight. It
performs real Hugging Face model loads, optional NanoQuant quantization, fixed-
token quality evaluation, fidelity comparison, and timed inference.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BenchmarkConfig, SCHEMA_VERSION
from .metrics import compare_results, timing_summary
from .storage import atomic_write_json, path_size_bytes, read_json, sha256_file


Progress = Callable[[str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _tensor_bytes(model: torch.nn.Module) -> dict[str, int]:
    """Count unique registered parameter and buffer storage without repacking."""

    seen: set[tuple[str, int]] = set()

    def count(tensors: Any) -> int:
        total = 0
        for tensor in tensors:
            if getattr(tensor, "is_meta", False):
                continue
            try:
                identity = (str(tensor.device), int(tensor.untyped_storage().data_ptr()))
                bytes_used = int(tensor.untyped_storage().nbytes())
            except (AttributeError, RuntimeError):
                identity = (str(tensor.device), id(tensor))
                bytes_used = int(tensor.numel() * tensor.element_size())
            if identity not in seen:
                seen.add(identity)
                total += bytes_used
        return total

    parameter_bytes = count(model.parameters())
    buffer_bytes = count(model.buffers())
    return {
        "resident_parameter_bytes": parameter_bytes,
        "resident_buffer_bytes": buffer_bytes,
        "resident_registered_tensor_bytes": parameter_bytes + buffer_bytes,
        "parameter_elements": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "transformers", "datasets", "accelerate", "safetensors", "nanoquant-x"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda: dict[str, Any] = {
        "runtime": torch.version.cuda,
        "available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        cuda.update({
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        })
    return {
        "timestamp_utc": _utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "tf32_matmul_precision": "high" if device.type == "cuda" else None,
    }


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


class RealModelRuntime:
    def __init__(self, config: BenchmarkConfig, progress: Progress = print):
        self.config = config.validate()
        self.profile = self.config.profile
        self.progress = progress
        self.device = torch.device(self.config.device)
        self.dtype = _torch_dtype(self.config.dtype)
        self.run_dir = self.config.run_directory
        self.run_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            torch.set_float32_matmul_precision("high")

    def _hf_common_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "revision": self.config.revision,
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }
        return {key: value for key, value in kwargs.items() if value is not None}

    def load_tokenizer(self):
        self.progress(f"Loading tokenizer: {self.profile.model_id}")
        kwargs = self._hf_common_kwargs()
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.profile.model_id, use_fast=True, **kwargs)
        except (OSError, TypeError, ValueError):
            tokenizer = AutoTokenizer.from_pretrained(self.profile.model_id, use_fast=False, **kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        return tokenizer

    def load_evaluation_tokens(self, tokenizer) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.config.dataset != "wikitext2":
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")
        self.progress(f"Loading WikiText-2 split={self.config.dataset_split}")
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split=self.config.dataset_split,
        )
        text = "\n\n".join(dataset["text"])
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, verbose=False)
        ids = encoded.input_ids[0].to(dtype=torch.long, device="cpu")
        bos_prepended = False
        if tokenizer.bos_token_id is not None and (ids.numel() == 0 or ids[0].item() != tokenizer.bos_token_id):
            ids = torch.cat([torch.tensor([tokenizer.bos_token_id], dtype=torch.long), ids])
            bos_prepended = True
        required = max(
            2,
            self.config.fidelity_length,
            max(self.config.prefill_lengths),
        )
        if ids.numel() < required:
            raise RuntimeError(f"Tokenized dataset has {ids.numel()} tokens but the benchmark needs {required}")
        array = ids.numpy()
        metadata = {
            "name": "Salesforce/wikitext",
            "configuration": "wikitext-2-raw-v1",
            "split": self.config.dataset_split,
            "available_tokens": int(ids.numel()),
            "token_ids_sha256": _array_digest(array),
            "tokenizer": getattr(tokenizer, "name_or_path", self.profile.model_id),
            "resolved_tokenizer_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            "bos_prepended": bos_prepended,
        }
        return ids, metadata

    def _move_and_prepare(self, model: torch.nn.Module, variant: str) -> tuple[torch.nn.Module, float, int]:
        model.to(self.device)
        model.eval()
        if hasattr(model, "seqlen"):
            model.seqlen = self.config.sequence_length
        if hasattr(model, "config"):
            model.config.use_cache = False
        prepared_modules = 0
        prepare_seconds = 0.0
        if variant != "baseline" and self.config.backend != "torch":
            from nanoquant.modules.linear import NanoQuantLinear

            modules = [module for module in model.modules() if isinstance(module, NanoQuantLinear)]
            if not modules:
                raise RuntimeError("No NanoQuantLinear modules were found for kernel preparation")
            self.progress(f"Preparing {len(modules)} {self.config.backend} kernels")
            started = time.perf_counter()
            for module in modules:
                module.to(self.device)
                module._prepare_kernel(kernel_type=self.config.backend, dtype=self.dtype)
            _sync(self.device)
            prepare_seconds = time.perf_counter() - started
            prepared_modules = len(modules)
        return model, prepare_seconds, prepared_modules

    def load_baseline(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        _reset_cuda_peak(self.device)
        started = time.perf_counter()
        kwargs = self._hf_common_kwargs()
        kwargs.update({
            "torch_dtype": self.dtype,
            "device_map": {"": str(self.device)},
            "low_cpu_mem_usage": True,
            "attn_implementation": self.config.attn_implementation,
        })
        model = AutoModelForCausalLM.from_pretrained(self.profile.model_id, **kwargs)
        model, prepare_seconds, prepared_modules = self._move_and_prepare(model, "baseline")
        _sync(self.device)
        elapsed = time.perf_counter() - started
        return model, {
            "operation": "huggingface_load",
            "seconds": elapsed,
            "kernel_prepare_seconds": prepare_seconds,
            "prepared_modules": prepared_modules,
            "cuda": _cuda_memory(self.device),
        }

    def _checkpoint_metadata_path(self, checkpoint: Path) -> Path:
        return checkpoint.with_suffix(checkpoint.suffix + ".metadata.json")

    def _validate_checkpoint_provenance(self, variant: str, checkpoint: Path) -> dict[str, Any] | None:
        metadata_path = self._checkpoint_metadata_path(checkpoint)
        if not metadata_path.is_file():
            if self.config.allow_unverified_checkpoint:
                return None
            raise RuntimeError(
                f"Checkpoint {checkpoint} has no provenance sidecar. Refusing to guess its quantization settings; "
                "use --allow-unverified-checkpoint only if you created and verified it yourself."
            )
        metadata = read_json(metadata_path)
        expected = self.config.quantization_fingerprint(variant)
        actual = metadata.get("quantization_fingerprint") if isinstance(metadata, dict) else None
        if actual != expected:
            raise RuntimeError(
                f"Checkpoint configuration mismatch for {checkpoint}: expected {expected}, found {actual}. "
                "Use a separate checkpoint directory or remove/rename the stale checkpoint."
            )
        return metadata

    def load_candidate(self, variant: str) -> tuple[torch.nn.Module, dict[str, Any], Path]:
        from nanoquant.modules.hub import NanoQuantConfigDataclass, NanoQuantModel

        checkpoint = self.config.checkpoint_path(variant)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        exists = checkpoint.is_file()
        provenance = self._validate_checkpoint_provenance(variant, checkpoint) if exists else None
        if not exists and not self.config.quantize_if_missing:
            raise FileNotFoundError(
                f"Missing {variant} checkpoint: {checkpoint}. Re-run with --quantize-if-missing to create it."
            )

        quantization_mapping = self.config.quantization_config(variant)
        quantization_config = NanoQuantConfigDataclass.from_dict(quantization_mapping).validate()
        temporary: Path | None = None
        qmodel_path = checkpoint
        operation = "checkpoint_load"
        if not exists:
            temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.partial")
            temporary.unlink(missing_ok=True)
            qmodel_path = temporary
            operation = "quantize_and_load"

        _reset_cuda_peak(self.device)
        started = time.perf_counter()
        try:
            wrapper = NanoQuantModel.from_pretrained_quantize(
                model_id=self.profile.model_id,
                qmodel_path=str(qmodel_path),
                quant_config=quantization_config,
                dtype=self.dtype,
                device_map=str(self.device),
            )
            model = wrapper.model
            # Version 0.3 checkpoints may be wrapped twice by the legacy Hub
            # loader. Unwrap defensively so generation and module traversal see
            # the actual causal LM.
            while isinstance(model, NanoQuantModel):
                model = model.model
            if temporary is not None:
                if not temporary.is_file():
                    raise RuntimeError("Quantization completed without producing the requested checkpoint")
                os.replace(temporary, checkpoint)
                provenance = {
                    "schema_version": "nqx-checkpoint-provenance/v1",
                    "created_at": _utc_now(),
                    "model_id": self.profile.model_id,
                    "variant": variant,
                    "quantization_fingerprint": self.config.quantization_fingerprint(variant),
                    "quantization_config": quantization_mapping,
                    "resolved_base_model_commit": getattr(getattr(model, "config", None), "_commit_hash", None),
                }
                atomic_write_json(self._checkpoint_metadata_path(checkpoint), provenance)
            model, prepare_seconds, prepared_modules = self._move_and_prepare(model, variant)
            _sync(self.device)
            elapsed = time.perf_counter() - started
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

        return model, {
            "operation": operation,
            "seconds": elapsed,
            "kernel_prepare_seconds": prepare_seconds,
            "prepared_modules": prepared_modules,
            "checkpoint_was_created": not exists,
            "checkpoint_provenance_verified": provenance is not None,
            "cuda": _cuda_memory(self.device),
        }, checkpoint

    def cleanup_model(self, model: torch.nn.Module | None) -> None:
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def evaluate_perplexity(self, model: torch.nn.Module, token_ids: torch.Tensor) -> dict[str, Any]:
        """Sliding-window perplexity with exact valid-token accounting."""

        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = False
        total_tokens = min(int(token_ids.numel()), self.config.max_eval_tokens)
        if total_tokens < 2:
            raise RuntimeError("At least two evaluation tokens are required")
        total_nll = 0.0
        scored_tokens = 0
        windows = 0
        previous_end = 0
        for begin in range(0, total_tokens - 1, self.config.stride):
            end = min(begin + self.config.sequence_length, total_tokens)
            target_length = end - previous_end
            if target_length <= 0:
                break
            input_ids = token_ids[begin:end].unsqueeze(0).to(self.device, non_blocking=True)
            labels = input_ids.clone()
            if target_length < labels.shape[1]:
                labels[:, :-target_length] = -100
            attention_mask = torch.ones_like(input_ids)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            valid_tokens = int((labels[:, 1:] != -100).sum().item())
            if valid_tokens:
                loss = outputs.loss.detach().float()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite perplexity loss in window {windows}")
                total_nll += float(loss.item()) * valid_tokens
                scored_tokens += valid_tokens
            windows += 1
            previous_end = end
            del input_ids, labels, attention_mask, outputs
            if end == total_tokens:
                break
        if scored_tokens == 0:
            raise RuntimeError("Perplexity evaluation did not score any tokens")
        mean_nll = total_nll / scored_tokens
        finite = mean_nll < math.log(float.fromhex("0x1.fffffffffffffp+1023"))
        value = math.exp(mean_nll) if finite else None
        return {
            "dataset": "wikitext2",
            "value": value,
            "finite": finite,
            "mean_negative_log_likelihood": mean_nll,
            "total_negative_log_likelihood": total_nll,
            "scored_tokens": scored_tokens,
            "input_tokens": total_tokens,
            "windows": windows,
            "sequence_length": self.config.sequence_length,
            "stride": self.config.stride,
        }

    def _fidelity_inputs(self, token_ids: torch.Tensor) -> np.ndarray:
        length = self.config.fidelity_length
        available_starts = int(token_ids.numel()) - length + 1
        if available_starts <= 0:
            raise RuntimeError("Evaluation corpus is too short for fidelity sequences")
        sample_count = min(self.config.fidelity_samples, available_starts)
        rng = np.random.default_rng(self.config.seed + 17_071)
        starts = np.sort(rng.choice(available_starts, size=sample_count, replace=False))
        return np.stack([
            token_ids[int(start):int(start) + length].numpy().astype(np.int32, copy=False)
            for start in starts
        ])

    @torch.inference_mode()
    def build_teacher_cache(self, model: torch.nn.Module, token_ids: torch.Tensor) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Cache compact baseline probabilities at deterministic token positions."""

        inputs = self._fidelity_inputs(token_ids)
        indices_parts: list[np.ndarray] = []
        probabilities_parts: list[np.ndarray] = []
        tail_parts: list[np.ndarray] = []
        top1_parts: list[np.ndarray] = []
        actual_topk: int | None = None
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = False
        for sample in inputs:
            current = torch.from_numpy(sample.astype(np.int64, copy=False)).unsqueeze(0).to(self.device)
            outputs = model(input_ids=current, attention_mask=torch.ones_like(current), use_cache=False)
            logits = outputs.logits[:, :-1, :]
            actual_topk = min(self.config.teacher_topk, int(logits.shape[-1]))
            values, indices = torch.topk(logits, k=actual_topk, dim=-1, sorted=True)
            log_partition = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
            probabilities = torch.exp(values.float() - log_partition)
            tail = (1.0 - probabilities.sum(dim=-1)).clamp(min=0.0, max=1.0)
            indices_parts.append(indices[0].to(torch.int32).cpu().numpy())
            probabilities_parts.append(probabilities[0].to(torch.float16).cpu().numpy())
            tail_parts.append(tail[0].to(torch.float32).cpu().numpy())
            top1_parts.append(indices[0, :, 0].to(torch.int32).cpu().numpy())
            del current, outputs, logits, values, indices, log_partition, probabilities, tail
        cache = {
            "input_ids": inputs.astype(np.int32, copy=False),
            "topk_indices": np.stack(indices_parts).astype(np.int32, copy=False),
            "topk_probabilities": np.stack(probabilities_parts).astype(np.float16, copy=False),
            "tail_probabilities": np.stack(tail_parts).astype(np.float32, copy=False),
            "teacher_top1": np.stack(top1_parts).astype(np.int32, copy=False),
        }
        metadata = {
            "schema_version": "nqx-teacher-cache/v1",
            "created_at": _utc_now(),
            "experiment_fingerprint": self.config.fingerprint,
            "model_id": self.profile.model_id,
            "resolved_base_model_commit": getattr(getattr(model, "config", None), "_commit_hash", None),
            "samples": int(inputs.shape[0]),
            "sequence_length": int(inputs.shape[1]),
            "scored_tokens": int(inputs.shape[0] * (inputs.shape[1] - 1)),
            "topk": int(actual_topk or 0),
            "input_ids_sha256": _array_digest(cache["input_ids"]),
            "mean_teacher_tail_probability": float(cache["tail_probabilities"].mean()),
            "storage": {
                key: {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": int(value.nbytes)}
                for key, value in cache.items()
            },
        }
        return cache, metadata

    def save_teacher_cache(self, cache: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        cache_path = self.run_dir / "teacher_cache.npz"
        metadata_path = self.run_dir / "teacher_cache.json"
        _atomic_save_npz(cache_path, **cache)
        value = dict(metadata)
        value["archive_sha256"] = sha256_file(cache_path)
        value["archive_bytes"] = cache_path.stat().st_size
        atomic_write_json(metadata_path, value)

    def load_teacher_cache(self) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
        cache_path = self.run_dir / "teacher_cache.npz"
        metadata_path = self.run_dir / "teacher_cache.json"
        if not cache_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = read_json(metadata_path)
            if metadata.get("schema_version") != "nqx-teacher-cache/v1":
                return None
            if metadata.get("experiment_fingerprint") != self.config.fingerprint:
                return None
            if metadata.get("archive_sha256") != sha256_file(cache_path):
                return None
            with np.load(cache_path, allow_pickle=False) as archive:
                expected = {"input_ids", "topk_indices", "topk_probabilities", "tail_probabilities", "teacher_top1"}
                if set(archive.files) != expected:
                    return None
                cache = {key: archive[key].copy() for key in expected}
            inputs = cache["input_ids"]
            indices = cache["topk_indices"]
            probabilities = cache["topk_probabilities"]
            tails = cache["tail_probabilities"]
            top1 = cache["teacher_top1"]
            if inputs.ndim != 2 or indices.ndim != 3 or probabilities.shape != indices.shape:
                return None
            expected_tokens = (inputs.shape[0], inputs.shape[1] - 1)
            if tails.shape != expected_tokens or top1.shape != expected_tokens or indices.shape[:2] != expected_tokens:
                return None
            if _array_digest(inputs) != metadata.get("input_ids_sha256"):
                return None
            if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(tails)):
                return None
            return cache, metadata
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @torch.inference_mode()
    def evaluate_fidelity(self, model: torch.nn.Module, cache: dict[str, np.ndarray]) -> dict[str, Any]:
        """Compute top-k-plus-tail KL and exact top-1 agreement to baseline."""

        total_kl = 0.0
        total_cross_entropy = 0.0
        total_teacher_entropy = 0.0
        total_tokens = 0
        top1_matches = 0
        sample_kls: list[float] = []
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = False
        for sample_index, input_array in enumerate(cache["input_ids"]):
            current = torch.from_numpy(input_array.astype(np.int64, copy=False)).unsqueeze(0).to(self.device)
            outputs = model(input_ids=current, attention_mask=torch.ones_like(current), use_cache=False)
            logits = outputs.logits[:, :-1, :]
            teacher_indices = torch.from_numpy(cache["topk_indices"][sample_index].astype(np.int64, copy=False)).to(
                self.device
            )
            teacher_probabilities = torch.from_numpy(
                cache["topk_probabilities"][sample_index].astype(np.float32)
            ).to(self.device)
            teacher_tail = torch.from_numpy(cache["tail_probabilities"][sample_index]).to(
                self.device, dtype=torch.float32
            )
            normalization = (teacher_probabilities.sum(dim=-1) + teacher_tail).clamp_min(1e-12)
            teacher_probabilities = teacher_probabilities / normalization.unsqueeze(-1)
            teacher_tail = teacher_tail / normalization

            selected_logits = torch.gather(logits[0], dim=-1, index=teacher_indices).float()
            log_partition = torch.logsumexp(logits[0].float(), dim=-1)
            selected_log_probabilities = selected_logits - log_partition.unsqueeze(-1)
            selected_mass = torch.exp(selected_log_probabilities).sum(dim=-1).clamp(max=1.0 - 1e-7)
            candidate_tail_log_probability = torch.log1p(-selected_mass).clamp_min(math.log(1e-12))

            teacher_log_probabilities = torch.log(teacher_probabilities.clamp_min(1e-12))
            teacher_tail_log_probability = torch.log(teacher_tail.clamp_min(1e-12))
            cross_entropy = -(
                (teacher_probabilities * selected_log_probabilities).sum(dim=-1)
                + teacher_tail * candidate_tail_log_probability
            )
            teacher_entropy = -(
                (teacher_probabilities * teacher_log_probabilities).sum(dim=-1)
                + teacher_tail * teacher_tail_log_probability
            )
            kl = (cross_entropy - teacher_entropy).clamp_min(0.0)
            candidate_top1 = logits[0].argmax(dim=-1)
            teacher_top1 = torch.from_numpy(cache["teacher_top1"][sample_index].astype(np.int64, copy=False)).to(
                self.device
            )

            count = int(kl.numel())
            sample_kls.append(float(kl.mean().item()))
            total_kl += float(kl.sum().item())
            total_cross_entropy += float(cross_entropy.sum().item())
            total_teacher_entropy += float(teacher_entropy.sum().item())
            top1_matches += int((candidate_top1 == teacher_top1).sum().item())
            total_tokens += count
            del current, outputs, logits, teacher_indices, teacher_probabilities, teacher_tail
            del selected_logits, log_partition, selected_log_probabilities, selected_mass
            del candidate_tail_log_probability, teacher_log_probabilities, teacher_tail_log_probability
            del cross_entropy, teacher_entropy, kl, candidate_top1, teacher_top1
        if total_tokens == 0:
            raise RuntimeError("Fidelity evaluation produced no tokens")
        return {
            "method": "teacher-topk-plus-tail-bucket",
            "topk": int(cache["topk_indices"].shape[-1]),
            "tokens": total_tokens,
            "topk_tail_kl_nats": total_kl / total_tokens,
            "bucket_cross_entropy_nats": total_cross_entropy / total_tokens,
            "teacher_bucket_entropy_nats": total_teacher_entropy / total_tokens,
            "top1_agreement": top1_matches / total_tokens,
            "per_sample_kl_nats": sample_kls,
            "interpretation": "KL is evaluated after grouping all non-top-k vocabulary items into one tail category.",
        }

    def _time_call(self, function: Callable[[], Any]) -> tuple[float, Any]:
        _sync(self.device)
        started = time.perf_counter()
        value = function()
        _sync(self.device)
        return time.perf_counter() - started, value

    @torch.inference_mode()
    def benchmark_prefill(self, model: torch.nn.Module, token_ids: torch.Tensor) -> list[dict[str, Any]]:
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = False
        rows: list[dict[str, Any]] = []
        for length in self.config.prefill_lengths:
            current = token_ids[:length].unsqueeze(0).to(self.device, non_blocking=True)
            attention_mask = torch.ones_like(current)

            def forward():
                return model(input_ids=current, attention_mask=attention_mask, use_cache=False)

            for _ in range(self.config.warmup_runs):
                output = forward()
                _sync(self.device)
                del output
            timings: list[float] = []
            for _ in range(self.config.repeat_runs):
                elapsed, output = self._time_call(forward)
                timings.append(elapsed)
                del output
            summary = timing_summary(timings, work_per_run=length)
            rows.append({
                "length": length,
                "batch_size": 1,
                "tokens_per_second": summary["units_per_second"],
                "timing": summary,
            })
            del current, attention_mask
        return rows

    def _generation_kwargs(self, tokenizer, fixed_length: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": self.config.generation_max_new_tokens,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if tokenizer.eos_token_id is not None:
            kwargs["eos_token_id"] = tokenizer.eos_token_id
        if fixed_length:
            kwargs["min_new_tokens"] = self.config.generation_max_new_tokens
        return kwargs

    @torch.inference_mode()
    def benchmark_generation(self, model: torch.nn.Module, tokenizer) -> dict[str, Any]:
        """End-to-end greedy generation timing; the reported time includes prefill."""

        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True
        inputs = tokenizer(self.config.prompts[0], return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", torch.ones_like(input_ids)).to(self.device)
        kwargs = self._generation_kwargs(tokenizer, fixed_length=True)

        def generate():
            return model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        for _ in range(self.config.warmup_runs):
            output = generate()
            _sync(self.device)
            del output
        timings: list[float] = []
        generated_counts: list[int] = []
        for _ in range(self.config.repeat_runs):
            elapsed, output = self._time_call(generate)
            timings.append(elapsed)
            generated_counts.append(int(output.shape[1] - input_ids.shape[1]))
            del output
        summary = timing_summary(timings)
        total_generated = sum(generated_counts)
        return {
            "definition": "end-to-end greedy generation including prompt prefill",
            "prompt": self.config.prompts[0],
            "prompt_tokens": int(input_ids.shape[1]),
            "requested_new_tokens": self.config.generation_max_new_tokens,
            "generated_tokens_per_run": generated_counts,
            "tokens_per_second": total_generated / sum(timings),
            "timing": summary,
        }

    @torch.inference_mode()
    def generate_samples(self, model: torch.nn.Module, tokenizer) -> list[dict[str, Any]]:
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True
        kwargs = self._generation_kwargs(tokenizer, fixed_length=False)
        samples: list[dict[str, Any]] = []
        for prompt in self.config.prompts:
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded.input_ids.to(self.device)
            attention_mask = getattr(encoded, "attention_mask", torch.ones_like(input_ids)).to(self.device)
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            new_ids = output[0, input_ids.shape[1]:].to(torch.int64).cpu().tolist()
            samples.append({
                "prompt": prompt,
                "prompt_tokens": int(input_ids.shape[1]),
                "new_token_ids": new_ids,
                "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
            })
            del input_ids, attention_mask, output
        return samples

    def evaluate_variant(
        self,
        variant: str,
        model: torch.nn.Module,
        tokenizer,
        token_ids: torch.Tensor,
        dataset_metadata: dict[str, Any],
        load_info: dict[str, Any],
        teacher_cache: dict[str, np.ndarray] | None,
        teacher_metadata: dict[str, Any] | None,
        checkpoint: Path | None,
        started_at: str,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray] | None, dict[str, Any] | None]:
        self.progress(f"[{variant}] WikiText-2 perplexity")
        _reset_cuda_peak(self.device)
        quality: dict[str, Any] = {"perplexity": self.evaluate_perplexity(model, token_ids)}
        if variant == "baseline":
            self.progress("[baseline] Building compact teacher reference")
            teacher_cache, teacher_metadata = self.build_teacher_cache(model, token_ids)
            self.save_teacher_cache(teacher_cache, teacher_metadata)
            quality["fidelity_reference"] = {
                "method": "teacher-topk-plus-tail-bucket",
                "topk": teacher_metadata["topk"],
                "tokens": teacher_metadata["scored_tokens"],
                "mean_tail_probability": teacher_metadata["mean_teacher_tail_probability"],
            }
        else:
            if teacher_cache is None:
                raise RuntimeError("Compressed variants require a valid baseline teacher cache")
            teacher_commit = (teacher_metadata or {}).get("resolved_base_model_commit")
            candidate_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
            if teacher_commit and candidate_commit and teacher_commit != candidate_commit:
                raise RuntimeError(
                    f"Base-model revision mismatch: teacher={teacher_commit}, candidate={candidate_commit}. "
                    "Pin --revision and rebuild the checkpoint."
                )
            self.progress(f"[{variant}] Baseline fidelity")
            quality["fidelity"] = self.evaluate_fidelity(model, teacher_cache)

        self.progress(f"[{variant}] Prefill throughput")
        prefill = self.benchmark_prefill(model, token_ids)
        self.progress(f"[{variant}] Generation throughput and samples")
        generation_throughput = self.benchmark_generation(model, tokenizer)
        generation_samples = self.generate_samples(model, tokenizer)
        inference_cuda = _cuda_memory(self.device)
        model_metrics = _tensor_bytes(model)
        model_metrics.update({
            "dtype": self.config.dtype,
            "backend": "torch" if variant == "baseline" else self.config.backend,
            "device": str(self.device),
            "requested_huggingface_revision": self.config.revision,
            "resolved_huggingface_commit": getattr(getattr(model, "config", None), "_commit_hash", None),
        })

        if checkpoint is None:
            checkpoint_metrics = {
                "kind": "huggingface-baseline",
                "path": None,
                "bytes": None,
                "sha256": None,
                "note": "Baseline storage is represented by resident parameter bytes; Hub cache size is not guessed.",
            }
        else:
            checkpoint_metrics = {
                "kind": "nanoquant-checkpoint",
                "path": str(checkpoint),
                "bytes": path_size_bytes(checkpoint),
                "sha256": sha256_file(checkpoint) if self.config.hash_checkpoint else None,
                "approximate_file_bits_per_original_parameter": (
                    path_size_bytes(checkpoint) * 8 / self.profile.parameters
                    if self.profile.parameters else None
                ),
                "note": (
                    "File-level BPW includes every tensor and metadata entry and uses the registry's rounded "
                    "official total-parameter count; use checkpoint bytes as the exact storage measurement."
                ),
            }

        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "experiment_fingerprint": self.config.fingerprint,
            "variant": variant,
            "model": self.profile.to_dict(),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "environment": _environment(self.device),
            "dataset": dataset_metadata,
            "configuration": self.config.to_dict(resolved=True),
            "quantization_configuration": None
            if variant == "baseline" else self.config.quantization_config(variant),
            "metrics": {
                "load": load_info,
                "model": model_metrics,
                "checkpoint": checkpoint_metrics,
                "quality": quality,
                "throughput": {
                    "prefill": prefill,
                    "generation": generation_throughput,
                },
                "generation": {"samples": generation_samples},
                "inference_cuda": inference_cuda,
            },
            "measurement_notes": [
                "Timing uses perf_counter with device synchronization before and after each measured call.",
                "Warmup runs are excluded; raw distribution summaries include mean, median, standard deviation, and p90.",
                "Generation is greedy and deterministic; generation tokens/second includes prompt prefill.",
                "The baseline and candidate use exactly the same cached token IDs and prompt strings.",
            ],
        }
        return result, teacher_cache, teacher_metadata

    def _result_path(self, variant: str) -> Path:
        return self.run_dir / f"result-{variant}.json"

    def _load_completed_result(self, variant: str) -> dict[str, Any] | None:
        path = self._result_path(variant)
        if not self.config.resume or not path.is_file():
            return None
        try:
            result = read_json(path)
        except (OSError, ValueError, TypeError):
            return None
        if (
            isinstance(result, dict)
            and result.get("status") == "complete"
            and result.get("experiment_fingerprint") == self.config.fingerprint
            and result.get("variant") == variant
        ):
            return result
        return None

    def _write_manifest(self, status: str, completed: list[str], active: str | None = None,
                        failure: dict[str, Any] | None = None) -> None:
        atomic_write_json(self.run_dir / "manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "experiment_fingerprint": self.config.fingerprint,
            "status": status,
            "model": self.profile.to_dict(),
            "variants": list(self.config.resolved_variants),
            "completed_variants": completed,
            "active_variant": active,
            "failure": failure,
            "updated_at": _utc_now(),
        })

    def run(self) -> dict[str, Any]:
        """Run all requested variants with resume and atomic per-stage output."""

        resolved_path = self.run_dir / "resolved_config.json"
        if resolved_path.is_file():
            existing = read_json(resolved_path)
            if existing.get("experiment_fingerprint") != self.config.fingerprint:
                raise RuntimeError(
                    f"Run directory {self.run_dir} belongs to a different experiment. Choose another --run-name."
                )
        atomic_write_json(resolved_path, {
            "schema_version": SCHEMA_VERSION,
            "experiment_fingerprint": self.config.fingerprint,
            "configuration": self.config.to_dict(resolved=True),
        })

        results: dict[str, dict[str, Any]] = {}
        for variant in self.config.resolved_variants:
            completed = self._load_completed_result(variant)
            if completed is not None:
                results[variant] = completed
                self.progress(f"[{variant}] Resuming complete result")

        pending_candidates = [
            variant for variant in self.config.resolved_variants
            if variant != "baseline" and variant not in results
        ]
        cached_teacher = self.load_teacher_cache()
        # A pending candidate cannot be compared safely if the baseline cache is
        # absent or corrupt, so force only the baseline phase to rerun.
        if pending_candidates and cached_teacher is None:
            results.pop("baseline", None)

        pending = [variant for variant in self.config.resolved_variants if variant not in results]
        if not pending:
            self.progress("All variants already complete; rebuilding summary only")
            return self._finalize(results)

        tokenizer = self.load_tokenizer()
        token_ids, dataset_metadata = self.load_evaluation_tokens(tokenizer)
        teacher_cache, teacher_metadata = cached_teacher if cached_teacher is not None else (None, None)
        self._write_manifest("running", list(results))

        for variant in self.config.resolved_variants:
            if variant in results:
                continue
            started_at = _utc_now()
            self._write_manifest("running", list(results), active=variant)
            model: torch.nn.Module | None = None
            checkpoint: Path | None = None
            try:
                self.progress(f"[{variant}] Loading model")
                if variant == "baseline":
                    model, load_info = self.load_baseline()
                else:
                    model, load_info, checkpoint = self.load_candidate(variant)
                result, teacher_cache, teacher_metadata = self.evaluate_variant(
                    variant=variant,
                    model=model,
                    tokenizer=tokenizer,
                    token_ids=token_ids,
                    dataset_metadata=dataset_metadata,
                    load_info=load_info,
                    teacher_cache=teacher_cache,
                    teacher_metadata=teacher_metadata,
                    checkpoint=checkpoint,
                    started_at=started_at,
                )
                atomic_write_json(self._result_path(variant), result)
                results[variant] = result
                self.progress(f"[{variant}] Complete: {self._result_path(variant)}")
            except BaseException as exc:
                failure = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "experiment_fingerprint": self.config.fingerprint,
                    "variant": variant,
                    "started_at": started_at,
                    "failed_at": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                atomic_write_json(self.run_dir / f"failure-{variant}.json", failure)
                self._write_manifest("failed", list(results), active=variant, failure={
                    "error_type": type(exc).__name__, "error": str(exc)
                })
                raise
            finally:
                self.cleanup_model(model)
                model = None
        return self._finalize(results)

    def _finalize(self, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if "baseline" not in results:
            raise RuntimeError("A benchmark summary requires a baseline result")
        comparisons = []
        for variant in self.config.resolved_variants:
            if variant == "baseline":
                continue
            comparison = compare_results(results["baseline"], results[variant])
            path = self.run_dir / f"comparison-{variant}.json"
            atomic_write_json(path, comparison)
            comparisons.append({"variant": variant, "path": str(path), "comparison": comparison})
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "experiment_fingerprint": self.config.fingerprint,
            "model": self.profile.to_dict(),
            "run_directory": str(self.run_dir),
            "results": {variant: str(self._result_path(variant)) for variant in self.config.resolved_variants},
            "comparisons": comparisons,
            "completed_at": _utc_now(),
        }
        atomic_write_json(self.run_dir / "summary.json", summary)
        self._write_manifest("complete", list(self.config.resolved_variants))
        return summary


def run_benchmarks(config: BenchmarkConfig, progress: Progress = print) -> dict[str, Any]:
    return RealModelRuntime(config, progress=progress).run()
