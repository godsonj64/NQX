# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Godson Johnson for NanoQuant-X, 2026.

"""NanoQuant core algorithms with lazy optional-PyTorch imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "factorize_admm_dbf": (".admm_dbf", "factorize_admm_dbf"),
    "factorize_admm_nanoquant": (".admm_nq", "factorize_admm_nanoquant"),
    "factorize_admm_nqx": (".admm_nqx", "factorize_admm_nqx"),
    "factorize_and_replace": (".compress_block", "factorize_and_replace"),
    "tune_fact": (".compress_block", "tune_fact"),
    "tune_nonfact": (".compress_block", "tune_nonfact"),
    "compress_block_recon": (".compress_model", "compress_block_recon"),
    "compress_model_recon": (".compress_model", "compress_model_recon"),
    "collect_stats": (".importance", "collect_stats"),
    "get_shrunk_stats": (".importance", "get_shrunk_stats"),
    "register_stats": (".importance", "register_stats"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
