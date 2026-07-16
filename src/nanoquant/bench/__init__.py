"""Real-model benchmarking for NanoQuant-X.

The public configuration and registry modules intentionally depend only on the
Python standard library.  Heavy ML dependencies are imported only when a real
run starts, so model selection, validation, preflight, and dry-run work in a
minimal installation.
"""

from .config import BenchmarkConfig
from .models import ModelProfile, list_model_profiles, resolve_model

__all__ = [
    "BenchmarkConfig",
    "ModelProfile",
    "list_model_profiles",
    "resolve_model",
]

