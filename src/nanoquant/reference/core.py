"""Portable NumPy reference implementation of the NanoQuant-X core.

This module intentionally has no PyTorch dependency.  It provides an auditable
implementation of the enhanced matrix quantizer, exact deployed-objective
metrics, and factorized inference.  The production PyTorch path mirrors the
same representation in :mod:`nanoquant.core.admm_nqx`.

Copyright 2026 Godson Johnson
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import ceil
from typing import Any, Iterable, Literal

import numpy as np


Array = np.ndarray
InitKind = Literal["spectral", "random"]


def _sign_no_zero(x: Array) -> Array:
    """Return a deterministic {-1,+1} sign array."""
    return np.where(x < 0, -1, 1).astype(np.int8, copy=False)


def _positive_vector(x: Array | None, length: int, eps: float) -> Array:
    if x is None:
        return np.ones(length, dtype=np.float64)
    out = np.asarray(x, dtype=np.float64).reshape(-1)
    if out.size != length:
        raise ValueError(f"Expected a statistic of length {length}, got {out.size}.")
    out = np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0)
    return np.maximum(out, eps)


def _relative_weighted_error(
    target: Array,
    approximation: Array,
    output_weight: Array,
    input_weight: Array,
    eps: float,
) -> float:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(approximation, dtype=np.float64)
    weights = output_weight[:, None] * input_weight[None, :]
    numerator = float(np.sum(weights * residual * residual, dtype=np.float64))
    denominator = float(np.sum(weights * target * target, dtype=np.float64))
    return numerator / max(denominator, eps)


def _geometric_mean_abs(x: Array, eps: float) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(np.abs(x), eps)))))


@dataclass(frozen=True)
class NQXConfig:
    """Configuration for one matrix factorization.

    ``rank_scale=False`` is the strict two-scale representation from the paper.
    Enabling it stores one additional FP16 value per rank and generally improves
    accuracy for a very small metadata cost.
    """

    rank: int
    max_iters: int = 400
    min_iters: int = 64
    tolerance: float = 1e-4
    patience: int = 10_000
    rho_init: float = 0.0
    rho_final: float = 1.0
    rho_min: float = 0.0
    rho_max: float = 64.0
    rho_balance: float = 10.0
    rho_multiplier: float = 2.0
    adaptive_rho: bool = False
    ridge: float = 3e-2
    scale_ridge: float = 1e-6
    projection_iters: int = 5
    init: InitKind = "random"
    spectral_oversample: int = 8
    spectral_power_iters: int = 1
    scale_iters: int = 8
    polish_iters: int = 1
    reclaim_packed_padding: bool = True
    candidate_selection: bool = True
    storage_aware: bool = True
    storage_refine_iters: int = 2
    rank_scale: bool = False
    precondition_shrinkage: float = 0.2
    precondition_clip: float = 8.0
    seed: int = 0
    eps: float = 1e-10

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.max_iters < 1:
            raise ValueError("max_iters must be at least one")
        if not 0 <= self.min_iters <= self.max_iters:
            raise ValueError("min_iters must lie in [0, max_iters]")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.patience < 1:
            raise ValueError("patience must be at least one")
        if not 0 <= self.rho_min <= self.rho_max:
            raise ValueError("rho bounds are invalid")
        if not self.rho_min <= self.rho_init <= self.rho_max:
            raise ValueError("rho_init must lie within the rho bounds")
        if not self.rho_min <= self.rho_final <= self.rho_max:
            raise ValueError("rho_final must lie within the rho bounds")
        if self.rho_multiplier <= 1:
            raise ValueError("rho_multiplier must exceed one")
        if self.ridge < 0 or self.scale_ridge < 0:
            raise ValueError("ridge values must be non-negative")
        if self.projection_iters < 1:
            raise ValueError("projection_iters must be at least one")
        if self.scale_iters < 0 or self.polish_iters < 0 or self.storage_refine_iters < 0:
            raise ValueError("scale_iters, polish_iters, and storage_refine_iters must be non-negative")
        if not 0 <= self.precondition_shrinkage <= 1:
            raise ValueError("precondition_shrinkage must lie in [0, 1]")
        if self.precondition_clip < 1:
            raise ValueError("precondition_clip must be at least one")
        if self.init not in ("spectral", "random"):
            raise ValueError(f"Unsupported initialization: {self.init}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuantizationDiagnostics:
    iterations: int
    stopped_early: bool
    initial_rho: float
    final_rho: float
    continuous_error: float
    deployed_error_before_refit: float
    deployed_error: float
    weighted_deployed_error: float
    primal_residual: float
    dual_residual: float
    serialized_deployed_error: float | None = None
    selected_candidate: str = "final"
    candidates_evaluated: int = 1
    history: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuantizedMatrix:
    """A two-stage binary matrix with channel and optional rank scales."""

    u: Array  # [out_features, rank], {-1,+1}
    v: Array  # [in_features, rank], {-1,+1}
    scale_out: Array  # [out_features]
    scale_in: Array  # [in_features]
    rank_scale: Array | None = None  # [rank]
    diagnostics: QuantizationDiagnostics | None = None
    config: dict[str, Any] | None = None
    _runtime_cache: dict[str, tuple[Array, Array]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=np.int8)
        self.v = np.asarray(self.v, dtype=np.int8)
        self.scale_out = np.asarray(self.scale_out, dtype=np.float32).reshape(-1)
        self.scale_in = np.asarray(self.scale_in, dtype=np.float32).reshape(-1)
        if self.u.ndim != 2 or self.v.ndim != 2:
            raise ValueError("u and v must be two-dimensional")
        if self.u.shape[1] != self.v.shape[1]:
            raise ValueError("u and v must share the same rank")
        if self.scale_out.size != self.u.shape[0]:
            raise ValueError("scale_out length does not match u")
        if self.scale_in.size != self.v.shape[0]:
            raise ValueError("scale_in length does not match v")
        if not np.all(np.isin(self.u, (-1, 1))) or not np.all(np.isin(self.v, (-1, 1))):
            raise ValueError("binary factors must contain only -1 and +1")
        if self.rank_scale is not None:
            self.rank_scale = np.asarray(self.rank_scale, dtype=np.float32).reshape(-1)
            if self.rank_scale.size != self.rank:
                raise ValueError("rank_scale length does not match the factor rank")

    @property
    def out_features(self) -> int:
        return int(self.u.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.v.shape[0])

    @property
    def rank(self) -> int:
        return int(self.u.shape[1])

    def reconstruct(self, dtype: np.dtype[Any] = np.float32) -> Array:
        u = self.u.astype(dtype, copy=False)
        v = self.v.astype(dtype, copy=False)
        if self.rank_scale is not None:
            u = u * self.rank_scale.astype(dtype, copy=False)[None, :]
        core = u @ v.T
        return (self.scale_out.astype(dtype)[:, None] * core) * self.scale_in.astype(dtype)[None, :]

    def prepare_runtime(self, dtype: np.dtype[Any] = np.float32) -> "QuantizedMatrix":
        """Fuse scales into reusable factors for repeated NumPy inference.

        This cache is optional and is never serialized.  It replaces repeated
        sign-to-float conversion and per-call scale broadcasts with two BLAS
        matrix multiplications.  Call :meth:`clear_runtime_cache` after
        manually mutating any public factor or scale array.
        """
        work_dtype = np.dtype(dtype)
        if work_dtype.kind != "f":
            raise TypeError("runtime dtype must be floating point")
        key = work_dtype.str
        if key not in self._runtime_cache:
            input_factor = np.ascontiguousarray(
                self.v.astype(work_dtype) * self.scale_in.astype(work_dtype)[:, None]
            )
            output_factor = self.u.astype(work_dtype) * self.scale_out.astype(work_dtype)[:, None]
            if self.rank_scale is not None:
                output_factor *= self.rank_scale.astype(work_dtype)[None, :]
            self._runtime_cache[key] = (
                input_factor,
                np.ascontiguousarray(output_factor),
            )
        return self

    def clear_runtime_cache(self) -> None:
        self._runtime_cache.clear()

    @property
    def runtime_cache_bytes(self) -> int:
        return int(sum(left.nbytes + right.nbytes for left, right in self._runtime_cache.values()))

    def matmul(self, x: Array, *, prepared: bool = True) -> Array:
        """Compute ``x @ W.T`` without materializing the dense matrix.

        ``prepared=True`` lazily caches scale-fused low-rank factors.  Set it
        to ``False`` for a zero-cache, one-shot calculation.
        """
        x_arr = np.asarray(x)
        if x_arr.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected the final input dimension to be {self.in_features}, got {x_arr.shape[-1]}."
            )
        work_dtype = np.result_type(x_arr.dtype, np.float32)
        flat = x_arr.astype(work_dtype, copy=False).reshape(-1, self.in_features)
        if prepared:
            self.prepare_runtime(work_dtype)
            input_factor, output_factor = self._runtime_cache[np.dtype(work_dtype).str]
            output = (flat @ input_factor) @ output_factor.T
        else:
            hidden = (flat * self.scale_in.astype(work_dtype)) @ self.v.astype(work_dtype)
            if self.rank_scale is not None:
                hidden *= self.rank_scale.astype(work_dtype)
            output = (hidden @ self.u.astype(work_dtype).T) * self.scale_out.astype(work_dtype)
        return output.reshape(*x_arr.shape[:-1], self.out_features)

    def storage_bits(self, scale_bits: int = 16, word_bits: int = 32) -> int:
        words_per_row = ceil(self.rank / word_bits)
        factor_bits = word_bits * words_per_row * (self.out_features + self.in_features)
        metadata_bits = scale_bits * (self.out_features + self.in_features)
        if self.rank_scale is not None:
            metadata_bits += scale_bits * self.rank
        return int(factor_bits + metadata_bits)

    def effective_bpw(self, scale_bits: int = 16, word_bits: int = 32) -> float:
        return self.storage_bits(scale_bits, word_bits) / (self.out_features * self.in_features)


def rank_for_budget(
    out_features: int,
    in_features: int,
    target_bpw: float,
    *,
    rank_scale: bool = False,
    scale_bits: int = 16,
    word_bits: int = 32,
    alignment: int = 32,
    max_rank: int | None = None,
) -> int:
    """Return the largest aligned rank that fits the *packed* bit budget."""
    if out_features <= 0 or in_features <= 0:
        raise ValueError("matrix dimensions must be positive")
    if target_bpw <= 0:
        raise ValueError("target_bpw must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    upper = min(out_features, in_features) if max_rank is None else min(max_rank, out_features, in_features)
    budget = target_bpw * out_features * in_features
    best = 0
    candidates: Iterable[int]
    if alignment == 1:
        candidates = range(1, upper + 1)
    else:
        candidates = range(alignment, upper + 1, alignment)
    for rank in candidates:
        factor_bits = word_bits * ceil(rank / word_bits) * (out_features + in_features)
        metadata_bits = scale_bits * (out_features + in_features + (rank if rank_scale else 0))
        if factor_bits + metadata_bits <= budget + 1e-9:
            best = rank
        else:
            break
    if best == 0:
        minimum = 1 if alignment == 1 else min(alignment, upper)
        factor_bits = word_bits * ceil(minimum / word_bits) * (out_features + in_features)
        metadata_bits = scale_bits * (out_features + in_features + (minimum if rank_scale else 0))
        required = (factor_bits + metadata_bits) / (out_features * in_features)
        raise ValueError(
            f"A target of {target_bpw:.4f} BPW cannot hold the minimum rank {minimum}; "
            f"at least {required:.4f} BPW is required with the requested alignment."
        )
    return best


def reclaim_packed_rank(requested_rank: int, maximum_rank: int, *, word_bits: int = 32) -> int:
    """Use every rank lane already paid for by packed factor storage.

    A requested rank of 24 and a rank of 32 both occupy one 32-bit word per
    factor row.  Unless the matrix dimension prevents it, returning 32 uses
    those eight otherwise padded lanes without adding factor bits.
    """
    if requested_rank <= 0 or maximum_rank <= 0:
        raise ValueError("requested_rank and maximum_rank must be positive")
    if requested_rank > maximum_rank:
        raise ValueError("requested_rank exceeds maximum_rank")
    if word_bits <= 0:
        raise ValueError("word_bits must be positive")
    paid_rank = ceil(requested_rank / word_bits) * word_bits
    return min(paid_rank, maximum_rank)


def _robust_preconditioner(
    raw: Array | None,
    length: int,
    *,
    shrinkage: float,
    clip: float,
    eps: float,
) -> tuple[Array, Array]:
    """Return normalized sqrt-Hessian preconditioner and positive raw weights."""
    weights = _positive_vector(raw, length, eps)
    center = float(np.mean(weights, dtype=np.float64))
    weights = (1.0 - shrinkage) * weights + shrinkage * center
    root = np.sqrt(np.maximum(weights, eps))
    median = max(float(np.median(root)), eps)
    root = np.clip(root, median / clip, median * clip)
    # Unit geometric mean makes ADMM penalty parameters comparable across layers.
    root /= _geometric_mean_abs(root, eps)
    return root, weights


def _power_iteration_abs(matrix: Array, iterations: int, rng: np.random.Generator, eps: float) -> tuple[Array, float, Array]:
    v = rng.standard_normal(matrix.shape[1])
    v /= max(float(np.linalg.norm(v)), eps)
    absolute = np.abs(matrix)
    for _ in range(max(iterations, 1)):
        u = absolute @ v
        u /= max(float(np.linalg.norm(u)), eps)
        v = absolute.T @ u
        v /= max(float(np.linalg.norm(v)), eps)
    unnormalized = absolute @ v
    sigma = max(float(np.linalg.norm(unnormalized)), eps)
    return unnormalized / sigma, sigma, v


def _svid_projection(matrix: Array, iterations: int, rng: np.random.Generator, eps: float) -> Array:
    u, sigma, v = _power_iteration_abs(matrix, iterations, rng, eps)
    return np.outer(u * sigma, v) * _sign_no_zero(matrix)


def _randomized_spectral_init(
    matrix: Array,
    rank: int,
    rng: np.random.Generator,
    oversample: int,
    power_iters: int,
) -> tuple[Array, Array]:
    m, n = matrix.shape
    sketch_rank = min(rank + max(oversample, 0), m, n)
    if sketch_rank >= min(m, n):
        left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
        left = left[:, :rank]
        singular = singular[:rank]
        right_t = right_t[:rank]
    else:
        omega = rng.standard_normal((n, sketch_rank))
        q, _ = np.linalg.qr(matrix @ omega, mode="reduced")
        for _ in range(max(power_iters, 0)):
            z, _ = np.linalg.qr(matrix.T @ q, mode="reduced")
            q, _ = np.linalg.qr(matrix @ z, mode="reduced")
        small = q.T @ matrix
        left_small, singular, right_t = np.linalg.svd(small, full_matrices=False)
        left = (q @ left_small)[:, :rank]
        singular = singular[:rank]
        right_t = right_t[:rank]
    root = np.sqrt(np.maximum(singular, 0.0))
    return left * root[None, :], root[:, None] * right_t


def _initial_factors(matrix: Array, config: NQXConfig, rng: np.random.Generator) -> tuple[Array, Array]:
    m, n = matrix.shape
    rank = config.rank
    if rank > min(m, n):
        raise ValueError(f"rank {rank} exceeds the smaller matrix dimension {min(m, n)}")
    if config.init == "spectral":
        return _randomized_spectral_init(
            matrix,
            rank,
            rng,
            config.spectral_oversample,
            config.spectral_power_iters,
        )
    return (
        rng.standard_normal((m, rank)),
        rng.standard_normal((rank, n)),
    )


def _stable_admm_solve(
    x: Array,
    target: Array,
    proxy: Array,
    dual: Array,
    rho: float,
    ridge: float,
    eps: float,
) -> Array:
    gram = x.T @ x
    gram = 0.5 * (gram + gram.T)
    diagonal_scale = max(float(np.mean(np.diag(gram))), eps)
    penalty = max(rho * diagonal_scale, eps)
    system = gram.copy()
    system.flat[:: system.shape[0] + 1] += penalty + ridge * diagonal_scale + eps
    rhs = x.T @ target + penalty * (proxy - dual)
    try:
        factor = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        factor = np.linalg.lstsq(system, rhs, rcond=eps)[0]
    return factor


def _continuous_reconstruction(a: Array, b: Array, eps: float) -> Array:
    middle = np.maximum(np.linalg.norm(b, axis=1), eps)
    return (a / middle[None, :]) @ b


def _initial_binary_scales(
    a_proxy: Array,
    b_proxy: Array,
    output_preconditioner: Array,
    input_preconditioner: Array,
    eps: float,
) -> tuple[Array, Array, Array, Array]:
    a = a_proxy / output_preconditioner[:, None]
    b = b_proxy / input_preconditioner[None, :]
    norm_a = max(float(np.linalg.norm(a)), eps)
    norm_b = max(float(np.linalg.norm(b)), eps)
    balance = (norm_b / norm_a) ** 0.5
    a *= balance
    b /= balance
    # Match NanoQuant's rank normalization before extracting the two boundary
    # scales.  This affects the deployed signed representation even though it
    # leaves the continuous factor product scale-ambiguous.
    a /= np.maximum(np.linalg.norm(a_proxy, axis=0), eps)[None, :]
    u = _sign_no_zero(a)
    v = _sign_no_zero(b.T)
    scale_out = np.maximum(np.mean(np.abs(a), axis=1), eps)
    scale_in = np.maximum(np.mean(np.abs(b), axis=0), eps)
    return u, v, scale_out, scale_in


def _canonicalize(
    u: Array,
    v: Array,
    scale_out: Array,
    scale_in: Array,
    rank_scale: Array | None,
    eps: float,
) -> tuple[Array, Array, Array, Array, Array | None]:
    u = u.copy()
    v = v.copy()
    scale_out = scale_out.copy()
    scale_in = scale_in.copy()
    negative_out = scale_out < 0
    u[negative_out] *= -1
    scale_out = np.maximum(np.abs(scale_out), eps)
    negative_in = scale_in < 0
    v[negative_in] *= -1
    scale_in = np.maximum(np.abs(scale_in), eps)
    if rank_scale is not None:
        rank_scale = rank_scale.copy()
        negative_rank = rank_scale < 0
        u[:, negative_rank] *= -1
        rank_scale = np.maximum(np.abs(rank_scale), eps)
    return u, v, scale_out, scale_in, rank_scale


def _solve_rank_scale(
    target: Array,
    u: Array,
    v: Array,
    scale_out: Array,
    scale_in: Array,
    output_weight: Array,
    input_weight: Array,
    ridge: float,
    eps: float,
) -> Array:
    uf = u.astype(np.float64, copy=False)
    vf = v.astype(np.float64, copy=False)
    left_weight = output_weight * scale_out * scale_out
    right_weight = input_weight * scale_in * scale_in
    h_left = uf.T @ (left_weight[:, None] * uf)
    h_right = vf.T @ (right_weight[:, None] * vf)
    hessian = h_left * h_right
    weighted_v = vf * (input_weight * scale_in)[:, None]
    projected = ((output_weight * scale_out)[:, None] * target) @ weighted_v
    rhs = np.sum(uf * projected, axis=0)
    diag_scale = max(float(np.mean(np.diag(hessian))), eps)
    hessian.flat[:: hessian.shape[0] + 1] += ridge * diag_scale + eps
    try:
        return np.linalg.solve(hessian, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(hessian, rhs, rcond=eps)[0]


def _fit_scales(
    target: Array,
    u: Array,
    v: Array,
    scale_out: Array,
    scale_in: Array,
    output_weight: Array,
    input_weight: Array,
    *,
    iterations: int,
    use_rank_scale: bool,
    ridge: float,
    eps: float,
) -> tuple[Array, Array, Array, Array, Array | None]:
    uf = u.astype(np.float64, copy=False)
    vf = v.astype(np.float64, copy=False)
    so = np.asarray(scale_out, dtype=np.float64).copy()
    si = np.asarray(scale_in, dtype=np.float64).copy()
    g = np.ones(u.shape[1], dtype=np.float64) if use_rank_scale else None

    for _ in range(max(iterations, 0)):
        if g is not None:
            g = _solve_rank_scale(target, u, v, so, si, output_weight, input_weight, ridge, eps)
            core = (uf * g[None, :]) @ vf.T
        else:
            core = uf @ vf.T

        weighted_input_scale = input_weight * si
        numerator_out = np.sum(target * core * weighted_input_scale[None, :], axis=1)
        denominator_out = np.sum(
            core * core * (input_weight * si * si)[None, :], axis=1
        )
        so = numerator_out / np.maximum(denominator_out + ridge * np.mean(denominator_out), eps)

        weighted_output_scale = output_weight * so
        numerator_in = np.sum(target * core * weighted_output_scale[:, None], axis=0)
        denominator_in = np.sum(
            core * core * (output_weight * so * so)[:, None], axis=0
        )
        si = numerator_in / np.maximum(denominator_in + ridge * np.mean(denominator_in), eps)

        # Remove the two-sided scale ambiguity without changing the matrix.
        gm_out = _geometric_mean_abs(so, eps)
        gm_in = _geometric_mean_abs(si, eps)
        balance = (gm_in / max(gm_out, eps)) ** 0.5
        so *= balance
        si /= balance

    u_out, v_out, so, si, g = _canonicalize(u, v, so, si, g, eps)
    return u_out, v_out, so, si, g


def _reconstruct_factors(
    u: Array,
    v: Array,
    scale_out: Array,
    scale_in: Array,
    rank_scale: Array | None,
) -> Array:
    uf = u.astype(np.float64, copy=False)
    vf = v.astype(np.float64, copy=False)
    if rank_scale is not None:
        uf = uf * rank_scale[None, :]
    return (scale_out[:, None] * (uf @ vf.T)) * scale_in[None, :]


def _polish_binary_factors(
    target: Array,
    u: Array,
    v: Array,
    scale_out: Array,
    scale_in: Array,
    rank_scale: Array | None,
    output_weight: Array,
    input_weight: Array,
    config: NQXConfig,
) -> tuple[Array, Array, Array, Array, Array | None]:
    if config.polish_iters <= 0:
        return u, v, scale_out, scale_in, rank_scale

    u = u.copy()
    v = v.copy()
    so = scale_out.copy()
    si = scale_in.copy()
    g = np.ones(config.rank, dtype=np.float64) if rank_scale is None else rank_scale.copy()
    prediction = _reconstruct_factors(u, v, so, si, g if rank_scale is not None else None)
    best = (u.copy(), v.copy(), so.copy(), si.copy(), None if rank_scale is None else g.copy())
    best_error = _relative_weighted_error(target, prediction, output_weight, input_weight, config.eps)

    for _ in range(config.polish_iters):
        for k in range(config.rank):
            old_component = np.outer(so * u[:, k] * g[k], si * v[:, k])
            residual_excluding = target - prediction + old_component

            u_direction = si * v[:, k]
            score_u = (residual_excluding * input_weight[None, :]) @ u_direction
            score_u *= so * g[k]
            u[:, k] = _sign_no_zero(score_u)
            new_component = np.outer(so * u[:, k] * g[k], si * v[:, k])
            prediction += new_component - old_component

            residual_excluding = target - prediction + new_component
            v_direction = so * u[:, k]
            score_v = (residual_excluding * output_weight[:, None]).T @ v_direction
            score_v *= si * g[k]
            v[:, k] = _sign_no_zero(score_v)
            updated_component = np.outer(so * u[:, k] * g[k], si * v[:, k])
            prediction += updated_component - new_component

        u, v, so, si, fitted_g = _fit_scales(
            target,
            u,
            v,
            so,
            si,
            output_weight,
            input_weight,
            iterations=max(2, config.scale_iters // 2),
            use_rank_scale=rank_scale is not None,
            ridge=config.scale_ridge,
            eps=config.eps,
        )
        if fitted_g is not None:
            g = fitted_g
        prediction = _reconstruct_factors(u, v, so, si, g if rank_scale is not None else None)
        error = _relative_weighted_error(target, prediction, output_weight, input_weight, config.eps)
        if error < best_error:
            best_error = error
            best = (u.copy(), v.copy(), so.copy(), si.copy(), None if rank_scale is None else g.copy())

    return best


def _round_fp16_positive(values: Array, eps: float) -> Array:
    """Project positive scales to the exact values stored by ``.nqx``."""
    source = np.asarray(values, dtype=np.float64)
    maximum = float(np.finfo(np.float16).max)
    smallest = float(np.nextafter(np.float16(0), np.float16(1)))
    finite = np.nan_to_num(source, nan=eps, posinf=maximum, neginf=eps)
    rounded = np.clip(finite, smallest, maximum).astype(np.float16).astype(np.float64)
    return np.maximum(rounded, smallest)


def _round_stored_scales(
    u: Array,
    v: Array,
    scale_out: Array,
    scale_in: Array,
    rank_scale: Array | None,
    eps: float,
) -> tuple[Array, Array, Array, Array, Array | None]:
    u, v, scale_out, scale_in, rank_scale = _canonicalize(
        u,
        v,
        scale_out,
        scale_in,
        rank_scale,
        eps,
    )
    return (
        u,
        v,
        _round_fp16_positive(scale_out, eps),
        _round_fp16_positive(scale_in, eps),
        None if rank_scale is None else _round_fp16_positive(rank_scale, eps),
    )


def _refine_stored_scales(
    target: Array,
    u: Array,
    v: Array,
    scale_out: Array,
    scale_in: Array,
    rank_scale: Array | None,
    output_weight: Array,
    input_weight: Array,
    *,
    iterations: int,
    ridge: float,
    eps: float,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """Optimize while projecting every accepted scale to FP16 storage.

    The final global-gain search tries algebraically equivalent placements of
    the gain because FP16 rounding makes those placements numerically
    different.  Candidate acceptance is exact and monotone.
    """

    def objective(candidate: tuple[Array, Array, Array, Array, Array | None]) -> float:
        cu, cv, cso, csi, cg = candidate
        approximation = _reconstruct_factors(cu, cv, cso, csi, cg)
        return _relative_weighted_error(target, approximation, output_weight, input_weight, eps)

    best = _round_stored_scales(u, v, scale_out, scale_in, rank_scale, eps)
    best_error = objective(best)
    use_rank_scale = rank_scale is not None

    for _ in range(iterations):
        fitted = _fit_scales(
            target,
            best[0],
            best[1],
            best[2],
            best[3],
            output_weight,
            input_weight,
            iterations=1,
            use_rank_scale=use_rank_scale,
            ridge=ridge,
            eps=eps,
        )
        projected = _round_stored_scales(*fitted, eps)
        projected_error = objective(projected)
        if np.isfinite(projected_error) and projected_error + eps < best_error:
            best = projected
            best_error = projected_error
        else:
            break

    prediction = _reconstruct_factors(*best)
    weights = output_weight[:, None] * input_weight[None, :]
    denominator = float(np.sum(weights * prediction * prediction, dtype=np.float64))
    if denominator <= eps:
        return best
    gain = float(np.sum(weights * target * prediction, dtype=np.float64) / denominator)
    if not np.isfinite(gain) or gain <= 0:
        return best

    bu, bv, bso, bsi, bg = best
    gain_candidates: list[tuple[Array, Array, Array, Array, Array | None]] = []
    gain_candidates.append((bu, bv, bso * gain, bsi, bg))
    gain_candidates.append((bu, bv, bso, bsi * gain, bg))
    root = gain**0.5
    gain_candidates.append((bu, bv, bso * root, bsi * root, bg))
    if bg is not None:
        gain_candidates.append((bu, bv, bso, bsi, bg * gain))
        cube = gain ** (1.0 / 3.0)
        gain_candidates.append((bu, bv, bso * cube, bsi * cube, bg * cube))
    for candidate in gain_candidates:
        projected = _round_stored_scales(*candidate, eps)
        candidate_error = objective(projected)
        if np.isfinite(candidate_error) and candidate_error + eps < best_error:
            best = projected
            best_error = candidate_error
    return best


@dataclass(frozen=True)
class _DeploymentCandidate:
    name: str
    u: Array
    v: Array
    scale_out: Array
    scale_in: Array
    rank_scale: Array | None
    deployed_before_refit: float
    unweighted_error: float
    weighted_error: float
    serialized_error: float


def _deploy_proxy_candidate(
    name: str,
    target: Array,
    a_proxy: Array,
    b_proxy: Array,
    output_preconditioner: Array,
    input_preconditioner: Array,
    output_weight: Array,
    input_weight: Array,
    config: NQXConfig,
) -> _DeploymentCandidate:
    u, v, scale_out, scale_in = _initial_binary_scales(
        a_proxy,
        b_proxy,
        output_preconditioner,
        input_preconditioner,
        config.eps,
    )
    initial_rank_scale = np.ones(config.rank, dtype=np.float64) if config.rank_scale else None
    initial = (u, v, scale_out, scale_in, initial_rank_scale)
    if config.storage_aware:
        initial = _refine_stored_scales(
            target,
            *initial,
            output_weight,
            input_weight,
            iterations=0,
            ridge=config.scale_ridge,
            eps=config.eps,
        )
    before_matrix = _reconstruct_factors(*initial)
    deployed_before_refit = _relative_weighted_error(
        target,
        before_matrix,
        np.ones_like(output_weight),
        np.ones_like(input_weight),
        config.eps,
    )
    initial_weighted_error = _relative_weighted_error(
        target,
        before_matrix,
        output_weight,
        input_weight,
        config.eps,
    )

    refined = _fit_scales(
        target,
        u,
        v,
        scale_out,
        scale_in,
        output_weight,
        input_weight,
        iterations=config.scale_iters,
        use_rank_scale=config.rank_scale,
        ridge=config.scale_ridge,
        eps=config.eps,
    )
    refined = _polish_binary_factors(
        target,
        *refined,
        output_weight,
        input_weight,
        config,
    )
    if config.storage_aware:
        refined = _refine_stored_scales(
            target,
            *refined,
            output_weight,
            input_weight,
            iterations=config.storage_refine_iters,
            ridge=config.scale_ridge,
            eps=config.eps,
        )
    refined_matrix = _reconstruct_factors(*refined)
    refined_weighted_error = _relative_weighted_error(
        target,
        refined_matrix,
        output_weight,
        input_weight,
        config.eps,
    )
    if not np.isfinite(refined_weighted_error) or refined_weighted_error > initial_weighted_error:
        refined = initial
        refined_matrix = before_matrix
        refined_weighted_error = initial_weighted_error

    unweighted_error = _relative_weighted_error(
        target,
        refined_matrix,
        np.ones_like(output_weight),
        np.ones_like(input_weight),
        config.eps,
    )
    serialized = _round_stored_scales(*refined, config.eps)
    serialized_matrix = _reconstruct_factors(*serialized)
    serialized_error = _relative_weighted_error(
        target,
        serialized_matrix,
        np.ones_like(output_weight),
        np.ones_like(input_weight),
        config.eps,
    )
    return _DeploymentCandidate(
        name=name,
        u=refined[0],
        v=refined[1],
        scale_out=refined[2],
        scale_in=refined[3],
        rank_scale=refined[4],
        deployed_before_refit=deployed_before_refit,
        unweighted_error=unweighted_error,
        weighted_error=refined_weighted_error,
        serialized_error=serialized_error,
    )


def quantize_matrix(
    weight: Array,
    config: NQXConfig,
    *,
    input_hessian: Array | None = None,
    output_hessian: Array | None = None,
) -> QuantizedMatrix:
    """Quantize one dense matrix into an exact packed-runtime representation.

    The objective is K-FAC weighted when diagonal input/output Hessian estimates
    are supplied.  All candidate selection and diagnostics use the matrix that
    is actually deployed after sign extraction and scale application.
    """
    source_config = config
    target = np.asarray(weight)
    if target.ndim != 2:
        raise ValueError(f"weight must be a matrix, got shape {target.shape}")
    if not np.issubdtype(target.dtype, np.floating):
        raise TypeError("weight must have a floating-point dtype")
    if not np.all(np.isfinite(target)):
        raise ValueError("weight contains NaN or infinite values")
    target64 = target.astype(np.float64, copy=False)
    out_features, in_features = target64.shape
    if config.rank > min(out_features, in_features):
        raise ValueError(
            f"rank {config.rank} exceeds the smaller matrix dimension {min(out_features, in_features)}"
        )
    if config.reclaim_packed_padding:
        effective_rank = reclaim_packed_rank(config.rank, min(out_features, in_features))
        if effective_rank != config.rank:
            config = replace(config, rank=effective_rank)

    input_pre, input_weight = _robust_preconditioner(
        input_hessian,
        in_features,
        shrinkage=config.precondition_shrinkage,
        clip=config.precondition_clip,
        eps=config.eps,
    )
    output_pre, output_weight = _robust_preconditioner(
        output_hessian,
        out_features,
        shrinkage=config.precondition_shrinkage,
        clip=config.precondition_clip,
        eps=config.eps,
    )
    conditioned = (output_pre[:, None] * target64) * input_pre[None, :]
    conditioned_norm = max(float(np.linalg.norm(conditioned)), config.eps)

    rng = np.random.default_rng(config.seed)
    a_continuous, b_continuous = _initial_factors(conditioned, config, rng)
    a_proxy = _svid_projection(a_continuous, config.projection_iters, rng, config.eps)
    b_proxy = _svid_projection(b_continuous, config.projection_iters, rng, config.eps)
    # The scaled dual initialization follows the released NanoQuant solver and
    # is materially more stable than a zero dual for random initial factors.
    dual_a = a_continuous - a_proxy
    dual_b = b_continuous - b_proxy
    rho = float(np.clip(config.rho_init, config.rho_min, config.rho_max))
    initial_rho = rho
    best_error = float("inf")
    best_proxy = (a_proxy.copy(), b_proxy.copy())
    no_improvement = 0
    history: list[dict[str, float]] = []
    primal = float("inf")
    dual = float("inf")
    stopped_early = False

    for iteration in range(config.max_iters):
        if not config.adaptive_rho:
            progress = iteration / max(config.max_iters, 1)
            rho = float(
                np.clip(
                    config.rho_init + progress * (config.rho_final - config.rho_init),
                    config.rho_min,
                    config.rho_max,
                )
            )
        old_a = a_proxy.copy()
        old_b = b_proxy.copy()

        norm_b = np.maximum(np.linalg.norm(b_proxy, axis=1), config.eps)
        x_a = b_proxy.T / norm_b[None, :]
        a_continuous = _stable_admm_solve(
            x_a,
            conditioned.T,
            a_proxy.T,
            dual_a.T,
            rho,
            config.ridge,
            config.eps,
        ).T

        norm_a = np.maximum(np.linalg.norm(a_proxy, axis=0), config.eps)
        x_b = a_proxy / norm_a[None, :]
        b_continuous = _stable_admm_solve(
            x_b,
            conditioned,
            b_proxy,
            dual_b,
            rho,
            config.ridge,
            config.eps,
        )

        a_proxy = _svid_projection(a_continuous + dual_a, config.projection_iters, rng, config.eps)
        b_proxy = _svid_projection(b_continuous + dual_b, config.projection_iters, rng, config.eps)
        dual_a += a_continuous - a_proxy
        dual_b += b_continuous - b_proxy

        primal = float(
            np.sqrt(
                np.linalg.norm(a_continuous - a_proxy) ** 2
                + np.linalg.norm(b_continuous - b_proxy) ** 2
            )
        )
        dual = float(
            rho
            * np.sqrt(
                np.linalg.norm(a_proxy - old_a) ** 2
                + np.linalg.norm(b_proxy - old_b) ** 2
            )
        )
        continuous = _continuous_reconstruction(a_proxy, b_proxy, config.eps)
        continuous_error = float(np.linalg.norm(conditioned - continuous) ** 2 / (conditioned_norm**2))

        if continuous_error + config.eps < best_error:
            best_error = continuous_error
            best_proxy = (a_proxy.copy(), b_proxy.copy())
            no_improvement = 0
        else:
            no_improvement += 1

        normalizer = max(
            float(np.sqrt(np.linalg.norm(a_continuous) ** 2 + np.linalg.norm(b_continuous) ** 2)),
            config.eps,
        )
        relative_primal = primal / normalizer
        relative_dual = dual / normalizer
        history.append(
            {
                "iteration": float(iteration + 1),
                "rho": rho,
                "continuous_error": continuous_error,
                "relative_primal": relative_primal,
                "relative_dual": relative_dual,
            }
        )

        # Residual balancing avoids spending all layers on a fixed 400-step schedule.
        if config.adaptive_rho:
            if primal > config.rho_balance * max(dual, config.eps) and rho < config.rho_max:
                new_rho = min(rho * config.rho_multiplier, config.rho_max)
                scale = rho / new_rho
                dual_a *= scale
                dual_b *= scale
                rho = new_rho
            elif dual > config.rho_balance * max(primal, config.eps) and rho > config.rho_min:
                new_rho = max(rho / config.rho_multiplier, config.rho_min)
                scale = rho / new_rho
                dual_a *= scale
                dual_b *= scale
                rho = new_rho

        enough_iterations = iteration + 1 >= config.min_iters
        converged = relative_primal <= config.tolerance and relative_dual <= config.tolerance
        plateaued = no_improvement >= config.patience
        if enough_iterations and (converged or plateaued):
            stopped_early = True
            break

    # Continuous-factor quality and signed deployment quality are not
    # equivalent. Evaluate the final schedule state and, at negligible ADMM
    # cost, the best continuous iterate after exact scale/sign finalization.
    proxy_candidates = [("final", a_proxy, b_proxy)]
    if config.candidate_selection and not (
        np.array_equal(a_proxy, best_proxy[0]) and np.array_equal(b_proxy, best_proxy[1])
    ):
        proxy_candidates.append(("best_continuous", best_proxy[0], best_proxy[1]))
    deployment_candidates = [
        _deploy_proxy_candidate(
            name,
            target64,
            candidate_a,
            candidate_b,
            output_pre,
            input_pre,
            output_weight,
            input_weight,
            config,
        )
        for name, candidate_a, candidate_b in proxy_candidates
    ]
    selected = min(deployment_candidates, key=lambda candidate: candidate.weighted_error)

    diagnostics = QuantizationDiagnostics(
        iterations=len(history),
        stopped_early=stopped_early,
        initial_rho=initial_rho,
        final_rho=rho,
        continuous_error=best_error,
        deployed_error_before_refit=selected.deployed_before_refit,
        deployed_error=selected.unweighted_error,
        weighted_deployed_error=selected.weighted_error,
        primal_residual=primal,
        dual_residual=dual,
        serialized_deployed_error=selected.serialized_error,
        selected_candidate=selected.name,
        candidates_evaluated=len(deployment_candidates),
        history=history,
    )
    return QuantizedMatrix(
        u=selected.u,
        v=selected.v,
        scale_out=selected.scale_out,
        scale_in=selected.scale_in,
        rank_scale=selected.rank_scale,
        diagnostics=diagnostics,
        config={
            **source_config.to_dict(),
            "requested_rank": source_config.rank,
            "effective_rank": config.rank,
        },
    )


def paper_style_baseline(
    weight: Array,
    rank: int,
    *,
    iterations: int = 96,
    projection_iters: int = 5,
    seed: int = 0,
    input_hessian: Array | None = None,
    output_hessian: Array | None = None,
    eps: float = 1e-10,
) -> QuantizedMatrix:
    """Matrix-level paper-style baseline used by the included benchmark.

    This reproduces the fixed-schedule random initialization, linear penalty,
    SVID projection, magnitude extraction, and two-scale deployment path.  It is
    not a replacement for the paper's later block-level STE/KD stages.
    """
    target = np.asarray(weight, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("weight must be a matrix")
    m, n = target.shape
    if rank <= 0 or rank > min(m, n):
        raise ValueError("invalid rank")
    input_pre = np.sqrt(_positive_vector(input_hessian, n, eps))
    output_pre = np.sqrt(_positive_vector(output_hessian, m, eps))
    conditioned = output_pre[:, None] * target * input_pre[None, :]
    rng = np.random.default_rng(seed)
    a_continuous = rng.standard_normal((m, rank))
    b_continuous = rng.standard_normal((rank, n))
    a_proxy = _svid_projection(a_continuous, projection_iters, rng, eps)
    b_proxy = _svid_projection(b_continuous, projection_iters, rng, eps)
    dual_a = a_continuous - a_proxy
    dual_b = b_continuous - b_proxy
    history: list[dict[str, float]] = []
    primal = dual = 0.0
    for index in range(iterations):
        rho = (index / max(iterations, 1))
        old_a = a_proxy.copy()
        old_b = b_proxy.copy()
        norm_b = np.maximum(np.linalg.norm(b_proxy, axis=1), eps)
        x_a = b_proxy.T / norm_b[None, :]
        a_continuous = _stable_admm_solve(x_a, conditioned.T, a_proxy.T, dual_a.T, rho, 3e-2, eps).T
        norm_a = np.maximum(np.linalg.norm(a_proxy, axis=0), eps)
        x_b = a_proxy / norm_a[None, :]
        b_continuous = _stable_admm_solve(x_b, conditioned, b_proxy, dual_b, rho, 3e-2, eps)
        a_proxy = _svid_projection(a_continuous + dual_a, projection_iters, rng, eps)
        b_proxy = _svid_projection(b_continuous + dual_b, projection_iters, rng, eps)
        dual_a += a_continuous - a_proxy
        dual_b += b_continuous - b_proxy
        primal = float(np.linalg.norm(a_continuous - a_proxy) + np.linalg.norm(b_continuous - b_proxy))
        dual = float(rho * (np.linalg.norm(a_proxy - old_a) + np.linalg.norm(b_proxy - old_b)))
        history.append({"iteration": float(index + 1), "rho": rho})

    u, v, scale_out, scale_in = _initial_binary_scales(a_proxy, b_proxy, output_pre, input_pre, eps)
    deployed = _reconstruct_factors(u, v, scale_out, scale_in, None)
    error = _relative_weighted_error(
        target,
        deployed,
        np.ones(m),
        np.ones(n),
        eps,
    )
    diagnostics = QuantizationDiagnostics(
        iterations=iterations,
        stopped_early=False,
        initial_rho=0.0,
        final_rho=(iterations - 1) / max(iterations, 1),
        continuous_error=float(np.linalg.norm(conditioned - _continuous_reconstruction(a_proxy, b_proxy, eps)) ** 2)
        / max(float(np.linalg.norm(conditioned) ** 2), eps),
        deployed_error_before_refit=error,
        deployed_error=error,
        weighted_deployed_error=error,
        primal_residual=primal,
        dual_residual=dual,
        history=history,
    )
    return QuantizedMatrix(
        u=u,
        v=v,
        scale_out=scale_out,
        scale_in=scale_in,
        diagnostics=diagnostics,
        config={"kind": "paper_style_matrix_baseline", "rank": rank, "iterations": iterations, "seed": seed},
    )
