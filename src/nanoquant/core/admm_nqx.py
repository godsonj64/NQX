"""Enhanced NanoQuant factorization with exact deployment-objective refitting.

This module preserves the released NanoQuant ADMM/SVID initializer and adds:

* exact signed-runtime reconstruction and metrics;
* alternating K-FAC-weighted channel-scale least squares;
* an optional per-rank scale vector (balanced profile);
* chunked reconstruction to limit peak memory; and
* a non-regression guard against scale-refit numerical failures.

The base implementation is Apache-2.0 software from Samsung Electronics.  The
enhancements in this file are Copyright 2026 Godson Johnson and are also
licensed under Apache-2.0.
"""

from __future__ import annotations

from typing import Any

import torch

from .admm_nq import factorize_admm_nanoquant


def _sign_no_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))


def _normalized_positive(x: torch.Tensor, eps: float) -> torch.Tensor:
    value = torch.nan_to_num(x.detach().float().reshape(-1), nan=1.0, posinf=1.0, neginf=1.0)
    value = value.clamp_min(eps)
    return value / value.mean().clamp_min(eps)


def _round_bfloat16_scales(
    u: torch.Tensor,
    v: torch.Tensor,
    scale_out: torch.Tensor,
    scale_in: torch.Tensor,
    rank_scale: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Project scales to the values stored by ``NanoQuantLinear``."""
    u = u.clone()
    v = v.clone()
    maximum = torch.finfo(torch.bfloat16).max
    scale_out = torch.nan_to_num(scale_out.float(), nan=eps, posinf=maximum, neginf=-maximum).clone()
    scale_in = torch.nan_to_num(scale_in.float(), nan=eps, posinf=maximum, neginf=-maximum).clone()
    negative_out = scale_out < 0
    u[negative_out] = -u[negative_out]
    negative_in = scale_in < 0
    v[negative_in] = -v[negative_in]
    scale_out = scale_out.abs().clamp(min=eps, max=maximum).to(torch.bfloat16).float()
    scale_in = scale_in.abs().clamp(min=eps, max=maximum).to(torch.bfloat16).float()
    if rank_scale is not None:
        rank_scale = torch.nan_to_num(
            rank_scale.float(), nan=eps, posinf=maximum, neginf=-maximum
        ).clone()
        negative_rank = rank_scale < 0
        u[:, negative_rank] = -u[:, negative_rank]
        rank_scale = rank_scale.abs().clamp(min=eps, max=maximum).to(torch.bfloat16).float()
    return u, v, scale_out, scale_in, rank_scale


@torch.no_grad()
def _solve_rank_scale(
    weight: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    scale_out: torch.Tensor,
    scale_in: torch.Tensor,
    output_weight: torch.Tensor,
    input_weight: torch.Tensor,
    ridge: float,
    eps: float,
    chunk_rows: int,
) -> torch.Tensor:
    left_weight = output_weight * scale_out.square()
    right_weight = input_weight * scale_in.square()
    h_left = u.mT @ (left_weight[:, None] * u)
    h_right = v.mT @ (right_weight[:, None] * v)
    hessian = h_left * h_right
    weighted_v = v * (input_weight * scale_in)[:, None]
    rhs = torch.zeros(u.shape[1], device=weight.device, dtype=torch.float32)
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        projected = ((output_weight[start:end] * scale_out[start:end])[:, None] * weight[start:end].float()) @ weighted_v
        rhs.add_((u[start:end] * projected).sum(dim=0))
    diagonal_scale = hessian.diagonal().mean().abs().clamp_min(eps)
    hessian = 0.5 * (hessian + hessian.mT)
    hessian.diagonal().add_(ridge * diagonal_scale + eps)
    chol, info = torch.linalg.cholesky_ex(hessian)
    if int(info.item()) == 0:
        return torch.cholesky_solve(rhs[:, None], chol).squeeze(1)
    return torch.linalg.lstsq(hessian, rhs[:, None]).solution.squeeze(1)


@torch.no_grad()
def _fit_deployed_scales(
    weight: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    scale_out: torch.Tensor,
    scale_in: torch.Tensor,
    output_weight: torch.Tensor,
    input_weight: torch.Tensor,
    *,
    iterations: int,
    use_rank_scale: bool,
    ridge: float,
    eps: float,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    u = u.float().clone()
    v = v.float().clone()
    scale_out = scale_out.float().reshape(-1).clone()
    scale_in = scale_in.float().reshape(-1).clone()
    rank_scale = torch.ones(u.shape[1], device=u.device, dtype=torch.float32) if use_rank_scale else None

    for _ in range(max(iterations, 0)):
        if rank_scale is not None:
            rank_scale = _solve_rank_scale(
                weight,
                u,
                v,
                scale_out,
                scale_in,
                output_weight,
                input_weight,
                ridge,
                eps,
                chunk_rows,
            )

        numerator_out = torch.empty_like(scale_out)
        denominator_out = torch.empty_like(scale_out)
        input_linear = input_weight * scale_in
        input_quadratic = input_weight * scale_in.square()
        scaled_u = u if rank_scale is None else u * rank_scale[None, :]
        for start in range(0, weight.shape[0], chunk_rows):
            end = min(start + chunk_rows, weight.shape[0])
            core = scaled_u[start:end] @ v.mT
            current_weight = weight[start:end].float()
            numerator_out[start:end] = (current_weight * core * input_linear[None, :]).sum(dim=1)
            denominator_out[start:end] = (core.square() * input_quadratic[None, :]).sum(dim=1)
        regularizer = ridge * denominator_out.mean().abs()
        scale_out = numerator_out / (denominator_out + regularizer).clamp_min(eps)

        numerator_in = torch.zeros_like(scale_in)
        denominator_in = torch.zeros_like(scale_in)
        for start in range(0, weight.shape[0], chunk_rows):
            end = min(start + chunk_rows, weight.shape[0])
            core = scaled_u[start:end] @ v.mT
            current_weight = weight[start:end].float()
            output_linear = output_weight[start:end] * scale_out[start:end]
            output_quadratic = output_weight[start:end] * scale_out[start:end].square()
            numerator_in.add_((current_weight * core * output_linear[:, None]).sum(dim=0))
            denominator_in.add_((core.square() * output_quadratic[:, None]).sum(dim=0))
        regularizer = ridge * denominator_in.mean().abs()
        scale_in = numerator_in / (denominator_in + regularizer).clamp_min(eps)

        gm_out = torch.exp(torch.log(scale_out.abs().clamp_min(eps)).mean())
        gm_in = torch.exp(torch.log(scale_in.abs().clamp_min(eps)).mean())
        balance = torch.sqrt(gm_in / gm_out.clamp_min(eps))
        scale_out.mul_(balance)
        scale_in.div_(balance)

    negative_out = scale_out < 0
    u[negative_out] = -u[negative_out]
    scale_out.abs_().clamp_min_(eps)
    negative_in = scale_in < 0
    v[negative_in] = -v[negative_in]
    scale_in.abs_().clamp_min_(eps)
    if rank_scale is not None:
        negative_rank = rank_scale < 0
        u[:, negative_rank] = -u[:, negative_rank]
        rank_scale.abs_().clamp_min_(eps)
    return u, v, scale_out, scale_in, rank_scale


@torch.no_grad()
def _weighted_error(
    weight: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    scale_out: torch.Tensor,
    scale_in: torch.Tensor,
    rank_scale: torch.Tensor | None,
    output_weight: torch.Tensor,
    input_weight: torch.Tensor,
    eps: float,
    chunk_rows: int,
) -> torch.Tensor:
    numerator = torch.zeros((), device=weight.device, dtype=torch.float64)
    denominator = torch.zeros((), device=weight.device, dtype=torch.float64)
    scaled_u = u if rank_scale is None else u * rank_scale[None, :]
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        core = scaled_u[start:end] @ v.mT
        approximation = scale_out[start:end, None] * core * scale_in[None, :]
        target = weight[start:end].float()
        importance = output_weight[start:end, None] * input_weight[None, :]
        numerator.add_((importance * (target - approximation).square()).sum(dtype=torch.float64))
        denominator.add_((importance * target.square()).sum(dtype=torch.float64))
    return numerator / denominator.clamp_min(eps)


@torch.no_grad()
def _storage_aware_scales(
    weight: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    scale_out: torch.Tensor,
    scale_in: torch.Tensor,
    rank_scale: torch.Tensor | None,
    output_weight: torch.Tensor,
    input_weight: torch.Tensor,
    eps: float,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Round to BF16 and place a global gain where rounding is most accurate."""
    best = _round_bfloat16_scales(u, v, scale_out, scale_in, rank_scale, eps)
    best_error = _weighted_error(weight, *best, output_weight, input_weight, eps, chunk_rows)

    bu, bv, bso, bsi, bg = best
    numerator = torch.zeros((), device=weight.device, dtype=torch.float64)
    denominator = torch.zeros((), device=weight.device, dtype=torch.float64)
    scaled_u = bu if bg is None else bu * bg[None, :]
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        approximation = bso[start:end, None] * (scaled_u[start:end] @ bv.mT) * bsi[None, :]
        importance = output_weight[start:end, None] * input_weight[None, :]
        numerator.add_((importance * weight[start:end].float() * approximation).sum(dtype=torch.float64))
        denominator.add_((importance * approximation.square()).sum(dtype=torch.float64))
    gain = (numerator / denominator.clamp_min(eps)).float()
    if not bool(torch.isfinite(gain)) or float(gain.item()) <= 0:
        return best

    root = torch.sqrt(gain)
    candidates = [
        (bu, bv, bso * gain, bsi, bg),
        (bu, bv, bso, bsi * gain, bg),
        (bu, bv, bso * root, bsi * root, bg),
    ]
    if bg is not None:
        candidates.append((bu, bv, bso, bsi, bg * gain))
        cube = torch.pow(gain, 1.0 / 3.0)
        candidates.append((bu, bv, bso * cube, bsi * cube, bg * cube))
    for candidate in candidates:
        projected = _round_bfloat16_scales(*candidate, eps)
        error = _weighted_error(weight, *projected, output_weight, input_weight, eps, chunk_rows)
        if bool(torch.isfinite(error)) and bool(error + eps < best_error):
            best = projected
            best_error = error
    return best


@torch.no_grad()
def _materialize(
    weight: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    scale_out: torch.Tensor,
    scale_in: torch.Tensor,
    rank_scale: torch.Tensor | None,
    chunk_rows: int,
) -> torch.Tensor:
    output = torch.empty_like(weight)
    scaled_u = u if rank_scale is None else u * rank_scale[None, :]
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        core = scaled_u[start:end] @ v.mT
        output[start:end] = (scale_out[start:end, None] * core * scale_in[None, :]).to(weight.dtype)
    return output


@torch.no_grad()
def factorize_admm_nqx(
    weight: torch.Tensor,
    i_norm: torch.Tensor,
    o_norm: torch.Tensor,
    mid_rank: int,
    *,
    outer_iters: int = 400,
    inner_iters: int = 5,
    reg: float = 3e-2,
    is_transpose: bool = False,
    eps: float = 1e-8,
    rho_scheduler: str = "linear",
    print_admm_steps: bool = False,
    scale_iters: int = 4,
    scale_ridge: float = 1e-6,
    rank_scale: bool = True,
    chunk_rows: int = 256,
    storage_aware: bool = True,
) -> dict[str, Any]:
    """Run the official initializer, then optimize the exact signed runtime."""
    if mid_rank <= 0 or mid_rank > min(weight.shape):
        raise ValueError(f"mid_rank must lie in [1, {min(weight.shape)}], got {mid_rank}")
    if outer_iters < 1 or inner_iters < 1:
        raise ValueError("ADMM iteration counts must be positive")
    if scale_iters < 0 or scale_ridge < 0 or reg < 0:
        raise ValueError("scale iterations and regularization values must be non-negative")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if is_transpose:
        result = factorize_admm_nqx(
            weight.mT,
            o_norm,
            i_norm,
            mid_rank,
            outer_iters=outer_iters,
            inner_iters=inner_iters,
            reg=reg,
            is_transpose=False,
            eps=eps,
            rho_scheduler=rho_scheduler,
            print_admm_steps=print_admm_steps,
            scale_iters=scale_iters,
            scale_ridge=scale_ridge,
            rank_scale=rank_scale,
            chunk_rows=chunk_rows,
            storage_aware=storage_aware,
        )
        result["W_final"] = result["W_final"].mT
        result["A"], result["B"] = result["B"], result["A"]
        result["A_latent"], result["B_latent"] = result["B_latent"], result["A_latent"]
        result["scale_pre"], result["scale_post"] = result["scale_post"], result["scale_pre"]
        return result

    base = factorize_admm_nanoquant(
        weight,
        i_norm,
        o_norm,
        mid_rank=mid_rank,
        outer_iters=outer_iters,
        inner_iters=inner_iters,
        reg=reg,
        is_transpose=False,
        eps=eps,
        rho_scheduler=rho_scheduler,
        print_admm_steps=print_admm_steps,
    )
    u_initial = _sign_no_zero(base["A"].mT.float())
    v_initial = _sign_no_zero(base["B"].mT.float())
    scale_out_initial = base["scale_post"].float().reshape(-1)
    scale_in_initial = base["scale_pre"].float().reshape(-1)
    output_weight = _normalized_positive(o_norm, eps)
    input_weight = _normalized_positive(i_norm, eps)
    initial_rank_scale = torch.ones(mid_rank, device=weight.device, dtype=torch.float32) if rank_scale else None
    if storage_aware:
        (
            u_initial,
            v_initial,
            scale_out_initial,
            scale_in_initial,
            initial_rank_scale,
        ) = _storage_aware_scales(
            weight,
            u_initial,
            v_initial,
            scale_out_initial,
            scale_in_initial,
            initial_rank_scale,
            output_weight,
            input_weight,
            eps,
            chunk_rows,
        )

    initial_error = _weighted_error(
        weight,
        u_initial,
        v_initial,
        scale_out_initial,
        scale_in_initial,
        initial_rank_scale,
        output_weight,
        input_weight,
        eps,
        chunk_rows,
    )
    u, v, scale_out, scale_in, middle = _fit_deployed_scales(
        weight,
        u_initial,
        v_initial,
        scale_out_initial,
        scale_in_initial,
        output_weight,
        input_weight,
        iterations=scale_iters,
        use_rank_scale=rank_scale,
        ridge=scale_ridge,
        eps=eps,
        chunk_rows=chunk_rows,
    )
    if storage_aware:
        u, v, scale_out, scale_in, middle = _storage_aware_scales(
            weight,
            u,
            v,
            scale_out,
            scale_in,
            middle,
            output_weight,
            input_weight,
            eps,
            chunk_rows,
        )
    fitted_error = _weighted_error(
        weight,
        u,
        v,
        scale_out,
        scale_in,
        middle,
        output_weight,
        input_weight,
        eps,
        chunk_rows,
    )
    fitted_is_finite = bool(torch.isfinite(fitted_error)) and all(
        bool(torch.isfinite(tensor).all())
        for tensor in (u, v, scale_out, scale_in, middle)
        if tensor is not None
    )
    if not fitted_is_finite or bool(fitted_error > initial_error):
        u = u_initial
        v = v_initial
        scale_out = scale_out_initial
        scale_in = scale_in_initial
        middle = initial_rank_scale
        fitted_error = initial_error

    exact_weight = _materialize(weight, u, v, scale_out, scale_in, middle, chunk_rows)
    # STE starts from the exact selected signs while preserving a unit margin.
    a_latent = u.mT.contiguous()
    b_latent = v.mT.contiguous()
    result: dict[str, Any] = {
        "W_final": exact_weight,
        "A": u.mT.contiguous(),
        "B": v.mT.contiguous(),
        "A_latent": a_latent,
        "B_latent": b_latent,
        "scale_pre": scale_in.reshape(1, -1),
        "scale_post": scale_out.reshape(1, -1),
        "nqx_diagnostics": {
            "deployed_weighted_error_before_refit": float(initial_error.item()),
            "deployed_weighted_error": float(fitted_error.item()),
            "scale_refit_iterations": int(scale_iters),
            "rank_scale": bool(rank_scale),
            "storage_aware_bfloat16": bool(storage_aware),
        },
    }
    if middle is not None:
        result["scale_mid"] = middle.reshape(1, -1)
    return result
