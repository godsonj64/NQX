"""End-to-end portable NanoQuant-X example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nanoquant.reference import NQXConfig, load_nqx, quantize_matrix, save_nqx


def main() -> None:
    rng = np.random.default_rng(0)
    weight = (
        rng.standard_normal((128, 12)) @ rng.standard_normal((12, 96)) / np.sqrt(12)
        + 0.08 * rng.standard_normal((128, 96))
    ).astype(np.float32)
    input_hessian = np.exp(rng.normal(scale=0.5, size=96))
    output_hessian = np.exp(rng.normal(scale=0.5, size=128))

    quantized = quantize_matrix(
        weight,
        NQXConfig(rank=32, max_iters=96, min_iters=96, rank_scale=True, seed=0),
        input_hessian=input_hessian,
        output_hessian=output_hessian,
    )
    path = save_nqx(quantized, Path("matrix-demo.nqx"))
    restored = load_nqx(path)

    x = rng.standard_normal((4, weight.shape[1])).astype(np.float32)
    dense = x @ restored.reconstruct().T
    factorized = restored.matmul(x)
    print(f"artifact={path}")
    print(f"effective_bpw={restored.effective_bpw():.6f}")
    print(f"relative_weight_error={restored.diagnostics.deployed_error:.6f}")
    print(f"max_matmul_error={np.max(np.abs(dense - factorized)):.6e}")


if __name__ == "__main__":
    main()

