"""Global, packed-bit-aware rank allocation for NanoQuant-X.

Copyright 2026 Godson Johnson
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import ceil, isfinite


@dataclass(frozen=True)
class LayerBudget:
    name: str
    out_features: int
    in_features: int
    sensitivity: float
    distortion_by_rank: dict[int, float] | None = None

    @property
    def parameters(self) -> int:
        return self.out_features * self.in_features


@dataclass(frozen=True)
class Allocation:
    ranks: dict[str, int]
    used_bits: int
    budget_bits: int
    effective_bpw: float


def layer_storage_bits(
    layer: LayerBudget,
    rank: int,
    *,
    rank_scale: bool,
    scale_bits: int = 16,
    word_bits: int = 32,
) -> int:
    factor = word_bits * ceil(rank / word_bits) * (layer.out_features + layer.in_features)
    scales = scale_bits * (layer.out_features + layer.in_features + (rank if rank_scale else 0))
    return int(factor + scales)


def allocate_global_ranks(
    layers: list[LayerBudget],
    target_bpw: float,
    *,
    rank_scale: bool = True,
    alignment: int = 32,
    minimum_rank: int = 32,
    scale_bits: int = 16,
    word_bits: int = 32,
) -> Allocation:
    """Allocate rank increments by expected error reduction per packed bit.

    When a layer supplies ``distortion_by_rank``, measured pilot
    rate-distortion points take precedence over the sensitivity proxy. This
    keeps the fast zero-pilot path while allowing production runs to make
    allocation decisions from actual layer behavior.
    """
    if not layers:
        return Allocation({}, 0, 0, 0.0)
    if not isfinite(target_bpw) or target_bpw <= 0:
        raise ValueError("target_bpw must be positive")
    if alignment <= 0 or minimum_rank <= 0:
        raise ValueError("rank alignment and minimum rank must be positive")
    names = [layer.name for layer in layers]
    if len(names) != len(set(names)):
        raise ValueError("layer names must be unique")
    for layer in layers:
        if layer.out_features <= 0 or layer.in_features <= 0:
            raise ValueError(f"layer dimensions must be positive: {layer.name}")
        if not isfinite(layer.sensitivity) or layer.sensitivity < 0:
            raise ValueError(f"layer sensitivity must be finite and non-negative: {layer.name}")
        if layer.distortion_by_rank is not None:
            for rank, distortion in layer.distortion_by_rank.items():
                if rank <= 0 or not isfinite(distortion) or distortion < 0:
                    raise ValueError(f"invalid rate-distortion point for {layer.name}")
    total_parameters = sum(layer.parameters for layer in layers)
    budget_bits = int(target_bpw * total_parameters)
    ranks: dict[str, int] = {}
    used_bits = 0
    for layer in layers:
        maximum = min(layer.out_features, layer.in_features)
        initial = min(minimum_rank, maximum)
        initial = max(1, (initial // alignment) * alignment) if maximum >= alignment else maximum
        ranks[layer.name] = initial
        used_bits += layer_storage_bits(
            layer, initial, rank_scale=rank_scale, scale_bits=scale_bits, word_bits=word_bits
        )
    if used_bits > budget_bits:
        required = used_bits / total_parameters
        raise ValueError(
            f"The minimum aligned ranks require {required:.6f} BPW, exceeding the target {target_bpw:.6f}."
        )

    by_name = {layer.name: layer for layer in layers}
    heap: list[tuple[float, str, int, int]] = []

    def candidate(layer: LayerBudget, current: int) -> tuple[float, int, int] | None:
        maximum = (min(layer.out_features, layer.in_features) // alignment) * alignment
        proposed = current + alignment
        if proposed > maximum:
            return None
        old_bits = layer_storage_bits(
            layer, current, rank_scale=rank_scale, scale_bits=scale_bits, word_bits=word_bits
        )
        new_bits = layer_storage_bits(
            layer, proposed, rank_scale=rank_scale, scale_bits=scale_bits, word_bits=word_bits
        )
        cost = new_bits - old_bits
        if (
            layer.distortion_by_rank is not None
            and current in layer.distortion_by_rank
            and proposed in layer.distortion_by_rank
        ):
            gain = layer.parameters * max(
                float(layer.distortion_by_rank[current]) - float(layer.distortion_by_rank[proposed]),
                0.0,
            )
        else:
            # A conservative 1/r proxy for residual energy gives diminishing returns.
            energy = float(layer.sensitivity) * layer.parameters
            gain = energy * (1.0 / max(current, 1) - 1.0 / proposed)
        return gain / max(cost, 1), proposed, cost

    for layer in layers:
        item = candidate(layer, ranks[layer.name])
        if item is not None:
            score, proposed, cost = item
            heapq.heappush(heap, (-score, layer.name, proposed, cost))

    while heap:
        negative_score, name, proposed, cost = heapq.heappop(heap)
        if proposed != ranks[name] + alignment:
            continue
        if used_bits + cost > budget_bits:
            continue
        ranks[name] = proposed
        used_bits += cost
        next_item = candidate(by_name[name], proposed)
        if next_item is not None:
            score, next_rank, next_cost = next_item
            heapq.heappush(heap, (-score, name, next_rank, next_cost))

    return Allocation(
        ranks=ranks,
        used_bits=used_bits,
        budget_bits=budget_bits,
        effective_bpw=used_bits / total_parameters,
    )
