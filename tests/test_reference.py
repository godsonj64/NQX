"""Tests for the dependency-light NanoQuant-X reference path."""

from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from nanoquant.reference import (
    NQXConfig,
    QuantizedMatrix,
    load_nqx,
    pack_signs,
    paper_style_baseline,
    quantize_matrix,
    reclaim_packed_rank,
    rank_for_budget,
    save_nqx,
    unpack_signs,
)
from nanoquant.core.budget import LayerBudget, allocate_global_ranks, layer_storage_bits


class PackingTests(unittest.TestCase):
    def test_round_trip_with_padding(self) -> None:
        rng = np.random.default_rng(7)
        for rows, columns in ((1, 1), (3, 31), (4, 32), (5, 33), (7, 97)):
            signs = np.where(rng.integers(0, 2, size=(rows, columns)) == 0, -1, 1).astype(np.int8)
            packed = pack_signs(signs)
            restored = unpack_signs(packed, signs.shape)
            np.testing.assert_array_equal(restored, signs)

    def test_rejects_non_binary_input(self) -> None:
        with self.assertRaises(ValueError):
            pack_signs(np.array([[0, 1]], dtype=np.int8))


class BudgetTests(unittest.TestCase):
    def test_rank_respects_actual_packed_budget(self) -> None:
        rank = rank_for_budget(4096, 4096, 0.75, rank_scale=True)
        q = QuantizedMatrix(
            u=np.ones((4096, rank), dtype=np.int8),
            v=np.ones((4096, rank), dtype=np.int8),
            scale_out=np.ones(4096),
            scale_in=np.ones(4096),
            rank_scale=np.ones(rank),
        )
        self.assertLessEqual(q.effective_bpw(), 0.75)
        if rank + 32 <= 4096:
            q_larger = QuantizedMatrix(
                u=np.ones((4096, rank + 32), dtype=np.int8),
                v=np.ones((4096, rank + 32), dtype=np.int8),
                scale_out=np.ones(4096),
                scale_in=np.ones(4096),
                rank_scale=np.ones(rank + 32),
            )
            self.assertGreater(q_larger.effective_bpw(), 0.75)

    def test_global_allocator_honors_budget_and_sensitivity(self) -> None:
        layers = [
            LayerBudget("sensitive", 1024, 1024, 10.0),
            LayerBudget("ordinary", 1024, 1024, 1.0),
        ]
        allocation = allocate_global_ranks(layers, 1.0, rank_scale=True)
        self.assertLessEqual(allocation.used_bits, allocation.budget_bits)
        self.assertLessEqual(allocation.effective_bpw, 1.0)
        self.assertGreater(allocation.ranks["sensitive"], allocation.ranks["ordinary"])

    def test_measured_rate_distortion_overrides_proxy(self) -> None:
        measured = LayerBudget("measured", 1024, 1024, 0.1, {32: 1.0, 64: 0.1})
        proxy = LayerBudget("proxy", 1024, 1024, 10.0, {32: 1.0, 64: 0.99})
        base = layer_storage_bits(measured, 32, rank_scale=False) * 2
        increment = layer_storage_bits(measured, 64, rank_scale=False) - layer_storage_bits(
            measured, 32, rank_scale=False
        )
        target = (base + increment) / (measured.parameters + proxy.parameters)
        allocation = allocate_global_ranks([measured, proxy], target, rank_scale=False)
        self.assertEqual(allocation.ranks["measured"], 64)
        self.assertEqual(allocation.ranks["proxy"], 32)

    def test_allocator_rejects_duplicate_names(self) -> None:
        layers = [LayerBudget("same", 128, 128, 1.0), LayerBudget("same", 128, 128, 2.0)]
        with self.assertRaises(ValueError):
            allocate_global_ranks(layers, 2.0)


class QuantizerTests(unittest.TestCase):
    @staticmethod
    def _weight(seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return (
            rng.standard_normal((40, 6)) @ rng.standard_normal((6, 36)) / np.sqrt(6)
            + 0.08 * rng.standard_normal((40, 36))
        ).astype(np.float32)

    def test_factorized_matmul_matches_reconstruction(self) -> None:
        rng = np.random.default_rng(3)
        q = quantize_matrix(
            self._weight(),
            NQXConfig(
                rank=8,
                max_iters=32,
                min_iters=32,
                scale_iters=4,
                polish_iters=1,
                reclaim_packed_padding=False,
            ),
        )
        x = rng.standard_normal((2, 5, q.in_features)).astype(np.float32)
        expected = x @ q.reconstruct().T
        np.testing.assert_allclose(q.matmul(x), expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(q.matmul(x, prepared=False), expected, rtol=2e-5, atol=2e-5)
        self.assertGreater(q.runtime_cache_bytes, 0)
        q.clear_runtime_cache()
        self.assertEqual(q.runtime_cache_bytes, 0)

    def test_deterministic(self) -> None:
        config = NQXConfig(
            rank=8,
            max_iters=24,
            min_iters=24,
            scale_iters=3,
            polish_iters=0,
            reclaim_packed_padding=False,
            seed=19,
        )
        first = quantize_matrix(self._weight(), config)
        second = quantize_matrix(self._weight(), config)
        np.testing.assert_array_equal(first.u, second.u)
        np.testing.assert_array_equal(first.v, second.v)
        np.testing.assert_array_equal(first.scale_out, second.scale_out)
        np.testing.assert_array_equal(first.scale_in, second.scale_in)

    def test_exact_deployed_refit_does_not_regress_paper_style_candidate(self) -> None:
        weight = self._weight(4)
        baseline = paper_style_baseline(weight, rank=8, iterations=40, seed=0)
        enhanced = quantize_matrix(
            weight,
            NQXConfig(
                rank=8,
                max_iters=40,
                min_iters=40,
                patience=999,
                scale_iters=6,
                polish_iters=1,
                reclaim_packed_padding=False,
                seed=0,
            ),
        )
        self.assertLessEqual(enhanced.diagnostics.deployed_error, baseline.diagnostics.deployed_error + 1e-10)

    def test_balanced_profile_has_rank_scale(self) -> None:
        enhanced = quantize_matrix(
            self._weight(2),
            NQXConfig(
                rank=8,
                max_iters=24,
                min_iters=24,
                scale_iters=4,
                polish_iters=1,
                reclaim_packed_padding=False,
                rank_scale=True,
            ),
        )
        self.assertIsNotNone(enhanced.rank_scale)
        self.assertTrue(np.all(np.isfinite(enhanced.rank_scale)))

    def test_packed_padding_is_reclaimed_without_factor_cost(self) -> None:
        weight = self._weight(8)
        compact = quantize_matrix(
            weight,
            NQXConfig(
                rank=8,
                max_iters=32,
                min_iters=32,
                scale_iters=4,
                polish_iters=1,
                reclaim_packed_padding=False,
            ),
        )
        reclaimed = quantize_matrix(
            weight,
            NQXConfig(rank=8, max_iters=32, min_iters=32, scale_iters=4, polish_iters=1),
        )
        self.assertEqual(reclaim_packed_rank(8, 36), 32)
        self.assertEqual(reclaimed.rank, 32)
        self.assertEqual(compact.storage_bits(), reclaimed.storage_bits())
        self.assertLessEqual(reclaimed.diagnostics.deployed_error, compact.diagnostics.deployed_error)

    def test_candidate_selection_and_storage_projection_are_monotone(self) -> None:
        weight = self._weight(11)
        common = dict(
            rank=8,
            max_iters=36,
            min_iters=36,
            scale_iters=4,
            polish_iters=1,
            reclaim_packed_padding=False,
            storage_aware=True,
            seed=5,
        )
        final_only = quantize_matrix(weight, NQXConfig(**common, candidate_selection=False))
        selected = quantize_matrix(weight, NQXConfig(**common, candidate_selection=True))
        self.assertLessEqual(
            selected.diagnostics.weighted_deployed_error,
            final_only.diagnostics.weighted_deployed_error + 1e-12,
        )
        for scale in (selected.scale_out, selected.scale_in):
            np.testing.assert_array_equal(scale, scale.astype(np.float16).astype(np.float32))
        self.assertAlmostEqual(
            selected.diagnostics.deployed_error,
            selected.diagnostics.serialized_deployed_error,
            places=12,
        )


class ArtifactTests(unittest.TestCase):
    def test_safe_artifact_round_trip(self) -> None:
        q = quantize_matrix(
            QuantizerTests._weight(5),
            NQXConfig(
                rank=8,
                max_iters=20,
                min_iters=20,
                scale_iters=3,
                polish_iters=0,
                reclaim_packed_padding=False,
                rank_scale=True,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_nqx(q, Path(directory) / "matrix.nqx")
            loaded = load_nqx(path)
            np.testing.assert_array_equal(loaded.u, q.u)
            np.testing.assert_array_equal(loaded.v, q.v)
            # FP16 is the actual stored scale precision.
            np.testing.assert_array_equal(loaded.scale_out, q.scale_out.astype(np.float16).astype(np.float32))
            np.testing.assert_array_equal(loaded.scale_in, q.scale_in.astype(np.float16).astype(np.float32))

    def test_artifact_bytes_are_deterministic(self) -> None:
        q = QuantizedMatrix(
            u=np.ones((4, 3), dtype=np.int8),
            v=np.ones((5, 3), dtype=np.int8),
            scale_out=np.ones(4),
            scale_in=np.ones(5),
        )
        with tempfile.TemporaryDirectory() as directory:
            first = save_nqx(q, Path(directory) / "first.nqx")
            second = save_nqx(q, Path(directory) / "second.nqx")
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_checksum_detects_tampering(self) -> None:
        q = QuantizedMatrix(
            u=np.ones((4, 3), dtype=np.int8),
            v=np.ones((5, 3), dtype=np.int8),
            scale_out=np.ones(4),
            scale_in=np.ones(5),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_nqx(q, Path(directory) / "matrix.nqx")
            modified = Path(directory) / "modified.nqx"
            with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(modified, "w") as target:
                for name in source.namelist():
                    payload = source.read(name)
                    if name == "tensors/scale_in.npy":
                        payload = payload[:-1] + bytes([payload[-1] ^ 1])
                    target.writestr(name, payload)
            with self.assertRaises(ValueError):
                load_nqx(modified)

    def test_unexpected_zip_member_is_rejected(self) -> None:
        q = QuantizedMatrix(
            u=np.ones((4, 3), dtype=np.int8),
            v=np.ones((5, 3), dtype=np.int8),
            scale_out=np.ones(4),
            scale_in=np.ones(5),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_nqx(q, Path(directory) / "matrix.nqx")
            modified = Path(directory) / "extra.nqx"
            with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(modified, "w") as target:
                for name in source.namelist():
                    target.writestr(name, source.read(name))
                target.writestr("unexpected.bin", b"not allowed")
            with self.assertRaises(ValueError):
                load_nqx(modified)


if __name__ == "__main__":
    unittest.main()
