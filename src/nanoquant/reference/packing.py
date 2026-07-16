"""Portable row-major bit packing for binary factors.

Copyright 2026 Godson Johnson
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from math import ceil

import numpy as np


def pack_signs(signs: np.ndarray, *, word_bits: int = 32) -> np.ndarray:
    """Pack a two-dimensional {-1,+1} matrix into unsigned words.

    Bit ``0`` represents ``+1`` and bit ``1`` represents ``-1``.  Bits are
    little-endian within each word, and padding is canonicalized to ``+1``.
    """
    matrix = np.asarray(signs)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a two-dimensional matrix, got {matrix.ndim} dimensions.")
    if not np.all(np.isin(matrix, (-1, 1))):
        raise ValueError("The input must contain only -1 and +1.")
    if word_bits not in (8, 16, 32, 64):
        raise ValueError("word_bits must be one of 8, 16, 32, or 64")
    dtype = np.dtype(f"<u{word_bits // 8}")
    rows, columns = matrix.shape
    words = ceil(columns / word_bits)
    padded = np.ones((rows, words * word_bits), dtype=np.int8)
    padded[:, :columns] = matrix.astype(np.int8, copy=False)
    bits = (padded < 0).reshape(rows, words, word_bits).astype(dtype, copy=False)
    powers = np.left_shift(np.array(1, dtype=dtype), np.arange(word_bits, dtype=dtype))
    return np.sum(bits * powers[None, None, :], axis=2, dtype=dtype).astype(dtype, copy=False)


def unpack_signs(
    packed: np.ndarray,
    shape: tuple[int, int],
    *,
    word_bits: int = 32,
) -> np.ndarray:
    """Unpack words produced by :func:`pack_signs`."""
    words_array = np.asarray(packed)
    if words_array.ndim != 2:
        raise ValueError("packed must be two-dimensional")
    if word_bits not in (8, 16, 32, 64):
        raise ValueError("word_bits must be one of 8, 16, 32, or 64")
    rows, columns = (int(shape[0]), int(shape[1]))
    expected = (rows, ceil(columns / word_bits))
    if tuple(words_array.shape) != expected:
        raise ValueError(f"Expected packed shape {expected}, got {tuple(words_array.shape)}.")
    dtype = np.dtype(f"<u{word_bits // 8}")
    words_array = words_array.astype(dtype, copy=False)
    shifts = np.arange(word_bits, dtype=dtype)
    bits = np.bitwise_and(np.right_shift(words_array[:, :, None], shifts), 1)
    flat = bits.reshape(rows, -1)[:, :columns].astype(np.int8, copy=False)
    return (1 - 2 * flat).astype(np.int8, copy=False)


def packed_storage_bits(shape: tuple[int, int], *, word_bits: int = 32) -> int:
    rows, columns = shape
    return int(rows * ceil(columns / word_bits) * word_bits)

