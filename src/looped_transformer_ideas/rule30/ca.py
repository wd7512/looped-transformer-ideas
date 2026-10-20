"""Rule 30 cellular automaton on a ring (periodic boundary)."""

from __future__ import annotations

import numpy as np


def step(row: np.ndarray) -> np.ndarray:
    """One Rule 30 update: new[i] = left XOR (center OR right)."""
    left = np.roll(row, 1)
    center = row
    right = np.roll(row, -1)
    return np.bitwise_xor(left, np.bitwise_or(center, right)).astype(np.int64)


def rollout(x0: np.ndarray, n: int) -> np.ndarray:
    """Evolve initial rows x0 (B, W) for n steps. Returns (n+1, B, W), row 0 = x0."""
    rows = [x0.astype(np.int64)]
    for _ in range(n):
        rows.append(step(rows[-1]))
    return np.stack(rows, axis=0)


def random_rows(rng: np.random.Generator, batch: int, width: int) -> np.ndarray:
    """Random binary initial rows."""
    return rng.integers(0, 2, size=(batch, width)).astype(np.int64)
