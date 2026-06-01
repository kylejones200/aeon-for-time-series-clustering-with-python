"""DTW distance (O(n*m) dynamic programming)."""

from __future__ import annotations

import numpy as np


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    n, m = len(aa), len(bb)
    if n == 0 or m == 0:
        return float("inf")
    prev = np.full(m + 1, np.inf)
    curr = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[0] = np.inf
        for j in range(1, m + 1):
            cost = (aa[i - 1] - bb[j - 1]) ** 2
            curr[j] = cost + min(prev[j], prev[j - 1], curr[j - 1])
        prev, curr = curr, prev
    return float(np.sqrt(prev[m]))
