#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import dtw_distance  # noqa: E402

def main() -> None:
    a = np.ascontiguousarray(np.sin(np.arange(200) * 0.1))
    b = np.ascontiguousarray(np.cos(np.arange(180) * 0.12))
    t0 = time.perf_counter()
    for _ in range(200):
        dtw_distance(a, b)
    py_s = time.perf_counter() - t0
    try:
        import aeon_for_time_series_clustering_with_python_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(a, b, 2000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(dtw_distance(a, b), rs.dtw_distance_py(a, b), rtol=1e-10)
    print("Correctness: OK")

if __name__ == "__main__":
    main()
