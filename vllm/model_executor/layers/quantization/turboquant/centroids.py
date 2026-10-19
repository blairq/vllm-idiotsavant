# [Genesis wiring marker: Genesis PN57 TQ centroids disk cache (vllm#41418 inspiration)]
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lloyd-Max optimal scalar quantizer for TurboQuant.

After rotating a d-dimensional unit vector by a random orthogonal matrix,
each coordinate approximately follows N(0, 1/d) for d >= 64.
We solve the Lloyd-Max conditions to find optimal centroids.

Based on: turboquant-pytorch/lloyd_max.py (Zandieh et al.)
"""

import math
from functools import lru_cache

import torch


def _gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def _trapz(f, a: float, b: float, n: int = 200) -> float:
    """Trapezoidal numerical integration (replaces scipy.integrate.quad)."""
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def solve_lloyd_max(
    d: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d) distribution.

    Args:
        d: Vector dimension (determines variance = 1/d).
        bits: Number of quantization bits.
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance.

    Returns:
        centroids: Sorted tensor of 2^bits optimal centroids.
        boundaries: Sorted tensor of 2^bits - 1 decision boundaries.
    """
    n_levels = 2**bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    def pdf(x):
        return _gaussian_pdf(x, sigma2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = _trapz(lambda x: x * pdf(x), a, b)
            den = _trapz(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])

        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


# [Genesis PN57 vllm#41418-inspired] disk-persistent cache for centroids.
# Lloyd-Max solver is fully deterministic given (d, bits) — same input
# always gives bit-identical output. Pre-baked tables in upstream PR
# #41418 are equivalent to disk cache populated lazily.
# [Audit A-02 fix 2026-05-05] All `os.*` calls live inside helper bodies
# with LOCAL `import os`. Module-level constants do NOT use `os` so we
# never require `import os` at target module top level.
_GENESIS_PN57_CACHE_PATH = None  # lazily resolved on first use
_GENESIS_PN57_DISK = None  # lazily loaded

def _genesis_pn57_path():
    import os as _os  # LOCAL — no top-level dependency
    global _GENESIS_PN57_CACHE_PATH
    if _GENESIS_PN57_CACHE_PATH is None:
        _GENESIS_PN57_CACHE_PATH = _os.path.expanduser(
            "~/.cache/genesis/turboquant_centroids.pkl"
        )
    return _GENESIS_PN57_CACHE_PATH

def _genesis_pn57_load_disk():
    import os as _os  # LOCAL
    global _GENESIS_PN57_DISK
    if _GENESIS_PN57_DISK is not None:
        return _GENESIS_PN57_DISK
    try:
        import pickle
        _path = _genesis_pn57_path()
        if _os.path.isfile(_path):
            with open(_path, "rb") as _f:
                _GENESIS_PN57_DISK = pickle.load(_f)
        else:
            _GENESIS_PN57_DISK = {}
    except Exception:
        _GENESIS_PN57_DISK = {}
    return _GENESIS_PN57_DISK

def _genesis_pn57_save_disk():
    try:
        import os as _os  # LOCAL
        import pickle, tempfile
        _path = _genesis_pn57_path()
        _os.makedirs(_os.path.dirname(_path), exist_ok=True)
        _dirn = _os.path.dirname(_path)
        with tempfile.NamedTemporaryFile("wb", dir=_dirn, delete=False) as _tf:
            pickle.dump(_GENESIS_PN57_DISK, _tf)
            _tmp = _tf.name
        _os.replace(_tmp, _path)
    except Exception:
        pass  # cache write failure non-fatal

@lru_cache(maxsize=32)
def get_centroids(d: int, bits: int) -> torch.Tensor:
    """Get precomputed Lloyd-Max centroids (in-memory + disk cached)."""
    _disk = _genesis_pn57_load_disk()
    _key = (int(d), int(bits))
    if _key in _disk:
        try:
            return torch.tensor(_disk[_key], dtype=torch.float32)
        except Exception:
            pass
    centroids, _ = solve_lloyd_max(d, bits)
    try:
        _disk[_key] = tuple(centroids.cpu().tolist())
        _genesis_pn57_save_disk()
    except Exception:
        pass
    return centroids
