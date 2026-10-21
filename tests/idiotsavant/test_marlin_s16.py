# SPDX-License-Identifier: Apache-2.0
"""Marlin W4A8 con escalas int16 (PN130): exactitud y ancho de banda con las formas del modelo.

El GEMM se llama como en produccion (``rot_down._gemm_int8``): activacion ya cuantizada a int8 por
token, escala por token por la global, peso int4 por grupo de 128 con escalas int16 con signo.
La referencia es el mismo producto en fp32 sobre el peso decuantizado y la activacion cuantizada,
asi que el error que queda es el de las escalas int16 y el de la acumulacion (deberia ser ~1e-3).

    pytest -q tests/idiotsavant/test_marlin_s16.py
    python3 tests/idiotsavant/test_marlin_s16.py --medir        # us y GB/s por forma y M

Para comparar los defines opcionales de build.py (MARLIN_L2_HINT, MARLIN_SHARED_JUSTA), recompilar
la extension en un proceso y medir en OTRO (el import de marlin_s16 carga la .so que haya):

    IDIOTSAVANT_MARLIN_DEFS="-DMARLIN_L2_HINT=1" python3 -c \\
        "from vllm.idiotsavant import marlin_s16; marlin_s16.construir()"
    pytest -q tests/idiotsavant/test_marlin_s16.py && \\
        python3 tests/idiotsavant/test_marlin_s16.py --medir > l2hint.txt

Y volver a la de produccion con IDIOTSAVANT_MARLIN_DEFS="" (construir() recompila al cambiar).
"""
import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="necesita GPU")

# (nombre, N, K) por rank con TP=2
FORMAS = [
    ("o_proj/out_proj", 5120, 5120),
    ("qkv / in_proj_qkvz", 7168, 5120),
    ("gate_up", 17408, 5120),
    ("down", 5120, 8704),
    ("lm_head", 124160, 5120),
]
MS = [1, 9, 18, 36, 81, 99, 880]


class Lineal:
    def __init__(self, n, k, seed=0, dev="cuda"):
        from vllm import _custom_ops as ops
        from vllm.idiotsavant.lm_head_int4 import G, cuantizar_int4_grupo
        from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
        g = torch.Generator(device=dev).manual_seed(seed)
        w = (torch.randn(n, k, generator=g, device=dev) * 0.02).half()
        qp, esc = cuantizar_int4_grupo(w)
        self.n, self.k = n, k
        # peso de referencia: lo que el kernel ve despues de cuantizar (sin las escalas int16)
        q = torch.stack([(qp >> (4 * i)) & 15 for i in range(8)], 1).reshape(k, n).float() - 8
        self.w_ref = (q.view(k // G, G, n) * esc.float().view(k // G, 1, n)).view(k, n)
        self.w = ops.gptq_marlin_repack(qp, torch.empty(0, dtype=torch.int32, device=dev), k, n, 4,
                                        is_a_8bit=True)
        em = mu.marlin_permute_scales(esc, size_k=k, size_n=n, group_size=G, is_a_8bit=True)
        self.s, self.igs = mu.marlin_act_int8_process_scales(em)
        self.ws = mu.marlin_make_workspace_new(torch.device(dev))
        self.vacio = torch.empty(0, dtype=torch.int32, device=dev)

    def cuantizar(self, x):
        s = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
        q = (x.float() / s).round().clamp(-127, 127).to(torch.int8)
        return q, s

    def gemm(self, q, s):
        from vllm.idiotsavant import marlin_s16
        from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
        from vllm.scalar_type import scalar_types
        m = q.shape[0]
        atom = mu.should_use_atomic_add_reduce(m=m, n=self.n, k=self.k, device=q.device,
                                               dtype=torch.float16)
        return marlin_s16.marlin_gemm(
            q, None, self.w, None, self.s, (s * self.igs).float(), None, self.vacio, self.vacio,
            self.vacio, self.ws, scalar_types.uint4b8, size_m=m, size_n=self.n, size_k=self.k,
            is_k_full=True, use_atomic_add=atom, use_fp32_reduce=True, is_zp_float=False)

    def bytes_peso(self):
        return self.n * self.k // 2 + self.n * (self.k // 128) * 2


def _requiere_s16():
    from vllm.idiotsavant import marlin_s16
    if not marlin_s16.ACTIVO_Y_CARGADO:
        pytest.skip("la extension idiotsavant_marlin_s16 no esta compilada")


@pytest.mark.parametrize("m", [1, 9, 36, 81, 99, 300])
@pytest.mark.parametrize("n,k", [(5120, 5120), (17408, 5120), (5120, 8704)])
def test_exactitud(n, k, m):
    _requiere_s16()
    lin = Lineal(n, k)
    x = torch.randn(m, k, device="cuda").half()
    q, s = lin.cuantizar(x)
    out = lin.gemm(q, s).float()
    ref = (q.float() * s) @ lin.w_ref
    err = ((out - ref).norm(dim=-1) / ref.norm(dim=-1)).max()
    assert err < 5e-3, float(err)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--medir", action="store_true")
    ap.add_argument("--m", type=int, nargs="+", default=MS)
    args = ap.parse_args()
    if not args.medir:
        return pytest.main([__file__, "-q"])
    from atencion_sintetica import medir
    print("IDIOTSAVANT_MARLIN_DEFS =", repr(os.environ.get("IDIOTSAVANT_MARLIN_DEFS", "")))
    print(f"{'forma':<20} {'N':>6} {'K':>5} {'M':>4} {'us':>8} {'GB/s peso':>9}")
    for nombre, n, k in FORMAS:
        lin = Lineal(n, k)
        for m in args.m:
            q, s = lin.cuantizar(torch.randn(m, k, device="cuda").half())
            us = medir(lambda: lin.gemm(q, s), reps=100)
            print(f"{nombre:<20} {n:>6} {k:>5} {m:>4} {us:>8.1f} {lin.bytes_peso() / us / 1e3:>9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
