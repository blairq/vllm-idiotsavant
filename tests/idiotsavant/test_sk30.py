# SPDX-License-Identifier: Apache-2.0
"""SK-30 (decode de atencion en arbol sobre la KV int8 de PN131): variantes contra produccion.

Que garantiza
-------------
* ``SK30_VCG`` y ``SK30_PING`` solo cambian como se cargan los tiles, no la cuenta: la salida tiene
  que ser bit a bit igual a la de la configuracion de referencia (VCG=0, PING=0).
* El decode entero queda cerca de la atencion fp32 sobre k/v sin cuantizar (error relativo por fila).
* ``SK30_PHL=0`` cambia la cuenta: el test solo informa su error, la decision es de calidad.

Como correrlo (en la placa, con la imagen del fork)::

    pytest -q tests/idiotsavant/test_sk30.py
    python3 tests/idiotsavant/test_sk30.py --medir            # tabla de tiempos por variante
    python3 tests/idiotsavant/test_sk30.py --medir --ctx 60000 --pedidos 1 4

Con el reloj fijo las diferencias chicas se leen mejor (``nvidia-smi -lgc 1400``).
"""
import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="necesita GPU")

VARIANTES = {
    "referencia": dict(SK30_VCG=0, SK30_PING=0, SK30_PHL=1),
    "produccion": dict(),
    "vcg": dict(SK30_VCG=1, SK30_PING=0, SK30_PHL=1),
    "ping": dict(SK30_VCG=1, SK30_PING=1, SK30_PHL=1),
    "phl0": dict(SK30_VCG=1, SK30_PING=0, SK30_PHL=0),
    "nq4": dict(SK30_VCG=1, SK30_PING=0, SK30_PHL=1, SK30_NQ=4),
}


def _correr(ctx, variante_, reps=0, seed=0):
    from atencion_sintetica import Escenario, configurar, error_relativo, medir, variante
    configurar(max_num_seqs=max(11, len(ctx)), max_model_len=max(ctx) + 4096)
    esc = Escenario(ctx=ctx, L=9, seed=seed)
    esc.escribir_contexto()
    q, md = esc.paso_decode()
    ref = esc.referencia(q, md)
    with variante(**VARIANTES[variante_]):
        out = esc.forward(q, md)
        us = medir(lambda: esc.forward(q, md), reps=reps) if reps else None
    return out, error_relativo(out, ref), us


@pytest.mark.parametrize("ctx", [[3000], [20000, 7001, 900]])
def test_variantes_de_carga_bit_a_bit(ctx):
    base, err, _ = _correr(ctx, "referencia")
    assert err < 0.08, err
    for nombre in ("produccion", "vcg", "ping"):
        out, _, _ = _correr(ctx, nombre)
        assert torch.equal(out, base), f"{nombre} no es bit a bit igual a la referencia"


def test_nq4_solo_difiere_en_redondeo():
    base, err, _ = _correr([20000, 3001], "referencia")
    out, err4, _ = _correr([20000, 3001], "nq4")
    dif = (out.float() - base.float()).abs().max()
    print(f"\nNQ=4: error relativo {err4:.4f} (NQ=2: {err:.4f}), max |dif| contra NQ=2 {float(dif):.2e}")
    assert err4 < err + 0.005, (err4, err)


def test_phl0_informa_error():
    _, err1, _ = _correr([20000], "referencia")
    _, err0, _ = _correr([20000], "phl0")
    print(f"\nerror relativo maximo: PHL=1 {err1:.4f}  PHL=0 {err0:.4f}")
    assert err0 < 0.15, err0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--medir", action="store_true")
    ap.add_argument("--ctx", type=int, nargs="+", default=[8000, 60000, 100000])
    ap.add_argument("--pedidos", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    if not args.medir:
        return pytest.main([__file__, "-q", "-s"])
    print(f"{'variante':<11} {'pedidos':>7} {'ctx':>7} {'us':>9} {'err_rel':>8}")
    for c in args.ctx:
        for b in args.pedidos:
            ctx = [c - 37 * i for i in range(b)]
            for nombre in VARIANTES:
                _, err, us = _correr(ctx, nombre, reps=args.reps)
                print(f"{nombre:<11} {b:>7} {c:>7} {us:>9.1f} {err:>8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
