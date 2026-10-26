# SPDX-License-Identifier: Apache-2.0
"""Prefill de atencion entero (SK-30 aplanado, PREFILL_SK30) contra decuantizar + FlashInfer.

Los dos caminos leen la misma KV int8 de PN131. El de produccion la decuantiza a fp16 y corre
FlashInfer (HMMA con acumulador fp32); el nuevo corre SK-30 sobre pseudo-secuencias de 10 tokens
(Q.K en IMMA int8, softmax fp32, P.V int8 en dos planos), sin decuantizar. Cambia la cuenta: Q queda
cuantizada a int8 por fila, como en el decode. El test compara los dos contra atencion fp32 sobre las
k/v originales.

    pytest -q -s tests/idiotsavant/test_prefill_sk30.py
    python3 tests/idiotsavant/test_prefill_sk30.py --medir      # ms por camino, contexto y chunk

Decision: prender PREFILL_SK30 en vllm/idiotsavant/sk18_attn.py solo si (1) el error del camino
entero queda en el orden del de FlashInfer + la cuantizacion de Q (el test lo imprime), (2) es mas
rapido en los contextos que importan y (3) la KL contra BF16 y tools/idiotsavant/bateria.py (en
especial tool calls con contexto largo) no empeoran.
"""
import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="necesita GPU")


def _hay_flashinfer():
    try:
        import flashinfer  # noqa: F401
        return True
    except ImportError:
        return False


def _caso(ctx, chunk, con_ref=True, reps=0, seed=0):
    from atencion_sintetica import Escenario, configurar, error_relativo, medir, variante
    configurar(max_model_len=max(ctx) + 4096, max_num_batched_tokens=max(8192, chunk * len(ctx)))
    esc = Escenario(ctx=ctx, L=9, seed=seed)
    esc.escribir_contexto()
    q, md, _ = esc.paso_prefill(chunk)
    ref = esc.referencia(q, md) if con_ref else None
    res = {}
    for nombre, activo in (("flashinfer", False), ("entero", True)):
        with variante(PREFILL_SK30=activo):
            out = esc.forward(q, md)
            if activo:
                from vllm.idiotsavant import sk18_attn as A
                assert A._bufs_pf[esc.dev.index].plan is not None, "el prefill entero no corrio"
            err = error_relativo(out, ref) if con_ref else None
            us = medir(lambda: esc.forward(q, md), reps=reps, calentar=2) if reps else None
        res[nombre] = (out, err, us)
    return res


@pytest.mark.parametrize("ctx,chunk", [
    ([3000], 880),            # chunk tipico con decode en curso
    ([20000, 5000], 300),     # dos pedidos, contexto largo
    ([100], 109),             # prefill desde cero (contexto + L = chunk)
    ([4000, 900, 50], 25),    # chunk corto: pseudo-secuencias con relleno
])
def test_prefill_entero_contra_fp32(ctx, chunk):
    if not _hay_flashinfer():
        pytest.skip("sin flashinfer no hay camino de referencia de produccion")
    res = _caso(ctx, chunk)
    e_fi, e_en = res["flashinfer"][1], res["entero"][1]
    print(f"\nctx={ctx} chunk={chunk}: error relativo maximo  flashinfer {e_fi:.4f}  entero {e_en:.4f}")
    assert e_en < 0.08, e_en
    assert e_en < 3 * e_fi + 0.02, (e_en, e_fi)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--medir", action="store_true")
    ap.add_argument("--ctx", type=int, nargs="+", default=[8000, 32000, 64000, 100000])
    ap.add_argument("--chunk", type=int, nargs="+", default=[880, 2640])
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    if not args.medir:
        return pytest.main([__file__, "-q", "-s"])
    print(f"{'ctx':>7} {'chunk':>5} {'flashinfer ms':>14} {'entero ms':>10} {'x':>6}")
    for c in args.ctx:
        for ch in args.chunk:
            res = _caso([c], ch, con_ref=False, reps=args.reps)
            a, b = res["flashinfer"][2] / 1e3, res["entero"][2] / 1e3
            print(f"{c:>7} {ch:>5} {a:>14.2f} {b:>10.2f} {a / b:>6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
