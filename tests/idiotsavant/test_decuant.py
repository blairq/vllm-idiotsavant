# SPDX-License-Identifier: Apache-2.0
"""sk18h_decuant_tile contra sk18h_decuant: paginas int8 de PN131 -> fp16 para el prefill.

El kernel por tiles (8 warps, V leida contigua y traspuesta en shared) tiene que dar exactamente los
mismos bits que el de siempre (1 warp por token, V leida byte a byte con paso BS).

    pytest -q tests/idiotsavant/test_decuant.py
    python3 tests/idiotsavant/test_decuant.py --medir          # us y GB/s efectivos por variante
"""
import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="necesita GPU")


def _escenario(ctx):
    from atencion_sintetica import Escenario, configurar
    configurar(max_model_len=max(ctx) + 4096)
    esc = Escenario(ctx=ctx, L=9)
    esc.escribir_contexto()
    esc.paso_decode()
    return esc


def _ids(esc):
    ids = torch.cat([esc.block_table[b, :n] for b, n in enumerate(esc.npag)])
    return ids.to(torch.int64)


def _decuantizar(esc, ids, tt):
    from atencion_sintetica import variante
    from vllm.idiotsavant import sk18_attn as A
    with variante(DECUANT_TT=tt):
        return A.decuantizar_kv(esc.impl, esc.kv_cache, ids).clone()


@pytest.mark.parametrize("tt", [16, 32])
def test_tile_bit_a_bit(tt):
    esc = _escenario([5000, 1800])
    ids = _ids(esc)
    viejo = _decuantizar(esc, ids, 0)
    nuevo = _decuantizar(esc, ids, tt)
    assert torch.equal(viejo.view(torch.int16), nuevo.view(torch.int16))


def test_tile_reconstruye_kv():
    """Sanidad: la KV decuantizada se parece a la original (error de la cuantizacion int8)."""
    esc = _escenario([3000])
    kv = _decuantizar(esc, _ids(esc), 32)
    from atencion_sintetica import BS
    n = 3000 + esc.L
    k = kv[:, 0].reshape(-1, *kv.shape[-2:])[:n].float()
    v = kv[:, 1].reshape(-1, *kv.shape[-2:])[:n].float()
    # K quedo rotada (Hadamard ortonormal): se compara la norma por token-cabeza
    rk = (k.norm(dim=-1) / esc.k[0][:n].float().norm(dim=-1) - 1).abs().max()
    rv = ((v - esc.v[0][:n].float()).norm(dim=-1) / esc.v[0][:n].float().norm(dim=-1)).max()
    assert rk < 0.05 and rv < 0.05, (float(rk), float(rv), BS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--medir", action="store_true")
    ap.add_argument("--paginas", type=int, default=64)
    args = ap.parse_args()
    if not args.medir:
        return pytest.main([__file__, "-q", "-s"])
    from atencion_sintetica import BS, NH, QD, medir, variante
    from vllm.idiotsavant import sk18_attn as A
    esc = _escenario([args.paginas * BS - 9])
    ids = _ids(esc)
    out = torch.empty((ids.shape[0], 2, BS, NH, QD), dtype=torch.float16, device="cuda")
    byts = ids.shape[0] * BS * NH * (516 + 2 * 2 * QD)       # leido + escrito
    for tt in (0, 16, 32):
        with variante(DECUANT_TT=tt):
            us = medir(lambda: A.decuantizar_kv(esc.impl, esc.kv_cache, ids, out=out))
        print(f"DECUANT_TT={tt:<3} {ids.shape[0]} paginas: {us:8.1f} us  {byts / us / 1e3:7.1f} GB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
