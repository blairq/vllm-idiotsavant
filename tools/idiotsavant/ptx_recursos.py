"""Registros, derrame, shared y bloques por SM de los kernels PTX de produccion en sm_86.

Compila (nvcc -ptx + ptxas -v) cada kernel con los defines con que lo construye el servidor, y estima
cuantos bloques entran por SM con los limites de una RTX 3090: 64K registros, 100 KB de shared, 16
bloques y 48 warps por SM. No hace falta GPU, solo el toolkit de CUDA (el mismo que usa ptx_lab).

    python3 tools/idiotsavant/ptx_recursos.py                 # tabla completa
    python3 tools/idiotsavant/ptx_recursos.py --solo sk30     # filtra por nombre
    python3 tools/idiotsavant/ptx_recursos.py --json r.json   # para comparar dos ramas

Para comparar una rama contra otra: correrlo en cada una con --json y diferenciar los archivos. Un
kernel que pasa a derramar (derrame > 0) o que pierde bloques por SM es una regresion aunque mida
igual en un microbenchmark aislado.
"""
import argparse
import json
import sys

from vllm.idiotsavant import sk18_attn as A
from vllm.idiotsavant.kernels.ptx_lab import Kernel

SM_REGS, SM_SMEM, SM_BLOQUES, SM_WARPS = 65536, 100 * 1024, 16, 48
SMEM_RESERVADA = 1024          # sm_86 reserva 1 KB de shared por bloque

# Formas del Qwen3.8-27B con TP=2 y arbol de 9 tokens (num_speculative_tokens 8).
H, HV, TM, T1 = 8, 24, 8, 9


def _produccion():
    qa, qb = A._coef()
    d = [f"-DQA={qa}", f"-DQB={qb}"]
    arb = ["-DARBOL=1"]
    # (archivo, entrada, defines, warps, shared dinamica)
    return [
        ("sk18h_escribir2.cu", "sk18h_escribir2", A._defs_escribir(), 1, 0),
        ("sk18h_prep2.cu", "sk18h_prep2", arb + ["-DQSF=1"], 1, 0),
        ("sk30_decode_1pasada.cu", "sk30_decode", A._defs_sk30(), 4 * A.SK30_NQ, A.sh_30()),
        ("sk30_decode_1pasada.cu", "sk30_union", [], 8, 0),
        ("sk18h_decuant.cu", "sk18h_decuant", A._defs_decuant(), A._warps_decuant(), 0),
        ("sk18h_batch2.cu", "sk18h_batch2", d + arb, 4, A.SH_H),
        ("sk18h_batch2.cu", "sk18h_batch2", d + arb + [f"-DNWARPS={A.NW8}"], A.NW8, A.sh_h(A.NW8)),
        ("sk18h_union4.cu", "sk18h_union4", d, 1, 0),
        ("sk18h_salida.cu", "sk18h_salida", [], 1, 0),
        ("gdn_arbol.cu", "gdn_arbol",
         [f"-DH={H}", f"-DHV={HV}", f"-DTM={TM}", f"-DTMAX={T1}", "-DHDT=0", "-DRPW=4"], 4, 0),
        ("pn122_cinta.cu", "pn122_cinta", [f"-DH={H}", f"-DHV={HV}", f"-DTM={TM}"], 8, 0),
        ("arbol_conv.cu", "arbol_conv",
         [f"-DTOK={T1}", "-DFPT=1", "-DNHILOS=32", "-DSILU=1", "-DESCRIBIR=1"], 1, 0),
        ("sk26_norma_q8.cu", "sk26_norma_q8", ["-DSUMA=3", "-DH=5120"], 20, 0),
        ("sk23_silu_had_q8.cu", "sk23_silu_had_q8", ["-DNW=17", "-DPORW=1"], 17, 0),
        ("sk25_cabeza_had_q8.cu", "sk25_cabeza_had_q8",
         ["-DMODO=0", "-DD=128", "-DNW=24", "-DPORW=1", "-DW32=0"], 24, 0),
        ("sk25_cabeza_had_q8.cu", "sk25_cabeza_had_q8",
         ["-DMODO=1", "-DD=256", "-DNW=12", "-DPORW=1", "-DW32=0"], 12, 0),
        ("sk24_ar_p2p.cu", "sk24_ar", ["-DHILOS=256"], 8, 0),
        ("sk24_ar_p2p.cu", "sk24_ar_i8", ["-DHILOS=256"], 8, 0),
        ("sk_fwht.cu", "sk_fwht128_f16", [], 4, 0),
    ]


def bloques_por_sm(regs: int, warps: int, smem: int) -> tuple[int, str]:
    """Bloques residentes por SM en sm_86 y el recurso que los limita."""
    regs_warp = -(-max(regs, 1) * 32 // 256) * 256          # granularidad de 256 registros por warp
    lim = {
        "registros": SM_REGS // (regs_warp * warps),
        "bloques": SM_BLOQUES,
        "warps": SM_WARPS // warps,
    }
    if smem:
        lim["shared"] = SM_SMEM // (smem + SMEM_RESERVADA)
    clave = min(lim, key=lim.get)
    return lim[clave], clave


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solo", default="", help="filtra por subcadena del nombre de la entrada")
    ap.add_argument("--json", default="", help="ademas guarda el resultado en este archivo")
    args = ap.parse_args()

    filas, malo = [], False
    for archivo, entrada, defs, warps, dyn in _produccion():
        if args.solo and args.solo not in entrada:
            continue
        r = Kernel(archivo, entrada, defs=defs, warps=warps).recursos()
        smem = r["smem"] + dyn
        bps, limite = bloques_por_sm(r["registros"], warps, smem)
        fila = dict(entrada=entrada, defs=" ".join(x for x in defs if not x.startswith("-DQ")),
                    warps=warps, **r, smem_total=smem, bloques_sm=bps, limite=limite,
                    ocupacion=round(100 * bps * warps / SM_WARPS))
        filas.append(fila)
        malo |= r["derrame_st"] > 0 or bps == 0

    print(f"{'entrada':<20} {'warps':>5} {'regs':>4} {'derr':>5} {'smem':>7} {'bl/SM':>5} "
          f"{'ocup':>5}  limite     defines")
    for f in filas:
        print(f"{f['entrada']:<20} {f['warps']:>5} {f['registros']:>4} {f['derrame_st']:>5} "
              f"{f['smem_total']:>7} {f['bloques_sm']:>5} {f['ocupacion']:>4}%  {f['limite']:<10} "
              f"{f['defs']}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(filas, fh, indent=1)
    if malo:
        print("\nAVISO: hay kernels con derrame a memoria local o que no entran en un SM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
