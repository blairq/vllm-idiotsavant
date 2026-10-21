"""Velocidad de prefill y decode con N pedidos simultaneos (greedy, ignore_eos: la misma carga en dos imagenes).

    python3 tools/idiotsavant/velocidad.py --url http://localhost:8000 --salida vel.json
    python3 tools/idiotsavant/velocidad.py --largo     # 8 decodes simultaneos sobre ~27k de contexto
"""
import argparse, json, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--clave", default="idiotsavantftw")
ap.add_argument("--salida", default="vel.json")
ap.add_argument("--largo", action="store_true")
args = ap.parse_args()

# (nombre, pedidos, notas por prompt, tokens de salida)
ESCENARIOS = [("decode_8x30k", 8, 1500, 6000)] if "--largo" in sys.argv else [
    ("decode_8x2k", 8, 100, 512),
    ("prefill_8x30k", 8, 1500, 128),
    ("prefill_1x60k", 1, 3000, 64),
    ("mixto_8x8k", 8, 400, 384),
]


def prompt(sal, i, notas):
    return f"Lote {sal}-{i}. " + "".join(
        f"Nota {j}: el item {(j * 13 + i) % 101} cuesta {(j * 7 + i) % 997} pesos. " for j in range(notas)
    ) + "\nResumí las notas en detalle:"


def uno(sal, i, notas, salida, res):
    cuerpo = {"model": "qwen3.8", "prompt": prompt(sal, i, notas), "max_tokens": salida,
              "temperature": 0, "ignore_eos": True, "stream": True,
              "stream_options": {"include_usage": True}}
    req = urllib.request.Request(args.url + "/v1/completions", json.dumps(cuerpo).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {args.clave}"})
    t0 = time.time(); t1 = None; uso = None; marcas = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for linea in r:
            linea = linea.decode().strip()
            if not linea.startswith("data: ") or linea == "data: [DONE]":
                continue
            d = json.loads(linea[6:])
            if d.get("usage"):
                uso = d["usage"]
            if any(c.get("text") for c in d.get("choices", [])):
                t1 = t1 or time.time()
                marcas.append(time.time())
    res[i] = dict(t0=t0, t1=t1, t2=time.time(), pt=uso["prompt_tokens"], ct=uso["completion_tokens"], marcas=marcas)


R = {}
for nombre, n, notas, salida in ESCENARIOS:
    R[nombre] = []
    for ronda in range(2):
        sal = f"{nombre}-{ronda}"
        res, hilos = [None] * n, []
        for i in range(n):
            hilos.append(threading.Thread(target=uno, args=(sal, i, notas, salida, res)))
            hilos[-1].start()
        for h in hilos:
            h.join()
        ini = min(r["t0"] for r in res)
        ult_t1 = max(r["t1"] for r in res)
        fin = max(r["t2"] for r in res)
        pt = sum(r["pt"] for r in res); ct = sum(r["ct"] for r in res)
        # decode: desde que todos terminaron el prefill hasta el final, tokens generados en ese tramo aprox.
        m = dict(
            prompt_tok=pt, out_tok=ct, total_s=round(fin - ini, 2),
            prefill_tok_s=round(pt / (ult_t1 - ini), 1),           # prompts / hasta el ultimo primer token
            ttft_medio=round(sum(r["t1"] - r["t0"] for r in res) / n, 2),
            ttft_max=round(ult_t1 - ini, 2),
            decode_tok_s_pedido=round(sum((r["ct"] - 1) / (r["t2"] - r["t1"]) for r in res) / n, 1),
            decode_tok_s_agregado=round(ct / (fin - ini), 1),
        )
        # decode en regimen: chunks recibidos con todos ya decodificando, hasta que termina el primero
        fin1 = min(r["t2"] for r in res)
        if fin1 > ult_t1 + 1:
            tok_por_chunk = ct / sum(len(r["marcas"]) for r in res)
            chunks = sum(1 for r in res for t in r["marcas"] if ult_t1 < t <= fin1)
            m["decode_regimen_tok_s"] = round(chunks * tok_por_chunk / (fin1 - ult_t1), 1)
        R[nombre].append(m)
        print(nombre, ronda, json.dumps(m), flush=True)
json.dump(R, open(args.salida, "w"), indent=1)
