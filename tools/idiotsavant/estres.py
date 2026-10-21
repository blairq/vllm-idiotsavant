"""Estres de memoria y admision: N pedidos largos simultaneos (por defecto 13, mas que --max-num-seqs 11).

Tres variantes, las que mostraron OOM antes de los arreglos del 20-10:

    python3 tools/idiotsavant/estres.py                        # prompts de 1-4k notas, prefijo compartido
    python3 tools/idiotsavant/estres.py --distintos            # contextos distintos de ~30-110k tokens
    python3 tools/idiotsavant/estres.py --distintos --agente primary --rondas 1
                                                               # alta prioridad: todos prefillean juntos

Imprime por ronda los errores y los tokens generados; cualquier error es una falla.
"""
import argparse
import json
import os
import threading
import time
import urllib.request

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--clave", default=os.environ.get("VLLM_API_KEY", "idiotsavantftw"))
ap.add_argument("--pedidos", type=int, default=13)
ap.add_argument("--rondas", type=int, default=3)
ap.add_argument("--distintos", action="store_true", help="contextos distintos y largos (sin prefijo comun)")
ap.add_argument("--agente", default=None,
                help="etiqueta de agente en kv_transfer_params (primary = alta prioridad, sin serializar)")
args = ap.parse_args()
URL = args.url.rstrip("/")


def prompt(i, ronda):
    if args.distintos:
        notas = [1500, 4000, 6000][i % 3]
        cab, base = f"Lote {ronda}-{i}. ", i * 7
    else:
        notas = [50, 400, 1500, 4000][i % 4]
        cab, base = "", 0
    return cab + "".join(f"Nota {j + base}: el item {j * 13 % 101} cuesta {j * 7 % 997} pesos. "
                         for j in range(notas))


def uno(i, ronda, res):
    cuerpo = {"model": "qwen3.8", "max_tokens": 300, "temperature": 0.6, "seed": i,
              "chat_template_kwargs": {"enable_thinking": i % 3 == 0},
              "messages": [{"role": "user", "content": prompt(i, ronda)
                            + f"\nResumí en dos líneas las notas (ronda {ronda}, pedido {i})."}]}
    if args.agente:
        cuerpo["kv_transfer_params"] = {"agent": args.agente}
    cab = {"Content-Type": "application/json", "Authorization": f"Bearer {args.clave}",
           "X-Session-Id": f"estres-{i}", "X-Session-Agent": args.agente or "coder"}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(cuerpo).encode(), cab)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
        res[i] = (d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"], round(time.time() - t0, 1),
                  d["choices"][0]["finish_reason"])
    except Exception as e:
        res[i] = f"ERROR {type(e).__name__}: {e}"


fallas = 0
for ronda in range(args.rondas):
    res, hilos = [None] * args.pedidos, []
    t0 = time.time()
    for i in range(args.pedidos):
        hilos.append(threading.Thread(target=uno, args=(i, ronda, res)))
        hilos[-1].start()
    for h in hilos:
        h.join()
    errores = [r for r in res if isinstance(r, str)]
    fallas += len(errores)
    gen = sum(r[1] for r in res if not isinstance(r, str))
    dt = time.time() - t0
    print(f"ronda {ronda}: {len(errores)} errores, {gen} tokens generados en {dt:.1f} s", flush=True)
    for r in res:
        print("   ", r)
raise SystemExit(1 if fallas else 0)
