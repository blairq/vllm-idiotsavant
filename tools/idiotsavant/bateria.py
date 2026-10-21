"""Bateria funcional contra un servidor vllm-idiotsavant (o cualquier vLLM con el mismo modelo).

Prueba texto greedy (sirve para comparar dos servidores token por token), razonamiento, tool calls,
imagenes (incluido el limite por turno), cabeceras de sesion, un prompt de ~160k tokens frio y
desde la prefix cache, concurrencia y velocidad de decode. Escribe el resultado en un JSON.

    python3 tools/idiotsavant/bateria.py --url http://localhost:8000 --salida resultado.json
    python3 tools/idiotsavant/bateria.py --grupos det,vel     # solo algunos grupos
"""
import argparse, base64, json, os, struct, threading, time, urllib.error, urllib.request, zlib

TODOS = ["det", "chat", "tool", "img", "sesion", "largo", "conc", "vel"]
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--clave", default=os.environ.get("VLLM_API_KEY", "idiotsavantftw"))
ap.add_argument("--salida", default="bateria.json")
ap.add_argument("--grupos", default=",".join(TODOS), help="separados por coma: " + ",".join(TODOS))
args = ap.parse_args()
URL, CLAVE, SALIDA, GRUPOS = args.url.rstrip("/"), args.clave, args.salida, args.grupos.split(",")
R = {}


def post(ruta, cuerpo, cab=None, stream=False, timeout=1800):
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {CLAVE}"}
    h.update(cab or {})
    req = urllib.request.Request(URL + ruta, json.dumps(cuerpo).encode(), h)
    t0 = time.time()
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), time.time() - t0
    ttft, partes, uso = None, [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for linea in r:
            linea = linea.decode().strip()
            if not linea.startswith("data: ") or linea == "data: [DONE]":
                continue
            d = json.loads(linea[6:])
            if d.get("usage"):
                uso = d["usage"]
            for c in d.get("choices", []):
                t = c.get("text") or (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content") or (c.get("delta") or {}).get("reasoning") or ""
                if t:
                    ttft = ttft or time.time() - t0
                    partes.append(t)
    return {"texto": "".join(partes), "uso": uso, "ttft": ttft}, time.time() - t0


def png(w=256, h=256):
    filas = b"".join(b"\x00" + b"".join(bytes((x * 255 // w if (x // 32 + y // 32) % 2 else 20, y * 255 // h, 200 if x < w // 2 else 30))
                                        for x in range(w)) for y in range(h))
    ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(filas)) + ch(b"IEND", b"")


def correr(nombre, f):
    if nombre not in GRUPOS:
        return
    try:
        R[nombre] = f()
        print(f"[ok] {nombre}: {json.dumps(R[nombre], ensure_ascii=False)[:600]}", flush=True)
    except Exception as e:
        R[nombre] = {"error": f"{type(e).__name__}: {e}", "cuerpo": getattr(e, "read", lambda: b"")().decode()[:2000]}
        print(f"[FALLA] {nombre}: {R[nombre]}", flush=True)


PROMPTS = ["La capital de Francia es",
           "def quicksort(arr):\n    ",
           "Explicá en tres oraciones qué es la entropía en termodinámica.",
           "Translate to English: 'El perro que ladra no muerde, pero el gato que maúlla tampoco araña.'",
           "1, 1, 2, 3, 5, 8, 13,",
           "Escribí un haiku sobre un servidor con dos GPUs."]


def det():
    out = []
    for p in PROMPTS:
        d, t = post("/v1/completions", {"model": "qwen3.8", "prompt": p, "max_tokens": 200, "temperature": 0, "seed": 0})
        out.append({"prompt": p, "texto": d["choices"][0]["text"], "tok": d["usage"]["completion_tokens"], "s": round(t, 2)})
    return out


def chat():
    d, t = post("/v1/chat/completions", {"model": "qwen3.8", "max_tokens": 2048, "temperature": 0,
                                         "messages": [{"role": "user", "content": "¿Cuánto es 17*23? Respondé solo el número."}]})
    m = d["choices"][0]["message"]
    return {"content": m.get("content"), "razon_len": len(m.get("reasoning_content") or m.get("reasoning") or ""), "fin": d["choices"][0]["finish_reason"], "s": round(t, 2)}


def tool():
    tools = [{"type": "function", "function": {"name": "clima", "description": "Clima actual de una ciudad",
                                               "parameters": {"type": "object", "properties": {"ciudad": {"type": "string"}}, "required": ["ciudad"]}}}]
    d, t = post("/v1/chat/completions", {"model": "qwen3.8", "max_tokens": 2048, "temperature": 0, "tools": tools, "tool_choice": "auto",
                                         "messages": [{"role": "user", "content": "¿Qué clima hace en Rosario? Usá la herramienta."}]})
    m = d["choices"][0]["message"]
    return {"tool_calls": [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])], "fin": d["choices"][0]["finish_reason"], "s": round(t, 2)}


def img():
    u = "data:image/png;base64," + base64.b64encode(png()).decode()
    pide = lambda n: {"model": "qwen3.8", "max_tokens": 1024, "temperature": 0,
                      "chat_template_kwargs": {"enable_thinking": False},
                      "messages": [{"role": "user", "content": [*([{"type": "image_url", "image_url": {"url": u}}] * n),
                                                                {"type": "text", "text": f"Describí brevemente {'la imagen' if n == 1 else 'las imágenes'}."}]}]}
    d1, t1 = post("/v1/chat/completions", pide(1))
    res = {"una": d1["choices"][0]["message"].get("content", "")[:200], "s": round(t1, 2)}
    try:   # 3 imagenes en un turno: limite por turno = 2 -> la tercera pasa a nota de texto (no 400)
        d3, _ = post("/v1/chat/completions", pide(3))
        res["tres"] = d3["choices"][0]["message"].get("content", "")[:200]
        res["tres_prompt_tok"] = d3["usage"]["prompt_tokens"]
    except urllib.error.HTTPError as e:
        res["tres"] = f"HTTP {e.code}: {e.read().decode()[:300]}"
    return res


def sesion():
    d, _ = post("/v1/chat/completions", {"model": "qwen3.8", "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                                         "messages": [{"role": "user", "content": "Hola"}]},
                cab={"X-Session-Id": "ses-prueba-0001", "X-Session-Agent": "coder", "X-Session-Root": "ses-prueba-0001"})
    return {"ok": bool(d["choices"][0]["message"].get("content"))}


def largo():
    base = "".join(f"Registro {i}: el sensor {i % 97} midió {i * 7 % 1000} unidades a las {i % 24}:{i % 60:02d}. " for i in range(6000))
    pide = lambda q: {"model": "qwen3.8", "prompt": base + f"\nPregunta: {q}\nRespuesta:", "max_tokens": 32, "temperature": 0, "stream": True,
                      "stream_options": {"include_usage": True}}
    a, ta = post("/v1/completions", pide("¿Qué midió el sensor 5 en el registro 5?"), stream=True)
    b, tb = post("/v1/completions", pide("¿Qué midió el sensor 9 en el registro 9?"), stream=True)
    return {"prompt_tok": (a["uso"] or {}).get("prompt_tokens"), "ttft_frio": round(a["ttft"], 2), "ttft_tibio": round(b["ttft"], 2),
            "cache_tok": ((b["uso"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens"), "resp": a["texto"][:120]}


def conc():
    res, hilos = [None] * 6, []

    def uno(i):
        d, t = post("/v1/completions", {"model": "qwen3.8", "prompt": PROMPTS[i % len(PROMPTS)] + " " * i, "max_tokens": 400, "temperature": 0.7, "seed": i})
        res[i] = (d["usage"]["completion_tokens"], round(t, 2))
    for i in range(6):
        hilos.append(threading.Thread(target=uno, args=(i,)))
        hilos[-1].start()
    for h in hilos:
        h.join()
    return {"pedidos": res, "tok_s_agregado": round(sum(r[0] for r in res) / max(r[1] for r in res), 1)}


def vel():
    out = {}
    for nombre, p in [("prosa", "Escribí un cuento largo sobre un faro en la Patagonia."), ("codigo", "Escribí en Python un servidor HTTP mínimo con asyncio, comentado.")]:
        d, t = post("/v1/chat/completions", {"model": "qwen3.8", "max_tokens": 1024, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                                             "messages": [{"role": "user", "content": p}], "stream": True, "stream_options": {"include_usage": True}}, stream=True)
        n = (d["uso"] or {}).get("completion_tokens", 0)
        out[nombre] = {"tok": n, "ttft": round(d["ttft"], 3), "decode_tok_s": round((n - 1) / (t - d["ttft"]), 1)}
    return out


for n, f in [("det", det), ("chat", chat), ("tool", tool), ("img", img), ("sesion", sesion), ("largo", largo), ("conc", conc), ("vel", vel)]:
    correr(n, f)
json.dump(R, open(SALIDA, "w"), ensure_ascii=False, indent=1)
