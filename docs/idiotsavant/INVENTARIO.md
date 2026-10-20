# Inventario de la integración (línea base)

Fuente: contenedor de producción `genesis-27b-idiotsavant`
(`genesis-vllm-patches/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml`,
vLLM v0.29.0, 2x RTX 3090), relevado el 2026-10-18.

## Qué llega realmente a `vllm serve`

El entrypoint de producción corre `python3 -m vllm._genesis.patches.apply_all` en un
proceso aparte que termina antes del `exec vllm serve`. Por eso, al servidor solo le
llegan dos cosas:

1. **Text-patches**: reescrituras de archivos de vLLM, que quedan en disco.
   Hay 52 archivos y +2082/−169 líneas. Están en el commit "Línea base: text-patches".
2. **Lo que carga `plugins_arranque` en cada proceso.** Se engancha por el text-patch
   de PN143 en `vllm/plugins/__init__.py`. Ahí entran el backend SK-18, el árbol del
   borrador y las escalas de DFlash2 (PN144).

Lo que el orquestador registra en memoria se pierde al salir de ese proceso.
`apply_all`, `dispatcher` y `wiring/` no forman parte del runtime.

## Números

| | |
|---|---|
| Parches reportados como aplicados en el último arranque | 83 (`parches_aplicados.txt`) |
| Archivos de vLLM modificados | 52 |
| Módulos de soporte (`vllm/_genesis`, hoy `vllm/idiotsavant`) | 81 `.py`, ~21k líneas |
| Kernels `.cu` usados | 28 de 63 (+ `sk18h_comun.cuh`) |
| Extensión C++/CUDA | Marlin S16 (PN130), hoy compilada en runtime |
| Kernels Triton en el camino activo | `gdn_cinta`, `arbol_conv`, `arbol_borrador`, `gdn_meta`, `logits_rango`, `silu_had`, `kernels/pn50_gdn_fused_proj` |

## Variables (`variables.tsv`)

| estado | cantidad | qué pasa en el fork |
|---|---|---|
| `compose` | 155 | Las fija el compose. Las `ENABLE=1` pasan a estar siempre activas o a autodetectarse; las `ENABLE=0` se borran con su código; los ajustes pasan a parámetros de vLLM. |
| `default` | 241 | Las lee el código activo, pero producción usa su valor por defecto. Pasan a ser constantes. |
| `huerfana` | 5 | Están en el compose y nadie las lee. Se borran. |

## Kernels

Hoy cada `.cu` se compila en runtime: `nvcc -ptx` → `ptxas` → `cuModuleLoadData`
(`kernels/ptx_lab.py`), con defines que dependen del modelo. El caché queda en
`~/.cache/genesis/ptx_lab`, que el compose no monta, así que **se recompila en cada
arranque**. Decisión: los kernels siguen en PTX y no hay fallback a torch ni a Triton.

## Notas para la limpieza

- `mamba_utils.py`, `gpu_model_runner.py` y `mamba_hybrid.py` evalúan
  `tl.constexpr(env)` al importar. Sin driver CUDA fallan, porque Triton se
  desactiva. Hay que moverlo a la configuración.
- PN81, PN90 y PN91 actúan sobre el tier L3 (disco), que está apagado desde el 06-10.
- El plugin de opencode usa los headers `X-Genesis-*` (PN169). Si se renombran,
  hay que cambiar también el plugin.
