# SPDX-License-Identifier: Apache-2.0
"""PN173 — tamano del chunk de prefill segun haya o no pedidos decodificando.

Medido 06-10 (prefill de 125k y, durante el, un turno con cache que genera 64 tokens):
    chunk   prefill solo   turno que decodifica
     880       83,7 s             16 s
    1760       76,3 s             31 s
    2640       74,0 s             49 s
El decode de los demas va al ritmo de los pasos, y el paso lo fija el chunk del prefill. Con nadie decodificando
conviene el chunk grande (prefill -3%); con alguien decodificando (o con poco prefill pendiente), el chico (su decode 2x mas rapido, +10% de prefill
solo mientras dura). Se fija una vez por paso en scheduler_config.long_prefill_token_threshold, que vLLM usa en los
chunks de running y waiting y en la alineacion de Mamba: los tres quedan coherentes. Multiplos de 880 (bloque).

Parametros: --long-prefill-token-threshold (sin decodes) y --long-prefill-token-threshold-with-decode (con
decodes; 0 lo apaga).
"""
from __future__ import annotations

# Tambien cuenta como interactivo un pedido al que le falta poco prefill (un turno con cache): sin esto su primer
# token salia a 15 s en vez de 12 (su propio prefill corto corria con pasos de 2640 del pedido grande).
POCO = 8192


def ajustar(scheduler) -> None:
    sc = scheduler.scheduler_config
    con_decode = sc.long_prefill_token_threshold_with_decode
    if not con_decode:
        return
    # El campo se reescribe en cada paso: el valor configurado se guarda la primera vez.
    sin_decode = getattr(scheduler, "_umbral_sin_decode", None)
    if sin_decode is None:
        sin_decode = scheduler._umbral_sin_decode = sc.long_prefill_token_threshold
    try:
        hay = any(r.num_prompt_tokens - r.num_computed_tokens <= POCO for r in scheduler.running)
        sc.long_prefill_token_threshold = con_decode if hay else sin_decode
    except Exception:
        pass
