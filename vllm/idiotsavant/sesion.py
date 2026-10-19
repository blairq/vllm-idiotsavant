# SPDX-License-Identifier: Apache-2.0
"""PN169 — conversacion de cada pedido, desde las cabeceras que pone el plugin de sesion de opencode.

X-Session-Id / -Parent / -Root / -Agent se copian a ``kv_transfer_params`` (session_id, session_parent,
session_root, session_agent): ese campo llega intacto al Request del scheduler, donde lo leen el diagnostico
de prefijo y la politica de desalojo por conversacion. Un pedido sin cabeceras no cambia. Solo ids opacos,
nunca otras cabeceras. Se aceptan tambien las cabeceras X-Genesis-Sesion / -Padre / -Raiz / -Agente de las
versiones anteriores del plugin.
"""
from __future__ import annotations

import logging

log = logging.getLogger("idiotsavant.pn169")
# (cabecera, cabecera anterior, clave en kv_transfer_params)
CAMPOS = (("x-session-id", "x-genesis-sesion", "session_id"),
          ("x-session-parent", "x-genesis-padre", "session_parent"),
          ("x-session-root", "x-genesis-raiz", "session_root"),
          ("x-session-agent", "x-genesis-agente", "session_agent"))
_vistas: set = set()


def agente(kv_transfer_params) -> str | None:
    """El agente que declara el cliente en ``kv_transfer_params``: ``agent``, o ``genesis_agent`` de los
    clientes anteriores."""
    if not isinstance(kv_transfer_params, dict):
        return None
    return kv_transfer_params.get("agent") or kv_transfer_params.get("genesis_agent")


def de_cabeceras(raw_request) -> dict:
    hd = getattr(raw_request, "headers", None) or {}
    res = {}
    for h, anterior, k in CAMPOS:
        v = hd.get(h) or hd.get(anterior)
        if v:
            res[k] = v[:128]
    return res


def etiquetar(request, raw_request) -> None:
    try:
        et = de_cabeceras(raw_request)
        if not et:
            return
        if et.get("session_id") and et["session_id"] not in _vistas:
            _vistas.add(et["session_id"])
            if len(_vistas) > 4096:
                _vistas.clear()
            log.warning("PN169 sesion nueva %s (padre %s, raiz %s, agente %s)", et["session_id"][-12:],
                        et.get("session_parent", "")[-12:] or "-", et.get("session_root", "")[-12:],
                        et.get("session_agent"))
        ktp = dict(getattr(request, "kv_transfer_params", None) or {})
        ktp.update(et)
        request.kv_transfer_params = ktp
    except Exception as e:                       # nunca rompe el pedido
        log.warning("PN169: no se pudo etiquetar (%s: %s)", type(e).__name__, e)


def de_request(req) -> dict:
    """Las etiquetas de un Request del scheduler ({} si no vino del plugin)."""
    ktp = getattr(req, "kv_transfer_params", None) or {}
    return {k: ktp[k] for _, _, k in CAMPOS if k in ktp}
