# SPDX-License-Identifier: Apache-2.0
"""Opciones de vLLM que leen los modulos de idiotsavant, resueltas una vez por proceso.

vLLM manda su configuracion a cada proceso (frontend, EngineCore, workers), pero los modulos de
idiotsavant no la reciben como argumento. ``configurar`` se llama en cada proceso justo antes de
``load_general_plugins()``. Leer antes es un error de orden y lanza, en vez de seguir en silencio
con un valor que no es el pedido.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

log = logging.getLogger("idiotsavant.opciones")

# Clave del config.json con la que un checkpoint de residuo rotado declara su rotacion. La segunda es
# la de los checkpoints publicados hasta ahora.
CLAVES_ROTACION = ("residual_rotation", "genesis_rotacion")


def rotacion(config: Any) -> dict:
    """La declaracion de rotacion de un config (objeto de HF o dict), o {} si no es rotado."""
    for clave in CLAVES_ROTACION:
        rc = config.get(clave) if isinstance(config, dict) else getattr(config, clave, None)
        if rc:
            return dict(rc)
    return {}


@dataclass(frozen=True)
class Opciones:
    tree_drafting: bool
    draft_vocab_size: int
    draft_residual_scale: int
    max_num_seqs: int
    max_model_len: int
    max_num_batched_tokens: int
    prefix_cache_retention_interval: int
    kv_metric_agents: tuple[str, ...]
    # Del checkpoint target (ver rotacion()): 0 / {} / False si no lo declara.
    rot_down_block: int = 0
    had_entrada: dict = field(default_factory=dict)
    gdn_gates_in_qkvz: bool = False


_actual: Opciones | None = None


def configurar(vllm_config: VllmConfig) -> None:
    global _actual
    spec = vllm_config.speculative_config
    rc = rotacion(vllm_config.model_config.hf_config)
    _actual = Opciones(
        tree_drafting=bool(spec is not None and spec.tree_drafting),
        draft_vocab_size=spec.draft_vocab_size if spec is not None else 0,
        draft_residual_scale=spec.draft_residual_scale if spec is not None else 0,
        max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
        max_model_len=vllm_config.model_config.max_model_len,
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        prefix_cache_retention_interval=(
            vllm_config.cache_config.prefix_cache_retention_interval or 0
        ),
        kv_metric_agents=tuple(
            (kv.kv_connector_extra_config.get("agents") or ())
            if (kv := vllm_config.kv_transfer_config) is not None
            else ()
        ),
        rot_down_block=int(rc.get("bloque_down") or 0),
        had_entrada=dict(rc.get("had_entrada") or {}),
        # El checkpoint del 28-09 lo declara en "requiere"; uno nuevo puede usar la clave propia.
        gdn_gates_in_qkvz=bool(rc.get("gdn_gates_in_qkvz")) or "PN155" in str(rc.get("requiere", "")),
    )
    _al_configurar()


def _al_configurar() -> None:
    # Las lecturas perezosas se resuelven ya: si la primera fuera dentro de una traza de dynamo,
    # el import y la escritura del global quedarian adentro.
    from vllm.idiotsavant import gdn_cinta, sk18_attn

    gdn_cinta.arbol()
    sk18_attn._arbol()
    try:
        from vllm.idiotsavant import dflash2_escala

        dflash2_escala.enganchar()
    except Exception as e:  # noqa: BLE001
        log.error("[PN144] no se pudo escalar el residual del borrador DFlash2 (%s: %s). "
                  "En fp16 el borrador va a dar NaN y aceptar 0.", type(e).__name__, e)
    if _actual.tree_drafting:
        # Arbol de borrador de DFlash2: envuelve el borrador, el runner v2 y la conv de GDN en
        # este proceso. Idempotente.
        try:
            from vllm.idiotsavant import arbol_runner

            arbol_runner.instalar()
        except Exception as e:  # noqa: BLE001
            log.error("[ARBOL] no se pudo instalar: %s: %s", type(e).__name__, e)


def actual() -> Opciones:
    if _actual is None:
        raise RuntimeError(
            "vllm.idiotsavant.opciones se leyo antes de configurar(): algun modulo lee una "
            "opcion al importarse, antes de que el proceso tenga su VllmConfig"
        )
    return _actual
