# SPDX-License-Identifier: Apache-2.0
"""Memoria que estos cambios piden despues del perfil de vLLM, creada DENTRO del perfil.

vLLM dimensiona la KV con lo que queda despues de ``profile_run``, que corre sin atencion
(``skip_attn``): los transitorios de la atencion y del GDN los cubre solo la holgura de
``--gpu-memory-utilization``. Lo que se crea recien en la primera inferencia (buffers de SK-18, el
workspace de FlashInfer, el pool de salida del GDN) o al enlazar la KV (la cinta del GDN) quedaba
fuera de la cuenta y se comia esa holgura: el 20-10, un paso con 8 prefills sobre contexto cacheado
mas 3 decodes dio OOM por 42 MiB. Aca se crea todo eso antes de medir; lo que necesita la KV (la
cinta) se aparta con un tensor del mismo tamano que se suelta justo antes de crearla.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("idiotsavant.reserva")
_apartado: dict[str, torch.Tensor] = {}


def antes_del_perfil(runner) -> None:
    """Se llama al empezar ``profile_run``, en cada worker."""
    from vllm.idiotsavant import gdn_cinta, opciones, sk18_attn
    from vllm.idiotsavant.kernels.gdn_scratch_pool import GdnScratchPool

    dev = runner.device
    cfg = runner.vllm_config
    mc, pc = cfg.model_config, cfg.parallel_config
    op = opciones.actual()
    partes = {}
    antes = torch.cuda.memory_allocated(dev)

    if sk18_attn.dtype_activo(cfg.cache_config.cache_dtype) and sk18_attn._habilitado():
        partes["sk18"] = sk18_attn.reservar(dev, mc.get_num_kv_heads(pc), cfg.cache_config.block_size,
                                            mc.get_num_attention_heads(pc))

    capas = [m for m in runner.model.modules()
             if hasattr(m, "num_v_heads") and getattr(m, "num_spec", 0) > 0]
    if capas:
        m = capas[0]
        H, HV, K, V = gdn_cinta.dims(m)
        x = torch.cuda.memory_allocated(dev)
        GdnScratchPool.reservar(op.max_num_batched_tokens, HV, V, mc.dtype, dev)
        partes["gdn_salida"] = torch.cuda.memory_allocated(dev) - x
        n = sum(gdn_cinta.bytes_cinta(c, op.max_num_seqs) for c in capas)
        _apartado["cinta"] = torch.empty(n, dtype=torch.uint8, device=dev)
        partes["cinta"] = n

    log.info("[reserva] %.0f MiB creados antes del perfil de memoria: %s",
             (torch.cuda.memory_allocated(dev) - antes) / 2**20,
             ", ".join(f"{k} {v / 2**20:.0f} MiB" for k, v in partes.items()))


def soltar(nombre: str) -> None:
    """Suelta lo apartado para ``nombre`` justo antes de crear lo de verdad."""
    _apartado.pop(nombre, None)
