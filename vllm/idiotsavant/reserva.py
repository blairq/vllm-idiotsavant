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
_sonda_pendiente = False
_inicio = 0          # memoria asignada al terminar la reserva (base de los logs de la sonda)


def antes_del_perfil(runner) -> None:
    """Se llama al empezar ``profile_run``, en cada worker."""
    global _sonda_pendiente, _inicio
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
        _sonda_pendiente = True

    # El buffer que el top-k/top-p de Triton (upstream) cachea en el primer pedido con top_k/top_p:
    # num_sm filas x vocabulario. Se crea con la funcion real y logits de prueba.
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
    x = torch.cuda.memory_allocated(dev)
    filas = torch.cuda.get_device_properties(dev).multi_processor_count
    logits = torch.zeros(filas, mc.get_vocab_size(), dtype=torch.float32, device=dev)
    apply_top_k_top_p_triton(logits, torch.full((filas,), 20, dtype=torch.int32, device=dev),
                             torch.full((filas,), 0.95, dtype=torch.float32, device=dev))
    del logits
    partes["sampler"] = torch.cuda.memory_allocated(dev) - x

    _inicio = torch.cuda.memory_allocated(dev)
    log.info("[reserva] %.0f MiB creados antes del perfil de memoria: %s",
             (torch.cuda.memory_allocated(dev) - antes) / 2**20,
             ", ".join(f"{k} {v / 2**20:.0f} MiB" for k, v in partes.items()))


def soltar(nombre: str) -> None:
    """Suelta lo apartado para ``nombre`` justo antes de crear lo de verdad."""
    _apartado.pop(nombre, None)


def sonda_gdn(layer, mixed_qkv: torch.Tensor) -> None:
    """Prefill GDN del peor caso, una vez, dentro del forward de prueba del perfil.

    El perfil corre la capa GDN sin metadata y el warmup de upstream pasa solo 64 tokens de un
    pedido (calienta el autotune, no mide memoria): el pico del nucleo (conv, post-conv,
    chunk_gated_delta_rule con su h, w, u, v_new y o) nunca entraba en la cuenta. El 20-10, 13
    pedidos de alta prioridad prefilleando juntos (~7,9k tokens) pasaron el pico medido y dieron OOM.
    Aca corre ese mismo camino con todos los tokens del paso repartidos en max_num_seqs pedidos,
    mientras las activaciones del forward siguen vivas, asi el pico que mide vLLM lo incluye."""
    global _sonda_pendiente
    if not _sonda_pendiente:
        return
    _sonda_pendiente = False
    from vllm.idiotsavant import gdn_cinta, opciones
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fused_post_conv_prep

    H, HV, K, V = gdn_cinta.dims(layer)
    dev, dt = mixed_qkv.device, mixed_qkv.dtype
    T = mixed_qkv.shape[0]
    N = max(1, min(opciones.actual().max_num_seqs, T // 64))
    base = torch.cuda.memory_allocated(dev)
    ancho = 2 * H * K + HV * V
    # Lo que el camino real tiene vivo durante el chunk en un paso mixto (prefills + decodes spec):
    # la copia contigua de en_op (PN164), la del index_select de la parte no spec y la salida del conv.
    copia = torch.empty(T, ancho, device=dev, dtype=dt)
    no_spec = torch.empty(T, ancho, device=dev, dtype=dt)
    conv = torch.zeros(T, ancho, device=dev, dtype=dt)
    a = torch.zeros(T, HV, device=dev, dtype=dt)
    b = torch.zeros(T, HV, device=dev, dtype=dt)
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv, a=a, b=b, A_log=layer.A_log, dt_bias=layer.dt_bias, num_k_heads=H,
        head_k_dim=K, head_v_dim=V, apply_l2norm=True, output_g_exp=False)
    estado = torch.zeros(N, HV, V, K, device=dev, dtype=layer.get_state_dtype()[1])
    cu = torch.linspace(0, T, N + 1, device=dev).round().to(torch.int32)
    ci = co = None
    if layer.gdn_prefill_backend == "cutedsl":
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import prepare_metadata_cutedsl
        ci, co = prepare_metadata_cutedsl(cu, T)
    layer.chunk_gated_delta_rule(
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0), g=g.unsqueeze(0),
        beta=beta.unsqueeze(0), initial_state=estado, output_final_state=True, cu_seqlens=cu,
        chunk_indices=ci, chunk_offsets=co, use_qk_l2norm_in_kernel=False)
    pico = torch.cuda.max_memory_allocated(dev) - base
    del copia, no_spec, conv
    log.info("[reserva] sonda del prefill GDN: %d tokens en %d pedidos; vivo al entrar a la capa %.0f MiB, "
             "pico del perfil hasta aca %.0f MiB (sobre lo vivo: %.0f MiB)", T, N, (base - _inicio) / 2**20,
             (torch.cuda.max_memory_allocated(dev) - _inicio) / 2**20, pico / 2**20)
