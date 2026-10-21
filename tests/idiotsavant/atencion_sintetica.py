# SPDX-License-Identifier: Apache-2.0
"""Escenario sintetico de una capa de atencion del Qwen3.8-27B (TP=2) sobre la KV de PN131.

Arma lo que el backend le pasa a ``sk18_attn`` (impl, layer, KV paginada int8_per_token_head y
la metadata del paso) sin levantar vLLM, para probar los kernels PTX del camino de atencion con
las formas de produccion: 12 cabezas Q y 2 KV por rank, head_dim 256, paginas de 880 tokens y
decode en arbol de 9 tokens (num_speculative_tokens 8).

La referencia es atencion en fp32 sobre las k/v originales (sin cuantizar). La rotacion Hadamard
de q y k es ortonormal, asi que no cambia q.k; la V no se rota (ROTV apagado).

Uso tipico::

    esc = Escenario(ctx=[30000, 12000], L=9)
    esc.escribir_contexto()
    q = esc.paso_decode()                 # escribe los L tokens nuevos y devuelve la query
    out = esc.forward(q)                  # camino de produccion (SK-30 si hay arbol)
    ref = esc.referencia(q)
"""
from __future__ import annotations

import contextlib
import math
import types

import numpy as np
import torch

from vllm.idiotsavant import opciones
from vllm.idiotsavant import sk18_attn as A

QD = 256
NH = 2          # cabezas KV por rank
G = 6           # cabezas Q por cabeza KV
BS = 880        # tokens por pagina (bloque del hibrido)


def configurar(max_num_seqs=11, max_model_len=131072, max_num_batched_tokens=8192, arbol=True,
               max_tok_decode=9):
    """Opciones del proceso como las fija vLLM en el servidor (sin VllmConfig)."""
    opciones._actual = opciones.Opciones(
        tree_drafting=arbol, draft_vocab_size=0, draft_residual_scale=0,
        max_num_seqs=max_num_seqs, max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens, prefix_cache_retention_interval=0,
        kv_metric_agents=())
    A._ARBOL = arbol
    A._MTD = max_tok_decode


@contextlib.contextmanager
def variante(**constantes):
    """Cambia constantes de ``sk18_attn`` (p. ej. SK30_PING=1) y recompila los kernels.

    Los PTX quedan en la cache de ptx_lab por defines, asi que volver atras no recompila.
    """
    viejas = {k: getattr(A, k) for k in constantes}
    try:
        for k, v in constantes.items():
            setattr(A, k, v)
        A._k.clear()
        yield
    finally:
        for k, v in viejas.items():
            setattr(A, k, v)
        A._k.clear()


def _kv_realista(n, gen, dev, outliers=4):
    """k/v fp16 [n, NH, 256] con escala por token variable y algunos canales outlier (como las
    activaciones reales despues de RoPE), que es lo que estresa la cuantizacion por token-cabeza."""
    x = torch.randn(n, NH, QD, generator=gen, device=dev)
    x *= torch.rand(n, NH, 1, generator=gen, device=dev) * 1.5 + 0.25
    canales = torch.randint(0, QD, (outliers,), generator=gen, device=dev)
    x[..., canales] *= 6.0
    return x.half()


class Escenario:
    def __init__(self, ctx, L=9, seed=0, dev="cuda", nombre="model.layers.7.self_attn.attn"):
        self.dev = torch.device(dev)
        self.ctx = list(ctx)
        self.B = len(self.ctx)
        self.L = L
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        from vllm.v1.kv_cache_interface import KVQuantMode
        self.impl = types.SimpleNamespace(
            num_heads=NH * G, num_kv_heads=NH, head_size=QD, scale=QD ** -0.5,
            _kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, alibi_slopes=None, sinks=None,
            sliding_window=(-1, -1), logits_soft_cap=0, chunk_lookback=-1, use_td=False)
        # _capas va por id(impl): un impl nuevo puede heredar el id de uno ya liberado y, con el, las
        # referencias de escala de otro escenario.
        A._capas.pop(id(self.impl), None)
        # Nombre fuera de _DIAG (capa 3): si no, forward compara contra el prefill en cada paso.
        self.layer = types.SimpleNamespace(
            layer_name=nombre, _k_scale=torch.ones((), device=self.dev),
            _v_scale=torch.ones((), device=self.dev))
        total = [c + L for c in self.ctx]
        self.npag = [-(-t // BS) for t in total]
        nb = sum(self.npag) + 4
        self.kv_cache = torch.zeros(nb, NH, BS, 520, dtype=torch.int8, device=self.dev)
        # paginas fisicas desordenadas, como despues de un rato de servir
        perm = torch.randperm(nb, generator=torch.Generator().manual_seed(seed)).tolist()
        maxp = max(self.npag)
        self.block_table = torch.zeros(self.B, maxp, dtype=torch.int32, device=self.dev)
        i = 0
        for b, n in enumerate(self.npag):
            self.block_table[b, :n] = torch.tensor(perm[i:i + n], dtype=torch.int32)
            i += n
        self.k = [_kv_realista(t, self.gen, self.dev) for t in total]
        self.v = [_kv_realista(t, self.gen, self.dev) for t in total]

    # ── escritura ─────────────────────────────────────────────────────────────
    def slots(self, b, ini, fin):
        pos = torch.arange(ini, fin, device=self.dev)
        return (self.block_table[b, pos // BS].to(torch.int64) * BS + pos % BS)

    def escribir(self, b, ini, fin):
        A.escribir(self.impl, self.layer, self.k[b][ini:fin], self.v[b][ini:fin], self.kv_cache,
                   self.slots(b, ini, fin))

    def escribir_contexto(self):
        """Contexto de cada pedido, por trozos como lo haria el chunked prefill."""
        for b, c in enumerate(self.ctx):
            for ini in range(0, c, 8192):
                self.escribir(b, ini, min(c, ini + 8192))
        torch.cuda.synchronize()

    # ── metadata ──────────────────────────────────────────────────────────────
    def metadata(self, qlens, seq_lens):
        qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
        seq = torch.tensor(seq_lens, dtype=torch.int32, device=self.dev)
        nt = int(qsl[-1])
        return types.SimpleNamespace(
            query_start_loc=torch.from_numpy(qsl).to(self.dev),
            isv_qsl_cpu=torch.from_numpy(qsl),
            seq_lens=seq, block_table=self.block_table, max_seq_len=int(max(seq_lens)),
            max_query_len=int(max(qlens)), num_actual_tokens=nt, causal=True,
            seq_threshold_3D=None, num_par_softmax_segments=None, softmax_segm_output=None,
            softmax_segm_max=None, softmax_segm_expsum=None, mm_prefix_range_tensor=None,
            rswa_prefix_lens=None, rswa_window=None)

    def query(self, n):
        return (torch.randn(n, NH * G, QD, generator=self.gen, device=self.dev) * 1.5).half()

    def paso_decode(self):
        """Escribe los L tokens nuevos de cada pedido y devuelve (query, metadata) del decode."""
        for b, c in enumerate(self.ctx):
            self.escribir(b, c, c + self.L)
        md = self.metadata([self.L] * self.B, [c + self.L for c in self.ctx])
        return self.query(self.B * self.L), md

    def paso_prefill(self, chunk):
        """Prefill de los ultimos ``chunk`` tokens de cada pedido sobre el resto como contexto.

        Reescribe la KV del chunk (idempotente) y devuelve (query, metadata, ini por pedido)."""
        inis = []
        for b, c in enumerate(self.ctx):
            ini = c + self.L - chunk
            self.escribir(b, ini, c + self.L)
            inis.append(ini)
        md = self.metadata([chunk] * self.B, [c + self.L for c in self.ctx])
        return self.query(self.B * chunk), md, inis

    # ── caminos ───────────────────────────────────────────────────────────────
    def forward(self, q, md):
        out = torch.zeros_like(q)
        A.forward(self.impl, self.layer, q, self.kv_cache, md, out)
        return out

    def referencia(self, q, md):
        """Atencion causal en fp32 sobre k/v sin cuantizar."""
        qsl = md.isv_qsl_cpu.tolist()
        out = torch.empty(q.shape, dtype=torch.float32, device=self.dev)
        for b in range(self.B):
            n = int(md.seq_lens[b])
            q0, q1 = qsl[b], qsl[b + 1]
            qb = q[q0:q1].float().view(q1 - q0, NH, G, QD)
            k = self.k[b][:n].float()
            v = self.v[b][:n].float()
            s = torch.einsum("thgd,nhd->thgn", qb, k) * self.impl.scale
            pos = torch.arange(n - (q1 - q0), n, device=self.dev)
            mask = torch.arange(n, device=self.dev)[None, :] > pos[:, None]
            s.masked_fill_(mask[:, None, None, :], -math.inf)
            p = torch.softmax(s, -1)
            out[q0:q1] = torch.einsum("thgn,nhd->thgd", p, v).reshape(q1 - q0, NH * G, QD)
        return out


def error_relativo(out, ref):
    """Maximo sobre filas (token, cabeza) de ||out - ref|| / ||ref||."""
    a = out.float().reshape(-1, QD)
    r = ref.float().reshape(-1, QD)
    return float(((a - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-6)).max())


def medir(fn, reps=50, calentar=5):
    """Mediana en microsegundos de ``fn()`` con eventos de CUDA."""
    for _ in range(calentar):
        fn()
    torch.cuda.synchronize()
    tiempos = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        tiempos.append(e0.elapsed_time(e1) * 1000)
    return float(np.median(tiempos))
