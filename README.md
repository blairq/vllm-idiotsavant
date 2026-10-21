<!-- markdownlint-disable MD001 MD041 -->
<p align="right"><b>EN</b> | <a href="README.es.md">ES</a></p>

<p align="center">
  <a href="https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86">
    <img alt="Idiot Savant" src="https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86/resolve/main/idiotsavant.png" width="300">
  </a>
</p>

<h1 align="center">vllm-idiotsavant</h1>

<p align="center">
vLLM v0.29.0 with the optimizations needed to serve
<a href="https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86"><b>qwen3.8_27b_idiotSavant_sm_86</b></a>
on 2× RTX 3090 (sm_86), configured with vLLM flags only.
</p>

---

## What it is for (and what it is not)

This fork exists for a single model:
**[BlairQ/qwen3.8_27b_idiotSavant_sm_86](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86)**
(uncensored Qwen3.8-27B in W4, with a rotated residual stream) and its drafter
**[BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2)**.
It is not a general-purpose vLLM: with any other checkpoint it adds nothing at best, and produces
garbage at worst. The reasons:

- **The weights live in a rotated basis.** The checkpoint was quantized with the residual rotated by
  Hadamard transforms (QuaRot style): `down_proj` expects its input multiplied by a 512-block
  Hadamard, and `o_proj` / `out_proj` by a one-head Hadamard (256 / 128). The GDN gates are packed
  inside `in_proj_qkvz`. This fork applies those rotations online in the forward pass; a stock vLLM
  loads the model without errors and generates garbage. They are switched on automatically from the
  checkpoint's `config.json` (`residual_rotation`), so another model does not get them, but the rest
  of the fork assumes this model's shapes.
- **The kernels are built for this exact combination**, all in PTX for sm_86:
  - W4A8 linears (int4 weights × per-token int8 activations) in Marlin with int16 scales;
  - integer attention decode over `int8_per_token_head` KV, which with the rotated residual gives
    more quality, not just more capacity;
  - Hadamards fused with activation quantization;
  - GDN (Gated DeltaNet) of the Qwen3.8 hybrid.

  The shapes (heads, groups, blocks, TP=2) are this model's.
- **Speculation is tuned to its drafter.** Tree-shaped DFlash2, with the candidate vocabulary and
  the drafter's residual scale fitted against this model.
- **Cache and scheduler are tuned on its real load.** Sparse retention of GDN checkpoints in the
  prefix cache, KV offload to RAM, prefill admission and dynamic chunking, tuned for 60–100k-token
  agent sessions on 2×24 GB.

Quality (KL 0.0178 against BF16, half that of the usual int4) and benchmarks are in the
[model card](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86) and in the
[technical report](https://huggingface.co/spaces/BlairQ/idiotSavant-sm86-report).

### Why "idiot savant"

At **18.6 GB**, the model beats the full **55 GB BF16** Qwen3.8-27B on some benchmarks of the
official card, and loses on others. Brilliant at some things, clumsy at others: hence the name.

| Benchmark | idiotSavant (18.6 GB) | Qwen3.8-27B BF16 (55 GB, card) | Δ | n |
|---|---|---|---|---|
| IFBench (prompt-level loose) | **83.0** ±4.3 | 79.5 | +3.5 | 300 |
| GPQA Diamond | **91.9** ±3.8 | 89.2 | +2.7 | 198 |
| LiveCodeBench v6 (2025-02..05) | **90.8** ±5.0 | 90.3 | +0.5 | 131 |
| HLE (fixed sample) | 23.6 ±3.7 (28.2 closing `</think>`¹) | **30.8** | −7.2 | 500 of 2500 |

± is the 95% interval. ¹ Grading the answers that never closed their reasoning; `--reasoning-force-close`
does that close at serving time. The BF16 column is Qwen's number for the official model, while this one
is quantized from an abliterated BF16, so the gap mixes abliteration, quantization and harness. Full
table (including vision, where it loses) and methodology in the
[model card](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86#benchmarks-against-the-bf16-model-card).

## What this fork adds to vLLM

Everything below is on by default unless it names a flag or key. Optimizations that only make
sense for this model are fixed in the code; the ones that depend on the checkpoint are read from
its `config.json`; the tunable ones are vLLM flags or config keys. The full list of changes against
vLLM v0.29.0 is in [`docs/idiotsavant/`](docs/idiotsavant/); the code is in
[`vllm/idiotsavant/`](vllm/idiotsavant/).

### Rotated-residual model

Activated only when the checkpoint's `config.json` declares `residual_rotation` (older checkpoints:
`genesis_rotacion`).

- **Hadamard before `down_proj`**: online 512-block Hadamard on the MLP intermediate, fused with
  SiluAndMul in one kernel (Triton; SK-23 on the int8 path). Key: `bloque_down`.
- **Per-head Hadamard at the input of `o_proj` (256) and GDN `out_proj` (128)**, fused with the
  int8 quantization of the input (SK-25). Key: `had_entrada`.
- **GDN gates inside `in_proj_qkvz`**: `in_proj_a` / `in_proj_b` are packed as extra shards of the
  `in_proj_qkvz` Marlin GEMM (Marlin cannot run them alone at TP=2). Key: `gdn_gates_in_qkvz`.
- **Drafter in the original basis**: the DFlash2 drafter keeps its residual unrotated; the shared
  embedding is unrotated on the way in and the hidden state rotated before the shared `lm_head`.
- **q/k rotation**: integer FWHT on q and k after RoPE, before the KV cache is quantized, to spread
  channel outliers.

### Quantized kernels (PTX, sm_86)

Compiled at runtime (`nvcc -ptx` → `ptxas` → `cuModuleLoadData`) and cached in
`~/.cache/vllm/idiotsavant/ptx`. No torch or Triton fallback.

- **Marlin W4A8 with signed int16 group scales**: vLLM's int8-activation Marlin reads the group
  scales as unsigned, which breaks checkpoints with negative scales. This is a fixed Marlin built as
  its own C++/CUDA extension (`idiotsavant_marlin::marlin_gemm_s16`), compiled for sm_86 in the
  Docker image. W4A8 is selected with the upstream `VLLM_MARLIN_INPUT_DTYPE=int8`.
- **Norm that writes int8 (SK-26)**: residual add + RMSNorm + per-token int8 quantization in one
  kernel, feeding Marlin W4A8 directly before `qkv_proj`, `in_proj_qkvz` and `gate_up_proj`.
- **Integer attention decode (SK-18)** over `--kv-cache-dtype int8_per_token_head`: own KV layout
  inside vLLM's allocation, Q·K in int8, integer streaming softmax, w·V in int8. Registered as a
  subclass of the `TRITON_ATTN` backend (`--attention-backend TRITON_ATTN`); steps with prefill
  dequantize the touched blocks and use vLLM's Triton kernel. CUDA-graph safe.
- **TRITON_ATTN tuned for Ampere at head_dim 256**: prefill tile sizes that fit sm_86 shared
  memory, and a decode path for multi-query (speculative) steps.
- **`lm_head` in int4** per group (128), W4A8, instead of fp16/fp8.

### Tensor parallelism (TP=2 over PCIe)

- **INT8 all-reduce for prefill**: row-parallel all-reduces of large batches (≥ 512 rows) are
  sent as int8 with per-group (64) scales.
- **P2P all-reduce for decode (SK-24)**: small all-reduces go by direct peer writes with a flag,
  double-buffered, graph-safe, instead of NCCL. TP=2 only, messages up to 1 MiB.
- **Vocab embedding all-reduce over P2P** (fp16, SK-24).
- **Per-rank logits in tree decode**: each rank sends only its local top candidates and its
  Gumbel-max winner instead of an AllGather of the full logits.
- **Custom all-reduce**: 32 MiB buffer for TP=2 and bypass during CUDA-graph capture.

### GDN (Gated DeltaNet)

- **Tape-based rollback for speculative decode**: instead of K full state copies per request,
  stores `(k, v, g, beta)` per token and replays the accepted rows; frees the KV blocks the copies
  used to take.
- **GDN metadata in one kernel** for tree decode steps (replaces ~35 torch ops per group per step).
- **No zero-fill/copy of the GDN output** in speculative steps, and GDN inputs as strided views of
  the `in_proj` output (no contiguous copies).
- Fused projection split (Triton), persistent gating buffers, preallocated scratch pools sized from
  the scheduler budget.

### Speculative decoding (DFlash2)

- **DFlash2 on v0.29.0**, including the fix for drafters that do not share target KV
  (vllm#53505).
- **Tree drafting**: an 8-node tree of candidates per step instead of a single chain, verified in
  one forward pass; GDN causal conv computed per path and compacted after acceptance, so the rest of
  vLLM sees a plain chain. Flag: `"tree_drafting": true` in `--speculative-config`.
- **Restricted draft vocabulary (FR-Spec)**: the drafter only proposes among the most frequent
  tokens of the real workload; the target still verifies with the full vocabulary. The frequency
  order ships in [`vllm/idiotsavant/datos/`](vllm/idiotsavant/datos/). Key: `"draft_vocab_size"`.
- **Drafter residual scale**: divides the drafter's residual stream at load time so it stays within
  fp16 (exact). Key: `"draft_residual_scale"`.
- **Drafter context projection in CUDA graphs**, and the drafter's dense linears on Marlin.
- **CUDA-graph sizes for speculation**: with speculative decoding, capture sizes are generated as
  multiples of K+1 (up to 4 requests) so no captured graph mixes query lengths; the
  `--cudagraph-capture-sizes` list is replaced.
- **`min_p` / `logit_bias` with speculation** are dropped with a warning instead of rejecting the
  request.

### Prefix cache and KV offloading

- **Reuse edge with EAGLE-family drafters**: keeps the GDN checkpoints at the prompt's reuse
  boundary, so sparse retention (upstream `--prefix-cache-retention-interval`) still hits with
  DFlash; plus fixes for caching the last Mamba page with EAGLE.
- **Block and group sizes for the hybrid**: sliding-window block size chosen from the backend's
  multiples, KV group size 8 (more blocks for the same memory).
- **GDN state in L2**: the RAM tier (`OffloadingConnector` with `TieringOffloadingSpec`) can serve
  GDN state with a speculative drafter; without this the external hit rate was 0.
- **L2 in RAM**: per-rank pinned registration (each rank pins only its slots), shared region
  unlinked after setup, recovery from a sticky CUDA error on host registration, and a sizing report
  at startup (L2 vs. L1 vs. `max_model_len`).
- **L3 disk tier** (`"type": "fs"` in `secondary_tiers`):
  - exclusive: blocks go to disk only when evicted from L2 (instead of write-through);
  - only persistent agents write to disk (`kv_transfer_params.agent` in a built-in list of primary
    agents, or explicit `kv_transfer_params.persist_disk`);
  - disk quota with oldest-first pruning: `max_size_gb`, `target_ratio`, `quota_check_secs`;
  - cleanup of sibling cache directories (other models/configs) unused for
    `orphan_max_age_days` days, every `orphan_check_secs`;
  - KV format fingerprint in the cache namespace, so changing the KV dtype, rotation or drafter
    scale never reads old blocks with a new meaning;
  - bounded deferral of disk → RAM promotions, so a request is not held for whole steps.
- **Per-agent offload limit**: when the client sends no `max_offload_tokens`, it is derived from the
  shared prefix measured for that agent.
- **Prefix-cache diagnostics per request**: shared hash blocks with the most similar recent request
  and tokens hit, to tell "prompt changed early" from "cache lost it" (numbers only, no content).

### Scheduling and admission

- **Dynamic prefill chunk**: `--long-prefill-token-threshold` when nobody is decoding,
  `--long-prefill-token-threshold-with-decode` when some request is decoding or has little prefill
  left.
- **Prefill serialization**: at most `--max-concurrent-prefills` requests prefilling at once,
  except prompts up to `--short-prompt-bypass-tokens` and requests whose prompt minus local
  prefix-cache hit fits in `--remaining-prefill-bypass-tokens`.
- **KV admission headroom**: a waiting request is not started if it would leave less than
  `--kv-admission-headroom` of GPU KV blocks free; never below `--kv-admission-min-concurrency`
  running requests.
- **Priority bypass**: at `max_num_seqs`, a higher-priority request can preempt a lower one.
  Priority comes from the request's `priority`, from `kv_transfer_params.priority`, or from a
  built-in agent → priority map.
- **Preemption cascade guard** in the scheduler.
- The admission controller can be inspected and adjusted at runtime (`/v1/kv-offload/pid`, below).

### Request and session tagging

- **Session headers**: `X-Session-Id`, `X-Session-Parent`, `X-Session-Root`, `X-Session-Agent` are
  copied into `kv_transfer_params` (`session_id`, `session_parent`, `session_root`,
  `session_agent`), up to 128 characters, for prefix diagnostics. The older `X-Genesis-Sesion`,
  `X-Genesis-Padre`, `X-Genesis-Raiz`, `X-Genesis-Agente` headers are still accepted.
- **Agent label**: `kv_transfer_params.agent` (older clients: `genesis_agent`) drives disk
  persistence, priority, the per-agent offload limit and metric labels. In metrics, names outside
  `kv_connector_extra_config.agents` collapse to `other` (bounded cardinality).

### HTTP API and administration

All routes below require the API key (`--api-key` / `VLLM_API_KEY`); the mutating ones require the
admin key.

- **Admin key**: `--admin-api-key` or `VLLM_ADMIN_API_KEY`. Without it, the regular API key is
  accepted on admin routes; with it, it is also accepted on regular routes.
- `GET /v1/kv-offload/requests`: ring buffer of recent chat completions.
- `POST /v1/kv-offload/reset` (admin): clears L1/L2/L3 caches, Prometheus metrics and the request
  history; query parameters `clear_l1`, `clear_l2`, `clear_l3`, `clear_metrics`, `clear_history`,
  `force`, `notify_clients` (open streams get a clean termination).
- `GET /v1/kv-offload/pid`: admission controller status. `POST /v1/kv-offload/pid` (admin): changes
  its parameters live (validated keys and ranges).

### Multimodal, reasoning and tool calls

- **`--limit-mm-per-turn N`**: limits new images per user turn instead of per request. Extra older
  items of a turn, and the oldest items beyond `--limit-mm-per-prompt`, are replaced by a text note
  instead of rejecting the request; earlier turns are left intact so the cached prefix is kept.
- **`--reasoning-force-close`**: if a chat completion with reasoning ends without `</think>`, the
  server appends it and continues so the model writes its answer (capped at 4096 tokens) instead of
  returning everything as reasoning. Requires `--reasoning-parser`.
- **Multi-turn reasoning fix** (vllm#41696): a `</think>` from a previous turn no longer marks the
  current reasoning as ended.
- **Tool-call reminder**: when a request has tools, `tool_choice` is `auto` or unset and the prompt
  is longer than 1000 characters, a reminder of the tool-call format is appended to the last user
  message.

### Observability and diagnostics

- **Prometheus metrics per KV tier** (`vllm:kv_tier_*`): bytes and operations by tier and
  direction, latency, lookups and hits, evictions by reason, occupancy and capacity, block residency
  and time to first reuse, disk I/O syscalls.
- **Admission metrics** (`vllm:pid_*`) and request metrics (`vllm:requests_by_agent_total`,
  `vllm:requests_by_priority_total`, `vllm:request_last_*`).
- **Startup analysis**: where the VRAM went, how much context really fits, known risks of the
  configuration.
- **Diagnostic dumps**: on a fatal exception (uncaught or logged), the traceback with summarized
  locals of every frame, `VLLM_*` variables (without keys), GPU memory and rank are written to
  `/root/.cache/vllm/idiotsavant/volcados` (up to 20 files).
- Warning at startup when the JIT cache disk has less than 10 GiB free.

### Upstream fixes carried

vllm#35288 (Mamba align bounds), vllm#37521 (warmup with the real number of speculative tokens),
vllm#35975 (no `inputs_embeds` buffer on text-only models), vllm#41696 (multi-turn reasoning),
vllm#53505 (DFlash KV).

### New flags and config keys

| Flag / key | Where | Default | Meaning |
|---|---|---|---|
| `--long-prefill-token-threshold-with-decode` | CLI (scheduler) | `0` (off) | Prefill chunk while some request is decoding |
| `--max-concurrent-prefills` | CLI (scheduler) | `0` (no limit) | Requests prefilling at the same time |
| `--short-prompt-bypass-tokens` | CLI (scheduler) | `0` (off) | Prompts up to this size skip the prefill limit |
| `--remaining-prefill-bypass-tokens` | CLI (scheduler) | `0` (off) | Requests with this much prefill left skip it |
| `--kv-admission-headroom` | CLI (scheduler) | `0.10` | Fraction of GPU KV blocks kept free on admission |
| `--kv-admission-min-concurrency` | CLI (scheduler) | `2` | Admission never limits below this many running |
| `--limit-mm-per-turn` | CLI (multimodal) | `0` (off) | New multimodal items per user turn |
| `--reasoning-force-close` | CLI (frontend) | off | Close unterminated reasoning and continue |
| `--admin-api-key` / `VLLM_ADMIN_API_KEY` | CLI / env | unset | Key for admin routes |
| `tree_drafting` | `--speculative-config` | `false` | Tree drafting (dflash only) |
| `draft_vocab_size` | `--speculative-config` | `0` (full) | Drafter candidate vocabulary (FR-Spec) |
| `draft_residual_scale` | `--speculative-config` | `0` (off) | Divisor for the drafter residual |
| `agents` | `kv_connector_extra_config` | built-in list | Allowlist of agent labels in metrics |
| `max_size_gb` | `fs` tier in `secondary_tiers` | `30.0` | Disk quota per rank |
| `target_ratio` | `fs` tier | `0.85` | Prune down to this fraction of the quota |
| `quota_check_secs` | `fs` tier | `60.0` | Seconds between quota checks |
| `orphan_max_age_days` | `fs` tier | `0.0` (off) | Delete sibling cache dirs unused this long |
| `orphan_check_secs` | `fs` tier | `3600.0` | Seconds between orphan sweeps |
| `residual_rotation` | model `config.json` | — | Declares the rotation (`bloque_down`, `had_entrada`, `gdn_gates_in_qkvz`) |
| `agent`, `persist_disk`, `priority` | request `kv_transfer_params` | — | Agent label, disk persistence, priority |
| `X-Session-Id`, `-Parent`, `-Root`, `-Agent` | HTTP headers | — | Session tagging |

## Usage

The image is published on GitHub Container Registry and the compose file pulls it by default:

```bash
cd docker/idiotsavant
docker compose -f docker-compose.qwen38-27b-sm86.yml up -d   # pulls ghcr.io/blairq/vllm-idiotsavant:v0.29.0-idiotsavant.1
```

To build it yourself (about 15 GB; it compiles Marlin S16 for sm_86) and use it instead:

```bash
docker build -f docker/Dockerfile.idiotsavant -t vllm-idiotsavant:latest .
VLLM_IDIOTSAVANT_IMAGE=vllm-idiotsavant:latest docker compose -f docker/idiotsavant/docker-compose.qwen38-27b-sm86.yml up -d
```

The compose file [`docker/idiotsavant/docker-compose.qwen38-27b-sm86.yml`](docker/idiotsavant/docker-compose.qwen38-27b-sm86.yml)
is the production configuration, commented flag by flag. It needs, besides the new flags,
`VLLM_MARLIN_INPUT_DTYPE=int8` (W4A8; without it the model runs W4A16), `--attention-backend
TRITON_ATTN` and `--kv-cache-dtype int8_per_token_head` (integer decode). Host paths, IP and the
external `lanbridge` network are those of the author's machine; adjust them. The first start with
empty caches also compiles the PTX kernels. To clear the JIT caches, with the server stopped:

```bash
docker compose -f docker-compose.qwen38-27b-sm86.yml run --rm limpiar-caches
```

## Origin and credits

This integrates into the vLLM tree the patches from
[blairq/genesis-vllm-patches](https://github.com/blairq/genesis-vllm-patches), derived from Genesis
(Sandermage, Apache-2.0); see [`vllm/idiotsavant/NOTICE`](vllm/idiotsavant/NOTICE). With equivalent
caches, greedy output is identical token by token to that of the container with the Genesis
patches. Everything else is [vLLM](https://github.com/vllm-project/vllm), whose README follows.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
