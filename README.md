<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <a href="https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86">
    <img alt="Idiot Savant" src="https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86/resolve/main/idiotsavant.png" width="300">
  </a>
</p>

<h1 align="center">vllm-idiotsavant</h1>

<p align="center">
vLLM v0.29.0 con las optimizaciones para servir
<a href="https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86"><b>qwen3.8_27b_idiotSavant_sm_86</b></a>
en 2× RTX 3090 (sm_86), configurado solo con flags de vLLM.
</p>

---

## Para qué sirve (y para qué no)

Este fork existe para un único modelo:
**[BlairQ/qwen3.8_27b_idiotSavant_sm_86](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86)**
(Qwen3.8-27B sin censura en W4, con el residuo rotado) y su borrador
**[BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2)**.
No es un vLLM de uso general: con otro checkpoint, en el mejor caso no aporta nada; en el peor, da
basura. Las razones:

- **Los pesos viven en una base rotada.** El checkpoint se cuantizó con el residuo rotado por
  Hadamard (estilo QuaRot): `down_proj` espera su entrada multiplicada por un Hadamard de bloques de
  512, y `o_proj` / `out_proj` por uno de una cabeza (256 / 128). Las compuertas del GDN van dentro
  de `in_proj_qkvz`. Este fork aplica esas rotaciones en línea en el forward; un vLLM normal carga
  el modelo sin error y genera basura. Se activan solas a partir del `config.json` del checkpoint
  (`residual_rotation`), así que otro modelo no las recibe, pero el resto del fork asume sus formas.
- **Los kernels están hechos para esta combinación exacta**, todos en PTX para sm_86:
  - lineales W4A8 (pesos int4 × activaciones int8 por token) en Marlin con escalas int16;
  - decode de atención entero sobre KV `int8_per_token_head`, que con el residuo rotado da más
    calidad, no solo más capacidad;
  - Hadamards fusionados con la cuantización de activaciones;
  - GDN (Gated DeltaNet) del híbrido Qwen3.8.

  Las formas (cabezas, grupos, bloques, TP=2) son las de este modelo.
- **La especulación está ajustada a su borrador.** DFlash2 en árbol, con el vocabulario de
  candidatos y la escala del residual del borrador entrenados contra este modelo.
- **La cache y el scheduler están medidos con su carga real.** Retención rala de los checkpoints
  GDN en la prefix cache, offload de KV a RAM, admisión de prefills y chunk dinámico, ajustados para
  sesiones de agentes de 60–100k tokens en 2×24 GB.

La calidad (KL 0,0178 contra BF16, la mitad que el int4 habitual) y los benchmarks están en la
[tarjeta del modelo](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86) y en el
[informe técnico](https://huggingface.co/spaces/BlairQ/idiotSavant-sm86-report).

## Uso

```bash
docker build -f docker/Dockerfile.idiotsavant -t vllm-idiotsavant:latest .
cd docker/idiotsavant
docker compose -f docker-compose.qwen38-27b-sm86.yml up -d
```

La compose [`docker/idiotsavant/docker-compose.qwen38-27b-sm86.yml`](docker/idiotsavant/docker-compose.qwen38-27b-sm86.yml)
es la configuración de producción, comentada flag por flag. Las optimizaciones fijas están en el
código; las que dependen del checkpoint se leen de su `config.json`; las ajustables son flags de
vLLM (`--max-concurrent-prefills`, `--kv-admission-headroom`, `--limit-mm-per-turn`,
`"tree_drafting"` en `--speculative-config`, …). Para limpiar las caches JIT, con el servidor parado:

```bash
docker compose -f docker-compose.qwen38-27b-sm86.yml run --rm limpiar-caches
```

El inventario de cambios respecto de vLLM v0.29.0 está en [`docs/idiotsavant/`](docs/idiotsavant/).

## Origen y créditos

Integra en el árbol de vLLM los parches de
[blairq/genesis-vllm-patches](https://github.com/blairq/genesis-vllm-patches), derivado de Genesis
(Sandermage, Apache-2.0); ver [`vllm/idiotsavant/NOTICE`](vllm/idiotsavant/NOTICE). Con caches
equivalentes, la salida greedy es idéntica token por token a la del contenedor con los parches
Genesis. Todo lo demás es [vLLM](https://github.com/vllm-project/vllm), cuyo README sigue abajo.

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
