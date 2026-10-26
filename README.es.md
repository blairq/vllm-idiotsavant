<!-- markdownlint-disable MD001 MD041 -->
<p align="right"><a href="README.md">EN</a> | <b>ES</b></p>

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

<p align="center"><b>¿Solo querés ponerlo a andar? <a href="#arranque-rápido">Ir al arranque rápido</a></b> (instalar, bajar, arrancar y probar, todo copiar y pegar).</p>

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
  candidatos y la escala del residual del borrador ajustados contra este modelo.
- **La cache y el scheduler están medidos con su carga real.** Retención rala de los checkpoints
  GDN en la prefix cache, offload de KV a RAM, admisión de prefills y chunk dinámico, ajustados para
  sesiones de agentes de 60–100k tokens en 2×24 GB.

La calidad (KL 0,0178 contra BF16, la mitad que el int4 habitual) y los benchmarks están en la
[tarjeta del modelo](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86) y en el
[informe técnico](https://huggingface.co/spaces/BlairQ/idiotSavant-sm86-report).

### Por qué "idiot savant"

Con **18,6 GB**, el modelo le gana al Qwen3.8-27B **BF16 completo de 55 GB** en algunos benchmarks de
la tarjeta oficial, y pierde en otros. Brillante para algunas cosas y torpe para otras: de ahí el nombre.

| Benchmark | idiotSavant (18,6 GB) | Qwen3.8-27B BF16 (55 GB, tarjeta) | Δ | n |
|---|---|---|---|---|
| IFBench (prompt-level loose) | **83,0** ±4,3 | 79,5 | +3,5 | 300 |
| GPQA Diamond | **91,9** ±3,8 | 89,2 | +2,7 | 198 |
| LiveCodeBench v6 (2025-02..05) | **90,8** ±5,0 | 90,3 | +0,5 | 131 |
| HLE (muestra fija) | 23,6 ±3,7 (28,2 cerrando `</think>`¹) | **30,8** | −7,2 | 500 de 2500 |

± es el intervalo del 95%. ¹ Corrigiendo las respuestas que nunca cerraron el razonamiento;
`--reasoning-force-close` hace ese cierre al servir. La columna BF16 es el número de Qwen para el modelo
oficial, y este se cuantizó desde un BF16 abliterado: la diferencia mezcla abliteración, cuantización y
harness. La tabla completa (con visión, donde pierde) y la metodología están en la
[tarjeta del modelo](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86#benchmarks-against-the-bf16-model-card).

## Qué le agrega este fork a vLLM

Todo lo que sigue está prendido por defecto, salvo que nombre un flag o una clave. Lo que solo tiene
sentido para este modelo está fijo en el código; lo que depende del checkpoint se lee de su
`config.json`; lo ajustable son flags o claves de configuración de vLLM. La lista completa de
cambios respecto de vLLM v0.29.0 está en [`docs/idiotsavant/`](docs/idiotsavant/); el código, en
[`vllm/idiotsavant/`](vllm/idiotsavant/).

### Modelo con residuo rotado

Se activa solo si el `config.json` del checkpoint declara `residual_rotation` (en checkpoints
anteriores, `genesis_rotacion`).

- **Hadamard antes de `down_proj`**: Hadamard de bloques de 512 en línea sobre el intermedio del
  MLP, fusionado con SiluAndMul en un kernel (Triton; SK-23 en el camino int8). Clave:
  `bloque_down`.
- **Hadamard por cabeza en la entrada de `o_proj` (256) y de `out_proj` del GDN (128)**, fusionado
  con la cuantización int8 de la entrada (SK-25). Clave: `had_entrada`.
- **Compuertas del GDN dentro de `in_proj_qkvz`**: `in_proj_a` / `in_proj_b` van como trozos extra
  del GEMM Marlin de `in_proj_qkvz` (Marlin no puede con ellas solas con TP=2). Clave:
  `gdn_gates_in_qkvz`.
- **Borrador en la base original**: el borrador DFlash2 deja su residuo sin rotar; el embedding
  compartido se desrota a la entrada y el estado oculto se rota antes del `lm_head` compartido.
- **Rotación de q/k**: FWHT entera sobre q y k después de RoPE, antes de cuantizar la KV, para
  repartir los outliers de canal.

### Kernels cuantizados (PTX, sm_86)

Se compilan en runtime (`nvcc -ptx` → `ptxas` → `cuModuleLoadData`) y quedan en
`~/.cache/vllm/idiotsavant/ptx`. Sin fallback a torch ni a Triton.

- **Marlin W4A8 con escalas de grupo int16 con signo**: el Marlin de vLLM con activaciones int8 lee
  las escalas de grupo sin signo, y eso rompe los checkpoints con escalas negativas. Acá va un
  Marlin corregido como extensión C++/CUDA propia (`idiotsavant_marlin::marlin_gemm_s16`),
  compilada para sm_86 en la imagen de Docker. W4A8 se elige con `VLLM_MARLIN_INPUT_DTYPE=int8`
  (de upstream).
- **Norma que escribe int8 (SK-26)**: suma del residuo + RMSNorm + cuantización int8 por token en
  un kernel, que alimenta directo a Marlin W4A8 antes de `qkv_proj`, `in_proj_qkvz` y
  `gate_up_proj`.
- **Decode de atención entero (SK-18)** sobre `--kv-cache-dtype int8_per_token_head`: layout propio
  de la KV dentro de la reserva de vLLM, Q·K en int8, softmax entero en streaming, w·V en int8. Se
  registra como subclase del backend `TRITON_ATTN` (`--attention-backend TRITON_ATTN`); los pasos
  con prefill decuantizan los bloques que tocan y usan el kernel Triton de vLLM. Apto para grafos
  CUDA.
- **TRITON_ATTN afinado para Ampere con head_dim 256**: tiles de prefill que entran en la memoria
  compartida de sm_86 y un camino de decode para pasos con varias queries (especulación).
- **`lm_head` en int4** por grupo (128), W4A8, en vez de fp16/fp8.

### Paralelismo de tensores (TP=2 sobre PCIe)

- **All-reduce INT8 en el prefill**: los all-reduce de las lineales row-parallel con lotes grandes
  (≥ 512 filas) viajan en int8 con escalas por grupo de 64.
- **All-reduce P2P en el decode (SK-24)**: los all-reduce chicos van por escritura directa en la
  otra placa con una bandera, con doble buffer y aptos para grafos, en vez de NCCL. Solo TP=2,
  mensajes de hasta 1 MiB.
- **All-reduce del embedding de vocabulario por P2P** (fp16, SK-24).
- **Logits por rango en el decode en árbol**: cada rango manda solo sus mejores candidatos locales y
  su ganador del Gumbel-max, en vez de un AllGather de los logits enteros.
- **Custom all-reduce**: buffer de 32 MiB para TP=2 y bypass durante la captura de grafos CUDA.

### GDN (Gated DeltaNet)

- **Rollback con cinta en la decodificación especulativa**: en vez de K copias del estado por
  pedido, guarda `(k, v, g, beta)` por token y reproduce las filas aceptadas; libera los bloques de
  KV que ocupaban las copias.
- **Metadata del GDN en un kernel** en los pasos de decode en árbol (reemplaza ~35 ops de torch por
  grupo y por paso).
- **Sin relleno de ceros ni copia de la salida del GDN** en los pasos especulativos, y entradas del
  GDN como vistas con stride de la salida de `in_proj` (sin copias contiguas).
- Separación fusionada de la proyección (Triton), buffers persistentes de las compuertas y pools de
  scratch prerreservados con el presupuesto del scheduler.

### Decodificación especulativa (DFlash2)

- **DFlash2 en v0.29.0**, con el arreglo para borradores que no comparten la KV del target
  (vllm#53505).
- **Borrador en árbol**: un árbol de 8 nodos de candidatos por paso en vez de una cadena,
  verificado en una sola pasada; la conv causal del GDN se calcula por camino y se compacta después
  de aceptar, así que el resto de vLLM ve una cadena común. Flag: `"tree_drafting": true` en
  `--speculative-config`.
- **Vocabulario recortado del borrador (FR-Spec)**: el borrador solo propone entre los tokens más
  frecuentes de la carga real; el target sigue verificando con el vocabulario completo. El orden de
  frecuencias viene en [`vllm/idiotsavant/datos/`](vllm/idiotsavant/datos/). Clave:
  `"draft_vocab_size"`.
- **Escala del residual del borrador**: divide el residual del borrador al cargarlo para que entre
  en fp16 (exacto). Clave: `"draft_residual_scale"`.
- **Corrector DCOCO del borrador** ([arXiv 2610.12570](https://arxiv.org/abs/2610.12570)): un MLP
  chico vuelve a puntuar las transiciones entre los candidatos de posiciones vecinas, con un estado
  de par y el embedding de cada candidato, y se suma a la tabla del selector antes de armar la
  cadena o el árbol. Necesita un borrador DFlash2-DCOCO
  (`BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco`, que se entrena en
  `entrenamiento/dflash2-dcoco/` de genesis-vllm-patches). Clave: `"dcoco"` (por defecto se usa si
  el checkpoint trae la cabeza; `true` la exige, `false` la descarta).
- **Proyección del contexto del borrador en grafos CUDA**, y las lineales densas del borrador en
  Marlin.
- **Tamaños de grafo CUDA para la especulación**: con decodificación especulativa, los tamaños de
  captura se generan como múltiplos de K+1 (hasta 4 pedidos) para que ningún grafo capturado mezcle
  largos de query; la lista de `--cudagraph-capture-sizes` se reemplaza.
- **`min_p` / `logit_bias` con especulación** se anulan con un aviso en vez de rechazar el pedido.

### Prefix cache y offload de KV

- **Borde de reuso con borradores de la familia EAGLE**: conserva los checkpoints GDN en el borde de
  reuso del prompt, así la retención rala (`--prefix-cache-retention-interval`, de upstream) sigue
  acertando con DFlash; más arreglos para cachear la última página de Mamba con EAGLE.
- **Tamaños de bloque y de grupo para el híbrido**: el bloque de la ventana deslizante se elige
  entre los múltiplos del backend, y el grupo de KV es de 8 (más bloques con la misma memoria).
- **Estado GDN en L2**: el tier de RAM (`OffloadingConnector` con `TieringOffloadingSpec`) puede
  servir el estado GDN con un borrador especulativo; sin esto el acierto externo era 0.
- **L2 en RAM**: registro pinneado por rango (cada rango fija solo sus slots), región compartida
  sin nombre después de armarla, recuperación de un error de CUDA pegajoso al registrar memoria del
  host, e informe de dimensionamiento al arrancar (L2 contra L1 y contra `max_model_len`).
- **Tier L3 en disco** (`"type": "fs"` en `secondary_tiers`):
  - exclusivo: un bloque baja a disco solo cuando lo desalojan de L2 (en vez de write-through);
  - solo escriben a disco los agentes persistentes (`kv_transfer_params.agent` en una lista fija de
    agentes primarios, o `kv_transfer_params.persist_disk` explícito);
  - cuota de disco con poda de lo más viejo: `max_size_gb`, `target_ratio`, `quota_check_secs`;
  - limpieza de directorios de cache vecinos (otros modelos o configuraciones) sin uso hace
    `orphan_max_age_days` días, cada `orphan_check_secs`;
  - huella del formato de la KV en el espacio de nombres de la cache, así cambiar el dtype de la
    KV, la rotación o la escala del borrador nunca lee bloques viejos con un significado nuevo;
  - diferimiento acotado de las promociones de disco a RAM, para que un pedido no quede parado
    pasos enteros.
- **Límite de offload por agente**: si el cliente no manda `max_offload_tokens`, sale del prefijo
  compartido medido para ese agente.
- **Diagnóstico de prefix cache por pedido**: bloques de hash compartidos con el pedido reciente
  más parecido y tokens acertados, para separar "el prompt cambió cerca del principio" de "la cache
  lo perdió" (solo números, nada del contenido).

### Scheduling y admisión

- **Chunk de prefill dinámico**: `--long-prefill-token-threshold` cuando nadie decodifica, y
  `--long-prefill-token-threshold-with-decode` cuando algún pedido decodifica o le queda poco
  prefill.
- **Serialización de prefills**: como mucho `--max-concurrent-prefills` pedidos en prefill a la
  vez, salvo los prompts de hasta `--short-prompt-bypass-tokens` y los pedidos cuyo prompt menos el
  acierto de la prefix cache local entra en `--remaining-prefill-bypass-tokens`.
- **Headroom de KV en la admisión**: un pedido en espera no arranca si dejaría libre menos de
  `--kv-admission-headroom` de los bloques de KV de la GPU; nunca por debajo de
  `--kv-admission-min-concurrency` pedidos corriendo.
- **Bypass por prioridad**: en `max_num_seqs`, un pedido de prioridad más alta puede desalojar a
  uno de prioridad más baja. La prioridad sale del `priority` del pedido, de
  `kv_transfer_params.priority` o de un mapa fijo agente → prioridad.
- **Guarda contra cascadas de preempción** en el scheduler.
- El controlador de admisión se puede consultar y ajustar en caliente (`/v1/kv-offload/pid`, más
  abajo).

### Etiquetado de pedidos y sesiones

- **Cabeceras de sesión**: `X-Session-Id`, `X-Session-Parent`, `X-Session-Root`, `X-Session-Agent`
  se copian a `kv_transfer_params` (`session_id`, `session_parent`, `session_root`,
  `session_agent`), hasta 128 caracteres, para el diagnóstico de prefijo. Se siguen aceptando las
  cabeceras anteriores `X-Genesis-Sesion`, `X-Genesis-Padre`, `X-Genesis-Raiz`, `X-Genesis-Agente`.
- **Etiqueta de agente**: `kv_transfer_params.agent` (clientes anteriores: `genesis_agent`) decide
  la persistencia en disco, la prioridad, el límite de offload por agente y las etiquetas de las
  métricas. En las métricas, los nombres fuera de `kv_connector_extra_config.agents` caen en
  `other` (cardinalidad acotada).

### API HTTP y administración

Todas las rutas de abajo piden la clave de la API (`--api-key` / `VLLM_API_KEY`); las que modifican
algo piden la clave de admin.

- **Clave de admin**: `--admin-api-key` o `VLLM_ADMIN_API_KEY`. Sin ella, las rutas de admin
  aceptan la clave común; con ella, también vale en las rutas comunes.
- `GET /v1/kv-offload/requests`: buffer circular de los últimos chat completions.
- `POST /v1/kv-offload/reset` (admin): vacía las caches L1/L2/L3, las métricas de Prometheus y el
  historial de pedidos; parámetros de query `clear_l1`, `clear_l2`, `clear_l3`, `clear_metrics`,
  `clear_history`, `force`, `notify_clients` (los streams abiertos reciben un cierre limpio).
- `GET /v1/kv-offload/pid`: estado del controlador de admisión. `POST /v1/kv-offload/pid` (admin):
  cambia sus parámetros en caliente (claves y rangos validados).

### Multimodal, razonamiento y llamadas a herramientas

- **`--limit-mm-per-turn N`**: limita las imágenes nuevas por turno del usuario en vez de por
  pedido. Las que sobran en un turno (las más viejas), y las más viejas de la conversación que
  pasen `--limit-mm-per-prompt`, se reemplazan por una nota de texto en vez de rechazar el pedido; los
  turnos anteriores quedan intactos, así se conserva el prefijo cacheado.
- **`--reasoning-force-close`**: si un chat completion con razonamiento termina sin `</think>`, el
  servidor lo agrega y sigue para que el modelo escriba la respuesta (hasta 4096 tokens), en vez de
  devolver todo como razonamiento. Requiere `--reasoning-parser`.
- **Arreglo del razonamiento multiturno** (vllm#41696): un `</think>` de un turno anterior ya no da
  por terminado el razonamiento actual.
- **Recordatorio de herramientas**: si un pedido trae herramientas, `tool_choice` es `auto` o no
  está, y el prompt pasa los 1000 caracteres, se agrega al último mensaje del usuario un
  recordatorio del formato de las llamadas a herramientas.

### Observabilidad y diagnóstico

- **Métricas de Prometheus por tier de KV** (`vllm:kv_tier_*`): bytes y operaciones por tier y
  dirección, latencia, búsquedas y aciertos, desalojos por motivo, ocupación y capacidad, vida de
  un bloque en el tier y tiempo hasta su primer reuso, syscalls de I/O de disco.
- **Métricas de admisión** (`vllm:pid_*`) y de pedidos (`vllm:requests_by_agent_total`,
  `vllm:requests_by_priority_total`, `vllm:request_last_*`).
- **Análisis de arranque**: a dónde se fue la VRAM, cuánto contexto entra de verdad, riesgos
  conocidos de la configuración.
- **Volcados de diagnóstico**: ante una excepción fatal (sin atrapar o logueada), se escriben el
  traceback con las variables locales resumidas de cada frame, las variables `VLLM_*` (sin claves),
  la memoria de GPU y el rank en `/root/.cache/vllm/idiotsavant/volcados` (hasta 20 archivos).
- Aviso al arrancar si al disco de las caches JIT le quedan menos de 10 GiB.

### Arreglos de upstream incorporados

vllm#35288 (límites del align de Mamba), vllm#37521 (warmup con la cantidad real de tokens
especulativos), vllm#35975 (sin buffer de `inputs_embeds` en modelos solo texto), vllm#41696
(razonamiento multiturno), vllm#53505 (KV de DFlash).

### Flags y claves nuevas

| Flag / clave | Dónde | Por defecto | Qué hace |
|---|---|---|---|
| `--long-prefill-token-threshold-with-decode` | CLI (scheduler) | `0` (apagado) | Chunk de prefill mientras algún pedido decodifica |
| `--max-concurrent-prefills` | CLI (scheduler) | `0` (sin límite) | Pedidos en prefill a la vez |
| `--short-prompt-bypass-tokens` | CLI (scheduler) | `0` (apagado) | Los prompts de hasta este tamaño no esperan |
| `--remaining-prefill-bypass-tokens` | CLI (scheduler) | `0` (apagado) | Tampoco los que tienen hasta esto de prefill pendiente |
| `--kv-admission-headroom` | CLI (scheduler) | `0.10` | Fracción de bloques de KV de la GPU que la admisión deja libre |
| `--kv-admission-min-concurrency` | CLI (scheduler) | `2` | La admisión nunca limita por debajo de estos pedidos |
| `--limit-mm-per-turn` | CLI (multimodal) | `0` (apagado) | Ítems multimodales nuevos por turno del usuario |
| `--reasoning-force-close` | CLI (frontend) | apagado | Cierra el razonamiento sin terminar y sigue |
| `--admin-api-key` / `VLLM_ADMIN_API_KEY` | CLI / entorno | sin definir | Clave de las rutas de admin |
| `tree_drafting` | `--speculative-config` | `false` | Borrador en árbol (solo dflash) |
| `draft_vocab_size` | `--speculative-config` | `0` (completo) | Vocabulario de candidatos del borrador (FR-Spec) |
| `draft_residual_scale` | `--speculative-config` | `0` (apagado) | Divisor del residual del borrador |
| `dcoco` | `--speculative-config` | sin definir (si el checkpoint la trae) | Corrector DCOCO del borrador DFlash2: `true` lo exige, `false` lo descarta |
| `agents` | `kv_connector_extra_config` | lista fija | Allowlist de etiquetas de agente en las métricas |
| `max_size_gb` | tier `fs` en `secondary_tiers` | `30.0` | Cuota de disco por rango |
| `target_ratio` | tier `fs` | `0.85` | Poda hasta esta fracción de la cuota |
| `quota_check_secs` | tier `fs` | `60.0` | Segundos entre chequeos de cuota |
| `orphan_max_age_days` | tier `fs` | `0.0` (apagado) | Borra directorios de cache vecinos sin uso hace tanto |
| `orphan_check_secs` | tier `fs` | `3600.0` | Segundos entre barridos de huérfanos |
| `residual_rotation` | `config.json` del modelo | — | Declara la rotación (`bloque_down`, `had_entrada`, `gdn_gates_in_qkvz`) |
| `agent`, `persist_disk`, `priority` | `kv_transfer_params` del pedido | — | Etiqueta de agente, persistencia en disco, prioridad |
| `X-Session-Id`, `-Parent`, `-Root`, `-Agent` | cabeceras HTTP | — | Etiquetado de sesión |

## Arranque rápido

**Requisitos:** Linux x86-64 con **2× RTX 3090** (24 GB, sm_86: los kernels están hechos solo para esta
placa), driver de NVIDIA **580 o más nuevo** (CUDA 13; se recomienda el de P2P de abajo, 610.57.04), **32 GB de RAM** y **~60 GB libres de disco**
(imagen ~15 GB, modelo y borrador ~20 GB, caches). Los comandos son para Ubuntu/Debian; se copian y
pegan en orden.

**1. Driver de NVIDIA con P2P** (muy recomendado). Las dos placas intercambian sus resultados parciales en
cada capa; con copias GPU a GPU (P2P) eso va directo por PCIe. El driver estándar no habilita P2P en placas
GeForce: sin él el servidor igual funciona (el all-reduce P2P se apaga solo y NCCL pasa por la memoria del
host), pero el decode es más lento. El driver P2P es el módulo abierto de NVIDIA con el parche de
[aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules).
En el BIOS, activar *Above 4G Decoding* y *Resizable BAR* (y desactivar ACS si la placa madre lo ofrece). Después:

```bash
# IOMMU en modo passthrough (AMD; en Intel, intel_iommu=on en vez de amd_iommu=on)
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/&amd_iommu=on iommu=pt /' /etc/default/grub && sudo update-grub
# Reemplazar el driver de la distribución: librerías oficiales 610.57.04, sin sus módulos de kernel
sudo apt-get remove --purge -y '^nvidia-.*' '^libnvidia-.*' '^xserver-xorg-video-nvidia.*'
sudo apt-get install -y build-essential dkms git linux-headers-$(uname -r)
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/610.57.04/NVIDIA-Linux-x86_64-610.57.04.run
sudo sh NVIDIA-Linux-x86_64-610.57.04.run --silent --no-kernel-modules
# Módulos abiertos con el parche P2P, por DKMS (se recompilan solos al actualizar el kernel)
sudo git clone -b 610.57.04-p2p https://github.com/aikitoria/open-gpu-kernel-modules.git /usr/src/nvidia-p2p-610.57.04
sudo tee /usr/src/nvidia-p2p-610.57.04/dkms.conf > /dev/null <<'EOF'
PACKAGE_NAME="nvidia-p2p"
PACKAGE_VERSION="610.57.04"
AUTOINSTALL="yes"
BUILT_MODULE_NAME[0]="nvidia";         BUILT_MODULE_LOCATION[0]="kernel-open"; DEST_MODULE_LOCATION[0]="/updates/dkms"
BUILT_MODULE_NAME[1]="nvidia-uvm";     BUILT_MODULE_LOCATION[1]="kernel-open"; DEST_MODULE_LOCATION[1]="/updates/dkms"
BUILT_MODULE_NAME[2]="nvidia-modeset"; BUILT_MODULE_LOCATION[2]="kernel-open"; DEST_MODULE_LOCATION[2]="/updates/dkms"
BUILT_MODULE_NAME[3]="nvidia-drm";     BUILT_MODULE_LOCATION[3]="kernel-open"; DEST_MODULE_LOCATION[3]="/updates/dkms"
BUILT_MODULE_NAME[4]="nvidia-peermem"; BUILT_MODULE_LOCATION[4]="kernel-open"; DEST_MODULE_LOCATION[4]="/updates/dkms"
MAKE[0]="env -u KERNELRELEASE make -j$(nproc) KERNELRELEASE=\"\" KERNEL_UNAME=${kernelver} modules"
EOF
sudo dkms install nvidia-p2p/610.57.04
sudo reboot
```

Después de reiniciar, verificar que el P2P esté prendido (`OK` entre las dos placas, no `NS`) y que el
BAR1 sea grande (32768 MiB en una 3090):

```bash
nvidia-smi topo -p2p r
nvidia-smi -q | grep -A1 "BAR1 Memory Usage"
```

**2. Docker** (saltear si `docker --version` ya funciona)

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER" && newgrp docker
```

**3. NVIDIA Container Toolkit** (saltear si `docker run --rm --gpus all ubuntu nvidia-smi` ya muestra las dos placas)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi
```

**4. Compose y plantilla de chat**

```bash
mkdir -p ~/idiotsavant && cd ~/idiotsavant
curl -fsSLO https://raw.githubusercontent.com/blairq/vllm-idiotsavant/main/docker/idiotsavant/docker-compose.qwen38-27b-sm86.yml
curl -fsSLO https://raw.githubusercontent.com/blairq/vllm-idiotsavant/main/docker/idiotsavant/qwen-froggeric-chat-template.jinja
```

**5. Arrancar.** El primer arranque baja la imagen (~7 GB), el modelo y el borrador (~20 GB, a
`./models`) y compila los kernels y los grafos CUDA: hay que contar 15-30 minutos según la conexión.
Los siguientes tardan unos minutos.

```bash
docker compose -f docker-compose.qwen38-27b-sm86.yml up -d
until curl -sf http://localhost:8000/health > /dev/null; do sleep 10; done; echo "listo"
```

**6. Probar**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer idiotsavantftw" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8",
       "messages": [{"role": "user", "content": "¿Cuál es la capital de Francia? Respondé en una oración."}],
       "chat_template_kwargs": {"enable_thinking": false},
       "max_tokens": 64}'
```

La API es compatible con OpenAI en `http://<host>:8000/v1`, con el modelo `qwen3.8`.

**Configuración.** Sin archivo `.env` el servidor escucha en el puerto 8000 con la clave
`idiotsavantftw` y guarda modelos y caches en `./models` y `./cache`. Para cambiar algo de eso:

```bash
curl -fsSLo .env https://raw.githubusercontent.com/blairq/vllm-idiotsavant/main/docker/idiotsavant/.env.example
nano .env   # clave, puerto, rutas, límite de memoria
docker compose -f docker-compose.qwen38-27b-sm86.yml up -d
```

Para conectar el servidor a una red de Docker existente con IP fija (un proxy inverso, otros
contenedores) o fijarlo a ciertas CPUs, bajar también `docker-compose.network.yml` y agregar
`-f docker-compose.network.yml` a los comandos.

**Comandos de todos los días**

```bash
docker compose -f docker-compose.qwen38-27b-sm86.yml logs -f vllm-server   # logs
docker compose -f docker-compose.qwen38-27b-sm86.yml down                  # parar
docker compose -f docker-compose.qwen38-27b-sm86.yml run --rm clear-caches # limpiar caches JIT (con el servidor parado)
```

## Desarrollo

**Estructura**

| Ruta | Qué hay |
|---|---|
| `vllm/idiotsavant/` | Los módulos del fork: opciones, reserva de memoria, lado Python de los kernels, lógica de scheduler y caches |
| `vllm/idiotsavant/kernels/` | Kernels PTX (`.cu`, se compilan en runtime con `nvcc`/`ptxas` y quedan en cache) y la extensión Marlin S16 (se compila en la imagen) |
| `vllm/` (el resto) | vLLM v0.29.0; cada cambio a un archivo de upstream lleva un comentario `[PNxxx]` |
| `docs/idiotsavant/` | Inventario de los parches integrados y de las viejas variables de entorno → flags nuevos |
| `docker/` | `Dockerfile.idiotsavant` y las compose |
| `tools/idiotsavant/` | Herramientas de prueba en GPU: batería funcional y estrés de memoria y admisión |

`main` es vLLM v0.29.0 más el fork; el tag `idiotsavant-base` marca la primera integración. Para
seguir a upstream: `git remote add upstream https://github.com/vllm-project/vllm.git`.

**Construir y correr una imagen propia**

```bash
git clone https://github.com/blairq/vllm-idiotsavant.git && cd vllm-idiotsavant
docker build -f docker/Dockerfile.idiotsavant -t vllm-idiotsavant:dev .
cd docker/idiotsavant
VLLM_IDIOTSAVANT_IMAGE=vllm-idiotsavant:dev docker compose -f docker-compose.qwen38-27b-sm86.yml up -d
```

La imagen parte de la oficial `vllm/vllm-openai:v0.29.0` y superpone el paquete `vllm/` de este repo,
así que un cambio solo de Python no necesita compilar vLLM. Los kernels `.cu` editados se recompilan
en el primer uso; después de cambiarlos, o para comparar dos builds, correr antes `clear-caches`.

**Pruebas**

```bash
ruff check vllm/idiotsavant tools/idiotsavant
python3 tools/idiotsavant/bateria.py --url http://localhost:8000 --salida resultado.json
python3 tools/idiotsavant/estres.py                                   # 13 pedidos simultáneos, prefijo compartido
python3 tools/idiotsavant/estres.py --distintos                       # contextos distintos de 30-110k
python3 tools/idiotsavant/estres.py --distintos --agente primary --rondas 1   # alta prioridad: todos prefillean juntos
```

El grupo `det` de la batería guarda textos greedy que se pueden comparar token por token entre dos
builds. Hay que compararlos con las caches vacías de los dos lados: el estado de la cache de autotune
de Inductor cambia los últimos bits de la salida. Cualquier error en el estrés es una falla.

**Depuración.** Al arrancar, el log informa la memoria reservada antes de que vLLM dimensione la KV
(`[reserva] ... MiB creados antes del perfil de memoria`) y el tamaño de la KV. Ante una excepción de
GPU, el driver escribe un volcado en `CACHE_DIR/vllm/idiotsavant/volcados/` (varios GB; se lee con
`cuda-gdb` y después se borra).

## Origen y créditos

Integra en el árbol de vLLM los parches de
[blairq/genesis-vllm-patches](https://github.com/blairq/genesis-vllm-patches), derivado de Genesis
(Sandermage, Apache-2.0); ver [`vllm/idiotsavant/NOTICE`](vllm/idiotsavant/NOTICE). Con caches
equivalentes, la salida greedy es idéntica token por token a la del contenedor con los parches
Genesis. Todo lo demás es [vLLM](https://github.com/vllm-project/vllm): su README está en
[`README.md`](README.md#about) (en inglés, debajo de la parte de este fork) y en el
[repositorio de vLLM](https://github.com/vllm-project/vllm#readme); la documentación, en
[docs.vllm.ai](https://docs.vllm.ai).
