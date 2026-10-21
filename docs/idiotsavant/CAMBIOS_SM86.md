# Optimizaciones del camino sm_86: qué se cambió y cómo probarlo

Rama: `optimizaciones/sm86`. Base: `dacb2e5` (main del fork).

Este documento cubre dos cosas:

- qué se cambió en cada commit, archivo por archivo, y qué queda prendido o apagado;
- cómo probarlo en las 2× RTX 3090, paso a paso, desde la compilación hasta el servidor completo.

El análisis que motivó cada cambio está en [`informe_kernels_sm86.md`](informe_kernels_sm86.md).

> **Importante.** Todo se escribió en un contenedor **sin GPU**. Lo que sí se verificó ahí:
>
> - cada kernel modificado compila para sm_86 (NVRTC 12.8 + `ptxas`), con sus registros, derrame
>   y shared revisados;
> - la lógica de partición del prefill int8 respeta la causalidad, comprobado en CPU sobre 300 lotes
>   aleatorios.
>
> **Ningún kernel se ejecutó en una GPU.** Por eso:
>
> - lo que es bit a bit igual y no puede ser más lento quedó **prendido**;
> - lo que cambia la cuenta, o cuyo efecto en velocidad hay que medir, quedó **apagado** detrás de
>   una constante o un define.
>
> Los tests de `tests/idiotsavant/` existen para que lo verifiques en la placa antes de usarlo.

---

## 1. Resumen

| # | Commit | Cambio | Estado | ¿Cambia la cuenta? |
|---|---|---|---|---|
| 1 | `72e33b8` | `ptx_lab`: `-lineinfo` opcional y `Kernel.recursos()`; herramienta `ptx_recursos.py` | — | no |
| 2 | `0afbb00` | SK-30: V por `cp.async.cg`; `PING` y `PHL` como constantes | VCG **prendido**, PING/PHL apagados | VCG/PING no; PHL sí |
| 3 | `b5fffac` | `sk18h_decuant_tile`: decuantización del prefill por tiles | **prendido** | no (bit a bit) |
| 4 | `d7fda19` | Marlin S16: hint `evict_first` de L2 y shared justa | **apagados** (defines) | no |
| 5 | `899ceca` | Prefill de atención entero: SK-30 sobre pseudo-secuencias | **apagado** (`PREFILL_SK30`) | sí (Q en int8) |
| 6 | `4648b40` | SK-30 sin derrame (`#pragma unroll 1`) y variante `NQ=4` | unroll **prendido**, NQ=4 apagado | unroll no; NQ=4 solo redondeo |
| 7 | `67212ff` | `tests/idiotsavant/README.md` y `correr_todo.sh` | — | — |
| 8 | `87c1afe` | Informe: estado de implementación; clave única en el escenario de test | — | — |

**Qué cambia en el servidor con la rama tal cual** (sin tocar ninguna constante):

- SK-30 (decode de atención en árbol) carga V sin pasar por L1 y ya no derrama registros: de 255 a
  177 registros por hilo. Mismo resultado bit a bit.
- La decuantización del prefill usa el kernel por tiles. Mismo resultado bit a bit.

Todo lo demás corre igual que en `main` hasta que lo prendas.

---

## 2. Qué se hizo, commit por commit

### Commit 1 · `72e33b8` · Herramientas de perfilado

**Archivos:** `vllm/idiotsavant/kernels/ptx_lab.py`, `vllm/idiotsavant/sk18_attn.py`,
`tools/idiotsavant/ptx_recursos.py` (nuevo).

- **`ptx_lab.py`:**
  - `-lineinfo` estaba clavado en `if False`. Ahora se prende con
    `VLLM_IDIOTSAVANT_PTX_LINEINFO=1`, para que Nsight Compute atribuya las métricas a líneas de
    fuente.
  - La variable entra en la clave de la caché de PTX: prenderla compila de nuevo y apagarla vuelve
    al PTX de siempre.
  - `Kernel.recursos()` corre `ptxas -v` y devuelve registros, derrame, stack y shared estática. No
    necesita GPU.
- **`sk18_attn.py`:** los defines de `escribir`, `decuant` y `sk30` salen de funciones (`_defs_*`),
  para que la herramienta compile exactamente lo mismo que el servidor. En este commit los defines
  son idénticos a los de antes.
- **`ptx_recursos.py`:**
  - compila los ~19 kernels PTX de producción con sus defines reales;
  - estima los bloques por SM en sm_86 (64K registros, 100 KB de shared, 16 bloques, 48 warps) y
    qué recurso los limita;
  - con `--json` guarda el resultado para comparar dos commits.

### Commit 2 · `0afbb00` · SK-30: carga de V y variantes

**Archivos:** `kernels/cuda/sk30_decode_1pasada.cu`, `sk18_attn.py`,
`tests/idiotsavant/atencion_sintetica.py` (nuevo), `tests/idiotsavant/test_sk30.py` (nuevo).

- **`VCG` (nuevo define):** V se carga con `cp.async.cg` en vez de `.ca`.
  - V no se reusa dentro del bloque ni entre bloques, y con 86,5 KB de shared el SM queda con
    28 KB de L1: `.ca` solo la ensuciaba.
  - Mismo resultado bit a bit. Constante `SK30_VCG = 1`.
- **`sh_30()` reemplaza a `SH_30`:** con `PING` el anillo de K es de 3 tiles (94 720 B en vez de
  86 528 B). La constante vieja no lo contemplaba, así que prender `PING` desbordaba la shared
  pedida.
- **`SK30_PING = 0`:** emite el Q·K del tile siguiente junto con el softmax del actual. Es bit a bit
  igual, pero sube los registros, así que queda apagado hasta medirlo.
- **`SK30_PHL = 1`:** P·V en dos planos int8. Con `0` va en un plano: la mitad de los mma, pero
  cambia la cuenta.
- **`atencion_sintetica.py`:** arma una capa de atención del Qwen3.8-27B con TP=2, sin levantar
  vLLM, y la reusan los demás tests:
  - 12 cabezas Q y 2 KV, head_dim 256, páginas de 880 tokens, árbol de 9 tokens;
  - KV paginada y desordenada en el layout de PN131;
  - k/v con canales outlier;
  - referencia en fp32 sobre k/v sin cuantizar.

### Commit 3 · `b5fffac` · Decuantización por tiles

**Archivos:** `kernels/cuda/sk18h_decuant.cu`, `sk18_attn.py`, `tools/idiotsavant/ptx_recursos.py`,
`tests/idiotsavant/test_decuant.py` (nuevo).

**Problema.** El prefill decuantiza todas las páginas que toca (int8 → fp16) antes de FlashInfer.
`sk18h_decuant` usa un bloque de **1 warp** por (página, token), y eso trae dos problemas:

- sm_86 admite 16 bloques por SM, así que el kernel nunca pasa del 33 % de ocupación;
- V está traspuesta (`[NH][256][BS]`): para un token, cada lane lee bytes separados por 880 B, o
  sea 256 sectores de 32 B por cada 256 B útiles.

**`sk18h_decuant_tile`** (nuevo, en el mismo archivo y con la misma firma):

- un bloque de 8 warps por (página, tile de 32 tokens);
- escalas del tile y V traspuesta en shared, con V leída de a 16 B contiguos;
- K leída de a 16 B y escrita de a 32 B.

Según `ptxas`: 46 registros, 17 KB de shared, 5 bloques/SM (83 % de ocupación). Hace la misma
cuenta por elemento (`q_a_fp16`), así que el resultado es bit a bit igual.

- `DECUANT_TT = 32` lo prende; `0` vuelve al kernel viejo.
- Si `BS % 16 != 0` o hay más de 2 cabezas KV por rank, cae solo al kernel viejo.

### Commit 4 · `d7fda19` · Marlin S16: opcionales de L2 y shared

**Archivos:** `kernels/marlin_s16/src/marlin.cuh`, `src/marlin_template.h`, `src/marlin.cu`,
`kernels/marlin_s16/build.py`, `marlin_s16.py`, `tests/idiotsavant/test_marlin_s16.py` (nuevo).

- **`MARLIN_L2_HINT`:** el peso se carga con `createpolicy.fractional.L2::evict_first` +
  `cp.async.cg...L2::cache_hint` (el `cp_async4_stream` del Marlin original).
  - El peso se lee una vez por GEMM.
  - Sin política, puede sacar de los 6 MB de L2 lo que sí se relee: A, escalas, `C_tmp` y locks.
- **`MARLIN_SHARED_JUSTA`:** el decode usa ~41 KB de shared, pero el lanzamiento pedía los 99 KB.
  - Ahora pide `get_kernel_cache_size` + el prefijo `sh_a_s`/`sh_a_sum` del template
    (64 B × m_blocks × (stages + 1)) + 1 KB.
  - No cambia la ocupación de Marlin, que ya queda en 1 bloque por SM por registros. Deja lugar
    para que otro kernel comparta el SM.
- **`build.py`:** lee `IDIOTSAVANT_MARLIN_DEFS` (por ejemplo `"-DMARLIN_L2_HINT=1"`).
- **`marlin_s16._huella()`:** incluye esa variable, así `construir()` recompila cuando cambia.
- **Ambos defines vienen apagados.** Marlin corre en todas las lineales del modelo y no pude
  ejecutarlo.

### Commit 5 · `899ceca` · Prefill de atención entero (`PREFILL_SK30`)

**Archivos:** `sk18_attn.py`, `tests/idiotsavant/test_prefill_sk30.py` (nuevo),
`tests/idiotsavant/atencion_sintetica.py`.

**Hoy.** El prefill decuantiza la KV a fp16 y llama a FlashInfer, que corre en HMMA fp16 con
acumulador fp32: **71 TFLOPS** en GeForce Ampere. A contexto largo, la atención domina el prefill.

**`_prefill_sk30`** reusa SK-30, que ya corre Q·K en IMMA int8 (**284 TOPS**) sobre esta misma KV.

**Partición en pseudo-secuencias.**

- Cada pedido del chunk se parte en pseudo-secuencias de `L = 64 // G = 10` tokens, que son 60 filas
  de las 64 que admite SK-30.
- La pseudo-secuencia `p` termina `L·(P-1-p)` tokens antes del final del pedido, con su `seq_len`
  recortado.
- La causalidad sale de la máscara de cadena `(1 << j) - 1` del camino de árbol.
- Si un pedido no llena la primera pseudo-secuencia, las filas de relleno repiten una query y se
  descartan.

**Pipeline por capa:**

1. junta Q en un buffer contiguo;
2. `sk18h_prep2` la cuantiza y la rota, igual que en el decode;
3. `sk30_decode` con `BAL=0` corre cada pseudo-secuencia en `g` grupos (el `g` más chico que deja la
   última ola de bloques llena al 90 %);
4. `sk30_union` combina los grupos;
5. la salida se reparte a las posiciones reales.

El plan se arma una vez por paso y lo reusan las 16 capas. No decuantiza nada, porque las K/V del
chunk ya están en el pool.

**Memoria.** Con `PREFILL_SK30`, `reservar()` crea estos buffers (~130 MB con 8192 tokens por paso)
y no crea el de decuantización ni los wrappers de FlashInfer (~250 MB).

**Por qué viene apagado.** Q queda cuantizada a int8 por fila en el prefill, igual que en el decode.
Es otra cuenta que la de FlashInfer.

### Commit 6 · `4648b40` · SK-30 sin derrame y `NQ=4`

**Archivos:** `kernels/cuda/sk30_decode_1pasada.cu`, `sk18_attn.py`, `tests/idiotsavant/test_sk30.py`.

`paso_grupo` suma los `seq_len` del lote. Ese bucle, desenrollado por el compilador, dejaba
`sk30_decode` en **255 registros con 28 B de derrame**. Con `#pragma unroll 1` queda en **177
registros y sin derrame**. Es la misma suma en el mismo orden, así que el resultado es bit a bit
igual.

| Variante de `sk30_decode` (ptxas sm_86) | Antes | Ahora |
|---|---|---|
| producción (NQ=2, PV8, PHL=1) | 255 reg, 28 B derrame | **177 reg, 0** |
| `PING=1` | 255 reg, 52 B | 255 reg, 24 B |
| `NQ=4` | 128 reg, 88 B | 128 reg, 16 B |
| `BAL=0` (prefill entero) | — | 175 reg, 0 |

`SK30_NQ = 4` (16 warps por bloque) se agrega como variante a medir. Cambia el orden de las sumas
de `l`, así que la diferencia con `NQ=2` es solo de redondeo.

### Commits 7 y 8 · Documentación

- **`tests/idiotsavant/README.md` y `correr_todo.sh`:** el orden de validación y un script que corre
  todo y guarda los resultados.
- **Sección 7 del informe:** qué quedó hecho y qué no, con el motivo.
- **Escenario de test:** cada metadata lleva una clave de paso única, para que un `id(md)` reciclado
  no reuse el plan de otro caso.

---

## 3. Cómo probarlo

### 3.0 Preparar el entorno

Los tests importan `vllm.idiotsavant` del paquete instalado. Para que usen el código de la rama hay
dos caminos.

**A. Imagen nueva desde la rama.** Es lo que vas a usar al final.

```bash
git fetch origin optimizaciones/sm86
git checkout optimizaciones/sm86
docker build -f docker/Dockerfile.idiotsavant -t vllm-idiotsavant:sm86opt .
```

La imagen recompila la extensión Marlin, porque cambió su fuente.

**B. Sin reconstruir, montando el paquete de la rama** (rápido para iterar):

```bash
SITE=/usr/local/lib/python3.12/dist-packages
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/ws -w /ws \
  -v "$PWD/vllm/idiotsavant":$SITE/vllm/idiotsavant \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  --entrypoint bash ghcr.io/blairq/vllm-idiotsavant:v0.29.0-idiotsavant.1
```

Adentro del contenedor, en los dos casos:

```bash
pip install -q pytest
# solo en el camino B: compila la extension Marlin de la rama (~1 min, queda en el repo, ignorada por git)
TORCH_CUDA_ARCH_LIST=8.6 python3 -c "from vllm.idiotsavant import marlin_s16; print(marlin_s16.construir())"
```

**Para el camino A**, además, montá el repo para tener `tests/` y `tools/`:

```bash
docker run --rm -it --gpus all --ipc=host -v "$PWD":/ws -w /ws --entrypoint bash vllm-idiotsavant:sm86opt
```

**Reloj fijo**, para que las diferencias chicas se lean (desde el host):

```bash
sudo nvidia-smi -i 0 -lgc 1400      # al terminar: sudo nvidia-smi -i 0 -rgc
```

Los tests usan una sola GPU (la 0). No hace falta parar el servidor si tenés memoria libre, pero las
mediciones salen más limpias con la GPU sola.

### 3.1 Paso 1 · Recursos de los kernels (sin GPU)

```bash
python3 tools/idiotsavant/ptx_recursos.py --json recursos.json
```

**Qué esperar** (puede variar algún registro con otra versión de nvcc):

| entrada | regs | derrame | bloques/SM |
|---|---|---|---|
| `sk30_decode` | ~177 | **0** | 1 |
| `sk18h_decuant_tile` | ~46 | 0 | 5 |
| el resto | sin cambios respecto de `main` | | |

**Si `sk30_decode` muestra derrame > 0**, tu nvcc desenrolla distinto. Avisame con la salida.

### 3.2 Paso 2 · Exactitud de lo prendido por defecto

```bash
pytest -q -s tests/idiotsavant/test_sk30.py tests/idiotsavant/test_decuant.py
```

| Test | Comprueba | Si falla |
|---|---|---|
| `test_variantes_de_carga_bit_a_bit` | producción, `vcg` y `ping` dan **exactamente** la misma salida que la referencia (VCG=0, PING=0); error contra fp32 < 8 % | No usar la rama: `SK30_VCG = 0` en `sk18_attn.py` y avisar |
| `test_tile_bit_a_bit` | `sk18h_decuant_tile` (TT 16 y 32) da los mismos bits que el kernel viejo | `DECUANT_TT = 0` en `sk18_attn.py` |
| `test_tile_reconstruye_kv` | la KV decuantizada se parece a la original | ídem |
| `test_nq4_solo_difiere_en_redondeo` | `NQ=4` no empeora el error más que 0,005 | no prender `NQ=4` |
| `test_phl0_informa_error` | imprime el error de `PHL=0`; falla si supera 15 % | no prender `PHL=0` |

### 3.3 Paso 3 · Marlin

```bash
pytest -q tests/idiotsavant/test_marlin_s16.py                      # exactitud (error < 0,5 %)
python3 tests/idiotsavant/test_marlin_s16.py --medir > marlin_base.txt
```

**La tabla** da µs y GB/s del peso para o_proj, qkv, gate_up, down y lm_head, con M = 1, 9, 18, 36,
81, 99 y 880. Es la línea base del decode: en la 3090 el techo es ~936 GB/s, y el informe estima
~500-600 GB/s para las formas medianas.

**Variantes:** recompilar en un proceso y medir en otro, porque el import carga la `.so` que haya.

```bash
IDIOTSAVANT_MARLIN_DEFS="-DMARLIN_L2_HINT=1" python3 -c \
    "from vllm.idiotsavant import marlin_s16; marlin_s16.construir()"
pytest -q tests/idiotsavant/test_marlin_s16.py
python3 tests/idiotsavant/test_marlin_s16.py --medir > marlin_l2.txt

IDIOTSAVANT_MARLIN_DEFS="-DMARLIN_SHARED_JUSTA=1" python3 -c \
    "from vllm.idiotsavant import marlin_s16; marlin_s16.construir()"
pytest -q tests/idiotsavant/test_marlin_s16.py
python3 tests/idiotsavant/test_marlin_s16.py --medir > marlin_shared.txt

# reduccion por atomicAdd de upstream, sin recompilar:
VLLM_MARLIN_USE_ATOMIC_ADD=1 python3 tests/idiotsavant/test_marlin_s16.py --medir > marlin_atomic.txt

# volver a la de produccion:
IDIOTSAVANT_MARLIN_DEFS="" python3 -c "from vllm.idiotsavant import marlin_s16; marlin_s16.construir()"
```

**Si `MARLIN_SHARED_JUSTA` da error de memoria ilegal o el test de exactitud falla**, el cálculo de
shared no coincide con el layout del template. No usarlo y avisar.

**Criterio:** prender un define si baja los µs en M = 9 a 36 (decode con 1 a 4 pedidos) sin
empeorar M = 880 (prefill).

### 3.4 Paso 4 · Variantes de SK-30

```bash
python3 tests/idiotsavant/test_sk30.py --medir > sk30.txt
# o mas corto:
python3 tests/idiotsavant/test_sk30.py --medir --ctx 60000 100000 --pedidos 1 4
```

**Columnas:** variante, pedidos, contexto, µs por capa (prep + sk30 + union) y error relativo
máximo.

| Variante | Qué prender en `sk18_attn.py` si gana | Condición |
|---|---|---|
| `ping` | `SK30_PING = 1` | menos µs que `produccion` en 60k y 100k; es bit a bit igual |
| `nq4` | `SK30_NQ = 4` | menos µs que `produccion`; solo cambia el redondeo |
| `phl0` | `SK30_PHL = 0` | menos µs **y** KL y batería sin empeorar (paso 6), porque cambia la cuenta |

`produccion` contra `referencia` muestra el efecto de VCG más el unroll (los dos prendidos).

### 3.5 Paso 5 · Decuantización y prefill entero

```bash
python3 tests/idiotsavant/test_decuant.py --medir
```

Imprime µs y GB/s con `DECUANT_TT` = 0 (kernel viejo), 16 y 32. El tile debería quedar bastante por
encima.

```bash
pytest -q -s tests/idiotsavant/test_prefill_sk30.py
python3 tests/idiotsavant/test_prefill_sk30.py --medir > prefill.txt
```

**El pytest** imprime el error relativo máximo contra fp32 de los dos caminos en 4 casos:

- chunk de 880;
- dos pedidos largos;
- prefill desde cero;
- chunk corto con relleno.

Exige que el camino entero quede por debajo de 8 % y de `3 × error_flashinfer + 0,02`.

**`--medir`** da ms por capa de FlashInfer contra entero, con 8k, 32k, 64k y 100k de contexto y
chunks de 880 y 2640, más la columna `x` de aceleración.

**Si `test_prefill_sk30` falla** con `assert ... plan is not None`, el camino entero no corrió (el
plan no entró en los buffers). Avisame con la salida.

### 3.6 Paso 6 · De punta a punta con el servidor

Todo lo anterior en un solo comando, para guardar y comparar:

```bash
bash tests/idiotsavant/correr_todo.sh resultados-$(git rev-parse --short HEAD)
```

**Primero, la rama tal cual** (solo los cambios prendidos por defecto):

```bash
# en el host, con la imagen del paso 3.0 A
cd docker/idiotsavant
VLLM_IDIOTSAVANT_IMAGE=vllm-idiotsavant:sm86opt docker compose -f docker-compose.qwen38-27b-sm86.yml up -d
python3 ../../tools/idiotsavant/bateria.py --url http://localhost:8320 --salida rama.json
```

Compará `rama.json` contra una corrida con la imagen publicada
(`ghcr.io/blairq/vllm-idiotsavant:v0.29.0-idiotsavant.1`).

| Grupo de `bateria.py` | Qué mirar |
|---|---|
| `det` | texto greedy: con los cambios bit a bit iguales tiene que salir **idéntico** |
| `vel` | velocidad de decode |
| `largo` | prompt de ~160k frío y desde la prefix cache (TTFT) |
| `tool` | tool calls |

**Después, cada cosa que ganó en los pasos 3 a 5.** Cambiá la constante en
`vllm/idiotsavant/sk18_attn.py` (o el define en `kernels/marlin_s16/build.py`), reconstruí la imagen
y repetí la batería.

**Con `PHL=0` o `PREFILL_SK30`**, `det` puede cambiar porque la cuenta es otra. En ese caso:

- `tool` y `largo` no tienen que empeorar;
- conviene medir la KL contra BF16 con el mismo procedimiento de la tarjeta del modelo antes de
  dejarlo en producción.

### 3.7 Perfilar un kernel con Nsight Compute

```bash
VLLM_IDIOTSAVANT_PTX_LINEINFO=1 ncu --set full --import-source yes \
    -k regex:sk30_decode -c 10 -o sk30 \
    python3 tests/idiotsavant/test_sk30.py --medir --ctx 60000 --pedidos 1
```

Métricas útiles:

- `dram__throughput.avg.pct_of_peak_sustained_elapsed`;
- `sm__pipe_tensor_op_imma_cycles_active`;
- `smsp__warp_issue_stalled_long_scoreboard` y `..._barrier`;
- `lts__t_sector_hit_rate`.

---

## 4. Cómo volver atrás

| Para desactivar | Dónde | Valor |
|---|---|---|
| V por `.cg` en SK-30 | `vllm/idiotsavant/sk18_attn.py` | `SK30_VCG = 0` |
| Decuantización por tiles | `vllm/idiotsavant/sk18_attn.py` | `DECUANT_TT = 0` |
| Unroll del bucle de SK-30 | `kernels/cuda/sk30_decode_1pasada.cu` | borrar el `#pragma unroll 1` de `paso_grupo` |
| Todo | git | `git checkout main` y reconstruir la imagen |

El resto ya viene apagado.

## 5. Qué no se hizo

El motivo de cada punto está en la sección 7 del informe.

- **GEMV W4A8 propio para M chico.** Es la palanca más grande del decode, pero es un kernel nuevo
  sobre el layout de Marlin y no se podía probar.
- **`pn122_cinta` dentro de `gdn_arbol`.** Sería una carrera sobre las mismas filas de la cinta.
- **Cinta del GDN en fp16.** Habría que tocar 7 kernels para ahorrar ~40 MB.
- **`sk30_union` dentro de SK-30.** Sería más lento que el lanzamiento aparte.
- **`sk24` + `sk26`.** Exige rediseñar el protocolo P2P.
- **`escribir2` + `prep2`.** Corren en ganchos distintos de vLLM.

## 6. Archivos tocados

```
docs/idiotsavant/informe_kernels_sm86.md        informe (y seccion 7: estado)
docs/idiotsavant/CAMBIOS_SM86.md                este documento
tools/idiotsavant/ptx_recursos.py               nuevo
tests/idiotsavant/README.md                     nuevo
tests/idiotsavant/correr_todo.sh                nuevo
tests/idiotsavant/atencion_sintetica.py         nuevo
tests/idiotsavant/test_sk30.py                  nuevo
tests/idiotsavant/test_decuant.py               nuevo
tests/idiotsavant/test_marlin_s16.py            nuevo
tests/idiotsavant/test_prefill_sk30.py          nuevo
vllm/idiotsavant/kernels/ptx_lab.py             lineinfo, recursos()
vllm/idiotsavant/sk18_attn.py                   constantes, sh_30(), decuant tile, prefill entero
vllm/idiotsavant/kernels/cuda/sk30_decode_1pasada.cu   VCG, unroll 1
vllm/idiotsavant/kernels/cuda/sk18h_decuant.cu         sk18h_decuant_tile
vllm/idiotsavant/kernels/marlin_s16/src/marlin.cuh     cp_async4_stream, defines
vllm/idiotsavant/kernels/marlin_s16/src/marlin_template.h  MARLIN_L2_HINT
vllm/idiotsavant/kernels/marlin_s16/src/marlin.cu      MARLIN_SHARED_JUSTA
vllm/idiotsavant/kernels/marlin_s16/build.py           IDIOTSAVANT_MARLIN_DEFS
vllm/idiotsavant/marlin_s16.py                         huella con los defines
```

---

## Resultados en 2× RTX 3090 (21-10)

Medido contra `main` (`dacb2e5`) con la misma compose, caches vacías al principio y los mismos prompts.

**Exactitud (tests de `tests/idiotsavant/`):**

- Bit a bit iguales, como se esperaba: la decuantización por tiles (TT=16 y 32) contra el kernel viejo, y
  SK-30 `produccion`/`vcg`/`ping` contra la referencia.
- La salida greedy del servidor completo es idéntica token por token a la de `main` (6 prompts × 2 corridas).
- Fallan 4 tests por umbrales puestos sin GPU:
  - `test_tile_reconstruye_kv` y `test_variantes_de_carga_bit_a_bit[ctx0]`: el error contra fp32 (0,075 y
    0,0876) también lo tiene el kernel viejo; el umbral de 0,05/0,08 es demasiado justo para la KV sintética
    con outliers.
  - `test_prefill_entero_contra_fp32` (2 casos): `PREFILL_SK30` pierde precisión de verdad (error contra
    fp32 0,077 → 0,088 y 0,053 → 0,072 en prefill desde cero).
- `correr_todo.sh` llama a `git`, que no está en la imagen, y su pytest levanta el `tests/conftest.py`
  general de vLLM: hay que copiar `tests/idiotsavant/` fuera del repo y correr pytest ahí.

**Kernels:**

| Cambio | Tiempo | ¿Pierde calidad? |
|---|---|---|
| Decuantización por tiles (prendida) | 955 → 338 µs por 64 páginas (2,8×) | no |
| SK-30 sin derrame + VCG (prendido) | 3–11 % menos en el decode de atención | no |
| `PREFILL_SK30` (apagado) | 1,25–1,7× más rápido que FlashInfer | sí |
| `SK30_PHL=0` (apagado) | hasta 10 % más rápido | sí, poco (0,0689 → 0,0696) |
| `SK30_PING`, `SK30_NQ=4` (apagados) | más lentos | — |
| Marlin `L2_HINT` / `SHARED_JUSTA` (apagados) | mixto: `down` M=36 +20 %, `gate_up` M=36 −14 % | no |

**Servidor completo** (`tools/idiotsavant/bateria.py`, `estres.py` y `velocidad.py`):

| Escenario | `main` | rama |
|---|---|---|
| 8 decodes simultáneos sobre ~27k de contexto (régimen) | 1006 / 1030 tok/s | **1065 / 1072 tok/s (+4–6 %)** |
| 8 × 2k, 512 de salida (decode agregado) | 387 / 426 tok/s | 385 / 400 tok/s |
| Prefill 8 × 27k | 2391 / 2447 tok/s | 2403 / 2470 tok/s |
| Prefill 1 × 55k | 2236 / 2227 tok/s | 2256 / 2252 tok/s |
| Prefill frío de 163k | 113,6 / 112,1 s | 113,1 / 110,5 s |
| Estrés (prefijo compartido, distintos de 30–110k, alta prioridad) | 0 errores | 0 errores |

Con lo que viene prendido, la rama gana 4–6 % en decode con contexto largo y muchos pedidos; en el resto
queda dentro del ruido. El prefill no se mueve: lo dominan Marlin y el GDN. Lo único que lo movería es
`PREFILL_SK30`, que pierde precisión.
