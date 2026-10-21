# Informe: kernels de vllm-idiotsavant frente a la microarquitectura de la RTX 3090 (GA102, sm_86)

> **Alcance.** Este informe se armó leyendo el código, sin ejecutar nada. No modifica ningún archivo
> de código. Cada número marcado como *estimado* sale de una cuenta de servilleta hecha con las
> formas del modelo y las cifras del marco de referencia, no de una medición. Los números marcados
> como *medido (repo)* están citados de comentarios del propio código: los midió el autor en su
> momento y no se volvieron a verificar acá. Antes de implementar cualquier oportunidad hay que
> medirla, y eso es un paso **separado y posterior** (sección 6).

Configuración de referencia: `docker/idiotsavant/docker-compose.qwen38-27b-sm86.yml`.

- **Modelo y precisión:** `--dtype float16`, `--mamba-ssm-cache-dtype float16`,
  `--kv-cache-dtype int8_per_token_head`, `--attention-backend TRITON_ATTN`.
- **Paralelismo y lote:** TP=2, `--max-num-seqs 11`.
- **Especulación:** DFlash2 con `num_speculative_tokens: 8` y `tree_drafting: true`, o sea
  **9 tokens por pedido en cada paso de decode**.
- **Chunked prefill:** chunk de 2640 tokens si nadie está decodificando, de 880 si alguno sí.
- **Grafos CUDA:** se capturan tamaños de 4 a 40 tokens, es decir hasta 4 pedidos.

Formas por rank:

- hidden 5120, `gate_up` 2×8704.
- Atención: 12 cabezas Q y 2 cabezas KV, head_dim 256, 16 capas.
- GDN: 8 cabezas k y 24 cabezas v, dimensión 128, 48 capas.
- lm_head con N=124160.

---

## 0. Resumen ejecutivo

| # | Kernel / camino | Clase | Brecha principal | Impacto est. | Esfuerzo |
|---|---|---|---|---|---|
| 1 | **Marlin W4A8-S16, decode** (todas las lineales) | Memory-bound (streaming de pesos) | Llega a ~50-65 % de los 936 GB/s (*medido (repo)*). En decode, los pesos son la mayor parte de los bytes del paso. | **Alto** (decode) | Alto |
| 2 | **Atención de prefill** (decuantizar a fp16 → FlashInfer) | Compute-bound | Corre en HMMA fp16 con acumulador fp32, que en GeForce Ampere rinde **71 TFLOPS**. La KV y la Q ya están en int8 y rotadas, y IMMA int8 rinde **284 TOPS**. | **Alto** (contexto de 60-100k) | Alto |
| 3 | **Cadena de kernels chicos en decode** (GDN + norma/cuantización + all-reduce) | Limitada por latencia y lanzamientos | Del orden de ~800+ lanzamientos por paso (*estimado*). Los intermedios ya caen en L2: el costo es lanzar y drenar cada kernel, no el ancho de banda. | Medio-alto | Medio |
| 4 | **SK-30**, decode de atención en árbol | Memory-bound **cerca del ridge** | 1 bloque/SM y pipeline de 2 etapas sin solapar el softmax con la MMA (`PING=0`). Con 54 filas y P·V en dos planos int8, la intensidad queda en ~190 op/B contra un ridge de ~300. | Medio | Bajo (`PING` ya existe) |
| 5 | Marlin con lote alto (M > 80) | Memory-bound | Árbol de 9 tokens × más de 8 pedidos da M > 80, y entonces el peso se relee entero en 2 pasadas. Con más de 4 pedidos además no hay grafo. | Medio (con carga alta) | Medio |
| 6 | Shared memory pedida de más en Marlin y carveout sin fijar | — | Pide 99 KB y usa 41. Hoy no cambia la occupancy (manda el límite de registros), pero impide que otro kernel comparta el SM. | Bajo | Bajo |
| 7 | Cinta GDN en fp32 (el docstring dice fp16) | Memoria | ~80 MB de VRAM por rank (*estimado*) que podrían ser la mitad. | Bajo | Bajo |
| 8 | **`sk18h_decuant`** y los demás kernels PTX de 1 warp por bloque | Memory-bound (prefill) | Tope estructural de 16 bloques por SM, o sea 33 % de occupancy, en un kernel de streaming. | Bajo-medio (prefill) | Bajo |

**Para el camino PTX de idiot savant en particular** (sección 2.1), el orden queda así:

1. **SK-30:** probar `PING` y `PHL`.
2. **Fusionar los PTX chicos de decode:** `escribir2`+`prep2`, `prep2`→`sk30`, `pn122`→`gdn_arbol`,
   `sk24`+`sk26`.
3. **Un kernel PTX de prefill con Q·K int8** que reemplace `decuant` + FlashInfer.
4. **`sk18h_decuant`** con más warps por bloque.

**Conclusión general.** El fork ya está muy trabajado para sm_86:

- Hay mediciones con reloj fijo y razonamiento de occupancy y de derrame de registros en los
  comentarios.
- La reutilización de datos está explícita en shared memory.
- No se usa BF16 ni TF32 en los caminos calientes.

Las brechas que quedan no son errores de configuración del tipo «tile mal calculado». Son tres
cosas:

- la eficiencia de streaming del GEMM de decode;
- el tipo de tensor core del prefill de atención;
- el costo fijo de tener muchos kernels chicos.

---

## 1. Ajustes al marco de referencia antes de aplicarlo

1. **Una RTX 3090 tiene 82 SM, no 84.**
   - 10496 CUDA cores son 82 × 128. El GA102 completo tiene 84 SM, que es la configuración de la
     3090 Ti y la A6000.
   - El código ya lo asume: SK-30 usa `GMAX = 41 = 82 / 2` y Marlin lanza `sms × 1` bloques.
2. **Shared memory en sm_86.**
   - El máximo es 100 KB por SM y 99 KB por bloque (opt-in).
   - Los carveouts posibles son 0, 8, 16, 32, 64 y 100 KB. El «128/0, 64/64, 28/100» del marco
     corresponde a la partición L1+shared de 128 KB.
   - Sin `PreferredSharedMemoryCarveout`, el driver elige el carveout mínimo que permite la
     occupancy pedida. Para un kernel de 77-91 KB eso fuerza 100 KB de shared y deja 28 KB de L1.
3. **Tensor cores en GeForce Ampere.** Sigue el whitepaper del GA102. La ventaja real no está en
   FP16 contra BF16 sino en el tipo de acumulador y en int8:

   | Tipo | Rendimiento denso |
   |---|---|
   | FP16 con acumulador FP16 | 142 TFLOPS |
   | FP16 o BF16 con acumulador FP32 | **71 TFLOPS** |
   | TF32 | 35,6 TFLOPS |
   | INT8 (IMMA) | **284 TOPS** |

4. **«L2 ~2 % de hit rate» no se traslada tal cual.** La cifra del paper sobre PagedAttention viene
   de bloques chicos (16 tokens) dispersos. Este fork:
   - usa páginas de **~880 tokens** con un layout propio, contiguo dentro de la página:
     `K [BS][NH][256] | V [NH][256][BS] | escalas`, unos 915 KB por página y por capa;
   - hace que todas las filas que reutilizan un tile (9 tokens × 6 cabezas Q) estén en el mismo
     bloque, así que el reúso queda en **shared** y no depende de L2.

   El working set de KV de una capa a 60k tokens es de ~62 MB por rank, unas 10 veces la L2. Por
   eso un hit rate bajo en L2 durante la atención es **esperable y no es un defecto de layout**.
   Corresponde al caso «el working set no entra», no al caso «el layout impide cualquier tile».
5. **Prefetch de KV a L2 (AAAI 2026).** Rinde cuando se solapa con cómputo que no usa DRAM. En el
   decode de este fork todo lo que corre antes de la atención (GEMMs Marlin) es memory-bound, así
   que el prefetch competiría por el mismo ancho de banda. Además el `cp.async` de SK-30 ya
   prefetchea a shared. Su potencial acá es bajo (ver sección 5).

---

## 2. Inventario: qué corre en producción

| Kernel | Fase | Estado |
|---|---|---|
| `sk30_decode` + `sk18h_prep2 -DQSF` + `sk30_union` + `sk18h_escribir2` | decode atención (árbol, MB=64) | **Activo** |
| `sk18h_batch2` (2 pasadas sobre K) + `sk18h_union4` + `sk18h_salida` | decode atención sin árbol | Respaldo |
| `sk18h_decuant` + FlashInfer `BatchPrefillWithPagedKVCacheWrapper` | prefill atención | **Activo** (Triton PN124 si no hay FlashInfer) |
| `marlin_gemm_s16` (extensión C++) | todas las lineales W4A8, lm_head int4 | **Activo** |
| `arbol_conv` → `gdn_arbol` → `pn122_cinta` | decode GDN en árbol | **Activo** |
| FLA Triton (`chunk_*`, `solve_tril`) vía `streaming_gdn_driver` | prefill GDN | **Activo** |
| `sk26_norma_q8`, `sk23_silu_had_q8`, `sk25_cabeza_had_q8` | ambas | **Activo** |
| `sk24_ar_p2p` (`sk24_ar_i8`) | decode (m < 512) | **Activo** |
| `sk_fwht` | solo el borrador (D=128) | Activo |
| `sk31_borrador_attn` (`ACTIVO=False`), `sk27` (`DIN=0`), `sk28`/`sk29` (sin llamador), `pn50` (anulado por PN164), `router_softmax` (MoE), `kernels_legacy/lm_head_fp8_compressor.py` | — | **Inactivos o código muerto** |

Lanzamiento de los kernels PTX:

- `kernels/ptx_lab.py` compila con `nvcc -ptx -arch=sm_86 -O3 --use_fast_math` y lanza con
  `cuLaunchKernel`.
- Solo fija el atributo 8 (`MAX_DYNAMIC_SHARED_SIZE_BYTES`), y únicamente si se piden más de 48 KB
  (`ptx_lab.py:131`).
- **No hay carveout (atributo 9) en ningún lugar del repo**, ni `cudaFuncSetCacheConfig`, ni
  ventanas de persistencia de L2, ni `createpolicy`/`L2::cache_hint`.
- `-lineinfo` está desactivado con `if False` (`ptx_lab.py:90,104`).

---

## 2.1 El camino PTX de idiot savant, en orden de ejecución

Esta sección es el centro del informe. Recorre todos los kernels `.cu` que `ptx_lab.Kernel` compila
en runtime (`nvcc -ptx` → `ptxas` → `cuModuleLoadData`) y que corren con la configuración de
producción. El análisis detallado de cada uno está en la sección 3. Marlin S16 no es PTX de
`ptx_lab` (es una extensión C++ compilada en la imagen), pero aparece entre los PTX en cada lineal,
así que se lo marca como `[Marlin]` para ver la cadena completa.

### Un paso de decode (árbol de 9 tokens por pedido, T = 9·B filas)

**Capa de atención (16 por paso):**

| # | Kernel PTX | Grilla × bloque | Shared | Clase | Notas |
|---|---|---|---|---|---|
| 1 | `sk26_norma_q8` | `(T,1)` × 20 warps | estática chica | latencia | suma de residuo + RMSNorm + int8 |
| 2 | `[Marlin]` qkv_proj | 82 × 8 warps | pide 99 KB | memory-bound | |
| 3 | `sk18h_escribir2` | `(9B, 2)` × **1 warp** | 0 | latencia | k rotada entera + escritura de KV int8. **18 bloques con B=1** |
| 4 | `sk18h_prep2 -DQSF` | `(9B, 12)` × **1 warp** | 0 | latencia | Q rotada → int8, lim/abase/amask |
| 5 | `sk30_decode` | `(41, 2B)` × 8 warps | 86,5 KB | memory-bound, cerca del ridge | 1 ola, 1 bloque/SM |
| 6 | `sk30_union` | `(64, 2B)` × 8 warps | ~9 KB estática | latencia | 10 de 64 bloques salen enseguida (54 filas útiles) |
| 7 | `sk25_cabeza_had_q8` (modo 1) | `(T,1)` × 12 warps | chica | latencia | sigmoide(g)·x + Hadamard 256 + int8 |
| 8 | `[Marlin]` o_proj | | | | |
| 9 | `sk24_ar_i8` | 8 × 8 warps | 0 | latencia (PCIe P2P) | |
| 10 | `sk26_norma_q8` | | | | |
| 11 | `[Marlin]` gate_up | | | | |
| 12 | `sk23_silu_had_q8` | `(T,1)` × 17 warps | chica | latencia | SiLU·mul + Hadamard 512 + int8 |
| 13 | `[Marlin]` down_proj | | | | |
| 14 | `sk24_ar_i8` | | | | |

**Capa GDN (48 por paso):** igual que la de atención, salvo los pasos 3 a 7, que se reemplazan por
`[Marlin]` in_proj_qkvz → `arbol_conv` `(N,160)` × 1 warp → `gdn_arbol` `(8, 24N)` × 4 warps →
`pn122_cinta` `(N, 8, 4)` × 8 warps → `sk25` (modo 0, 24 warps) → `[Marlin]` out_proj.

**Total aproximado de kernels por paso:**

- PTX de `ptx_lab`: ~9 por capa de atención y ~9 por capa GDN, o sea **~580 lanzamientos PTX**.
- Más ~4 Marlin por capa, unos **~256** más.
- Total del orden de **~840 kernels por paso** (*estimado*), sin contar RoPE, embedding, muestreo
  ni el borrador.

### Prefill (chunk de 880 o 2640 tokens)

- `sk26`, `sk23`, `sk25` y `sk24` no corren en prefill (`m ≥ 512` va por int8 + NCCL).
- Para la escritura de la KV, `sk18h_escribir2` corre igual, con `(T, 2)` bloques de 1 warp.
- `sk18h_decuant` corre con `(páginas, 880)` bloques de **1 warp**: decuantiza todo el contexto
  tocado antes de FlashInfer.
- GDN de prefill va por FLA (Triton), no por PTX.

### Hallazgos que aparecen al mirar el camino PTX en conjunto

1. **Bloques de un solo warp.** `escribir2`, `prep2`, `decuant`, `union4`, `salida` y `arbol_conv`
   usan `__launch_bounds__(32)`, bloques de 32 hilos.
   - sm_86 admite como máximo **16 bloques residentes por SM**, así que esos kernels nunca pasan de
     16 de 48 warps (**33 % de occupancy**), sin importar registros ni shared.
   - En decode no importa: son grillas chicas.
   - En **`sk18h_decuant` durante el prefill sí puede importar.** Es streaming puro (lee ~1 KB y
     escribe ~2 KB por token-página) sobre decenas de miles de bloques, justo el caso en que el
     marco pide occupancy para esconder la latencia de DRAM, y hay un tope estructural del 33 %.
   - *Qué medir:* `launch__occupancy_limit_blocks` y `dram__throughput` de `sk18h_decuant`.
   - *Oportunidad:* 4-8 tokens por bloque (4-8 warps). Esfuerzo bajo.
2. **Decode con grillas sub-ocupadas.** En decode, `escribir2` lanza 18 bloques de 1 warp con B=1,
   y `sk26`, `sk23` y `sk25` lanzan T = 9-36 bloques. Es decir, **menos de la mitad de los 82 SM, y
   la mayoría con un warp**. Su duración es casi pura latencia de arranque y de una ida a DRAM o L2.
   Son los candidatos naturales a fusionarse con el kernel vecino: `escribir2` + `prep2` (misma
   forma de grilla `(token, cabeza)`), `prep2` en el prólogo de `sk30`, `sk24` + `sk26`, y
   `pn122` dentro de `gdn_arbol`.
3. **Todo el PTX sale con `--use_fast_math` global** (`ptx_lab.py:91`). Los kernels que necesitan
   ser bit a bit iguales a Triton (`pn122_cinta`, `arbol_conv`) lo esquivan con PTX en línea. Si se
   fusionan kernels de los dos mundos, el kernel resultante hereda fast-math, y hay que revisar
   esas secciones a mano.
4. **No hay `-lineinfo`, `-maxrregcount` ni carveout en `ptx_lab`.** Los límites de registros salen
   solo de `__launch_bounds__`. Para perfilar con Nsight Compute a nivel de línea hay que cambiar
   el `if False` de `ptx_lab.py:90,104`, y como es parte de la clave de caché, eso genera PTX
   nuevo.
5. **El repo ya tiene su propio microbenchmark de tensor cores**, `kernels/cuda/lab_mma.cu`: un warp
   repite `mma` s8 m16n8k32 contra s4 m16n8k64 sobre registros fijos. Es el equivalente local del
   `arithmtest` de `gpu-micro-bench` y sirve para la medición 6.1 extendiéndolo a HMMA f16 con
   acumulador f16 y f32.
6. **Camino int4 de la KV (`sk18i_*`, espejo int8 de la ventana reciente):** completo, pero **no
   corre en producción**, que usa `int8_per_token_head`. Queda fuera de la priorización.

---

## 3. Análisis por kernel

### 3.1 SK-30: decode de atención en una pasada (`kernels/cuda/sk30_decode_1pasada.cu`)

**Clasificación: memory-bound y disperso**, lectura de KV paginada. En el caso árbol queda **cerca
del punto de equilibrio** (ver la brecha).

**Estado actual**

- **Grilla y bloques:**
  - grilla `(GMAX=41, B·NH)` con 8 warps (`NQ=2`);
  - el reparto balanceado (`BAL=1`) deja unos 82 bloques con trabajo, una sola ola de 1 bloque por
    SM;
  - shared dinámica `SH_30 = 86 528 B` (`sk18_attn.py:134`), lo que da **1 bloque/SM, 8 de 48 warps
    (16,7 %)**.
- **Dtypes y MMA:**
  - Q·K con `mma.m16n8k32 s8·s8→s32`;
  - softmax en fp32 (`exp2f`, flash-decoding);
  - P·V con `PV8=1, PHL=1`: P' en 16 bits partido en dos planos int8, o sea **dos** MMA s8 por tile;
  - salida parcial O/l en fp16, combinada en `sk30_union`.
- **Carga:**
  - `cp.async` de 16 B en doble buffer (`__pipeline_wait_prior(1)`);
  - K con `.cg` (sin L1);
  - V con `.ca` (se aloja en L1 sin reúso posible);
  - escalas con `.ca` de 4 B.
- **Carveout:** no se fija. El driver queda forzado a 100 KB de shared y 28 KB de L1, lo cual es
  coherente porque K esquiva L1 igual.
- **Fusiones existentes:** la rotación Hadamard de q/k ya está dentro de `prep` y `escribir`
  (`ROT_PTX=True`), y la desrotación de la salida dentro de `union`.
- **Pasos que siguen separados** en cada capa de atención, todos chicos:
  - `escribir` (actualización de KV);
  - `prep30` (cuantizar Q);
  - `sk30` (atención);
  - `sk30_union` (combinar parciales).

**Brecha**

- **Occupancy.** El marco advierte que en decode conviene priorizar occupancy. Acá la occupancy es
  mínima a propósito, y por Little no hace falta más (*estimado*):
  - para saturar 936 GB/s con ~0,6 µs de latencia hay que tener ~560 KB en vuelo, ~7 KB por SM;
  - SK-30 tiene en vuelo una etapa completa, ~33 KB por SM (K 16 KB + V 16 KB + escalas).
- **Equilibrio entre carga y cómputo.** El riesgo real no es la occupancy, es que se solapen mal la
  carga y el cómputo. Por tile de 64 keys (*estimado*):
  - Q·K: 64 × 64 × 256 × 2 = 2,1 MOP;
  - P·V en dos planos: 4,2 MOP;
  - unos 33 KB leídos, o sea **~190 op/B**;
  - el ridge int8 de la 3090 es 284 T / 936 G ≈ 300 op/B, y ~210 con un rendimiento realista de
    `mma.sync`.

  Con un solo bloque por SM no hay otro bloque que esconda las barreras (`__syncthreads` por tile)
  ni la fase de softmax con `exp2f` en la SFU. Así, carga y cómputo se serializan en parte. La
  variante `PING` (Q·K del tile i+1 junto con el softmax del tile i, anillo de 3 para K) ya está
  escrita y está apagada (`PING 0`).
- **Filas útiles.** En el árbol se usan 54 de 64 filas (9 tokens × 6 cabezas), un 84 %.
- **`.ca` en V.** Usar `.ca` en las cargas de V no aporta nada: no hay reúso de V dentro del bloque
  ni entre bloques.

**Oportunidades** (en orden)

1. Medir `PING=1` contra `PING=0` en la configuración de producción. Esfuerzo bajo: es un `-D`.
2. Evaluar `PHL=0` (P·V en un solo plano int8) o `PV8=0` (P·V en fp16 con acumulador fp16,
   142 TFLOPS). Bajaría la intensidad hacia la zona claramente memory-bound. Es un compromiso de
   calidad que el repo midió (`PHL` existe para eliminar el error de P), así que hay que volver a
   evaluar KL y benchmarks.
3. Fusionar `prep30` en el prólogo de `sk30`, y la combinación de `sk30_union` con el patrón
   «último bloque reduce» (contador atómico). Ahorra 2 lanzamientos por capa.
   - Los parciales de `union` son ~2,7 MB y entran en L2, así que el ahorro es latencia, no
     ancho de banda.
   - En el árbol, 41 bloques por cabeza recalcularían la cuantización de Q (16 KB fp16 desde L2,
     barato), pero cambia la semántica de las escalas de fila: hay que verificar que el resultado
     sea bit a bit igual.
4. `cp.async.cg` para V (cosmético; sirve para limpiar las métricas de L1).

### 3.2 SK-18h `batch2`: decode de atención de respaldo (`kernels/cuda/sk18h_batch2.cu`)

- **Clasificación:** memory-bound.
- **Estado:**
  - lee K **dos veces** (pasada A para el máximo exacto, pasada B para pesos y w·V);
  - shared de 77 KB (`SH_H`) o 89 KB (`sh_h(8)`), 1 bloque/SM;
  - el comentario de SK-30 dice que llegaba al **55 % del ancho de banda** a 62k (*medido (repo)*).
- **Brecha:** solo corre si `tree_drafting` está apagado, o si `MB ≠ 64`. Con la configuración de
  producción no está en el camino caliente.
- **Oportunidad:**
  - si se piensa soportar MTP o decode sin árbol, generalizar SK-30 para `MB=32`;
  - si no, documentar que `batch2` es solo respaldo.

  Prioridad baja.

### 3.3 Atención de prefill: decuantizar y luego FlashInfer (`sk18_attn.py:921-1006`)

**Clasificación: compute-bound y determinístico** (tiling fijo de FlashAttention-2 paginado).

**Estado actual**

- **Flujo:**
  - `sk18h_decuant` convierte las páginas que toca el paso, de int8 a fp16, en un buffer fijo de
    `PAG_TOPE=64` páginas (~115 MB por GPU);
  - FlashInfer hace el prefill paginado en fp16, con acumulador fp32;
  - los contextos que no entran se procesan por tramos y se combinan por LSE.
- **Q:** se rota con FWHT y se pasa a fp16.
- **Respaldo:** `unified_attention` de Triton con configuraciones afinadas para sm_86 (PN124):
  `BLOCK_M=64, TILE=64, stages=1`, 29,1 ms contra 82,4 ms (*medido (repo)*).
- **Viaje extra por memoria (decuantización):**
  - por cada chunk se lee la KV int8 del contexto previo, se escribe en fp16 y FlashInfer la
    vuelve a leer;
  - son ~5,1 KB por token de contexto por capa, contra ~1 KB si se leyera int8 directo.

**Brecha**

- **El viaje extra no es lo que importa** (*estimado*, contexto C, chunk Lq, por capa):
  - decuantizar cuesta ~5,5 ns × C;
  - el cómputo ideal a 71 TFLOPS cuesta 4·Lq·256·12 / 71 T ≈ 152 ns × C para Lq=880;
  - o sea ~3,6 % del tiempo con chunk de 880 y ~1,2 % con chunk de 2640.
- **Lo que importa es el tipo de tensor core.**
  - Con 100k de contexto y chunk de 880, la atención cuesta del orden de ~240 ms por chunk sumando
    las 16 capas (*estimado*, ideal). Eso es comparable o mayor que los GEMM W4A8 del mismo chunk:
    a contexto largo, el **prefill está dominado por la atención**.
  - Esa atención corre en HMMA con acumulador fp32 (71 TFLOPS), mientras K ya está en int8 con
    escala por token-cabeza y Q ya está rotada, que es justo la precondición de un Q·K al estilo
    SageAttention en IMMA (284 TOPS).
  - P·V podría ir en fp16 con acumulador fp16 (142 TFLOPS), con el mismo truco `ACC16` que ya usa
    SK-30.
- **Fusión:** QKᵀ → softmax → ×V ya está fusionado (FlashInfer o Triton). No hay atención naive.
- **Dtype:** FP16 en todos lados. No hay BF16 innecesario y no hay TF32.

**Oportunidad**

- **Kernel de prefill propio con Q·K en int8 leyendo directo el layout de PN131** (sin
  decuantizar), con softmax en fp32 y P·V en fp16 con acumulador fp16.
  - Techo teórico: hasta ~2-4× en el tramo Q·K y ~2× en P·V; en total ~2× de la atención de
    prefill a contexto largo (*estimado*).
  - Esfuerzo alto. Exige evaluar calidad: KL contra BF16, y los benchmarks de tool-calls que el
    repo ya cuida en el espejo int8.
- **Efecto secundario:** elimina el buffer de decuantización de ~115 MB por GPU, lo que deja más
  bloques de KV.

### 3.4 Marlin W4A8 con escalas int16 (`kernels/marlin_s16/`)

**Clasificación:** con el árbol de 9 tokens, M = 9·B.

- **Decode (M = 9-36 con grafo, hasta 99 sin grafo): memory-bound.** Streaming de pesos int4 sin
  reúso, más de 6,7 GB por rank y por paso (*estimado*).
- **Prefill (M ≥ 512): compute-bound y determinístico.**

**Estado actual**

- **Instrucciones:**
  - `mma.m16n8k32 s8·s8→s32 .satfinite`;
  - `ldmatrix.x4`;
  - `cp.async.cg` de 16 B para A, B y escalas, con 4 etapas y 2 en vuelo;
  - acumulación entera hasta el epílogo (escala de grupo int16 aplicada en int32, después
    `int2float` × escala por token) y salida en fp16.
- **Grilla:** `sms × 1` = 82 bloques de 256 hilos. `MAX_BLOQUES_POR_SM=1` está justificado con una
  tabla medida con reloj fijo: con 2 bloques/SM era hasta 43 % peor (`marlin.cu:140-168`).
- **Límite de M:** tope de M=80 por bloque. Con `thread_m_blocks ≥ 6` hay derrame de registros:
  23,4 M sectores leídos de memoria local (`marlin.cu:492-511`).
- **Shared:**
  - la configuración de decode usa ~41 KB y **pide los 99 KB** (`max_shared_mem_new` no se ajusta
    con 1 bloque/SM, `marlin.cu:574-575, 612, 620`);
  - no hay carveout ni `__launch_bounds__`;
  - con 242-255 registros × 256 hilos, el banco de registros ya limita a 1 bloque/SM.
- **Fusiones:**
  - la cuantización de activaciones viene fusionada *antes*, en SK-26, SK-23 y SK-25;
  - escalas, bias y reducción split-K van dentro del GEMM;
  - el all-reduce va *después* (sk24 o int8+NCCL).

**Brecha**

- **Ancho de banda en decode** (*medido (repo)*, con reloj fijo, `marlin.cu:161-165`):
  - 5120×5120: 25,5 µs ≈ **514 GB/s**;
  - 17408×5120: 74,2 µs ≈ **600 GB/s**;
  - 7168×5120 con M=32: 40 µs ≈ **460 GB/s**;
  - en cambio el lm_head grande (328 MB en 380 µs) llega a ≈ 863 GB/s.

  Las matrices medianas pagan el arranque del pipeline, la cadena de barreras y locks del split-K
  global (reducción en fp32 a `C_tmp`) y la cola del reparto en franjas. Como en decode los pesos
  son la mayor parte de los bytes del paso, **esta es la palanca más grande del decode**: pasar de
  ~550 a ~800 GB/s en las lineales recortaría del orden de un 25-30 % del paso (*estimado*).
- **Lote alto:** con más de 8 pedidos en el árbol, M > 80 y el peso se relee entero en dos pasadas.
  Con más de 4 pedidos además no hay grafo CUDA (los tamaños capturados llegan a 40). Ese régimen
  no aparece en las tablas medidas.
- **Carveout y shared pedida de más:** hoy no limitan la occupancy (mandan los registros). El costo
  es indirecto: un bloque Marlin ocupa toda la shared de su SM, y nada puede convivir en el SM
  mientras corre (por ejemplo, un all-reduce o una norma en otro stream).
- **L2:** no hay política. Los pesos (streaming) pueden desalojar de los 6 MB de L2 a `C_tmp`, los
  locks y A, que sí se releen.

**Oportunidades**

1. Un GEMV/GEMM específico para M ≤ 16-36: stream-K o split-K con reducción atómica, tiles más
   chicos y más etapas en vuelo. Objetivo: llevar las matrices medianas a ≥ 80 % de DRAM. Esfuerzo
   alto, impacto alto.
2. Política de L2 por carga: `createpolicy.fractional.L2::evict_first` en B (pesos) y `evict_last`
   en `C_tmp`, locks y A. Esfuerzo bajo, impacto incierto: hay que medirlo.
3. Pedir `cache_size` en vez de `max_shared_mem` (`marlin.cu:612/620`). Esfuerzo bajo. Solo rinde
   si después se superponen kernels en streams.
4. Para M de 81 a 99: evaluar `thread_m_blocks=7` con `__launch_bounds__(256,1)` y acumulación
   reorganizada para no derramar, o asumir que ese régimen corre sin grafo y medir cuánto pesa.

### 3.5 lm_head int4 (`lm_head_int4.py`)

- **Clasificación:** memory-bound. N=124160, K=5120, grupo 128, sobre el mismo Marlin.
- **Estado:** 380 µs contra 749 µs del fp8 (*medido (repo)*), ≈ 863 GB/s, prácticamente en el techo.
- **Brecha:** en el camino genérico, la cuantización de la activación va en dos kernels separados
  (`per_token_quant_int8` y la multiplicación de escala, `marlin_utils.py:737-738`).
- **Oportunidad:** usar el mismo SK-26 (norma final → int8) que ya alimenta a las otras lineales.
  Impacto bajo, unos µs por paso.

### 3.6 Cadena GDN de decode en árbol: `arbol_conv` → `gdn_arbol` → `pn122_cinta` → SK-25

**Clasificación: limitada por latencia.** Recurrencia serial por token, reducciones por shuffle y
pocos bytes. No es compute-bound, y tampoco es memory-bound en sentido estricto.

**Estado actual**

- **`gdn_arbol`:**
  - grilla `(8, 24·N)`, 128 hilos, `__launch_bounds__(128)`, ~19 KB de shared estática;
  - estado **fp16** (786 KB por capa y por slot), cuentas en fp32;
  - ya fusiona l2-norm, compuertas g/β, reproducción de la cinta y máscaras del árbol;
  - lee q/k/v como vistas con stride (PN164, sin copias).
- **`arbol_conv`:**
  - bloques de 1 warp, grilla `(N, 160)`;
  - cargas escalares de 2 B (`FPT=1`, el mejor valor medido);
  - escribe `out` [9N, 5120] fp16, que `gdn_arbol` relee.
- **`pn122_cinta`:** relee k/v/a/b y vuelve a calcular la l2-norm, g y β, que `gdn_arbol` ya había
  calculado, para escribir la cinta.
- **Cinta:** se reserva en **fp32** (`gdn_cinta.py:323`), 16,6 KB por token-fila. El docstring
  (`:18`) habla de ~8 KB, que corresponde a fp16.

**Brecha**

- Los intermedios de esta cadena son chicos: 9 tokens × 5120 × 2 B ≈ 92 KB por pedido. **Ya caen en
  L2.** El costo de no fusionar es de lanzamiento, cola y dependencia entre kernels (unos µs cada
  uno, × 48 capas), no de DRAM.
- **El marco de «fusionar para no pagar el viaje L1→L2→DRAM» aplica en su versión de latencia, no
  en su versión de ancho de banda.**
- El estado de 786 KB por capa y por pedido sí va a DRAM en cada paso: ~75 MB por pedido y por paso
  sumando lectura y escritura (*estimado*), chico frente a los pesos.

**Oportunidades**

1. Meter `pn122_cinta` en el epílogo o prólogo de `gdn_arbol`: los bloques con `i_v==0` escriben
   k, g y β, y cada bloque su tajada de v. Ahorra 48 lanzamientos por paso y una lectura. Esfuerzo
   bajo-medio. Ojo: hoy el orden de reducción es bit a bit igual al de Triton, y con la fusión
   dejaría de serlo.
2. Meter la conv causal + SiLU de `arbol_conv` en el prólogo de `gdn_arbol`. Esfuerzo medio: el
   estado de la conv es por feature y los features de q/k se comparten entre 3 cabezas v, así que
   hace falta una regla de pertenencia.
3. Cinta en fp16: ahorra ~40 MB de VRAM por rank que pueden ir a KV (*estimado*: 11 slots × 9 × 48
   capas). Hay que validar la precisión de la reproducción de la cinta.

### 3.7 GDN de prefill (FLA Triton vía `streaming_gdn_driver.py`)

- **Clasificación:** compute-bound por chunks de 64 tokens y determinístico.
- **Estado:**
  - pipeline `cumsum → kkt → solve_tril → w_u → h → o`;
  - los productos en fp32 usan `allow_tf32=False` o `input_precision="ieee"` (`FLA_TRIL_PRECISION`
    por defecto `ieee`): **TF32 está apagado a propósito por precisión**, lo que es coherente con el
    marco;
  - el driver procesa en ventanas para no materializar h completo cuando T > 256;
  - el estado se junta y se dispersa en fp16 (`ssm_state[idx]` y `.to(fp16)`).
- **Ojo con TF32:** `VLLM_FLOAT32_MATMUL_PRECISION=high` activa TF32 para cualquier matmul fp32 de
  *torch* (no de Triton). Conviene confirmar que ninguna matmul fp32 de torch del camino caliente
  dependa de esa precisión, o dejarlo explícito.
- **Brecha y oportunidad:** baja frente a la atención de prefill. Es candidato a medir antes que a
  tocar.

### 3.8 Kernels fusionados de norma, cuantización y Hadamard (SK-26, SK-23, SK-25) y all-reduce (SK-24)

**Clasificación:** memory-bound elementwise. En decode quedan **limitados por latencia**.

**Estado**

- **SK-26:** suma de residuo + RMSNorm + int8 por token. 640 hilos, `uint4`.
- **SK-23:** SiLU·mul + Hadamard de 512 + int8. Un warp por bloque de 512, butterflies con
  `shfl_xor`.
- **SK-25:** norma con compuerta o sigmoide + Hadamard por cabeza + int8.

Los tres:

- tienen una grilla `(T,1)` con un bloque por token;
- usan cargas vectorizadas;
- no usan shared salvo para la reducción.

**SK-24:** all-reduce P2P de 4-8 bloques con bandera de época, en int8 por grupos de 64.

**Brecha**

- La cuantización por token necesita el absmax de toda la fila, y eso impide fusionarla en el
  epílogo de Marlin, que reparte la fila entre bloques. Por eso cada lineal tiene un kernel
  separado antes.
- **Con T = 9·B filas y un bloque por fila, el decode lanza solo 9-36 bloques: menos de la mitad de
  los SM.** Son kernels de latencia pura.

**Oportunidades**

1. Fusionar SK-24 con SK-26 (all-reduce + residuo + norma + int8) en decode: el bucle de suma de
   SK-24 ya tiene todos los elementos. Ahorra un lanzamiento y una ida y vuelta por cada lineal
   row-parallel (2 por capa × 64 capas). Esfuerzo medio: SK-24 usa 4-8 bloques y SK-26 necesita la
   fila entera por bloque.
2. En SK-25 para GDN: calcular la norma con compuerta y el Hadamard por cabeza dentro de
   `gdn_arbol`, y dejar solo el absmax y la cuantización por token como pasada chica.

### 3.9 `sk_fwht` (borrador, D=128) y otros chicos

- **`sk_fwht`:** cargas escalares de 2 B in place, una fila por warp. Se puede vectorizar (`uint2`)
  o meterlo en la RoPE del borrador. Impacto bajo.
- **`gdn_meta`:** Triton con `num_warps=1`, una vez por paso. Ya reemplazó ~35 ops de torch. Sin
  acción.

---

## 4. Carveout de shared memory: resumen por kernel

| Kernel | Shared/bloque | Categoría | Carveout implícito | ¿Coherente? |
|---|---|---|---|---|
| `sk30_decode` | 86,5 KB | memory-bound, reúso explícito en shared | 100 KB shared / 28 KB L1 | Sí. K esquiva L1 (`.cg`); el diseño es 1 ola con 1 bloque/SM. |
| `sk18h_batch2` | 77-89 KB | ídem | 100 / 28 | Sí (respaldo) |
| Marlin decode | usa 41, **pide 99** | memory-bound | 100 / 28 | La occupancy no cambia (registros), pero bloquea la convivencia en el SM. |
| Marlin prefill | ~51, pide 99 | compute-bound, tiling fijo | 100 / 28 | Sí, coherente con maximizar shared en GEMM. |
| FlashInfer / Triton prefill | lo fija la biblioteca | compute-bound | — | — |
| `gdn_arbol` | ~19 KB estática | latencia | el driver puede elegir 32/96 | Sí |
| SK-23/25/26, `arbol_conv`, `pn122` | ~0 | elementwise | máxima L1 | Sí. Igual no hay reúso que aprovechar en L1. |

Fijar `PreferredSharedMemoryCarveout` explícito aporta poco: el driver ya llega a la misma
configuración. La única decisión con peso real es lo que se *pide*, como en Marlin.

---

## 5. Técnicas mencionadas como oportunidad (no implementar todavía)

- **Prefetch asíncrono de KV a L2 solapado con cómputo** (arXiv:2504.06319, 2,15× medido en
  Hopper):
  - en decode, potencial **bajo**: lo que corre antes de la atención también es memory-bound, y la
    L2 (6 MB) es diez veces más chica que la KV de una capa;
  - lo único plausible es prefetchear los primeros tiles de KV de la capa durante los kernels chicos
    de latencia que vienen justo antes (`prep30`, `escribir`), cuando la DRAM está ociosa:
    ganancia de unos pocos µs por capa.
- **`cp.async` con bypass según patrón:**
  - SK-30 y Marlin ya usan `.cg`, que esquiva L1 para el streaming;
  - lo pendiente es la política de L2 (`evict_first` para el streaming de pesos y KV, `evict_last`
    para lo que se relee: parciales, locks, estado GDN entre capas), y pasar V a `.cg` en SK-30.
- **INT8 para la KV:** ya está hecho (`int8_per_token_head` con rotación). La extensión pendiente
  es usar esa KV int8 *también en la aritmética del prefill* (3.3), no solo en el decode.

---

## 6. Qué medir antes de decidir (paso posterior y separado)

Condiciones de medición:

- Igual que en las tablas del repo: **reloj de GPU fijo** (por ejemplo 1400 MHz), procesos
  alternados y varias repeticiones con dispersión informada.
- Para atribuir métricas a líneas de fuente en ncu hay que activar `-lineinfo` en `ptx_lab.py`. Eso
  ya es un cambio de código y pertenece a la etapa de medición, no a este informe.

### 6.1 Microbenchmarks de plataforma (`arabel1a/gpu-micro-bench`)

| Prueba | Para decidir |
|---|---|
| `arithmtest`: IMMA s8 m16n8k32 contra HMMA f16 con acumulador f32 y con acumulador f16, en modo throughput | Confirma en *esta* placa la relación 4:1:2 que justifica la oportunidad 3.3, y el ridge real de SK-30 (3.1). |
| Pointer chasing con `cn` (L2 vaciada) y `cg` contra `ca` | Latencia DRAM y L2 reales, para recalcular el «bytes en vuelo» de Little de SK-30 y Marlin. |
| Prueba de solapamiento de latencia DRAM entre warps | Si con 8 warps/SM y `cp.async` de 2 etapas ya se satura DRAM, o si hace falta `PING` o más etapas. |

### 6.2 Nsight Compute, por kernel

| Kernel | Métricas |
|---|---|
| `sk30_decode` (B=1 y 4; contexto 8k, 60k y 100k) | `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_tensor_op_imma_cycles_active`, stalls `long_scoreboard` contra `barrier`, `launch__occupancy_limit_shared_mem`, `l1tex__t_sector_hit_rate` (V con `.ca`) y `lts__t_sector_hit_rate`. Comparar `PING` 0 contra 1, y `PHL` 0 contra 1. |
| `marlin_gemm_s16` (5120×5120, 7168×5120, 17408×5120, 2·8704×5120; M=9, 18, 36, 81, 99) | `dram__throughput`, `lts__t_sector_hit_rate` de A y `C_tmp`, `smsp__warp_issue_stalled_*` (barrera de locks), `local_load/store` sectors (derrame), `achieved_occupancy`. |
| `sk18h_decuant` + FlashInfer prefill (contexto 32k, 64k y 100k; chunk 880 y 2640) | Tiempo de decuantización contra tiempo de atención; `sm__pipe_tensor_op_hmma_cycles_active` (cuánto falta para el techo de 71 T). |
| `gdn_arbol`, `pn122_cinta`, `arbol_conv`, SK-23/25/26, SK-24 | Duración contra `gpu__time_duration` mínima posible (~2-3 µs), `sm__throughput` y `lts__t_sector_hit_rate` (confirmar que los intermedios caen en L2). |

### 6.3 Nsight Systems, por paso de decode

- Cantidad de kernels por paso y suma de *gaps* entre kernels dentro del grafo, contra la suma de
  duraciones. Si los kernels de menos de 10 µs más sus gaps superan ~15 % del paso, la oportunidad
  3 de la tabla (fusiones) sube de prioridad.
- El mismo perfil con 5-11 pedidos (fuera de grafo): cuánto pesa el lanzamiento por ctypes y Python.

### 6.4 Memoria pico (advertencia de KernelBench-Verified)

Para cada oportunidad implementada:

- registrar `torch.cuda.max_memory_allocated` y la cantidad de bloques de KV que reserva vLLM al
  arrancar (el informe de dimensionamiento de `analisis_arranque.py`);
- una mejora de latencia que agrande buffers (decuantización, parciales, cinta) se paga en contexto
  servible;
- comparar siempre contra la base fuerte que ya existe (FlashInfer, Marlin upstream y Triton
  afinado), no contra torch sin optimizar;
- para cambios que tocan la salida numérica (`PHL`, `PV8`, int8 en prefill, cinta fp16), correr la
  evaluación de KL contra BF16 y los benchmarks de la tarjeta del modelo.

---

## 7. Estado de implementación (rama `claude/clever-ride-uycaba`)

Todo se escribió sin GPU. Cada kernel modificado se compiló para sm_86 (NVRTC 12.8 + ptxas) y la
lógica del plan del prefill entero se verificó en CPU, pero **nada se ejecutó en una 3090**. Los
tests y el orden de validación están en `tests/idiotsavant/README.md`
(`bash tests/idiotsavant/correr_todo.sh <dir>`).

| Oportunidad (sección) | Estado | Por defecto |
|---|---|---|
| `-lineinfo` y recursos de ptxas para perfilar (2.1, 6) | `ptx_lab`: `VLLM_IDIOTSAVANT_PTX_LINEINFO=1` y `Kernel.recursos()`; `tools/idiotsavant/ptx_recursos.py` | — |
| SK-30: derrame de registros (3.1) | `#pragma unroll 1` en la suma de `seq_len`: **255 → 177 registros, derrame 28 B → 0**, bit a bit igual | prendido |
| SK-30: V con `.cg` (3.1, 5) | `SK30_VCG`, bit a bit igual | prendido |
| SK-30: `PING` (3.1) | arreglada la shared que pedía (`sh_30()`); `SK30_PING` | apagado: medir |
| SK-30: `PHL=0` (3.1) | `SK30_PHL` | apagado: cambia la cuenta |
| SK-30: 16 warps (`NQ=4`) | `SK30_NQ`, 128 registros y 16 B de derrame | apagado: medir |
| `sk18h_decuant` de 1 warp y V sin coalescer (2.1) | `sk18h_decuant_tile`: 8 warps, V contigua traspuesta en shared, 5 bloques/SM; bit a bit igual | prendido (`DECUANT_TT = 32`) |
| Marlin: hint de L2 para el peso (3.4) | `MARLIN_L2_HINT` (define de build) | apagado: medir |
| Marlin: shared pedida de más (3.4, 4) | `MARLIN_SHARED_JUSTA` (define de build) | apagado: medir |
| Prefill de atención en int8 (3.3) | `PREFILL_SK30`: SK-30 sobre pseudo-secuencias de 10 tokens, sin decuantizar | apagado: cambia la cuenta |

### No implementado, y por qué

- **GEMV/GEMM W4A8 propio para M chico (3.4, oportunidad 1).** Es la palanca más grande del decode,
  pero exige un kernel nuevo que lea el layout reempaquetado de Marlin (o un segundo
  reempaquetado, con el doble de pesos en memoria). Escribirlo sin poder correrlo era un riesgo
  sobre todas las lineales del modelo. `test_marlin_s16.py --medir` deja la línea base (µs y GB/s
  por forma y M) para cuando se encare.
- **`pn122_cinta` dentro de `gdn_arbol` (3.6).** `gdn_arbol` reproduce las filas de la cinta del paso
  anterior y `pn122` escribe las del paso actual **en el mismo slot**. Dentro de un solo kernel no
  hay orden entre bloques, así que fusionarlos es una carrera, salvo que se duplique la cinta por
  paridad de paso (+80 MB por rank).
- **Cinta en fp16 (3.6).** La leen o escriben siete kernels (Triton y PTX: `_k_spec`,
  `_k_spec_arbol`, `_k_escribir`, `_k_escribir_par`, `_k_materializar`, `gdn_arbol`, `pn122`). El
  ahorro es ~40 MB por rank. No compensa el riesgo de tocarlos a ciegas.
- **`sk30_union` dentro de `sk30` («último bloque reduce», 3.1).** Un solo bloque tendría que
  combinar 41 grupos × 54 filas × 256 dims (~1 MB) a ancho de banda de un SM: más lento que el
  lanzamiento aparte que ahorra.
- **`sk24` + `sk26` (3.8).** SK-24 reparte los elementos entre 4-8 bloques y SK-26 necesita filas
  enteras por bloque. Fusionarlos implica rediseñar el protocolo P2P, que no se puede probar sin
  las dos placas.
- **`escribir2` + `prep2`.** Corren en dos ganchos distintos de vLLM (`do_kv_cache_update` y
  `forward`), y el primero no recibe la query.
