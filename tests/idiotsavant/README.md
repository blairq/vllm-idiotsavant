# Tests de los kernels del camino sm_86

Tests y mediciones de los kernels PTX y del Marlin S16 que usa vllm-idiotsavant con
qwen3.8_27b_idiotSavant_sm_86 en 2× RTX 3090. Corren **en la placa**, dentro de la imagen del fork.
No levantan vLLM: arman una capa sintética con las formas del modelo (`atencion_sintetica.py`).

| Archivo | Qué prueba | Cambio que valida |
|---|---|---|
| `tools/idiotsavant/ptx_recursos.py` | registros, derrame, shared y bloques por SM de cada kernel PTX (sin GPU) | todos |
| `test_sk30.py` | variantes de SK-30 bit a bit iguales, error contra fp32, tiempos | `SK30_VCG`, `SK30_PING`, `SK30_PHL`, `SK30_NQ`, unroll 1 |
| `test_decuant.py` | `sk18h_decuant_tile` bit a bit igual al kernel viejo, GB/s | `DECUANT_TT` |
| `test_marlin_s16.py` | exactitud del GEMM W4A8 y GB/s del peso por forma y M | `MARLIN_L2_HINT`, `MARLIN_SHARED_JUSTA` |
| `test_prefill_sk30.py` | prefill entero contra decuantizar+FlashInfer: error contra fp32 y ms | `PREFILL_SK30` |

## Orden sugerido

Con el reloj fijo las diferencias chicas se leen mejor:

```bash
sudo nvidia-smi -i 0 -lgc 1400        # y al final: sudo nvidia-smi -i 0 -rgc
```

1. Recursos (sin GPU), para tener la foto de esta rama:

   ```bash
   python3 tools/idiotsavant/ptx_recursos.py --json recursos.json
   ```

   `sk30_decode` tiene que mostrar derrame 0.

2. Exactitud de todo lo que está prendido por defecto:

   ```bash
   pytest -q -s tests/idiotsavant/
   ```

3. Mediciones de lo que está apagado y hay que decidir:

   ```bash
   python3 tests/idiotsavant/test_sk30.py --medir           # ping / phl0 / nq4 contra produccion
   python3 tests/idiotsavant/test_decuant.py --medir
   python3 tests/idiotsavant/test_prefill_sk30.py --medir   # entero contra FlashInfer
   python3 tests/idiotsavant/test_marlin_s16.py --medir > marlin_base.txt
   ```

4. Marlin con defines opcionales: recompilar en un proceso y medir en otro.

   ```bash
   IDIOTSAVANT_MARLIN_DEFS="-DMARLIN_L2_HINT=1" python3 -c \
       "from vllm.idiotsavant import marlin_s16; marlin_s16.construir()"
   pytest -q tests/idiotsavant/test_marlin_s16.py
   python3 tests/idiotsavant/test_marlin_s16.py --medir > marlin_l2.txt
   ```

   La reducción por `atomicAdd` de upstream se prueba con la variable de vLLM, sin recompilar:

   ```bash
   VLLM_MARLIN_USE_ATOMIC_ADD=1 python3 tests/idiotsavant/test_marlin_s16.py --medir
   ```

5. De punta a punta: prender lo que ganó en `vllm/idiotsavant/sk18_attn.py` (o en `build.py`, para
   Marlin), reconstruir la imagen y correr la batería contra el servidor:

   ```bash
   python3 tools/idiotsavant/bateria.py --url http://localhost:8320 --salida despues.json
   ```

   Comparala contra una corrida de la imagen anterior. Mirá sobre todo `det`, `tool` y `largo`.

`tests/idiotsavant/correr_todo.sh` hace los pasos 1 a 3 y deja todo en un directorio.

## Qué decide cada medición

- **`SK30_PING`**: es bit a bit igual. Prenderlo si `ping` mide menos que `produccion` con 60k y
  100k de contexto.
- **`SK30_NQ = 4`**: difiere solo en redondeo. Prenderlo si `nq4` gana.
- **`SK30_PHL = 0`**: cambia la cuenta. Prenderlo solo si gana tiempo **y** la KL contra BF16 y la
  batería no empeoran.
- **`PREFILL_SK30`**: cambia la cuenta (Q en int8 en el prefill). Mismo criterio que `PHL`, más el
  tiempo de prefill en contexto largo (grupo `largo` de la batería).
- **`MARLIN_L2_HINT`** y **`MARLIN_SHARED_JUSTA`**: bit a bit iguales en exactitud. Prenderlos si
  bajan los µs en las formas de decode (M = 9 a 36).
- **`DECUANT_TT`**: ya viene prendido en 32. Si el test bit a bit falla, `DECUANT_TT = 0` vuelve al
  kernel viejo.

Perfilar un kernel en Nsight Compute con atribución a líneas de fuente:

```bash
VLLM_IDIOTSAVANT_PTX_LINEINFO=1 ncu --set full --import-source yes -k regex:sk30_decode -c 10 \
    -o sk30 python3 tests/idiotsavant/test_sk30.py --medir --ctx 60000 --pedidos 1
```
