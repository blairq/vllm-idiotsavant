#!/bin/bash
# Pasos 1-3 de tests/idiotsavant/README.md: recursos, exactitud y mediciones. Deja todo en $1.
#   bash tests/idiotsavant/correr_todo.sh resultados-$(git rev-parse --short HEAD)
set -u
OUT=${1:-resultados}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv > "$OUT/gpu.txt" 2>&1
git log --oneline -1 > "$OUT/commit.txt"
python3 tools/idiotsavant/ptx_recursos.py --json "$OUT/recursos.json" | tee "$OUT/recursos.txt"
pytest -q -s tests/idiotsavant/ 2>&1 | tee "$OUT/pytest.txt"
for t in test_sk30 test_decuant test_prefill_sk30 test_marlin_s16; do
    python3 "tests/idiotsavant/$t.py" --medir 2>&1 | tee "$OUT/$t.txt"
done
echo "listo: $OUT"
