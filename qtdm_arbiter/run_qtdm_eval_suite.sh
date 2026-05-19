#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

RUN_ID="eval-$(date +%s)"
OUT_DIR="output/eval_runs/${RUN_ID}"

echo "=== QTDM Comparative Eval Suite ==="
echo "Run ID: ${RUN_ID}"
echo "Output: ${OUT_DIR}"
echo ""

# Health check
if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "WARNING: retrieval service not running — baselines requiring live retrieval will refuse"
fi

# Run eval
python3 tools/eval_qtdm_comparative.py \
    --test-set tools/test_sets/seed.jsonl \
    --retrieval-url http://localhost:8000 \
    --run-id "${RUN_ID}" \
    --out-dir "${OUT_DIR}"

# Render report
python3 tools/render_eval_report.py \
    --results "${OUT_DIR}/results.json" \
    --out "${OUT_DIR}/report.md"

echo ""
echo "Eval complete. Report at: ${OUT_DIR}/report.md"
