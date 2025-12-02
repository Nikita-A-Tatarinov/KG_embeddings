#!/usr/bin/env bash
# Run RSCF experiments for final report
# Execute configs one by one for careful monitoring
set -euo pipefail

echo "=========================================="
echo "Final Report: RSCF Experiments"
echo "=========================================="
echo ""
echo "Running MED + RSCF experiments"
echo "Models: ComplEx, RotatE, TransE"
echo "Datasets: FB15k-237, WN18RR"
echo "Total: 6 experiments"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ No GPU - using CPU (slower)"
fi
echo ""

ROOT_DIR=$(dirname "$(dirname "${BASH_SOURCE[0]}")")

# RSCF experiments - ordered for systematic execution
RSCF_CONFIGS=(
    "configs/final/fb15k237/complex_med_rscf.yaml"
    "configs/final/fb15k237/rotate_med_rscf.yaml"
    "configs/final/fb15k237/transe_med_rscf.yaml"
    "configs/final/wn18rr/complex_med_rscf.yaml"
    "configs/final/wn18rr/rotate_med_rscf.yaml"
    "configs/final/wn18rr/transe_med_rscf.yaml"
)

echo "Experiments to run: ${#RSCF_CONFIGS[@]}"
echo ""

TOTAL=${#RSCF_CONFIGS[@]}
CURRENT=0

for config in "${RSCF_CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "========================================"
    echo "Progress: ${CURRENT}/${TOTAL}"
    echo "Config: $config"
    echo "========================================"
    echo ""
    
    python "${ROOT_DIR}/train.py" --config "${ROOT_DIR}/${config}"
    
    echo ""
    echo "✓ Completed: $config"
    echo ""
done

echo ""
echo "========================================"
echo "✅ All RSCF experiments complete!"
echo "========================================"
echo ""
echo "Results saved to: ./workdir/runs/final/"
echo ""
echo "Next: Run MI experiments with scripts/run_final_mi.sh"
echo ""
