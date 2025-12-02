#!/usr/bin/env bash
# Run RSCF+MI experiments for final report
set -euo pipefail

echo "=========================================="
echo "Final Report: RSCF+MI Experiments"
echo "=========================================="
echo ""
echo "Running MED + RSCF + MI experiments"
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

# RSCF+MI experiments
RSCF_MI_CONFIGS=(
    "configs/final/fb15k237/complex_med_rscf_mi.yaml"
    "configs/final/fb15k237/rotate_med_rscf_mi.yaml"
    "configs/final/fb15k237/transe_med_rscf_mi.yaml"
    "configs/final/wn18rr/complex_med_rscf_mi.yaml"
    "configs/final/wn18rr/rotate_med_rscf_mi.yaml"
    "configs/final/wn18rr/transe_med_rscf_mi.yaml"
)

echo "Experiments to run: ${#RSCF_MI_CONFIGS[@]}"
echo ""

TOTAL=${#RSCF_MI_CONFIGS[@]}
CURRENT=0

for config in "${RSCF_MI_CONFIGS[@]}"; do
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
echo "✅ All RSCF+MI experiments complete!"
echo "========================================"
echo ""
echo "Results saved to: ./workdir/runs/final/"
echo ""
