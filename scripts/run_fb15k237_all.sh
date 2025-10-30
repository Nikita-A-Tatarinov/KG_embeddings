#!/usr/bin/env bash
# Run ALL FB15k-237 experiments (all models, all dimensions)
# Ready for GPU cluster with sampling enabled
set -euo pipefail

echo "=========================================="
echo "FB15k-237: All Models (Sampled - Fast)"
echo "=========================================="
echo ""
echo "Config: 10% training + 20% validation"
echo "Expected time: ~40-50 min per 100 epochs"
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

# All FB15k-237 configs
CONFIGS=(
    # RotatE
    "configs/fb15k237/rotate_d10.yaml"
    "configs/fb15k237/rotate_d20.yaml"
    "configs/fb15k237/rotate_d40.yaml"
    "configs/fb15k237/rotate_d80.yaml"
    "configs/fb15k237/rotate_med.yaml"
    "configs/fb15k237/rotate_med_mi.yaml"
    "configs/fb15k237/rotate_med_rscf.yaml"
    "configs/fb15k237/rotate_med_rscf_mi.yaml"
    
    # ComplEx
    "configs/fb15k237/complex_d10.yaml"
    "configs/fb15k237/complex_d20.yaml"
    "configs/fb15k237/complex_d40.yaml"
    "configs/fb15k237/complex_d80.yaml"
    "configs/fb15k237/complex_med.yaml"
    "configs/fb15k237/complex_med_mi.yaml"
    "configs/fb15k237/complex_med_rscf.yaml"
    "configs/fb15k237/complex_med_rscf_mi.yaml"
    
    # DistMult
    "configs/fb15k237/distmult_d10.yaml"
    "configs/fb15k237/distmult_d20.yaml"
    "configs/fb15k237/distmult_d40.yaml"
    "configs/fb15k237/distmult_d80.yaml"
    "configs/fb15k237/distmult_med.yaml"
    "configs/fb15k237/distmult_med_mi.yaml"
    "configs/fb15k237/distmult_med_rscf.yaml"
    "configs/fb15k237/distmult_med_rscf_mi.yaml"
    
    # TransE
    "configs/fb15k237/transe_d10.yaml"
    "configs/fb15k237/transe_d20.yaml"
    "configs/fb15k237/transe_d40.yaml"
    "configs/fb15k237/transe_d80.yaml"
    "configs/fb15k237/transe_med.yaml"
    "configs/fb15k237/transe_med_mi.yaml"
    "configs/fb15k237/transe_med_rscf.yaml"
    "configs/fb15k237/transe_med_rscf_mi.yaml"
    
    # TransH
    "configs/fb15k237/transh_d10.yaml"
    "configs/fb15k237/transh_d20.yaml"
    "configs/fb15k237/transh_d40.yaml"
    "configs/fb15k237/transh_d80.yaml"
    "configs/fb15k237/transh_med.yaml"
    "configs/fb15k237/transh_med_mi.yaml"
    "configs/fb15k237/transh_med_rscf.yaml"
    "configs/fb15k237/transh_med_rscf_mi.yaml"
    
    # PairRE
    "configs/fb15k237/pairre_d10.yaml"
    "configs/fb15k237/pairre_d20.yaml"
    "configs/fb15k237/pairre_d40.yaml"
    "configs/fb15k237/pairre_d80.yaml"
    "configs/fb15k237/pairre_med.yaml"
    "configs/fb15k237/pairre_med_mi.yaml"
    "configs/fb15k237/pairre_med_rscf.yaml"
    "configs/fb15k237/pairre_med_rscf_mi.yaml"
)

echo "Running ${#CONFIGS[@]} experiments..."
echo ""

for config in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Running: $config"
    echo "========================================"
    python "${ROOT_DIR}/train.py" --config "${ROOT_DIR}/${config}"
    echo ""
done

echo ""
echo "✅ All FB15k-237 experiments complete!"
