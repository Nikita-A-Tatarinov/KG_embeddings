#!/usr/bin/env bash
# Run ALL WN18RR experiments (all models, all dimensions)
# Ready for GPU cluster with sampling enabled
set -euo pipefail

echo "=========================================="
echo "WN18RR: All Models (Sampled - Fast)"
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

# All WN18RR configs
CONFIGS=(
    # RotatE
    "configs/wn18rr/rotate_d10.yaml"
    "configs/wn18rr/rotate_d20.yaml"
    "configs/wn18rr/rotate_d40.yaml"
    "configs/wn18rr/rotate_d80.yaml"
    "configs/wn18rr/rotate_med.yaml"
    "configs/wn18rr/rotate_med_mi.yaml"
    "configs/wn18rr/rotate_med_rscf.yaml"
    "configs/wn18rr/rotate_med_rscf_mi.yaml"
    
    # ComplEx
    "configs/wn18rr/complex_d10.yaml"
    "configs/wn18rr/complex_d20.yaml"
    "configs/wn18rr/complex_d40.yaml"
    "configs/wn18rr/complex_d80.yaml"
    "configs/wn18rr/complex_med.yaml"
    "configs/wn18rr/complex_med_mi.yaml"
    "configs/wn18rr/complex_med_rscf.yaml"
    "configs/wn18rr/complex_med_rscf_mi.yaml"
    
    # DistMult
    "configs/wn18rr/distmult_d10.yaml"
    "configs/wn18rr/distmult_d20.yaml"
    "configs/wn18rr/distmult_d40.yaml"
    "configs/wn18rr/distmult_d80.yaml"
    "configs/wn18rr/distmult_med.yaml"
    "configs/wn18rr/distmult_med_mi.yaml"
    "configs/wn18rr/distmult_med_rscf.yaml"
    "configs/wn18rr/distmult_med_rscf_mi.yaml"
    
    # TransE
    "configs/wn18rr/transe_d10.yaml"
    "configs/wn18rr/transe_d20.yaml"
    "configs/wn18rr/transe_d40.yaml"
    "configs/wn18rr/transe_d80.yaml"
    "configs/wn18rr/transe_med.yaml"
    "configs/wn18rr/transe_med_mi.yaml"
    "configs/wn18rr/transe_med_rscf.yaml"
    "configs/wn18rr/transe_med_rscf_mi.yaml"
    
    # TransH
    "configs/wn18rr/transh_d10.yaml"
    "configs/wn18rr/transh_d20.yaml"
    "configs/wn18rr/transh_d40.yaml"
    "configs/wn18rr/transh_d80.yaml"
    "configs/wn18rr/transh_med.yaml"
    "configs/wn18rr/transh_med_mi.yaml"
    "configs/wn18rr/transh_med_rscf.yaml"
    "configs/wn18rr/transh_med_rscf_mi.yaml"
    
    # PairRE
    "configs/wn18rr/pairre_d10.yaml"
    "configs/wn18rr/pairre_d20.yaml"
    "configs/wn18rr/pairre_d40.yaml"
    "configs/wn18rr/pairre_d80.yaml"
    "configs/wn18rr/pairre_med.yaml"
    "configs/wn18rr/pairre_med_mi.yaml"
    "configs/wn18rr/pairre_med_rscf.yaml"
    "configs/wn18rr/pairre_med_rscf_mi.yaml"
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
echo "✅ All WN18RR experiments complete!"
