#!/usr/bin/env bash
# Run MED + RSCF, MED + MI, and MED + RSCF + MI experiments
# This script runs the three key combinations for Phase 1 validation
set -euo pipefail

echo "=========================================="
echo "MED Combinations Experiments"
echo "=========================================="
echo ""
echo "Running: MED + RSCF, MED + MI, MED + RSCF + MI"
echo "Models: RotatE, ComplEx, DistMult, TransE, TransH, PairRE"
echo "Datasets: FB15k-237, WN18RR"
echo ""
echo "Config: 10% training + 20% validation (sampled)"
echo "Expected time per config: ~40-50 min per 100 epochs"
echo "Total configs: 36 (6 models × 2 datasets × 3 combinations)"
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

# All MED combination configs (3 per model per dataset)
CONFIGS=(
    # ===============================================
    # FB15k-237
    # ===============================================
    
    # RotatE
    "configs/fb15k237/rotate_med_rscf.yaml"
    "configs/fb15k237/rotate_med_mi.yaml"
    "configs/fb15k237/rotate_med_rscf_mi.yaml"
    
    # ComplEx
    "configs/fb15k237/complex_med_rscf.yaml"
    "configs/fb15k237/complex_med_mi.yaml"
    "configs/fb15k237/complex_med_rscf_mi.yaml"
    
    # DistMult
    "configs/fb15k237/distmult_med_rscf.yaml"
    "configs/fb15k237/distmult_med_mi.yaml"
    "configs/fb15k237/distmult_med_rscf_mi.yaml"
    
    # TransE
    "configs/fb15k237/transe_med_rscf.yaml"
    "configs/fb15k237/transe_med_mi.yaml"
    "configs/fb15k237/transe_med_rscf_mi.yaml"
    
    # TransH
    "configs/fb15k237/transh_med_rscf.yaml"
    "configs/fb15k237/transh_med_mi.yaml"
    "configs/fb15k237/transh_med_rscf_mi.yaml"
    
    # PairRE
    "configs/fb15k237/pairre_med_rscf.yaml"
    "configs/fb15k237/pairre_med_mi.yaml"
    "configs/fb15k237/pairre_med_rscf_mi.yaml"
    
    # ===============================================
    # WN18RR
    # ===============================================
    
    # RotatE
    "configs/wn18rr/rotate_med_rscf.yaml"
    "configs/wn18rr/rotate_med_mi.yaml"
    "configs/wn18rr/rotate_med_rscf_mi.yaml"
    
    # ComplEx
    "configs/wn18rr/complex_med_rscf.yaml"
    "configs/wn18rr/complex_med_mi.yaml"
    "configs/wn18rr/complex_med_rscf_mi.yaml"
    
    # DistMult
    "configs/wn18rr/distmult_med_rscf.yaml"
    "configs/wn18rr/distmult_med_mi.yaml"
    "configs/wn18rr/distmult_med_rscf_mi.yaml"
    
    # TransE
    "configs/wn18rr/transe_med_rscf.yaml"
    "configs/wn18rr/transe_med_mi.yaml"
    "configs/wn18rr/transe_med_rscf_mi.yaml"
    
    # TransH
    "configs/wn18rr/transh_med_rscf.yaml"
    "configs/wn18rr/transh_med_mi.yaml"
    "configs/wn18rr/transh_med_rscf_mi.yaml"
    
    # PairRE
    "configs/wn18rr/pairre_med_rscf.yaml"
    "configs/wn18rr/pairre_med_mi.yaml"
    "configs/wn18rr/pairre_med_rscf_mi.yaml"
)

echo "Total experiments to run: ${#CONFIGS[@]}"
echo ""
echo "Estimated total time:"
echo "  - With sampling: ~24-30 hours"
echo "  - Full dataset: ~72-96 hours"
echo ""

read -p "Press Enter to start experiments (Ctrl+C to cancel)..."
echo ""

TOTAL=${#CONFIGS[@]}
CURRENT=0

for config in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "========================================"
    echo "Progress: ${CURRENT}/${TOTAL}"
    echo "Running: $config"
    echo "========================================"
    
    python "${ROOT_DIR}/train.py" --config "${ROOT_DIR}/${config}"
    
    echo ""
    echo "✓ Completed: $config"
done

echo ""
echo "========================================"
echo "✅ All MED combination experiments complete!"
echo "========================================"
echo ""
echo "Results saved to: ./workdir/runs/"
echo ""
echo "Next steps:"
echo "  1. Analyze results with: python tools/plot_logs.py"
echo "  2. Run full evaluation with: python evaluate.py --config <config_path> --checkpoint <path>"
echo ""
