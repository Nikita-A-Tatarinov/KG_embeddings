#!/bin/bash
# Essential Stability Evaluation Script (512d configs only)
# Processes only the 16 most important configurations for report Table 3
# 
# Usage: bash scripts/run_stability_evaluation_essential.sh [--ultra-fast|--fast]
#   --ultra-fast: 20 samples, L=1 (~5 hours total)
#   --fast: 30 samples, L=1 (~8 hours total)
#   (default): 50 samples, L=1 (~16 hours total)

set -e  # Exit on error

# Parse arguments
if [[ "$1" == "--ultra-fast" ]]; then
    SAMPLES=20
    LAYERS=1
    echo "=== ULTRA-FAST MODE: samples=20, layers=1 ==="
elif [[ "$1" == "--fast" ]]; then
    SAMPLES=30
    LAYERS=1
    echo "=== FAST MODE: samples=30, layers=1 ==="
else
    SAMPLES=50
    LAYERS=1
    echo "=== ESSENTIAL MODE: samples=50, layers=1 ==="
fi

# Configuration
BASE_DIR="workdir/runs/final"
SCRIPT="evaluate_stability.py"

# Counter for progress tracking
TOTAL_CONFIGS=24
CURRENT=0

echo "Processing 24 essential configurations (512d + MED for both models)"
echo "Estimated time per checkpoint: ~20-60 minutes"
echo ""

# Function to run stability evaluation
run_stability() {
    local model=$1
    local dataset=$2
    local hf_dataset=$3
    local ckpt_path=$4
    local config_name=$5
    
    CURRENT=$((CURRENT + 1))
    
    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL_CONFIGS] Evaluating: $config_name"
    echo "Model: $model | Dataset: $dataset"
    echo "Checkpoint: $ckpt_path"
    echo "=========================================="
    
    # Check if checkpoint exists
    if [[ ! -f "$ckpt_path" ]]; then
        echo "⚠️  WARNING: Checkpoint not found, skipping..."
        return
    fi
    
    # Check if stability.json already exists
    local out_dir=$(dirname "$ckpt_path")
    local out_file="$out_dir/stability.json"
    
    if [[ -f "$out_file" ]]; then
        echo "✓ Stability results already exist, skipping..."
        return
    fi
    
    # Run evaluation
    echo "Running stability evaluation..."
    python $SCRIPT \
        --model "$model" \
        --ckpt "$ckpt_path" \
        --dataset "$dataset" \
        --use-hf \
        --hf-name "$hf_dataset" \
        --filtered \
        --batch-size 128 \
        --device auto \
        --skip-standard-eval \
        --compute-stability \
        --stability-samples $SAMPLES \
        --stability-layers $LAYERS \
        --out "$out_file"
    
    if [[ $? -eq 0 ]]; then
        echo "✓ Success: Results saved to $out_file"
    else
        echo "✗ Failed to evaluate $config_name"
    fi
}

# ============================================
# WN18RR - 512d configurations only
# ============================================
echo ""
echo "############################################"
echo "# WN18RR - Essential Configurations"
echo "############################################"

# RotatE 512d
run_stability "RotatE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/mi_512d/ckpt_final.pt" \
    "WN18RR/RotatE/MI-512d"

run_stability "RotatE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/rscf_512d/ckpt_final.pt" \
    "WN18RR/RotatE/RSCF-512d"

run_stability "RotatE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/mi_rscf_512d/ckpt_final.pt" \
    "WN18RR/RotatE/MI+RSCF-512d"

run_stability "RotatE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/med_mi/ckpt_final.pt" \
    "WN18RR/RotatE/MED+MI"

run_stability "RotatE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/med_rscf/ckpt_final.pt" \
    "WN18RR/RotatE/MED+RSCF"

run_stability "RotatE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/med_mi_rscf/ckpt_final.pt" \
    "WN18RR/RotatE/MED+MI+RSCF"

# TransE 512d
run_stability "TransE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/transe/mi_512d/ckpt_final.pt" \
    "WN18RR/TransE/MI-512d"

run_stability "TransE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/transe/rscf_512d/ckpt_final.pt" \
    "WN18RR/TransE/RSCF-512d"

run_stability "TransE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/transe/mi_rscf_512d/ckpt_final.pt" \
    "WN18RR/TransE/MI+RSCF-512d"

run_stability "TransE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/transe/med_mi/ckpt_final.pt" \
    "WN18RR/TransE/MED+MI"

run_stability "TransE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/transe/med_rscf/ckpt_final.pt" \
    "WN18RR/TransE/MED+RSCF"

run_stability "TransE" "WN18RR" "KGraph/WN18RR" \
    "$BASE_DIR/wn18rr/transe/med_mi_rscf/ckpt_final.pt" \
    "WN18RR/TransE/MED+MI+RSCF"

# ============================================
# FB15k-237 - 512d configurations only
# ============================================
echo ""
echo "############################################"
echo "# FB15k-237 - Essential Configurations"
echo "############################################"

# RotatE 512d
run_stability "RotatE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/mi_512d/ckpt_final.pt" \
    "FB15k237/RotatE/MI-512d"

run_stability "RotatE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/rscf_512d/ckpt_final.pt" \
    "FB15k237/RotatE/RSCF-512d"

run_stability "RotatE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/mi_rscf_512d/ckpt_final.pt" \
    "FB15k237/RotatE/MI+RSCF-512d"

run_stability "RotatE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/med_mi/ckpt_final.pt" \
    "FB15k237/RotatE/MED+MI"

run_stability "RotatE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/med_rscf/ckpt_final.pt" \
    "FB15k237/RotatE/MED+RSCF"

run_stability "RotatE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/med_mi_rscf/ckpt_final.pt" \
    "FB15k237/RotatE/MED+MI+RSCF"

# TransE 512d
run_stability "TransE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/mi_512d/ckpt_final.pt" \
    "FB15k237/TransE/MI-512d"

run_stability "TransE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/rscf_512d/ckpt_final.pt" \
    "FB15k237/TransE/RSCF-512d"

run_stability "TransE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/mi_rscf_512d/ckpt_final.pt" \
    "FB15k237/TransE/MI+RSCF-512d"

run_stability "TransE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/med_mi/ckpt_final.pt" \
    "FB15k237/TransE/MED+MI"

run_stability "TransE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/med_rscf/ckpt_final.pt" \
    "FB15k237/TransE/MED+RSCF"

run_stability "TransE" "FB15k-237" "KGraph/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/med_mi_rscf/ckpt_final.pt" \
    "FB15k237/TransE/MED+MI+RSCF"

# ============================================
# Summary
# ============================================
echo ""
echo "############################################"
echo "# Stability Evaluation Complete!"
echo "############################################"
echo ""
echo "Processed 24 essential configurations (512d + MED for both models)"
echo ""
echo "Results saved in:"
echo "  - workdir/runs/final/wn18rr/*/stability.json"
echo "  - workdir/runs/final/fb15k237/*/stability.json"
echo ""
echo "To collect results into a summary table:"
echo "  python scripts/collect_stability_results.py"
echo ""
