#!/bin/bash
# Batch Stability Evaluation Script
# Computes RTMD stability metrics for all trained checkpoints
# 
# Usage: bash scripts/run_stability_evaluation.sh [--quick]
#   --quick: Use fewer samples (100) and lower depth (L=2) for faster testing
#   (default): Use 500 samples and L=3 for comprehensive evaluation

set -e  # Exit on error

# Parse arguments
MODE="comprehensive"
if [[ "$1" == "--ultra-fast" ]]; then
    MODE="ultra-fast"
    SAMPLES=20
    LAYERS=1
    echo "=== ULTRA-FAST MODE: samples=20, layers=1 (20 min/checkpoint) ==="
elif [[ "$1" == "--quick" ]]; then
    MODE="quick"
    SAMPLES=50
    LAYERS=1
    echo "=== QUICK MODE: samples=50, layers=1 (60 min/checkpoint) ==="
elif [[ "$1" == "--fast" ]]; then
    MODE="fast"
    SAMPLES=30
    LAYERS=1
    echo "=== FAST MODE: samples=30, layers=1 (30 min/checkpoint) ==="
else
    SAMPLES=100
    LAYERS=2
    echo "=== STANDARD MODE: samples=100, layers=2 (6 hours/checkpoint) ==="
    echo "WARNING: This will take ~288 hours for all 48 checkpoints!"
    echo "Consider using --ultra-fast (16 hours total) or --fast (24 hours total)"
fi

# Configuration
BASE_DIR="workdir/runs/final"
SCRIPT="evaluate_stability.py"

# Counter for progress tracking
TOTAL_CONFIGS=48
CURRENT=0

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
# WN18RR - RotatE
# ============================================
echo ""
echo "############################################"
echo "# WN18RR - RotatE Configurations"
echo "############################################"

# RotatE + MI (128d, 256d, 512d)
run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/mi_128d/ckpt_final.pt" \
    "WN18RR/RotatE/MI-128d"

run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/mi_256d/ckpt_final.pt" \
    "WN18RR/RotatE/MI-256d"

run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/mi_512d/ckpt_final.pt" \
    "WN18RR/RotatE/MI-512d"

# RotatE + RSCF (128d, 256d, 512d)
run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/rscf_128d/ckpt_final.pt" \
    "WN18RR/RotatE/RSCF-128d"

run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/rscf_256d/ckpt_final.pt" \
    "WN18RR/RotatE/RSCF-256d"

run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/rscf_512d/ckpt_final.pt" \
    "WN18RR/RotatE/RSCF-512d"

# RotatE + MI+RSCF (128d, 256d, 512d)
run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/mi_rscf_128d/ckpt_final.pt" \
    "WN18RR/RotatE/MI+RSCF-128d"

run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/mi_rscf_256d/ckpt_final.pt" \
    "WN18RR/RotatE/MI+RSCF-256d"

run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/mi_rscf_512d/ckpt_final.pt" \
    "WN18RR/RotatE/MI+RSCF-512d"

# RotatE + MED combinations
run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/med_mi/ckpt_final.pt" \
    "WN18RR/RotatE/MED+MI"

run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/med_rscf/ckpt_final.pt" \
    "WN18RR/RotatE/MED+RSCF"

run_stability "RotatE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/rotate/med_mi_rscf/ckpt_final.pt" \
    "WN18RR/RotatE/MED+MI+RSCF"

# ============================================
# WN18RR - TransE
# ============================================
echo ""
echo "############################################"
echo "# WN18RR - TransE Configurations"
echo "############################################"

# TransE + MI (128d, 256d, 512d)
run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/mi_128d/ckpt_final.pt" \
    "WN18RR/TransE/MI-128d"

run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/mi_256d/ckpt_final.pt" \
    "WN18RR/TransE/MI-256d"

run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/mi_512d/ckpt_final.pt" \
    "WN18RR/TransE/MI-512d"

# TransE + RSCF (128d, 256d, 512d)
run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/rscf_128d/ckpt_final.pt" \
    "WN18RR/TransE/RSCF-128d"

run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/rscf_256d/ckpt_final.pt" \
    "WN18RR/TransE/RSCF-256d"

run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/rscf_512d/ckpt_final.pt" \
    "WN18RR/TransE/RSCF-512d"

# TransE + MI+RSCF (128d, 256d, 512d)
run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/mi_rscf_128d/ckpt_final.pt" \
    "WN18RR/TransE/MI+RSCF-128d"

run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/mi_rscf_256d/ckpt_final.pt" \
    "WN18RR/TransE/MI+RSCF-256d"

run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/mi_rscf_512d/ckpt_final.pt" \
    "WN18RR/TransE/MI+RSCF-512d"

# TransE + MED combinations
run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/med_mi/ckpt_final.pt" \
    "WN18RR/TransE/MED+MI"

run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/med_rscf/ckpt_final.pt" \
    "WN18RR/TransE/MED+RSCF"

run_stability "TransE" "WN18RR" "VLyb/WN18RR" \
    "$BASE_DIR/wn18rr/transe/med_mi_rscf/ckpt_final.pt" \
    "WN18RR/TransE/MED+MI+RSCF"

# ============================================
# FB15k-237 - RotatE
# ============================================
echo ""
echo "############################################"
echo "# FB15k-237 - RotatE Configurations"
echo "############################################"

# RotatE + MI (128d, 256d, 512d)
run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/mi_128d/ckpt_final.pt" \
    "FB15k237/RotatE/MI-128d"

run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/mi_256d/ckpt_final.pt" \
    "FB15k237/RotatE/MI-256d"

run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/mi_512d/ckpt_final.pt" \
    "FB15k237/RotatE/MI-512d"

# RotatE + RSCF (128d, 256d, 512d)
run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/rscf_128d/ckpt_final.pt" \
    "FB15k237/RotatE/RSCF-128d"

run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/rscf_256d/ckpt_final.pt" \
    "FB15k237/RotatE/RSCF-256d"

run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/rscf_512d/ckpt_final.pt" \
    "FB15k237/RotatE/RSCF-512d"

# RotatE + MI+RSCF (128d, 256d, 512d)
run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/mi_rscf_128d/ckpt_final.pt" \
    "FB15k237/RotatE/MI+RSCF-128d"

run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/mi_rscf_256d/ckpt_final.pt" \
    "FB15k237/RotatE/MI+RSCF-256d"

run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/mi_rscf_512d/ckpt_final.pt" \
    "FB15k237/RotatE/MI+RSCF-512d"

# RotatE + MED combinations
run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/med_mi/ckpt_final.pt" \
    "FB15k237/RotatE/MED+MI"

run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/med_rscf/ckpt_final.pt" \
    "FB15k237/RotatE/MED+RSCF"

run_stability "RotatE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/rotate/med_mi_rscf/ckpt_final.pt" \
    "FB15k237/RotatE/MED+MI+RSCF"

# ============================================
# FB15k-237 - TransE
# ============================================
echo ""
echo "############################################"
echo "# FB15k-237 - TransE Configurations"
echo "############################################"

# TransE + MI (128d, 256d, 512d)
run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/mi_128d/ckpt_final.pt" \
    "FB15k237/TransE/MI-128d"

run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/mi_256d/ckpt_final.pt" \
    "FB15k237/TransE/MI-256d"

run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/mi_512d/ckpt_final.pt" \
    "FB15k237/TransE/MI-512d"

# TransE + RSCF (128d, 256d, 512d)
run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/rscf_128d/ckpt_final.pt" \
    "FB15k237/TransE/RSCF-128d"

run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/rscf_256d/ckpt_final.pt" \
    "FB15k237/TransE/RSCF-256d"

run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/rscf_512d/ckpt_final.pt" \
    "FB15k237/TransE/RSCF-512d"

# TransE + MI+RSCF (128d, 256d, 512d)
run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/mi_rscf_128d/ckpt_final.pt" \
    "FB15k237/TransE/MI+RSCF-128d"

run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/mi_rscf_256d/ckpt_final.pt" \
    "FB15k237/TransE/MI+RSCF-256d"

run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/mi_rscf_512d/ckpt_final.pt" \
    "FB15k237/TransE/MI+RSCF-512d"

# TransE + MED combinations
run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/med_mi/ckpt_final.pt" \
    "FB15k237/TransE/MED+MI"

run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
    "$BASE_DIR/fb15k237/transe/med_rscf/ckpt_final.pt" \
    "FB15k237/TransE/MED+RSCF"

run_stability "TransE" "FB15k-237" "VLyb/FB15k-237" \
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
echo "Results saved in:"
echo "  - workdir/runs/final/wn18rr/*/stability.json"
echo "  - workdir/runs/final/fb15k237/*/stability.json"
echo ""
echo "To collect all results into a summary table:"
echo "  python scripts/collect_stability_results.py"
echo ""
