#!/bin/bash
# Run KGExplainer distillation on all 48 Phase 1 checkpoints
# Estimated time on GPU: ~4-8 hours total (500 steps, ~5-10 min per checkpoint)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate KGExplainer environment
source "$PROJECT_ROOT/.venv_kgexplainer/bin/activate"

# Configuration
NUM_STEPS=500         # REDUCED from 5000 for speed (10x faster)
BATCH_SIZE=64         # INCREASED from 32 (2x faster batches)
K_HOP=2
LR=0.001
DEVICE="cuda"         # GPU ENABLED! (H200)

# Output directory
OUT_DIR="$PROJECT_ROOT/workdir/kgexplainer"
mkdir -p "$OUT_DIR"

# Log file
LOG_FILE="$OUT_DIR/distillation_log.txt"
echo "KGExplainer Distillation (FAST MODE) - Started at $(date)" | tee "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  Steps: $NUM_STEPS (reduced for speed)" | tee -a "$LOG_FILE"
echo "  Batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  K-hop: $K_HOP" | tee -a "$LOG_FILE"
echo "  Device: $DEVICE" | tee -a "$LOG_FILE"
echo "  Output: $OUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Counter
TOTAL=48
CURRENT=0

# Function to run distillation
run_distillation() {
    local model=$1
    local dataset=$2
    local variant=$3
    local dim=$4
    
    CURRENT=$((CURRENT + 1))
    
    # Construct paths
    local dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    local model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
    
    local ckpt_path="$PROJECT_ROOT/workdir/runs/final/${dataset_lower}/${model_lower}/${variant}_${dim}/ckpt_final.pt"
    local out_path="$OUT_DIR/${dataset_lower}_${model_lower}_${variant}_${dim}_evaluator.pt"
    
    echo "[$CURRENT/$TOTAL] Processing: $dataset $model $variant $dim" | tee -a "$LOG_FILE"
    echo "  Checkpoint: $ckpt_path" | tee -a "$LOG_FILE"
    echo "  Output: $out_path" | tee -a "$LOG_FILE"
    
    # Check if checkpoint exists
    if [[ ! -f "$ckpt_path" ]]; then
        echo "  ⚠️  WARNING: Checkpoint not found, skipping..." | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        return
    fi
    
    # Check if output already exists
    if [[ -f "$out_path" ]]; then
        echo "  ✓ Evaluator already exists, skipping..." | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        return
    fi
    
    # Run distillation
    local start_time=$(date +%s)
    
    python "$PROJECT_ROOT/KGExplainer/distill_fixed.py" \
        --ckpt "$ckpt_path" \
        --model "$model" \
        --dataset "$dataset" \
        --use-hf \
        --k-hop $K_HOP \
        --num-steps $NUM_STEPS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --device $DEVICE \
        --out "$out_path" 2>&1 | tee -a "$LOG_FILE"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    if [[ -f "$out_path" ]]; then
        echo "  ✓ Completed in ${minutes}m ${seconds}s" | tee -a "$LOG_FILE"
    else
        echo "  ✗ FAILED" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
}

# Main execution
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  Steps: $NUM_STEPS" | tee -a "$LOG_FILE"
echo "  Batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  K-hop: $K_HOP" | tee -a "$LOG_FILE"
echo "  Device: $DEVICE" | tee -a "$LOG_FILE"
echo "  Output directory: $OUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# WN18RR - RotatE
echo "=== WN18RR - RotatE ===" | tee -a "$LOG_FILE"
run_distillation "RotatE" "WN18RR" "mi" "128d"
run_distillation "RotatE" "WN18RR" "mi" "256d"
run_distillation "RotatE" "WN18RR" "mi" "512d"
run_distillation "RotatE" "WN18RR" "rscf" "128d"
run_distillation "RotatE" "WN18RR" "rscf" "256d"
run_distillation "RotatE" "WN18RR" "rscf" "512d"
run_distillation "RotatE" "WN18RR" "mi_rscf" "128d"
run_distillation "RotatE" "WN18RR" "mi_rscf" "256d"
run_distillation "RotatE" "WN18RR" "mi_rscf" "512d"
run_distillation "RotatE" "WN18RR" "med_mi" "128d"
run_distillation "RotatE" "WN18RR" "med_mi" "256d"
run_distillation "RotatE" "WN18RR" "med_mi" "512d"
run_distillation "RotatE" "WN18RR" "med_rscf" "128d"
run_distillation "RotatE" "WN18RR" "med_rscf" "256d"
run_distillation "RotatE" "WN18RR" "med_rscf" "512d"
run_distillation "RotatE" "WN18RR" "med_mi_rscf" "128d"
run_distillation "RotatE" "WN18RR" "med_mi_rscf" "256d"
run_distillation "RotatE" "WN18RR" "med_mi_rscf" "512d"

# WN18RR - TransE
echo "=== WN18RR - TransE ===" | tee -a "$LOG_FILE"
run_distillation "TransE" "WN18RR" "mi" "128d"
run_distillation "TransE" "WN18RR" "mi" "256d"
run_distillation "TransE" "WN18RR" "mi" "512d"
run_distillation "TransE" "WN18RR" "rscf" "128d"
run_distillation "TransE" "WN18RR" "rscf" "256d"
run_distillation "TransE" "WN18RR" "rscf" "512d"
run_distillation "TransE" "WN18RR" "mi_rscf" "128d"
run_distillation "TransE" "WN18RR" "mi_rscf" "256d"
run_distillation "TransE" "WN18RR" "mi_rscf" "512d"
run_distillation "TransE" "WN18RR" "med_mi" "128d"
run_distillation "TransE" "WN18RR" "med_mi" "256d"
run_distillation "TransE" "WN18RR" "med_mi" "512d"
run_distillation "TransE" "WN18RR" "med_rscf" "128d"
run_distillation "TransE" "WN18RR" "med_rscf" "256d"
run_distillation "TransE" "WN18RR" "med_rscf" "512d"
run_distillation "TransE" "WN18RR" "med_mi_rscf" "128d"
run_distillation "TransE" "WN18RR" "med_mi_rscf" "256d"
run_distillation "TransE" "WN18RR" "med_mi_rscf" "512d"

# FB15k-237 - RotatE
echo "=== FB15k-237 - RotatE ===" | tee -a "$LOG_FILE"
run_distillation "RotatE" "FB15k-237" "mi" "128d"
run_distillation "RotatE" "FB15k-237" "mi" "256d"
run_distillation "RotatE" "FB15k-237" "mi" "512d"
run_distillation "RotatE" "FB15k-237" "rscf" "128d"
run_distillation "RotatE" "FB15k-237" "rscf" "256d"
run_distillation "RotatE" "FB15k-237" "rscf" "512d"
run_distillation "RotatE" "FB15k-237" "mi_rscf" "128d"
run_distillation "RotatE" "FB15k-237" "mi_rscf" "256d"
run_distillation "RotatE" "FB15k-237" "mi_rscf" "512d"
run_distillation "RotatE" "FB15k-237" "med_mi" "128d"
run_distillation "RotatE" "FB15k-237" "med_mi" "256d"
run_distillation "RotatE" "FB15k-237" "med_mi" "512d"
run_distillation "RotatE" "FB15k-237" "med_rscf" "128d"
run_distillation "RotatE" "FB15k-237" "med_rscf" "256d"
run_distillation "RotatE" "FB15k-237" "med_rscf" "512d"
run_distillation "RotatE" "FB15k-237" "med_mi_rscf" "128d"
run_distillation "RotatE" "FB15k-237" "med_mi_rscf" "256d"
run_distillation "RotatE" "FB15k-237" "med_mi_rscf" "512d"

# FB15k-237 - TransE
echo "=== FB15k-237 - TransE ===" | tee -a "$LOG_FILE"
run_distillation "TransE" "FB15k-237" "mi" "128d"
run_distillation "TransE" "FB15k-237" "mi" "256d"
run_distillation "TransE" "FB15k-237" "mi" "512d"
run_distillation "TransE" "FB15k-237" "rscf" "128d"
run_distillation "TransE" "FB15k-237" "rscf" "256d"
run_distillation "TransE" "FB15k-237" "rscf" "512d"
run_distillation "TransE" "FB15k-237" "mi_rscf" "128d"
run_distillation "TransE" "FB15k-237" "mi_rscf" "256d"
run_distillation "TransE" "FB15k-237" "mi_rscf" "512d"
run_distillation "TransE" "FB15k-237" "med_mi" "128d"
run_distillation "TransE" "FB15k-237" "med_mi" "256d"
run_distillation "TransE" "FB15k-237" "med_mi" "512d"
run_distillation "TransE" "FB15k-237" "med_rscf" "128d"
run_distillation "TransE" "FB15k-237" "med_rscf" "256d"
run_distillation "TransE" "FB15k-237" "med_rscf" "512d"
run_distillation "TransE" "FB15k-237" "med_mi_rscf" "128d"
run_distillation "TransE" "FB15k-237" "med_mi_rscf" "256d"
run_distillation "TransE" "FB15k-237" "med_mi_rscf" "512d"

# Summary
echo "========================================" | tee -a "$LOG_FILE"
echo "Distillation Complete!" | tee -a "$LOG_FILE"
echo "Finished at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $OUT_DIR" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Count completed evaluators
NUM_COMPLETED=$(ls -1 "$OUT_DIR"/*_evaluator.pt 2>/dev/null | wc -l)
echo "Completed: $NUM_COMPLETED / $TOTAL evaluators" | tee -a "$LOG_FILE"
