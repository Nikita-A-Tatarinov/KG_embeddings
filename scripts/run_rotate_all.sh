#!/usr/bin/env bash
# Run all RotatE experiments (FB15k-237 and WN18RR) with training and evaluation
set -euo pipefail

ROOT_DIR=$(dirname "$(dirname "${BASH_SOURCE[0]}")")
SCRIPTDIR="$ROOT_DIR/configs"

# Check GPU availability
echo "=========================================="
echo "KG Embeddings Training & Evaluation"
echo "=========================================="
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DEVICE="cuda (auto-detected)"
else
    echo "⚠ No GPU detected. Training will use CPU (slower)."
    DEVICE="cpu (fallback)"
fi
echo "Device: $DEVICE"
echo ""

CONFIGS=(
  "${ROOT_DIR}/configs/fb15k237/rotate_d10.yaml"
  "${ROOT_DIR}/configs/fb15k237/rotate_d20.yaml"
  "${ROOT_DIR}/configs/fb15k237/rotate_d40.yaml"
  "${ROOT_DIR}/configs/fb15k237/rotate_d80.yaml"
  "${ROOT_DIR}/configs/fb15k237/rotate_med.yaml"
  "${ROOT_DIR}/configs/fb15k237/rotate_med_rscf.yaml"
  "${ROOT_DIR}/configs/fb15k237/rotate_med_mi.yaml"
  "${ROOT_DIR}/configs/fb15k237/rotate_med_rscf_mi.yaml"

  "${ROOT_DIR}/configs/wn18rr/rotate_d10.yaml"
  "${ROOT_DIR}/configs/wn18rr/rotate_d20.yaml"
  "${ROOT_DIR}/configs/wn18rr/rotate_d40.yaml"
  "${ROOT_DIR}/configs/wn18rr/rotate_d80.yaml"
  "${ROOT_DIR}/configs/wn18rr/rotate_med.yaml"
  "${ROOT_DIR}/configs/wn18rr/rotate_med_rscf.yaml"
  "${ROOT_DIR}/configs/wn18rr/rotate_med_mi.yaml"
  "${ROOT_DIR}/configs/wn18rr/rotate_med_rscf_mi.yaml"
)

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_CONFIGS=()

for cfg in "${CONFIGS[@]}"; do
  echo "=========================================="
  echo "Config: $cfg"
  echo "=========================================="
  
  # Extract dataset name and output dir from config for evaluation
  DATASET=$(grep "name:" "$cfg" | head -1 | awk '{print $2}')
  OUT_DIR=$(grep "out_dir:" "$cfg" | awk '{print $2}')
  
  # Check if using HuggingFace dataset
  if grep -q "source: hf" "$cfg"; then
    USE_HF="--use-hf"
    HF_NAME=$(grep "hf_name:" "$cfg" | awk '{print $2}')
    if [ -n "$HF_NAME" ]; then
      HF_NAME="--hf-name $HF_NAME"
    else
      HF_NAME=""
    fi
  else
    USE_HF=""
    HF_NAME=""
    DATA_ROOT=$(grep "root:" "$cfg" | awk '{print $2}')
    DATA_ROOT="--data-root $DATA_ROOT"
  fi
  
  # Run training (includes final test evaluation in trainer)
  echo "[1/2] Training model..."
  if PYTHONPATH=. python3 train.py --config "$cfg"; then
    echo "✓ Training completed successfully"
    
    # Run explicit evaluation on best checkpoint if it exists
    BEST_CKPT="${OUT_DIR}/ckpt_best_mrr.pt"
    if [ -f "$BEST_CKPT" ]; then
      echo "[2/2] Evaluating best checkpoint..."
      
      # Run evaluation and save metrics
      METRICS_FILE="${OUT_DIR}/final_test_metrics.json"
      if PYTHONPATH=. python3 evaluate.py \
          --model RotatE \
          --ckpt "$BEST_CKPT" \
          $DATA_ROOT \
          --dataset "$DATASET" \
          $USE_HF \
          $HF_NAME \
          --batch-size 128 \
          --filtered \
          --device auto \
          --out "$METRICS_FILE"; then
        echo "✓ Evaluation completed successfully"
        echo "  Metrics saved to: $METRICS_FILE"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      else
        echo "⚠ Evaluation failed (but training succeeded)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))  # Count as success since training worked
      fi
    else
      echo "[2/2] No best checkpoint found (ckpt_best_mrr.pt), skipping explicit evaluation"
      echo "  (Test metrics should be in training logs)"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
    echo ""
  else
    echo "✗ Training failed: $cfg" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAILED_CONFIGS+=("$cfg")
    echo ""
  fi
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total configs: ${#CONFIGS[@]}"
echo "Successful: $SUCCESS_COUNT ✓"
echo "Failed: $FAIL_COUNT ✗"

if [ $FAIL_COUNT -gt 0 ]; then
  echo ""
  echo "Failed configurations:"
  for failed in "${FAILED_CONFIGS[@]}"; do
    echo "  - $failed"
  done
fi

echo ""
echo "All done. Check workdir/runs for outputs."
echo "Metrics saved in <out_dir>/final_test_metrics.json"
