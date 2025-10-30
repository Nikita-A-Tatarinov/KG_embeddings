#!/usr/bin/env bash
# Quick test script - run one config from each model to verify setup
# Perfect for testing GPU cluster setup before running full experiments
set -euo pipefail

echo "=========================================="
echo "Quick Test: One Config Per Model"
echo "=========================================="
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ No GPU - using CPU"
fi
echo ""

ROOT_DIR=$(dirname "$(dirname "${BASH_SOURCE[0]}")")

# One quick config from each model (d10 variants)
CONFIGS=(
    "configs/fb15k237/rotate_d10.yaml"
    "configs/fb15k237/complex_d10.yaml"
    "configs/fb15k237/distmult_d10.yaml"
    "configs/fb15k237/transe_d10.yaml"
    "configs/fb15k237/transh_d10.yaml"
    "configs/fb15k237/pairre_d10.yaml"
)

echo "Running ${#CONFIGS[@]} quick tests..."
echo "Expected time: ~4 hours total (~40 min each)"
echo ""

for config in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Testing: $config"
    echo "========================================"
    python "${ROOT_DIR}/train.py" --config "${ROOT_DIR}/${config}"
    echo ""
done

echo ""
echo "✅ Quick test complete! All models working."
echo "   Ready to run full experiments."
