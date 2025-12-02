#!/usr/bin/env bash
# Quick verification script to test MED combinations setup
# Tests one config from each combination type
set -euo pipefail

echo "=========================================="
echo "MED Combinations - Quick Verification"
echo "=========================================="
echo ""
echo "This script runs 3 quick tests (1 epoch each):"
echo "  1. MED + RSCF (ComplEx, FB15k-237)"
echo "  2. MED + MI (ComplEx, FB15k-237)"
echo "  3. MED + RSCF + MI (ComplEx, FB15k-237)"
echo ""
echo "Expected time: ~5-10 minutes total"
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
TEMP_DIR="${ROOT_DIR}/workdir/verification_test_$(date +%s)"

mkdir -p "${TEMP_DIR}"

# Test configs - we'll modify them temporarily to run for just 1 epoch
TEST_CONFIGS=(
    "configs/fb15k237/complex_med_rscf.yaml"
    "configs/fb15k237/complex_med_mi.yaml"
    "configs/fb15k237/complex_med_rscf_mi.yaml"
)

echo "Testing 3 MED combination configs..."
echo ""

for i in "${!TEST_CONFIGS[@]}"; do
    config="${TEST_CONFIGS[$i]}"
    config_name=$(basename "$config" .yaml)
    
    echo "========================================"
    echo "Test $((i+1))/3: $config"
    echo "========================================"
    
    # Create temporary config with 1 epoch only
    temp_config="${TEMP_DIR}/${config_name}_test.yaml"
    cat "${ROOT_DIR}/${config}" | sed 's/epochs: [0-9]\+/epochs: 1/' > "${temp_config}"
    
    # Run test
    if python "${ROOT_DIR}/train.py" --config "${temp_config}"; then
        echo "✓ Test passed: $config"
    else
        echo "✗ Test failed: $config"
        exit 1
    fi
    echo ""
done

echo ""
echo "========================================"
echo "✅ All verification tests passed!"
echo "========================================"
echo ""
echo "The MED combinations are working correctly:"
echo "  ✓ MED + RSCF"
echo "  ✓ MED + MI"
echo "  ✓ MED + RSCF + MI"
echo ""
echo "You can now run full experiments with:"
echo "  bash scripts/run_med_combinations.sh"
echo ""
echo "Cleaning up temporary files..."
rm -rf "${TEMP_DIR}"
echo "Done!"
