#!/usr/bin/env bash
# Run ALL final report experiments
set -euo pipefail

echo "=========================================="
echo "Final Report: ALL Experiments"
echo "=========================================="
echo ""
echo "Running all MED combinations:"
echo "  - MED + RSCF (6 experiments)"
echo "  - MED + MI (6 experiments)"
echo "  - MED + RSCF + MI (6 experiments)"
echo ""
echo "Models: ComplEx, RotatE, TransE"
echo "Datasets: FB15k-237, WN18RR"
echo "Total: 18 experiments"
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

echo "========================================"
echo "Phase 1: RSCF Experiments (6)"
echo "========================================"
bash "${ROOT_DIR}/scripts/run_final_rscf.sh"

echo ""
echo "========================================"
echo "Phase 2: MI Experiments (6)"
echo "========================================"
bash "${ROOT_DIR}/scripts/run_final_mi.sh"

echo ""
echo "========================================"
echo "Phase 3: RSCF+MI Experiments (6)"
echo "========================================"
bash "${ROOT_DIR}/scripts/run_final_rscf_mi.sh"

echo ""
echo "========================================"
echo "✅ ALL EXPERIMENTS COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to: ./workdir/runs/final/"
echo ""
echo "Summary:"
echo "  - RSCF: 6 experiments ✓"
echo "  - MI: 6 experiments ✓"
echo "  - RSCF+MI: 6 experiments ✓"
echo "  - Total: 18 experiments ✓"
echo ""
