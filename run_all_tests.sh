#!/bin/bash
# Run all tests for KG embeddings repository
# Usage: ./run_all_tests.sh

set -e  # Exit on first failure

echo "=========================================="
echo "KG Embeddings Test Suite"
echo "=========================================="
echo ""

# Store the repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Track test results
TOTAL=0
PASSED=0
FAILED=0

run_test() {
    local test_name=$1
    local test_path=$2
    
    TOTAL=$((TOTAL + 1))
    echo "[$TOTAL] Running: $test_name"
    
    if PYTHONPATH=. python3 "$test_path" > /dev/null 2>&1; then
        echo "    ✅ PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "    ❌ FAILED"
        FAILED=$((FAILED + 1))
        echo "    Run manually for details: PYTHONPATH=. python3 $test_path"
    fi
    echo ""
}

echo "Running Smoke Tests..."
echo "----------------------------------------"
run_test "Dataset I/O" "tests/smoke_test_dataset_io.py"
run_test "Models" "tests/smoke_test_models.py"
run_test "Evaluation" "tests/smoke_test_eval.py"
run_test "MED Wrapper" "tests/smoke_test_med.py"
run_test "RSCF Wrapper" "tests/smoke_test_rscf.py"
run_test "MI Wrapper" "tests/smoke_test_mi.py"

echo ""
echo "Running Integration Tests..."
echo "----------------------------------------"
run_test "End-to-End Pipeline" "tests/test_end_to_end_minimal.py"
run_test "Wrappers Integration" "tests/test_wrappers_integration.py"

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo "Total:  $TOTAL"
echo "Passed: $PASSED ✅"
echo "Failed: $FAILED ❌"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed. See output above for details."
    exit 1
fi
