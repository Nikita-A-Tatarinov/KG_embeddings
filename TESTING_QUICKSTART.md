# Testing Quick Start

## Run All Tests

```bash
./run_all_tests.sh
```

This will run all 8 tests in sequence:
- ✅ Dataset I/O
- ✅ All 7 Models (TransE, TransH, DistMult, ComplEx, RotatE, RotatEv2, PairRE)
- ✅ Evaluation Metrics
- ✅ MED Wrapper
- ✅ RSCF Wrapper
- ✅ MI Wrapper
- ✅ End-to-End Pipeline
- ✅ Wrappers Integration

**Runtime**: ~10-15 seconds on CPU

## Run Individual Tests

```bash
# Example: Test a specific component
PYTHONPATH=. python3 tests/smoke_test_models.py

# Example: Test end-to-end pipeline
PYTHONPATH=. python3 tests/test_end_to_end_minimal.py

# Example: Test wrapper integration
PYTHONPATH=. python3 tests/test_wrappers_integration.py
```

## Test Documentation

See [TESTING.md](TESTING.md) for:
- Detailed test descriptions
- CPU-friendly testing guidelines
- How to add new tests
- Common issues and solutions

## What's Tested

✅ **Data Loading**: Synthetic datasets, filtering, negative sampling  
✅ **Models**: All 7 architectures with forward/backward passes  
✅ **Evaluation**: Filtered MRR, Hits@k metrics  
✅ **MED**: Multi-dimensional training with mutual learning  
✅ **RSCF**: Relation-specific confidence filtering  
✅ **MI**: Mutual information enhancement  
✅ **Integration**: Full pipeline with checkpoints  
✅ **Evaluation Script**: `evaluate.py` compatibility

All tests pass on CPU with minimal resource requirements.
