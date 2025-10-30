# KG Embeddings Testing Guide

## Overview

This document describes the comprehensive test suite for the KG embeddings repository. All tests are designed to be **CPU-friendly** and run quickly for development and CI purposes.

## Test Structure

The repository follows a **bottom-up testing approach**:

```
tests/
├── smoke_test_dataset_io.py      # Data loading and filtering
├── smoke_test_models.py           # All 7 base models
├── smoke_test_eval.py             # KGC evaluation metrics
├── smoke_test_med.py              # MED wrapper
├── smoke_test_rscf.py             # RSCF wrapper
├── smoke_test_mi.py               # MI wrapper
├── test_end_to_end_minimal.py     # Full pipeline (NEW)
└── test_wrappers_integration.py   # Wrappers with eval (NEW)
```

## Running Tests

All tests should be run with `PYTHONPATH=.` from the repository root:

```bash
cd /path/to/KG_embeddings
```

### Quick Test (Run All)

```bash
# Run all smoke tests
PYTHONPATH=. python3 tests/smoke_test_dataset_io.py
PYTHONPATH=. python3 tests/smoke_test_models.py
PYTHONPATH=. python3 tests/smoke_test_eval.py
PYTHONPATH=. python3 tests/smoke_test_med.py
PYTHONPATH=. python3 tests/smoke_test_rscf.py
PYTHONPATH=. python3 tests/smoke_test_mi.py

# Run comprehensive integration tests
PYTHONPATH=. python3 tests/test_end_to_end_minimal.py
PYTHONPATH=. python3 tests/test_wrappers_integration.py
```

### Individual Test Details

#### 1. Dataset I/O Test (`smoke_test_dataset_io.py`)

**Purpose**: Verify data loading, KGIndex filtering, and train/test loaders.

**What it tests**:
- Loading raw triples from text files
- Entity/relation mapping
- Filtered negative sampling for training
- Filtered evaluation candidate generation
- Gold triple placement at column 0

**Expected output**: ✓ marks for train and test loaders

```bash
PYTHONPATH=. python3 tests/smoke_test_dataset_io.py
```

#### 2. Models Test (`smoke_test_models.py`)

**Purpose**: Verify all 7 base model architectures work correctly.

**Models tested**:
- TransE
- TransH
- DistMult
- ComplEx
- RotatE
- RotatEv2
- PairRE

**What it tests**:
- Forward pass in all modes (single, head-batch, tail-batch)
- Shape correctness
- Dimension cropping (for MED)
- Backward pass (gradient computation)

**Expected output**: ✓ marks for each model with loss values

```bash
PYTHONPATH=. python3 tests/smoke_test_models.py
```

#### 3. Evaluation Test (`smoke_test_eval.py`)

**Purpose**: Verify KGC evaluation metrics calculation.

**What it tests**:
- Filtered MRR computation
- Hits@k metrics (k=1,3,10)
- Head and tail evaluation
- Rank calculation

**Expected output**: Metrics dictionary with MRR and Hits values

```bash
PYTHONPATH=. python3 tests/smoke_test_eval.py
```

#### 4. MED Wrapper Test (`smoke_test_med.py`)

**Purpose**: Verify Mixture of Embedding Dimensions (MED) training.

**What it tests**:
- Multi-dimensional training
- Mutual learning loss (L_ml)
- Embedding independence loss (L_ei)
- Backward pass for all models with MED

**Expected output**: Loss values for each model with L_total, L_ml, L_ei

```bash
PYTHONPATH=. python3 tests/smoke_test_med.py
```

#### 5. RSCF Wrapper Test (`smoke_test_rscf.py`)

**Purpose**: Verify Relation-Specific Confidence Filtering.

**What it tests**:
- RSCF attachment to models
- Forward pass with filtering

**Expected output**: "RSCF smoke test passed"

```bash
PYTHONPATH=. python3 tests/smoke_test_rscf.py
```

#### 6. MI Wrapper Test (`smoke_test_mi.py`)

**Purpose**: Verify Mutual Information enhancement.

**What it tests**:
- MI module attachment
- MI loss computation
- Backward pass with MI

**Expected output**: "MI smoke test passed"

```bash
PYTHONPATH=. python3 tests/smoke_test_mi.py
```

#### 7. End-to-End Minimal Test (`test_end_to_end_minimal.py`) ⭐ NEW

**Purpose**: Comprehensive test of the full training pipeline.

**What it tests**:
1. Synthetic dataset creation
2. Data loading
3. Model creation (TransE)
4. Training for 3 steps
5. Evaluation (filtered MRR/Hits)
6. Checkpoint saving
7. Checkpoint loading
8. Re-evaluation (metrics match)
9. Compatibility with `evaluate.py` script

**CPU-friendly settings**:
- 6 entities, 3 relations
- 8-dimensional embeddings
- Batch size: 2
- Only 3 training steps

**Expected output**: 
- Training progress with loss values
- Evaluation metrics before and after checkpoint
- ✅ END-TO-END TEST PASSED!

```bash
PYTHONPATH=. python3 tests/test_end_to_end_minimal.py
```

**Runtime**: ~1-2 seconds on CPU

#### 8. Wrappers Integration Test (`test_wrappers_integration.py`) ⭐ NEW

**Purpose**: Test all three wrappers (MED, RSCF, MI) with full training and evaluation.

**What it tests**:

1. **RSCF Wrapper Integration**:
   - Training with RSCF filtering
   - Evaluation with RSCF
   - Checkpoint save/load

2. **MI Wrapper Integration**:
   - Training with MI loss
   - Combined KGE + MI loss
   - Evaluation
   - Checkpoint save/load

3. **MED Wrapper Integration**:
   - Multi-dimensional training
   - Evaluation of underlying model
   - Checkpoint save/load

4. **MED + MI Combined**:
   - Advanced scenario with both wrappers
   - Ensures compatibility

**CPU-friendly settings**:
- 4 entities, 2 relations
- 8-dimensional embeddings
- 2 training steps per test
- Minimal batch sizes

**Expected output**:
- ✅ marks for each wrapper test
- Loss/metric values
- ✅ ALL WRAPPER INTEGRATION TESTS PASSED!

```bash
PYTHONPATH=. python3 tests/test_wrappers_integration.py
```

**Runtime**: ~3-5 seconds on CPU

## Test Results Summary

All tests have been verified to pass on CPU:

| Test | Status | Notes |
|------|--------|-------|
| smoke_test_dataset_io | ✅ PASS | Train/test loaders work |
| smoke_test_models | ✅ PASS | All 7 models work |
| smoke_test_eval | ✅ PASS | Evaluation metrics correct |
| smoke_test_med | ✅ PASS | MED wrapper works |
| smoke_test_rscf | ✅ PASS | RSCF wrapper works |
| smoke_test_mi | ✅ PASS | MI wrapper works |
| test_end_to_end_minimal | ✅ PASS | Full pipeline works |
| test_wrappers_integration | ✅ PASS | All wrappers integrate correctly |

## Evaluation Script (`evaluate.py`)

The `evaluate.py` script is **fully functional** and tested. It can evaluate saved checkpoints:

```bash
PYTHONPATH=. python3 evaluate.py \
  --model TransE \
  --ckpt path/to/ckpt_test.pt \
  --data-root path/to/data \
  --dataset FB15k-237 \
  --batch-size 64 \
  --filtered \
  --device cpu
```

**Key features**:
- Works with all model architectures
- Supports filtered evaluation
- Compatible with checkpoint format
- Can save metrics to JSON with `--out metrics.json`

## CPU-Friendly Testing Guidelines

To keep tests fast on CPU:

1. **Small datasets**: Use 4-10 entities, 2-3 relations
2. **Low dimensions**: 8-12 dimensional embeddings
3. **Minimal batches**: Batch size 2-4
4. **Few steps**: 2-3 training steps per test
5. **No workers**: Set `num_workers=0` in dataloaders

## Common Issues and Solutions

### Issue: `ModuleNotFoundError: No module named 'dataset'`

**Solution**: Always use `PYTHONPATH=.` before running tests:
```bash
PYTHONPATH=. python3 tests/test_name.py
```

### Issue: Tests are too slow

**Solution**: All new tests are designed for CPU with minimal sizes. If still slow:
- Check you're using `device="cpu"`
- Verify small dataset sizes
- Reduce number of training steps

### Issue: `python` command not found

**Solution**: Use `python3` explicitly:
```bash
PYTHONPATH=. python3 tests/test_name.py
```

## Adding New Tests

When adding new tests, follow these guidelines:

1. **Use minimal sizes** for CPU efficiency
2. **Seed RNG** for reproducibility: `seed_all(42)`
3. **Test full pipeline**: data → train → eval → checkpoint
4. **Verify numerically**: Check metrics match after checkpoint load
5. **Print progress**: Use clear print statements
6. **Assert conditions**: Use assertions for critical checks

Example template:

```python
def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

def test_my_feature(device="cpu"):
    print("Testing my feature...")
    seed_all(42)
    
    # Create minimal data
    # ... 
    
    # Test
    # ...
    
    # Verify
    assert condition, "Error message"
    print("✅ Test passed!")

if __name__ == "__main__":
    test_my_feature(device="cpu")
```

## Continuous Integration

All tests can be run sequentially in CI:

```bash
#!/bin/bash
set -e  # Exit on first failure

cd /path/to/KG_embeddings

# Smoke tests
PYTHONPATH=. python3 tests/smoke_test_dataset_io.py
PYTHONPATH=. python3 tests/smoke_test_models.py
PYTHONPATH=. python3 tests/smoke_test_eval.py
PYTHONPATH=. python3 tests/smoke_test_med.py
PYTHONPATH=. python3 tests/smoke_test_rscf.py
PYTHONPATH=. python3 tests/smoke_test_mi.py

# Integration tests
PYTHONPATH=. python3 tests/test_end_to_end_minimal.py
PYTHONPATH=. python3 tests/test_wrappers_integration.py

echo "✅ All tests passed!"
```

**Total runtime**: ~10-15 seconds on modern CPU

## Next Steps

Future improvements:

1. **pytest integration**: Convert tests to pytest format
2. **Coverage reports**: Add code coverage tracking
3. **GPU tests**: Optional GPU-specific tests (when available)
4. **Performance benchmarks**: Track training speed over time
5. **Dataset tests**: Add tests for FB15k-237 and WN18RR loaders

## Summary

The test suite provides comprehensive coverage of:
- ✅ Data loading and processing
- ✅ All 7 base model architectures  
- ✅ Evaluation metrics (MRR, Hits@k)
- ✅ MED wrapper functionality
- ✅ RSCF wrapper functionality
- ✅ MI wrapper functionality
- ✅ Full training pipeline
- ✅ Checkpoint save/load
- ✅ Wrapper integration with evaluation

All tests are **CPU-friendly** and complete in seconds, making them ideal for rapid development and testing.
