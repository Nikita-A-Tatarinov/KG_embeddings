# Training Issues Investigation

## Issue 1: Evaluation Script Failed ✅ FIXED

**Problem**: `evaluate.py` couldn't find dataset at `./data/FB15k-237`

**Root Cause**: Training uses HuggingFace datasets (`source: hf` in config), but evaluation script only supported file-based loading.

**Fix Applied**:
1. Updated `evaluate.py` to support `--use-hf` flag
2. Updated script to pass correct flags based on config
3. Auto-infer `base_dim` from checkpoint (was hardcoded to 100!)

## Issue 2: Terrible Training Metrics ⚠️ NEEDS INVESTIGATION

**Observed**: Test MRR = 0.003 (should be ~0.33 for RotatE on FB15k-237)

**Possible Causes**:

### 1. Wrong Dimensions
- Training shows `base_dim: 20` but evaluation might have used wrong dim
- **Status**: FIXED in evaluate.py (now infers from checkpoint)

### 2. Data Loading Issue
- HuggingFace data might be in wrong format
- **Test**: Run `PYTHONPATH=. python3 tests/test_hf_data_loading.py`
- **Expected**: 272,115 train / 17,535 valid / 20,466 test triples

### 3. Training Issue
Looking at your output:
```
[99/100] | step 52600 | loss 0.9396
[100/100] | step 53200 | loss 0.9293
Test: {'MR': 5712.5706, 'MRR': 0.003, 'H@1': 0.0008, 'H@3': 0.002, 'H@10': 0.0051}
```

**Observations**:
- Loss ~0.93 is reasonable for early/small models
- But MRR 0.003 is TERRIBLE (random would be ~0.0001, so it's learning something)
- MR 5712 means on average ranking ~5712th out of 14541 entities (random!)

**Potential Issues**:
- ❓ Model dimension too small (d=20 is very small, typical is 100-200)
- ❓ Learning rate too high/low
- ❓ Not enough epochs (100 might not be enough for d=20)
- ❓ Evaluation on wrong split?

### 4. Quick Tests to Run

```bash
# 1. Verify data loading
PYTHONPATH=. python3 tests/test_hf_data_loading.py

# 2. Re-evaluate the checkpoint with fixed script
PYTHONPATH=. python3 evaluate.py \
  --model RotatE \
  --ckpt ./workdir/runs/fb15k237/rotate/d20/ckpt_best_mrr.pt \
  --dataset FB15k-237 \
  --use-hf \
  --filtered \
  --device auto

# 3. Check training logs
cat ./workdir/runs/fb15k237/rotate/d20/train.csv | tail -20
```

## Recommended Actions

### Immediate:
1. ✅ Re-run evaluation with fixed script
2. 📊 Check if d=20 model actually achieved 0.003 or if it was evaluation bug
3. 🔍 Verify data loaded correctly

### If metrics are still bad:
1. Try d=40 or d=80 (more reasonable dimensions)
2. Check if validation metrics during training were also bad
3. Inspect training logs for anomalies

### Long-term:
1. Add base_dim to checkpoint metadata for easier recovery
2. Add data loading validation to training script
3. Add early stopping if metrics are suspiciously bad

## Expected Performance Baselines

For RotatE on FB15k-237 (from literature):

| Dimension | MRR (approx.) | Notes |
|-----------|---------------|-------|
| d=20      | 0.25-0.28     | Very small |
| d=40      | 0.30-0.32     | Small |
| d=100     | 0.33-0.34     | Standard |
| d=200     | 0.33-0.35     | Large |

If you're getting 0.003, something is seriously wrong.

## Files Modified

1. `evaluate.py`:
   - Added `--use-hf` and `--hf-name` flags
   - Auto-infer `base_dim` from checkpoint
   - Better device handling (auto)
   - Added data source info printing

2. `scripts/run_rotate_all.sh`:
   - Correctly parse and pass HF flags
   - Pass dataset-specific parameters

3. New test: `tests/test_hf_data_loading.py`
   - Validates HF data loads correctly
   - Checks expected triple counts
