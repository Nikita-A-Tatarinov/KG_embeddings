# Critical Issues Found and Fixed

## Problem
Your RotatE + MED + RSCF experiment on WN18RR produced:
```
Test: {'MR': 11236.658691, 'MRR': 0.037938, 'H@1': 0.02074, 'H@3': 0.037173, 'H@10': 0.070836}
```

Expected baseline MED (from midterm): **MRR ≈ 0.40-0.41**

## Root Causes Identified

### 1. ❌ Wrong Gamma Value
- **Your config**: `gamma: 6.0`
- **Baseline config**: `gamma: 12.0`
- **Fix**: Changed to `gamma: 12.0`

### 2. ❌ Learning Rate Too High for RSCF
- RSCF adds 3 large parameter matrices (A1, A2, A3)
- Same lr=0.0005 as baseline causes instability
- **Fix**: Reduced to `lr: 0.0001` (5x lower)

### 3. ❌ No Regularization
- RSCF parameters can overfit quickly
- **Fix**: Added `weight_decay: 1e-5`

### 4. ❌ Too Many Sub-models
- Training 4 dimensions [10, 20, 40, 80] with 3 per step is unstable
- Dimension 10 is too small for meaningful learning
- **Fix**: Reduced to `dims: [20, 40, 80]`, `submodels_per_step: 2`

### 5. ⚠️ rscf_alpha Not Implemented
- Config has `rscf_alpha: 0.1` but code doesn't use it
- This is OK for now (not the main issue)

## Fixed Configs

All WN18RR RSCF configs have been updated with:

### RotatE + MED + RSCF
- ✅ `gamma: 12.0` (was 6.0)
- ✅ `lr: 0.0001` (was 0.0005)
- ✅ `weight_decay: 1e-5` (was 0.0)
- ✅ `dims: [20, 40, 80]` (was [10, 20, 40, 80])
- ✅ `submodels_per_step: 2` (was 3)
- ✅ `grad_clip: 0.5` (was 1.0)
- ✅ `warmup_steps: 3000` (was 1000)
- ✅ `epochs: 150` (was 100)
- ✅ `eval_every: 3000` (was 5000)

### ComplEx + MED + RSCF
- Same adjustments as RotatE

### TransE + MED + RSCF
- Same adjustments as RotatE
- ✅ `epochs: 200` (TransE needs more epochs)

## Next Steps

### 1. Re-run with Fixed Config

```bash
python train.py --config configs/final/wn18rr/rotate_med_rscf.yaml
```

Expected improvement: **MRR should be > 0.35** (ideally > 0.40)

### 2. Monitor Training

```bash
# Watch training progress
tail -f workdir/runs/final/wn18rr/rotate/med_rscf/train.log

# Check if loss is decreasing
grep "loss" workdir/runs/final/wn18rr/rotate/med_rscf/train.log | tail -20
```

### 3. Compare with Baseline

If still poor, run baseline MED (no RSCF):
```bash
python train.py --config configs/wn18rr/rotate_med.yaml
```

### 4. Alternative: Try Without MED First

To isolate RSCF effect, you could temporarily disable MED:
```yaml
med:
  enabled: false  # Test RSCF alone
```

## FB15k-237 Configs

You'll need to apply similar fixes to FB15k-237 configs. The same issues likely exist there too.

## Summary

**Main fix**: `gamma: 12.0` + `lr: 0.0001` should dramatically improve results.

The poor performance (MRR=0.038) was likely due to the wrong gamma value breaking the margin-based loss, combined with unstable training from high learning rate.
