# Parameter Tuning Guide for MED + RSCF/MI Experiments

## Issues Identified

### 1. **RSCF Alpha Parameter Not Used**
The config files specify `rscf_alpha: 0.1`, but this parameter is **not implemented** in the RSCF code. The RSCF module (`models/rscf.py`) doesn't use any scaling factor.

### 2. **Inconsistent Gamma Values**
- Baseline `rotate_med.yaml`: `gamma: 12.0`
- Your `rotate_med_rscf.yaml`: `gamma: 6.0`
- Midterm report suggests RotatE needs margin 6.0-12.0

### 3. **Potential RSCF Transformation Issues**
The RSCF transformations `(change_n + 1.0) * e` might be too aggressive, especially when combined with MED's multi-dimensional training.

### 4. **Learning Rate Might Be Too High**
With RSCF adding extra parameters (A1, A2, A3 matrices), the same lr=0.0005 might cause instability.

---

## Recommended Parameter Adjustments

### Option 1: Conservative Tuning (Start Here)

```yaml
model:
  name: RotatE
  base_dim: 80
  gamma: 12.0              # ✓ Match baseline (was 6.0)
  use_rscf: true

med:
  enabled: true
  dims: [10, 20, 40, 80]
  submodels_per_step: 3

optim:
  name: Adam
  lr: 0.0001               # ✓ Reduce from 0.0005 (5x lower)
  weight_decay: 1e-5       # ✓ Add regularization (was 0.0)
  betas: [0.9, 0.999]
  grad_clip: 1.0

train:
  epochs: 150              # ✓ More epochs (was 100)
```

### Option 2: Moderate Tuning

```yaml
model:
  name: RotatE
  base_dim: 80
  gamma: 12.0
  use_rscf: true

optim:
  name: Adam
  lr: 0.0002               # ✓ Reduce from 0.0005 (2.5x lower)
  weight_decay: 1e-6       # ✓ Light regularization
  betas: [0.9, 0.999]
  grad_clip: 1.0

train:
  epochs: 120
```

### Option 3: Aggressive Learning Rate Reduction

```yaml
optim:
  name: Adam
  lr: 0.00005              # ✓ 10x lower than baseline
  weight_decay: 1e-5
  
train:
  epochs: 200              # More epochs with slower learning
```

---

## Model-Specific Recommendations

### RotatE + RSCF
- **gamma**: 12.0 (not 6.0)
- **lr**: 0.0001 - 0.0002
- **weight_decay**: 1e-6 to 1e-5
- **epochs**: 120-150

### ComplEx + RSCF
- **gamma**: 12.0
- **lr**: 0.0001 - 0.0002
- **weight_decay**: 1e-5
- **epochs**: 120-150

### TransE + RSCF
- **gamma**: 2.0 (TransE uses smaller margins)
- **lr**: 0.0001
- **weight_decay**: 1e-5
- **epochs**: 150-200

---

## Additional Tuning Strategies

### 1. Warmup Steps
```yaml
sched:
  name: linear
  warmup_steps: 5000       # ✓ Increase from 1000
  total_steps: 0
```

### 2. MED Configuration
```yaml
med:
  enabled: true
  dims: [20, 40, 80]       # ✓ Remove dimension 10 (might be too small)
  submodels_per_step: 2    # ✓ Train fewer dimensions simultaneously
```

### 3. Batch Size Adjustment
```yaml
data:
  train_bs: 512            # ✓ Reduce from 1024 for stability
  test_bs: 128
```

### 4. Gradient Clipping
```yaml
optim:
  grad_clip: 0.5           # ✓ Stricter clipping (was 1.0)
```

---

## Debugging Steps

### 1. Check Training Loss
```bash
grep "loss" workdir/runs/final/wn18rr/rotate/med_rscf/train.log | tail -20
```

If loss is:
- **Not decreasing**: Learning rate too low OR model broken
- **Exploding (NaN)**: Learning rate too high OR gradient issues
- **Decreasing but val_MRR low**: Overfitting OR RSCF interfering

### 2. Compare with Baseline MED
Run baseline MED (no RSCF) first:
```bash
python train.py --config configs/wn18rr/rotate_med.yaml
```

Expected baseline: MRR ≈ 0.40-0.41 (from midterm Table 1: 0.4084-0.4132)

### 3. Test Without MED
Temporarily disable MED to isolate RSCF effect:
```yaml
med:
  enabled: false
```

### 4. Monitor RSCF Parameters
Add to training loop to check if RSCF matrices are learning:
```python
# Check RSCF parameter norms
if hasattr(model, '_rscf'):
    print(f"A1 norm: {model._rscf.A1.norm():.4f}")
```

---

## Expected Results

### Baseline MED (WN18RR, RotatE, dims 128/256/512)
From midterm report Table 1:
- **128d**: MRR = 0.4132, H@10 = 0.4858
- **256d**: MRR = 0.4084, H@10 = 0.4877
- **512d**: MRR = 0.3989, H@10 = 0.4865

### Target with RSCF
- Should be **comparable or better** than baseline MED
- Minimum acceptable: MRR > 0.35
- Good result: MRR > 0.40
- Your result (MRR = 0.038) suggests **major issue**

---

## Quick Fix: Updated Config

Here's a complete fixed config to try first:

```yaml
seed: 42
device: auto

dataset:
  name: WN18RR
  source: hf
  hf_name: VLyb/WN18RR
  filtered_eval: true

data:
  neg_size: 64
  train_bs: 512              # Reduced for stability
  test_bs: 128
  num_workers: 4

model:
  name: RotatE
  base_dim: 80
  gamma: 12.0                # Fixed from 6.0
  use_rscf: true

med:
  enabled: true
  dims: [20, 40, 80]         # Removed 10d
  submodels_per_step: 2      # Reduced from 3

optim:
  name: Adam
  lr: 0.0001                 # Reduced from 0.0005
  weight_decay: 1e-5         # Added regularization
  betas: [0.9, 0.999]
  grad_clip: 0.5             # Stricter clipping

sched:
  name: linear
  warmup_steps: 3000         # Increased warmup
  total_steps: 0

train:
  epochs: 150                # More epochs
  log_every: 100
  eval_every: 3000           # More frequent eval
  save_every: 10000
  out_dir: ./workdir/runs/final/wn18rr/rotate/med_rscf_tuned
  save_best_metric: MRR
```

---

## Implementation Note

The `rscf_alpha` parameter in configs is currently **not used**. If you want to implement it, you'd need to modify `models/rscf.py` to scale the transformations:

```python
def transform_entity(self, e: torch.Tensor, r: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    change = torch.matmul(r, A1)
    change_n = self._normalize(change)
    return (alpha * change_n + 1.0) * e  # Scale the change
```

But for now, try the parameter adjustments above first!
