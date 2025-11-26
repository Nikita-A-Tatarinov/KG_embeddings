# MI Device Mismatch Fix

## What Was Wrong

The MI module's projection layers (`q_proj` and `t_proj`) were created on **CPU** but the embeddings were on **GPU**, causing:
```
RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cuda:0, different from other tensors on cpu
```

## Root Cause

In `runner/trainer.py`:
1. Model created and moved to GPU: `model.to(self.device)` (line 240)
2. MI module attached **after**: `attach_mi(model, ...)` (line 260)
3. `attach_mi` creates new `nn.Linear` layers but they stay on CPU
4. When training starts, embeddings (GPU) × MI projections (CPU) → crash

## The Fix

Added one line to move MI module to device after attaching:
```python
mi_module = attach_mi(model, ...)
mi_module.to(self.device)  # NEW: Move to GPU
```

## Now You Can Run

```bash
python train.py --config configs/final/wn18rr/rotate_mi_512d.yaml
```

Or with memory fragmentation fix:
```bash
./run_with_mem_fix.sh python train.py --config configs/final/wn18rr/rotate_mi_512d.yaml
```
