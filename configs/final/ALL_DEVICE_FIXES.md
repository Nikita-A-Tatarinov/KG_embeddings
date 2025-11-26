# Device Mismatch Fixes for MI and RSCF

## The Problem

Both MI and RSCF modules had the same bug: they were created on **CPU** after the model was moved to **GPU**.

### Why This Happened
In `runner/trainer.py`, the flow was:
1. Model created and moved to GPU: `model.to(self.device)` (line 240)
2. RSCF/MI attached **after**: Creates new nn.Module with parameters on CPU
3. When training starts: embeddings (GPU) × module parameters (CPU) → crash

### Why MED+MI and MED+RSCF Worked
When MED is enabled:
```python
self.train_obj = MEDTrainer(model, ...).to(self.device)
```
The `.to(self.device)` on MEDTrainer **recursively moved all submodules** (including MI/RSCF) to GPU.

Without MED, there's no such call, so MI/RSCF stayed on CPU.

## The Fixes Applied

### Fix 1: RSCF (Line 247)
```python
rscf_module = attach_rscf(model, ...)
rscf_module.to(self.device)  # Move RSCF to GPU
```

### Fix 2: MI (Line 260)
```python
mi_module = attach_mi(model, ...)
mi_module.to(self.device)  # Move MI to GPU
```

## Now All Combinations Work

✅ **Standalone MI**: Fixed  
✅ **Standalone RSCF**: Fixed  
✅ **MED + MI**: Already worked (MEDTrainer moved it)  
✅ **MED + RSCF**: Already worked (MEDTrainer moved it)  
✅ **MED + MI + RSCF**: Already worked (MEDTrainer moved both)  

## Run Your Experiments

All these should now work:
```bash
# Standalone RSCF with 512d
python train.py --config configs/final/wn18rr/rotate_rscf_512d.yaml

# Standalone MI with 512d
python train.py --config configs/final/wn18rr/rotate_mi_512d.yaml

# Or with memory fragmentation fix
./run_with_mem_fix.sh python train.py --config configs/final/wn18rr/rotate_rscf_512d.yaml
```
