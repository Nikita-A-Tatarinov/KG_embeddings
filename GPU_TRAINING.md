# GPU Training Guide

## Quick GPU Check

Before starting GPU training, verify your setup:

```bash
PYTHONPATH=. python3 tests/test_gpu_ready.py
```

This will check:
- ✅ CUDA availability
- ✅ GPU device properties
- ✅ Model operations on GPU
- ✅ Training steps on GPU
- ✅ All wrappers (MED, RSCF, MI) on GPU
- ✅ Data loading to GPU

## GPU Configuration

All configuration files are **already set up for GPU** with `device: auto`:

```yaml
# configs/fb15k237/rotate_d40.yaml
seed: 42
device: auto  # ← Automatically uses GPU if available, falls back to CPU
```

The `device: auto` setting means:
- On GPU systems: Uses CUDA automatically
- On CPU systems: Falls back to CPU automatically
- No manual configuration needed!

## Training on GPU

### Single Configuration

Train a single configuration:

```bash
PYTHONPATH=. python3 train.py --config configs/fb15k237/rotate_d40.yaml
```

The trainer will:
1. Auto-detect GPU and print device info
2. Train the model on GPU
3. Evaluate during training (on GPU)
4. Save checkpoints
5. Run final test evaluation (on GPU)

### All Configurations

Run all experiments (improved script with evaluation):

```bash
./scripts/run_rotate_all.sh
```

**What the script does**:
1. ✅ Checks GPU availability at start
2. ✅ Trains each configuration
3. ✅ Evaluates best checkpoint after training
4. ✅ Saves final metrics to JSON
5. ✅ Provides summary of successes/failures

**Features**:
- Runs 16 configurations (8 for FB15k-237, 8 for WN18RR)
- Tests: d20, d40, d80, d160 (base dimensions)
- Tests: MED, MED+RSCF, MED+MI, MED+RSCF+MI
- Continues on failure (won't stop entire batch)
- Tracks success/failure counts

### Output Structure

```
workdir/runs/
├── fb15k237/
│   └── rotate/
│       ├── d20/
│       │   ├── ckpt_last.pt
│       │   ├── ckpt_best_mrr.pt
│       │   ├── final_test_metrics.json  ← New!
│       │   └── train.csv
│       ├── d40/
│       ├── d80/
│       └── med_dims_20_40_80_160/
└── wn18rr/
    └── rotate/
        └── ...
```

## GPU Memory Management

### Recommended Batch Sizes by GPU Memory

| GPU Memory | train_bs | test_bs | Notes |
|------------|----------|---------|-------|
| 8 GB       | 512      | 64      | Small GPU |
| 16 GB      | 1024     | 128     | Default (current configs) |
| 24 GB      | 2048     | 256     | Large GPU |
| 40+ GB     | 4096     | 512     | Multi-GPU or A100 |

Current configs use `train_bs: 1024` and `test_bs: 128`, which work well on 16GB GPUs.

### If You Run Out of Memory

**Option 1**: Reduce batch size in config:
```yaml
data:
  train_bs: 512  # Reduce from 1024
  test_bs: 64    # Reduce from 128
```

**Option 2**: Reduce embedding dimension:
```yaml
model:
  base_dim: 80   # Reduce from 160
```

**Option 3**: Reduce workers (saves a bit of memory):
```yaml
data:
  num_workers: 2  # Reduce from 4
```

### Multi-GPU Training

The current codebase uses single GPU. For multi-GPU:

1. The model is already DataParallel-compatible
2. Wrap model after `.to(device)`:
   ```python
   if torch.cuda.device_count() > 1:
       model = torch.nn.DataParallel(model)
   ```

This is not implemented in the current trainer but can be added if needed.

## Performance Expectations

### Training Speed (GPU vs CPU)

Approximate speedup with GPU (depends on GPU model):

| Component | CPU (est.) | GPU (est.) | Speedup |
|-----------|------------|------------|---------|
| TransE d40 | ~2-3 h/epoch | ~5-10 min/epoch | 10-30x |
| RotatE d160 | ~5-8 h/epoch | ~15-30 min/epoch | 10-20x |
| MED (multi-dim) | ~10-15 h/epoch | ~30-60 min/epoch | 10-20x |

**Note**: These are rough estimates. Actual speed depends on:
- GPU model (V100, A100, RTX 3090, etc.)
- Dataset size (FB15k-237 is larger than WN18RR)
- Batch size
- Number of workers

### Full Training Time

For 100 epochs on FB15k-237:

| Configuration | GPU Time (est.) | CPU Time (est.) |
|---------------|-----------------|-----------------|
| RotatE d20    | ~3-5 hours      | ~100+ hours     |
| RotatE d40    | ~5-8 hours      | ~150+ hours     |
| RotatE d160   | ~20-40 hours    | ~500+ hours     |
| MED (all dims)| ~30-50 hours    | ~800+ hours     |

**Recommendation**: Use GPU for production training. CPU is only for testing.

## Monitoring GPU Usage

### During Training

**Terminal 1**: Run training
```bash
./scripts/run_rotate_all.sh
```

**Terminal 2**: Monitor GPU
```bash
watch -n 1 nvidia-smi
```

This shows:
- GPU utilization (should be 90-100% during training)
- Memory usage
- Temperature
- Power consumption

### Check GPU Utilization

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1
```

Expected during training:
- GPU Utilization: 90-100%
- Memory Used: 4-12 GB (depending on batch size)

If GPU utilization is low (<50%):
- Increase `num_workers` in config
- Increase batch size
- Check if data loading is bottleneck

## Troubleshooting

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce `train_bs` in config (e.g., 1024 → 512)
2. Reduce `test_bs` (e.g., 128 → 64)
3. Reduce `base_dim` (e.g., 160 → 80)
4. Reduce `num_workers` (frees up some memory)

### Issue: "GPU not detected" but GPU is present

**Check**:
```bash
nvidia-smi  # Should show GPU
python3 -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

If `nvidia-smi` works but PyTorch doesn't see GPU:
- Reinstall PyTorch with CUDA support
- Check CUDA version compatibility

### Issue: Training is slow on GPU

**Possible causes**:
1. **Data loading bottleneck**: Increase `num_workers` to 4-8
2. **Small batches**: GPU works best with larger batches
3. **CPU-GPU transfer**: Using `pin_memory=True` in dataloaders (already enabled)

### Issue: Different results on GPU vs CPU

Small numerical differences are normal due to:
- Different floating-point operations
- Atomic operations in CUDA
- Reduced precision in some GPU operations

These differences should be tiny (<0.01 in metrics).

## Best Practices

### 1. Always Test First

Before running full training:
```bash
# Quick GPU verification
PYTHONPATH=. python3 tests/test_gpu_ready.py

# Quick training test (1 epoch)
# Modify config: epochs: 1
PYTHONPATH=. python3 train.py --config configs/fb15k237/rotate_d20.yaml
```

### 2. Monitor First Epoch

Watch the first epoch carefully:
- Check GPU utilization (should be high)
- Check memory usage (should be stable)
- Check training speed (should be fast)
- Check loss (should decrease)

If anything looks wrong, stop and investigate.

### 3. Use Checkpointing

The trainer automatically saves:
- `ckpt_last.pt`: Latest checkpoint
- `ckpt_best_mrr.pt`: Best validation MRR
- `ckpt_step*.pt`: Periodic checkpoints

If training crashes, you can resume (requires manual implementation).

### 4. Save Disk Space

If disk space is limited:
- Disable periodic checkpoints: Set `save_every` very high
- Only keep best checkpoint
- Delete intermediate checkpoints after training

### 5. Batch Processing

Don't run all 16 configs at once:
```bash
# Run FB15k-237 first
./scripts/run_rotate_all.sh  # Comment out WN18RR configs

# Then run WN18RR
./scripts/run_rotate_all.sh  # Comment out FB15k-237 configs
```

## Configuration Files Summary

### FB15k-237 Configurations

| Config | Features | Training Time (GPU) |
|--------|----------|---------------------|
| `rotate_d20.yaml` | Base, d=20 | ~3-5 hours |
| `rotate_d40.yaml` | Base, d=40 | ~5-8 hours |
| `rotate_d80.yaml` | Base, d=80 | ~10-15 hours |
| `rotate_d160.yaml` | Base, d=160 | ~20-30 hours |
| `rotate_med.yaml` | MED [20,40,80,160] | ~30-40 hours |
| `rotate_med_rscf.yaml` | MED + RSCF | ~35-45 hours |
| `rotate_med_mi.yaml` | MED + MI | ~35-45 hours |
| `rotate_med_rscf_mi.yaml` | MED + RSCF + MI | ~40-50 hours |

### WN18RR Configurations

Same structure as FB15k-237. WN18RR is smaller, so training is ~20-30% faster.

## Expected Results

### FB15k-237 (Test Set)

Approximate expected MRR (after 100 epochs):

| Configuration | MRR (approx.) |
|---------------|---------------|
| RotatE d160   | 0.33-0.34     |
| MED           | 0.34-0.35     |
| MED + RSCF    | 0.34-0.36     |
| MED + MI      | 0.34-0.36     |
| MED + RSCF + MI | 0.35-0.37  |

**Note**: These are rough estimates. Actual results vary based on:
- Random seed
- Hyperparameters
- Training dynamics

## Post-Training

After training completes:

### 1. Check Metrics

```bash
# View final metrics
cat workdir/runs/fb15k237/rotate/d40/final_test_metrics.json
```

### 2. Compare Results

```bash
# Compare all results
for d in workdir/runs/fb15k237/rotate/*/final_test_metrics.json; do
  echo "$d:"
  cat "$d"
  echo ""
done
```

### 3. Plot Learning Curves

Use the plotting tool:
```bash
PYTHONPATH=. python3 tools/plot_logs.py --log workdir/runs/fb15k237/rotate/d40/train.csv
```

### 4. Re-evaluate if Needed

```bash
PYTHONPATH=. python3 evaluate.py \
  --model RotatE \
  --ckpt workdir/runs/fb15k237/rotate/d40/ckpt_best_mrr.pt \
  --data-root ./data \
  --dataset FB15k-237 \
  --batch-size 128 \
  --filtered \
  --device auto
```

## Summary

✅ **All configs are GPU-ready** with `device: auto`  
✅ **Script includes evaluation** after training  
✅ **GPU verification test** available  
✅ **Automatic fallback to CPU** if no GPU  
✅ **Full pipeline tested** on CPU (GPU will be faster)  

**To start GPU training**:
```bash
# 1. Verify GPU
PYTHONPATH=. python3 tests/test_gpu_ready.py

# 2. Run all experiments
./scripts/run_rotate_all.sh

# 3. Monitor progress
# (in another terminal)
watch -n 1 nvidia-smi
```

Good luck with your training! 🚀
