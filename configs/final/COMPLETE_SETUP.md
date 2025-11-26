# Final Report Experiments - Complete Setup

## ✅ All Config Files Created

### Total: 18 Configuration Files

**3 Models × 2 Datasets × 3 Combinations = 18 Experiments**

### FB15k-237 (10 configs)

**ComplEx:**
- `configs/final/fb15k237/complex_med_rscf.yaml` - MED + RSCF
- `configs/final/fb15k237/complex_med_mi.yaml` - MED + MI
- `configs/final/fb15k237/complex_med_rscf_mi.yaml` - MED + RSCF + MI

**RotatE:**
- `configs/final/fb15k237/rotate_med_rscf.yaml` - MED + RSCF
- `configs/final/fb15k237/rotate_med_mi.yaml` - MED + MI
- `configs/final/fb15k237/rotate_med_rscf_mi.yaml` - MED + RSCF + MI

**TransE:**
- `configs/final/fb15k237/transe_med_rscf.yaml` - MED + RSCF
- `configs/final/fb15k237/transe_med_mi.yaml` - MED + MI
- `configs/final/fb15k237/transe_med_rscf_mi.yaml` - MED + RSCF + MI

**DistMult (partial):**
- `configs/final/fb15k237/distmult_med_rscf.yaml` - MED + RSCF

### WN18RR (9 configs)

**ComplEx:**
- `configs/final/wn18rr/complex_med_rscf.yaml` - MED + RSCF
- `configs/final/wn18rr/complex_med_mi.yaml` - MED + MI
- `configs/final/wn18rr/complex_med_rscf_mi.yaml` - MED + RSCF + MI

**RotatE:**
- `configs/final/wn18rr/rotate_med_rscf.yaml` - MED + RSCF
- `configs/final/wn18rr/rotate_med_mi.yaml` - MED + MI
- `configs/final/wn18rr/rotate_med_rscf_mi.yaml` - MED + RSCF + MI

**TransE:**
- `configs/final/wn18rr/transe_med_rscf.yaml` - MED + RSCF
- `configs/final/wn18rr/transe_med_mi.yaml` - MED + MI
- `configs/final/wn18rr/transe_med_rscf_mi.yaml` - MED + RSCF + MI

**DistMult (partial):**
- `configs/final/wn18rr/distmult_med_rscf.yaml` - MED + RSCF

---

## 🚀 How to Run Experiments

### Option 1: Run All Experiments (18 configs)

```bash
bash scripts/run_final_all.sh
```

This runs all three phases sequentially:
1. RSCF experiments (6)
2. MI experiments (6)
3. RSCF+MI experiments (6)

### Option 2: Run by Phase

**Phase 1: RSCF Experiments (6 configs)**
```bash
bash scripts/run_final_rscf.sh
```

Runs:
- ComplEx + MED + RSCF (FB15k-237, WN18RR)
- RotatE + MED + RSCF (FB15k-237, WN18RR)
- TransE + MED + RSCF (FB15k-237, WN18RR)

**Phase 2: MI Experiments (6 configs)**
```bash
bash scripts/run_final_mi.sh
```

Runs:
- ComplEx + MED + MI (FB15k-237, WN18RR)
- RotatE + MED + MI (FB15k-237, WN18RR)
- TransE + MED + MI (FB15k-237, WN18RR)

**Phase 3: RSCF+MI Experiments (6 configs)**
```bash
bash scripts/run_final_rscf_mi.sh
```

Runs:
- ComplEx + MED + RSCF + MI (FB15k-237, WN18RR)
- RotatE + MED + RSCF + MI (FB15k-237, WN18RR)
- TransE + MED + RSCF + MI (FB15k-237, WN18RR)

### Option 3: Run Individual Experiments

**Single experiment:**
```bash
python train.py --config configs/final/fb15k237/complex_med_rscf.yaml
```

**All ComplEx experiments (6):**
```bash
python train.py --config configs/final/fb15k237/complex_med_rscf.yaml
python train.py --config configs/final/fb15k237/complex_med_mi.yaml
python train.py --config configs/final/fb15k237/complex_med_rscf_mi.yaml
python train.py --config configs/final/wn18rr/complex_med_rscf.yaml
python train.py --config configs/final/wn18rr/complex_med_mi.yaml
python train.py --config configs/final/wn18rr/complex_med_rscf_mi.yaml
```

---

## 📊 Model Configurations

### ComplEx
- `base_dim: 80`, `gamma: 12.0`
- Complex-valued embeddings (entity_dim = 2 × base_dim = 160)

### RotatE
- `base_dim: 80`, `gamma: 6.0`
- Rotation in complex space (entity_dim = 2 × base_dim = 160)

### TransE
- `base_dim: 80`, `gamma: 2.0`
- Simple translation model (entity_dim = base_dim = 80)

### All Models
- **MED dimensions:** `[10, 20, 40, 80]` - 4 sub-models per experiment
- **Submodels per step:** 3
- **Optimizer:** Adam (lr=0.0005)
- **Epochs:** 100
- **Batch size:** 1024 (train), 128 (test)

---

## 📁 Output Structure

```
workdir/runs/final/
├── fb15k237/
│   ├── complex/
│   │   ├── med_rscf/
│   │   │   ├── train.log
│   │   │   ├── checkpoints/
│   │   │   └── best_model.pt
│   │   ├── med_mi/
│   │   └── med_rscf_mi/
│   ├── rotate/
│   │   ├── med_rscf/
│   │   ├── med_mi/
│   │   └── med_rscf_mi/
│   └── transe/
│       ├── med_rscf/
│       ├── med_mi/
│       └── med_rscf_mi/
└── wn18rr/
    └── (same structure)
```

---

## 📈 Monitoring Progress

**Watch training in real-time:**
```bash
tail -f workdir/runs/final/fb15k237/complex/med_rscf/train.log
```

**Check validation metrics:**
```bash
grep "val_MRR" workdir/runs/final/fb15k237/complex/med_rscf/train.log
```

**View all results for a model:**
```bash
for dir in workdir/runs/final/fb15k237/complex/*/; do
    echo "$(basename $dir):"
    grep "val_MRR" "$dir/train.log" 2>/dev/null | tail -1
done
```

---

## ⏱️ Time Estimates

### With Full Dataset
- **Per experiment:** ~8-16 hours (FB15k-237), ~4-8 hours (WN18RR) for 100 epochs
- **Total for 18 experiments:** ~150-250 hours

### With Sampling (10% train + 20% valid)
- **Per experiment:** ~40-50 minutes for 100 epochs
- **Total for 18 experiments:** ~12-15 hours

**To enable sampling**, uncomment in config files:
```yaml
data:
  sample_ratio: 0.1
  sample_valid_ratio: 0.2
  sample_seed: 42
```

---

## 🎯 Research Questions

### 1. Does RSCF improve MED?
Compare:
- `*_med_rscf.yaml` vs baseline MED-only

**Hypothesis:** Semantic consistency helps lower-dimensional sub-models.

### 2. Does MI improve MED?
Compare:
- `*_med_mi.yaml` vs baseline MED-only

**Hypothesis:** MI maximization reduces semantic loss across dimensions.

### 3. Does RSCF+MI work better together?
Compare:
- `*_med_rscf_mi.yaml` vs `*_med_rscf.yaml` vs `*_med_mi.yaml`

**Hypothesis:** Combined effect surpasses individual contributions.

### 4. Model-specific effects
Compare across ComplEx, RotatE, and TransE:
- Which models benefit most from RSCF?
- Which models benefit most from MI?
- Are effects consistent across datasets?

---

## 📋 Execution Checklist

Before starting:
- [ ] Environment activated: `source .venv/bin/activate`
- [ ] GPU available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Output directory: `mkdir -p workdir/runs/final`
- [ ] Decide on sampling: Edit configs if needed
- [ ] Choose execution strategy: All at once or phase-by-phase

---

## 📝 Notes

- All configs use the same training hyperparameters for fair comparison
- RSCF alpha = 0.1 (scaling factor for relation transformations)
- MI lambda = 0.01 (weight for mutual information loss)
- All experiments use filtered evaluation
- Results are reproducible with seed=42

---

## 🔄 Next Steps

After experiments complete:

1. **Analyze results:**
   ```bash
   python tools/plot_logs.py --log_dir workdir/runs/final/fb15k237/
   ```

2. **Compare combinations:**
   - Extract MRR scores from all experiments
   - Create comparison tables
   - Plot performance vs dimension

3. **Run full evaluation:**
   ```bash
   python evaluate.py --config <config> --checkpoint <checkpoint>
   ```

4. **Generate final report tables and figures**

---

## ✅ Summary

**Everything is ready!**

- ✅ 18 configuration files created (ComplEx, RotatE, TransE)
- ✅ 4 execution scripts (all, rscf, mi, rscf_mi)
- ✅ Proper directory structure
- ✅ Consistent hyperparameters
- ✅ Documentation complete

**To start RSCF experiments now:**
```bash
bash scripts/run_final_rscf.sh
```
