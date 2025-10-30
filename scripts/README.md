# Batch Execution Scripts

All scripts are ready for GPU cluster execution with sampling enabled.

## Quick Start

**Test setup (recommended first):**
```bash
bash scripts/quick_test.sh
```
Runs 6 configs (one per model) in ~4 hours to verify everything works.

---

## Run All Experiments

**By dataset:**
```bash
bash scripts/run_fb15k237_all.sh    # 48 configs, ~32 hours
bash scripts/run_wn18rr_all.sh      # 48 configs, ~32 hours
```

**By model (both datasets):**
```bash
bash scripts/run_rotate_all.sh      # 16 configs, ~11 hours
bash scripts/run_complex_all.sh     # 16 configs, ~11 hours
bash scripts/run_distmult_all.sh    # 16 configs, ~11 hours
bash scripts/run_transe_all.sh      # 16 configs, ~11 hours
bash scripts/run_transh_all.sh      # 16 configs, ~11 hours
bash scripts/run_pairre_all.sh      # 16 configs, ~11 hours
```

---

## What's Enabled

All configs run with:
- ✅ `sample_ratio: 0.1` (10% training data)
- ✅ `sample_valid_ratio: 0.2` (20% validation data)
- ✅ `sample_seed: 42` (reproducibility)

~40-50 minutes per 100 epochs (vs 8-16 hours full dataset)

---

## Single Config Execution

```bash
python train.py --config configs/fb15k237/rotate_d10.yaml
```

All 96 configs ready to run individually.
