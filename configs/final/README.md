# Final Report Experiments

## Structure

```
configs/final/
├── fb15k237/
│   ├── complex_med_rscf.yaml
│   ├── complex_med_mi.yaml
│   ├── complex_med_rscf_mi.yaml
│   ├── rotate_med_rscf.yaml
│   ├── rotate_med_mi.yaml
│   ├── rotate_med_rscf_mi.yaml
│   ├── transe_med_rscf.yaml
│   ├── transe_med_mi.yaml
│   ├── transe_med_rscf_mi.yaml
│   ├── distmult_med_rscf.yaml
│   └── ... (DistMult MI and RSCF+MI to be added)
└── wn18rr/
    ├── complex_med_rscf.yaml
    ├── complex_med_mi.yaml
    ├── complex_med_rscf_mi.yaml
    ├── rotate_med_rscf.yaml
    ├── rotate_med_mi.yaml
    ├── rotate_med_rscf_mi.yaml
    ├── transe_med_rscf.yaml
    ├── transe_med_mi.yaml
    ├── transe_med_rscf_mi.yaml
    ├── distmult_med_rscf.yaml
    └── ... (DistMult MI and RSCF+MI to be added)
```

## Running Experiments

### All Experiments (18 configs)

```bash
bash scripts/run_final_all.sh
```

Runs all RSCF, MI, and RSCF+MI experiments sequentially.

### RSCF Experiments (6 configs)

```bash
bash scripts/run_final_rscf.sh
```

**Configs:**
- ComplEx + MED + RSCF (FB15k-237, WN18RR)
- RotatE + MED + RSCF (FB15k-237, WN18RR)
- TransE + MED + RSCF (FB15k-237, WN18RR)

### MI Experiments (6 configs)

```bash
bash scripts/run_final_mi.sh
```

**Configs:**
- ComplEx + MED + MI (FB15k-237, WN18RR)
- RotatE + MED + MI (FB15k-237, WN18RR)
- TransE + MED + MI (FB15k-237, WN18RR)

### RSCF+MI Experiments (6 configs)

```bash
bash scripts/run_final_rscf_mi.sh
```

**Configs:**
- ComplEx + MED + RSCF + MI (FB15k-237, WN18RR)
- RotatE + MED + RSCF + MI (FB15k-237, WN18RR)
- TransE + MED + RSCF + MI (FB15k-237, WN18RR)

### Individual Execution

```bash
# Run one experiment at a time
python train.py --config configs/final/fb15k237/complex_med_rscf.yaml
```

## Output Location

All results go to:
```
workdir/runs/final/
├── fb15k237/
│   ├── complex/
│   │   ├── med_rscf/
│   │   ├── med_mi/
│   │   └── med_rscf_mi/
│   └── ...
└── wn18rr/
    └── ...
```

## Monitoring

```bash
# Watch training
tail -f workdir/runs/final/fb15k237/complex/med_rscf/train.log

# Check validation metrics
grep "val_MRR" workdir/runs/final/fb15k237/complex/med_rscf/train.log
```

## Time Estimates

- **With sampling** (10% train + 20% valid): ~40-50 min per 100 epochs
- **Full dataset**: ~8-16 hours per 100 epochs (FB15k-237)

To enable sampling, uncomment in config files:
```yaml
data:
  sample_ratio: 0.1
  sample_valid_ratio: 0.2
  sample_seed: 42
```
