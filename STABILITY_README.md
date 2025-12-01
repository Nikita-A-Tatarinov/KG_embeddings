### Key Metrics
- **Stability (C_f)**: Higher is better ↑ (similar to MRR, H@10)
  - Measures robustness to input changes
  - C_f = 1/η̂_f
  
- **Lipschitz Constant (η̂_f)**: Lower is better ↓
  - Measures model sensitivity
  - η̂_f = max |f(S1) - f(S2)| / RTMD(S1, S2)

### Basic Usage
```bash
python evaluate_stability.py \
    --model TransE \
    --ckpt workdir/runs/final/wn18rr/transe/med_mi/ckpt_final.pt \
    --dataset WN18RR \
    --use-hf \
    --hf-name VLyb/WN18RR \
    --filtered \
    --batch-size 128 \
    --device auto \
    --skip-standard-eval \
    --compute-stability \
    --stability-samples 30 \
    --stability-layers 2 \
    --out workdir/runs/final/wn18rr/transe/med_mi/test_stability.json
```

### Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--compute-stability` | False | Enable stability computation |
| `--stability-samples` | 30 | Number of subgraphs to sample |
| `--stability-layers` | 2 | Depth L for RTMD computation |
| `--skip-standard-eval` | False | Skip MRR/Hits evaluation (faster) |
| `--seed` | 42 | Random seed for reproducibility |

## Output Format
The script saves results to a JSON file:

```json
{
  "stability": 0.152,
  "num_subgraph_pairs": 435
}
```

If `--skip-standard-eval` is not used, it also includes standard metrics:

```json
{
  "MRR": 0.226,
  "Hits@1": 0.028,
  "Hits@3": 0.369,
  "Hits@10": 0.501,
  "stability": 0.152,
  "num_subgraph_pairs": 435
}
```

## Interpreting Results

### Stability (C_f)
- **Range**: (0, ∞)
- **Direction**: Higher is better ✅
- **Interpretation**: 
  - Higher values → More stable/robust model
  - More likely to generalize well
  - Less sensitive to subgraph perturbations