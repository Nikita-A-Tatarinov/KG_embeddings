# KG_embeddings

## Project setup in Pace

```
git clone #URL/KG_embeddings
cd KG_embeddings

mkdir -p ~/scratch/KG_embeddings/workdir
ln -s ~/scratch/KG_embeddings/workdir workdir

uv venv --python 3.12 --seed ~/scratch/KG_embeddings/.venv
ln -s ~/scratch/KG_embeddings/.venv .venv
source .venv/bin/activate
uv sync
```

## Running a single experiment

```
python train.py --config configs/final/wn18rr/transe_med_mi.yaml
```

If a memory error occurs:

```
./run_with_mem_fix.sh python train.py --config configs/final/wn18rr/transe_med_mi.yaml
```

The results will be in `workdir/runs/final/`.