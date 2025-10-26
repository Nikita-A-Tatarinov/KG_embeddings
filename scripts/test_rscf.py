# scripts/test_rscf.py
import torch
from models.transe import TransE          # or rotate/rotatev2/distmult/complex_
from models.rscf import RSCFPlugin

# ---- dummy setup ----
nentity, nrelation = 100, 20
base_dim = 64
gamma = 12.0

# 1) make a real KGE model (not KGModel)
base = TransE(nentity=nentity, nrelation=nrelation, base_dim=base_dim, gamma=gamma)

# 2) wrap with RSCF
model = RSCFPlugin(base, use_rt=True, use_rp=True, rp_weight=0.1)

# ---- random batch (correct ranges) ----
B, K = 4, 3
heads  = torch.randint(0, nentity,  (B,))
rels   = torch.randint(0, nrelation,(B,))   # <-- relation ids from nrelation
tails  = torch.randint(0, nentity,  (B,))
pos    = torch.stack([heads, rels, tails], dim=1)  # (B,3)

neg = torch.randint(0, nentity, (B, K))     # entity ids for corrupted head/tail

# ---- run forwards ----
# positives
logits_pos, aux = model(pos, mode="single", crop_dim=32, return_aux=True)
print("pos logits:", logits_pos.shape, logits_pos[:3])
print("RP loss:", float(aux["rp_loss"]))

# tail-batch negatives
logits_tail, _ = model((pos, neg), mode="tail-batch", crop_dim=32)
print("tail-batch logits:", logits_tail.shape)

# head-batch negatives
logits_head, _ = model((pos, torch.randint(0, nentity, (B, K))), mode="head-batch", crop_dim=32)
print("head-batch logits:", logits_head.shape)