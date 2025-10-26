import torch
import torch.nn.functional as F
from models.transe import TransE              # simple, real-valued KGE
from models.rscf import RSCFPlugin            # your plugin (RP disabled)
from med.med_wrapper import MEDTrainer        # MED wrapper

def main():
    torch.manual_seed(0)

    # --- Small sizes
    nentity, nrelation = 500, 30
    base_dim = 64            # max width
    d_list = [16, 32, 64]    # sub-widths MED will sample from
    B, K = 32, 16            # batch size and #negatives per positive
    steps = 50               # keep short for a quick sanity run

    # --- Base model (TransE) ---
    base = TransE(nentity=nentity, nrelation=nrelation, base_dim=base_dim, gamma=12.0)

    # --- RSCF plugin (RP off so forward returns only logits) ---
    model = RSCFPlugin(base, use_rt=True, use_rp=False)

    # --- MED trainer around the model ---
    med = MEDTrainer(model, d_list=d_list, submodels_per_step=3, huber_delta=1.0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(steps):
        # Random mini-batch
        heads  = torch.randint(0, nentity,  (B,))
        rels   = torch.randint(0, nrelation,(B,))
        tails  = torch.randint(0, nentity,  (B,))
        pos    = torch.stack([heads, rels, tails], dim=1)  # (B,3)
        neg    = torch.randint(0, nentity, (B, K))         # (B,K) tail candidates

        # MED loss (tail-batch here; head-batch also works)
        loss, stats = med(pos, neg, mode='tail-batch')

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:03d} | L_total={stats['L_total']:.4f} "
                  f"L_ml={stats['L_ml']:.4f} L_ei={stats['L_ei']:.4f} dims={stats['dims']}")

if __name__ == "__main__":
    main()
