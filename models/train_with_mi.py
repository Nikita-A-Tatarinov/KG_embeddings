# train_with_mi.py
from types import SimpleNamespace
import torch, torch.nn.functional as F
from models import create_model
from dataset.fb15k237 import prepare_fb15k237

def info_nce_inbatch(model, pos_triples, tau=0.1, crop_dim=None):
    B = pos_triples.size(0)
    device = pos_triples.device
    idx = torch.arange(B, device=device)

    # ---- (h,r) -> t with in-batch tails as negatives ----
    tails = pos_triples[:, 2]                                    # (B,)
    cands_t = tails.unsqueeze(0).expand(B, B)                    # (B,B)
    s_tail = model((pos_triples, cands_t), mode="tail-batch", crop_dim=crop_dim)  # (B,B)
    L_tail = F.cross_entropy(s_tail / tau, idx)

    # ---- (r,t) -> h with in-batch heads as negatives ----
    heads = pos_triples[:, 0]                                    # (B,)
    cands_h = heads.unsqueeze(0).expand(B, B)                    # (B,B)
    s_head = model((pos_triples, cands_h), mode="head-batch", crop_dim=crop_dim)  # (B,B)
    L_head = F.cross_entropy(s_head / tau, idx)

    return L_tail, L_head

def bce_loss_openke_style(model, pos_triples, neg_cands, mode, crop_dim=None):
    # Positive logits (B,)
    s_pos = model(pos_triples, mode="single", crop_dim=crop_dim).squeeze(-1)
    # Negative logits (B,K) -> flatten to (B*K,)
    s_neg = model((pos_triples, neg_cands), mode=mode, crop_dim=crop_dim).reshape(-1)

    ones = torch.ones_like(s_pos)
    zeros = torch.zeros_like(s_neg)
    L_pos = F.binary_cross_entropy_with_logits(s_pos, ones, reduction="mean")
    L_neg = F.binary_cross_entropy_with_logits(s_neg, zeros, reduction="mean")
    return L_pos + L_neg

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load data ---
    loaders = prepare_fb15k237(
        root="./data", prefer_ids=False, filtered_eval=True,
        neg_size=64, train_bs=1024, test_bs=128, num_workers=4,
        use_hf=False, hf_name=None, hf_revision=None
    )
    (train_iter, steps_per_epoch, nentity, nrelation,
     head_loader, tail_loader) = loaders

    # --- build base model ---
    cfg = SimpleNamespace(model=SimpleNamespace(name="rotate", base_dim=256, gamma=9.0))
    model = create_model(
        name=cfg.model.name,
        nentity=nentity, nrelation=nrelation,
        base_dim=cfg.model.base_dim, gamma=cfg.model.gamma
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    # --- knobs for MI ---
    tau = 0.1
    lam_tail = 1.0
    lam_head = 1.0
    crop_dim = None 

    for epoch in range(5):
        for step in range(steps_per_epoch):
            pos, neg, _, mode = next(train_iter)
            pos, neg = pos.to(device), neg.to(device)

            opt.zero_grad(set_to_none=True)
            # Base BCE loss (same scoring API as repo)
            L_bce = bce_loss_openke_style(model, pos, neg, mode, crop_dim=crop_dim)
            # MI loss over in-batch negatives (two directions)
            L_tail, L_head = info_nce_inbatch(model, pos, tau=tau, crop_dim=crop_dim)

            loss = L_bce + lam_tail * L_tail + lam_head * L_head
            loss.backward()
            opt.step()

        print(f"epoch {epoch}  loss={loss.item():.4f}  (bce={L_bce.item():.4f}, mi={L_tail.item()+L_head.item():.4f})")

if __name__ == "__main__":
    main()
