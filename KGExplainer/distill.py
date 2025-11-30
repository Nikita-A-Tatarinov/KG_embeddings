
# This is the main “distillation” loop. It:

# Loads the trained KGE model from a checkpoint.

# Builds the DGL graph from training triples.

# In each step:

# Sample positive triple (h,r,t)

# Sample negative tail t′

# Extract k-hop enclosing subgraphs for (h,t) and (h,t′)

# Compute teacher scores φ_pos, φ_neg

# Run evaluator on those subgraphs → Z_pos, Z_neg

# Minimize:

# MSE(φ_pos, Z_pos) + MSE(φ_neg, Z_neg) (distillation)

# margin ranking loss on scores



# May need to tweak base_dim / gamma inference using the checkpoint metadata


from __future__ import annotations

import argparse
import os
import random

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from dataset.kg_dataset import load_kg
from models.registry import create_model
from runner.checkpoint import load_checkpoint

from .graph import build_dgl_graph, k_hop_enclosing_subgraph
from .evaluator import HeteroGAT


def teacher_score(model, triple, device):
    """
    triple: (3,) LongTensor [h, r, t]
    returns scalar teacher score
    """
    triple = triple.view(1, 3).to(device)
    with torch.no_grad():
        logits = model(triple, mode="single")  # shape (1, 1) or (1,)
    return logits.view(-1)[0]  # scalar


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to trained KGE checkpoint")
    ap.add_argument("--model", required=True, help="Model name (e.g. ComplEx, TransE)")
    ap.add_argument("--data-root", required=True, help="Path to dataset root folder")
    ap.add_argument("--dataset", required=True, help="Dataset name (FB15k-237, WN18RR, ...)")
    ap.add_argument("--k-hop", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default=None, help="Where to save evaluator .pt")
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Load triples
    train_ids, valid_ids, test_ids, ent2id, rel2id = load_kg(args.data_root, args.dataset)
    nentity = len(ent2id) if ent2id else int(train_ids[:, [0, 2]].max()) + 1
    nrelation = len(rel2id) if rel2id else int(train_ids[:, 1].max()) + 1

    # 2) Build DGL graph from train triples
    g_full = build_dgl_graph(train_ids, nentity, nrelation, add_reverse=True)

    # 3) Load teacher KGE model
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    base_dim = 100  # or infer from ckpt metadata if available
    gamma = 12.0
    teacher = create_model(args.model, nentity=nentity, nrelation=nrelation, base_dim=base_dim, gamma=gamma)
    load_checkpoint(args.ckpt, teacher)
    teacher.to(device)
    teacher.eval()

    # 4) Build evaluator
    etypes = g_full.canonical_etypes
    evaluator = HeteroGAT(etypes, num_nodes=nentity, in_size=32, hid_size=64, out_size=32, n_heads=4)
    evaluator.to(device)

    optimizer = Adam(evaluator.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    margin_loss = nn.MarginRankingLoss(margin=1.0, reduction="mean")

    train_triples = train_ids.clone()
    all_entities = torch.arange(nentity)

    for step in tqdm(range(args.num_steps), desc="Distilling evaluator"):
        evaluator.train()

        batch_idx = torch.randint(0, len(train_triples), (args.batch_size,))
        pos_triples = train_triples[batch_idx]

        pos_graphs = []
        neg_graphs = []
        pos_scores_teacher = []
        neg_scores_teacher = []

        for (h, r, t) in pos_triples.tolist():
            # positive graph
            g_pos = k_hop_enclosing_subgraph(g_full, h, t, k_hop=args.k_hop)
            pos_graphs.append(g_pos)
            pos_scores_teacher.append(teacher_score(teacher, torch.tensor([h, r, t]), device))

            # sample negative tail
            # (simple random negative, can be improved using true_tail filtering)
            t_neg = random.randint(0, nentity - 1)
            while t_neg == t:
                t_neg = random.randint(0, nentity - 1)

            g_neg = k_hop_enclosing_subgraph(g_full, h, t_neg, k_hop=args.k_hop)
            neg_graphs.append(g_neg)
            neg_scores_teacher.append(teacher_score(teacher, torch.tensor([h, r, t_neg]), device))

        # batch graphs
        b_pos = dgl.batch(pos_graphs).to(device)
        b_neg = dgl.batch(neg_graphs).to(device)
        pos_scores_teacher = torch.stack(pos_scores_teacher).to(device)
        neg_scores_teacher = torch.stack(neg_scores_teacher).to(device)

        # evaluator scores
        _, pos_scores_eval = evaluator(b_pos)
        _, neg_scores_eval = evaluator(b_neg)

        # distillation loss (match teacher)
        loss_distill = mse(pos_scores_eval, pos_scores_teacher) + mse(neg_scores_eval, neg_scores_teacher)

        # margin ranking (pos > neg)
        y = torch.ones_like(pos_scores_eval, device=device)
        loss_margin = margin_loss(pos_scores_eval, neg_scores_eval, y)

        loss = loss_distill + loss_margin

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            tqdm.write(f"[step {step+1}] loss={loss.item():.4f}")

    out_path = args.out or os.path.join(
        os.path.dirname(args.ckpt),
        f"kgexplainer_evaluator_{args.dataset}_{args.model}.pt",
    )
    torch.save({"state_dict": evaluator.state_dict(), "nentity": nentity}, out_path)
    print(f"Saved evaluator to: {out_path}")


if __name__ == "__main__":
    main()
