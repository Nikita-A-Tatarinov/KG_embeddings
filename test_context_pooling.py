#!/usr/bin/env python
"""
Quick diagnostic to check if Context Pooling is working correctly
"""
import torch
from dataset.wn18rr import prepare_wn18rr
from models import create_model

print("Loading WN18RR...")
train_iter, _, _, meta = prepare_wn18rr(
    root='./data',
    use_hf=True,
    hf_name='VLyb/WN18RR',
    train_bs=64,
    test_bs=64,
    neg_size=128,
    num_workers=0,
    filtered_eval=True
)

nentity = meta['nentity']
nrelation = meta['nrelation']
train_triples = train_iter._src_head.dataset.triples

print(f"Entities: {nentity}, Relations: {nrelation}, Triples: {len(train_triples)}")

print("\nCreating Context Pooling model...")
model = create_model('cp',
                     nentity=nentity,
                     nrelation=nrelation,
                     base_dim=128,
                     gamma=12.0,
                     n_layer=5,
                     attn_dim=5,
                     dropout=0.02,
                     train_triples=train_triples,
                     accuracy_threshold=0.89,
                     recall_threshold=0.53,
                     accuracy_graph=False,
                     recall_graph=True,
                     accuracy_graph_complement=False,
                     recall_graph_complement=True)

model = model.cuda() if torch.cuda.is_available() else model
model.train()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"Model device: {model.entity_embedding.device}")

# Get a batch
pos, neg, w, mode = next(train_iter)
pos = pos.cuda() if torch.cuda.is_available() else pos
neg = neg.cuda() if torch.cuda.is_available() else neg

print(f"\nBatch info:")
print(f"  Positive shape: {pos.shape}")
print(f"  Negative shape: {neg.shape}")
print(f"  Mode: {mode}")

# Forward pass
print("\nForward pass...")
with torch.no_grad():
    pos_score = model(pos, mode='single')
    neg_score = model((pos, neg), mode=mode)
    
    print(f"  Positive scores shape: {pos_score.shape}")
    print(f"  Positive scores range: [{pos_score.min().item():.4f}, {pos_score.max().item():.4f}]")
    print(f"  Positive scores mean: {pos_score.mean().item():.4f}")
    
    print(f"  Negative scores shape: {neg_score.shape}")
    print(f"  Negative scores range: [{neg_score.min().item():.4f}, {neg_score.max().item():.4f}]")
    print(f"  Negative scores mean: {neg_score.mean().item():.4f}")
    
    # Check if positive scores are higher than negative (as they should be)
    pos_mean = pos_score.mean()
    neg_mean = neg_score.mean()
    print(f"\n  Positive > Negative? {pos_mean > neg_mean} (diff: {(pos_mean - neg_mean).item():.4f})")

print("\n✓ Context Pooling diagnostic complete")
