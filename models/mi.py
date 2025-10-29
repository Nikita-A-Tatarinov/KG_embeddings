from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleCritic(nn.Module):
    """Small MLP critic scoring pairs of vectors."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a, b: (..., D)
        x = torch.cat([a, b], dim=-1)
        return self.net(x).squeeze(-1)


class MI_Module(nn.Module):
    """
    Minimal mutual information module exposing InfoNCE and JSD losses.

    Accepts different input dims for query and target and projects both to a shared proj_dim.
    """

    def __init__(self, q_in_dim: int, t_in_dim: int, proj_dim: int = 128, critic_hidden: int = 256):
        super().__init__()
        self.q_in_dim = int(q_in_dim)
        self.t_in_dim = int(t_in_dim)
        self.proj_dim = int(proj_dim)
        # projectors to a common space
        self.q_proj = nn.Linear(self.q_in_dim, self.proj_dim)
        self.t_proj = nn.Linear(self.t_in_dim, self.proj_dim)
        self.critic = SimpleCritic(self.proj_dim, hidden=critic_hidden)

    def encode_q(self, q: torch.Tensor) -> torch.Tensor:
        return F.relu(self.q_proj(q))

    def encode_t(self, t: torch.Tensor) -> torch.Tensor:
        return F.relu(self.t_proj(t))

    def info_nce_loss(self, q: torch.Tensor, pos_t: torch.Tensor, neg_t: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        """
        q: (B, D)
        pos_t: (B, D)
        neg_t: (B, N, D)
        returns scalar loss (avg over batch)
        """
        B = q.size(0)
        q_z = self.encode_q(q)  # (B,proj_dim)
        pos_z = self.encode_t(pos_t)  # (B,proj_dim)
        neg_z = self.encode_t(neg_t.view(-1, neg_t.size(-1))).view(B, neg_t.size(1), -1)  # (B,N,proj_dim)

        # compute scores: positive and negatives
        pos_score = self.critic(q_z, pos_z) / temperature  # (B,)
        neg_score = self.critic(
            q_z.unsqueeze(1).expand(-1, neg_z.size(1), -1).reshape(-1, q_z.size(-1)), neg_z.view(-1, neg_z.size(-1))
        ) / temperature
        neg_score = neg_score.view(B, -1)  # (B, N)

        denom = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)  # (B, N+1)
        # stable log-softmax over axis 1
        logits = denom
        labels = torch.zeros(B, dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def jsd_loss(self, q: torch.Tensor, pos_t: torch.Tensor, neg_q: torch.Tensor) -> torch.Tensor:
        """
        JSD estimator per paper: E_P[-sp(T(q, t))] - E_{P'}[sp(T(q', t))]
        q: (B, D)
        pos_t: (B, D)
        neg_q: (B, M, D)  # negative queries sampled from batch
        returns scalar loss (we minimize negative JSD so use -I_jSD as loss)
        """
        q_z = self.encode_q(q)
        t_z = self.encode_t(pos_t)
        # positive scores
        pos_s = self.critic(q_z, t_z)
        E_pos = torch.log1p(torch.exp(pos_s)).mean()

        B, M, D = neg_q.size()
        neg_q_flat = neg_q.reshape(B * M, D)
        neg_q_z = self.encode_q(neg_q_flat)
        t_rep = t_z.unsqueeze(1).expand(-1, M, -1).contiguous().reshape(B * M, -1)
        neg_s = self.critic(neg_q_z, t_rep)
        E_neg = torch.log1p(torch.exp(neg_s)).mean()

        # JSD estimate is E_neg - E_pos per paper; we want to maximize it so minimize negative
        loss = E_neg - E_pos
        return loss
