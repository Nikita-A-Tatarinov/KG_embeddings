from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def _pnorm_unit(x: torch.Tensor, p: int = 2, eps: float = 1e-12):
    denom = x.norm(p=p, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom

class RSCFPlugin(nn.Module):
    def __init__(self, base_model, p_norm: int = 2, use_rt: bool = True, use_rp: bool = True, rp_weight: float = 0.1):
        super().__init__()
        self.M = base_model
        self.p = int(p_norm)
        self.use_rt = bool(use_rt)
        self.use_rp = bool(use_rp)
        self.rp_weight = float(rp_weight)

        Dmax = self.M.max_base_dim
        ef, rf = int(self.M.ENTITY_FACTOR), int(self.M.RELATION_FACTOR)
        De_max, Dr_max = ef * Dmax, rf * Dmax

        device = self.M.entity_embedding.device
        dtype  = self.M.entity_embedding.dtype

        # store MAX-size weights and slice per crop_dim
        self.W_A1 = nn.Parameter(torch.empty(De_max, Dr_max, device=device, dtype=dtype))  # Dr -> De
        self.W_A2 = nn.Parameter(torch.empty(Dr_max, De_max, device=device, dtype=dtype))  # De -> Dr
        self.W_A3 = nn.Parameter(torch.empty(Dr_max, De_max, device=device, dtype=dtype))  # De -> Dr
        for W in (self.W_A1, self.W_A2, self.W_A3):
            nn.init.xavier_uniform_(W, gain=0.5)

    def _sizes_at(self, d: int):
        return int(self.M.ENTITY_FACTOR * d), int(self.M.RELATION_FACTOR * d)  # (De, Dr)

    def _entity_transform(self, e: torch.Tensor, r: torch.Tensor, d: int) -> torch.Tensor:
        De, Dr = self._sizes_at(d)
        # F.linear expects weight[out, in]; slice weight to (De, Dr)
        delta = F.linear(r, self.W_A1[:De, :Dr])          # (..., De)
        delta = _pnorm_unit(delta, p=self.p)
        return e * (1.0 + delta)

    def _relation_transform(self, r: torch.Tensor, h: torch.Tensor, t: torch.Tensor, d: int) -> torch.Tensor:
        if not self.use_rt:
            return r
        De, Dr = self._sizes_at(d)
        m_h = _pnorm_unit(F.linear(h, self.W_A2[:Dr, :De]), p=self.p)  # (..., Dr)
        m_t = _pnorm_unit(F.linear(t, self.W_A3[:Dr, :De]), p=self.p)  # (..., Dr)
        return r * (1.0 + m_h) * (1.0 + m_t)

    def forward(self, sample, mode: str = "single", crop_dim: int | None = None, return_aux: bool = False):
        h, r, t, rel_ids = self.M._index_full(sample, mode)
        d = crop_dim or self.M.max_base_dim
        self.M._check_dim_ok(d)

        h = self.M._slice_entity(h, d)    # (..., De)
        t = self.M._slice_entity(t, d)    # (..., De)
        r = self.M._slice_relation(r, d)  # (..., Dr)

        h_t = self._entity_transform(h, r, d)
        t_t = self._entity_transform(t, r, d)
        r_t = self._relation_transform(r, h, t, d)

        logits = self.M.score(h_t, r_t, t_t, mode, rel_ids=rel_ids, crop_dim=d)

        if not self.use_rp:
            return logits
        aux = {"rp_loss": self._relation_prediction_loss(h_t, t_t, rel_ids, d)}
        return logits, aux

    def _relation_prediction_loss(self, h_t: torch.Tensor, t_t: torch.Tensor, rel_ids: torch.Tensor, d: int) -> torch.Tensor:
        B = rel_ids.size(0)
        De, Dr = self._sizes_at(d)

        if h_t.dim() == 3:
            h_vec = h_t.mean(dim=1)
            t_vec = t_t.mean(dim=1)
        else:
            h_vec, t_vec = h_t, t_t
        joint = h_vec * t_vec                                      # (B, De)

        R = self.M._slice_relation(self.M.relation_embedding, d)   # (nrel, Dr)
        if self.use_rt:
            m_h = _pnorm_unit(F.linear(h_vec, self.W_A2[:Dr, :De]), p=self.p)  # (B, Dr)
            m_t = _pnorm_unit(F.linear(t_vec, self.W_A3[:Dr, :De]), p=self.p)  # (B, Dr)
            R = R.unsqueeze(0) * (1.0 + m_h).unsqueeze(1) * (1.0 + m_t).unsqueeze(1)  # (B,nrel,Dr)
        else:
            R = R.unsqueeze(0).expand(B, -1, -1)

        joint_proj = _pnorm_unit(F.linear(joint, self.W_A2[:Dr, :De]), p=self.p)      # (B, Dr)
        logits_rel = (joint_proj.unsqueeze(1) * R).sum(dim=-1)                        # (B, nrel)
        return F.cross_entropy(logits_rel, rel_ids)
    # ---- proxy KGModel-style attributes/methods so wrappers behave like a KGModel ----
    @property
    def entity_embedding(self):
        return self.M.entity_embedding

    @property
    def relation_embedding(self):
        return self.M.relation_embedding

    @property
    def max_base_dim(self):
        return self.M.max_base_dim

    @property
    def ENTITY_FACTOR(self):
        return self.M.ENTITY_FACTOR

    @property
    def RELATION_FACTOR(self):
        return self.M.RELATION_FACTOR

    # some utilities MEDTrainer may call during init/forward
    def _check_dim_ok(self, d: int):
        return self.M._check_dim_ok(d)

    def _slice_entity(self, E: torch.Tensor, d: int) -> torch.Tensor:
        return self.M._slice_entity(E, d)

    def _slice_relation(self, R: torch.Tensor, d: int) -> torch.Tensor:
        return self.M._slice_relation(R, d)