import torch

from models import create_model


def test_mi_forward_backward():
    # small model
    model = create_model("PairRE", nentity=20, nrelation=6, base_dim=8, gamma=12.0)
    from models.mi_wrapper import attach_mi

    attach_mi(model, use_info_nce=True, use_jsd=True)

    # sample batch using model sizes
    n_ent = 20
    n_rel = 6
    sample = torch.stack([
        torch.randint(0, n_ent, (4,)),
        torch.randint(0, n_rel, (4,)),
        torch.randint(0, n_ent, (4,)),
    ], dim=1).long()
    # compute mi loss
    loss = model.compute_mi_loss(sample, neg_size=4)
    assert torch.isfinite(loss)

    # ensure backward passes
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    opt.zero_grad()
    loss.backward()
    opt.step()


if __name__ == "__main__":
    test_mi_forward_backward()
    print("MI smoke test passed")
