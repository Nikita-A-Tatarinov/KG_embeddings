import torch

from models import create_model


def test_rscf_forward():
    # tiny model
    model = create_model("PairRE", nentity=10, nrelation=5, base_dim=8, gamma=12.0)
    # attach RSCF dynamically
    from models.rscf_wrapper import attach_rscf

    attach_rscf(model)

    # single sample (h,r,t)
    sample = torch.tensor([[0, 1, 2]], dtype=torch.long)
    logits = model(sample)
    assert logits.shape == (1, 1)


if __name__ == "__main__":
    test_rscf_forward()
    print("RSCF smoke test passed")
