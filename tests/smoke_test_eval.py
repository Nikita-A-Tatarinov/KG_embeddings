"""Smoke test for KGC evaluator: tiny dataset and model."""
import torch

from models.transe import TransE
from eval.kgc_eval import evaluate_model
from dataset.kg_dataset import build_test_loaders, KGIndex


def run():
    # tiny synthetic KG: 4 entities, 2 relations
    train = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 1, 3]], dtype=torch.long)
    valid = torch.tensor([[0, 0, 2]], dtype=torch.long)
    test = torch.tensor([[1, 0, 2]], dtype=torch.long)
    nentity = 4
    nrelation = 2

    all_true = KGIndex(
        torch.cat([train, valid, test], dim=0).tolist(), nentity, nrelation)
    dl_head, dl_tail = build_test_loaders(
        test, nentity, batch_size=2, filtered=True, all_true=all_true)

    model = TransE(nentity=nentity, nrelation=nrelation,
                   base_dim=10, gamma=12.0)
    # random init
    metrics = evaluate_model(model, dl_head, dl_tail,
                             device=torch.device("cpu"))
    print("Evaluator smoke metrics:", metrics)


if __name__ == "__main__":
    run()
