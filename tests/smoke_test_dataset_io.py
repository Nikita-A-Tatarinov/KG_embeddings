# tests/smoke_test_dataloaders.py
import os
import random
import tempfile

import torch

from dataset.kg_dataset import BidirectionalOneShotIterator, KGIndex, build_test_loaders, build_train_loaders, load_kg


def _write_raw_dataset(root, name, train, valid, test):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)

    def w(fname, triples):
        with open(os.path.join(d, fname), "w") as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")

    w("train.txt", train)
    w("valid.txt", valid)
    w("test.txt", test)
    return d


def run(device=None):
    random.seed(0)
    torch.manual_seed(0)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    with tempfile.TemporaryDirectory() as tmp:
        # Tiny KG where filtered sampling can actually exclude something:
        # True triples:
        # (e0, r0, e1), (e2, r0, e1)  -> for (t=e1,r0), banned heads = {e0,e2}
        # (e0, r1, e2)                -> for (h=e0,r1), banned tails = {e2}
        train = [("e0", "r0", "e1"), ("e2", "r0", "e1"), ("e0", "r1", "e2")]
        valid = [("e2", "r1", "e0")]
        test = [("e1", "r0", "e0")]
        _write_raw_dataset(tmp, "FB15k-237", train, valid, test)

        tr, va, te, e2id, r2id = load_kg(tmp, "FB15k-237", prefer_ids=False)
        nentity, nrelation = len(e2id), len(r2id)
        all_true = KGIndex(tr.tolist() + va.tolist() + te.tolist(), nentity, nrelation)

        # ---------- Train loaders with filtering ----------
        neg_size = 8
        dl_h, dl_t = build_train_loaders(
            tr,
            nentity,
            nrelation,
            negative_size=neg_size,
            batch_size=4,
            num_workers=0,
            use_filtered=True,
            all_true=all_true,
        )
        it = BidirectionalOneShotIterator(dl_h, dl_t)

        # Head-batch batch
        pos, neg, w, mode = next(it)
        assert mode == "head-batch"
        assert pos.shape == (min(4, tr.shape[0]), 3)
        assert neg.shape[-1] == neg_size and w.shape[0] == pos.shape[0]

        # For any row with (t=e1, r=r0), ensure heads {e0,e2} are not sampled
        inv_e = {v: k for k, v in e2id.items()}
        inv_r = {v: k for k, v in r2id.items()}
        for i in range(pos.size(0)):
            h, r, t = pos[i].tolist()
            if inv_r[r] == "r0" and inv_e[t] == "e1":
                banned_heads = {e2id["e0"], e2id["e2"]}
                assert not any(int(x) in banned_heads for x in neg[i].tolist()), (
                    "Filtered head-batch produced banned head"
                )

        # Tail-batch batch
        pos2, neg2, w2, mode2 = next(it)
        assert mode2 == "tail-batch"
        assert pos2.shape == (min(4, tr.shape[0]), 3)
        # For any row with (h=e0, r=r1), ensure tails {e2} are not sampled
        for i in range(pos2.size(0)):
            h, r, t = pos2[i].tolist()
            if inv_r[r] == "r1" and inv_e[h] == "e0":
                banned_tails = {e2id["e2"]}
                assert not any(int(x) in banned_tails for x in neg2[i].tolist()), (
                    "Filtered tail-batch produced banned tail"
                )

        print("Train loaders (filtered negatives) ✓")

        # ---------- Test loaders (full ranking) ----------
        v_head, v_tail = build_test_loaders(va, nentity, batch_size=2, num_workers=0, filtered=True, all_true=all_true)
        t_head, t_tail = build_test_loaders(te, nentity, batch_size=2, num_workers=0, filtered=True, all_true=all_true)

        # Check gold at column 0 and filtering effect
        for loader, mode in [
            (v_head, "head-batch"),
            (v_tail, "tail-batch"),
            (t_head, "head-batch"),
            (t_tail, "tail-batch"),
        ]:
            for pos_b, cands_b, mode_b in loader:
                assert mode_b == mode
                B, N = cands_b.shape
                assert pos_b.shape == (B, 3)
                assert N == nentity
                for i in range(B):
                    h, r, t = pos_b[i].tolist()
                    gold = h if mode == "head-batch" else t
                    assert cands_b[i, 0].item() == gold, "Gold is not at column 0"

                    # Basic filtered sanity on our tiny KG:
                    if mode == "head-batch":
                        banned = set(all_true.tr2h.get((t, r), set()))
                        banned.discard(h)
                        # No banned in first few entries (not guaranteed for *all*),
                        # but our builder tries to push banned to the tail.
                        assert not any(int(x) in banned for x in cands_b[i, 1:5].tolist()), (
                            "Filtered eval still showing banned heads up front"
                        )
                    else:
                        banned = set(all_true.hr2t.get((h, r), set()))
                        banned.discard(t)
                        assert not any(int(x) in banned for x in cands_b[i, 1:5].tolist()), (
                            "Filtered eval still showing banned tails up front"
                        )
        print("Test loaders (gold@0, filtered) ✓")

    print("\nAll dataloader smoke tests passed.")


if __name__ == "__main__":
    run()
