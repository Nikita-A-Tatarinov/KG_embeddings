import os
from typing import Any, Optional


# dataset/utils.py (patch)
def _extract_triples_from_hf_split(split):
    """
    Robustly pull (head, relation, tail) from an HF split.

    Supports:
      - explicit columns: head, relation, tail / head_id, relation_id, tail_id
      - short names: h, r, t  / s, p, o
      - auto CSV cols: col_0, col_1, col_2 (or col0/column_0 variants)
      - single 'text' column with 'h<sep>r<sep>t' per line (tabs/spaces)
    """
    cols = set(split.column_names)

    def pick(*cands):
        for c in cands:
            if c in cols:
                return c
        return None

    # 1) Try explicit columns first
    h_col = pick("head_id", "head", "h", "subject", "s", "subj", "col_0", "col0", "column_0")
    r_col = pick("relation_id", "relation", "r", "predicate", "p", "rel", "col_1", "col1", "column_1")
    t_col = pick("tail_id", "tail", "t", "object", "o", "obj", "col_2", "col2", "column_2")
    if h_col and r_col and t_col:
        H = split[h_col].to_pylist() if hasattr(split[h_col], "to_pylist") else list(split[h_col])
        R = split[r_col].to_pylist() if hasattr(split[r_col], "to_pylist") else list(split[r_col])
        T = split[t_col].to_pylist() if hasattr(split[t_col], "to_pylist") else list(split[t_col])
        return H, R, T

    # 2) Fallback: single 'text' column, parse "h<sep>r<sep>t"
    if "text" in cols:
        lines = split["text"].to_pylist() if hasattr(split["text"], "to_pylist") else list(split["text"])
        H, R, T = [], [], []
        for line in lines:
            s = str(line).strip()
            if not s:
                continue
            # split on tabs first, then spaces
            parts = s.split("\t")
            if len(parts) == 1:
                parts = s.replace("\t", " ").split()
            if len(parts) != 3:
                raise ValueError(f"Cannot parse triple from text line: {line!r}")
            h, r, t = parts
            H.append(h)
            R.append(r)
            T.append(t)

        # Try to cast to int if all are numeric
        def maybe_int(lst):
            try:
                # only cast if ALL entries are integer-like
                cast = [int(x) for x in lst]
                return cast
            except Exception:
                return lst

        # If the dataset already stores ids in the text, this preserves ints
        H = maybe_int(H)
        R = maybe_int(R)
        T = maybe_int(T)
        return H, R, T

    raise ValueError(f"Cannot find H/R/T columns in HF split; columns={list(cols)}")


def _encode_maybe_str_triples(all_splits: dict[str, tuple[list[Any], list[Any], list[Any]]]):
    """
    If splits are string-labeled, build ent2id / rel2id and encode to ids.
    If already ints, just stack and infer nentity/nrelation from max+1.
    Returns (train_ids, valid_ids, test_ids, ent2id, rel2id)
    """
    # Peek types
    some = next(iter(all_splits.values()))
    H, R, T = some
    is_int_based = isinstance(H[0], int) and isinstance(R[0], int) and isinstance(T[0], int)

    if is_int_based:

        def stack(hrts):
            H, R, T = hrts
            import torch

            return torch.tensor(list(zip(H, R, T, strict=True)), dtype=torch.long)

        train = stack(all_splits["train"])
        valid = stack(all_splits["valid"])
        test = stack(all_splits["test"])
        # If ids are dense 0..N-1, ent2id/rel2id can be None; callers don’t need names.
        return train, valid, test, {}, {}

    # Build maps over ALL splits
    ent2id: dict[str, int] = {}
    rel2id: dict[str, int] = {}

    def _id(d: dict[str, int], k: str):
        if k not in d:
            d[k] = len(d)
            return d[k]
        return d[k]

    import torch

    def encode_split(hrts):
        H, R, T = hrts
        ids = []
        for h, r, t in zip(H, R, T, strict=True):
            ids.append((_id(ent2id, str(h)), _id(rel2id, str(r)), _id(ent2id, str(t))))
        return torch.tensor(ids, dtype=torch.long)

    train = encode_split(all_splits["train"])
    valid = encode_split(all_splits["valid"])
    test = encode_split(all_splits["test"])
    return train, valid, test, ent2id, rel2id


def load_kg_hf(
    name_or_ds: Any,  # "KGraph/FB15k-237" or a DatasetDict
    revision: Optional[str] = None,
):
    """
    Load an HF dataset into (train_ids, valid_ids, test_ids, ent2id, rel2id),
    matching the signature of load_kg().
    """
    from datasets import load_dataset

    # Resolve DatasetDict
    if isinstance(name_or_ds, str):
        ds = load_dataset(name_or_ds, revision=revision)
    else:
        ds = name_or_ds

    # Map common split names
    def pick_split(*names):
        for n in names:
            if n in ds:
                return ds[n]
        return None

    train = pick_split("train", "training")
    valid = pick_split("validation", "valid", "dev")
    test = pick_split("test", "testing")
    if train is None or valid is None or test is None:
        raise ValueError(f"Could not find (train/validation/test) in dataset; splits={list(ds.keys())}")

    # Extract columns robustly and encode
    splits = {
        "train": _extract_triples_from_hf_split(train),
        "valid": _extract_triples_from_hf_split(valid),
        "test": _extract_triples_from_hf_split(test),
    }
    return _encode_maybe_str_triples(splits)


def _read_triples_txt(path: str) -> list[tuple[str, str, str]]:
    """Read 'h\\t r\\t t' text triples (strings)."""
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:
                h, r, t = parts
            else:
                # tolerate tab / space mixes
                parts = line.replace("\t", " ").split()
                if len(parts) != 3:
                    raise ValueError(f"Bad triple line in {path}: {line}")
                h, r, t = parts
            triples.append((h, r, t))
    return triples


def _read_triples_id_txt(path: str) -> list[tuple[int, int, int]]:
    """Read 'h t r' or 'h r t' id files (OpenKE-like).
    We auto-detect by counting columns and looking at header.
    """
    triples = []
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip()
        # Many OpenKE files have first line as count; if so, skip it
        try:
            _ = int(first)
            # read rest as 3 ints
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split()
                if len(cols) != 3:
                    raise ValueError(f"Bad triple line in {path}: {line}")
                h, t, r = map(int, cols)  # OpenKE order (h, t, r)
                triples.append((h, r, t))
        except ValueError:
            # no count header; assume 'h r t'
            col = first.split()
            if len(col) != 3:
                raise ValueError(f"Bad triple line in {path}: {first}")
            h, r, t = map(int, col)
            triples.append((h, r, t))
            for line in f:
                line = line.strip()
                if not line:
                    continue
                h, r, t = map(int, line.split())
                triples.append((h, r, t))
    return triples


def _maybe_read_id_mapping(path: str) -> Optional[dict[str, int]]:
    """Read 'name id' mapping when present (OpenKE: entity2id.txt / relation2id.txt)."""
    if not os.path.exists(path):
        return None
    mapping = {}
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip()
        try:
            _ = int(first)  # header count
        except ValueError:
            # first line is data
            k, v = first.split()
            mapping[k] = int(v)
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split()
            mapping[k] = int(v)
    return mapping
