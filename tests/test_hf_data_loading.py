"""Quick diagnostic to check if HuggingFace data loading works correctly.

This will verify:
1. Data loads from HuggingFace
2. Number of entities and relations are correct
3. Data format is correct
"""


def test_hf_loading():
    print("=" * 60)
    print("Testing HuggingFace Data Loading")
    print("=" * 60)
    print()

    from dataset.utils import load_kg_hf

    # Load FB15k-237
    print("Loading FB15k-237 from HuggingFace...")
    train, valid, test, ent2id, rel2id = load_kg_hf("KGraph/FB15k-237")

    print("✓ Data loaded successfully")
    print(f"  Train: {len(train)} triples")
    print(f"  Valid: {len(valid)} triples")
    print(f"  Test: {len(test)} triples")
    print(f"  Entities: {len(ent2id) if ent2id else train[:, [0, 2]].max().item() + 1}")
    print(f"  Relations: {len(rel2id) if rel2id else train[:, 1].max().item() + 1}")
    print()

    # Check data format
    print("Sample triples from train:")
    for i in range(min(5, len(train))):
        h, r, t = train[i].tolist()
        print(f"  ({h}, {r}, {t})")
    print()

    # Check if IDs are in valid range
    max_ent = train[:, [0, 2]].max().item()
    max_rel = train[:, 1].max().item()
    print(f"Max entity ID in train: {max_ent}")
    print(f"Max relation ID in train: {max_rel}")
    print()

    # Expected values for FB15k-237
    expected_train = 272115
    expected_valid = 17535
    expected_test = 20466
    expected_entities = 14541
    expected_relations = 237

    print("Expected vs Actual:")
    print(f"  Train: {expected_train} vs {len(train)} {'✓' if abs(len(train) - expected_train) < 100 else '✗'}")
    print(f"  Valid: {expected_valid} vs {len(valid)} {'✓' if abs(len(valid) - expected_valid) < 100 else '✗'}")
    print(f"  Test: {expected_test} vs {len(test)} {'✓' if abs(len(test) - expected_test) < 100 else '✗'}")

    nentity = len(ent2id) if ent2id else max_ent + 1
    nrelation = len(rel2id) if rel2id else max_rel + 1
    print(f"  Entities: {expected_entities} vs {nentity} {'✓' if nentity == expected_entities else '✗'}")
    print(f"  Relations: {expected_relations} vs {nrelation} {'✓' if nrelation == expected_relations else '✗'}")
    print()

    if nentity == expected_entities and nrelation == expected_relations:
        print("✅ Data loading appears correct!")
    else:
        print("⚠️  Data counts don't match expected values - investigate further")

    print()
    return train, valid, test, ent2id, rel2id


if __name__ == "__main__":
    test_hf_loading()
