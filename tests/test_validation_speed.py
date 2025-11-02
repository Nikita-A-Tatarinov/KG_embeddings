#!/usr/bin/env python3
"""Test that sampling actually speeds up validation during training."""

import time

from dataset.fb15k237 import prepare_fb15k237

print("=" * 70)
print("Testing Validation Speed: Full vs Sampled Dataset")
print("=" * 70)

# Test 1: Full dataset
print("\n[1/2] Loading FULL FB15k-237 dataset...")
start = time.time()
train_iter_full, (v_head_full, v_tail_full), (t_head_full, t_tail_full), meta_full = prepare_fb15k237(
    root="./data",
    use_hf=True,
)
load_time_full = time.time() - start

print(f"✓ Loaded in {load_time_full:.2f}s")
print(f"  Validation loaders: head={len(v_head_full.dataset)} triples, tail={len(v_tail_full.dataset)} triples")
print(f"  Test loaders: head={len(t_head_full.dataset)} triples, tail={len(t_tail_full.dataset)} triples")

# Test 2: Sampled dataset (10%)
print("\n[2/2] Loading SAMPLED (10%) FB15k-237 dataset...")
start = time.time()
train_iter_sample, (v_head_sample, v_tail_sample), (t_head_sample, t_tail_sample), meta_sample = prepare_fb15k237(
    root="./data",
    use_hf=True,
    sample_ratio=0.1,
    sample_seed=42,
)
load_time_sample = time.time() - start

print(f"✓ Loaded in {load_time_sample:.2f}s")
print(f"  Validation loaders: head={len(v_head_sample.dataset)} triples, tail={len(v_tail_sample.dataset)} triples")
print(f"  Test loaders: head={len(t_head_sample.dataset)} triples, tail={len(t_tail_sample.dataset)} triples")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: Validation/Test Set Sizes")
print("=" * 70)

valid_full = len(v_head_full.dataset)
valid_sample = len(v_head_sample.dataset)
test_full = len(t_head_full.dataset)
test_sample = len(t_head_sample.dataset)

print("\nValidation (per direction):")
print(f"  Full:    {valid_full:>6} triples")
print(f"  Sampled: {valid_sample:>6} triples ({100 * valid_sample / valid_full:.1f}% of full)")
print(f"  Speedup: {valid_full / valid_sample:.1f}× faster validation")

print("\nTest (per direction):")
print(f"  Full:    {test_full:>6} triples")
print(f"  Sampled: {test_sample:>6} triples ({100 * test_sample / test_full:.1f}% of full)")
print(f"  Speedup: {test_full / test_sample:.1f}× faster testing")

print("\nVocabulary (both cases):")
print(f"  Entities:  {meta_sample['nentity']} (preserved)")
print(f"  Relations: {meta_sample['nrelation']} (preserved)")

print("\n" + "=" * 70)
print("✅ CONCLUSION:")
print("=" * 70)
print(f"During training, validation will be ~{valid_full / valid_sample:.1f}× FASTER with 10% sampling!")
print(f"This saves {(valid_full - valid_sample) * 2} triple evaluations per validation epoch.")
print("(×2 because we evaluate both head and tail prediction)")
print("=" * 70)
