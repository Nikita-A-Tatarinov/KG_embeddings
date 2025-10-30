"""Quick GPU verification test for KG embeddings.

This test verifies that:
1. GPU is available and detected
2. Models can be moved to GPU
3. Training works on GPU
4. Evaluation works on GPU
5. Data loading works with GPU

Run this before starting large-scale GPU training.
"""
import torch
import sys


def check_gpu():
    """Check GPU availability and properties."""
    print("=" * 60)
    print("GPU Verification Test")
    print("=" * 60)
    print()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   PyTorch version:", torch.__version__)
        print("   CUDA compiled:", torch.version.cuda)
        print()
        print("   This means training will fall back to CPU.")
        print("   To use GPU, ensure:")
        print("   1. NVIDIA GPU is installed")
        print("   2. CUDA drivers are installed")
        print("   3. PyTorch with CUDA support is installed")
        return False

    print("✅ CUDA is available!")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    print()

    # Check GPU devices
    n_gpus = torch.cuda.device_count()
    print(f"✅ Found {n_gpus} GPU(s):")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"   GPU {i}: {props.name}")
        print(f"           Memory: {mem_gb:.2f} GB")
        print(f"           Compute Capability: {props.major}.{props.minor}")
    print()

    return True


def test_model_on_gpu():
    """Test that models work on GPU."""
    print("Testing model on GPU...")

    from models.registry import create_model

    device = torch.device("cuda:0")

    # Create a small model
    model = create_model("TransE", nentity=100,
                         nrelation=10, base_dim=50, gamma=12.0)
    model = model.to(device)

    # Create sample data
    sample = torch.randint(0, 100, (8, 3), dtype=torch.long, device=device)

    # Forward pass
    with torch.no_grad():
        output = model(sample, mode="single")

    assert output.device.type == "cuda", "Output should be on GPU"
    print("✅ Model forward pass on GPU works")
    print()


def test_training_step_on_gpu():
    """Test that training step works on GPU."""
    print("Testing training step on GPU...")

    from models.registry import create_model

    device = torch.device("cuda:0")

    # Create model and optimizer
    model = create_model("RotatE", nentity=100,
                         nrelation=10, base_dim=50, gamma=12.0)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create sample batch
    pos = torch.randint(0, 100, (16, 3), dtype=torch.long, device=device)
    neg = torch.randint(0, 100, (16, 32), dtype=torch.long, device=device)

    # Training step
    optimizer.zero_grad()

    # Negative scores
    neg_scores = model((pos, neg), mode="tail-batch")
    # Positive scores
    pos_scores = model(pos, mode="single")

    # Simple loss
    pos_loss = -torch.nn.functional.logsigmoid(pos_scores).mean()
    neg_loss = -torch.nn.functional.logsigmoid(-neg_scores).mean()
    loss = (pos_loss + neg_loss) / 2.0

    # Backward
    loss.backward()
    optimizer.step()

    print(f"✅ Training step works (loss={loss.item():.4f})")
    print()


def test_wrappers_on_gpu():
    """Test that MED, RSCF, MI wrappers work on GPU."""
    print("Testing wrappers on GPU...")

    from models.registry import create_model
    from models.rscf_wrapper import attach_rscf
    from models.mi_wrapper import attach_mi
    from med.med_wrapper import MEDTrainer

    device = torch.device("cuda:0")

    # Test RSCF
    model_rscf = create_model("TransE", nentity=50,
                              nrelation=5, base_dim=32, gamma=12.0)
    attach_rscf(model_rscf)
    model_rscf = model_rscf.to(device)
    sample = torch.randint(0, 50, (4, 3), dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model_rscf(sample, mode="single")
    print("  ✓ RSCF wrapper works on GPU")

    # Test MI
    model_mi = create_model("TransE", nentity=50,
                            nrelation=5, base_dim=32, gamma=12.0)
    attach_mi(model_mi, use_info_nce=True)
    model_mi = model_mi.to(device)
    pos = torch.randint(0, 50, (4, 3), dtype=torch.long, device=device)
    mi_loss = model_mi.compute_mi_loss(pos, neg_size=8)
    assert mi_loss.device.type == "cuda", "MI loss should be on GPU"
    print("  ✓ MI wrapper works on GPU")

    # Test MED
    base_model = create_model("TransE", nentity=50,
                              nrelation=5, base_dim=32, gamma=12.0)
    model_med = MEDTrainer(base_model, d_list=[
                           8, 16, 32], submodels_per_step=2)
    model_med = model_med.to(device)
    pos = torch.randint(0, 50, (4, 3), dtype=torch.long, device=device)
    neg = torch.randint(0, 50, (4, 8), dtype=torch.long, device=device)
    loss, stats = model_med(pos, neg, mode="tail-batch")
    assert loss.device.type == "cuda", "MED loss should be on GPU"
    print("  ✓ MED wrapper works on GPU")
    print()


def test_dataloader_gpu_transfer():
    """Test that data can be efficiently transferred to GPU."""
    print("Testing data loading and GPU transfer...")

    from dataset.kg_dataset import KGIndex, build_train_loaders, BidirectionalOneShotIterator

    device = torch.device("cuda:0")

    # Create small synthetic dataset
    train_ids = torch.tensor([
        [0, 0, 1], [1, 0, 2], [2, 1, 3],
        [0, 1, 3], [3, 0, 1], [1, 1, 0],
    ], dtype=torch.long)

    nentity, nrelation = 10, 5
    all_true = KGIndex(train_ids.tolist(), nentity, nrelation)

    # Build dataloaders
    dl_head, dl_tail = build_train_loaders(
        train_ids,
        nentity,
        nrelation,
        negative_size=16,
        batch_size=4,
        num_workers=0,  # Use 0 for testing
        use_filtered=True,
        all_true=all_true,
    )

    train_iter = BidirectionalOneShotIterator(dl_head, dl_tail)

    # Get a batch and transfer to GPU
    pos, neg, weight, mode = next(train_iter)
    pos = pos.to(device)
    neg = neg.to(device)
    weight = weight.to(device)

    assert pos.device.type == "cuda", "Positive samples should be on GPU"
    assert neg.device.type == "cuda", "Negative samples should be on GPU"

    print("✅ Data loading and GPU transfer works")
    print()


def main():
    # Check GPU availability
    if not check_gpu():
        print("=" * 60)
        print("GPU verification failed. Training will use CPU.")
        print("=" * 60)
        sys.exit(1)

    try:
        # Test model on GPU
        test_model_on_gpu()

        # Test training step
        test_training_step_on_gpu()

        # Test wrappers
        test_wrappers_on_gpu()

        # Test dataloader
        test_dataloader_gpu_transfer()

        print("=" * 60)
        print("✅ ALL GPU VERIFICATION TESTS PASSED!")
        print("=" * 60)
        print()
        print("Your system is ready for GPU training!")
        print("The trainer will automatically use GPU with 'device: auto' in configs.")

    except Exception as e:
        print("=" * 60)
        print("❌ GPU verification test failed!")
        print("=" * 60)
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
