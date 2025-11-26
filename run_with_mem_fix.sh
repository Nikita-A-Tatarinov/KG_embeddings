#!/bin/bash
# Run training with memory fragmentation fixes enabled

# Enable expandable segments to prevent fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Set max split size to further reduce fragmentation
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

echo "================================================================"
echo "Memory Fragmentation Fixes Enabled"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "================================================================"
echo ""

# Run the training command passed as arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <training command>"
    echo "Example: $0 python train.py --config configs/final/wn18rr/rotate_med_rscf.yaml"
    exit 1
fi

# Execute the training command
exec "$@"
