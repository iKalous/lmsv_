#!/bin/bash
# Fix script for msadapter environment compatibility issues

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate msadapter

echo "Attempting to fix MSA environment..."

# The torchvision nms issue is related to msadapter not supporting all torch ops
# Try to use a compatible torchvision version
pip uninstall -y torchvision
echo "Removed torchvision to avoid compatibility issues"

# Check if we can import transformers without torchvision
echo "Testing imports..."
python3 -c "from transformers import PreTrainedModel; print('transformers OK')" 2>&1 || echo "transformers still has issues"

echo "Done"
