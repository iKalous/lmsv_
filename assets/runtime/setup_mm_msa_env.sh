#!/bin/bash
# Setup script for mm-msa environment
# This installs required packages in the msadapter conda environment

set -e

echo "Setting up mm-msa environment..."

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate msadapter

# Install required packages that might be missing
pip install beartype -q

# Check if peft is installed and has issues with torch.fft
python3 << 'PYEOF'
try:
    import peft
    print(f"peft version: {peft.__version__}")
    # Try importing the C3A tuner to check for torch.fft issue
    try:
        from peft.tuners import C3A
        print("C3A tuner available")
    except ImportError as e:
        print(f"C3A import error: {e}")
except ImportError:
    print("peft not installed")
PYEOF

echo "mm-msa environment setup complete"
