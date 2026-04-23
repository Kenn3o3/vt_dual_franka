#!/usr/bin/env bash
set -euo pipefail

# Environment setup for movement-primitive-diffusion training.
# Run from the vt_franka repo root.

ENV_NAME="${1:-mpd}"
MPD_DIR="robot_workspace/third_parties/movement-primitive-diffusion"

if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. Install miniconda/miniforge first."
    exit 1
fi

if [ ! -d "$MPD_DIR" ]; then
    echo "Error: $MPD_DIR not found. Run from vt_franka repo root."
    exit 1
fi

echo "Creating conda environment: $ENV_NAME (Python 3.10)"
conda create -n "$ENV_NAME" python=3.10 -y

echo "Installing MP_PyTorch dependency..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

pip install -e "$MPD_DIR/dependencies/MP_PyTorch"

echo "Installing movement-primitive-diffusion..."
pip install -e "$MPD_DIR"

echo "Installing extra dependencies..."
pip install wandb opencv-python

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
echo "Test with: python -c 'import movement_primitive_diffusion; print(\"OK\")'"
