#!/bin/bash
set -e

echo "======================================================================"
echo "   Abu Align & Train - PFGS360 Setup Helper (Linux / RunPod)"
echo "   (CVPR 2026: Pose-Free Omnidirectional Gaussian Splatting for 360 Videos)"
echo "======================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Activate venv if present
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "$HOME/miniconda3/envs/abu_align" ]; then
    source "$HOME/miniconda3/bin/activate" abu_align
elif [ -d "$HOME/miniconda3/envs/vggt" ]; then
    source "$HOME/miniconda3/bin/activate" vggt
fi

# 2. Install dependencies
pip install ninja jaxtyping typeguard rich open3d timm scikit-image || true

# 3. Clone PFGS360
if [ ! -d "PFGS360" ]; then
    echo "[*] Cloning PFGS360 repository..."
    git clone https://github.com/zcq15/PFGS360.git
fi

# 4. Install PFGS360
if [ -d "PFGS360" ]; then
    cd PFGS360
    if [ -f "setup.py" ]; then
        echo "[*] Installing PFGS360 in editable mode..."
        pip install -e . --no-build-isolation || true
    fi
    cd ..
fi

mkdir -p checkpoints/pfgs360

echo ""
echo "======================================================================"
echo "   PFGS360 Setup Completed Successfully!"
echo "   Ready for Pose-Free 360° GS training and OpenSfM / 360DVO handoff."
echo "======================================================================"
echo ""
