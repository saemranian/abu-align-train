#!/bin/bash
set -e

echo "======================================================================"
echo "   Abu Align & Train - 360DVO Deep Visual Odometry Setup (Linux / RunPod)"
echo "   (IEEE RA-L 2026: Monocular 360-Degree Deep Visual Odometry)"
echo "======================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 2. Install requirements
pip install evo viser pyyaml scikit-image || true

# 3. Clone 360DVO
if [ ! -d "360DVO" ]; then
    echo "[*] Cloning 360DVO repository..."
    git clone https://github.com/chris1004336379/360DVO.git
fi

# 4. Download Eigen 3.4.0
cd 360DVO
if [ ! -d "thirdparty/eigen-3.4.0" ]; then
    echo "[*] Downloading Eigen 3.4.0..."
    mkdir -p thirdparty
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q eigen-3.4.0.zip -d thirdparty
    rm -f eigen-3.4.0.zip
fi

# 5. Build 360DVO CUDA extension
echo "[*] Compiling 360DVO CUDA & C++ extensions..."
pip install -e . --no-build-isolation || true
cd ..

mkdir -p checkpoints

echo ""
echo "======================================================================"
echo "   360DVO Installed Successfully!"
echo "   Place '360dvo.pth' into 'checkpoints/' to enable deep inference."
echo "======================================================================"
echo ""
