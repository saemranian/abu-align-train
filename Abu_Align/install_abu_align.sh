#!/bin/bash
set -e

echo "======================================================================"
echo "   Abu Align & Train - Automated One-Click Installer (Linux / RunPod)"
echo "======================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Install System Dependencies
echo "[*] Updating system packages and installing system libraries..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y || apt-get update -y
    sudo apt-get install -y build-essential cmake git curl libceres-dev libgoogle-glog-dev libgl1-mesa-glx libglib2.0-0 python3-dev python3-pip python3-venv || \
    apt-get install -y build-essential cmake git curl libceres-dev libgoogle-glog-dev libgl1-mesa-glx libglib2.0-0 python3-dev python3-pip python3-venv
fi

# 2. Setup Fast Python Virtual Environment (prefer uv if available)
VENV_PATH="$SCRIPT_DIR/venv"

if ! command -v uv >/dev/null 2>&1; then
    echo "[*] Installing uv for lightning-fast environment resolution..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || true
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if command -v uv >/dev/null 2>&1; then
    echo "[*] Creating Python 3.11 environment with uv..."
    uv venv "$VENV_PATH" --python 3.11 || uv venv "$VENV_PATH"
    PIP_CMD="uv pip"
else
    echo "[*] Creating standard Python venv at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    PIP_CMD="$VENV_PATH/bin/pip"
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# 3. Install PyTorch with CUDA
echo "[*] Installing PyTorch with CUDA 12.1..."
if command -v uv >/dev/null 2>&1; then
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# 4. Install Abu Align requirements
echo "[*] Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv pip install -r requirements.txt
    else
        pip install -r requirements.txt
    fi
fi

# 5. Install editable package dependencies
if [ -f "setup.py" ]; then
    pip install -e . --no-deps
fi
if [ -d "../vggt-omega" ] && [ -f "../vggt-omega/setup.py" ]; then
    pip install -e "../vggt-omega" --no-deps
fi

# 6. Install Mapillary OpenSfM (360° Spherical Photogrammetry)
echo "[*] Installing Mapillary OpenSfM..."
pip install pyyaml joblib scikit-image || true
if ! command -v opensfm >/dev/null 2>&1; then
    if [ ! -d "OpenSfM" ]; then
        git clone --recursive https://github.com/mapillary/OpenSfM.git || true
    fi
    if [ -d "OpenSfM" ]; then
        cd OpenSfM
        python setup.py build || true
        pip install -e . || true
        cd ..
    fi
fi

# 7. Create convenient launcher script
cat << 'EOF' > run_abu_align.sh
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
python "$SCRIPT_DIR/align_cli.py" "$@"
EOF
chmod +x run_abu_align.sh

echo ""
echo "======================================================================"
echo "   Abu Align & Train Installed Successfully!"
echo "   Run alignment via:"
echo "     ./run_abu_align.sh /path/to/dataset --engine vggt_omega"
echo "======================================================================"
echo ""
