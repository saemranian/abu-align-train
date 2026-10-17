#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$ROOT_DIR/installer/dist"
BUILD_DIR="$ROOT_DIR/installer/build/AppDir"

VERSION="$(tr -d '[:space:]vV' < "$ROOT_DIR/VERSION.txt" || echo "0.1.5.5")"

echo "[*] Building Linux AppImage for Abu Align & Train v${VERSION}..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/usr/share/abu-align" "$DIST_DIR"

# Copy core application files
cp "$ROOT_DIR"/*.py "$BUILD_DIR/usr/share/abu-align/" 2>/dev/null || true
cp "$ROOT_DIR"/*.sh "$BUILD_DIR/usr/share/abu-align/" 2>/dev/null || true
cp "$ROOT_DIR"/*.txt "$BUILD_DIR/usr/share/abu-align/" 2>/dev/null || true
cp "$ROOT_DIR"/*.md "$BUILD_DIR/usr/share/abu-align/" 2>/dev/null || true
cp "$ROOT_DIR"/*.png "$BUILD_DIR/usr/share/abu-align/" 2>/dev/null || true
cp "$ROOT_DIR"/*.ico "$BUILD_DIR/usr/share/abu-align/" 2>/dev/null || true
cp "$ROOT_DIR/pyproject.toml" "$BUILD_DIR/usr/share/abu-align/" 2>/dev/null || true

for subdir in vggt training docs; do
    if [ -d "$ROOT_DIR/$subdir" ]; then
        cp -r "$ROOT_DIR/$subdir" "$BUILD_DIR/usr/share/abu-align/"
    fi
done

# Create AppRun entry point
cat << 'EOF' > "$BUILD_DIR/AppRun"
#!/usr/bin/env bash
set -e

HERE="$(dirname "$(readlink -f "${0}")")"
APP_SRC="$HERE/usr/share/abu-align"
USER_DATA="${XDG_DATA_HOME:-$HOME/.local/share}/AbuAlign"
APP_HOME="$USER_DATA/app"
VENV_DIR="$USER_DATA/env"

mkdir -p "$APP_HOME"
cp -ru "$APP_SRC/"* "$APP_HOME/" 2>/dev/null || cp -r "$APP_SRC/"* "$APP_HOME/"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "======================================================================"
    echo "   Abu Align & Train - First-Time Linux Runtime Provisioning"
    echo "======================================================================"
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || true
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    if command -v uv >/dev/null 2>&1; then
        uv venv "$VENV_DIR" --python 3.11 || uv venv "$VENV_DIR"
        uv pip install --python "$VENV_DIR/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
        uv pip install --python "$VENV_DIR/bin/python" torch torchvision
        uv pip install --python "$VENV_DIR/bin/python" -r "$APP_HOME/requirements.txt"
    else
        python3 -m venv "$VENV_DIR"
        "$VENV_DIR/bin/pip" install --upgrade pip
        "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
        "$VENV_DIR/bin/pip" install torch torchvision
        "$VENV_DIR/bin/pip" install -r "$APP_HOME/requirements.txt"
    fi
fi

cd "$APP_HOME"
if [ "${1:-}" = "--cli" ]; then
    shift
    exec "$VENV_DIR/bin/python" "$APP_HOME/align_cli.py" "$@"
else
    exec "$VENV_DIR/bin/python" "$APP_HOME/abu_align_gui.py" "$@"
fi
EOF
chmod +x "$BUILD_DIR/AppRun"

# Desktop Entry
cat << EOF > "$BUILD_DIR/abu-align.desktop"
[Desktop Entry]
Type=Application
Name=Abu Align & Train
Comment=Neural Photogrammetry & 3DGS Training Studio (v${VERSION})
Exec=AppRun
Icon=abu_icon
Categories=Graphics;3DGraphics;Science;
Terminal=false
EOF

if [ -f "$ROOT_DIR/abu_icon.png" ]; then
    cp "$ROOT_DIR/abu_icon.png" "$BUILD_DIR/abu_icon.png"
fi

# Download appimagetool if not present
APPIMAGETOOL="/tmp/appimagetool-x86_64.AppImage"
if [ ! -x "$APPIMAGETOOL" ]; then
    curl -fsSL -o "$APPIMAGETOOL" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
    chmod +x "$APPIMAGETOOL"
fi

ARCH=x86_64 "$APPIMAGETOOL" --appimage-extract-and-run "$BUILD_DIR" "$DIST_DIR/Abu_Align_and_Train_Linux_x86_64.AppImage"
chmod +x "$DIST_DIR/Abu_Align_and_Train_Linux_x86_64.AppImage"
echo "[SUCCESS] Built: $DIST_DIR/Abu_Align_and_Train_Linux_x86_64.AppImage"
