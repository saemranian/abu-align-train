#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$ROOT_DIR/installer/dist"
STAGING_DIR="$ROOT_DIR/installer/build/dmg_staging"
APP_BUNDLE="$STAGING_DIR/Abu Align & Train.app"

VERSION="$(tr -d '[:space:]vV' < "$ROOT_DIR/VERSION.txt" || echo "0.1.5.5")"

echo "[*] Building macOS Application Bundle & DMG for Abu Align & Train v${VERSION}..."
rm -rf "$STAGING_DIR"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources/app" "$DIST_DIR"

# Copy application source files
cp "$ROOT_DIR"/*.py "$APP_BUNDLE/Contents/Resources/app/" 2>/dev/null || true
cp "$ROOT_DIR"/*.sh "$APP_BUNDLE/Contents/Resources/app/" 2>/dev/null || true
cp "$ROOT_DIR"/*.txt "$APP_BUNDLE/Contents/Resources/app/" 2>/dev/null || true
cp "$ROOT_DIR"/*.md "$APP_BUNDLE/Contents/Resources/app/" 2>/dev/null || true
cp "$ROOT_DIR"/*.png "$APP_BUNDLE/Contents/Resources/app/" 2>/dev/null || true
cp "$ROOT_DIR"/*.ico "$APP_BUNDLE/Contents/Resources/app/" 2>/dev/null || true
cp "$ROOT_DIR/pyproject.toml" "$APP_BUNDLE/Contents/Resources/app/" 2>/dev/null || true

for subdir in vggt training docs; do
    if [ -d "$ROOT_DIR/$subdir" ]; then
        cp -r "$ROOT_DIR/$subdir" "$APP_BUNDLE/Contents/Resources/app/"
    fi
done

# Create macOS Info.plist
cat << EOF > "$APP_BUNDLE/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>AbuAlign</string>
    <key>CFBundleIdentifier</key>
    <string>io.github.saemranian.abu-align-train</string>
    <key>CFBundleName</key>
    <string>Abu Align &amp; Train</string>
    <key>CFBundleDisplayName</key>
    <string>Abu Align &amp; Train</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Create macOS executable bootstrapper
cat << 'EOF' > "$APP_BUNDLE/Contents/MacOS/AbuAlign"
#!/usr/bin/env bash
set -e

CONTENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_SRC="$CONTENTS_DIR/Resources/app"
USER_SUPPORT="$HOME/Library/Application Support/AbuAlign"
APP_HOME="$USER_SUPPORT/app"
VENV_DIR="$USER_SUPPORT/env"

mkdir -p "$APP_HOME"
cp -R "$APP_SRC/"* "$APP_HOME/"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    osascript -e 'display notification "Provisioning Python & Apple Silicon MPS runtime (first launch only)..." with title "Abu Align & Train"' || true
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || true
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
    fi
    if command -v uv >/dev/null 2>&1; then
        uv venv "$VENV_DIR" --python 3.11 || uv venv "$VENV_DIR"
        uv pip install --python "$VENV_DIR/bin/python" torch torchvision
        uv pip install --python "$VENV_DIR/bin/python" -r "$APP_HOME/requirements.txt"
    else
        python3 -m venv "$VENV_DIR"
        "$VENV_DIR/bin/pip" install --upgrade pip
        "$VENV_DIR/bin/pip" install torch torchvision
        "$VENV_DIR/bin/pip" install -r "$APP_HOME/requirements.txt"
    fi
fi

cd "$APP_HOME"
exec "$VENV_DIR/bin/python" "$APP_HOME/abu_align_gui.py" "$@"
EOF
chmod +x "$APP_BUNDLE/Contents/MacOS/AbuAlign"

# Create Applications symlink for drag-and-drop installation
ln -s /Applications "$STAGING_DIR/Applications"

DMG_OUT="$DIST_DIR/Abu_Align_and_Train_macOS_Universal.dmg"
rm -f "$DMG_OUT"
hdiutil create -volname "Abu Align & Train v${VERSION}" -srcfolder "$STAGING_DIR" -ov -format UDZO "$DMG_OUT"
echo "[SUCCESS] Built: $DMG_OUT"
