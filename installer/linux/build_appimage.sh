#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APPDIR="$ROOT_DIR/installer/build/AppDir"
test -x "$ROOT_DIR/installer/dist/AbuRuntime/AbuRuntime"
mkdir -p "$APPDIR/usr/lib/abu-align"
cp -R "$ROOT_DIR/installer/dist/AbuRuntime/." "$APPDIR/usr/lib/abu-align/"
cat > "$APPDIR/AppRun" <<'SH'
#!/usr/bin/env sh
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$HERE/usr/lib/abu-align/AbuRuntime" "$@"
SH
chmod +x "$APPDIR/AppRun"
cat > "$APPDIR/abu-align.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=Abu Align CPU Core Candidate
Exec=AppRun
Icon=abu_icon
Categories=Graphics;Science;
Terminal=false
DESKTOP
cp "$ROOT_DIR/abu_icon.png" "$APPDIR/abu_icon.png"
: "${ABU_APPIMAGETOOL:?Set ABU_APPIMAGETOOL to a verified appimagetool executable}"
ARCH=x86_64 "$ABU_APPIMAGETOOL" --appimage-extract-and-run -n "$APPDIR" "$ROOT_DIR/installer/dist/Abu_Align_and_Train_Linux_x86_64.AppImage"
