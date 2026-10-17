#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="$ROOT_DIR/installer/build/macos/Abu Align and Train.app"
test -x "$ROOT_DIR/installer/dist/AbuRuntime/AbuRuntime"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp -R "$ROOT_DIR/installer/dist/AbuRuntime/." "$APP/Contents/MacOS/"
python - "$ROOT_DIR/VERSION.txt" "$APP/Contents/Info.plist" <<'PY'
import plistlib,sys
from pathlib import Path
v=Path(sys.argv[1]).read_text().strip().lstrip('vV')
plist={'CFBundleExecutable':'AbuRuntime','CFBundleIdentifier':'io.github.saemranian.abu-align-train',
       'CFBundleName':'Abu Align CPU Core Candidate','CFBundleVersion':v,
       'CFBundleShortVersionString':'.'.join(v.split('.')[:3]),'CFBundlePackageType':'APPL',
       'NSHighResolutionCapable':True}
with open(sys.argv[2],'wb') as f:plistlib.dump(plist,f)
PY
if [[ -n "${ABU_MAC_SIGN_IDENTITY:-}" ]]; then
  codesign --force --deep --options runtime --timestamp --sign "$ABU_MAC_SIGN_IDENTITY" "$APP"
fi
ARCH="$(uname -m)"
OUT="$ROOT_DIR/installer/dist/Abu_Align_and_Train_macOS_${ARCH}.dmg"
hdiutil create -volname 'Abu CPU Core Candidate' -srcfolder "$ROOT_DIR/installer/build/macos" -ov -format UDZO "$OUT"
if [[ -n "${ABU_NOTARY_PROFILE:-}" ]]; then
  xcrun notarytool submit "$OUT" --keychain-profile "$ABU_NOTARY_PROFILE" --wait
  xcrun stapler staple "$OUT"
fi
echo "Candidate only until architecture, signing and clean-machine checks pass: $OUT"
