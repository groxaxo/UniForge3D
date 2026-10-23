#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TASK_MODE=${1:-simulator}
cd "$TASK_ROOT"
mkdir -p outputs/ios
if command -v xcodegen >/dev/null 2>&1; then xcodegen generate --spec ios/project.yml; fi
case "$TASK_MODE" in
  simulator)
    xcodebuild -project ios/UniForgeStudio.xcodeproj -scheme UniForgeStudio \
      -configuration Debug -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' \
      -derivedDataPath outputs/ios/DerivedData CODE_SIGNING_ALLOWED=NO build
    echo "Simulator app: $TASK_ROOT/outputs/ios/DerivedData/Build/Products/Debug-iphonesimulator/UniForgeStudio.app"
    ;;
  device)
    TASK_TEAM=${UNIFORGE_DEVELOPMENT_TEAM:-GV8YC56858}
    TASK_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    TASK_ARCHIVE="outputs/ios/UniForgeStudio-$TASK_STAMP.xcarchive"
    TASK_EXPORT="outputs/ios/Exports/$TASK_STAMP"
    xcodebuild -project ios/UniForgeStudio.xcodeproj -scheme UniForgeStudio \
      -configuration Release -destination 'generic/platform=iOS' \
      -derivedDataPath outputs/ios/DeviceDerivedData -archivePath "$TASK_ARCHIVE" \
      CODE_SIGN_STYLE=Automatic "DEVELOPMENT_TEAM=$TASK_TEAM" archive
    python3 - "$TASK_TEAM" <<'PY'
from pathlib import Path
import plistlib,sys
options=plistlib.loads(Path('ios/ExportOptions.plist').read_bytes())
options['teamID']=sys.argv[1]
Path('outputs/ios/ExportOptions-current.plist').write_bytes(plistlib.dumps(options))
PY
    xcodebuild -exportArchive -archivePath "$TASK_ARCHIVE" \
      -exportOptionsPlist outputs/ios/ExportOptions-current.plist -exportPath "$TASK_EXPORT"
    cp "$TASK_EXPORT/UniForgeStudio.ipa" outputs/ios/UniForgeStudio.ipa
    echo "Signed development IPA: $TASK_ROOT/outputs/ios/UniForgeStudio.ipa"
    ;;
  *) echo 'Usage: build_ios.sh [simulator|device]' >&2; exit 2 ;;
esac
