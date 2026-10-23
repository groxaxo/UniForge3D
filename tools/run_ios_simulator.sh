#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
TASK_SIMULATOR=${UNIFORGE_SIMULATOR_ID:-$(cat outputs/ios/simulator-id 2>/dev/null || true)}
if [ -z "$TASK_SIMULATOR" ]; then
  echo 'Set UNIFORGE_SIMULATOR_ID to the intended iPhone simulator. See ios/README.md.' >&2
  exit 1
fi
xcrun simctl bootstatus "$TASK_SIMULATOR" -b
xcrun simctl install "$TASK_SIMULATOR" outputs/ios/DerivedData/Build/Products/Debug-iphonesimulator/UniForgeStudio.app
xcrun simctl launch "$TASK_SIMULATOR" com.groxaxo.uniforge.ios
if open -Ra Simulator >/dev/null 2>&1; then
  open -a Simulator --args -CurrentDeviceUDID "$TASK_SIMULATOR"
else
  echo "UniForge Studio is running in simulator $TASK_SIMULATOR."
  echo 'This Xcode installation has no registered Simulator viewer.'
  echo "Capture it with: xcrun simctl io $TASK_SIMULATOR screenshot outputs/ios/Studio-iPhone.png"
fi
