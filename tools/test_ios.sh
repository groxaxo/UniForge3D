#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
TASK_SIMULATOR=${UNIFORGE_SIMULATOR_ID:-$(cat outputs/ios/simulator-id 2>/dev/null || true)}
if [ -z "$TASK_SIMULATOR" ]; then
  echo 'Set UNIFORGE_SIMULATOR_ID from xcrun simctl list devices available. See ios/README.md.' >&2
  exit 1
fi
if command -v xcodegen >/dev/null 2>&1; then xcodegen generate --spec ios/project.yml; fi
xcrun simctl bootstatus "$TASK_SIMULATOR" -b
# The dedicated QA simulator receives one public demo image for native picker tests.
xcrun simctl addmedia "$TASK_SIMULATOR" assets/case01/001.png
TASK_RESULT="outputs/ios/Tests-$(date -u +%Y%m%dT%H%M%SZ).xcresult"
xcodebuild -project ios/UniForgeStudio.xcodeproj -scheme UniForgeStudio \
  -destination "platform=iOS Simulator,id=$TASK_SIMULATOR" \
  -derivedDataPath outputs/ios/DerivedData -resultBundlePath "$TASK_RESULT" \
  -parallel-testing-enabled NO -collect-test-diagnostics never CODE_SIGNING_ALLOWED=NO test
