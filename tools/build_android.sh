#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TASK_JAVA_HOME=${JAVA_HOME:-/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home}
TASK_ANDROID_HOME=${ANDROID_HOME:-/opt/homebrew/share/android-commandlinetools}
test -x "$TASK_JAVA_HOME/bin/java" || { echo 'Set JAVA_HOME to a JDK 17 installation.' >&2; exit 1; }
test -f "$TASK_ANDROID_HOME/platforms/android-36/android.jar" || { echo 'Install Android SDK platform 36 and set ANDROID_HOME.' >&2; exit 1; }
export JAVA_HOME="$TASK_JAVA_HOME" ANDROID_HOME="$TASK_ANDROID_HOME"
cd "$TASK_ROOT"
python3 android/scripts/prepare_signing.py --keytool "$JAVA_HOME/bin/keytool"
./android/gradlew -p android testReleaseUnitTest assembleRelease lintRelease --console=plain
"$ANDROID_HOME/build-tools/36.0.0/apksigner" verify --verbose --print-certs android/app/build/outputs/apk/release/app-release.apk
python3 - <<'PY'
import hashlib,json,shutil
from pathlib import Path
source=Path('android/app/build/outputs/apk/release')
info=json.loads((source/'output-metadata.json').read_text())['elements'][0]
output=Path('outputs/android')/f"UniForgeStudio-{info['versionName']}.apk"
output.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(source/info['outputFile'],output)
(output.with_suffix('.apk.sha256')).write_text(hashlib.sha256(output.read_bytes()).hexdigest()+'  '+output.name+'\n')
print(f'Signed APK: {output.resolve()}')
PY
