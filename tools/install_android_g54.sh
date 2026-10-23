#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TASK_APK=${1:-"$TASK_ROOT/outputs/android/UniForgeStudio-0.1.1.apk"}
TASK_SERIAL=127.0.0.1:15555
test -f "$TASK_APK" || { echo 'Build the signed APK first with tools/build_android.sh.' >&2; exit 1; }
"$HOME/.local/bin/g54-cua" start
test "$(adb -s "$TASK_SERIAL" shell getprop ro.product.model | tr -d '\r')" = 'moto g54 5G' || { echo 'Refusing to install on an unexpected device.' >&2; exit 1; }
adb -s "$TASK_SERIAL" install -r "$TASK_APK"
adb -s "$TASK_SERIAL" push "$TASK_APK" "/sdcard/Download/$(basename "$TASK_APK")"
adb -s "$TASK_SERIAL" shell am start -W -n com.groxaxo.uniforge/.MainActivity
# Let Android settle before accepting a launch; am start can report success even
# when Activity.onCreate crashes immediately afterwards.
adb -s "$TASK_SERIAL" shell uiautomator dump /sdcard/uniforge-install-window.xml >/dev/null
TASK_FOCUS=$(adb -s "$TASK_SERIAL" shell dumpsys window | sed -n '/mCurrentFocus=/p')
case "$TASK_FOCUS" in
  *com.groxaxo.uniforge/com.groxaxo.uniforge.MainActivity*) ;;
  *) echo 'UniForge Studio did not remain open. Check its device crash log.' >&2; exit 1 ;;
esac
echo 'Installed and launched UniForge Studio. Confirm its page on the phone.'
