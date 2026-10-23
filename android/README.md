# UniForge Studio for Android

A small signed Android app for the existing UniForge Studio service. It opens
Studio in a hardware-accelerated WebView, supports the native image/document
picker (multiple reference views or one face close-up), and saves GLB/receipt
downloads through Android DownloadManager. It does not ship or run MLX/CUDA models
on the phone. The Mac or remote GPU continues to run generation and face correction.

The native top bar switches between **Mac · Local MLX** on
`http://100.85.200.56:7861/`, **RTX 3060 · .54** through the independent queue on
port 7863, and **RTX 3090 · .51 · GPUQ** on port 7864. The `.51` route uses only
the existing port-12434 proxy and shared GPUQ. See [backend routing](../BACKENDS.md).
The selected engine survives relaunch; existing two-engine preferences migrate.
The current web UI,
including **Preserve face**, is served by that backend, so frontend improvements
do not require another APK. Keep the Mac service and Tailscale running.

## Build

The project pins AGP 8.13.2, Gradle 8.13 (official distribution SHA256 verified),
JDK 17, Android SDK/platform/build tools 36. It has no third-party runtime Android
library dependency. Set `JAVA_HOME` and `ANDROID_HOME` when they differ from the
Homebrew paths used on this Mac, then run from the repository root:

```sh
./tools/build_android.sh
```

This runs the origin-policy tests, release build and Android lint, verifies the
APK signature, and writes `outputs/android/UniForgeStudio-0.2.0.apk` plus its SHA256.
The private upgrade signing key is created once under
`~/.local/share/uniforge3d/android/`, with private directory/file permissions.
Keep it for upgrades. It is not stored in this repository. An existing or partial
identity is never replaced automatically. `UNIFORGE_SIGNING_PROPERTIES` can select
an existing signing properties file for the Gradle build; prepare that file before
calling Gradle directly. The standard build helper maintains the default identity.

## Install on the Moto G54

```sh
./tools/install_android_g54.sh
```

The helper uses the existing `g54-cua` SSH/ADB tunnel, checks the exact phone model,
installs without clearing app data, copies the APK to the phone's Downloads folder,
and launches `com.groxaxo.uniforge/.MainActivity`. It checks that the activity stays
in the foreground after Android settles, so an immediate startup crash fails the
install check. It never connects to a public ADB
listener. Unlock the phone normally if it requests a PIN.

## Behavior

- Version 0.1.1 installs the window content before configuring the system-bar
  controller, fixing the Moto G54 startup crash.
- React, local WebGL/model-viewer and DOM storage are enabled for Studio. No
  JavaScript-to-native bridge or arbitrary server input is exposed.
- In-app navigation is limited to the three private Studio origins. Cleartext HTTP
  is permitted only to that Mac's Tailscale address; certificate errors are cancelled.
- File selection uses Android's system document picker. Only user-selected image
  URIs are returned; broad photo/storage permissions are not requested.
- Downloads get unique timestamped names in the phone's public Downloads folder.
- Backup and device-transfer rules exclude the app's private data.
- Android 13+ uses the platform predictive-back callback; older Android releases
  retain the Activity's default Back behavior. The minimum Android version is 9.
- If the backend cannot be reached, the app shows a retry screen. A connected
  page can also show an unavailable GPU independently of the app connection.

Neither installation nor page loading proves new 3D inference completed. Device
acceptance should separately check the loaded UI, face checkbox, image picker,
3D preview and an actual download. Do not queue expensive inference just to test
APK installation.
