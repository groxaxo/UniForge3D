# UniForge Studio for iOS

A SwiftUI iPhone/iPad app hosting the existing Studio generation interface in
WKWebView. Native controls switch between **Mac · Local MLX** and **RTX 3060 · .54**,
refresh the page, and show saved downloads with system sharing / Save to Files.
The web interface includes **Preserve face**, ordered reference images, job
progress, cancellation and the interactive textured/clay GLB preview.

Generation and optional face-photo correction run on the existing Mac/GPU
backends. No MLX/CUDA weights are bundled or executed on iOS. Keep Studio and
Tailscale running; the private queues are `100.85.200.56:7861` and `:7863`.

## Open, build and run

Open `UniForgeStudio.xcodeproj` in Xcode. The checked-in project is generated from
`project.yml`; XcodeGen is only needed when regenerating it. This project was
built with **Xcode 27.1 / Swift 6.4**, in **Swift 6 language mode**, with complete
strict-concurrency checking and Swift warnings treated as errors. Deployment is
**iOS 18.0+**, on iPhone and iPad. No third-party Swift packages are required.

From the repository root:

```sh
./tools/build_ios.sh simulator
xcrun simctl list devices available
# Select a dedicated compatible iPhone simulator, then boot it:
export UNIFORGE_SIMULATOR_ID=YOUR_SIMULATOR_UUID
xcrun simctl boot "$UNIFORGE_SIMULATOR_ID"
./tools/run_ios_simulator.sh
```

On the development Mac, `outputs/ios/simulator-id` records the dedicated
**UniForge Studio QA** iPhone 15 Pro simulator. The helper reuses that ID when the
environment override is absent. Other simulator sessions are not reset or reused.
The run helper opens Apple's Simulator viewer when it is registered. This Mac's
Xcode installation currently provides headless simulator tools without that
viewer; the helper still installs/launches the app and prints a screenshot command.

## Tests

```sh
./tools/test_ios.sh
```

Unit tests cover exact-origin restrictions, filename/path handling, unique file
names and benign-vs-real navigation errors. Live UI tests require the Mac Studio
service and its existing completed assets. They exercise native engine switching,
the Preserve face checkbox, a completed GLB download into native Downloads,
native photo selection using a public repository demo image, and the completed
Monkey King asset's textured/clay WebGL preview. They do not submit
a GPU generation job. Unit-only runs can use `xcodebuild ...
-only-testing:UniForgeStudioTests test` without the backend UI test prerequisite.

The test helper seeds only its explicitly selected QA simulator with that demo
image. `-collect-test-diagnostics never` avoids Xcode's 600-second post-failure
simulator-diagnostics collection; ordinary test results and screenshots remain in
the `.xcresult`. Failed-run bundles are retained separately under `outputs/ios`.

## Signed development build

```sh
./tools/build_ios.sh device
```

This uses an existing local Apple Development identity and matching profile, then
exports a **development IPA**, not an App Store/TestFlight submission. The current
development team is configured in `project.yml` and `ExportOptions.plist`; use
`UNIFORGE_DEVELOPMENT_TEAM` to override it in the build helper. Signing keys and
provisioning profiles remain in the existing Keychain/Xcode locations. They are
not copied into source control, and the helper does not enable automatic portal
updates or change Apple Account credentials.

The resulting IPA is `outputs/ios/UniForgeStudio.ipa`; timestamped archives and
exports are retained as well. Installation requires a connected, unlocked iPhone
registered in its development profile and normal Developer Mode/trust. A signed
archive is not evidence of completed physical-device installation.

## Implementation boundaries

- `StudioSession` is a final `@MainActor` owner of WKWebView and its delegates;
  UI state uses Observation. No `@unchecked Sendable`, `@preconcurrency`, forced
  unwraps or manual JavaScript-to-native bridge is required.
- Native Safari-style image selection is left to WebKit. Implementing the custom
  open-panel delegate would unnecessarily require iOS 18.4; omitting it supports
  the deployment target and the built-in picker.
- Navigation/download redirects are limited to the exact private host and ports.
  ATS permits insecure HTTP only for that numeric address (supported since iOS17),
  without global arbitrary-load exceptions. Local-network/photo/camera purpose
  strings describe the user-selected reference flow.
- Downloads use safe unique names, stage in Caches, and move into app
  Documents/Downloads only after WebKit reports completion and the file is
  nonempty. Incomplete downloads are not restored as saved assets. Completed files
  are excluded from backup and visible in Files under UniForge Studio.
- A successful download clears the progress state; WebKit's expected policy-handoff
  error is not shown as an offline failure. Actual connection errors show retry UI.
- GLB preview is the existing WebGL renderer. Native Quick Look GLB support is not
  assumed. Native download rows provide ShareLink for local export.

ATS reference: [Apple's exception-domain documentation](https://developer.apple.com/documentation/bundleresources/information-property-list/nsapptransportsecurity/nsexceptiondomains).
