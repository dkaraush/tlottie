# tlottie Android example

This minimal app loads a bundled animation, a `.json` file selected with the
Android document picker, or a fixture from the device corpus at
`/storage/emulated/0/Android/data/com.example.tlottie/files/tgs_dump`. The fixture selector first chooses a pack and then
an animation.

The backend button switches between tlottie's CPU renderer and its `vulkan`
feature. Vulkan renders directly into a `SurfaceView` swapchain through
`ANativeWindow`; there is no GPU readback or Android `Bitmap` on that path. The
CPU comparison path writes an Android `Bitmap` displayed by a regular `View`.
If the surface does not expose `B8G8R8A8_UNORM`, the Vulkan path keeps tlottie's
BGRA render target offscreen and blits it into an `R8G8B8A8_UNORM` swapchain on
the GPU before presentation.

## Build and run

Requirements: Android SDK 35, NDK `27.2.12479018`, a JDK 17+, Rust, and an
attached arm64 Android device.

```sh
cd examples/android
export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
export JAVA_HOME="/path/to/jdk-17-or-newer"
./gradlew :app:assembleDebug
"$ANDROID_SDK_ROOT/platform-tools/adb" install -r app/build/outputs/apk/debug/app-debug.apk
"$ANDROID_SDK_ROOT/platform-tools/adb" shell am start -n com.example.tlottie/.MainActivity
```

Prepare the full fixture corpus once. The helper uploads it from
`~/Documents/fixtures-full` when missing and otherwise only verifies permissions:

```sh
./prepare-device-fixtures.sh RZCX41P9KTM
```

Override the host source with `HOST_FIXTURES=/another/path` or the device
destination with `DEVICE_FIXTURES=/another/path` when needed.

`build-rust.sh` installs the Rust `aarch64-linux-android` standard library on
first use and cross-compiles the JNI library. Override `ANDROID_NDK_HOME` or
`ANDROID_NDK_VERSION` when using a different NDK.
