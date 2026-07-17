#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Android/Sdk}}"
NDK_VERSION="${ANDROID_NDK_VERSION:-27.2.12479018}"
NDK="${ANDROID_NDK_HOME:-$ANDROID_SDK_ROOT/ndk/$NDK_VERSION}"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
TARGET="aarch64-linux-android"
API=28

if ! rustup target list --installed | grep -qx "$TARGET"; then
    rustup target add "$TARGET"
fi

export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$TOOLCHAIN/${TARGET}${API}-clang"
export CC_aarch64_linux_android="$TOOLCHAIN/${TARGET}${API}-clang"
export AR_aarch64_linux_android="$TOOLCHAIN/llvm-ar"

cargo build \
    --manifest-path "$ROOT/examples/android/native/Cargo.toml" \
    --target "$TARGET" \
    --release

DEST="$ROOT/examples/android/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$DEST"
cp "$ROOT/examples/android/native/target/$TARGET/release/libtlottie_android.so" "$DEST/"
