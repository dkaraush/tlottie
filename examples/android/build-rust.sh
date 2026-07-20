#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -z "${ANDROID_SDK_ROOT:-}" ]]; then
    if [[ -n "${ANDROID_HOME:-}" ]]; then
        ANDROID_SDK_ROOT="$ANDROID_HOME"
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        ANDROID_SDK_ROOT="$HOME/Library/Android/sdk"
    else
        ANDROID_SDK_ROOT="$HOME/Android/Sdk"
    fi
fi
NDK_VERSION="${ANDROID_NDK_VERSION:-27.2.12479018}"
NDK="${ANDROID_NDK_HOME:-$ANDROID_SDK_ROOT/ndk/$NDK_VERSION}"
case "$(uname -s)" in
    Darwin) HOST_TAG="darwin-x86_64" ;;
    Linux) HOST_TAG="linux-x86_64" ;;
    *) echo "unsupported Android NDK host: $(uname -s)" >&2; exit 1 ;;
esac
HOST_TAG="${ANDROID_NDK_HOST_TAG:-$HOST_TAG}"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin"
TARGET="aarch64-linux-android"
API=28

if [[ ! -x "$TOOLCHAIN/${TARGET}${API}-clang" ]]; then
    echo "Android NDK toolchain not found: $TOOLCHAIN" >&2
    exit 1
fi

if command -v rustup >/dev/null 2>&1; then
    RUST_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-$(rustup show active-toolchain | awk '{print $1}')}"
    if ! rustup target list --installed --toolchain "$RUST_TOOLCHAIN" | grep -qx "$TARGET"; then
        rustup target add --toolchain "$RUST_TOOLCHAIN" "$TARGET"
    fi
    CARGO=("$(rustup which --toolchain "$RUST_TOOLCHAIN" cargo)")
    export RUSTC="$(rustup which --toolchain "$RUST_TOOLCHAIN" rustc)"
else
    CARGO=(cargo)
fi

export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$TOOLCHAIN/${TARGET}${API}-clang"
export CC_aarch64_linux_android="$TOOLCHAIN/${TARGET}${API}-clang"
export AR_aarch64_linux_android="$TOOLCHAIN/llvm-ar"

"${CARGO[@]}" build \
    --manifest-path "$ROOT/examples/android/native/Cargo.toml" \
    --target "$TARGET" \
    --release

DEST="$ROOT/examples/android/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$DEST"
cp "$ROOT/examples/android/native/target/$TARGET/release/libtlottie_android.so" "$DEST/"
