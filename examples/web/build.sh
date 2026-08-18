#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
target=wasm32-unknown-unknown
profile=release
rustup_toolchain=

if command -v rustup >/dev/null 2>&1; then
  rustup_toolchain=$(rustup show active-toolchain | awk 'NR == 1 { print $1 }')
  if ! rustup target list --toolchain "$rustup_toolchain" --installed | grep -qx "$target"; then
    echo "error: Rust target '$target' is not installed for '$rustup_toolchain'" >&2
    echo "install it with: rustup target add --toolchain $rustup_toolchain $target" >&2
    exit 1
  fi
  RUSTC=$(rustup which --toolchain "$rustup_toolchain" rustc)
  export RUSTC
else
  target_libdir=$(rustc --print target-libdir --target "$target" 2>/dev/null || true)
  if [ ! -d "$target_libdir" ] || ! ls "$target_libdir"/libstd-*.rlib >/dev/null 2>&1; then
    echo "error: the active Rust toolchain does not include target '$target'" >&2
    echo "install the target with your Rust distribution and try again" >&2
    exit 1
  fi
fi

export CARGO_TARGET_DIR="$repo_root/target"
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-feature=+simd128"

if [ -n "$rustup_toolchain" ]; then
  rustup run "$rustup_toolchain" cargo build \
    --manifest-path "$repo_root/Cargo.toml" \
    --target "$target" \
    --profile "$profile" \
    --no-default-features \
    --features wasm,std
else
  cargo build \
    --manifest-path "$repo_root/Cargo.toml" \
    --target "$target" \
    --profile "$profile" \
    --no-default-features \
    --features wasm,std
fi

cp "$CARGO_TARGET_DIR/$target/$profile/tlottie.wasm" "$script_dir/tlottie.wasm"
echo "Built $script_dir/tlottie.wasm"
