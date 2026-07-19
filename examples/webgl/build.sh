#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
target=wasm32-unknown-unknown
profile=release-size

local_wasm_bindgen="$repo_root/target/tools/bin/wasm-bindgen"
if command -v wasm-bindgen >/dev/null 2>&1; then
  wasm_bindgen=wasm-bindgen
elif [ -x "$local_wasm_bindgen" ]; then
  wasm_bindgen="$local_wasm_bindgen"
else
  echo "error: wasm-bindgen CLI 0.2.126 is required" >&2
  echo "install it with: cargo install --root target/tools wasm-bindgen-cli --version 0.2.126 --locked" >&2
  exit 1
fi

export CARGO_TARGET_DIR="$repo_root/target"
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-feature=+simd128"

cargo build \
  --manifest-path "$repo_root/Cargo.toml" \
  --target "$target" \
  --profile "$profile" \
  --no-default-features \
  --features webgl

rm -rf "$script_dir/pkg"
"$wasm_bindgen" \
  --target web \
  --out-dir "$script_dir/pkg" \
  --out-name tlottie \
  "$CARGO_TARGET_DIR/$target/$profile/tlottie.wasm"

echo "Built $script_dir/pkg (generated JS binding + WebAssembly)"
echo "Serve the repository, then open /examples/webgl/"
