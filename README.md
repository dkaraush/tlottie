# tlottie

Library for drawing [lottie animations](https://en.wikipedia.org/wiki/Lottie_(file_format)), written in Rust.

- Micro-optimized and [benchmarked](#benchmarks) across 17k+ animations from Telegram, against [rlottie](https://github.com/Samsung/rlottie) and [thorvg](https://github.com/thorvg/thorvg).
- SIMD on ARM NEON, and WASM: [web demo](https://dkaraush.github.io/tlottie/examples/web/).
- Support of `fitz` modifier and color replacements by layer name prefixes.
- Support of rendering into only alpha channel bitmap.
- Safe. Lottie JSON is treated as untrusted input.
- Zero dependencies[^1].

### Building (Native)

```bash
cargo rustc --release --features c-api --lib --crate-type staticlib
# library will be built at target/release/libtlottie.a
# include headers are at include/tlottie.h
```

### Building (Web)

```bash
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release --no-default-features --features wasm --lib
# .wasm will be built at target/wasm32-unknown-unknown/release-size/tlottie.wasm
```
Example on how to work with `tlottie.wasm` is at [examples/web/](https://github.com/dkaraush/tlottie/tree/dev/examples/web).

### Benchmarks

Frame time, mainly compared to [rlottie2019](https://github.com/TelegramMessenger/rlottie).

- [android arm, Samsung Fold7](https://dkaraush.github.io/tlottie/benchmarks/android-fold.html): **-40.6%** at 64px, **-45.8%** at 320px, **-47.6%** at 720px
- [macOS arm](https://dkaraush.github.io/tlottie/benchmarks/macos.html): **-61.5%** at 64px, **-43.5%** at 320px, **-32%** at 720px
- [linux x86](https://dkaraush.github.io/tlottie/benchmarks/linux.html)[^2]: **-93.2%** at 64px, **-82.3%** at 320px, **-68.8%** at 720px

[^1]: Listed dependencies are only for GPU renderers, that are currently work in progress, expected to be useful only on large canvas sizes.
[^2]: Tested on Threadripper running at all cores, a bit skeptical about such results.
