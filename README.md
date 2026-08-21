# tlottie

Library for drawing [lottie animations](https://en.wikipedia.org/wiki/Lottie_(file_format)), written in Rust.

- Micro-optimized and [benchmarked](#benchmarks) across 17k+ animations from Telegram, against [rlottie](https://github.com/Samsung/rlottie) and [thorvg](https://github.com/thorvg/thorvg).
- SIMD on ARM NEON, x86_64 SSE2 AVX2 AVX512, and WASM: [web demo](https://dkaraush.github.io/tlottie/examples/web/).
- Support of `fitz` modifier and color replacements.
- Support of rendering into only alpha channel bitmap.
- Safe. Lottie JSON is treated as untrusted input.
- Zero dependencies.

### Building (Native)

```bash
cargo rustc --release --features c-api --lib --crate-type staticlib
# library will be built at target/release/libtlottie.a
# include headers are at include/tlottie.h
```

### Building (Web)

```bash
./examples/web/build.sh
# .wasm will be copied to examples/web/tlottie.wasm
```

Example on how to work with `tlottie.wasm` is at [examples/web/](https://github.com/dkaraush/tlottie/tree/dev/examples/web).

### Benchmarks

Frame time, mainly compared to [rlottie2019](https://github.com/TelegramMessenger/rlottie).

- [android arm, Samsung Fold7](https://dkaraush.github.io/tlottie/benchmarks/android-fold.html): **-64.7%** at 64px, **-55.8%** at 320px, **-52.0%** at 720px
- [android arm, Samsung F15](https://dkaraush.github.io/tlottie/benchmarks/android-f15.html): **-62.7%** at 64px, **-42.0%** at 320px, **-23.0%** at 720px
- [macOS arm](https://dkaraush.github.io/tlottie/benchmarks/macos.html): **-61.5%** at 64px, **-47.6%** at 320px, **-27.3%** at 720px
- [linux x86](https://dkaraush.github.io/tlottie/benchmarks/linux.html): **-70.0%** at 64px, **-50.2%** at 320px, **-36.4%** at 720px
