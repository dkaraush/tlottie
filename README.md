# tlottie

Library for drawing [lottie animations](https://en.wikipedia.org/wiki/Lottie_(file_format)), written in Rust.

- Micro-optimized and [benchmarked](#benchmarks) across 17k+ animations from Telegram, against [rlottie](https://github.com/Samsung/rlottie) and [thorvg](https://github.com/thorvg/thorvg).
- SIMD on ARM NEON, x86_64 SSE2 AVX2 AVX512, and WASM: [web demo](https://dkaraush.github.io/tlottie/examples/web/).
- Support of `fitz` modifier and color replacements.
- Support of rendering into only alpha channel bitmap.
- Safe. Lottie JSON is treated as untrusted input.
- No runtime dependencies in default `std` builds; opt-in `no_std` builds use
  only `hashbrown` for the map type missing from `alloc`.[^1]

### Building (Native)

```bash
cargo rustc --release --features c-api --lib --crate-type staticlib
# library will be built at target/release/libtlottie.a
# include headers are at include/tlottie.h
```

Native CPU builds use the Rust standard library by default. To build the
experimental `core` + `alloc` C API static library explicitly:

```bash
cargo rustc --profile release-nostd --lib \
  --no-default-features --features cpu,no-std,c-api \
  --crate-type staticlib
```

The no-std C API aborts on an unexpected panic, so Rust cannot unwind through
the C ABI, and supplies a small allocator backed by the host C runtime.

### Building (Web)

```bash
./examples/web/build.sh
# .wasm will be copied to examples/web/tlottie.wasm
```
The WebAssembly script keeps `std` enabled. An opt-in no-std build is also
supported with `--profile release-nostd --no-default-features --features
wasm,no-std`; it uses dlmalloc directly over WebAssembly linear-memory growth
and traps on an unexpected panic.
Example on how to work with `tlottie.wasm` is at [examples/web/](https://github.com/dkaraush/tlottie/tree/dev/examples/web).

### Benchmarks

Frame time, mainly compared to [rlottie2019](https://github.com/TelegramMessenger/rlottie).

- [android arm, Samsung Fold7](https://dkaraush.github.io/tlottie/benchmarks/android-fold.html): **-64.7%** at 64px, **-55.8%** at 320px, **-52.0%** at 720px
- [android arm, Samsung F15](https://dkaraush.github.io/tlottie/benchmarks/android-f15.html): **-62.7%** at 64px, **-42.0%** at 320px, **-23.0%** at 720px
- [macOS arm](https://dkaraush.github.io/tlottie/benchmarks/macos.html): **-61.5%** at 64px, **-47.6%** at 320px, **-27.3%** at 720px
- [linux x86](https://dkaraush.github.io/tlottie/benchmarks/linux.html)[^2]: **-70.0%** at 64px, **-50.2%** at 320px, **-36.4%** at 720px

[^1]: GPU renderer dependencies are currently work in progress and expected to be useful only on large canvas sizes.
[^2]: Tested on Threadripper PRO 9995WX at full core count, running tlottie's SSE2+AVX2+AVX-512 backend against rlottie's. Relative parity with rlottie2019 is worse here than on smaller cores because the vector span kernels are memory-bound at 720px; tlottie is faster than this on typical x86_64 laptops (see the AVX-512 breakdown in the report).
