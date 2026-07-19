//! Optional native and WebAssembly bindings.

#[cfg(all(feature = "c-api", not(target_arch = "wasm32")))]
pub(crate) mod c_api;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub(crate) mod wasm;
