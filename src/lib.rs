//! WGPU LTC Area Lighting Demo
//!
//! Shared library code for both native and WebAssembly builds.
//!
//! ## Building for WebAssembly
//!
//! ```bash
//! # Install wasm-bindgen-cli if not installed
//! cargo install wasm-bindgen-cli
//!
//! # Add wasm32 target if not installed
//! rustup target add wasm32-unknown-unknown
//!
//! # Build for wasm32
//! cargo build --release --target wasm32-unknown-unknown --lib
//!
//! # Generate JS bindings
//! wasm-bindgen --target web --out-dir pkg target/wasm32-unknown-unknown/release/wgpu_linear_cosine_lighting.wasm
//!
//! # Serve locally (use any web server)
//! python -m http.server 8080
//! # Then open http://localhost:8080 in a WebGPU-enabled browser
//! ```

mod app;
mod ltc_lut;

pub use app::run;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Debug).expect("Failed to initialize logger");

    run();
}
