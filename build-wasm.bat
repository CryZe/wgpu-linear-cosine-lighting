@echo off
echo Building for WebAssembly...

:: Check if wasm-bindgen is installed
where wasm-bindgen >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo wasm-bindgen-cli not found. Installing...
    cargo install wasm-bindgen-cli
)

:: Build with cargo for wasm32 target
cargo build --release --target wasm32-unknown-unknown --lib

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

:: Create pkg directory
if not exist pkg mkdir pkg

:: Run wasm-bindgen to generate JS bindings
wasm-bindgen --target web --out-dir pkg target/wasm32-unknown-unknown/release/wgpu_linear_cosine_lighting.wasm

if %ERRORLEVEL% equ 0 (
    echo.
    echo Build successful!
    echo.
    echo To run locally, use a web server:
    echo   python -m http.server 8080
    echo   or
    echo   npx serve .
    echo.
    echo Then open http://localhost:8080 in a WebGPU-enabled browser
    echo ^(Chrome, Edge, or Firefox Nightly with WebGPU enabled^)
) else (
    echo wasm-bindgen failed!
)
