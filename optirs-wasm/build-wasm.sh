#!/bin/bash
# Build script for optirs-wasm
# Uses cargo + wasm-bindgen-cli directly for nightly compatibility.
# Falls back to wasm-pack if available and compatible.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_NAME="@cooljapan/optirs"
OUT_DIR="$SCRIPT_DIR/pkg"

echo "Building optirs-wasm..."

# Check for wasm-bindgen CLI
if ! command -v wasm-bindgen &> /dev/null; then
    echo "Installing wasm-bindgen-cli..."
    cargo install wasm-bindgen-cli
fi

build_target() {
    local target=$1
    local out="$OUT_DIR/$target"
    local wasm_file="$SCRIPT_DIR/../target/wasm32-unknown-unknown/release/optirs_wasm.wasm"

    echo "Building for target: $target -> $out"
    mkdir -p "$out"

    # Step 1: Build the WASM binary with cargo
    cargo build \
        --manifest-path "$SCRIPT_DIR/Cargo.toml" \
        --lib \
        --release \
        --target wasm32-unknown-unknown \
        --features wasm

    # Step 2: Run wasm-bindgen to generate JS bindings
    wasm-bindgen \
        "$wasm_file" \
        --out-dir "$out" \
        --out-name optirs \
        --target "$target" \
        --typescript

    # Step 3: Patch package.json with scoped name
    if [ -f "$out/package.json" ]; then
        sed -i.bak "s/\"name\": \"optirs-wasm\"/\"name\": \"$PACKAGE_NAME\"/" "$out/package.json"
        rm -f "$out/package.json.bak"
    fi

    echo "Built: $out"
}

case "${1:-web}" in
    web)
        build_target "web"
        ;;
    bundler)
        build_target "bundler"
        ;;
    nodejs)
        build_target "nodejs"
        ;;
    all)
        build_target "web"
        build_target "bundler"
        build_target "nodejs"
        ;;
    *)
        echo "Usage: $0 {web|bundler|nodejs|all}"
        exit 1
        ;;
esac

echo "Build complete!"
