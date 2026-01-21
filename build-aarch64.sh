#!/bin/bash
# Cross-compile platinenmachergpt for aarch64

set -e

TARGET="aarch64-unknown-linux-gnu"
BUILD_TYPE="${1:-release}"

echo "Building for $TARGET ($BUILD_TYPE)..."

if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --target "$TARGET" --release
    OUTPUT="target/$TARGET/release/platinenmachergpt"
else
    cargo build --target "$TARGET"
    OUTPUT="target/$TARGET/debug/platinenmachergpt"
fi

echo ""
echo "Build complete!"
echo "Binary: $OUTPUT"
echo ""
echo "To deploy, copy the following files to the target device:"
echo "  - $OUTPUT"
echo "  - lib/librkllmrt.so -> /usr/lib/librkllmrt.so"
echo ""
echo "On the target device, ensure librkllmrt.so is in the library path:"
echo "  export LD_LIBRARY_PATH=/usr/lib:\$LD_LIBRARY_PATH"
