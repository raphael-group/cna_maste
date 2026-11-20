#!/usr/bin/env bash
# Build cnamaste's alpha expansion, `libcnamaste_expansion.so`, into `$CNAMASTE_EXPANSION_HOME`.
#
# The crate beside this script has no dependencies; `cargo` (rust-toolchain stable) builds it and the library is copied
# where `cnamaste.expansion` finds it, its tests skipping without. It fetches nothing.
#
#   CNAMASTE_EXPANSION_HOME=/some/dir infra/expansion/build_expansion.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "build_expansion: $1" >&2; exit 2; }

[ -n "${CNAMASTE_EXPANSION_HOME:-}" ] || die "set \$CNAMASTE_EXPANSION_HOME to the directory the library is written to"
home="$CNAMASTE_EXPANSION_HOME"
[ -d "$home" ] || die "\$CNAMASTE_EXPANSION_HOME ($home) does not exist; create it first"
command -v cargo >/dev/null || die "cargo is not on PATH"

cargo build --release --locked --manifest-path "$here/Cargo.toml" --target-dir "$here/target"
cp "$here/target/release/libcnamaste_expansion.so" "$home/libcnamaste_expansion.so"
echo "build_expansion: $home/libcnamaste_expansion.so ($(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo unversioned), $(cargo --version))"
