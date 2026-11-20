#!/usr/bin/env bash
# Build cnamaste's OpenGM library, `libcnamaste_opengm.so`, into a cache.
#
# OpenGM is not on PyPI, and its inference is header-only C++
# templates. This compiles `infra/opengm/cnamaste_opengm.cxx` against a
# checkout the caller supplies, and writes the library, the commit and the
# licence terms beside it, where `cnamaste.external` finds them and its tests
# skip without. No HDF5, no Python binding and no downloaded external solver
# is built: every algorithm in scope is OpenGM's own MIT code, and the min s-t
# cut alpha expansion and the swap need is Boost Graph's Boykov-Kolmogorov
# (headers, required) or infra/opengm/dinic.cxx.
#
#   CNAMASTE_OPENGM_HOME=/some/dir infra/opengm/build_opengm.sh
#
# **It fetches nothing and creates nothing.** `$CNAMASTE_OPENGM_HOME` is
# required, must already exist, and must already hold OpenGM's checkout at the
# pinned commit under `src/`; `OPENGM.md` states the clone. Anything
# missing or at another commit is reported and the script exits without
# touching it, because the directory is the caller's and a checkout is theirs
# to place, move or keep.
#
# Idempotent: a build whose stamp (commit, driver source, compiler, flags)
# matches is kept, and anything else is rebuilt.
set -euo pipefail

# NB the driver is this script's neighbour, so the path is resolved from the
#    script rather than from the repository root: one less thing to re-point
#    if either moves again, and cwd still does not matter.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#: The commit built: OpenGM's last, 2019-07-26; the project is unmaintained.
commit="decdacf4caad223b0ab5478d38a855f8767a394f"
repository="https://github.com/opengm/opengm"
source_file="$here/cnamaste_opengm.cxx"
dinic_file="$here/dinic.cxx"
flags="-std=c++11 -O2 -shared -fPIC -w"
compiler="${CXX:-g++}"

die() { echo "build_opengm: $1" >&2; exit 2; }

# NB no default home: a cache this script invents is a directory nobody chose
#    and nobody prunes. The caller names it, and `cnamaste.external.home`
#    reads the same variable, so the two cannot disagree.
[ -n "${CNAMASTE_OPENGM_HOME:-}" ] || die "set \$CNAMASTE_OPENGM_HOME to the directory holding OpenGM's checkout (OPENGM.md)"
home="$CNAMASTE_OPENGM_HOME"

[ -d "$home" ] || die "\$CNAMASTE_OPENGM_HOME ($home) does not exist; create it and clone OpenGM into src/ (OPENGM.md)"

# NB the checkout is read, never written: no fetch, no checkout, no `rm -rf`.
#    A wrong or missing one is the caller's to fix, and deleting a directory
#    they named is not this script's to do.
[ -d "$home/src/include/opengm" ] || die "no OpenGM headers at $home/src/include/opengm; clone the pinned commit there (OPENGM.md)"

found="$(git -C "$home/src" rev-parse HEAD 2>/dev/null || true)"
if [ "$found" != "$commit" ]; then
  die "$home/src is at ${found:-no commit}, not the pinned $commit; check it out there (OPENGM.md)"
fi

# NB after the checkout, not before it. The stamp records the commit a build
#    was asked for, never the tree it actually read, so a stamp hit on a
#    checkout that has since moved would report a stale library as current.
# NB Boost Graph's Boykov-Kolmogorov max flow, header-only: from the compiler's own include path, or
#    $BOOST_INCLUDEDIR where set. Its version is read off the headers found, so the stamp names what was built.
boost=()
[ -n "${BOOST_INCLUDEDIR:-}" ] && boost=(-I"$BOOST_INCLUDEDIR")
boost_version="$(printf '#include <boost/version.hpp>\nBOOST_LIB_VERSION\n' | "$compiler" -E -P -x c++ "${boost[@]}" - 2>/dev/null | tail -1 | tr -d '"')" \
  || die "no Boost headers: install them (apt install libboost-dev, brew install boost, conda install boost-cpp) or set \$BOOST_INCLUDEDIR (OPENGM.md)"
[ -n "$boost_version" ] || die "no Boost headers: install them or set \$BOOST_INCLUDEDIR (OPENGM.md)"

stamp="$commit $(cat "$source_file" "$dinic_file" | sha256sum | cut -d' ' -f1) $compiler $flags boost-$boost_version"
if [ -f "$home/STAMP" ] && [ "$(cat "$home/STAMP")" = "$stamp" ] && [ -f "$home/libcnamaste_opengm.so" ]; then
  echo "opengm: current at $home"
  exit 0
fi

start=$(date +%s)
# shellcheck disable=SC2086 # the flags are words
"$compiler" $flags -I"$home/src/include" "${boost[@]}" "$source_file" -o "$home/libcnamaste_opengm.so.tmp"
mv "$home/libcnamaste_opengm.so.tmp" "$home/libcnamaste_opengm.so"
echo "$commit" > "$home/COMMIT"
cat > "$home/LICENCES" <<TERMS
OpenGM $commit ($repository): MIT, Licence-OpenGM.txt.
Built: OpenGM's header-only ICM, loopy BP, A*, TRWSi (Savchynskyy's TRW-S),
dual decomposition (subgradient, dynamic-programming subproblems), alpha
expansion and alpha-beta swap. Their min s-t cut: Boost Graph's
Boykov-Kolmogorov max flow (Boost $boost_version, Boost Software License 1.0,
header-only) or infra/opengm/dinic.cxx's Dinic.
Not fetched or built: the externals OpenGM's CMake downloads (Kolmogorov's
TRW-S v1.3 and MRF-LIB, maxflow v3.02, QPBO v1.3: research-only or GPL;
gco-v3.0: research-only; IBFS, AD3, ConicBundle, and the rest).
TERMS
echo "$stamp" > "$home/STAMP"
echo "opengm: built $home/libcnamaste_opengm.so in $(( $(date +%s) - start )) s"
