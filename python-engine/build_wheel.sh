#!/bin/sh
# Build release artifacts: a pure sdist, then a platform wheel with the
# native engine (librt_c) bundled inside the relativedb_engine package, where
# native.py looks first.
#
#   ./build_wheel.sh                 # uses python3
#   PYTHON=.venv/bin/python ./build_wheel.sh
#   SKIP_SDIST=1 ./build_wheel.sh    # wheel only (CI: sdist is built once)
#
# Installs from the sdist have no bundled library and fall back to
# RELATIVEDB_RT_LIB or a monorepo cpp/build tree at runtime.
#
# CI (.github/workflows/wheels.yml) calls this same script inside the
# manylinux container and on the macOS runner, so the local and released
# artifacts come off one code path.
set -eu
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

rm -rf dist build src/relativedb_engine.egg-info
rm -f src/relativedb_engine/librt_c.dylib src/relativedb_engine/librt_c.so \
      src/relativedb_engine/librt_c.dll src/relativedb_engine/rt_c.dll

# sdist first, while the tree is pure source
if [ -z "${SKIP_SDIST:-}" ]; then
  "$PY" -m build --sdist
fi

# native library -> package -> platform wheel. macOS builds target 13.0
# (rt_metal guards newer APIs with @available) as a universal arm64 +
# x86_64 dylib, so one wheel serves both Mac architectures. setup.py reads
# the wheel's platform tag straight from the bundled dylib — no
# environment variables involved.
#
# Everywhere: rt_c is a SHARED library linking the STATIC rt, so the static
# objects must be position-independent (ELF refuses non-PIC objects in a .so).
set -- -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
case "$(uname)" in
  Darwin)
    set -- "$@" -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
           "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64" ;;
  Linux)
    # Fold the C++ runtime into librt_c. The manylinux policy allows only an
    # old GLIBCXX/CXXABI symbol set, and a ctypes leaf library has no C++ ABI
    # boundary that needs a shared libstdc++, so static linking removes the
    # whole class of "imports here, not on the user's box" failures. What is
    # left (libm/libgcc_s/libc/ld-linux) is inside the manylinux_2_28
    # whitelist, so auditwheel has nothing to vendor.
    set -- "$@" "-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc" ;;
esac
# The dedicated wheel build tree keeps arch/deployment flags from fighting
# the development build in cpp/build.
BUILD_DIR=../cpp/build-wheel
rm -rf "$BUILD_DIR"
cmake -S ../cpp -B "$BUILD_DIR" "$@" >/dev/null
cmake --build "$BUILD_DIR" -j --target rt_c
for f in "$BUILD_DIR"/librt_c.dylib "$BUILD_DIR"/librt_c.so \
         "$BUILD_DIR"/rt_c.dll; do
  [ -f "$f" ] && cp "$f" src/relativedb_engine/
done
case "$(uname)" in Darwin)
  echo "dylib architectures: $(lipo -archs src/relativedb_engine/librt_c.dylib)" ;;
esac
"$PY" -m build --wheel

echo
echo "artifacts:"
ls -l dist
echo
echo "native lib in wheel:"
"$PY" -c "import glob, zipfile; w = glob.glob('dist/*.whl')[0]; [print(' ', n) for n in zipfile.ZipFile(w).namelist() if 'librt_c' in n or 'rt_c' in n]"
