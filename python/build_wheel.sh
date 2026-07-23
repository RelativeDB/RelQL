#!/bin/sh
# Build release artifacts: a pure sdist, then a platform wheel with the
# native engine (librt_c) bundled inside the relativedb package, where
# rt_native/csc_native look first.
#
#   ./build_wheel.sh                 # uses python3
#   PYTHON=.venv/bin/python ./build_wheel.sh
#
# Installs from the sdist have no bundled library and fall back to
# RELATIVEDB_RT_LIB or a monorepo cpp/build tree at runtime.
set -eu
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

rm -rf dist build src/relativedb.egg-info
rm -f src/relativedb/librt_c.dylib src/relativedb/librt_c.so \
      src/relativedb/librt_c.dll src/relativedb/rt_c.dll

# sdist first, while the tree is pure source
"$PY" -m build --sdist

# native library -> package -> platform wheel. macOS builds target 13.0
# (rt_metal guards newer APIs with @available) as a universal arm64 +
# x86_64 dylib, so one wheel serves both Mac architectures. setup.py reads
# the wheel's platform tag straight from the bundled dylib — no
# environment variables involved.
MAC_FLAGS=""
case "$(uname)" in Darwin)
  MAC_FLAGS='-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DCMAKE_OSX_ARCHITECTURES=arm64;x86_64' ;;
esac
# The dedicated wheel build tree keeps arch/deployment flags from fighting
# the development build in cpp/build.
BUILD_DIR=../cpp/build-wheel
rm -rf "$BUILD_DIR"
cmake -S ../cpp -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release $MAC_FLAGS \
  >/dev/null
cmake --build "$BUILD_DIR" -j --target rt_c
for f in "$BUILD_DIR"/librt_c.dylib "$BUILD_DIR"/librt_c.so \
         "$BUILD_DIR"/rt_c.dll; do
  [ -f "$f" ] && cp "$f" src/relativedb/
done
case "$(uname)" in Darwin)
  echo "dylib architectures: $(lipo -archs src/relativedb/librt_c.dylib)" ;;
esac
"$PY" -m build --wheel

echo
echo "artifacts:"
ls -l dist
echo
echo "native lib in wheel:"
"$PY" -c "import glob, zipfile; w = glob.glob('dist/*.whl')[0]; [print(' ', n) for n in zipfile.ZipFile(w).namelist() if 'librt_c' in n or 'rt_c' in n]"
