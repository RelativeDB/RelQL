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

# native library -> package -> platform wheel. Pin the macOS deployment
# target so the wheel installs on more than the build host's OS release
# (rt_metal guards newer APIs with @available).
case "$(uname)" in Darwin)
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}" ;;
esac
cmake -S ../cpp -B ../cpp/build -DCMAKE_BUILD_TYPE=Release \
  ${MACOSX_DEPLOYMENT_TARGET:+-DCMAKE_OSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET} \
  >/dev/null
cmake --build ../cpp/build -j --target rt_c
for f in ../cpp/build/librt_c.dylib ../cpp/build/librt_c.so \
         ../cpp/build/rt_c.dll; do
  [ -f "$f" ] && cp "$f" src/relativedb/
done
"$PY" -m build --wheel

echo
echo "artifacts:"
ls -l dist
echo
echo "native lib in wheel:"
"$PY" -c "import glob, zipfile; w = glob.glob('dist/*.whl')[0]; [print(' ', n) for n in zipfile.ZipFile(w).namelist() if 'librt_c' in n or 'rt_c' in n]"
