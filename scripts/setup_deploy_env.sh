#!/usr/bin/env bash
# Build the uv-managed real-robot deployment venv (.venv-deploy).
#
# Why this exists: unitree_sdk2py pins cyclonedds==0.10.2, which has no wheel
# for modern CPython/Linux, and three upstream packaging defects break a naive
# install:
#   1. Ubuntu's apt cyclonedds is 0.10.4 with a patched soname
#      (libddsc.so.0debian); the 0.10.2 python binding aborts inside it with
#      "buffer overflow detected" at ChannelFactoryInitialize.
#   2. cyclonedds' loader ignores its own recorded library_path for source
#      builds (in_wheel=False) and falls back to find_library("ddsc"), which
#      resolves to the broken system 0.10.4 — so even a correct source build
#      crashes unless CYCLONEDDS_HOME is exported at every run. We flip
#      in_wheel=True so the recorded absolute path is used unconditionally.
#   3. unitree_sdk2py's pip install drops utils/lib/*.so (the CRC natives);
#      they are copied in from the git checkout.
#
# After this script: no environment variables needed at runtime.
#   .venv-deploy/bin/python src/mjlab_homierl/scripts/deploy_g1_homie.py \
#     --onnx <policy.onnx> --net <iface>
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PREFIX=$ROOT/.local/cyclonedds-0.10.2
VENV=$ROOT/.venv-deploy
PY=3.11
BUILD=$(mktemp -d)

uv tool install --quiet cmake
uv tool install --quiet patchelf
export PATH=$HOME/.local/bin:$PATH

if [ ! -f "$PREFIX/lib/libddsc.so" ]; then
  git clone --depth 1 -b 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds "$BUILD/cyclonedds"
  cmake -S "$BUILD/cyclonedds" -B "$BUILD/build" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DENABLE_SSL=OFF -DENABLE_SECURITY=OFF
  cmake --build "$BUILD/build" -j"$(nproc)"
  cmake --install "$BUILD/build"
fi

[ -d "$VENV" ] || uv venv --python $PY "$VENV"
CYCLONEDDS_HOME=$PREFIX uv pip install --python "$VENV/bin/python" \
  --no-cache-dir cyclonedds==0.10.2
uv pip install --python "$VENV/bin/python" numpy onnxruntime \
  "unitree_sdk2py @ git+https://github.com/unitreerobotics/unitree_sdk2_python.git"

SITE=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

# Fix 2: use the recorded absolute libddsc path unconditionally.
sed -i "s/^in_wheel = False/in_wheel = True/" "$SITE/cyclonedds/__library__.py"
grep -q "in_wheel = True" "$SITE/cyclonedds/__library__.py"

# Belt-and-braces: rpath the extension modules to our prefix too.
for so in "$SITE"/cyclonedds/_*.so; do
  patchelf --set-rpath "$PREFIX/lib" "$so"
done

# Fix 3: restore the CRC native libraries dropped by pip.
if [ ! -f "$SITE/unitree_sdk2py/utils/lib/crc_amd64.so" ]; then
  SRC=$(find "$HOME/.cache/uv/git-v0" -path "*unitree_sdk2py/utils/lib" -type d | head -1)
  [ -n "$SRC" ] || { echo "CRC libs not found in uv git cache"; exit 1; }
  mkdir -p "$SITE/unitree_sdk2py/utils/lib"
  cp "$SRC"/*.so "$SITE/unitree_sdk2py/utils/lib/"
fi

"$VENV/bin/python" -c "
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
print('crc ok:', CRC().Crc(unitree_hg_msg_dds__LowCmd_()))
import cyclonedds.__library__ as l
assert '.local/cyclonedds-0.10.2' in l.library_path, l.library_path
print('libddsc:', l.library_path)
"
echo "Deploy venv ready: $VENV"
