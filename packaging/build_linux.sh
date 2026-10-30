#!/usr/bin/env bash
# Build dist/slopcad-linux.AppImage.  Expects an Ubuntu 24.04 userland with
# the apt packages in packaging/Containerfile — the CI runner, or that
# container (packaging/release.sh runs it there; don't build on a rolling
# distro directly: the AppImage would need its newer glibc).
#
#   PYTHON=python3.12  VENV=.build-venv  packaging/build_linux.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.build-venv}"
CACHE="packaging/.cache"
mkdir -p "$CACHE"

"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt -r packaging/requirements-build.txt

pyinstaller --noconfirm --clean slopcad.spec

# Use the user's system GL: bundled driver-coupled libGL/libEGL/libxcb-glx
# cause GLX version-mismatch crashes.
( cd dist/slopcad/_internal
  rm -f libGL.so* libGL-*.so* libEGL.so* libEGL-*.so* libxcb-glx.so* )

# libGLU IS bundled: the app calls gluProject/gluPerspective, GLU is pure
# vendor-neutral math, and many minimal distros don't ship it.
glu="$(ldconfig -p | awk '/libGLU\.so\.1/ {print $NF; exit}')"
[ -n "$glu" ] || { echo "libGLU.so.1 not found on build host" >&2; exit 1; }
cp -L "$glu" dist/slopcad/_internal/libGLU.so.1

rm -rf AppDir
mkdir -p AppDir/usr/bin
cp -r dist/slopcad/* AppDir/usr/bin/
cat > AppDir/slopcad.desktop <<'DESKTOP'
[Desktop Entry]
Name=SlopCAD
Exec=slopcad
Icon=slopcad
Type=Application
Categories=Graphics;
DESKTOP
cp brand/avatar-640.png AppDir/slopcad.png
cat > AppDir/AppRun <<'APPRUN'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:$HERE/usr/bin:$HERE/usr/bin/_internal"
exec "$HERE/usr/bin/slopcad" "$@"
APPRUN
chmod +x AppDir/AppRun

tool="$CACHE/appimagetool-x86_64.AppImage"
if [ ! -x "$tool" ]; then
  curl -fsSL -o "$tool" \
    https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$tool"
fi
# extract-and-run: no FUSE needed (containers, CI)
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 "$tool" AppDir dist/slopcad-linux.AppImage
echo "built dist/slopcad-linux.AppImage"
