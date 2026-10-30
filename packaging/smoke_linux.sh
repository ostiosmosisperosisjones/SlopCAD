#!/usr/bin/env bash
# Headless checks of a built AppImage: kernel workers (--selftest) and the
# real window under Xvfb (--smoke-gui).  Exit 0 only if both pass.
#
#   packaging/smoke_linux.sh [dist/slopcad-linux.AppImage]
set -uo pipefail
APP="$(readlink -f "${1:-dist/slopcad-linux.AppImage}")"
export APPIMAGE_EXTRACT_AND_RUN=1          # no FUSE in containers/CI
out="$(mktemp -d)"
rc=0

echo "== selftest"
"$APP" --selftest "$out/selftest.json"; st=$?
cat "$out/selftest.json" 2>/dev/null
[ $st -eq 0 ] || { echo "selftest FAILED (exit $st)"; rc=1; }

echo "== gui smoke (xvfb)"
timeout 120 xvfb-run -a -s "-screen 0 1600x1000x24" "$APP" --smoke-gui "$out/gui.json"; st=$?
cat "$out/gui.json" 2>/dev/null
[ $st -eq 0 ] || { echo "gui smoke FAILED (exit $st)"; rc=1; }

if [ $rc -ne 0 ]; then
  log="${XDG_CONFIG_HOME:-$HOME/.config}/cadapp/slopcad.log"
  echo "== tail of $log"; tail -40 "$log" 2>/dev/null
fi
exit $rc
