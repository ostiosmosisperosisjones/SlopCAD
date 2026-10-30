#!/usr/bin/env bash
# Local release builds — the same scripts CI runs.
#
#   packaging/release.sh linux              AppImage in an ubuntu:24.04 container,
#                                           then smoke-tested in the container
#                                           AND on this host
#   packaging/release.sh windows HOST       copy the working tree to HOST over
#                                           ssh, build + selftest there, fetch
#                                           dist/slopcad-windows.zip
#   packaging/release.sh all HOST           both
#
# Env: ENGINE=podman|docker (default: podman if present), WIN_PYTHON (python
# on the Windows box, default the per-user python.org 3.12 install), WIN_DIR
# (build dir there, default slopcad-build under the remote user's home).
# Assumes the box's ssh shell is PowerShell (the Windows OpenSSH default
# when configured).
set -euo pipefail
cd "$(dirname "$0")/.."

ENGINE="${ENGINE:-$(command -v podman >/dev/null && echo podman || echo docker)}"
IMAGE=slopcad-build:24.04

linux() {
  "$ENGINE" build -q -t "$IMAGE" -f packaging/Containerfile packaging >/dev/null
  # venv + pip/appimagetool caches persist across runs (.build-venv-linux and
  # packaging/.cache in the tree, pip cache in a named volume).
  "$ENGINE" run --rm -v "$PWD:/src:Z" -v slopcad-pip-cache:/root/.cache/pip \
    -e PYTHON=python3.12 -e VENV=.build-venv-linux \
    "$IMAGE" bash -c 'packaging/build_linux.sh && packaging/smoke_linux.sh'
  echo "== smoke on this host ($(. /etc/os-release; echo "$NAME"))"
  packaging/smoke_linux.sh dist/slopcad-linux.AppImage
}

windows() {
  local host="$1" dir="${WIN_DIR:-slopcad-build}"
  # per-user python.org 3.12 (the WindowsApps "python" is a Store stub)
  local py="${WIN_PYTHON:-\$env:LOCALAPPDATA\\Programs\\Python\\Python312\\python.exe}"
  echo "== copying working tree to $host:$dir"
  # tarball via scp: Windows PowerShell 5.1 mangles binary stdin to tar
  local tarball; tarball="$(mktemp --suffix=.tar)"
  git ls-files -c -o --exclude-standard -z | tar --null -T - -cf "$tarball"
  ssh "$host" "New-Item -ItemType Directory -Force '$dir' | Out-Null"
  scp -q "$tarball" "$host:$dir/src.tar"
  rm -f "$tarball"
  ssh "$host" "tar -xf '$dir/src.tar' -C '$dir'; Remove-Item '$dir/src.tar'"
  ssh "$host" "powershell -NoProfile -ExecutionPolicy Bypass -File '$dir/packaging/build_windows.ps1' -Python \"$py\""
  mkdir -p dist
  scp "$host:$dir/dist/slopcad-windows.zip" dist/
  echo "fetched dist/slopcad-windows.zip"
}

case "${1:-}" in
  linux)   linux ;;
  windows) windows "${2:?usage: release.sh windows HOST}" ;;
  all)     linux; windows "${2:?usage: release.sh all HOST}" ;;
  *)       sed -n '2,14p' "$0"; exit 2 ;;
esac
