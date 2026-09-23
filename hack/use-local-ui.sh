#!/usr/bin/env bash
#
# Install a locally built web console into this repository.
#
# Why this exists: this repository does not compile the console. `make install`
# downloads a published tarball from a CDN bucket, so out of the box the product
# serves whatever build that bucket holds — logos, links and all — regardless of
# what the console's own source says. That is fine for upstream and wrong for a
# fork. Two ways out: publish your own console artifacts and point
# UI_RELEASE_BASE_URL at them, or build the console yourself and put it here.
# This script is the second one, and the one that works today.
#
# It does exactly the local half of `download_ui()` in hack/install.sh — unpack
# a build into gpustack/ui — and nothing else, so the pieces that are not part of
# the console bundle (repository static files, the packaged helm chart) still come
# from `make install` with UI_DOWNLOAD=false.
#
# Usage:
#   hack/use-local-ui.sh /path/to/gpustack-ui/dist
#   UI_DOWNLOAD=false make install
#
# The second command is required, not optional: it copies static/ into
# gpustack/ui/static and packages the helm chart into gpustack/ui/static/charts,
# both of which the server serves. Installing the console without it gives a UI
# whose chart download and extra static assets are missing.
#
# Run with --clean to delete gpustack/ui first. Use it when switching between
# console builds: without it, files that no longer exist in the new build stay
# behind and can be served. With it, re-run `UI_DOWNLOAD=false make install`
# afterwards or the static assets and chart are gone.
#
# This script deliberately does not source hack/lib/init.sh, unlike its
# neighbours. That library calls `gpustack::version::get_version_vars` when it
# loads, and on a dirty working tree that function prints the repository's entire
# git status and diff (hack/lib/version.sh:69-76) to explain why the version
# degrades to v0.0.0. Measured here: 4400 lines of diff around ten lines of actual
# output. The library is needed for logging and for the version, and this script
# needs neither, so it logs with printf.

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

log_info() { printf '[INFO] %s\n' "$1"; }
log_warn() { printf '[WARN] %s\n' "$1"; }
log_fatal() {
  printf '[FATA] %s\n' "$1" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: hack/use-local-ui.sh [--clean] <built-console-dir>

  <built-console-dir>  a console build directory containing index.html
                       (the `dist/` produced by `pnpm build` in gpustack-ui)

  --clean              remove gpustack/ui first, then re-run
                       `UI_DOWNLOAD=false make install` to restore the
                       static assets and the packaged helm chart
EOF
}

clean="false"
src=""
for arg in "$@"; do
  case "${arg}" in
  --clean)
    clean="true"
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    if [[ -n "${src}" ]]; then
      log_fatal "unexpected extra argument: ${arg}"
    fi
    src="${arg}"
    ;;
  esac
done

if [[ -z "${src}" ]]; then
  usage
  exit 1
fi

if [[ ! -d "${src}" ]]; then
  log_fatal "${src} is not a directory"
fi

# index.html is what makes a directory a console build rather than, say, the
# frontend source tree or an empty dist. Checking it here turns a silent broken
# install into an error at the point of the mistake.
if [[ ! -f "${src}/index.html" ]]; then
  log_fatal "${src} has no index.html; point this at the console's built dist directory"
fi

src_abs="$(cd "${src}" && pwd -P)"
ui_path="${ROOT_DIR}/gpustack/ui"

case "${src_abs}/" in
"${ui_path}"/*)
  log_fatal "refusing to copy ${src_abs} into itself"
  ;;
esac

log_info "installing console build from ${src_abs}"

if [[ "${clean}" == "true" ]]; then
  log_info "--clean: removing ${ui_path}"
  rm -rf "${ui_path}"
fi

mkdir -p "${ui_path}"
cp -a "${src_abs}/." "${ui_path}"

if [[ ! -f "${ui_path}/index.html" ]]; then
  log_fatal "copy finished but ${ui_path}/index.html is missing"
fi

# An audit line, not a failure: the console legitimately contains the upstream
# name — X-GPUStack-* are the wire headers and some help links point at upstream
# documentation on purpose. What this catches is the case where the directory
# handed to this script is itself an upstream build, which is otherwise
# indistinguishable from success.
#
# Counted with a loop rather than `grep -rl | wc -l` because grep exits 1 when it
# matches nothing and `set -o pipefail` turns that into a failed pipeline: the
# quietest, most normal outcome — a build with no upstream strings left — would be
# treated as an error. `if grep -q` has no status to propagate.
upstream_hits=0
while IFS= read -r -d '' file; do
  if grep -q 'GPUStack' "${file}" 2>/dev/null; then
    upstream_hits=$((upstream_hits + 1))
  fi
done < <(find "${ui_path}" -type f -print0 2>/dev/null)
log_info "files containing 'GPUStack': ${upstream_hits} (expected non-zero: wire headers and upstream docs links)"

if [[ ! -d "${ui_path}/static" ]]; then
  log_warn "${ui_path}/static is missing: run 'UI_DOWNLOAD=false make install' to add it"
fi

if [[ ! -f "${ui_path}/static/charts/gpustack-chart.tgz" ]]; then
  log_warn "the packaged helm chart is missing: run 'UI_DOWNLOAD=false make install'"
fi

log_info "done. Next: UI_DOWNLOAD=false make install"
