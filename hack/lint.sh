#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function lint() {
  local path="$1"

  gpustack::log::info "linting ${path}"
  uv run pre-commit run --all-files --show-diff-on-failure
}

# Brand and attribution guard, kept separate from pre-commit because it is not a
# code style check and must not be skippable: it is what notices a merge putting
# an upstream string back into a user-visible surface, or a cleanup deleting the
# upstream copyright line.
function check_brand() {
  gpustack::log::info "checking brand and attribution"
  bash "${ROOT_DIR}/hack/check-brand.sh"
}

#
# main
#
# Both checks run, and both are reported, before this script decides the exit
# code. That ordering is not polish: `set -o errexit` would otherwise let a lint
# failure hide the brand guard entirely, and the guard is the check that protects
# the licence. A branch with an unrelated flake8 complaint — this one currently
# has four, in the billing code — would then never reach the attribution check,
# and a red `make lint` that mentions only flake8 is exactly how an Apache-2.0
# §4(c) breach survives review.
status=0

gpustack::log::info "+++ LINT +++"
if ! lint "gpustack"; then
  gpustack::log::error "style checks reported failures"
  status=1
fi
gpustack::log::info "--- LINT ---"

gpustack::log::info "+++ BRAND +++"
if ! check_brand; then
  gpustack::log::error "brand and attribution guard reported failures"
  status=1
fi
gpustack::log::info "--- BRAND ---"

exit "${status}"
