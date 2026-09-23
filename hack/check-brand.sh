#!/usr/bin/env bash
#
# Brand and attribution guard.
#
# Two jobs, and they pull in opposite directions, which is why they live in one
# script:
#
#   1. Keep the rebrand from regressing — a new hardcoded "GPUStack" in a
#      user-visible string is invisible in review and permanent in the product.
#   2. Keep the rebrand from over-reaching — the compatibility surfaces that
#      deliberately keep their upstream spelling (env prefix, wire headers,
#      metric prefix, package and data directories, Kubernetes resource names)
#      must still be there. Renaming one of those does not improve the brand, it
#      breaks a deployed environment, and it breaks quietly: an unread
#      GPUSTACK_DATABASE_URL falls back to a default rather than raising.
#
# It also checks the Apache-2.0 obligations that are easy to lose while editing
# READMEs, because losing them is not a cosmetic regression — §4 breaches
# terminate the license.
#
# Usage:  hack/check-brand.sh          (exit non-zero on any finding)

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT_DIR}"

failures=0
note() { printf '  %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; failures=$((failures + 1)); }
ok()   { printf 'ok:   %s\n' "$1"; }

# ---------------------------------------------------------------------------
# 1. Attribution and license files (Apache-2.0 §4(a), (b), (c))
# ---------------------------------------------------------------------------
echo "== license and attribution =="

for f in LICENSE NOTICE CHANGES.md; do
  if [[ -f "${f}" ]]; then ok "${f} present"; else fail "${f} is missing"; fi
done

# The license text must be the unmodified Apache-2.0, and must still carry the
# upstream copyright line in its appendix.
if grep -q "Apache License" LICENSE && grep -q "Copyright (c) 2024 The GPUStack authors" LICENSE; then
  ok "LICENSE retains the upstream copyright notice"
else
  fail "LICENSE lost the Apache text or the upstream copyright line"
fi

if grep -q "Copyright (c) 2024-2026 The GPUStack authors" README.md; then
  ok "README retains the upstream copyright line"
else
  fail "README lost 'Copyright (c) 2024-2026 The GPUStack authors' (§4(c))"
fi

if grep -qi "modified from\|derivative work\|基于.*修改" README.md; then
  ok "README states that this is a modified work (§4(b))"
else
  fail "README does not say the work was modified (§4(b))"
fi

# ---------------------------------------------------------------------------
# 2. Compatibility surfaces must still carry their upstream spelling
# ---------------------------------------------------------------------------
echo
echo "== compatibility surfaces intact =="

expect() { # <description> <file> <pattern> <minimum count>
  local desc="$1" file="$2" pattern="$3" min="$4"
  local found
  found="$(grep -cE "${pattern}" "${file}" 2>/dev/null || true)"
  if [[ "${found:-0}" -ge "${min}" ]]; then
    ok "${desc} (${found})"
  else
    fail "${desc}: expected >= ${min} in ${file}, found ${found:-0}"
  fi
}

expect "GPUSTACK_ env prefix"          gpustack/envs/__init__.py   'GPUSTACK_'                 50
expect "X-GPUStack-* wire headers"     gpustack/api/auth.py        'X-GPUStack-'                4
expect "gpustack: metric prefix"       gpustack/utils/name.py      'METRIC_PREFIX = "gpustack:"' 1
expect "gpustack data dir"             gpustack/config/config.py   'app_name = "gpustack"'      1
expect "gpustack-chart static path"    gpustack/k8s/chart.py       'gpustack-chart\.tgz'        1
expect "gpustack_* cookie names"       gpustack/api/auth.py        'gpustack_(oidc_state|sso_login)' 2

# ---------------------------------------------------------------------------
# 3. No new hardcoded brand strings in user-visible Python
# ---------------------------------------------------------------------------
echo
echo "== no hardcoded brand in user-visible code =="

# branding.py is the single source of truth; everything else must import it.
# The scan is a Python program rather than a grep pipeline, because the
# distinction that matters is not expressible in grep: a brand name in a *string
# literal* reaches a user (a log line, an error message, --help output, an HTML
# fragment in an API response), while the same name in a comment or docstring
# reaches only a developer. Grepping both returns a wall of docstrings, and the
# person reading it starts ignoring the output — which is how a real leak
# survives. hack/check_brand.py walks tokens and reports strings only.
# Prefer the project's interpreter so the result does not depend on which python
# CI happens to have. It matters more than it looks: PEP 701 (3.12) retokenized
# f-strings, and a checker that only inspects STRING tokens silently skips every
# brand string inside an f-string on 3.12 while reporting it on 3.9. The checker
# handles both, but a stable interpreter keeps the two from ever disagreeing.
if [[ -x ".venv/bin/python" ]]; then
  BRAND_PY=".venv/bin/python"
else
  BRAND_PY="$(command -v python3)"
fi
if "${BRAND_PY}" hack/check_brand.py gpustack; then
  ok "no user-visible brand strings outside gpustack/branding.py"
else
  fail "user-visible brand strings remain (see above)"
fi

# ---------------------------------------------------------------------------
# 4. No upstream endpoints left in shipped code
# ---------------------------------------------------------------------------
echo
echo "== no upstream endpoints =="

# branding.py may name the upstream repository for attribution; nothing else
# should point a running product at an upstream service. Two classes of
# exception, both deliberate:
#   * the operator chart repository — a fetch location for `helm dependency
#     update`, so changing it breaks the build rather than the brand;
#   * CRD API groups and label keys under `*.gpustack.ai`, which are part of the
#     Kubernetes contract with the operator.
endpoints="$(
  grep -rn "gpustack\.ai" gpustack/ charts/ pack/ \
    --include="*.py" --include="*.yaml" --include="*.sh" --include="Dockerfile" 2>/dev/null \
    | grep -v "^gpustack/branding.py:" \
    | grep -v "^gpustack/migrations/" \
    | grep -vE "docs\.gpustack\.ai/gpustack-operator/charts" \
    | grep -vE "[a-z]\.gpustack\.ai|gpustack\.ai/|gpustack\.ai\"" \
    || true
)"
if [[ -n "${endpoints}" ]]; then
  fail "references to upstream endpoints remain:"
  printf '%s\n' "${endpoints}" | head -10 | sed 's/^/      /'
  note "(the operator chart repo and CRD API groups are allowed: contracts, not branding)"
else
  ok "no upstream endpoints (chart repo and CRD groups excluded as contracts)"
fi

# The chart must not advertise an upstream-hosted logo.
if grep -q "^icon:.*gpustack\.ai" charts/gpustack-chart/Chart.yaml; then
  fail "chart icon still points at an upstream-hosted logo"
else
  ok "chart icon does not point upstream"
fi

# ---------------------------------------------------------------------------
# 4b. The console bundle is a named, overridable dependency
# ---------------------------------------------------------------------------
echo
echo "== console bundle source is a knob, not a literal =="

# The web console is not built by this repository: `make install` downloads a
# published tarball. Whoever controls that URL controls the product users
# actually see, so a rebrand that leaves it hardcoded is a rebrand that never
# ships. The default still points upstream — that is the only published build —
# but it must be reachable by one variable, in both installers, or the fork has
# no way to ship its own console without editing build scripts under a deadline.
for installer in hack/install.sh hack/windows/install.ps1; do
  if grep -q "UI_RELEASE_BASE_URL" "${installer}"; then
    ok "${installer} reads the console bundle location from UI_RELEASE_BASE_URL"
  else
    fail "${installer} hardcodes where the web console is downloaded from"
  fi
done

# `|| true` on both substitutions, and not only as belt-and-braces: `grep -c`
# exits 1 when a file has no match, and `find` exits 1 when the directory is gone.
# Under `set -o errexit` that aborts the script *before* the failure branch runs,
# so the regression these two checks exist to catch — a bucket literal deleted, an
# asset directory removed — would terminate the guard silently, printing no FAIL
# and skipping every later section. The exit code would still be non-zero, which
# is exactly what makes it dangerous: red, unexplained, and covering less.
bucket_literals="$(grep -rc "gpustack-ui-1303613262" hack/install.sh hack/windows/install.ps1 2>/dev/null | awk -F: '{s+=$2} END {print s+0}' || true)"
if [[ "${bucket_literals}" == "2" ]]; then
  ok "the upstream bucket appears once per installer, as the default value only"
else
  fail "expected the upstream bucket literal exactly twice (one default per installer), found ${bucket_literals}"
fi

# ---------------------------------------------------------------------------
# 6. Operator-facing docs: brand strings pinned to the reviewed set
# ---------------------------------------------------------------------------
# The four areas an operator actually follows were rebranded (installation,
# upgrade, cli-reference, environment-variables). Two occurrences remain, both
# deliberate and both about someone else's property:
#
#   * the link to the upstream repository that maintains the docker-compose
#     files — renaming its link text would claim that repository as ours;
#   * a shell comment inside a code fence naming the chart it pins.
#
# The count is pinned rather than zeroed: more means un-reviewed prose arrived
# with an upstream merge, fewer means the debt was paid and this expectation
# should be tightened to 1, then 0. The rest of docs/ is out of scope on purpose
# — a blanket rewrite of prose that mixes product names with configuration
# contracts is the riskiest possible pass over this repository.
echo
echo "== operator-facing docs rebranded, remainder pinned =="

doc_scope=(docs/installation docs/upgrade docs/cli-reference docs/environment-variables.md)
docs_hits="$(grep -ro 'GPUStack' "${doc_scope[@]}" 2>/dev/null | wc -l | tr -d ' ' || true)"
if [[ "${docs_hits:-0}" == "2" ]]; then
  ok "installation/upgrade/cli-reference/environment-variables: 2 upstream mentions, both reviewed"
else
  fail "expected 2 remaining 'GPUStack' mentions in the operator docs, found ${docs_hits:-0}"
  note "review each one: a contract token must survive, prose must not"
  grep -rn 'GPUStack' "${doc_scope[@]}" 2>/dev/null | head -5 | sed 's/^/      /' || true
fi

# The env-var prefix and the CLI names those pages document must still be the
# upstream spelling, or the documentation now describes commands that do not
# exist. That is the failure mode a documentation rebrand produces, and it is
# silent: nothing tests a prose file.
#
# `-r` because half the scope is directories — without it grep refuses a
# directory argument, and the check reports a missing token that is sitting right
# there. `-e` because a pattern may begin with `-`, which grep would otherwise
# read as an option. Both failure modes were observed while writing this block.
for token in 'GPUSTACK_RUNTIME_DOCKER_RESOURCE_INJECTION_POLICY' 'gpustack copy-images' 'gpustack-chart' '--ota-server-url'; do
  if grep -rq -e "${token}" "${doc_scope[@]}" 2>/dev/null; then
    ok "docs still document '${token}' verbatim"
  else
    fail "docs no longer mention '${token}': a rebrand rewrote a contract token"
  fi
done

# ---------------------------------------------------------------------------
# 7. Third-party marks untouched
# ---------------------------------------------------------------------------
echo
echo "== third-party assets untouched =="

# These are other companies' trademarks, reproduced to identify compatible
# hardware and model families. A rebrand must not replace or restyle them.
count_dir() { # <description> <dir> <expected>
  local desc="$1" dir="$2" expected="$3" found
  found="$(find "${dir}" -type f 2>/dev/null | wc -l | tr -d ' ' || true)"
  if [[ "${found}" -eq "${expected}" ]]; then
    ok "${desc} (${found} files)"
  else
    fail "${desc}: expected ${expected} files, found ${found}"
  fi
}
count_dir "docs/assets/logos (chip vendors)"  docs/assets/logos    9
count_dir "static/catalog_icons (model pubs)" static/catalog_icons 47

# ---------------------------------------------------------------------------
echo
if [[ "${failures}" -gt 0 ]]; then
  echo "brand guard: ${failures} finding(s)"
  exit 1
fi
echo "brand guard: all checks passed"
