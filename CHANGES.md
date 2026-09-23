# Changes from the upstream work

This file exists to satisfy Apache License, Version 2.0 §4(b), which requires
that modified files carry prominent notices stating that the work was changed.
It is a summary; the authoritative record is the git history of this repository,
which is preserved in full rather than squashed, so every change below can be
read as a diff.

| | |
| --- | --- |
| Derivative work | **OriginHub** |
| Upstream project | [GPUStack](https://github.com/gpustack/gpustack) (Apache-2.0) |
| Upstream base commit | `80ff3ac6121ba7f0ea0332b64f751280206b43e4` |
| Upstream describe at fork point | `v2.3.0rc1` |
| Original copyright | Copyright (c) 2024 The GPUStack authors; Copyright (c) 2024-2026 The GPUStack authors |
| Upstream license text | Retained unmodified in [LICENSE](./LICENSE) |

To reproduce the full set of modifications against the upstream base:

```bash
git diff 80ff3ac6121ba7f0ea0332b64f751280206b43e4...HEAD
git log --stat 80ff3ac6121ba7f0ea0332b64f751280206b43e4..HEAD
```

---

## 1. Brand replacement (OriginHub)

The product name shown to users was changed from GPUStack to OriginHub. This is
a naming change only; it does not alter behaviour.

* Added `gpustack/branding.py` as the single source of truth for user-visible
  brand strings and for the upstream attribution, so that a future upstream
  merge conflicts in one file rather than in hundreds.
* Rewired user-visible strings to it: the FastAPI application title (which drives
  `/docs`, `/redoc` and `openapi.json`'s `info.title`), CLI help and description
  text, the version log line, one API error message, and the generated
  Prometheus configuration comment.
* Packaging metadata now declares the license and names both the original and
  the current authors (`pyproject.toml`), and the container image carries OCI
  labels identifying vendor and license (`pack/Dockerfile`).
* Added `NOTICE` and this file.
* The Helm chart's `icon` field no longer points at an upstream-hosted logo.

### Deliberately **not** renamed

The following keep their upstream spelling on purpose. They are compatibility
surfaces rather than branding, and renaming any of them breaks deployed
environments instead of improving them. The reasoning per item is in
`docs/prd/15-品牌重塑与合规指南.md` §3.2 of the accompanying product
documentation, which lives outside this repository.

* The `gpustack/` Python package directory and every import path.
* The `GPUSTACK_` environment variable prefix — 128 distinct names read in the
  Python sources:

  ```sh
  grep -rhoE 'GPUSTACK_[A-Z0-9_]+' gpustack/ --include='*.py' | sort -u | wc -l
  ```

  They are read by explicit `os.getenv` calls, so renaming makes existing
  deployments silently fall back to defaults rather than fail: an unread
  `GPUSTACK_DATABASE_URL` does not raise, it connects somewhere else.
* The `X-GPUStack-*` request headers — a wire protocol with the wasm plugin
  shipped inside the separate `gpustack-higress-plugins` distribution.
* The `gpustack:` Prometheus metric prefix, the `/var/lib/gpustack` data
  directory, the `gpustack_*` cookie names, the `gpustack-system` namespace, the
  `gpustack-*` gateway plugin names and the `gpustack-chart` chart name.
* The dependency names `gpustack-runtime` and `gpustack-higress-plugins`.

Database table names contain no brand string and were not changed, so no data
migration accompanies the rebrand.

### Behaviour change: update check disabled by default

`gpustack/server/update_check.py` previously defaulted to
`https://update-service.gpustack.ai`. It now defaults to no URL, which makes the
check a no-op returning the running version. A rebranded deployment should not
report its version to the upstream project's service. Operators who run their own
update service can still set `--update-check-url`, and `--disable-update-check`
remains the explicit off switch.

### Behaviour change: the OFFICIAL model catalog is not polled by default

`gpustack/server/sources/probe.py` defaulted to the upstream OTA server, which
populates the OFFICIAL model-catalog rows. It now defaults to no server, and
`_refresh_official` returns early instead of logging a failure every tick, so a
fresh deployment sees only the catalog content shipped with the release. This is
a product behaviour change, not only a rebrand: restoring the old behaviour
means setting `--ota-server-url` (or `GPUSTACK_OTA_SERVER_URL`) to a server this
fork actually operates. It is listed here because it needs a product decision, and
`git diff` will not surface it.

### Why these two are switched off rather than removed

Removing the code was considered and rejected. From a user's perspective the two
outcomes are the same — no automatic update notification, no OFFICIAL catalog
refresh — but they differ in what they cost later:

* The switches already exist (`--update-check-url`, `--ota-server-url`, and their
  `GPUSTACK_*` variables), so a deployment that operates its own service keeps
  working by configuration alone. Deleting the code deletes that option, and
  restoring it means writing the feature again.
* Both are upstream modules. Deleting them guarantees a conflict on every future
  merge in files that otherwise merge cleanly, which is the cost this fork's
  brand work is specifically arranged to avoid.
* Neither sends anything while it is off. There is no traffic to remove, no
  scheduled task to cancel, no data to clean — the only thing deletion would
  change is what remains possible.

With the upstream address gone the OFFICIAL source rows simply stop being
refreshed and the console shows the catalog content that shipped with the
release; there is no error surfaced to an administrator, by design. That is
deliberate: a disabled feature should look unused, not broken. The debug-level
log line in `gpustack/server/sources/probe.py` names the flag to set, so the
behaviour is discoverable without being noisy.

### The web console is downloaded, not built

This repository does not compile the console: `make install` fetches a published
tarball from a CDN bucket (and `hack/windows/install.ps1` does the same), which
means the console a user sees in the browser is whatever that bucket serves — with
its own logo, links and update feed, independent of this repository's source.

Both installers now read the location from `UI_RELEASE_BASE_URL` instead of
inlining it, so the fork's own console build can be substituted with one
variable. The default still points at the upstream bucket deliberately: that is
the only published artifact today, and pointing the variable at a bucket with
nothing in it produces a broken install rather than an upstream-branded one.
Repointing it is a release-engineering decision that has to be made together
with publishing the console, which is why it is recorded here rather than
changed.

`hack/check-brand.sh` asserts the variable exists in both installers and that the
bucket literal appears exactly once per file, as a default value — the coupling
can be reintroduced as a hardcode, and that is the failure worth a machine check.

**Until this fork publishes console artifacts, `hack/use-local-ui.sh` is the way
to ship the rebranded console.** It takes a locally built console directory and
unpacks it into `gpustack/ui`, which is the local half of `download_ui()`; you
then run `UI_DOWNLOAD=false make install` so the steps that are not part of the
console bundle (repository static files, the packaged helm chart) still run. It
validates that the directory it was given is actually a console build — an
`index.html` must be present — and reports how many installed files still
mention the upstream name, which is the check that distinguishes "a rebranded
build" from "a directory that happens to be a build". Verified by running it
against a fixture, not by reading it: argument validation, a missing
`index.html`, the `--clean` path (a stale file left behind by an earlier bundle
is removed), the refusal to copy a directory into itself, and the count at 0 and
at 1.

Running it also settled a question that reading it would not have. It originally
sourced `hack/lib/init.sh` for its logging helpers, like every other script in
this directory, and the first real run produced **4406 lines of output** for what
should have been six. The cause was not the script: `init.sh` calls
`gpustack::version::get_version_vars` at load time, and on a dirty working tree
that function prints the repository's entire `git status` and `git diff`
(`hack/lib/version.sh:69-76`) to explain why the version degrades to `v0.0.0`.
Any hack script does this — it was simply invisible until a script produced
little enough output to read. This one needs neither the version nor the lib's
logging, so it logs with `printf` and stays quiet; the neighbours are unchanged,
since changing shared version handling is not this change's business.

### The rebranded console is now what this repository serves

Done, not proposed. The console was built from this fork's sources and installed
into `gpustack/ui`:

```sh
# in gpustack-ui
pnpm install --frozen-lockfile
pnpm run check:locales        # All keys are consistent!
pnpm run build                # Compiled successfully in 3.90m
# here
hack/use-local-ui.sh /path/to/gpustack-ui/dist
UI_DOWNLOAD=false CHART_PACKAGE=false make install
```

`gpustack/ui/index.html` now carries `<title>OriginHub</title>`, `gpustack/ui/static`
holds 152 files including the 47 catalog icons, and the served console is the
fork's build rather than the tarball from the upstream bucket.

Evidence that it is really being served, rather than merely present:
`tests/routes/test_ui_static.py::test_static_mount_is_not_shadowed_by_the_docs_mount`
skips itself with "UI assets not downloaded; create_app requires gpustack/ui".
It no longer skips. The suite moves from `19 failed, 4379 passed, 71 skipped` to
**`19 failed, 4380 passed, 70 skipped`** — the same 19 failures
(`tests/k8s/test_bootstrap_manifest.py`, documented in section 4) and one test
that went from unverifiable to passing.

`CHART_PACKAGE=false` because `helm` is not installed on this machine, so
`gpustack/ui/static/charts/gpustack-chart.tgz` was not produced — which is
exactly what keeps those 19 tests failing. `brew install helm` followed by
`UI_DOWNLOAD=false make install` closes that, and would be the first time this
checkout has run the chart-packaging step end to end.

### Guard is wired into the build

`hack/check-brand.sh` (plus its Python-side companion `hack/check_brand.py`) runs
from `hack/lint.sh`, which `make ci` invokes — so it runs on every pull request
via `.github/workflows/ci.yml` and `pr.yml`. The tokenizer half,
`hack/check_brand.py`, is additionally registered as a `repo: local` pre-commit
hook with `always_run: true`, stdlib-only, so a developer sees a brand string in a
user-visible literal before committing it. The shell half is deliberately not a
pre-commit hook: it is bash, and registering it would break `git commit` on
Windows. There is no path to a green pipeline that does not pass the full guard.

## 2. Billing engine (new subsystem)

A complete prepaid billing subsystem was added. It is new code rather than a
modification of upstream behaviour: the existing usage-collection pipeline
(`model_usage_details`, `metered_usage`) is consumed unchanged, and no upstream
table or endpoint was altered in shape.

* Eight new tables under the `billing_*` prefix, with hand-written Alembic
  migrations (`2026_09_22_1000-a7f3c9e21b40`, `2026_09_23_1000-c5a9f31d7e20`,
  `2026_09_23_1200-d8e5b21c4f90`).
* New modules: `server/billing_pricing.py` (price resolution and invariants),
  `server/billing_rater.py` (usage → ledger), `server/billing_settlement.py`
  (wallets, settlement, redemption, manual adjustments),
  `server/billing_enforcement.py` (suspension reaching the request path),
  `server/billing_quota.py` (ceilings and the pre-request gate),
  `server/billing_invoice.py` (period statements), `server/billing_alerts.py`
  (deduplicated alerts), `exporter/billing_metrics.py` (Prometheus collector).
* New routes under `/billing/*` and one tenant-facing pair; new columns
  `api_keys.suspended` and `api_keys.suspension_reason`.
* Existing files touched, and how:
  * `server/server.py` — three new leader-only background loops registered
    alongside the existing ones.
  * `server/gateway_auth_reconciler.py` — suspended keys are excluded from the
    gateway's local auth table, and a key with an active quota loses the
    `unrestricted` bit so the gateway asks the server. Both follow the file's
    existing exclusion patterns; neither changes the plugin contract.
  * `routes/token.py`, `routes/openai.py` — a suspension check and a quota check
    added to the authorization path, answering 402 and 429 respectively.
  * `routes/api_keys.py` — a new key inherits its wallet's suspension.
  * `api/exceptions.py` — added 402 and 429 exception types.
  * `envs/__init__.py` — added `GPUSTACK_BILLING_*` settings, which follow the
    existing prefix rather than introducing a new one.
* A master switch, `GPUSTACK_BILLING_MODE`, defaults to `shadow`: the ledger is
  written but no wallet is ever debited. Nothing in this subsystem can charge a
  tenant until an operator explicitly sets it to `enforce`.
* Verification tooling under `hack/billing/`: a shadow-rating drill, a
  reconciliation tool and a rating-pipeline stress run, each exit-code gated.
* Design and operational documentation lives outside this repository, in
  `docs/prd/07`, `12`, `13`, `14` of the accompanying product documentation.

## 3. Tests

New tests were added under `tests/server/`, `tests/routes/`, `tests/schemas/`
and `tests/exporter/`. Existing test files modified to keep their fixtures or
assertions aligned with the new behaviour:

* `tests/gateway/test_ext_auth.py` — its fake session answered every query with
  the same rows; it now distinguishes the quota-subject query from the key query,
  and the row shape gained the `owner_principal_id` column the reconciler reads.
* `tests/server/test_billing_rater.py` — one expectation was corrected: usage
  reporting more cached tokens than prompt tokens is now clamped to the prompt
  count instead of being billed as reported, because billing the larger figure
  charges for tokens the request's own accounting says do not exist.
* `tests/routers/test_models.py` — assertions on the CSV export header and on an
  alert string now read `branding.PRODUCT_NAME` instead of a literal product name.
  They were asserting the *word*, so a rename failed them while proving nothing
  about the behaviour under test (that the header carries the product name, and
  that the alert names the caller's own product).
* `tests/routes/test_source_routes.py` — the official-source tests no longer read
  the shipped OTA default; they patch `probe_module.OTA_SERVER_URL` to a test
  address, so they assert which address resolves to the OFFICIAL slot rather than
  what a release happens to point at. Reviewing this file surfaced a defect in the
  test itself: one `monkeypatch` had been written *after* the call whose behaviour
  it was supposed to control, so it was a no-op and the assertion passed for the
  wrong reason. It now runs before the call.

## 4. Known upstream test failures

`tests/k8s/test_bootstrap_manifest.py` fails in a checkout that has not run
`hack/install.sh`, because it asserts on `gpustack/ui/static/charts/gpustack-chart.tgz`
— a build artifact produced by that script's chart-packaging step, not a source
file. This is pre-existing upstream behaviour and is unrelated to the changes
above.

## 5. Verification of the rebrand

    uv run pytest -q            →  19 failed, 4380 passed, 70 skipped
    bash hack/lint.sh           →  exit 0: all 8 pre-commit hooks pass, then the
                                   full guard runs
    bash hack/check-brand.sh    →  25 checks pass

The 19 failures are the pre-existing `tests/k8s/test_bootstrap_manifest.py` set
from section 4. The passed/skipped pair moved by one (`4379/71` → `4380/70`)
because the console is now installed in this checkout and a UI-serving test that
used to skip itself runs — see "The rebranded console is now what this repository
serves" above. A targeted run of the billing suites (81 tests) also passes,
which is what makes the formatting churn in section 6 safe to keep.

## 6. Tooling conflict found while verifying (not caused by the rebrand)

`.pre-commit-config.yaml` pins black `24.4.2` while the project venv resolves
black `26.5.1`. The billing work had been formatted by the newer binary, so
`pre-commit run --all-files` — the command CI runs — reformatted 26 of those
files, i.e. **`make lint` was red on this branch before any of the rebrand work**,
for a reason unrelated to it.

Two decisions taken here, both reversible:

* The pinned formatter's output was kept rather than reverted, because CI runs the
  pinned version and the venv version is not what gates a merge. The 26 files are
  formatting-only; they belong in their own commit, not in a branding commit.
* Four flake8 findings in the same pre-existing billing code were fixed so that
  the lint gate can actually gate: an unused unpacked variable in
  `gpustack/server/billing_alerts.py` (the count column is selected but never
  reported, so it is no longer bound), and `C901` complexity in three
  `hack/billing/` drill scripts, exempted by a commented `per-file-ignores` entry
  in `.flake8` — their branching is the workload, and they ship in no
  distribution. Every importable module is still held to the limit.

`hack/lint.sh` now runs both the style checks and the brand guard and aggregates
their exit status. It previously ran under `set -o errexit` with lint first, so
any flake8 complaint — including the four above — meant the brand guard never
executed. A licence obligation checked only when nothing else is broken is not
checked.

The version mismatch itself is upstream's configuration and was not changed:
bumping the pinned black would reformat a large share of the repository and bury
the rebrand diff. It is recorded here so the next person to hit it does not
assume the rebrand did it.

## 7. Review follow-ups

A review pass over the uncommitted changes found four things worth fixing, and
one proposed fix that turned out to be wrong. Both categories are recorded, since
the second is the more expensive kind.

### Fixed

* **The Japanese README had kept the upstream product name in nine places.** The
  earlier pass used a Unicode-aware `\w` inside a negative lookahead, so
  `GPUStackは` and `GPUStackの` — a brand name followed directly by a Japanese
  particle — suppressed their own replacement. English and Chinese were fine
  because a space followed. The same bug class had already been fixed once in the
  console's locale files; this is the second instance, which is why the rule is
  now "use ASCII character classes for ASCII brand names".
* **Three quick-start config comments** still said "... GPUStack server URL". Those
  blocks are copied by users, so they are user-facing text, not comments.
* **The community sections advertised the upstream project's channels** — a
  Discord invite in English and Japanese, an upstream WeChat group QR code in
  Chinese. All three now point at this fork's issue tracker, matching the choice
  already made in the console (`externalLinks.discord` defaults to empty, for the
  same reason: an invitation to someone else's community is worse than none).
* **Ten relative documentation links 404'd.** The earlier pass rewrote absolute
  `docs.gpustack.ai` links into repository-relative ones and dropped the `.md`
  (`./docs/installation/requirements/`), which GitHub cannot resolve — replacing
  a working upstream link with a broken local one.
* **`USER_AGENT` had lost its version.** It read `ORIGINHUB_VERSION`, which
  nothing in this repository sets, so every outbound request announced a bare
  `originhub`; the previous code sent `gpustack/<version>`. It now defaults to
  the package's own `__version__`, with the variable kept as an override.
* **The guard's own failure branches were unreachable.** Two command
  substitutions (`grep -c` for the bucket literal, `find` for the asset counts)
  exit non-zero on exactly the condition they are meant to detect — no match, no
  directory — and under `set -o errexit` that terminated the script before the
  `fail` branch ran. The regression would have gone unreported and every later
  section skipped, with a non-zero exit code and no explanation. Demonstrated
  both ways in the shell before and after the fix.
* **`branding.py` said 276 environment variables** where the reproducible count
  is 128, and pointed at a `docs/prd/` path that does not exist inside this
  repository (the product documentation lives outside it).

### A proposed fix that was wrong, and reverted

The review reported that the new `process.env.ORIGINHUB_*` webpack `define`
values were missing `JSON.stringify`, on the grounds that webpack splices define
values in as *code*. That is true of webpack, and false of this project: umi
stringifies every define value itself — `define[key] = JSON.stringify(
userConfig.define[key])` in `@umijs/bundler-webpack/dist/config/definePlugin.js`.

It was applied, and a build with `ORIGINHUB_COMPANY="Acme Ltd"` proved it wrong:
the bundle carried `o='"Acme Ltd"'`, with the quotes as part of the value, and the
unset case would have become the two-character string `""` instead of falling
back to the product name. Reverted, and the mechanism is now recorded in
`config/config.ts`, in the console's `CHANGES.md`, and in a guard assertion
(`scripts/check-brand.cjs`) that checks the three variables carry raw values and
that the links blob — the one deliberate exception, because it is parsed at
runtime — keeps its stringify.

The lesson generalises: a reproduction built on the underlying library is not a
reproduction of this project. The reviewer's harness used webpack directly, and
webpack's behaviour was not the behaviour in play.
