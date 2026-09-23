"""Single source of truth for product branding.

Every user-visible brand string is defined here and nowhere else, so that
re-applying the rebrand after an upstream merge touches one file instead of
hundreds, and so a CI guard (``hack/check-brand.sh``) can forbid new hardcoded
occurrences.

Deliberately **not** here — these are compatibility surfaces, not branding, and
renaming any of them breaks deployed environments rather than improving them
(see ``docs/prd/15-品牌重塑与合规指南.md`` in the accompanying product
 documentation — outside this repository — §3.2 for the reasoning per item):

* the ``GPUSTACK_`` environment variable prefix (128 distinct names, read by
  explicit ``os.getenv`` calls; a rename makes existing deployments silently fall
  back to defaults, which is how a database URL goes missing without an error).
  The count is reproducible:
  ``grep -rhoE 'GPUSTACK_[A-Z0-9_]+' gpustack/ --include='*.py' | sort -u | wc -l``;
* the ``X-GPUStack-*`` request headers (``api/auth.py``), which are a wire
  protocol with the wasm plugin shipped inside the ``gpustack-higress-plugins``
  distribution — renaming one side alone breaks gateway authentication;
* the ``gpustack:`` metric prefix (``utils/name.py``), which every existing
  dashboard and alerting rule is written against;
* the ``gpustack`` package directory, the ``/var/lib/gpustack`` data directory,
  the ``gpustack_*`` cookie names, the ``gpustack-system`` namespace and the
  ``gpustack-*`` gateway plugin names.

Attribution lives beside the brand names on purpose: a rebrand that deletes the
upstream copyright notice is the single easiest way to breach Apache-2.0 §4(c),
and putting the two next to each other makes that hard to do by accident.
"""

import os

# ``gpustack/__init__.py`` holds nothing but version constants, so importing it
# from here cannot create a cycle with the modules that import branding.
from gpustack import __version__

# ---------------------------------------------------------------------------
# Product identity
# ---------------------------------------------------------------------------

#: Display name. Used for the API docs title, CLI help and user-facing messages.
PRODUCT_NAME = "OriginHub"

#: Lowercase slug, for file names and identifiers that need one.
PRODUCT_NAME_LOWER = "originhub"

#: One-line description, for packaging metadata and the docs site.
PRODUCT_TAGLINE = "AI infrastructure and LLM inference serving platform"

# ---------------------------------------------------------------------------
# Endpoints. Empty means "not configured", and every caller must treat empty as
# absent rather than rendering a broken link — these are left blank on purpose
# so a placeholder URL can never ship and quietly point a customer somewhere
# unintended. Fill them in per deployment or at release time.
# ---------------------------------------------------------------------------

DOCS_URL = os.getenv("ORIGINHUB_DOCS_URL", "")
SUPPORT_URL = os.getenv("ORIGINHUB_SUPPORT_URL", "")
SUPPORT_EMAIL = os.getenv("ORIGINHUB_SUPPORT_EMAIL", "")
REPO_URL = os.getenv("ORIGINHUB_REPO_URL", "")

#: Update check endpoint. Empty disables the check, which is the default: a
#: rebranded deployment should not report its version to the upstream project,
#: and the ``UpdateChecker`` treats an empty URL as "nothing to ask". Overridable
#: through the existing ``--update-check-url`` flag for anyone who runs their own
#: update service.
UPDATE_CHECK_URL = os.getenv("ORIGINHUB_UPDATE_CHECK_URL", "")

#: Content source (model catalog, community backend list) published as an OTA
#: bundle. Empty by default, which turns the OFFICIAL source slot into
#: "packaged content only" — see ``server/sources/probe.py``.
#:
#: This one is a functional dependency and not only a brand leak: with it set,
#: the platform keeps fetching its catalog from whoever owns the URL. A
#: rebranded product should point it at a mirror it controls, both so the catalog
#: does not change under a customer without a release, and so the dependency is
#: one the operator chose. Set ``--ota-server-url`` /
#: ``GPUSTACK_OTA_SERVER_URL`` to restore the upstream feed.
OTA_SERVER_URL = os.getenv("ORIGINHUB_OTA_SERVER_URL", "")

#: User-Agent for outbound calls that identify this product.
#:
#: The version comes from the package itself, which a release build rewrites.
#: ``ORIGINHUB_VERSION`` overrides it for a build that wants to say something
#: else. An earlier version of this line read *only* the variable, which nothing
#: in this repository sets, so every outbound request announced a bare
#: ``originhub`` — dropping the version number, which is the first thing anyone
#: asks for when diagnosing an outbound call.
USER_AGENT = f"{PRODUCT_NAME_LOWER}/{os.getenv('ORIGINHUB_VERSION') or __version__}"

# ---------------------------------------------------------------------------
# Upstream attribution — do not remove (Apache-2.0 §4(b), (c))
# ---------------------------------------------------------------------------

UPSTREAM_NAME = "GPUStack"
UPSTREAM_URL = "https://github.com/gpustack/gpustack"
UPSTREAM_LICENSE = "Apache-2.0"
UPSTREAM_COPYRIGHT = "Copyright (c) 2024-2026 The GPUStack authors"

#: The acknowledgement string for user-facing surfaces (an About page, the
#: ``/version`` response, a docs footer). Keeping it as one constant means the
#: attribution cannot drift between surfaces or be dropped from one of them.
ATTRIBUTION = (
    f"{PRODUCT_NAME} is built on the open-source {UPSTREAM_NAME} project, "
    f"licensed under the {UPSTREAM_LICENSE} License. {UPSTREAM_COPYRIGHT}."
)


def docs_link(path: str, text: str) -> str:
    """An HTML link into the documentation, or just the text when there is none.

    For messages rendered in the web console. Falling back to plain text rather
    than emitting an ``href`` that goes nowhere matters: a broken link in an
    error message is worse than no link, because the reader clicks it at the
    exact moment they are trying to fix something.
    """
    if not DOCS_URL:
        return text
    return f"<a href='{DOCS_URL.rstrip('/')}/{path.lstrip('/')}'>{text}</a>"
