"""Prometheus metrics for billing, pulled from the alert registry's snapshot.

Nothing here touches the database. ``/metrics`` is served from a scrape thread on
an interval Prometheus controls, so a query in this path would make every billing
table's latency a scrape-latency problem — and a slow scrape is reported as the
exporter being down, which sends an operator looking in the wrong place. Instead
the leader-only ``BillingAlertDetector`` refreshes a snapshot each scan and this
collector only reads it.

The consequence is worth knowing: these numbers lag by up to one detector
interval (``GPUSTACK_BILLING_ALERT_INTERVAL_SECONDS``, 300s by default). That is
the right trade for every condition here — an unpriced SKU, an invoice unpaid for
days, a wallet suspended for arrears — none of which is a sub-minute phenomenon.
Before the first scan the gauges are absent rather than zero, so a dashboard can
tell "nothing is wrong" from "the detector has not run".
"""

from typing import Dict, Iterator

from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from gpustack.server.billing_alerts import (
    BillingAlertKind,
    billing_alerts,
)
from gpustack.utils.name import metric_name

# Snapshot keys the detector publishes, with the help text a dashboard reader
# needs. Kept next to the collector rather than the detector because the wording
# is part of the metric's contract, and this is where the contract is published.
_GAUGES = (
    (
        "unpriced_skus",
        "Number of SKUs with usage the rater could not price. Usage under these "
        "is accruing unbilled; add a price-book entry to clear it.",
    ),
    (
        "unpriced_entries",
        "Number of ledger placeholders (VOID) standing for usage that has no "
        "price. They are retried every rating sweep until priced.",
    ),
    (
        "unpaid_invoices",
        "Invoices issued and unpaid past their grace period. Each suspends its "
        "org until it is collected.",
    ),
    (
        "unpaid_amount",
        "Total currency outstanding on invoices past their grace period.",
    ),
    (
        "suspended_wallets",
        "Wallets suspended for arrears, i.e. orgs whose api keys are refused "
        "with 402 on both request paths.",
    ),
    (
        "pending_realtime_amount",
        "Rated realtime charges not yet collected by the settler. This should "
        "stay near zero: a figure that grows across scrapes means the settlement "
        "loop is behind or failing, which tenants cannot see until it is large.",
    ),
    (
        "pending_deferred_amount",
        "Rated deferred (resource) charges awaiting their period's invoice. "
        "Expected to grow through a period and drop when it closes.",
    ),
)


class BillingMetricsCollector(Collector):
    """Expose billing health as Prometheus metrics, from the registry snapshot."""

    def collect(self) -> Iterator[Metric]:
        # One series per (kind, severity) for active alerts, so a rule can fire on
        # a single kind without summing across unrelated problems, and on a
        # severity without listing kinds.
        active = GaugeMetricFamily(
            metric_name("billing_alerts_active"),
            "Billing problems currently firing, by kind and severity. A problem "
            "is raised once and stays active until the condition clears, so a "
            "sustained 1 is one unresolved problem rather than one per scan.",
            labels=["kind", "severity"],
        )
        raised = GaugeMetricFamily(
            metric_name("billing_alerts_raised"),
            "Cumulative billing alerts raised since process start, by kind. "
            "Monotonic within a process; a restart resets it, which is why "
            "alerting on the active gauge is preferred.",
            labels=["kind"],
        )
        occurrences = GaugeMetricFamily(
            metric_name("billing_alert_occurrences"),
            "Scans that observed each active problem. Rising fast with the alert "
            "still active means a long-running problem, which is the difference "
            "between 'just appeared' and 'unpriced for nine days'.",
            labels=["kind", "key"],
        )
        age = GaugeMetricFamily(
            metric_name("billing_alert_age_seconds"),
            "Age in seconds of each active billing problem, measured from when "
            "this process first observed it.",
            labels=["kind", "key"],
        )

        # Copy before iterating: the detector writes on the event loop while a
        # scrape reads here, and a dict changed mid-iteration would fail the whole
        # scrape rather than one series.
        alerts = billing_alerts.active()
        counts: Dict[tuple, int] = {}
        for alert in alerts:
            counts[(alert.kind.value, alert.severity.value)] = (
                counts.get((alert.kind.value, alert.severity.value), 0) + 1
            )
            occurrences.add_metric([alert.kind.value, alert.key], alert.occurrences)
            age.add_metric([alert.kind.value, alert.key], alert.age.total_seconds())
        for kind in BillingAlertKind:
            for severity in ("warning", "critical"):
                counts.setdefault((kind.value, severity), 0)
        for (kind, severity), count in counts.items():
            active.add_metric([kind, severity], count)

        totals = billing_alerts.raised_total()
        for kind in BillingAlertKind:
            raised.add_metric([kind.value], totals.get(kind.value, 0))

        snapshot = billing_alerts.metrics()
        for key, help_text in _GAUGES:
            if key not in snapshot:
                # Absent, not zero: the detector has not run yet, and a zero here
                # would read as "all clear" on a server that has not scanned.
                continue
            family = GaugeMetricFamily(
                metric_name(f"billing_{key}"), help_text, labels=[]
            )
            family.add_metric([], snapshot[key])
            yield family

        yield active
        yield raised
        yield occurrences
        yield age
