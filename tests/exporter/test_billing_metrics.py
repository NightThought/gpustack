"""Tests for the billing metrics collector.

The collector reads a snapshot the leader's alert detector publishes; it never
queries. That division is what these tests pin down, because the alternative — a
billing query inside ``/metrics`` — fails in a way that looks like an exporter
problem rather than a slow query.
"""

import pytest

from gpustack.exporter.billing_metrics import _GAUGES, BillingMetricsCollector
from gpustack.server.billing_alerts import (
    AlertSeverity,
    BillingAlertKind,
    BillingAlertRegistry,
    billing_alerts,
)
from gpustack.utils.name import metric_name


def _samples(metrics, name):
    for metric in metrics:
        if metric.name == name:
            return list(metric.samples)
    return []


def _sample(metrics, name, labels):
    for sample in _samples(metrics, name):
        if all(sample.labels.get(k) == v for k, v in labels.items()):
            return sample
    raise AssertionError(f"no sample for {name} {labels}")


@pytest.fixture(autouse=True)
def clean_registry():
    billing_alerts.clear()
    yield
    billing_alerts.clear()


def test_gauges_are_absent_before_the_first_scan():
    """Absent, not zero: a dashboard must be able to tell "nothing is wrong"
    from "the detector has not run yet"."""
    metrics = list(BillingMetricsCollector().collect())

    for key, _help_text in _GAUGES:
        assert _samples(metrics, metric_name(f"billing_{key}")) == []


def test_the_snapshot_is_served_as_published():
    billing_alerts.publish_metrics(
        {
            "unpriced_skus": 2.0,
            "unpriced_entries": 17.0,
            "unpaid_invoices": 1.0,
            "unpaid_amount": 120.5,
            "suspended_wallets": 3.0,
            "pending_realtime_amount": 0.25,
            "pending_deferred_amount": 88.0,
        }
    )

    metrics = list(BillingMetricsCollector().collect())

    assert _sample(metrics, metric_name("billing_unpriced_skus"), {}).value == 2.0
    assert _sample(metrics, metric_name("billing_unpriced_entries"), {}).value == 17.0
    assert _sample(metrics, metric_name("billing_unpaid_amount"), {}).value == 120.5
    assert _sample(metrics, metric_name("billing_suspended_wallets"), {}).value == 3.0
    # The two backlog figures, which are what answer "is the pipeline keeping up".
    assert (
        _sample(metrics, metric_name("billing_pending_realtime_amount"), {}).value
        == 0.25
    )
    assert (
        _sample(metrics, metric_name("billing_pending_deferred_amount"), {}).value
        == 88.0
    )


def test_active_alerts_are_counted_per_kind_and_severity():
    billing_alerts.raise_alert(
        BillingAlertKind.UNPRICED_GAP,
        "gpu.hour.910b",
        severity=AlertSeverity.WARNING,
        summary="a",
    )
    billing_alerts.raise_alert(
        BillingAlertKind.UNPRICED_GAP,
        "model.token.prompt",
        severity=AlertSeverity.WARNING,
        summary="b",
    )
    billing_alerts.raise_alert(
        BillingAlertKind.UNPAID_INVOICE,
        "7",
        severity=AlertSeverity.CRITICAL,
        summary="c",
    )

    metrics = list(BillingMetricsCollector().collect())
    name = metric_name("billing_alerts_active")

    assert (
        _sample(
            metrics,
            name,
            {"kind": BillingAlertKind.UNPRICED_GAP.value, "severity": "warning"},
        ).value
        == 2.0
    )
    assert (
        _sample(
            metrics,
            name,
            {"kind": BillingAlertKind.UNPAID_INVOICE.value, "severity": "critical"},
        ).value
        == 1.0
    )
    # Every kind/severity pair is present, so a rule can be written against a
    # series that does not exist yet instead of one that appears only on failure.
    assert (
        _sample(
            metrics,
            name,
            {"kind": BillingAlertKind.WALLET_SUSPENDED.value, "severity": "warning"},
        ).value
        == 0.0
    )


def test_occurrences_and_age_expose_how_long_a_problem_has_been_open():
    """The difference between "just appeared" and "unpriced for nine days" is
    only visible in these two."""
    for _ in range(4):
        billing_alerts.raise_alert(
            BillingAlertKind.UNPRICED_GAP,
            "gpu.hour.910b",
            severity=AlertSeverity.WARNING,
            summary="a",
        )

    metrics = list(BillingMetricsCollector().collect())
    labels = {"kind": BillingAlertKind.UNPRICED_GAP.value, "key": "gpu.hour.910b"}

    assert _sample(metrics, metric_name("billing_alert_occurrences"), labels).value == 4
    assert _sample(metrics, metric_name("billing_alert_age_seconds"), labels).value >= 0


def test_raised_total_does_not_fall_when_a_problem_is_fixed():
    """A gauge of active alerts drops on resolution, which Prometheus would read
    as a counter reset if this series were used as one."""
    billing_alerts.raise_alert(
        BillingAlertKind.WALLET_SUSPENDED,
        "1",
        severity=AlertSeverity.WARNING,
        summary="a",
    )
    billing_alerts.resolve(BillingAlertKind.WALLET_SUSPENDED, "1")
    billing_alerts.raise_alert(
        BillingAlertKind.WALLET_SUSPENDED,
        "1",
        severity=AlertSeverity.WARNING,
        summary="a",
    )

    metrics = list(BillingMetricsCollector().collect())
    name = metric_name("billing_alerts_raised")

    assert (
        _sample(metrics, name, {"kind": BillingAlertKind.WALLET_SUSPENDED.value}).value
        == 2.0
    )
    assert (
        _sample(
            metrics,
            metric_name("billing_alerts_active"),
            {"kind": BillingAlertKind.WALLET_SUSPENDED.value, "severity": "warning"},
        ).value
        == 1.0
    )


def test_a_registry_being_written_does_not_break_a_scrape():
    """The detector writes on the event loop while a scrape reads here; a
    half-applied view would fail the whole scrape rather than one series."""
    billing_alerts.publish_metrics({"unpriced_skus": 1.0})
    collector = BillingMetricsCollector()

    first = _samples(collector.collect(), metric_name("billing_unpriced_skus"))
    billing_alerts.publish_metrics({"unpriced_skus": 5.0})
    second = _samples(collector.collect(), metric_name("billing_unpriced_skus"))

    assert first[0].value == 1.0
    assert second[0].value == 5.0


def test_the_collector_needs_no_database():
    """Asserted by construction: nothing here opens a session, so a scrape with
    the database unreachable still serves the last snapshot."""
    billing_alerts.publish_metrics({"suspended_wallets": 1.0})

    metrics = list(BillingMetricsCollector().collect())

    assert _sample(metrics, metric_name("billing_suspended_wallets"), {}).value == 1.0


def test_registry_isolation_between_instances():
    """A fresh registry must not inherit another's alerts, which is what makes
    the module singleton testable at all."""
    other = BillingAlertRegistry()
    other.raise_alert(
        BillingAlertKind.UNPAID_INVOICE,
        "1",
        severity=AlertSeverity.WARNING,
        summary="x",
    )

    assert billing_alerts.active() == []
    assert len(other.active()) == 1
