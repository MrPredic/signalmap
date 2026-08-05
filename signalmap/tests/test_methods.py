import json

import pytest

from signalmap.methods import (
    EVIDENCE_STATUSES,
    MethodCapability,
    MethodCatalog,
    route_capabilities,
)
from signalmap.qualification import SourceProfile


def _profile(channels=1, samples=1024, recordings=6):
    return SourceProfile(
        source_id="s", device_id="d", setup_id="u",
        metrics={
            "channels": channels, "samples_per_window": samples,
            "recordings": recordings, "finite_fraction": 1.0,
        }, signature="sig",
    )


def test_capability_schema_preserves_speculative_extensions():
    method = MethodCapability(
        name="new-envelope", family="envelope", status="speculative",
        requires={"channels": 1, "min_samples": 128}, cost="medium",
        evidence_required=("null", "logo", "stability"),
        extra={"author": "research-track", "custom_gate": {"x": 1}},
    )
    encoded = json.loads(json.dumps(method.to_dict()))
    back = MethodCapability.from_dict(encoded)
    assert back == method
    assert set(EVIDENCE_STATUSES) == {"candidate", "speculative", "confirmed"}


def test_router_hard_excludes_only_impossible_requirements():
    catalog = MethodCatalog([
        MethodCapability("coherence", "coherence", "confirmed", {"channels": 2}),
        MethodCapability("new-one-channel", "novel", "speculative", {"channels": 1}),
        MethodCapability("long-window", "novel", "candidate", {"min_samples": 2048}),
    ])
    decision = route_capabilities(_profile(channels=1), catalog)
    assert "new-one-channel" in decision.eligible
    assert "coherence" in decision.hard_excluded
    assert decision.reasons["coherence"] == "requires_channels>=2"
    assert "long-window" in decision.hard_excluded


def test_router_keeps_speculative_methods_visible_and_prioritizes_cost():
    catalog = MethodCatalog([
        MethodCapability("expensive", "novel", "candidate", cost="high"),
        MethodCapability("cheap-spec", "novel", "speculative", cost="low"),
    ])
    decision = route_capabilities(_profile(), catalog)
    assert decision.eligible == ("cheap-spec", "expensive")
    assert decision.hard_excluded == ()


def test_catalog_rejects_unknown_status_and_duplicate_names():
    with pytest.raises(ValueError, match="status"):
        MethodCapability("x", "x", "promoted")
    with pytest.raises(ValueError, match="duplicate"):
        MethodCatalog([
            MethodCapability("x", "x", "candidate"),
            MethodCapability("x", "y", "candidate"),
        ])
