"""Small, extensible capability catalog for SignalMap method modules.

The catalog only enforces hard technical preconditions.  Scientific strength
is represented by the evidence status and must be established by receipts;
heuristics may prioritize a method but never silently disqualify it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .qualification import SourceProfile

EVIDENCE_STATUSES = ("candidate", "speculative", "confirmed")
_COST_ORDER = {"low": 0, "medium": 1, "high": 2, "unknown": 3}
_CORE_REQUIREMENTS = {"channels", "min_samples", "min_recordings", "finite_fraction"}


@dataclass(frozen=True)
class MethodCapability:
    name: str
    family: str
    status: str = "candidate"
    requires: dict[str, Any] = field(default_factory=dict)
    cost: str = "unknown"
    evidence_required: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError(f"unsupported evidence status {self.status!r}")
        if self.cost not in _COST_ORDER:
            raise ValueError(f"unsupported method cost {self.cost!r}")
        unknown = set(self.requires) - _CORE_REQUIREMENTS
        if unknown:
            raise ValueError(f"unsupported hard requirement(s): {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "status": self.status,
            "requires": dict(self.requires),
            "cost": self.cost,
            "evidence_required": list(self.evidence_required),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MethodCapability":
        return cls(
            name=str(data["name"]), family=str(data["family"]),
            status=str(data.get("status", "candidate")),
            requires=dict(data.get("requires", {})),
            cost=str(data.get("cost", "unknown")),
            evidence_required=tuple(data.get("evidence_required", ())),
            extra=dict(data.get("extra", {})),
        )


class MethodCatalog:
    def __init__(self, methods: list[MethodCapability] | tuple[MethodCapability, ...] = ()):
        names = [m.name for m in methods]
        if len(names) != len(set(names)):
            raise ValueError("duplicate method capability name")
        self._methods = tuple(methods)

    @property
    def methods(self) -> tuple[MethodCapability, ...]:
        return self._methods

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "methods": [m.to_dict() for m in self._methods]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MethodCatalog":
        if data.get("version") != 1:
            raise ValueError("unsupported method catalog version")
        return cls(tuple(MethodCapability.from_dict(m) for m in data.get("methods", ())))


@dataclass(frozen=True)
class CapabilityDecision:
    eligible: tuple[str, ...]
    hard_excluded: tuple[str, ...]
    reasons: dict[str, str]
    priorities: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": list(self.eligible),
            "hard_excluded": list(self.hard_excluded),
            "reasons": dict(self.reasons),
            "priorities": dict(self.priorities),
        }


def _hard_failure(profile: SourceProfile, method: MethodCapability) -> str | None:
    metrics = profile.metrics
    req = method.requires
    if "channels" in req and int(metrics.get("channels", 0)) < int(req["channels"]):
        return f"requires_channels>={int(req['channels'])}"
    if "min_samples" in req and int(metrics.get("samples_per_window", 0)) < int(req["min_samples"]):
        return f"requires_samples>={int(req['min_samples'])}"
    if "min_recordings" in req and int(metrics.get("recordings", 0)) < int(req["min_recordings"]):
        return f"requires_recordings>={int(req['min_recordings'])}"
    if "finite_fraction" in req and float(metrics.get("finite_fraction", 0.0)) < float(req["finite_fraction"]):
        return f"requires_finite_fraction>={float(req['finite_fraction']):g}"
    return None


def route_capabilities(profile: SourceProfile, catalog: MethodCatalog) -> CapabilityDecision:
    """Route methods by hard preconditions, then rank eligible methods by cost.

    Noise or weak signal diagnostics are intentionally not exclusion rules.
    They belong in future module-specific priority functions.
    """
    eligible: list[MethodCapability] = []
    excluded: list[str] = []
    reasons: dict[str, str] = {}
    for method in catalog.methods:
        failure = _hard_failure(profile, method)
        if failure:
            excluded.append(method.name)
            reasons[method.name] = failure
        else:
            eligible.append(method)
    eligible.sort(key=lambda m: (_COST_ORDER[m.cost], m.status, m.name))
    names = tuple(m.name for m in eligible)
    return CapabilityDecision(
        eligible=names,
        hard_excluded=tuple(excluded),
        reasons=reasons,
        priorities={name: i for i, name in enumerate(names)},
    )
