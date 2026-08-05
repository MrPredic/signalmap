"""Label-free source qualification and deterministic method-family routing.

This module deliberately does not inspect ``Bank.y``.  It describes what a
measurement can support; it does not evaluate whether a method detects a label.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .distill import Bank

PROFILE_VERSION = "1"
_SPECTRAL_POOLS = {"specratio", "speccent", "specflat", "acflag"}


def _program_families(program) -> set[str]:
    """Every family a program depends on. A spectral pool applied to an
    envelope transform needs both, so classifying by the first match would
    run an envelope on a source where envelope was routed out."""
    families = set()
    if program.pool in _SPECTRAL_POOLS:
        families.add("spectral")
    if program.t1 == "env" or program.t2 == "env":
        families.add("envelope")
    return families or {"time_domain"}


def filter_programs(programs, compatible_families: set[str]):
    """Keep only base-grammar programs whose every family was routed compatible.

    Premium-only families (recurrence/coherence/causal_lag) are intentionally
    not synthesized here; their existing opt-in challengers remain separate.
    """
    return [p for p in programs
            if _program_families(p) <= set(compatible_families)]


def _primary(window: np.ndarray) -> np.ndarray:
    return np.asarray(window[0] if np.asarray(window).ndim == 2 else window, float)


def _finite_fraction(windows: list[np.ndarray]) -> float:
    total = sum(np.asarray(w).size for w in windows)
    finite = sum(int(np.isfinite(w).sum()) for w in windows)
    return float(finite / total) if total else 0.0


def _metric_median(windows: list[np.ndarray], fn) -> float:
    values = [float(fn(_primary(w))) for w in windows if len(_primary(w))]
    return float(np.median(values)) if values else 0.0


def _spectral_concentration(x: np.ndarray) -> float:
    x = np.nan_to_num(x - np.nanmean(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    total = float(power.sum())
    return float(power.max() / total) if total > 0 else 0.0


def _noise_proxy(x: np.ndarray) -> float:
    scale = float(np.std(x))
    return float(np.median(np.abs(np.diff(x))) / (scale + 1e-12)) if len(x) > 1 else 0.0


def _stationarity_drift(x: np.ndarray) -> float:
    n = len(x)
    if n < 16:
        return 0.0
    q = max(n // 4, 4)
    a, b = x[:q], x[-q:]
    return float(abs(np.mean(a) - np.mean(b)) + abs(np.std(a) - np.std(b)))


def _impulsiveness(x: np.ndarray) -> float:
    mean_abs = float(np.mean(np.abs(x)))
    return float(np.max(np.abs(x)) / (mean_abs + 1e-12)) if len(x) else 0.0


@dataclass(frozen=True)
class SourceProfile:
    source_id: str
    device_id: str
    setup_id: str
    metrics: dict[str, Any]
    signature: str
    profile_version: str = PROFILE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "identity": {
                "source_id": self.source_id,
                "device_id": self.device_id,
                "setup_id": self.setup_id,
            },
            "channels": int(self.metrics["channels"]),
            "metrics": self.metrics,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class RoutingDecision:
    compatible: tuple[str, ...]
    reasons: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"compatible": list(self.compatible), "reasons": self.reasons}


class MethodRegistry:
    """Human-readable qualification records keyed by explicit source identity."""

    def __init__(self, entries: dict[str, dict[str, Any]] | None = None):
        self._entries = entries or {}

    @staticmethod
    def _key(profile: SourceProfile) -> str:
        return f"{profile.source_id}:{profile.device_id}:{profile.setup_id}"

    def register(
        self,
        profile: SourceProfile,
        routing: RoutingDecision,
        *,
        receipts: dict[str, Any],
    ) -> None:
        """Record a completed qualification; callers supply validated receipts."""
        families = {
            name: (
                {"state": "confirmed", "reason": "qualified", "receipt": receipts[name]}
                if name in receipts
                else {"state": "candidate", "reason": "receipt_missing"}
            )
            for name in routing.compatible
        }
        families.update({
            name: {"state": "not_measured_incompatible", "reason": reason}
            for name, reason in routing.reasons.items()
        })
        self._entries[self._key(profile)] = {
            "profile": profile.to_dict(),
            "families": families,
            "receipts": receipts,
        }

    def entry(self, profile: SourceProfile) -> dict[str, Any]:
        return self._entries[self._key(profile)]

    def status(self, profile: SourceProfile) -> str:
        entry = self._entries.get(self._key(profile))
        if entry is None:
            # A known source with a changed device/setup is a re-qualification,
            # not an unrelated new source.
            for candidate in self._entries.values():
                identity = candidate.get("profile", {}).get("identity", {})
                if identity.get("source_id") == profile.source_id:
                    return "qualification_required"
            return "new_source"
        recorded = entry["profile"]
        if (
            recorded.get("profile_version") == profile.profile_version
            and recorded.get("signature") == profile.signature
        ):
            return "confirmed"
        return "qualification_required"

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"profile_version": PROFILE_VERSION, "entries": self._entries}
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MethodRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("profile_version") != PROFILE_VERSION:
            raise ValueError("unsupported method registry profile version")
        return cls(dict(payload.get("entries", {})))


def profile_bank(
    bank: Bank, *, source_id: str, device_id: str, setup_id: str
) -> SourceProfile:
    """Build a stable, label-free profile from a loaded recording bank."""
    if not bank.windows:
        raise ValueError("cannot profile an empty bank")
    channels = int(bank.windows[0].shape[0]) if bank.windows[0].ndim == 2 else 1
    lengths = [int(_primary(w).size) for w in bank.windows]
    metrics: dict[str, Any] = {
        "recordings": int(bank.n_recordings),
        "windows": len(bank.windows),
        "channels": channels,
        "samples_per_window": int(np.median(lengths)),
        "finite_fraction": round(_finite_fraction(bank.windows), 12),
        "noise_proxy": round(_metric_median(bank.windows, _noise_proxy), 12),
        "stationarity_drift": round(_metric_median(bank.windows, _stationarity_drift), 12),
        "spectral_concentration": round(_metric_median(bank.windows, _spectral_concentration), 12),
        "impulsiveness": round(_metric_median(bank.windows, _impulsiveness), 12),
        "channel_layout": list(bank.channels),
    }
    canonical = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return SourceProfile(source_id, device_id, setup_id, metrics, signature)


def route_method_families(profile: SourceProfile) -> RoutingDecision:
    """Select hard-compatible families without using labels or outcomes."""
    m = profile.metrics
    compatible: list[str] = []
    reasons: dict[str, str] = {}

    def allow(name: str, condition: bool, reason: str) -> None:
        if condition:
            compatible.append(name)
        else:
            reasons[name] = reason

    finite = float(m["finite_fraction"])
    n = int(m["samples_per_window"])
    channels = int(m["channels"])
    recordings = int(m["recordings"])
    allow("time_domain", n >= 16 and finite > 0.99, "needs_finite_windows_of_at_least_16_samples")
    allow("spectral", n >= 64 and finite > 0.99, "needs_finite_windows_of_at_least_64_samples")
    allow("envelope", n >= 128 and finite > 0.99, "needs_finite_windows_of_at_least_128_samples")
    allow("recurrence", n >= 128 and finite > 0.99, "needs_finite_windows_of_at_least_128_samples")
    allow("coherence", channels >= 2, "requires_at_least_two_channels")
    allow("causal_lag", channels >= 2 and recordings >= 2, "requires_two_channels_and_recordings")
    return RoutingDecision(tuple(compatible), reasons)
