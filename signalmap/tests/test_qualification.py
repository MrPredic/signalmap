import json

import numpy as np

from signalmap.distill import Bank
from signalmap.qualification import (
    MethodRegistry,
    filter_programs,
    profile_bank,
    route_method_families,
)


def _single_bank() -> Bank:
    windows = [
        np.sin(np.linspace(0, 8 * np.pi, 1024)),
        np.sin(np.linspace(0, 8 * np.pi, 1024) + 0.2),
    ]
    return Bank(
        windows=windows,
        y=np.array(["healthy", "fault"]),
        g=np.array([0, 1]),
        classes=["fault", "healthy"],
        n_recordings=2,
    )


def _multi_bank() -> Bank:
    t = np.linspace(0, 8 * np.pi, 1024)
    windows = [
        np.stack([np.sin(t), np.cos(t)]),
        np.stack([np.sin(t + 0.1), np.cos(t + 0.1)]),
    ]
    return Bank(
        windows=windows,
        y=np.array(["healthy", "fault"]),
        g=np.array([0, 1]),
        classes=["fault", "healthy"],
        n_recordings=2,
        channels=["accel", "microphone"],
    )


def test_profile_is_label_free_and_reproducible():
    bank = _single_bank()
    profile = profile_bank(
        bank, source_id="bench", device_id="dev-1", setup_id="setup-a"
    )
    payload = profile.to_dict()
    assert "healthy" not in json.dumps(payload)
    assert payload["identity"] == {
        "source_id": "bench", "device_id": "dev-1", "setup_id": "setup-a"
    }
    assert payload["channels"] == 1
    assert payload["signature"] == profile_bank(
        bank, source_id="bench", device_id="dev-1", setup_id="setup-a"
    ).signature
    assert payload["metrics"]["finite_fraction"] == 1.0


def test_single_channel_routes_without_multichannel_families():
    decision = route_method_families(
        profile_bank(_single_bank(), source_id="s", device_id="d", setup_id="u")
    )
    assert "time_domain" in decision.compatible
    assert "spectral" in decision.compatible
    assert "coherence" not in decision.compatible
    assert decision.reasons["coherence"] == "requires_at_least_two_channels"


def test_multichannel_routes_coherence_and_causal_lag():
    decision = route_method_families(
        profile_bank(_multi_bank(), source_id="s", device_id="d", setup_id="u")
    )
    assert "coherence" in decision.compatible
    assert "causal_lag" in decision.compatible
    assert decision.compatible == route_method_families(
        profile_bank(_multi_bank(), source_id="s", device_id="d", setup_id="u")
    ).compatible


def test_registry_roundtrip_and_exact_match(tmp_path):
    profile = profile_bank(_single_bank(), source_id="s", device_id="d", setup_id="u")
    registry = MethodRegistry()
    registry.register(profile, route_method_families(profile), receipts={"time_domain": "r1"})
    path = tmp_path / "registry.json"
    registry.save(path)
    loaded = MethodRegistry.load(path)
    assert loaded.status(profile) == "confirmed"
    assert loaded.entry(profile)["receipts"] == {"time_domain": "r1"}
    assert loaded.entry(profile)["families"]["time_domain"]["state"] == "confirmed"
    assert loaded.entry(profile)["families"]["spectral"]["state"] == "candidate"


def test_registry_requires_qualification_after_setup_change():
    bank = _single_bank()
    registry = MethodRegistry()
    first = profile_bank(bank, source_id="s", device_id="d", setup_id="u")
    registry.register(first, route_method_families(first), receipts={})
    changed = profile_bank(bank, source_id="s", device_id="d", setup_id="u2")
    assert registry.status(changed) == "qualification_required"


def test_qualify_cli_writes_profile_and_routing(tmp_path):
    from signalmap.cli import build_parser

    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    np.save(bank_dir / "A_0.npy", _single_bank().windows[0])
    out = tmp_path / "qualification.json"
    args = build_parser().parse_args([
        "qualify", "--bank", str(bank_dir), "--source-id", "s",
        "--device-id", "d", "--setup-id", "u", "--out", str(out),
    ])
    args.func(args)
    payload = json.loads(out.read_text())
    assert payload["status"] == "new_source"
    assert payload["profile"]["identity"]["device_id"] == "d"
    assert "coherence" in payload["routing"]["reasons"]


def test_qualify_cli_registers_only_receipted_families(tmp_path):
    from signalmap.cli import build_parser

    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    np.save(bank_dir / "A_0.npy", _single_bank().windows[0])
    out = tmp_path / "qualification.json"
    registry = tmp_path / "registry.json"
    receipts = tmp_path / "receipts.json"
    receipts.write_text(json.dumps({"families": {"time_domain": {"verdict": "PASS"}}}))
    args = build_parser().parse_args([
        "qualify", "--bank", str(bank_dir), "--source-id", "s",
        "--device-id", "d", "--setup-id", "u", "--out", str(out),
        "--register", str(registry), "--receipts", str(receipts),
    ])
    args.func(args)
    payload = json.loads(out.read_text())
    saved = json.loads(registry.read_text())
    entry = next(iter(saved["entries"].values()))
    assert payload["status"] == "new_source"
    assert entry["families"]["time_domain"]["state"] == "confirmed"
    assert entry["families"]["spectral"]["state"] == "candidate"


def test_qualify_cli_routes_optional_method_catalog(tmp_path):
    from signalmap.cli import build_parser

    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    np.save(bank_dir / "A_0.npy", _single_bank().windows[0])
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "version": 1,
        "methods": [
            {"name": "coherence-x", "family": "coherence", "status": "candidate",
             "requires": {"channels": 2}, "cost": "medium"},
            {"name": "new-time", "family": "novel", "status": "speculative",
             "requires": {"channels": 1}, "cost": "low"},
        ],
    }))
    out = tmp_path / "qualification.json"
    args = build_parser().parse_args([
        "qualify", "--bank", str(bank_dir), "--source-id", "s",
        "--device-id", "d", "--setup-id", "u", "--out", str(out),
        "--catalog", str(catalog),
    ])
    args.func(args)
    capability = json.loads(out.read_text())["capabilities"]
    assert capability["eligible"] == ["new-time"]
    assert capability["hard_excluded"] == ["coherence-x"]


def test_family_filter_removes_unqualified_program_families():
    from signalmap.distill import enumerate_programs

    filtered = filter_programs(enumerate_programs(), {"time_domain"})
    assert filtered
    assert all(p.pool not in {"specratio", "speccent", "specflat", "acflag"}
               and p.t1 != "env" and p.t2 != "env" for p in filtered)
    assert len(filtered) < len(enumerate_programs())
