"""Receipt format v1 + Ed25519.

Every verdict-producing command emits a versioned, signed JSON receipt:
claim, verdict (INCLUDED/EXCLUDED/PASS/REFUSED), evidence, input hashes,
tool+version, timestamp, countersignatures. The signing key lives outside
the repo (0600); only the public key travels inside the receipt. Red-team
gate: flipping a single byte of a signed receipt must make verify FAIL.
"""
import json
import os
import stat

import numpy as np
import pytest

from signalmap.receipt import (
    SCHEMA_V1,
    build_receipt,
    canonical_bytes,
    load_or_create_key,
    sign_receipt,
    verify_receipt,
    write_receipt,
)


@pytest.fixture
def key(tmp_path):
    return load_or_create_key(tmp_path / "signing_key")


def _receipt():
    return build_receipt(
        claim="rqa family separates healthy from faulty on bank X",
        verdict="INCLUDED",
        evidence={"nested_logo": 0.91, "perm_p": 0.002, "null_passed": True,
                  "cost": 12.5},
        input_hashes={"bank": "ab" * 32, "spec": "cd" * 32},
    )


def test_build_receipt_carries_schema_and_required_fields():
    r = _receipt()
    assert r["schema"] == SCHEMA_V1 == "signalmap.receipt/1"
    assert r["tool"] == "signalmap"
    assert r["tool_version"]
    assert r["verdict"] == "INCLUDED"
    assert r["countersignatures"] == []
    assert r["created_at"].endswith("Z")


def test_build_receipt_rejects_unknown_verdict():
    with pytest.raises(ValueError, match="verdict"):
        build_receipt(claim="c", verdict="MAYBE", evidence={}, input_hashes={})


def test_key_file_created_0600_and_reused(tmp_path):
    p = tmp_path / "signing_key"
    k1 = load_or_create_key(p)
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    k2 = load_or_create_key(p)
    assert (k1.public_key().public_bytes_raw()
            == k2.public_key().public_bytes_raw())


def test_sign_then_verify_roundtrip(key):
    signed = sign_receipt(_receipt(), key)
    assert signed["pubkey"] and signed["signature"]
    assert verify_receipt(signed) is True


def test_verify_fails_if_any_field_changes(key):
    signed = sign_receipt(_receipt(), key)
    tampered = dict(signed, verdict="EXCLUDED")
    assert verify_receipt(tampered) is False


def test_red_team_single_byte_flip_fails_verify(tmp_path, key):
    """DoD gate: 1 manipulated byte in the shipped file => verify FAIL."""
    p = tmp_path / "r.receipt.json"
    write_receipt(sign_receipt(_receipt(), key), p)
    raw = bytearray(p.read_bytes())
    idx = raw.index(b"INCLUDED"[0], raw.find(b'"claim"'))
    raw[idx] ^= 0x01
    p.write_bytes(bytes(raw))
    tampered = json.loads(p.read_text())
    assert verify_receipt(tampered) is False


def test_canonical_bytes_excludes_signature_and_is_stable(key):
    signed = sign_receipt(_receipt(), key)
    assert canonical_bytes(signed) == canonical_bytes(
        {k: signed[k] for k in reversed(list(signed))})
    assert b'"signature"' not in canonical_bytes(signed)


# --- product surface: fit + monitor emit signed receipts -------------------

from signalmap.distill import FeatureSpec, W


def _write_recordings(d, kind, n_rec=2, seed=0):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = 4 * W
    t = np.arange(n)
    for r in range(n_rec):
        if kind == "healthy":
            s = np.sin(2 * np.pi * 0.01 * t) + 0.1 * rng.standard_normal(n)
        else:
            s = (np.sin(2 * np.pi * 0.30 * t)
                 + 4.0 * (rng.random(n) < 0.02) * rng.standard_normal(n)
                 + 0.1 * rng.standard_normal(n))
        np.save(d / f"{kind}_{r}.npy", s.astype(np.float64))


def test_cli_fit_and_monitor_emit_signed_receipts(tmp_path, monkeypatch):
    """DoD: `signalmap fit … && signalmap monitor …` produces signed receipts
    that verify, with input hashes binding bank+spec/detector."""
    monkeypatch.setenv("SIGNALMAP_HOME", str(tmp_path / "home"))
    from signalmap.cli import build_parser, cmd_fit, cmd_monitor
    spec = FeatureSpec(programs=["acf1(id(id(x)))", "specratio(id(id(x)))"],
                       classes=["healthy", "faulty"])
    spec_p = tmp_path / "spec.json"
    spec.save(str(spec_p))
    _write_recordings(tmp_path / "healthy", "healthy")
    _write_recordings(tmp_path / "faulty", "faulty", seed=1)
    p = build_parser()
    out = str(tmp_path / "det.json")
    cmd_fit(p.parse_args(["fit", "--spec", str(spec_p),
                          "--bank", str(tmp_path / "healthy"), "--out", out]))
    fit_receipt = json.loads((tmp_path / "det.json.receipt.json").read_text())
    assert fit_receipt["verdict"] == "PASS"
    assert verify_receipt(fit_receipt) is True
    assert "bank" in fit_receipt["input_hashes"]
    assert "spec" in fit_receipt["input_hashes"]

    cmd_monitor(p.parse_args(["monitor", "--detector", out,
                              "--bank", str(tmp_path / "faulty"), "--quiet"]))
    mon_receipt = json.loads(
        (tmp_path / "det.json.monitor.receipt.json").read_text())
    assert mon_receipt["verdict"] == "PASS"
    assert verify_receipt(mon_receipt) is True
    assert "detector" in mon_receipt["input_hashes"]
    assert "rate" in mon_receipt["evidence"]
    # both receipts signed by the same persistent key
    assert fit_receipt["pubkey"] == mon_receipt["pubkey"]


def test_cryptography_is_declared_as_a_core_dependency():
    """Every verdict command signs its receipt, so signing is not optional.
    Shipped as an extra, `pip install signalmap` gives a package whose headline
    feature dies with ModuleNotFoundError (that happened in 0.5.0 and 0.5.1)."""
    import re
    import sys
    from pathlib import Path
    if sys.version_info >= (3, 11):
        import tomllib
    else:                                    # pragma: no cover - 3.9/3.10 CI legs
        tomllib = None
    root = Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text()
    if tomllib is not None:
        deps = tomllib.loads(text)["project"]["dependencies"]
    else:
        block = re.search(r"^dependencies = \[(.*?)\]", text, re.S | re.M).group(1)
        deps = re.findall(r'"([^"]+)"', block)
    assert any(d.split(">=")[0].split("[")[0].strip() == "cryptography" for d in deps), (
        f"cryptography missing from core dependencies: {deps}")
