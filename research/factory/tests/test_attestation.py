"""Targeted tests for attestation.py -- Ed25519 signed-receipt core.
Run: .venv-research/bin/python -m pytest tests/test_attestation.py -q
(run from research/factory/). Fully offline, no network, no real ledger I/O.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import attestation as at  # noqa: E402


SAMPLE_RECEIPT = {
    "config_hash": "sha256:abc123",
    "data_provenance_sha": "sha256:cwru-v2-def456",
    "verdict": "INCLUDED",
    "ci_low": 0.042,
    "ci_high": 0.236,
    "ledger_tip_hash": "9f8e7d6c5b4a",
    "ts": "2026-07-21T00:00:00Z",
}


def test_sign_verify_roundtrip():
    priv, pub = at.generate_keypair()
    sig = at.sign_receipt(SAMPLE_RECEIPT, priv)
    assert at.verify_receipt(SAMPLE_RECEIPT, sig, pub) is True


def test_tampered_receipt_fails_verify():
    priv, pub = at.generate_keypair()
    sig = at.sign_receipt(SAMPLE_RECEIPT, priv)

    tampered = copy.deepcopy(SAMPLE_RECEIPT)
    tampered["verdict"] = "EXCLUDED"  # flip the one field that matters most
    assert at.verify_receipt(tampered, sig, pub) is False

    tampered2 = copy.deepcopy(SAMPLE_RECEIPT)
    tampered2["ci_low"] = 0.999  # any single field changing must break verify
    assert at.verify_receipt(tampered2, sig, pub) is False


def test_wrong_pubkey_fails_verify():
    priv, _pub = at.generate_keypair()
    _other_priv, other_pub = at.generate_keypair()
    sig = at.sign_receipt(SAMPLE_RECEIPT, priv)
    assert at.verify_receipt(SAMPLE_RECEIPT, sig, other_pub) is False


def test_pubkey_from_privkey_matches_generated_pair():
    priv, pub = at.generate_keypair()
    assert at.pubkey_from_privkey(priv) == pub


def test_canonical_json_is_deterministic_regardless_of_key_order():
    a = {"z": 1, "a": 2, "nested": {"y": 1, "x": 2}}
    b = {"a": 2, "nested": {"x": 2, "y": 1}, "z": 1}
    assert at.canonical_json(a) == at.canonical_json(b)


def test_canonical_json_stable_across_calls():
    out1 = at.canonical_json(SAMPLE_RECEIPT)
    out2 = at.canonical_json(copy.deepcopy(SAMPLE_RECEIPT))
    assert out1 == out2
    # sorted keys, no whitespace -- matches receipt_ledger._canon's convention
    assert out1 == json.dumps(SAMPLE_RECEIPT, sort_keys=True, separators=(",", ":")).encode()


def test_sign_verify_roundtrip_survives_reserialization():
    """A receipt round-tripped through JSON (as it would be when stored to
    disk and reloaded) must still verify -- proves signing binds to the
    canonical VALUE, not incidental dict object identity/ordering."""
    priv, pub = at.generate_keypair()
    sig = at.sign_receipt(SAMPLE_RECEIPT, priv)
    reloaded = json.loads(json.dumps(SAMPLE_RECEIPT))  # dict -> str -> dict
    assert at.verify_receipt(reloaded, sig, pub) is True


def test_build_receipt_shape_and_verdict_validation():
    r = at.build_receipt("sha256:cfg", "sha256:data", "INCLUDED", 0.04, 0.24,
                         "tiphash123", timestamp="2026-07-21T00:00:00Z")
    assert r["verdict"] == "INCLUDED"
    assert r["ledger_tip_hash"] == "tiphash123"
    priv, pub = at.generate_keypair()
    sig = at.sign_receipt(r, priv)
    assert at.verify_receipt(r, sig, pub) is True

    try:
        at.build_receipt("h", "h", "MAYBE", 0, 1, "tip")
        assert False, "expected ValueError for invalid verdict"
    except ValueError:
        pass


def test_cli_keygen_sign_verify_tip(monkeypatch, tmp_path, capsys):
    """CLI smoke test, fully offline: keygen -> sign-tip (against a redirected
    fake ledger) -> verify-tip, using only stdout (no real ledger touched)."""
    import receipt_ledger
    monkeypatch.setattr(receipt_ledger, "LEDGER", str(tmp_path / "LEDGER.jsonl"))
    receipt_ledger.log_receipt("TEST", {"note": "fixture entry"})
    tip = receipt_ledger._last_hash()

    at.main(["keygen"])
    keys = json.loads(capsys.readouterr().out)
    priv_hex, pub_hex = keys["privkey_hex"], keys["pubkey_hex"]

    at.main(["sign-tip", priv_hex])
    signed = json.loads(capsys.readouterr().out)
    assert signed["receipt"]["ledger_tip_hash"] == tip

    rc = at.main(["verify-tip", signed["sig_hex"], pub_hex,
                 signed["receipt"]["ledger_tip_hash"], signed["receipt"]["ts"]])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "VALID"

    # tampered tip hash must fail verify
    rc_bad = at.main(["verify-tip", signed["sig_hex"], pub_hex,
                      "deadbeef", signed["receipt"]["ts"]])
    assert rc_bad == 1
    assert capsys.readouterr().out.strip() == "INVALID"
