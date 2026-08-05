"""Standalone verifier — the OSS wedge.

``tools/verify_receipt.py`` must verify a receipt with **zero** imports from
signalmap (HEDG3 ``verify_shipped`` pattern): stdlib + cryptography only.
Checks signature, schema, and internal consistency (REFUSED => no
deploy-spec entry). Exit 0 + PASS on good receipts, exit 1 + FAIL otherwise.

The subprocess tests poison ``signalmap`` on PYTHONPATH so any accidental
import inside the verifier explodes even though the suite venv has
signalmap installed; CI additionally runs it from a bare venv.
"""
import ast
import subprocess
import sys
from pathlib import Path

from signalmap.receipt import (
    build_receipt,
    load_or_create_key,
    sign_receipt,
    write_receipt,
)

VERIFIER = Path(__file__).resolve().parents[2] / "tools" / "verify_receipt.py"


def _signed(tmp_path, **overrides):
    fields = {"claim": "rqa separates classes on bank X",
              "verdict": "INCLUDED", "evidence": {"perm_p": 0.002},
              "input_hashes": {"bank": "ab" * 32}}
    fields.update(overrides)
    key = load_or_create_key(tmp_path / "k")
    return sign_receipt(build_receipt(**fields), key)


def _run(receipt_path, tmp_path, *extra):
    poison = tmp_path / "poison"
    poison.mkdir(exist_ok=True)
    (poison / "signalmap.py").write_text(
        "raise ImportError('verifier must not import signalmap')\n")
    return subprocess.run(
        [sys.executable, str(VERIFIER), str(receipt_path), *extra],
        capture_output=True, text=True, check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(poison)})


def test_verifier_source_has_no_signalmap_import():
    tree = ast.parse(VERIFIER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            continue
        for m in mods:
            assert not m.startswith("signalmap"), f"verifier imports {m}"


def test_verifier_passes_valid_receipt(tmp_path):
    p = tmp_path / "r.json"
    write_receipt(_signed(tmp_path), p)
    res = _run(p, tmp_path)
    assert res.returncode == 0, res.stderr
    assert "PASS" in res.stdout


def test_verifier_fails_tampered_receipt(tmp_path):
    p = tmp_path / "r.json"
    write_receipt(_signed(tmp_path), p)
    raw = bytearray(p.read_bytes())
    raw[raw.find(b"rqa")] ^= 0x01
    p.write_bytes(bytes(raw))
    res = _run(p, tmp_path)
    assert res.returncode == 1
    assert "FAIL" in res.stdout


def _resign(r, tmp_path):
    """Re-sign a mutated body so only the targeted check can fail."""
    body = {k: v for k, v in r.items() if k not in ("pubkey", "signature")}
    return sign_receipt(body, load_or_create_key(tmp_path / "k"))


def test_verifier_fails_wrong_schema(tmp_path):
    r = _signed(tmp_path)
    r["schema"] = "signalmap.receipt/999"
    r = _resign(r, tmp_path)
    p = tmp_path / "r.json"
    write_receipt(r, p)
    res = _run(p, tmp_path)
    assert res.returncode == 1
    assert "schema" in res.stdout.lower()


def test_verifier_fails_missing_required_field(tmp_path):
    r = _signed(tmp_path)
    del r["evidence"]
    r = _resign(r, tmp_path)
    p = tmp_path / "r.json"
    write_receipt(r, p)
    res = _run(p, tmp_path)
    assert res.returncode == 1
    assert "evidence" in res.stdout.lower()


def test_verifier_fails_refused_with_deploy_spec(tmp_path):
    """Internal consistency: a REFUSED verdict must not carry a deploy spec."""
    r = _signed(tmp_path, verdict="REFUSED",
                evidence={"deploy_spec": {"programs": ["acf1(x)"]}})
    p = tmp_path / "r.json"
    write_receipt(r, p)
    res = _run(p, tmp_path)
    assert res.returncode == 1
    assert "refused" in res.stdout.lower()


def test_verifier_pinned_pubkey_mismatch_fails(tmp_path):
    p = tmp_path / "r.json"
    write_receipt(_signed(tmp_path), p)
    other = load_or_create_key(tmp_path / "other_key")
    pin = other.public_key().public_bytes_raw().hex()
    res = _run(p, tmp_path, "--pubkey", pin)
    assert res.returncode == 1
    assert "pubkey" in res.stdout.lower()


def test_verifier_pinned_pubkey_match_passes(tmp_path):
    r = _signed(tmp_path)
    p = tmp_path / "r.json"
    write_receipt(r, p)
    res = _run(p, tmp_path, "--pubkey", r["pubkey"])
    assert res.returncode == 0, res.stdout + res.stderr
