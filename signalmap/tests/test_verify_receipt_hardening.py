"""Adversarial hardening of the standalone verifier.

The offline promise is: a third party reads ``tools/verify_receipt.py``,
runs it on a receipt, and learns what the receipt says — without installing
signalmap and without trusting us. Every test here is an attack that broke
that promise before it was fixed:

* the bytes that get signed must be the bytes a reader sees (no duplicate
  JSON keys, no non-RFC literals, no unpaired surrogates);
* the consistency rules must not be evadable by reshaping the JSON
  (nesting, wrong container type);
* nothing unverifiable may ride along under a PASS (countersignatures);
* malformed input must produce a clean FAIL, never a traceback;
* the output must not read as authenticity when no key is pinned.

Same subprocess + poisoned-``PYTHONPATH`` harness as
``test_verify_receipt_standalone``: an accidental ``import signalmap``
inside the verifier explodes.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from signalmap.receipt import build_receipt, load_or_create_key, sign_receipt

VERIFIER = Path(__file__).resolve().parents[2] / "tools" / "verify_receipt.py"
VERDICT_DIR = Path(__file__).resolve().parents[1] / "verdicts"


def _signed(tmp_path, key_name="k", **overrides):
    fields = {"claim": "rqa separates classes on bank X",
              "verdict": "INCLUDED", "evidence": {"perm_p": 0.002},
              "input_hashes": {"bank": "ab" * 32}}
    extra = {k: overrides.pop(k) for k in list(overrides)
             if k not in ("claim", "verdict", "evidence", "input_hashes")}
    fields.update(overrides)
    body = build_receipt(**fields)
    body.update(extra)
    return sign_receipt(body, load_or_create_key(tmp_path / key_name))


def _run(raw, tmp_path, *extra):
    """Write raw bytes as the receipt and verify them in a clean subprocess."""
    poison = tmp_path / "poison"
    poison.mkdir(exist_ok=True)
    (poison / "signalmap.py").write_text(
        "raise ImportError('verifier must not import signalmap')\n")
    p = tmp_path / "r.json"
    p.write_bytes(raw if isinstance(raw, bytes) else raw.encode())
    res = subprocess.run(
        [sys.executable, str(VERIFIER), str(p), *extra],
        capture_output=True, text=True, check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(poison)})
    return res


def _assert_clean_fail(res):
    """Exit 1 with a FAIL line — not a traceback, never exit 0."""
    assert res.returncode == 1, res.stdout + res.stderr
    assert "Traceback" not in res.stderr, res.stderr
    assert res.stdout.startswith("FAIL"), res.stdout


# --- 1. canonicalization: signed bytes == bytes a reader sees --------------

def test_verifier_rejects_duplicate_json_keys(tmp_path):
    """A repeated key makes the file say one thing and the signature another.

    ``json.load`` keeps the LAST value, so an attacker prepends a flattering
    copy of ``verdict``/``claim``: the signature still checks out over the
    honest last value while a human (and every first-wins parser) reads the
    forgery.
    """
    r = _signed(tmp_path, verdict="REFUSED",
                claim="bank too small — deployment refused")
    blob = json.dumps(r)
    forged = ('{"verdict": "INCLUDED", "claim": "rqa PROVEN on bank X", '
              + blob[1:])
    assert json.loads(forged)["verdict"] == "REFUSED"  # signature is intact
    res = _run(forged, tmp_path)
    _assert_clean_fail(res)
    assert "duplicate" in res.stdout.lower()


def test_verifier_rejects_duplicate_signature_key(tmp_path):
    """Same trick aimed at the signature field itself."""
    blob = json.dumps(_signed(tmp_path))
    res = _run('{"signature": "00", ' + blob[1:], tmp_path)
    _assert_clean_fail(res)
    assert "duplicate" in res.stdout.lower()


def test_verifier_rejects_nan_and_infinity_literals(tmp_path):
    """``NaN``/``Infinity`` are Python extensions, not JSON.

    Python signs and re-verifies them happily; a Go/Rust/JS verifier cannot
    parse the file at all, so the receipt is not offline-verifiable by the
    third party the promise is about.
    """
    r = _signed(tmp_path, evidence={"perm_p": float("nan"),
                                    "acc": float("inf")})
    raw = json.dumps(r)
    assert "NaN" in raw
    res = _run(raw, tmp_path)
    _assert_clean_fail(res)
    assert "nan" in res.stdout.lower() or "json" in res.stdout.lower()


def test_verifier_rejects_number_overflowing_to_infinity(tmp_path):
    """``1e999`` parses to inf and is re-serialized as ``Infinity``."""
    r = _signed(tmp_path, evidence={"acc": 1e999})
    res = _run(json.dumps(r), tmp_path)
    _assert_clean_fail(res)
    assert "finite" in res.stdout.lower() or "infinity" in res.stdout.lower()


def test_verifier_rejects_unpaired_surrogate(tmp_path):
    """A lone surrogate cannot be encoded as UTF-8 by other parsers."""
    r = _signed(tmp_path, claim="\ud800 payload")
    res = _run(json.dumps(r), tmp_path)
    _assert_clean_fail(res)
    assert "surrogate" in res.stdout.lower()


def test_verifier_rejects_integer_outside_double_range(tmp_path):
    """Beyond 2**53 a JSON number means different things in different hosts."""
    r = _signed(tmp_path, evidence={"n_windows": 2 ** 70})
    res = _run(json.dumps(r), tmp_path)
    _assert_clean_fail(res)
    assert "integer" in res.stdout.lower()


# --- 2. consistency rules must survive reshaping ---------------------------

def test_verifier_rejects_refused_with_nested_deploy_spec(tmp_path):
    """The deploy_spec rule only looked at the top level of evidence."""
    r = _signed(tmp_path, verdict="REFUSED",
                evidence={"result": {"deploy_spec": {"programs": ["acf1(x)"]}}})
    res = _run(json.dumps(r), tmp_path)
    _assert_clean_fail(res)
    assert "deploy_spec" in res.stdout


def test_verifier_rejects_refused_carrying_an_endorsing_verdict(tmp_path):
    """A refusal must not contain a nested PASS/INCLUDED endorsement."""
    r = _signed(tmp_path, verdict="REFUSED",
                evidence={"premium": {"family": "rqa", "verdict": "INCLUDED"},
                          "base_gate": {"verdict": "PASS"}})
    res = _run(json.dumps(r), tmp_path)
    _assert_clean_fail(res)
    assert "refused" in res.stdout.lower()


@pytest.mark.parametrize("evidence", [
    [{"deploy_spec": {"programs": ["acf1(x)"]}}],   # list, not object
    None,                                           # null
    '{"deploy_spec": ["acf1(x)"]}',                 # JSON *inside a string*
])
def test_verifier_rejects_non_object_evidence(tmp_path, evidence):
    """Every consistency check was guarded by ``isinstance(evidence, dict)``,
    so any other container silently skipped all of them."""
    r = _signed(tmp_path, verdict="REFUSED", evidence=evidence)
    res = _run(json.dumps(r), tmp_path)
    _assert_clean_fail(res)
    assert "evidence" in res.stdout.lower()


def test_verifier_rejects_wrongly_typed_input_hashes(tmp_path):
    r = _signed(tmp_path, input_hashes="none")
    res = _run(json.dumps(r), tmp_path)
    _assert_clean_fail(res)
    assert "input_hashes" in res.stdout


def test_verifier_rejects_non_object_provenance(tmp_path):
    """A list-shaped provenance skipped every archive_signature check while
    still reading as an archive signature."""
    r = _signed(tmp_path, evidence={"provenance": [{"mode":
                                                    "archive_signature"}]})
    res = _run(json.dumps(r), tmp_path)
    _assert_clean_fail(res)
    assert "provenance" in res.stdout


# --- 3. nothing unverifiable rides along under a PASS ----------------------

def test_verifier_rejects_unverifiable_countersignatures(tmp_path):
    """schema/1 defines no countersignature format, so the verifier cannot
    check one — it must not print PASS next to a claimed third-party seal."""
    r = _signed(tmp_path,
                countersignatures=[{"by": "TUV Rheinland", "sig": "de" * 32}])
    res = _run(json.dumps(r), tmp_path)
    _assert_clean_fail(res)
    assert "countersignature" in res.stdout.lower()


def test_verifier_rejects_wrongly_typed_countersignatures(tmp_path):
    r = _signed(tmp_path, countersignatures="countersigned by BSI")
    res = _run(json.dumps(r), tmp_path)
    _assert_clean_fail(res)
    assert "countersignature" in res.stdout.lower()


# --- 4. robustness: clean FAIL, never a traceback --------------------------

def test_verifier_clean_fail_on_invalid_utf8(tmp_path):
    _assert_clean_fail(_run(b'{"schema": "signalmap.receipt/1", "c": "\xff"}',
                            tmp_path))


def test_verifier_clean_fail_on_deeply_nested_json(tmp_path):
    _assert_clean_fail(_run("[" * 100000 + "]" * 100000, tmp_path))


def test_verifier_clean_fail_on_deeply_nested_object(tmp_path):
    raw = '{"schema": ' + '[' * 2000 + ']' * 2000 + "}"
    _assert_clean_fail(_run(raw, tmp_path))


def test_verifier_clean_fail_on_oversized_integer_literal(tmp_path):
    """Python 3.11 caps int<->str conversion; the raw ValueError escaped."""
    _assert_clean_fail(
        _run('{"schema": "signalmap.receipt/1", "claim": ' + "9" * 5000 + "}",
             tmp_path))


# --- 5. trust semantics ----------------------------------------------------

def test_unpinned_pass_is_labelled_integrity_only_and_warns(tmp_path):
    """An attacker's own key verifies fine — that is what integrity means.

    The output therefore must not read as authenticity to someone skimming
    it: PASS has to carry the scope on the same line plus an explicit
    warning that anyone can mint such a receipt.
    """
    r = _signed(tmp_path, key_name="attacker_key",
                claim="signalmap certifies bank X")
    res = _run(json.dumps(r), tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    first = res.stdout.splitlines()[0]
    assert "integrity only" in first.lower()
    assert "warning" in res.stdout.lower()
    assert "not prove who signed" in res.stdout.lower()


def test_pinned_pass_is_labelled_authentic(tmp_path):
    r = _signed(tmp_path)
    res = _run(json.dumps(r), tmp_path, "--pubkey", r["pubkey"])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "authentic" in res.stdout.lower()
    assert "warning" not in res.stdout.lower()


def test_pin_matches_regardless_of_hex_case(tmp_path):
    """Same 32 key bytes, different transcription — pinning compares keys,
    not strings, so a copy-pasted uppercase pin must not read as a forgery."""
    r = _signed(tmp_path)
    res = _run(json.dumps(r), tmp_path, "--pubkey", r["pubkey"].upper())
    assert res.returncode == 0, res.stdout + res.stderr


def test_malformed_pin_is_reported_as_a_bad_pin(tmp_path):
    """A typo'd pin must not masquerade as 'this receipt is a forgery'."""
    r = _signed(tmp_path)
    res = _run(json.dumps(r), tmp_path, "--pubkey", "not-hex")
    _assert_clean_fail(res)
    assert "--pubkey" in res.stdout


def test_pinned_wrong_key_still_fails(tmp_path):
    r = _signed(tmp_path)
    other = load_or_create_key(tmp_path / "other")
    res = _run(json.dumps(r), tmp_path,
               "--pubkey", other.public_key().public_bytes_raw().hex())
    _assert_clean_fail(res)
    assert "pubkey" in res.stdout.lower()


def test_tampered_body_does_not_also_report_a_false_pubkey_mismatch(tmp_path):
    """A wrong reason is a wrong verdict.

    Editing the body leaves the receipt's own pubkey untouched, so pinning
    that very key must report the broken signature and nothing else. Saying
    "not signed by the pinned key" here would send a reader hunting for a key
    substitution that never happened.
    """
    r = _signed(tmp_path)
    pinned = r["pubkey"]
    r["evidence"]["perm_p"] = 0.9          # body changed, key untouched
    res = _run(json.dumps(r), tmp_path, "--pubkey", pinned)
    _assert_clean_fail(res)
    assert "signature does not match receipt body" in res.stdout
    assert "pubkey mismatch" not in res.stdout, res.stdout


# --- 6. the hardening must not reject the receipts we ship -----------------

@pytest.mark.parametrize("path", sorted(VERDICT_DIR.glob("*.receipt.json")),
                         ids=lambda p: p.name)
def test_shipped_receipts_pass_the_hardened_verifier(path, tmp_path):
    res = _run(path.read_bytes(), tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
