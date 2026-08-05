"""``distill`` verdicts on ``signalmap.receipt/1``.

Until now only ``fit``/``monitor`` signed. ``distill`` is the command that
actually produces the verdicts we sell — the base gate (PASS, or a written
REFUSED) and one INCLUDED/EXCLUDED per premium family. Each becomes a signed
receipt whose evidence has the same shape as the archived corpus, so one
reader parses run receipts and archive receipts alike.

Refusal honesty is the load-bearing part: a base gate that fails signs
``REFUSED`` and must NOT carry a deploy_spec — the standalone verifier
enforces that pairing.

Speed: the heavy full-grammar path runs exactly once (the run_cli wiring
test); everything else drives ``distill`` directly with a small budget.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from signalmap import corpus
from signalmap.distill import distill, load_bank, run_cli
from signalmap.premium import PREMIUM_FAMILIES, PremiumFamily
from signalmap.receipt import hash_bank, hash_file, verify_receipt
from signalmap.run_receipts import emit_distill_receipts

VERIFIER = Path(__file__).resolve().parents[2] / "tools" / "verify_receipt.py"
SMALL = dict(C=5, kmax=3, thr=0.005, n_perm=25, trees=25, cand=15)


@pytest.fixture(autouse=True)
def _isolated_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALMAP_HOME", str(tmp_path / "home"))


@pytest.fixture
def oracle_family():
    fam = PremiumFamily(name="oracle",
                        featurize=lambda w: np.array([float(np.asarray(w)[0])]),
                        feature_names=["oracle_first"],
                        cost_note="O(1) test-only")
    PREMIUM_FAMILIES["oracle"] = fam
    yield fam
    del PREMIUM_FAMILIES["oracle"]


def _noise_bank(dirpath, n=2048, seed=1):
    rng = np.random.default_rng(seed)
    dirpath.mkdir(parents=True, exist_ok=True)
    for cls in ("A", "B"):
        for r in range(3):
            s = rng.standard_normal(n)
            np.save(dirpath / f"{cls}_{r}.npy", s.astype(np.float64))
    return dirpath


def _plant_marker(bank):
    """Only a family reading sample 0 can see the label."""
    for w, y in zip(bank.windows, bank.y):
        w[0] = 5.0 if y == "B" else -5.0
    return bank


def _emit(res, bank_dir, tmp_path, name="spec.json"):
    spec_path = tmp_path / name
    res.spec.save(spec_path)
    report = tmp_path / "spec_report.md"
    report.write_text(res.report)
    return emit_distill_receipts(res, bank_path=bank_dir, spec_path=spec_path,
                                 report_path=report), spec_path


def _strict(path):
    """Parse rejecting NaN/Infinity — receipts must be valid JSON for any
    third-party verifier, not just Python's lenient decoder."""
    def boom(x):
        raise AssertionError(f"non-JSON constant in receipt: {x}")
    return json.loads(Path(path).read_text(), parse_constant=boom)



def _result(tmp_path, *, passed=True, null=0.10, premium=()):
    """A DistillResult built directly — the unit under test is receipt
    emission, not the gauntlet (which has its own tests and costs minutes)."""
    from signalmap.distill import DistillResult, FeatureSpec
    spec = FeatureSpec(programs=["mean(abs(x))", "crest(diff(x))"],
                       classes=["A", "B"], window=1024, budget_c=5,
                       n_recordings=6, nested_acc=0.91, forged_acc=0.94,
                       chance=0.5, premium=[r["family"] for r in premium
                                            if r["included"]])
    return DistillResult(spec, "# report\n", 0.91, 0.5, 0.72, 0.94,
                         0.017 if passed else 0.42, null, 30, 2128, 0.135,
                         passed, list(premium))


def _prem(name="oracle", included=True):
    return {"family": name, "base_acc": 0.50, "aug_acc": 0.98,
            "delta": 0.48 if included else -0.01,
            "ci_lo": 0.31 if included else -0.09,
            "ci_hi": 0.62 if included else 0.07,
            "cost_ms": 12.5, "base_cost_ms": 0.125,
            "cost_note": "O(1) test-only", "included": included}


# --- base gate ---------------------------------------------------------

def test_passing_gate_signs_pass_with_the_deploy_spec(tmp_path):
    d = _noise_bank(tmp_path / "bank")
    res = _result(tmp_path, passed=True)
    (base, *rest), _ = _emit(res, d, tmp_path)
    assert rest == []

    r = _strict(base)
    assert r["schema"] == "signalmap.receipt/1"
    assert r["verdict"] == "PASS"
    assert verify_receipt(r)
    assert r["evidence"]["deploy_spec"] == res.spec.programs
    assert r["evidence"]["base_gate"]["nested_logo"] == pytest.approx(res.nested_acc)
    assert r["evidence"]["base_gate"]["perm_p"] == pytest.approx(res.p_forged)
    assert r["evidence"]["provenance"]["mode"] == "run"
    assert set(r["input_hashes"]) >= {"bank", "spec", "report"}


def test_failed_gate_signs_refused_without_a_deploy_spec(tmp_path):
    d = _noise_bank(tmp_path / "bank")
    res = _result(tmp_path, passed=False)
    (base,), _ = _emit(res, d, tmp_path)

    r = _strict(base)
    assert r["verdict"] == "REFUSED"
    assert "deploy_spec" not in r["evidence"]
    assert "no deploy spec is endorsed" in r["evidence"]["refusal"]
    assert verify_receipt(r)


def test_nan_null_selftest_is_json_null_not_nan(tmp_path):
    d = _noise_bank(tmp_path / "bank")
    res = _result(tmp_path, null=float("nan"))
    (base,), _ = _emit(res, d, tmp_path)
    assert _strict(base)["evidence"]["base_gate"]["null_selftest"] is None


def test_receipts_bind_bank_spec_and_report_digests(tmp_path):
    d = _noise_bank(tmp_path / "bank")
    res = _result(tmp_path)
    (base,), spec_path = _emit(res, d, tmp_path)
    r = _strict(base)
    assert r["input_hashes"]["bank"] == hash_bank(d)
    assert r["input_hashes"]["spec"] == hash_file(spec_path)
    assert r["input_hashes"]["report"] == hash_file(tmp_path / "spec_report.md")


# --- premium families --------------------------------------------------

def test_premium_verdicts_get_their_own_receipts(tmp_path, oracle_family):
    d = _noise_bank(tmp_path / "bank")
    bank = _plant_marker(load_bank(str(d), label_by="prefix"))
    res = distill(bank, premium=("oracle",), null_check=False, **SMALL)
    rec = res.premium_receipts[0]
    paths, _ = _emit(res, d, tmp_path)
    assert len(paths) == 2

    r = _strict(tmp_path / "spec.oracle.receipt.json")
    assert r["verdict"] == ("INCLUDED" if rec["included"] else "EXCLUDED")
    prem = r["evidence"]["premium"]
    assert prem["family"] == "oracle"
    assert prem["base_acc"] == pytest.approx(rec["base_acc"])
    assert prem["augmented_acc"] == pytest.approx(rec["aug_acc"])
    assert prem["paired_delta"] == pytest.approx(rec["delta"])
    assert prem["ci95"] == [pytest.approx(rec["ci_lo"]), pytest.approx(rec["ci_hi"])]
    assert prem["cost_note"] == rec["cost_note"]
    assert verify_receipt(r)


def test_premium_receipt_shape_matches_the_archive_corpus(tmp_path,
                                                          oracle_family):
    """One reader for run receipts and archive receipts."""
    d = _noise_bank(tmp_path / "bank")
    res = _result(tmp_path, premium=[_prem()])
    _emit(res, d, tmp_path)
    run_receipt = _strict(tmp_path / "spec.oracle.receipt.json")
    archive = corpus.load_corpus()[0]

    assert set(run_receipt["evidence"]) >= {"bank", "family", "premium",
                                            "base_gate", "provenance"}
    assert set(run_receipt["evidence"]["premium"]) >= set(
        archive["evidence"]["premium"])
    assert run_receipt["evidence"]["provenance"]["mode"] == "run"
    assert archive["evidence"]["provenance"]["mode"] == "archive_signature"
    assert "2 verdicts" in corpus.traction_line([run_receipt, archive])


# --- standalone verifier + red team ------------------------------------

def test_emitted_receipts_pass_the_standalone_verifier(tmp_path):
    d = _noise_bank(tmp_path / "bank")
    res = _result(tmp_path, premium=[_prem(), _prem("cheap", included=False)])
    paths, _ = _emit(res, d, tmp_path)

    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "signalmap.py").write_text(
        "raise ImportError('verifier must not import signalmap')\n")
    for p in paths:
        proc = subprocess.run([sys.executable, str(VERIFIER), str(p)],
                              capture_output=True, text=True, check=False,
                              env={"PATH": "/usr/bin:/bin",
                                   "PYTHONPATH": str(poison)})
        assert proc.returncode == 0, proc.stdout + proc.stderr


def test_tampered_run_receipt_fails_verification(tmp_path):
    d = _noise_bank(tmp_path / "bank")
    res = _result(tmp_path)
    (base,), _ = _emit(res, d, tmp_path)
    r = json.loads(Path(base).read_text())
    r["evidence"]["base_gate"]["nested_logo"] = 0.999
    assert not verify_receipt(r)


# --- CLI wiring --------------------------------------------------------

def test_run_cli_writes_receipts_next_to_the_spec(tmp_path, capsys,
                                                  monkeypatch):
    """Plumbing only: the gauntlet itself is stubbed (it has its own tests and
    costs minutes); real gauntlet numbers reach a receipt in the premium test
    above."""
    d = _noise_bank(tmp_path / "bank")
    res = _result(tmp_path, premium=[_prem()])
    monkeypatch.setattr("signalmap.distill.distill",
                        lambda bank, **kw: res)
    out = tmp_path / "spec.json"
    run_cli(str(d), "prefix", 5, str(out))

    printed = capsys.readouterr().out
    assert "receipt ->" in printed
    for name in ("spec.receipt.json", "spec.oracle.receipt.json"):
        r = _strict(tmp_path / name)
        assert verify_receipt(r)
        assert r["evidence"]["provenance"]["spec"] == str(out)
    assert _strict(tmp_path / "spec.receipt.json")["verdict"] == "PASS"
