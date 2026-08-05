"""The verdict corpus is the traction line.

The eight preregistered premium verdicts (rqa 1 IN/2 EX · coherence 1 IN/1 EX ·
envelope 0 IN/3 EX) predate the receipt format, so they are signed as
**archive signatures**, never as re-runs: every receipt says so in
``evidence.provenance`` and pins the sha256 of the report it was transcribed
from. The staleness gate below is the point — a shipped receipt whose source
report has moved on is exactly the failure mode that broke HEDG3's headline.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from signalmap import corpus
from signalmap.receipt import verify_receipt

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "tools" / "verify_receipt.py"


def _verifier_module():
    spec = importlib.util.spec_from_file_location("verify_receipt_mod", VERIFIER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALMAP_HOME", str(tmp_path / "home"))
    out = tmp_path / "receipts"
    paths = corpus.build_corpus(out_dir=out)
    return out, paths


# --- parsing -----------------------------------------------------------

def test_parse_cwru_report_extracts_base_and_premium():
    rep = corpus.parse_premium_report(
        ROOT / "research/factory/logs/distill_premium_cwru_report.md")
    base = rep["base"]
    assert base["verdict"] == "PASS"
    assert base["n_recordings"] == 24
    assert base["n_windows"] == 2849
    assert base["chance"] == 0.167
    assert base["nested_logo"] == 0.874
    assert base["deploy_selection"] == 0.914
    assert base["perm_p"] == 0.005
    assert base["null_selftest"] == 0.082
    assert base["cost_ms_per_window"] == 0.113

    (prem,) = rep["premium"]
    assert prem["family"] == "rqa"
    assert prem["verdict"] == "INCLUDED"
    assert prem["base_acc"] == 0.914
    assert prem["augmented_acc"] == 0.980
    assert prem["paired_delta"] == 0.066
    assert prem["ci95"] == [0.033, 0.106]
    assert prem["cost_ms_per_window"] == 66.03
    assert prem["cost_ratio_vs_base"] == 585.0


def test_parse_keeps_negative_delta_and_ci_signs():
    rep = corpus.parse_premium_report(
        ROOT / "research/factory/logs/distill_premium_calce_report.md")
    (prem,) = rep["premium"]
    assert prem["verdict"] == "EXCLUDED"
    assert prem["paired_delta"] == -0.030
    assert prem["ci95"] == [-0.072, 0.006]


# --- manifest ----------------------------------------------------------

def test_manifest_matches_the_preregistered_tally():
    entries = corpus.MANIFEST
    assert len(entries) == 8
    by_family = {}
    for e in entries:
        assert (ROOT / e.report).is_file(), e.report
        by_family.setdefault(e.family, []).append(e)
    assert set(by_family) == {"rqa", "coherence", "envelope"}
    assert len(by_family["rqa"]) == 3
    assert len(by_family["coherence"]) == 2
    assert len(by_family["envelope"]) == 3
    assert len({e.bank for e in entries}) == 6


def test_every_manifest_entry_is_present_in_its_report():
    for e in corpus.MANIFEST:
        rep = corpus.parse_premium_report(ROOT / e.report)
        families = {p["family"] for p in rep["premium"]}
        assert e.family in families, f"{e.report} has {families}, want {e.family}"


# --- receipts ----------------------------------------------------------

def test_build_corpus_signs_every_verdict(built):
    out, paths = built
    assert len(paths) == 8
    for p in paths:
        r = json.loads(Path(p).read_text())
        assert r["schema"] == "signalmap.receipt/1"
        assert r["verdict"] in ("INCLUDED", "EXCLUDED")
        assert verify_receipt(r), p


def test_receipts_are_labelled_archive_signatures_not_reruns(built):
    _, paths = built
    for p in paths:
        prov = json.loads(Path(p).read_text())["evidence"]["provenance"]
        assert prov["mode"] == "archive_signature"
        assert prov["rerun"] is False
        src = ROOT / prov["source_report"]
        assert prov["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
        assert "not" in prov["note"].lower()


def test_verdict_counts_match_the_reports(built):
    _, paths = built
    verdicts = [json.loads(Path(p).read_text())["verdict"] for p in paths]
    assert verdicts.count("INCLUDED") == 2
    assert verdicts.count("EXCLUDED") == 6


def test_built_receipts_pass_the_standalone_verifier(built, tmp_path):
    _, paths = built
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "signalmap.py").write_text(
        "raise ImportError('verifier must not import signalmap')\n")
    for p in paths:
        proc = subprocess.run(
            [sys.executable, str(VERIFIER), str(p)], capture_output=True,
            text=True, check=False,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(poison)})
        assert proc.returncode == 0, proc.stdout + proc.stderr


def test_rebuild_is_idempotent_when_sources_are_unchanged(built):
    out, paths = built
    before = {p: Path(p).read_text() for p in paths}
    again = corpus.build_corpus(out_dir=out)
    assert set(again) == set(paths)
    assert {p: Path(p).read_text() for p in again} == before


def test_rebuild_rewrites_when_the_source_report_changes(built, tmp_path,
                                                         monkeypatch):
    out, paths = built
    target = next(p for p in paths if "cwru_rqa" in Path(p).name)
    before = json.loads(Path(target).read_text())
    monkeypatch.setattr(corpus, "_source_digest",
                        lambda path: "00" * 32)
    corpus.build_corpus(out_dir=out)
    after = json.loads(Path(target).read_text())
    assert after["evidence"]["provenance"]["source_sha256"] == "00" * 32
    assert after["signature"] != before["signature"]


# --- staleness gate on the committed corpus ----------------------------

def test_committed_receipts_are_not_stale():
    """Every shipped receipt must pin the *current* report digest."""
    receipts = corpus.load_corpus()
    assert len(receipts) == 8, "committed corpus incomplete"
    for r in receipts:
        prov = r["evidence"]["provenance"]
        src = ROOT / prov["source_report"]
        assert prov["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest(), (
            f"{prov['source_report']} changed after the receipt was signed — "
            "rerun `signalmap corpus`")
        assert verify_receipt(r)


# --- verifier consistency check ----------------------------------------

def test_verifier_rejects_an_archive_receipt_claiming_a_rerun(tmp_path):
    from signalmap.receipt import build_receipt, load_or_create_key, sign_receipt
    mod = _verifier_module()
    ev = {"provenance": {"mode": "archive_signature", "rerun": True,
                         "source_report": "x.md", "source_sha256": "ab" * 32}}
    r = sign_receipt(build_receipt(claim="c", verdict="INCLUDED", evidence=ev,
                                   input_hashes={}),
                     load_or_create_key(tmp_path / "k"))
    errors = mod.check(r)
    assert any("rerun" in e for e in errors), errors


def test_verifier_rejects_an_archive_receipt_without_a_source_digest(tmp_path):
    from signalmap.receipt import build_receipt, load_or_create_key, sign_receipt
    mod = _verifier_module()
    ev = {"provenance": {"mode": "archive_signature", "rerun": False}}
    r = sign_receipt(build_receipt(claim="c", verdict="EXCLUDED", evidence=ev,
                                   input_hashes={}),
                     load_or_create_key(tmp_path / "k"))
    errors = mod.check(r)
    assert any("source_sha256" in e for e in errors), errors


def test_verifier_still_passes_a_plain_receipt(tmp_path):
    from signalmap.receipt import build_receipt, load_or_create_key, sign_receipt
    mod = _verifier_module()
    r = sign_receipt(build_receipt(claim="c", verdict="PASS", evidence={},
                                   input_hashes={}),
                     load_or_create_key(tmp_path / "k"))
    assert mod.check(r) == []


# --- traction line -----------------------------------------------------

def test_traction_line_counts_verdicts_banks_and_exclusions(built):
    _, paths = built
    line = corpus.traction_line([json.loads(Path(p).read_text()) for p in paths])
    assert "8 verdicts" in line
    assert "6 banks" in line
    assert "6 honest exclusions" in line
    assert "offline verifiable" in line


def test_cli_corpus_command_prints_the_traction_line(tmp_path, monkeypatch,
                                                     capsys):
    from signalmap.cli import main
    monkeypatch.setenv("SIGNALMAP_HOME", str(tmp_path / "home"))
    out = tmp_path / "receipts"
    rc = main(["corpus", "--out", str(out)])
    assert rc in (0, None)
    printed = capsys.readouterr().out
    assert "8 verdicts" in printed
    assert len(list(out.glob("*.receipt.json"))) == 8


def test_cli_corpus_shortens_paths_that_sit_under_the_working_directory(
        tmp_path, monkeypatch, capsys):
    """`corpus` defaults to an absolute path under the repo root, so the listing
    used to print the maintainer's home directory eight times. The traction line
    gets pasted into issues and READMEs — the paths above it should be local."""
    from signalmap.cli import main
    monkeypatch.setenv("SIGNALMAP_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    out = Path.cwd() / "receipts"          # absolute, but under the cwd
    rc = main(["corpus", "--out", str(out)])
    assert rc in (0, None)
    printed = capsys.readouterr().out
    assert "receipts/cwru_rqa.receipt.json" in printed
    assert str(out) not in printed


def test_cli_corpus_keeps_absolute_paths_outside_the_working_directory(
        tmp_path, monkeypatch, capsys):
    from signalmap.cli import main
    monkeypatch.setenv("SIGNALMAP_HOME", str(tmp_path / "home"))
    out = tmp_path / "receipts"
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    rc = main(["corpus", "--out", str(out)])
    assert rc in (0, None)
    printed = capsys.readouterr().out
    assert str(Path(out).resolve() / "cwru_rqa.receipt.json") in printed
