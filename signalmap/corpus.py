"""V4: the verdict corpus — archived premium verdicts as signed receipts.

Eight preregistered premium-family verdicts (rqa 1 IN/2 EX · coherence
1 IN/1 EX · envelope 0 IN/3 EX) were decided before the receipt format
existed. They are signed here as **archive signatures**, not as re-runs:
each receipt transcribes the numbers out of its gauntlet report, pins that
report's sha256, and says in ``evidence.provenance`` that no re-execution
happened. The signature therefore attests to origin and integrity of the
transcription — never to correctness of the statistics, and never to a
reproduction that did not take place.

A receipt whose source report has changed since signing is stale by
definition; ``load_corpus`` plus the suite gate in ``tests/test_corpus.py``
make that a failing test rather than a silent claim.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

LOGS = "research/factory/logs"
PREREG = "research/factory"
CORPUS_DIR = "research/factory/receipts"

NOTE = ("Archive signature: the verdict was decided in the preregistered run "
        "cited below and is transcribed here, NOT re-executed. The signature "
        "attests to origin and integrity of this transcription, not to "
        "correctness of the statistics.")


@dataclass(frozen=True)
class ArchiveEntry:
    bank: str
    family: str
    report: str
    spec: str
    prereg: str


def _e(bank: str, family: str, stem: str, prereg: str) -> ArchiveEntry:
    return ArchiveEntry(bank=bank, family=family,
                        report=f"{LOGS}/{stem}_report.md",
                        spec=f"{LOGS}/{stem}_spec.json",
                        prereg=f"{PREREG}/{prereg}")


MANIFEST = (
    _e("CWRU", "rqa", "distill_premium_cwru",
       "PREREG_DISTILL_PREMIUM_CWRU.md"),
    _e("CALCE", "rqa", "distill_premium_calce",
       "PREREG_DISTILL_PREMIUM_CALCE.md"),
    _e("HYD-cooler", "rqa", "distill_premium_hydcooler",
       "PREREG_DISTILL_PREMIUM_CALCE.md"),
    _e("HYD-cooler", "coherence", "distill_premium_coh_hydcooler",
       "PREREG_DISTILL_PREMIUM_COHERENCE.md"),
    _e("GAS-id", "coherence", "distill_premium_coh_gasid",
       "PREREG_DISTILL_PREMIUM_COHERENCE.md"),
    _e("CWRU", "envelope", "distill_premium_cwru_envelope",
       "PREREG_DISTILL_PREMIUM_ENVELOPE.md"),
    _e("IMS", "envelope", "distill_premium_ims_envelope",
       "PREREG_DISTILL_PREMIUM_ENVELOPE.md"),
    _e("MFPT", "envelope", "distill_premium_mfpt_envelope",
       "PREREG_DISTILL_PREMIUM_ENVELOPE.md"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# --- report parsing ----------------------------------------------------

_NUM = r"[-+]?[\d.]+"
_BASE_FIELDS = {
    "verdict": (r"^- verdict: (\w+)", str),
    "nested_logo": (rf"^- nested LOGO[^:]*: ({_NUM})", float),
    "deploy_selection": (rf"^- deploy selection[^:]*: ({_NUM})", float),
    "perm_p": (rf"^- group-permutation p: ({_NUM})", float),
    "null_selftest": (rf"^- NULL self-test[^:]*: ({_NUM})", float),
    "cost_ms_per_window": (rf"^- cost: ({_NUM}) ms/window", float),
}

_COUNTS = re.compile(rf"^- recordings: (\d+) . windows: (\d+) . chance: ({_NUM})",
                     re.M)

_PREMIUM = re.compile(
    rf"^- (?P<family>[a-z_]+): base (?P<base>{_NUM}) -> augmented "
    rf"(?P<aug>{_NUM}) \(paired (?P<delta>{_NUM}), 95% CI "
    rf"\[(?P<lo>{_NUM}), (?P<hi>{_NUM})\]\)[^\n]*?"
    rf"\*\*(?P<verdict>INCLUDED|EXCLUDED)\*\*[^\n]*\n"
    rf"\s+cost: (?P<pcost>{_NUM}) ms/window vs base (?P<bcost>{_NUM}) "
    rf"ms/window \(~(?P<ratio>{_NUM})x\)",
    re.M)


def parse_premium_report(path) -> dict:
    """Extract base gate metrics and premium-family verdicts from a report."""
    text = Path(path).read_text(encoding="utf-8")
    base: dict = {}
    for key, (pattern, cast) in _BASE_FIELDS.items():
        m = re.search(pattern, text, re.M)
        if m:
            base[key] = cast(m.group(1))
    m = _COUNTS.search(text)
    if m:
        base["n_recordings"] = int(m.group(1))
        base["n_windows"] = int(m.group(2))
        base["chance"] = float(m.group(3))

    premium = [{
        "family": g["family"],
        "verdict": g["verdict"],
        "base_acc": float(g["base"]),
        "augmented_acc": float(g["aug"]),
        "paired_delta": float(g["delta"]),
        "ci95": [float(g["lo"]), float(g["hi"])],
        "cost_ms_per_window": float(g["pcost"]),
        "base_cost_ms_per_window": float(g["bcost"]),
        "cost_ratio_vs_base": float(g["ratio"]),
    } for g in (mm.groupdict() for mm in _PREMIUM.finditer(text))]
    return {"base": base, "premium": premium}


# --- receipt construction ---------------------------------------------

def _source_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _claim(entry: ArchiveEntry, verdict: str) -> str:
    clears = "clears" if verdict == "INCLUDED" else "does not clear"
    return (f"premium family {entry.family!r} on bank {entry.bank} {clears} the "
            f"cost-receipted champion rule over the distilled base grammar")


def receipt_name(entry: ArchiveEntry) -> str:
    bank = re.sub(r"[^a-z0-9]", "", entry.bank.lower())
    return f"{bank}_{entry.family}.receipt.json"


def build_archive_receipt(entry: ArchiveEntry, root=None) -> dict:
    """Signed receipt for one archived premium verdict."""
    from .receipt import build_receipt, load_or_create_key, sign_receipt

    root = Path(root) if root is not None else repo_root()
    parsed = parse_premium_report(root / entry.report)
    try:
        prem = next(p for p in parsed["premium"] if p["family"] == entry.family)
    except StopIteration:
        raise ValueError(
            f"{entry.report} carries no verdict for family {entry.family!r}")

    evidence = {
        "bank": entry.bank,
        "family": entry.family,
        "premium": prem,
        "base_gate": parsed["base"],
        "provenance": {
            "mode": "archive_signature",
            "rerun": False,
            "source_report": entry.report,
            "source_sha256": _source_digest(root / entry.report),
            "prereg": entry.prereg,
            "note": NOTE,
        },
    }
    input_hashes = {"report": evidence["provenance"]["source_sha256"]}
    for label, rel in (("spec", entry.spec), ("prereg", entry.prereg)):
        p = root / rel
        if p.is_file():
            input_hashes[label] = _source_digest(p)

    receipt = build_receipt(claim=_claim(entry, prem["verdict"]),
                            verdict=prem["verdict"], evidence=evidence,
                            input_hashes=input_hashes)
    return sign_receipt(receipt, load_or_create_key())


def _unchanged(existing: dict, fresh: dict) -> bool:
    drop = ("created_at", "signature")
    return ({k: v for k, v in existing.items() if k not in drop}
            == {k: v for k, v in fresh.items() if k not in drop})


def build_corpus(out_dir=None, root=None) -> list[Path]:
    """(Re)build every archive receipt; untouched ones keep their signature."""
    from .receipt import write_receipt
    root = Path(root) if root is not None else repo_root()
    out = Path(out_dir) if out_dir is not None else root / CORPUS_DIR
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for entry in MANIFEST:
        path = out / receipt_name(entry)
        fresh = build_archive_receipt(entry, root=root)
        if path.is_file():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError:
                existing = None
            if existing is not None and _unchanged(existing, fresh):
                paths.append(path)
                continue
        write_receipt(fresh, path)
        paths.append(path)
    return sorted(paths)


def load_corpus(corpus_dir=None) -> list[dict]:
    root = repo_root()
    d = Path(corpus_dir) if corpus_dir is not None else root / CORPUS_DIR
    return [json.loads(p.read_text())
            for p in sorted(d.glob("*.receipt.json"))]


def traction_line(receipts) -> str:
    banks = {r["evidence"]["bank"] for r in receipts}
    excluded = sum(1 for r in receipts if r["verdict"] == "EXCLUDED")
    included = len(receipts) - excluded
    return (f"{len(receipts)} verdicts across {len(banks)} banks · "
            f"{included} included with a cost receipt · "
            f"{excluded} honest exclusions · 0 silent adoptions · "
            f"offline verifiable (tools/verify_receipt.py)")
