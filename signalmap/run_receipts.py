"""Signed receipts for `signalmap distill` runs.

``distill`` produces the verdicts the product is about: the base gate (PASS,
or a written REFUSED when the gauntlet does not clear) and one
INCLUDED/EXCLUDED per premium family. Each is emitted as a
``signalmap.receipt/1`` receipt whose evidence has the same shape as the
archived corpus (``signalmap.corpus``), so one reader parses both — the only
difference is ``evidence.provenance.mode``: ``run`` here, ``archive_signature``
there.

A refused gate carries no ``deploy_spec``: the spec file is still written for
inspection, but the receipt does not endorse it, and the standalone verifier
enforces that pairing.
"""
from __future__ import annotations

import math
import os
from pathlib import Path


def _num(x):
    """JSON-safe float: NaN/Inf become null rather than invalid JSON."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def base_evidence(res, *, bank_label: str) -> dict:
    return {
        "verdict": "PASS" if res.passed else "FAIL",
        "n_recordings": int(res.spec.n_recordings),
        "chance": _num(res.chance),
        "nested_logo": _num(res.nested_acc),
        "deploy_selection": _num(res.forged_acc),
        "lean_baseline": _num(res.lean_acc),
        "perm_p": _num(res.p_forged),
        "null_selftest": _num(res.null_acc),
        "cost_ms_per_window": _num(res.cost_ms),
        "budget": int(res.budget),
        "grammar_total": int(res.grammar_total),
        "bank": bank_label,
    }


def premium_evidence(rec: dict) -> dict:
    ratio = rec["cost_ms"] / max(rec["base_cost_ms"], 1e-9)
    return {
        "family": rec["family"],
        "verdict": "INCLUDED" if rec["included"] else "EXCLUDED",
        "base_acc": _num(rec["base_acc"]),
        "augmented_acc": _num(rec["aug_acc"]),
        "paired_delta": _num(rec["delta"]),
        "ci95": [_num(rec["ci_lo"]), _num(rec["ci_hi"])],
        "cost_ms_per_window": _num(rec["cost_ms"]),
        "base_cost_ms_per_window": _num(rec["base_cost_ms"]),
        "cost_ratio_vs_base": _num(ratio),
        "cost_note": rec["cost_note"],
    }


def _provenance(bank_path, spec_path, report_path) -> dict:
    prov = {
        "mode": "run",
        "command": "signalmap distill",
        "bank_path": str(bank_path),
        "spec": str(spec_path),
    }
    if report_path is not None:
        prov["report"] = str(report_path)
    return prov


def emit_distill_receipts(res, *, bank_path, spec_path,
                          report_path=None) -> list[Path]:
    """Sign the base verdict and every premium verdict of one distill run."""
    from .receipt import emit_signed_receipt, hash_bank, hash_file

    spec_path = Path(spec_path)
    stem = spec_path.with_suffix("")
    bank_label = os.path.basename(str(bank_path).rstrip("/")) or str(bank_path)
    prov = _provenance(bank_path, spec_path, report_path)
    hashes = {"bank": hash_bank(bank_path), "spec": hash_file(spec_path)}
    if report_path is not None and Path(report_path).is_file():
        hashes["report"] = hash_file(report_path)

    gate = base_evidence(res, bank_label=bank_label)
    evidence = {"bank": bank_label, "base_gate": gate, "provenance": prov}
    if res.passed:
        evidence["deploy_spec"] = list(res.spec.programs)
        claim = (f"distilled base grammar on bank {bank_label} clears the "
                 f"gauntlet (nested LOGO {gate['nested_logo']}, "
                 f"permutation p {gate['perm_p']})")
    else:
        evidence["refusal"] = (
            "gauntlet not cleared — no deploy spec is endorsed by this "
            "receipt; spec.json was written for inspection only")
        claim = (f"distilled base grammar on bank {bank_label} does NOT clear "
                 f"the gauntlet — deployment refused")

    paths = [Path(f"{stem}.receipt.json")]
    emit_signed_receipt(claim=claim,
                        verdict="PASS" if res.passed else "REFUSED",
                        evidence=evidence, input_hashes=hashes,
                        out_path=paths[0])

    for rec in res.premium_receipts:
        prem = premium_evidence(rec)
        clears = "clears" if rec["included"] else "does not clear"
        out = Path(f"{stem}.{rec['family']}.receipt.json")
        emit_signed_receipt(
            claim=(f"premium family {rec['family']!r} on bank {bank_label} "
                   f"{clears} the cost-receipted champion rule over the "
                   f"distilled base grammar"),
            verdict=prem["verdict"],
            evidence={"bank": bank_label, "family": rec["family"],
                      "premium": prem, "base_gate": gate, "provenance": prov},
            input_hashes=hashes, out_path=out)
        paths.append(out)
    return paths
