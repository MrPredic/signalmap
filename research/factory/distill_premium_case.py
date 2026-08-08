"""distill Premium-Familie Praxis-Case — Prereg: PREREG_DISTILL_PREMIUM_CALCE.md /
PREREG_DISTILL_PREMIUM_CWRU.md / PREREG_DISTILL_PREMIUM_ENVELOPE_{CWRU,MFPT,IMS}.md
(frozen VOR Readout). Fixe Parameter, keine Suche; EXCLUDED ist ein gültiges Ergebnis.

Run: cd research/factory && ../../.venv-research/bin/python3 distill_premium_case.py [--family=rqa|envelope] BANK...
Default family is "rqa" (unchanged, backward compatible).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # signalmap package
sys.path.insert(0, HERE)

from receipt_ledger import log_receipt  # noqa: E402
from retro_loaders import load_calce, load_hydraulic, load_ims  # noqa: E402

from signalmap.distill import Bank, distill  # noqa: E402

def _load_cwru():
    from feature_forge import load_cwru
    return load_cwru()


def _load_mfpt():
    from mfpt_run import load_mfpt
    return load_mfpt()


BANKS = {
    "calce": lambda: load_calce(per_stage=2),
    "hydcooler": lambda: load_hydraulic(target="cooler"),
    # PREREG_DISTILL_PREMIUM_CWRU.md (14. Jul, frozen before readout)
    "cwru": _load_cwru,
    # PREREG_DISTILL_PREMIUM_ENVELOPE_{CWRU,MFPT,IMS}.md (19. Jul, envelope family #3)
    "mfpt": _load_mfpt,
    "ims": load_ims,
}

PREREG = {
    "calce": "PREREG_DISTILL_PREMIUM_CALCE.md",
    "hydcooler": "PREREG_DISTILL_PREMIUM_CALCE.md",
    "cwru": "PREREG_DISTILL_PREMIUM_CWRU.md",
    "mfpt": "PREREG_DISTILL_PREMIUM_ENVELOPE.md",
    "ims": "PREREG_DISTILL_PREMIUM_ENVELOPE.md",
}

# envelope readout on cwru reuses the same bank as the rqa case but a
# different prereg document (family switched via --family=envelope); one
# combined prereg with a per-bank section, mirroring
# PREREG_DISTILL_PREMIUM_COHERENCE.md's structure (hydcooler+gasid).
PREREG_ENVELOPE = {
    "cwru": "PREREG_DISTILL_PREMIUM_ENVELOPE.md",
    "mfpt": "PREREG_DISTILL_PREMIUM_ENVELOPE.md",
    "ims": "PREREG_DISTILL_PREMIUM_ENVELOPE.md",
}


def to_bank(raw) -> Bank:
    wins = [np.asarray(r[0], float) for r in raw]
    y = np.array([r[1] for r in raw])
    g = np.array([r[2] for r in raw])
    return Bank(wins, y, g, sorted(set(y.tolist())), len(np.unique(g)))


def run(name: str, family: str = "rqa") -> None:
    bank = to_bank(BANKS[name]())
    print(f"== {name}: {bank.n_recordings} recordings, {len(bank.windows)} windows, "
          f"classes {bank.classes}", flush=True)
    t0 = time.time()
    res = distill(bank, C=50, kmax=5, thr=0.005, n_perm=200, trees=100,
                  cand=60, seed=0, null_check=True, premium=(family,))
    rec = res.premium_receipts[0]
    tag = name if family == "rqa" else f"{name}_{family}"
    report_path = os.path.join(HERE, "logs", f"distill_premium_{tag}_report.md")
    with open(report_path, "w") as fh:
        fh.write(res.report)
    res.spec.save(os.path.join(HERE, "logs", f"distill_premium_{tag}_spec.json"))
    prereg = PREREG_ENVELOPE[name] if family == "envelope" else PREREG[name]
    payload = {
        "prereg": prereg,
        "bank": name, "family": family, "n_recordings": bank.n_recordings,
        "n_windows": len(bank.windows), "chance": bank.chance,
        "distill_pass": bool(res.passed), "nested_acc": res.nested_acc,
        "forged_acc": res.forged_acc, "lean_acc": res.lean_acc,
        "p_forged": res.p_forged, "null_acc": res.null_acc,
        "premium": rec, "spec_premium": res.spec.premium,
        "runtime_s": round(time.time() - t0, 1),
    }
    ledger_tag = name.upper() if family == "rqa" else f"{name.upper()}-{family.upper()}"
    log_receipt(f"DISTILL-PREMIUM-{ledger_tag}", payload)
    print(res.report, flush=True)
    print(f"[{name}/{family}] PRIMARY verdict: "
          f"{'INCLUDED' if rec['included'] else 'EXCLUDED'} "
          f"(paired {rec['delta']:+.3f} CI [{rec['ci_lo']:+.3f}, {rec['ci_hi']:+.3f}], "
          f"~{rec['cost_ms'] / max(rec.get('base_cost_ms', 1e-9), 1e-9):.1f}x cost) "
          f"| distill gates: {'PASS' if res.passed else 'FAIL'} "
          f"| {payload['runtime_s']}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    fam = "rqa"
    for a in list(args):
        if a.startswith("--family="):
            fam = a.split("=", 1)[1]
            args.remove(a)
    for name in (args or ["calce", "hydcooler"]):
        run(name, family=fam)
