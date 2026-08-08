"""distill Premium-Familie COHERENCE Praxis-Case — Prereg:
PREREG_DISTILL_PREMIUM_COHERENCE.md (frozen VOR Readout, Ledger
DISTILL-PREMIUM-COH-PREREG). Fixe Produkt-Config c128b2, keine Suche;
EXCLUDED ist ein gültiges Ergebnis.

Run: cd research/factory && nice -n 19 ../../.venv-research/bin/python3 \
     distill_premium_coherence.py [hydcooler|gasid]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # signalmap package
sys.path.insert(0, HERE)

from scipy.signal import detrend  # noqa: E402

from receipt_ledger import log_receipt  # noqa: E402

from signalmap.distill import Bank, distill  # noqa: E402

PREREG = "PREREG_DISTILL_PREMIUM_COHERENCE.md"


def _znorm_channels(X):
    """Exact window() semantics per channel (detrend + z-norm)."""
    out = []
    for ch in X:
        s = detrend(np.ascontiguousarray(ch, float))
        out.append((s - s.mean()) / (s.std() + 1e-12))
    return np.stack(out)


def load_mc_bank(name: str) -> Bank:
    if name == "hydcooler":
        from hyd_multichannel import load_hyd_mc
        raw = [(X, c, gid) for X, c, gid in load_hyd_mc("cooler")]
        channels = [f"ps{i}" for i in range(1, 7)]
    elif name == "gasid":
        from gas_multichannel import load_gas_mc
        raw = [(X, c, gid) for X, c, gid, _ in load_gas_mc()]
        channels = [f"mox{i}" for i in range(1, 9)]
    else:
        raise SystemExit(f"unknown bank {name!r}")
    wins = [_znorm_channels(r[0]) for r in raw]
    y = np.array([r[1] for r in raw])
    g = np.array([r[2] for r in raw])
    return Bank(wins, y, g, sorted(set(y.tolist())), len(np.unique(g)), channels)


def prereg() -> None:
    tip = log_receipt("DISTILL-PREMIUM-COH-PREREG", {
        "prereg": PREREG,
        "family": "coherence", "config": "c128b2 fixed (package default), no grid",
        "banks": {"hydcooler": "hyd_multichannel.load_hyd_mc('cooler'), ch0=PS1",
                  "gasid": "gas_multichannel.load_gas_mc(), natural order ch0=mox1"},
        "windows": "(C,1024) detrend+z-norm per channel (window() semantics; "
                   "declared deviation: coherence_fair ran on raw windows)",
        "distill": "C=50 kmax=5 thr=0.005 n_perm=200 trees=100 cand=60 seed=0 "
                   "null_check=True",
        "primary": "champion rule: paired LOGO (aug-base) over recordings, "
                   "10k bootstrap, INCLUDED iff CI-lo > 0; base = full distill "
                   "base selection",
        "expectation": "HYD-cooler expected INCLUDED (fair aug 0.944 vs base "
                       "0.800); GAS-id open, EXCLUDED welcome (fixed config "
                       "costs it the c256b8 points)"})
    print(f"prereg ledger tip = {tip}", flush=True)


def run(name: str) -> None:
    bank = load_mc_bank(name)
    print(f"== {name}: {bank.n_recordings} recordings, {len(bank.windows)} windows, "
          f"{len(bank.channels)} channels, classes {bank.classes}", flush=True)
    t0 = time.time()
    res = distill(bank, C=50, kmax=5, thr=0.005, n_perm=200, trees=100,
                  cand=60, seed=0, null_check=True, premium=("coherence",))
    rec = res.premium_receipts[0]
    with open(os.path.join(HERE, "logs", f"distill_premium_coh_{name}_report.md"), "w") as fh:
        fh.write(res.report)
    res.spec.save(os.path.join(HERE, "logs", f"distill_premium_coh_{name}_spec.json"))
    payload = {
        "prereg": PREREG,
        "bank": name, "n_recordings": bank.n_recordings,
        "n_windows": len(bank.windows), "channels": bank.channels,
        "chance": bank.chance,
        "distill_pass": bool(res.passed), "nested_acc": res.nested_acc,
        "forged_acc": res.forged_acc, "lean_acc": res.lean_acc,
        "p_forged": res.p_forged, "null_acc": res.null_acc,
        "premium": rec, "spec_premium": res.spec.premium,
        "runtime_s": round(time.time() - t0, 1),
    }
    log_receipt(f"DISTILL-PREMIUM-COH-{name.upper()}", payload)
    print(res.report, flush=True)
    print(f"[{name}] PRIMARY verdict: "
          f"{'INCLUDED' if rec['included'] else 'EXCLUDED'} "
          f"(base {rec['base_acc']:.3f} -> aug {rec['aug_acc']:.3f}, paired "
          f"{rec['delta']:+.3f} CI [{rec['ci_lo']:+.3f}, {rec['ci_hi']:+.3f}], "
          f"~{rec['cost_ms'] / max(rec['base_cost_ms'], 1e-9):.0f}x cost) "
          f"| distill gates: {'PASS' if res.passed else 'FAIL'} "
          f"| {payload['runtime_s']}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["prereg"]:
        prereg()
    else:
        for name in (sys.argv[1:] or ["hydcooler", "gasid"]):
            run(name)
