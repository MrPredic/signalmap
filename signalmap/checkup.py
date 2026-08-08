"""`signalmap checkup` — does this method work on YOUR recordings?

The honest answer to that question is the first thing a newcomer needs, and
until now the project could not give it. Our own study measured why it matters:
across nine public domains, one frozen recipe separated cleanly on some and
not at all on others, and on two of them it ranked faults as *more normal than
normal*. Telling someone to run `fit` and `monitor` and hope is not good enough
when we have measured that the outcome swings that hard.

So checkup runs the study's own procedure on the user's data and prints a
verdict rather than a score:

    signalmap checkup --bank recordings/ --label-by prefix

  SEPARATES      the score tells the two classes apart, CI clear of chance
  REFUSED        it does not, or the direction cannot be pinned down
  ALARM READY    the calibrated cut actually fires on the faulty recordings

It needs labelled recordings — a handful of known-bad ones is enough. That is
not a workaround: the direction of the score is not identifiable from healthy
data alone, so a few labels are the cheapest thing that makes a decision
possible at all.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .distill import DistilledDetector, FeatureSpec, _rank_auc, enumerate_programs, gate, load_bank

# The deterministic lean-base set: the cheapest survivors of the capacity gate.
# Same spec the study used on every domain, so a checkup here is comparable
# with the numbers published in study/REPORT.md.
LEAN_BASE_N = 9
N_BOOT = 2000
SEED = 0


def lean_base_spec(n_recordings: int, budget_c: int = 50) -> FeatureSpec:
    names = [p.name for p in gate(enumerate_programs(), n_recordings=n_recordings,
                                  C=budget_c)][:LEAN_BASE_N]
    return FeatureSpec(programs=names, premium=[], window=1024)


def _boot_ci(y, s, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, y.size, y.size)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(_rank_auc(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def checkup(bank_dir: str, label_by: str = "prefix", pattern: str = "*",
            column: int = 0, healthy_class: str | None = None) -> dict:
    """Run the study's procedure on one bank and return the verdict payload."""
    bank = load_bank(bank_dir, label_by=label_by, pattern=pattern, column=column)
    classes = list(bank.classes)
    if len(classes) < 2:
        return {"verdict": "REFUSED", "reason":
                f"only one class present ({classes}); checkup needs both healthy "
                f"and known-bad recordings — a handful of bad ones is enough",
                "n_recordings": int(bank.n_recordings), "classes": classes}

    healthy = healthy_class or _guess_healthy(classes)
    if healthy is None:
        return {"verdict": "REFUSED", "reason":
                f"cannot tell which class is healthy from {classes}; pass "
                f"--healthy-class", "classes": classes}

    y_rec, groups = [], []
    for g, lab in zip(bank.g, bank.y):
        groups.append(int(g))
        y_rec.append(0 if lab == healthy else 1)
    groups = np.asarray(groups)
    y_win = np.asarray(y_rec)

    # Fit on half the healthy recordings, keep the rest for evaluation, so the
    # detector is never scored on what calibrated it.
    healthy_ids = sorted({int(g) for g, lab in zip(bank.g, bank.y) if lab == healthy})
    if len(healthy_ids) < 4:
        return {"verdict": "REFUSED", "reason":
                f"only {len(healthy_ids)} healthy recordings; need at least 4 to "
                f"hold any of them out", "n_recordings": int(bank.n_recordings)}
    split = max(2, len(healthy_ids) // 2)
    fit_ids = set(healthy_ids[:split])

    windows = list(bank.windows)
    fit_windows = [w for w, g in zip(windows, groups) if int(g) in fit_ids]
    spec = lean_base_spec(len(fit_ids))
    det = DistilledDetector.fit(spec, fit_windows, envelope=3.0)

    eval_idx = [i for i, g in enumerate(groups) if int(g) not in fit_ids]
    per_rec: dict[int, list[float]] = {}
    per_lab: dict[int, int] = {}
    for i in eval_idx:
        g = int(groups[i])
        per_rec.setdefault(g, []).append(det.score(windows[i]))
        per_lab[g] = int(y_win[i])
    rec_ids = sorted(per_rec)
    y = np.array([per_lab[g] for g in rec_ids])
    s = np.array([float(np.mean(per_rec[g])) for g in rec_ids])
    if np.unique(y).size < 2:
        return {"verdict": "REFUSED", "reason":
                "the held-out half contains only one class", "classes": classes}

    auc = _rank_auc(y, s)
    lo, hi = _boot_ci(y, s)
    if hi < 0.5:
        direction, separates = "inverted", True
    elif lo > 0.5:
        direction, separates = "aligned", True
    else:
        direction, separates = "undetermined", False

    verdict = "SEPARATES" if separates else "REFUSED"

    # Calibrate the decision from the same labelled recordings and report
    # whether an alarm would actually fire — the study found ranking and
    # firing are not the same thing.
    alarm = None
    if separates:
        anchors = [windows[i] for i in eval_idx]
        labels = [int(y_win[i]) for i in eval_idx]
        grp = [int(groups[i]) for i in eval_idx]
        dv = det.calibrate_direction(anchors, labels, groups=grp)
        if dv.identified:
            # The cut was calibrated on recording means, so judge recordings.
            # Counting windows against a recording-level cut is the same unit
            # mismatch that made calibration contradict itself.
            dcut, sign = det.decision_cut, det.direction
            fired = {g: (s_ >= dcut if sign == 1 else s_ <= dcut)
                     for g, s_ in zip(rec_ids, s)}
            hit = float(np.mean([fired[g] for g in rec_ids if per_lab[g] == 1]))
            fp = float(np.mean([fired[g] for g in rec_ids if per_lab[g] == 0]))
            alarm = {"sign": dv.sign, "cut": det.decision_cut,
                     "hit_rate": hit, "false_alarm_rate": fp,
                     "ready": bool(hit > 0.5 and fp < 0.2)}

    return {"verdict": verdict, "direction": direction,
            "auc": auc, "ci_lo": lo, "ci_hi": hi,
            "n_recordings": int(bank.n_recordings),
            "n_fit_recordings": len(fit_ids),
            "n_eval_recordings": len(rec_ids),
            "n_eval_faulty": int(y.sum()),
            "healthy_class": healthy, "classes": classes,
            "spec_programs": list(spec.programs),
            "alarm": alarm,
            "degenerate": [p for p, d in zip(spec.programs,
                                             np.asarray(det.degenerate, dtype=bool))
                           if d] if det.degenerate is not None else []}


def _guess_healthy(classes: list[str]) -> str | None:
    """Pick the healthy class by the naming the rest of the tool already uses."""
    for c in classes:
        low = c.lower()
        if low.startswith("normal") or low in ("healthy", "ok", "good", "baseline"):
            return c
    anomalyish = [c for c in classes if c.lower().startswith(("anomaly", "fault",
                                                              "bad", "broken"))]
    rest = [c for c in classes if c not in anomalyish]
    return rest[0] if len(rest) == 1 and anomalyish else None


def render(res: dict) -> str:
    """The whole point is that a newcomer can read this without a manual."""
    L = []
    if res["verdict"] == "REFUSED" and "auc" not in res:
        L.append("REFUSED — checkup could not run")
        L.append(f"  {res['reason']}")
        return "\n".join(L)

    a = res.get("alarm")
    L.append(f"{res['verdict']} — AUC {res['auc']:.3f}, 95% CI "
             f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}] over "
             f"{res['n_eval_recordings']} held-out recordings "
             f"({res['n_eval_faulty']} faulty)")
    L.append("")
    if res["verdict"] == "SEPARATES":
        if res["direction"] == "inverted":
            L.append("  Direction: INVERTED — your faults score LOWER than healthy.")
            L.append("  That is a real pattern, not a bug: faults here are more")
            L.append("  stereotyped than normal operation. A detector that assumed")
            L.append("  'far from healthy means faulty' would be worse than useless")
            L.append("  on this data.")
        else:
            L.append("  Direction: ALIGNED — faults score higher than healthy, the")
            L.append("  orientation most tools assume.")
    else:
        L.append("  The score does not tell your two classes apart: the confidence")
        L.append("  interval covers chance. No amount of threshold tuning fixes")
        L.append("  that, so nothing is claimed.")

    L.append("")
    if a and a.get("ready"):
        L.append(f"  ALARM READY — cut {a['cut']:.4g} catches "
                 f"{a['hit_rate'] * 100:.0f}% of faulty recordings at "
                 f"{a['false_alarm_rate'] * 100:.0f}% false alarms.")
        L.append("  Next: signalmap distill --bank <dir> --label-by prefix "
                 "--out spec.json")
    elif a:
        L.append(f"  Alarm NOT ready — the calibrated cut catches "
                 f"{a['hit_rate'] * 100:.0f}% at "
                 f"{a['false_alarm_rate'] * 100:.0f}% false alarms.")
        L.append("  Ranking works but a usable operating point does not exist yet;")
        L.append("  more labelled examples usually fix this before more features do.")
    else:
        L.append("  No alarm calibrated: without a decided direction there is no")
        L.append("  honest cut to place.")

    if res.get("degenerate"):
        L.append("")
        L.append(f"  Note: {', '.join(res['degenerate'])} carries no variation on "
                 f"your healthy data and was floored.")
    return "\n".join(L)


def run_cli(bank: str, label_by: str = "prefix", pattern: str = "*",
            column: int = 0, healthy_class: str | None = None,
            out: str | None = None) -> dict:
    res = checkup(bank, label_by=label_by, pattern=pattern, column=column,
                  healthy_class=healthy_class)
    print(render(res))
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2, sort_keys=True, default=float)
        print(f"\nwritten: {out}")
    return res
