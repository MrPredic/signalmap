"""DCASE2020 Task2 valve/id_00 external-user-simulation readout.
Prereg: PREREG_DCASE_VALVE_EXTERNAL.md (frozen before readout, git 9963266).
Banks built by dcase_valve_adapter.py. Run PRIMARY first (fast: fit+monitor+
AUC), then SECONDARY (distill+envelope premium receipt on the 30-recording
distill/ bank — this ALSO produces the spec.json that PRIMARY's fit uses, so
secondary runs first in wall-clock terms even though it is reported second).

Run: cd <local-path>/signalmap && source .venv-research/bin/activate && \
     nice -n 19 python research/factory/dcase_valve_readout.py
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from receipt_ledger import log_receipt  # noqa: E402

from signalmap.distill import Bank, DistilledDetector, FeatureSpec, distill, load_bank  # noqa: E402

BANK = os.path.join(ROOT, "data", "mimii", "valve_id00_bank")
LOGS = os.path.join(HERE, "logs")
os.makedirs(LOGS, exist_ok=True)

SPEC_PATH = os.path.join(LOGS, "dcase_valve_distill_envelope_spec.json")
DET_PATH = os.path.join(LOGS, "dcase_valve_detector.json")


def _sorted_files(bank_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(bank_dir, "*.npy")))


def _label(stem: str) -> int:
    return 1 if stem.startswith("anomaly_") else 0


# --------------------------------------------------------------- SECONDARY
def run_secondary() -> dict:
    """distill + envelope premium challenger on the 30-recording distill/ bank.
    Fixed params per prereg §Fixe Parameter. Also produces spec.json used by
    PRIMARY's fit step."""
    bank = load_bank(os.path.join(BANK, "distill"), label_by="prefix")
    print(f"[secondary] distill bank: {bank.n_recordings} recordings, "
          f"{len(bank.windows)} windows, classes {bank.classes}", flush=True)
    t0 = time.time()
    res = distill(bank, C=50, kmax=5, thr=0.005, n_perm=100, trees=100,
                  cand=60, seed=0, null_check=True, premium=("envelope",))
    runtime_s = round(time.time() - t0, 1)
    rec = res.premium_receipts[0]
    with open(os.path.join(LOGS, "dcase_valve_distill_envelope_report.md"), "w") as fh:
        fh.write(res.report)
    res.spec.save(SPEC_PATH)
    payload = {
        "prereg": "PREREG_DCASE_VALVE_EXTERNAL.md", "bank": "dcase_valve_distill",
        "family": "envelope", "n_recordings": bank.n_recordings,
        "n_windows": len(bank.windows), "chance": bank.chance,
        "distill_pass": bool(res.passed), "nested_acc": res.nested_acc,
        "forged_acc": res.forged_acc, "lean_acc": res.lean_acc,
        "p_forged": res.p_forged, "null_acc": res.null_acc,
        "premium": rec, "spec_premium": res.spec.premium, "runtime_s": runtime_s,
    }
    log_receipt("DCASE-VALVE-DISTILL-ENVELOPE", payload)
    print(res.report, flush=True)
    print(f"[secondary] verdict: {'INCLUDED' if rec['included'] else 'EXCLUDED'} "
          f"(paired {rec['delta']:+.3f} CI [{rec['ci_lo']:+.3f}, {rec['ci_hi']:+.3f}], "
          f"~{rec['cost_ms'] / max(rec.get('base_cost_ms', 1e-9), 1e-9):.1f}x cost, "
          f"cost_ms={rec['cost_ms']:.4f}) | distill gates: "
          f"{'PASS' if res.passed else 'FAIL'} | {runtime_s}s", flush=True)
    return payload


# ----------------------------------------------------------------- PRIMARY
def run_primary() -> dict:
    """fit on train/ (healthy) with the distilled spec, monitor test/ (204
    held-out clips), AUC + bootstrap CI + TPR@FPR=0.1."""
    if not os.path.exists(SPEC_PATH):
        raise SystemExit(f"{SPEC_PATH} missing — run secondary (distill) first, "
                          f"it produces the spec.json PRIMARY fits against")
    spec = FeatureSpec.load(SPEC_PATH)

    train_dir = os.path.join(BANK, "train")
    train_bank = load_bank(train_dir, label_by="stem")
    det = DistilledDetector.fit(spec, train_bank.windows, envelope=3.0)
    det.save(DET_PATH)
    print(f"[primary] fit on {len(train_bank.windows)} healthy windows from "
          f"{train_bank.n_recordings} train recordings -> threshold "
          f"{det.threshold:.4g}", flush=True)

    test_dir = os.path.join(BANK, "test")
    files = _sorted_files(test_dir)
    test_bank = load_bank(test_dir, label_by="stem")
    # bank.g is a per-file integer id assigned in the same sorted-glob order
    # load_bank uses internally -> gid indexes directly into `files`.
    per_clip_scores: dict[int, list[float]] = {}
    for w, g in zip(test_bank.windows, test_bank.g):
        per_clip_scores.setdefault(int(g), []).append(det.score(w))

    y_true, y_score, stems = [], [], []
    for gid in sorted(per_clip_scores):
        stem = os.path.splitext(os.path.basename(files[gid]))[0]
        stems.append(stem)
        y_true.append(_label(stem))
        y_score.append(float(np.mean(per_clip_scores[gid])))
    y_true = np.array(y_true); y_score = np.array(y_score)
    print(f"[primary] monitor: {len(y_true)} held-out clips "
          f"({int(y_true.sum())} anomaly, {int((1 - y_true).sum())} normal)", flush=True)

    from sklearn.metrics import roc_auc_score, roc_curve
    auc = float(roc_auc_score(y_true, y_score))

    rng = np.random.default_rng(0)
    n = len(y_true)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_score[idx]))
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    tpr_at_fpr01 = float(np.interp(0.1, fpr, tpr))

    print(f"[primary] AUC={auc:.4f} 95% CI [{ci_lo:.4f}, {ci_hi:.4f}] "
          f"(bootstrap n=2000 over {n} clips, seed=0) | "
          f"TPR@FPR=0.1 = {tpr_at_fpr01:.4f}", flush=True)

    payload = {
        "prereg": "PREREG_DCASE_VALVE_EXTERNAL.md", "n_clips": int(n),
        "n_anomaly": int(y_true.sum()), "n_normal": int((1 - y_true).sum()),
        "auc": auc, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "tpr_at_fpr01": tpr_at_fpr01, "threshold": det.threshold,
        "spec_programs": spec.programs, "spec_premium": spec.premium,
    }
    with open(os.path.join(LOGS, "dcase_valve_readout_report.md"), "w") as fh:
        fh.write(
            f"# DCASE valve id_00 external-user readout\n\n"
            f"spec programs: {spec.programs}\nspec premium: {spec.premium}\n\n"
            f"AUC = {auc:.4f}, 95% CI [{ci_lo:.4f}, {ci_hi:.4f}] "
            f"(bootstrap n=2000 over {n} held-out clips, seed=0)\n"
            f"TPR@FPR=0.1 = {tpr_at_fpr01:.4f}\n"
            f"threshold = {det.threshold:.6g}\n"
            f"n_clips={n} (anomaly={int(y_true.sum())}, normal={int((1 - y_true).sum())})\n")
    log_receipt("DCASE-VALVE-EXTERNAL", payload)
    return payload


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--primary-only"]:
        run_primary()
    elif args == ["--secondary-only"]:
        run_secondary()
    else:
        run_secondary()
        run_primary()
