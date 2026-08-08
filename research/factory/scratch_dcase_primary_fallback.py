"""FALLBACK PRIMARY readout for DCASE valve id_00.

The prereg-declared distilled spec is INFEASIBLE within the 30-min wall cap
(secondary distill LOGO timed out at 31:59, exit 124, even at the frozen
30-recording bank). Per the task's step-4 fallback, this produces the headline
AUC using a MINIMAL SPEC built another way: the 9 cheapest depth-0 O(n) grammar
programs (SignalMap's own complexity-ordered gate prefix), with NO LOGO
selection, NO RandomForest, NO premium family. This is explicitly NOT the
distilled spec.json; the AUC below is labelled as a fallback-spec result.

fit on ALL 891 healthy train windows, monitor ALL 204 held-out test clips
(100 normal + 104 anomaly), AUC + bootstrap 95% CI over clips + TPR@FPR=0.1.
Everything else (banks, K, clip-mean aggregation, seed=0, 2000 bootstraps)
matches the frozen prereg PRIMARY exactly.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from signalmap.distill import (  # noqa: E402
    DistilledDetector, FeatureSpec, enumerate_programs, load_bank,
)

BANK = os.path.join(ROOT, "data", "mimii", "valve_id00_bank")
LOGS = os.path.join(HERE, "logs")

# Minimal fallback spec: 9 cheapest depth-0 pools on the raw window.
CHEAP = [p.name for p in enumerate_programs() if p.complexity()[0] == 0
         and p.complexity()[1] == 1]
print(f"[fallback] minimal spec programs ({len(CHEAP)}): {CHEAP}", flush=True)
spec = FeatureSpec(programs=CHEAP, classes=["anomaly", "normal"], premium=[])


def _label(stem: str) -> int:
    return 1 if stem.startswith("anomaly_") else 0


def _sorted_files(d):
    return sorted(glob.glob(os.path.join(d, "*.npy")))


train_bank = load_bank(os.path.join(BANK, "train"), label_by="stem")
det = DistilledDetector.fit(spec, train_bank.windows, envelope=3.0)
print(f"[primary] fit on {len(train_bank.windows)} healthy windows from "
      f"{train_bank.n_recordings} train recordings -> threshold "
      f"{det.threshold:.4g}", flush=True)

test_dir = os.path.join(BANK, "test")
files = _sorted_files(test_dir)
test_bank = load_bank(test_dir, label_by="stem")
per_clip = {}
for w, g in zip(test_bank.windows, test_bank.g):
    per_clip.setdefault(int(g), []).append(det.score(w))

y_true, y_score = [], []
for gid in sorted(per_clip):
    stem = os.path.splitext(os.path.basename(files[gid]))[0]
    y_true.append(_label(stem))
    y_score.append(float(np.mean(per_clip[gid])))
y_true = np.array(y_true)
y_score = np.array(y_score)
print(f"[primary] monitor: {len(y_true)} clips "
      f"({int(y_true.sum())} anomaly, {int((1 - y_true).sum())} normal)", flush=True)

from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
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
tpr01 = float(np.interp(0.1, fpr, tpr))

print(f"[FALLBACK PRIMARY] AUC={auc:.4f} 95% CI [{ci_lo:.4f}, {ci_hi:.4f}] "
      f"(bootstrap n=2000 over {n} clips, seed=0) | TPR@FPR=0.1={tpr01:.4f}",
      flush=True)
with open(os.path.join(LOGS, "dcase_valve_readout_fallback_report.md"), "w") as fh:
    fh.write(
        "# DCASE valve id_00 external-user readout (FALLBACK minimal spec)\n\n"
        "NOT the distilled spec (distill LOGO infeasible in 30-min cap).\n"
        f"spec programs: {CHEAP}\nspec premium: [] (none)\n\n"
        f"AUC = {auc:.4f}, 95% CI [{ci_lo:.4f}, {ci_hi:.4f}] "
        f"(bootstrap n=2000 over {n} held-out clips, seed=0)\n"
        f"TPR@FPR=0.1 = {tpr01:.4f}\nthreshold = {det.threshold:.6g}\n"
        f"n_clips={n} (anomaly={int(y_true.sum())}, normal={int((1 - y_true).sum())})\n")
