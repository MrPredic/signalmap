"""Cross-domain family transfer: does each bank's stable winner family (found
by nested forge) also work on the OTHER banks? LOGO RF per bank on each fixed
3-program family (no selection => no selection bias, plain LOGO is honest).
Answers: is per-domain distillation necessary, or is there one universal set?
Run: .venv-research/bin/python research/factory/family_transfer.py
"""
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from feature_forge import programs, run_prog, load_ecn, load_cwru
from mfpt_run import load_mfpt

FAMILIES = {
    "ECN-family": ["specratio(sign(diff(x)))", "specratio(tanh(id(x)))", "hent(tanh(diff(x)))"],
    "CWRU-family": ["meanabs(diff(abs(x)))", "crest(diff2(abs(x)))", "meanabs(sq(sq(x)))"],
    "MFPT-family": ["hent(tanh(id(x)))", "speccent(env(id(x)))", "specratio(env(env(x)))"],
}

if __name__ == "__main__":
    progs = {p[0]: p for p in programs()}
    banks = [("ECN", load_ecn(), 0.167), ("MFPT", load_mfpt(), 0.333), ("CWRU", load_cwru(), 0.167)]
    for bname, raw, chance in banks:
        y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
        t0 = time.time()
        for fam, names in FAMILIES.items():
            sel = [progs[n] for n in names if n in progs]
            F = np.array([[run_prog(p, s) for p in sel] for s, _, _ in raw])
            acc = cross_val_score(
                make_pipeline(StandardScaler(), RandomForestClassifier(150, random_state=0, n_jobs=-1)),
                F, y, groups=g, cv=LeaveOneGroupOut()).mean()
            print(f"{bname} (chance {chance}): {fam:12s} -> {acc:.3f}  ({len(sel)}/{len(names)} progs found)", flush=True)
        print(f"  [{time.time()-t0:.0f}s]", flush=True)
