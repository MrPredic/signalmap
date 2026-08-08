"""Frozen, small cardiac-family readouts for the preregistered ECG screen.

This module is research-only until its receipts pass the preregistered gates.
It deliberately exposes no parameter grid or label-aware selection.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pywt
import wfdb
from scipy import signal
from scipy.stats import entropy

FS = 250.0
RECORDS = {
    "AF": ["04015", "04043", "04048", "04126", "04746", "04908"],
    "NSR": ["16265", "16272", "16273", "16420", "16483", "16539"],
}


def resample_fixed_fs(x, source_fs, target_fs=FS):
    n = int(round(len(x) * float(target_fs) / float(source_fs)))
    return signal.resample(np.asarray(x, float), n)


def _clean(x):
    x = signal.detrend(np.asarray(x, float))
    return (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))) + 1e-9)


def extract_beats(x, fs):
    """Fixed Pan-Tompkins-like detector; returns beat snippets and RR seconds."""
    x = _clean(x)
    lo, hi = 5.0, min(18.0, fs * 0.45)
    b, a = signal.butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    z = signal.filtfilt(b, a, x)
    integ = signal.savgol_filter(np.gradient(z) ** 2, int(fs * 0.12) // 2 * 2 + 1, 2)
    distance = int(0.30 * fs)
    peaks, props = signal.find_peaks(integ, distance=distance,
                                     prominence=max(np.std(integ) * 0.5, 1e-6))
    if len(peaks) < 2:
        return np.empty((0, int(0.6 * fs))), np.empty(0)
    amp = np.abs(z[peaks])
    keep = amp >= np.percentile(amp, 25)
    peaks = peaks[keep]
    rr = np.diff(peaks) / fs
    half_l, half_r = int(0.20 * fs), int(0.40 * fs)
    beats = [x[p - half_l:p + half_r] for p in peaks
             if p >= half_l and p + half_r < len(x)]
    return np.asarray(beats), rr


def _safe_stats(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return np.zeros(4)
    d = np.diff(v)
    return np.array([np.median(v), np.std(v), np.median(np.abs(d)) if d.size else 0.0,
                     np.std(d) if d.size else 0.0])


def cardiac_features(x, fs, family):
    beats, rr = extract_beats(x, fs)
    if family == "hrv":
        if len(rr) < 7:
            return np.full(4, np.nan)
        rr = rr[(rr >= 0.30) & (rr <= 2.0)]
        if len(rr) < 7:
            return np.full(4, np.nan)
        d = np.diff(rr)
        rmssd = np.sqrt(np.mean(d * d))
        pnn50 = np.mean(np.abs(d) > 0.05)
        acf1 = np.corrcoef(rr[:-1], rr[1:])[0, 1] if len(rr) > 8 else 0.0
        return np.array([np.std(rr), rmssd, pnn50, np.nan_to_num(acf1)])
    if family == "morphology":
        if len(beats) < 5:
            return np.full(6, np.nan)
        template = np.median(beats, axis=0)
        dev = np.mean(np.abs(beats - template), axis=1)
        width = np.sum(np.abs(template) >= 0.5 * np.max(np.abs(template))) / fs
        asym = np.sum(np.abs(template[:len(template)//2])) / (np.sum(np.abs(template[len(template)//2:])) + 1e-9)
        return np.array([np.median(dev), np.std(dev), width, asym,
                         np.median(np.max(beats, axis=1)), np.median(np.min(beats, axis=1))])
    if family == "wavelet":
        coeff = pywt.wavedec(_clean(x), "db4", level=5)
        e = np.array([np.mean(c * c) for c in coeff]) + 1e-12
        p = e / e.sum()
        return np.r_[e[1:] / e[:-1], entropy(p) / np.log(len(p))]
    raise ValueError(family)


def generic_features(x, fs):
    x = _clean(x)
    pe = np.argsort(np.argsort(np.lib.stride_tricks.sliding_window_view(x, 3), axis=1), axis=1)
    pe = entropy(np.bincount(pe[:, 0] * 9 + pe[:, 1] * 3 + pe[:, 2], minlength=27) + 1e-9)
    f, p = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    ok = (f > 0) & (p > 0)
    slope = np.polyfit(np.log(f[ok]), np.log(p[ok]), 1)[0]
    return np.array([pe, slope])


def phase_randomize(x, rng):
    """IAAFT-lite surrogate: preserves the power spectrum, randomizes phase."""
    x = np.asarray(x, float)
    f = np.fft.rfft(x - np.mean(x))
    phase = rng.uniform(0.0, 2.0 * np.pi, len(f))
    phase[0] = 0.0
    if len(x) % 2 == 0:
        phase[-1] = 0.0
    return np.fft.irfft(np.abs(f) * np.exp(1j * phase), n=len(x))


def load_records(seconds=30.0, windows=4, window_seconds=8.0):
    rows = []
    for label, records in RECORDS.items():
        for rec in records:
            db = "afdb" if label == "AF" else "nsrdb"
            h = wfdb.rdheader(rec, pn_dir=db)
            sampto = int(seconds * h.fs)
            sig = wfdb.rdrecord(rec, pn_dir=db, channels=[0], sampto=sampto).p_signal[:, 0]
            sig = resample_fixed_fs(sig, h.fs, FS)
            w = int(window_seconds * FS)
            for k in range(min(windows, len(sig) // w)):
                x = sig[k * w:(k + 1) * w]
                rows.append(((_clean(x)), label, rec))
    return rows


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", default="research/factory/logs/heart_method_families.json")
    args = ap.parse_args()
    rows = load_records()
    families = ["generic", "morphology", "hrv", "wavelet"]
    root = Path(__file__).resolve().parent
    out = {"prereg": _sha(root / "PREREG_HEART_METHOD_FAMILIES.md"), "n": len(rows), "families": {}}
    def evaluate(family, input_rows):
        X, y, groups = [], [], []
        for x, label, group in input_rows:
            f = generic_features(x, FS) if family == "generic" else cardiac_features(x, FS, family)
            if np.isfinite(f).all():
                X.append(f); y.append(label); groups.append(group)
        X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
        group_scores = {}
        for group in np.unique(groups):
            tr, te = groups != group, groups == group
            if len(np.unique(y[tr])) < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=0))
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            group_scores[group] = float(accuracy_score(y[te], pred))
        return group_scores, int(len(y))

    scores = {}
    for family in families:
        scores[family], n_valid = evaluate(family, rows)
        out["families"][family] = {
            "accuracy_recording_macro": float(np.mean(list(scores[family].values()))) if scores[family] else None,
            "n_valid_windows": n_valid,
            "n_groups": len(scores[family]),
        }
    base = scores["generic"]
    for family in families[1:]:
        shared = sorted(set(base) & set(scores[family]))
        delta = [scores[family][g] - base[g] for g in shared]
        out["families"][family]["paired_delta_vs_generic"] = float(np.mean(delta)) if delta else None
        out["families"][family]["shared_groups"] = len(shared)

    # Fixed, label-preserving mechanism stress: spectrum-preserving phase null.
    rng = np.random.default_rng(0)
    for family in families[1:]:
        null_rows = [(phase_randomize(x, rng), label, group) for x, label, group in rows]
        null_scores, _ = evaluate(family, null_rows)
        out["families"][family]["phase_null_recording_macro"] = (
            float(np.mean(list(null_scores.values()))) if null_scores else None)

    # Group-level label permutation, fixed 200 draws for this screening run.
    group_labels = {g: lab for _, lab, g in rows}
    groups = np.array(sorted(group_labels))
    observed = {f: out["families"][f]["accuracy_recording_macro"] for f in families}
    perm_ge = {f: 0 for f in families}
    for _ in range(200):
        shuffled = rng.permutation([group_labels[g] for g in groups])
        mapping = dict(zip(groups, shuffled))
        perm_rows = [(x, mapping[group], group) for x, _, group in rows]
        for family in families:
            ps, _ = evaluate(family, perm_rows)
            score = float(np.mean(list(ps.values()))) if ps else 0.0
            if score >= observed[family]:
                perm_ge[family] += 1
    for family in families:
        out["families"][family]["group_label_perm_p_200"] = (perm_ge[family] + 1) / 201
    receipt = Path(args.receipt)
    if not receipt.is_absolute():
        receipt = root.parent.parent / receipt
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
