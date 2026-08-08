"""Frontier-domain loaders (Jul 2): jump outside industrial sensing into
high-potential fields with clean group structure.

HAR (UCI 240): smartphone accel/gyro, 30 subjects x 6 activities. Recording =
subject x activity block (label-pure); signal = body-acc magnitude, re-windowed.
Group = subject -> no subject leaks across folds (the honest question for
wearables: does it generalize to a NEW person?).

ECG AF-vs-NSR (PhysioNet afdb/nsrdb via wfdb streaming): patient = recording.
Atrial-fibrillation detection — a real clinical screening task. Group = patient.
"""
import glob, os
import numpy as np
from scipy.signal import detrend

W = 1024
HAR = "<local-path>/signalmap/data/har/UCI HAR Dataset"

def _push(x, cls, gid, raw, w=W):
    x = np.asarray(x, float)
    for k in range(len(x) // w):
        s = detrend(np.ascontiguousarray(x[k * w:(k + 1) * w]))
        raw.append(((s - s.mean()) / (s.std() + 1e-12), cls, gid))

ACT = {1: "walk", 2: "upstairs", 3: "downstairs", 4: "sit", 5: "stand", 6: "lay"}

def load_har(split="train", w=256):
    base = f"{HAR}/{split}"
    ax = np.loadtxt(f"{base}/Inertial Signals/body_acc_x_{split}.txt")
    ay = np.loadtxt(f"{base}/Inertial Signals/body_acc_y_{split}.txt")
    az = np.loadtxt(f"{base}/Inertial Signals/body_acc_z_{split}.txt")
    subj = np.loadtxt(f"{base}/subject_{split}.txt").astype(int)
    y = np.loadtxt(f"{base}/y_{split}.txt").astype(int)
    # window overlap in HAR is 50%; take first half of each 128-frame to de-dup
    mag = np.sqrt(ax**2 + ay**2 + az**2)[:, :64]
    raw, gid = [], 0
    for s in np.unique(subj):
        for a in np.unique(y):
            m = (subj == s) & (y == a)
            if m.sum() < 4: continue
            sig = mag[m].ravel()  # concat this subject's frames for this activity
            if len(sig) >= 2 * w:
                _push(sig, ACT[a], gid, raw, w=w)
                gid += 1
    return raw

def load_ecg_af(n_patients=6, seg_per_patient=6, w=W):
    """AF (afdb) vs normal sinus (nsrdb). Streams short segments per record.
    Recording = patient. Requires internet (wfdb.rdrecord over HTTPS)."""
    import wfdb
    af = ["04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091"]
    ns = ["16265", "16272", "16273", "16420", "16483", "16539", "16773", "17052"]
    raw, gid = [], 0
    for lab, recs, db in [("AF", af, "afdb"), ("NSR", ns, "nsrdb")]:
        got = 0
        for r in recs:
            if got >= n_patients: break
            try:
                sig = wfdb.rdrecord(r, pn_dir=db, channels=[0],
                                    sampto=seg_per_patient * w).p_signal[:, 0]
            except Exception as e:
                print(f"  skip {db}/{r}: {e}", flush=True); continue
            if np.isnan(sig).any() or len(sig) < 2 * w: continue
            _push(sig, lab, gid, raw, w=w)
            gid += 1; got += 1
    return raw
