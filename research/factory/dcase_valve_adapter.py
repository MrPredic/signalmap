"""wav -> npy adapter for the DCASE2020 Task2 valve/id_00 external-user
simulation. Prereg: PREREG_DCASE_VALVE_EXTERNAL.md (frozen before readout) —
bank composition (which files go where, K per bank) is FIXED there; do not
change file selection or K after reading results.

Source (already on disk, Phase 1): data/mimii/valve_id00/{train,test}/*.wav
  mono, 16000 Hz, 160000 frames (10s), 16-bit PCM. train = 891 normal.
  test = 100 normal + 119 anomaly.

Output (three bank dirs, 1-D int16 .npy per recording, stem preserved so
`label_by="stem"` (fit/monitor) and `label_by="prefix"` (distill) both work
off the same filenames):
  data/mimii/valve_id00_bank/distill/  30 recordings (15 normal from train +
                                        15 anomaly from test, first-by-name),
                                        K=120 windows/clip worth of samples.
  data/mimii/valve_id00_bank/train/    all 891 train-normal clips, K=20.
  data/mimii/valve_id00_bank/test/     100 test-normal (all) + 104 test-anomaly
                                        (119 - 15 used by distill), K=20.

Run: cd <local-path>/signalmap && source .venv-research/bin/activate && \
     python research/factory/dcase_valve_adapter.py
"""
import glob
import os
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
SRC = os.path.join(ROOT, "data", "mimii", "valve_id00")
DST = os.path.join(ROOT, "data", "mimii", "valve_id00_bank")

W = 1024
K_DISTILL = 120   # distill/premium bank (30 recordings, precedent scale)
K_PRIMARY = 20    # fit/monitor bank (all clips, RAM-bounded)

N_DISTILL_PER_CLASS = 15  # -> 30 recordings total, chance=0.5


def _read_wav_int16_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, f"{path}: expected 16-bit PCM"
        assert w.getnchannels() == 1, f"{path}: expected mono"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16)


def _write_bank_file(dst_dir: str, stem: str, samples: np.ndarray, k: int) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    take = samples[: k * W]
    assert len(take) == k * W, (
        f"{stem}: only {len(samples)} samples, need {k * W} for K={k}")
    np.save(os.path.join(dst_dir, stem + ".npy"), take)


def build() -> None:
    train_files = sorted(glob.glob(os.path.join(SRC, "train", "normal_*.wav")))
    test_normal = sorted(glob.glob(os.path.join(SRC, "test", "normal_*.wav")))
    test_anomaly = sorted(glob.glob(os.path.join(SRC, "test", "anomaly_*.wav")))
    assert len(train_files) == 891, f"expected 891 train-normal, got {len(train_files)}"
    assert len(test_normal) == 100, f"expected 100 test-normal, got {len(test_normal)}"
    assert len(test_anomaly) == 119, f"expected 119 test-anomaly, got {len(test_anomaly)}"

    distill_normal = train_files[:N_DISTILL_PER_CLASS]
    distill_anomaly = test_anomaly[:N_DISTILL_PER_CLASS]
    eval_anomaly = test_anomaly[N_DISTILL_PER_CLASS:]  # 104, disjoint from distill

    # 1) distill/premium bank: 30 recordings, K=120
    distill_dir = os.path.join(DST, "distill")
    for f in distill_normal + distill_anomaly:
        stem = os.path.splitext(os.path.basename(f))[0]
        _write_bank_file(distill_dir, stem, _read_wav_int16_mono(f), K_DISTILL)
    print(f"distill/: {len(distill_normal)} normal + {len(distill_anomaly)} anomaly "
          f"-> {distill_dir}")

    # 2) fit bank: all 891 train-normal, K=20
    train_dir = os.path.join(DST, "train")
    for f in train_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        _write_bank_file(train_dir, stem, _read_wav_int16_mono(f), K_PRIMARY)
    print(f"train/: {len(train_files)} recordings -> {train_dir}")

    # 3) monitor/eval bank: 100 test-normal (all) + 104 test-anomaly, K=20
    test_dir = os.path.join(DST, "test")
    for f in test_normal + eval_anomaly:
        stem = os.path.splitext(os.path.basename(f))[0]
        _write_bank_file(test_dir, stem, _read_wav_int16_mono(f), K_PRIMARY)
    print(f"test/: {len(test_normal)} normal + {len(eval_anomaly)} anomaly "
          f"-> {test_dir}")


if __name__ == "__main__":
    build()
