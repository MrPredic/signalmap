"""wav -> npy bank builder for DCASE2020 Task2 (MIMII) fan/pump/slider id_00,
following the same source and id_00 convention as dcase_valve_adapter.py
(which already loaded data/mimii/valve_id00/{train,test}/ from the same
Zenodo record). Output follows DOMAIN_BANK_CONTRACT.md exactly:

  data/signdomains/mimii_<machine>_id00/
    fit/<stem>.npy              # train/id_00 normal_*.wav (healthy-only split)
    eval/normal_<stem>.npy      # test/id_00 normal_*.wav
    eval/anomaly_<stem>.npy     # test/id_00 anomaly_*.wav
    manifest.json

Source: Zenodo record 3678171, "DCASE 2020 Challenge Task 2 Development
Dataset", https://zenodo.org/api/records/3678171/files/dev_data_<machine>.zip/content
License: CC BY-NC-SA 4.0 (cc-by-nc-sa-4.0).
Verified via direct Zenodo API call (research/factory scratch, Aug 6 2026):
  dev_data_fan.zip    1354774772 bytes  md5:649bdfc06263ae7a838963f43b6641e6
  dev_data_pump.zip   1031279015 bytes  md5:90e7091ef722b7238a7f1009365779cd
  dev_data_slider.zip 1002998966 bytes  md5:da24a757719f0d94d5aa2d646bbfdc86

Decode: python stdlib `wave` module reads the PCM16 wav -> raw bytes ->
np.frombuffer(dtype="<i2") (little-endian int16) -> .astype(np.float64).
This is a pure widen-to-float64 cast of the on-disk PCM16 amplitude values;
NO division/scaling to [-1, 1], no normalization, no filtering, no detrend,
no resampling. mono (nchannels==1) asserted per file; channel 0 = the only
channel per source docs.

fit/ = ALL id_00 train/normal_*.wav (dataset's train split is all-normal,
per contract). eval/ = ALL id_00 test/normal_*.wav and test/anomaly_*.wav.
No sub-sampling: fit and eval use every id_00 recording in the respective
dev_data_<machine> train/test split, so the "first N by sorted filename"
carve-out rule in the contract does not apply here.
train/ and test/ are disjoint recording sets by MIMII/DCASE construction
(different underlying clips) -- verified class-blindly by check_domain_bank.py
via sha256 window digest, not asserted here.

Run:
  python research/factory/build_mimii_id00_banks.py
"""
import glob
import hashlib
import json
import os
import wave
import zipfile

import numpy as np

ROOT = os.environ.get("SIGNALMAP_ROOT",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW = os.path.join(ROOT, "data", "mimii_raw")
OUT_ROOT = os.path.join(ROOT, "data", "signdomains")

MACHINES = ["fan", "pump", "slider"]

ZIP_BYTES = {
    "fan": 1354774772,
    "pump": 1031279015,
    "slider": 1002998966,
}
ZIP_MD5 = {
    "fan": "649bdfc06263ae7a838963f43b6641e6",
    "pump": "90e7091ef722b7238a7f1009365779cd",
    "slider": "da24a757719f0d94d5aa2d646bbfdc86",
}
SOURCE_URL_TMPL = (
    "https://zenodo.org/api/records/3678171/files/dev_data_{m}.zip/content"
)
LICENSE = "CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International)"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wav_float64_mono(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, f"{path}: expected 16-bit PCM"
        assert w.getnchannels() == 1, f"{path}: expected mono"
        fs = w.getframerate()
        raw = w.readframes(w.getnframes())
    int16 = np.frombuffer(raw, dtype="<i2")
    return int16.astype(np.float64), fs


def extract_id00(machine):
    zip_path = os.path.join(RAW, f"dev_data_{machine}.zip")
    extract_dir = os.path.join(RAW, f"extract_{machine}")
    marker = os.path.join(extract_dir, machine, "id_00")
    if os.path.isdir(marker):
        return marker
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        id00_members = [n for n in names if "/id_00/" in n or n.startswith("id_00/")]
        assert id00_members, f"{machine}: no id_00 members found in zip; sample names: {names[:10]}"
        zf.extractall(extract_dir, members=id00_members)
    # locate wherever id_00 landed (zip root layout varies: <machine>/id_00/... )
    found = glob.glob(os.path.join(extract_dir, "**", "id_00"), recursive=True)
    assert found, f"{machine}: id_00 dir not found after extraction under {extract_dir}"
    return found[0]


def build_machine(machine):
    domain = f"mimii_{machine}_id00"
    out_dir = os.path.join(OUT_ROOT, domain)
    fit_dir = os.path.join(out_dir, "fit")
    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(fit_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    id00_dir = extract_id00(machine)
    train_files = sorted(glob.glob(os.path.join(id00_dir, "train", "normal_*.wav")))
    test_normal = sorted(glob.glob(os.path.join(id00_dir, "test", "normal_*.wav")))
    test_anomaly = sorted(glob.glob(os.path.join(id00_dir, "test", "anomaly_*.wav")))
    assert train_files, f"{machine}: no train/normal_*.wav under {id00_dir}"
    assert test_normal, f"{machine}: no test/normal_*.wav under {id00_dir}"
    assert test_anomaly, f"{machine}: no test/anomaly_*.wav under {id00_dir}"

    files_manifest = {}
    fs_seen = set()

    for f in train_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        arr, fs = read_wav_float64_mono(f)
        fs_seen.add(fs)
        assert arr.dtype == np.float64 and arr.ndim == 1
        dst = os.path.join(fit_dir, stem + ".npy")
        np.save(dst, arr)
        files_manifest[os.path.relpath(dst, out_dir)] = sha256_file(dst)

    for f in test_normal + test_anomaly:
        stem = os.path.splitext(os.path.basename(f))[0]
        arr, fs = read_wav_float64_mono(f)
        fs_seen.add(fs)
        dst = os.path.join(eval_dir, stem + ".npy")
        np.save(dst, arr)
        files_manifest[os.path.relpath(dst, out_dir)] = sha256_file(dst)

    assert len(fs_seen) == 1, f"{machine}: mixed sample rates {fs_seen}"
    fs_hz = fs_seen.pop()

    zip_path = os.path.join(RAW, f"dev_data_{machine}.zip")
    bytes_downloaded = os.path.getsize(zip_path)

    manifest = {
        "domain": domain,
        "source_url": SOURCE_URL_TMPL.format(m=machine),
        "source_record": "https://zenodo.org/record/3678171 (DCASE 2020 Challenge Task 2 Development Dataset)",
        "source_zip_md5": ZIP_MD5[machine],
        "source_zip_bytes": ZIP_BYTES[machine],
        "license": LICENSE,
        "modality": "acoustic (single-channel microphone recording of running machine)",
        "fs_hz": fs_hz,
        "channel": "0 = the only (mono) channel, per source docs",
        "decode": (
            "python stdlib wave.open() -> readframes() raw PCM16 bytes -> "
            "np.frombuffer(dtype='<i2') int16 -> .astype(np.float64). "
            "No scaling to [-1,1], no normalization, no filtering, no "
            "detrend, no resampling: float64 widen-cast of the raw PCM16 "
            "amplitude only."
        ),
        "anomaly_mapping": (
            "source filename prefix is the label as shipped by DCASE2020 "
            "Task2: test/id_00/normal_*.wav => normal, "
            "test/id_00/anomaly_*.wav => anomaly. train/id_00/normal_*.wav "
            "(all-normal split) => fit/. No severity levels for this task "
            "(binary normal/anomaly only)."
        ),
        "split_rule": (
            "fit/ = ALL id_00 recordings in the dataset's train/ split "
            "(healthy-only by construction). eval/ = ALL id_00 recordings "
            "in the dataset's test/ split (normal_ and anomaly_). No "
            "sub-sampling applied; train/ and test/ are disjoint recording "
            "sets by MIMII/DCASE construction."
        ),
        "n_fit": len(train_files),
        "n_eval_normal": len(test_normal),
        "n_eval_anomaly": len(test_anomaly),
        "bytes_downloaded": bytes_downloaded,
        "files": files_manifest,
        "notes": (
            f"Same source and id_00 machine-id convention as "
            f"research/factory/dcase_valve_adapter.py (valve_id00, already "
            f"on disk at data/mimii/valve_id00/). Zip downloaded via "
            f"Zenodo record API (nice -n 19 curl, download cap 4GB, "
            f"actual size {bytes_downloaded} bytes) to "
            f"data/mimii_raw/dev_data_{machine}.zip, id_00 subtree "
            f"extracted to data/mimii_raw/extract_{machine}/. "
            f"Reproduce: download dev_data_{machine}.zip from "
            f"{SOURCE_URL_TMPL.format(m=machine)}, verify md5 "
            f"{ZIP_MD5[machine]}, unzip, run this script."
        ),
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)

    print(f"{domain}: fit={len(train_files)} eval_normal={len(test_normal)} "
          f"eval_anomaly={len(test_anomaly)} fs_hz={fs_hz} "
          f"bytes_downloaded={bytes_downloaded}")


if __name__ == "__main__":
    for m in MACHINES:
        build_machine(m)
