#!/usr/bin/env python3
"""Sign-identifiability readout — the only script in this study that computes a
separation metric.

    nice -n 19 .venv/bin/python research/factory/sign_identifiability_readout.py

Prereg: research/factory/PREREG_SIGN_IDENTIFIABILITY.md (frozen, git 8c303c6).
Banks: data/signdomains/<domain>/, built to DOMAIN_BANK_CONTRACT.md by people
who were forbidden to compute any separation metric — so no bank was ever
selected against the number this script produces.

What is measured, per domain, with one frozen recipe shared by all domains:
  fit  DistilledDetector on healthy-only windows (never sees an anomaly)
  eval mean window score per held-out recording
  M1   AUC + percentile bootstrap CI over recordings
  M3   alarm-rate difference at the detector's own self-calibrated threshold
  N1   label-shuffle null (also calibrates the upward bias of max(AUC, 1-AUC))
  N2   healthy-vs-healthy split (catches ordering/drift artefacts N1 cannot)
  N3   sha256 leakage check between fit and eval windows
M2 (theory) lives in SIGN_IDENTIFIABILITY_THEORY.md; M4 (sign transfer across
domains) is computed here across the per-domain results.

Everything is seeded (0) and cached per domain, so a re-run reproduces the
report byte-for-byte and a crashed domain does not cost the finished ones.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from signalmap.distill import DistilledDetector, FeatureSpec, window  # noqa: E402
from signalmap.receipt import emit_signed_receipt  # noqa: E402

BANKS = ROOT / "data" / "signdomains"
LOGS = ROOT / "research" / "factory" / "logs"
CACHE = LOGS / "sign_identifiability_cache"
RECEIPTS = LOGS / "sign_receipts"
REPORT = LOGS / "sign_identifiability_report.md"

# --- frozen per prereg §Fixe Parameter; do not touch ------------------------
SPEC_PROGRAMS = [
    "acf1(id(id(x)))", "crest(id(id(x)))", "lcross(id(id(x)))",
    "meanabs(id(id(x)))", "peakcv(id(id(x)))", "runcv(id(id(x)))",
    "runmean(id(id(x)))", "std(id(id(x)))", "zcr(id(id(x)))",
]
K, W = 20, 1024
ENVELOPE = 3.0
N_BOOT = 2000
N_PERM = 2000
SEED = 0
PREREG = "PREREG_SIGN_IDENTIFIABILITY.md"


# ------------------------------------------------------------------ metrics
def auc(y_true, y_score):
    """Mann-Whitney AUC with tie-averaged ranks.

    Written out rather than imported so a reviewer can check the headline
    number against the definition without trusting a library version. The
    test suite pins it against sklearn.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(y_score.size, dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # average rank, 1-based
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def boot_ci(y_true, y_score, n=N_BOOT, seed=SEED):
    """Percentile bootstrap over recordings — the unit the CI must respect."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    m = y_true.size
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        if np.unique(y_true[idx]).size < 2:
            continue
        vals.append(auc(y_true[idx], y_score[idx]))
    if not vals:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def perm_null(y_true, y_score, n=N_PERM, seed=SEED):
    """N1: shuffle labels. Returns the null AUCs and the null of max(AUC,1-AUC).

    The second one matters: max() is biased upward under the null, so H2 must
    be tested against this distribution rather than against 0.5.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).copy()
    aucs = np.empty(n, dtype=np.float64)
    for i in range(n):
        rng.shuffle(y_true)
        aucs[i] = auc(y_true, y_score)
    return aucs, np.maximum(aucs, 1.0 - aucs)


def alarm_gap(y_true, y_score, threshold, seed=SEED):
    """M3: what the product actually ships — an alarm, not a ranking."""
    y_true = np.asarray(y_true)
    alarm = (np.asarray(y_score) > threshold).astype(float)
    rate_a = float(alarm[y_true == 1].mean())
    rate_n = float(alarm[y_true == 0].mean())
    rng = np.random.default_rng(seed)
    m = y_true.size
    gaps = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, m, m)
        yt, al = y_true[idx], alarm[idx]
        if np.unique(yt).size < 2:
            continue
        gaps.append(al[yt == 1].mean() - al[yt == 0].mean())
    lo, hi = (np.percentile(gaps, [2.5, 97.5]) if gaps else (float("nan"),) * 2)
    return {"alarm_rate_anomaly": rate_a, "alarm_rate_normal": rate_n,
            "gap": rate_a - rate_n, "gap_ci_lo": float(lo), "gap_ci_hi": float(hi)}


# -------------------------------------------------------------------- banks
def load_recording(path):
    """First K*W samples of a raw recording, windowed by the product's own
    windower (detrend + z-norm per window)."""
    arr = np.load(path, allow_pickle=False)
    wins = window(np.ascontiguousarray(arr[: K * W], dtype=np.float64))
    return wins[:K]


def digest_windows(wins):
    h = hashlib.sha256()
    for w in wins:
        h.update(np.ascontiguousarray(w, dtype=np.float64).tobytes())
    return h.hexdigest()


def load_domain(root):
    """Return fit windows, eval scores' raw material, and the leakage verdict."""
    fit_paths = sorted((root / "fit").glob("*.npy"))
    ev_paths = sorted((root / "eval").glob("normal_*.npy")) + \
        sorted((root / "eval").glob("anomaly_*.npy"))

    fit_wins, fit_digests = [], set()
    for p in fit_paths:
        wins = load_recording(p)
        fit_digests.add(digest_windows(wins))
        fit_wins.extend(wins)

    ev = []
    leaked = []
    for p in ev_paths:
        wins = load_recording(p)
        if digest_windows(wins) in fit_digests:
            leaked.append(p.name)
        ev.append((p.name, 1 if p.name.startswith("anomaly_") else 0, wins))
    return fit_wins, ev, leaked


# ------------------------------------------------------------------ readout
def run_domain(name, root):
    """One domain, one frozen recipe. Returns the payload that gets signed."""
    t0 = time.time()
    manifest = json.loads((root / "manifest.json").read_text())
    fit_wins, ev, leaked = load_domain(root)

    spec = FeatureSpec(programs=list(SPEC_PROGRAMS), premium=[], window=W)
    det = DistilledDetector.fit(spec, fit_wins, envelope=ENVELOPE)

    names = [n for n, _, _ in ev]
    y_true = np.array([lab for _, lab, _ in ev])
    y_score = np.array([float(np.mean([det.score(w) for w in wins])) for _, _, wins in ev])

    observed = auc(y_true, y_score)
    ci_lo, ci_hi, n_boot = boot_ci(y_true, y_score)
    null_auc, null_star = perm_null(y_true, y_score)
    auc_star = max(observed, 1.0 - observed)
    q95_star = float(np.percentile(null_star, 95))

    # N2: relabel the second half of the normal recordings as "anomaly".
    norm_idx = [i for i, lab in enumerate(y_true) if lab == 0]
    order = sorted(norm_idx, key=lambda i: names[i])
    half = len(order) // 2
    hh_true = np.array([0] * half + [1] * (len(order) - half))
    hh_score = y_score[np.array(order)]
    hh_auc = auc(hh_true, hh_score)
    hh_lo, hh_hi, _ = boot_ci(hh_true, hh_score)

    direction = ("inverted" if ci_hi < 0.5 else
                 "aligned" if ci_lo > 0.5 else "undetermined")
    n2_ok = bool(hh_lo <= 0.5 <= hh_hi)
    n1_ok = bool(np.percentile(null_auc, 2.5) <= 0.5 <= np.percentile(null_auc, 97.5))

    if leaked:
        verdict = "EXCLUDED"
    elif not (n1_ok and n2_ok):
        verdict = "REFUSED"
    elif direction == "undetermined":
        verdict = "REFUSED"
    else:
        verdict = "PASS"

    payload = {
        "domain": name, "prereg": PREREG, "verdict": verdict,
        "direction": direction,
        "n_fit_recordings": len(fit_wins) // K, "n_fit_windows": len(fit_wins),
        "n_eval": int(y_true.size), "n_anomaly": int(y_true.sum()),
        "n_normal": int((1 - y_true).sum()),
        "auc": observed, "ci_lo": ci_lo, "ci_hi": ci_hi, "n_boot_used": n_boot,
        "auc_star": auc_star, "null_star_q95": q95_star,
        "h2_separates": bool(auc_star > q95_star),
        "perm_p_two_sided": float(
            (np.abs(null_auc - 0.5) >= abs(observed - 0.5)).mean()),
        "n1_null_covers_half": n1_ok,
        "n1_null_ci": [float(np.percentile(null_auc, 2.5)),
                       float(np.percentile(null_auc, 97.5))],
        "n2_healthy_vs_healthy_auc": hh_auc,
        "n2_ci": [hh_lo, hh_hi], "n2_ok": n2_ok,
        "n3_leaked_recordings": leaked,
        "threshold": float(det.threshold),
        "alarm": alarm_gap(y_true, y_score, det.threshold),
        "spec_programs": SPEC_PROGRAMS, "spec_premium": [],
        "K": K, "W": W, "envelope": ENVELOPE, "seed": SEED,
        "manifest": {k: manifest.get(k) for k in
                     ("source_url", "license", "modality", "fs_hz",
                      "channel", "anomaly_mapping")},
        "runtime_s": round(time.time() - t0, 1),
    }
    return payload


def transfer_matrix(results):
    """M4: does the sign measured in one domain predict the sign in another?

    Only domains whose verdict is PASS count. A CI clear of 0.5 is not enough:
    if the healthy-vs-healthy control failed, the eval set is separable
    without any anomaly at all, so its "direction" may be an operating-point
    difference rather than a fault, and it must not prop up H1.
    """
    decided = [r for r in results
               if r["verdict"] == "PASS" and r["direction"] in ("inverted", "aligned")]
    hits = total = 0
    for a in decided:
        for b in decided:
            if a["domain"] == b["domain"]:
                continue
            total += 1
            hits += int(a["direction"] == b["direction"])
    return {"n_decided_domains": len(decided), "n_ordered_pairs": total,
            "sign_transfer_hit_rate": (hits / total) if total else float("nan"),
            "inverted": [r["domain"] for r in decided if r["direction"] == "inverted"],
            "aligned": [r["domain"] for r in decided if r["direction"] == "aligned"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", help="restrict to these domains")
    ap.add_argument("--refresh", action="store_true", help="ignore cached domains")
    ap.add_argument("--banks", type=Path, default=BANKS,
                    help="bank root (smoke tests point this away from the study)")
    ap.add_argument("--out", type=Path, default=LOGS,
                    help="output root (smoke tests point this away from the study)")
    args = ap.parse_args(argv)

    banks = args.banks
    cache, receipts, report = (args.out / "sign_identifiability_cache",
                               args.out / "sign_receipts",
                               args.out / "sign_identifiability_report.md")
    cache.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    if not banks.is_dir():
        raise SystemExit(f"{banks} missing — no domain banks built yet")

    results = []
    for root in sorted(p for p in banks.iterdir() if (p / "manifest.json").exists()):
        name = root.name
        if args.only and name not in args.only:
            continue
        cached = cache / f"{name}.json"
        if cached.exists() and not args.refresh:
            print(f"[{name}] cached", flush=True)
            results.append(json.loads(cached.read_text()))
            continue
        print(f"[{name}] running ...", flush=True)
        try:
            payload = run_domain(name, root)
        except Exception as exc:  # noqa: BLE001 - one bad domain must not sink the run
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        cached.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        results.append(payload)
        print(f"[{name}] {payload['verdict']} {payload['direction']} "
              f"AUC={payload['auc']:.4f} CI [{payload['ci_lo']:.4f}, "
              f"{payload['ci_hi']:.4f}] ({payload['runtime_s']}s)", flush=True)

    if not results:
        raise SystemExit("no domain produced a result")

    for payload in results:
        emit_signed_receipt(
            claim=(f"on domain '{payload['domain']}', a healthy-only calibrated "
                   f"signalmap detector ranks anomalies {payload['direction']} "
                   f"relative to normal (AUC {payload['auc']:.4f}, "
                   f"95% CI [{payload['ci_lo']:.4f}, {payload['ci_hi']:.4f}])"),
            verdict=payload["verdict"], evidence=payload,
            input_hashes={"manifest_sha256": hashlib.sha256(
                (banks / payload["domain"] / "manifest.json").read_bytes()).hexdigest()},
            out_path=receipts / f"sign_{payload['domain']}.receipt.json")

    xfer = transfer_matrix(results)
    h1 = bool(xfer["inverted"] and xfer["aligned"])
    margins = [r["auc_star"] - r["null_star_q95"] for r in results]
    n_clear = sum(m >= 0.02 for m in margins)
    n_marginal = sum(0 <= m < 0.02 for m in margins)

    lines = [
        "# Sign identifiability of label-free anomaly scores",
        "",
        f"Prereg: `{PREREG}` (frozen before any bank was built).",
        f"One frozen spec for every domain, K={K} windows of W={W}, "
        f"envelope={ENVELOPE}, seed={SEED}, {N_BOOT} bootstrap / {N_PERM} permutations.",
        "",
        "| domain | n | AUC | 95% CI | direction | verdict | perm p | N1 | N2 | AUC* vs null q95 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r["auc"]):
        margin = r["auc_star"] - r["null_star_q95"]
        lines.append(
            f"| {r['domain']} | {r['n_eval']} | {r['auc']:.4f} | "
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['direction']} | {r['verdict']} | "
            f"{r['perm_p_two_sided']:.4f} | "
            f"{'ok' if r['n1_null_covers_half'] else 'FAIL'} | "
            f"{'ok' if r['n2_ok'] else 'FAIL'} | "
            f"{r['auc_star']:.3f} vs {r['null_star_q95']:.3f} "
            f"({margin:+.3f}{', marginal' if 0 < margin < 0.02 else ''}) |")
    lines += [
        "",
        f"**H1 (sign not identifiable): {'CONFIRMED' if h1 else 'NOT CONFIRMED'}** — "
        f"inverted: {xfer['inverted'] or 'none'}; "
        f"aligned: {xfer['aligned'] or 'none'}. "
        f"Counted only where the verdict is PASS: a CI clear of 0.5 does not "
        f"count if a null control failed. Excluded on that ground: "
        f"{[r['domain'] for r in results if r['verdict'] != 'PASS' and r['direction'] != 'undetermined'] or 'none'}.",
        f"**H2 (magnitude carries information):** {n_clear}/{len(results)} domains clear "
        f"their own shuffle-null 95th percentile by more than 0.02"
        + (f"; {n_marginal} sit on it (margin < 0.02) and are counted as "
           f"undecided, not as support" if n_marginal else "") + ".",
        "**H3 (shipped alarm): NOT VALIDLY MEASURED** — see prereg AMENDMENT 3. "
        "The threshold is the 99th percentile of *window* scores, but this run "
        "compared it against the *recording* mean, so the alarm is structurally "
        "silent (0.000 on both classes in every domain) regardless of any anomaly. "
        "No alarm claim is made from this run, for or against H3.",
        f"**M4 sign transfer:** hit rate {xfer['sign_transfer_hit_rate']:.3f} over "
        f"{xfer['n_ordered_pairs']} ordered pairs of CI-fest domains "
        f"(0.5 = the sign of one domain says nothing about another).",
        "",
        "Reproduce: `nice -n 19 .venv/bin/python "
        "research/factory/sign_identifiability_readout.py --refresh`",
        f"Signed receipts: `{receipts.relative_to(ROOT)}/` — verify offline with "
        "`python tools/verify_receipt.py <receipt>`, which imports nothing from signalmap.",
    ]
    report.write_text("\n".join(lines) + "\n")

    emit_signed_receipt(
        claim=("the sign of a healthy-only calibrated anomaly score is not "
               "identifiable from the training data: it is CI-fest inverted in at "
               "least one domain and CI-fest aligned in another under one frozen recipe"),
        verdict="PASS" if h1 else "REFUSED",
        evidence={"prereg": PREREG, "h1_confirmed": h1, "transfer": xfer,
                  "n_domains": len(results),
                  "h2_domains_clearly_separating": int(n_clear),
                  "h2_domains_marginal": int(n_marginal),
                  "h3_status": "not validly measured (prereg AMENDMENT 3: "
                               "window-calibrated threshold compared against a "
                               "recording mean; alarm silent on both classes)",
                  "per_domain": {r["domain"]: {"auc": r["auc"], "ci": [r["ci_lo"], r["ci_hi"]],
                                               "direction": r["direction"],
                                               "verdict": r["verdict"]} for r in results}},
        input_hashes={"report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()},
        out_path=receipts / "sign_identifiability_overall.receipt.json")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
