"""Cross-station hardening (RESTART Prio A1+A3): does the SEIS-depth and
GEOMAG-storm signal survive at a second, independent station?

Design (strictest variant, kills station- AND event/day-leakage at once):
  cross-LOGO: for each held-out recording r, train lean on SOURCE station
  WITHOUT r, test on the TARGET station's windows of r only. The model never
  sees the target station, and never sees the tested event/day at any station.
  Both directions. Plus: target-internal LOGO + group-perm-p = independent
  replication of the original claim at the second station.

Banks are matched by construction (verified at fetch time):
  SEIS  ANMO/KONO: identical 16 events (deterministic USGS query), gid-aligned.
  GEOMAG BOU/FRD: same day list; FRD is missing 2023-12-09 -> align via `days`
  arrays (BOU cache predates the days field -> rebuilt from the day lists).

Run: .venv-research/bin/python research/factory/harden_transfer.py [seis|geomag]
"""
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_forge import lean_baseline, group_perm_p, programs, run_prog

BASE = "<local-path>/signalmap/data"


def _ci(a, seed=0):
    a = np.array(a, float); rng = np.random.default_rng(seed)
    m = rng.choice(a, (10000, len(a)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _clf():
    return make_pipeline(StandardScaler(),
                         RandomForestClassifier(150, random_state=0, n_jobs=-1))


def cross_logo(Fs, ys, gs, Ft, yt, gt, common):
    """Train on source minus rec r, test on target's windows of rec r."""
    accs = {}
    for r in common:
        tr = gs != r
        clf = _clf(); clf.fit(Fs[tr], ys[tr])
        te = gt == r
        accs[r] = float((clf.predict(Ft[te]) == yt[te]).mean())
    return accs


def report(tag, accs, chance):
    a = list(accs.values())
    lo, hi = _ci(a)
    n_ok = sum(v > chance for v in a)
    print(f"{tag}: {np.mean(a):.3f}  CI [{lo:.3f},{hi:.3f}]  "
          f">chance {n_ok}/{len(a)}  (chance {chance:.3f})", flush=True)
    return np.mean(a), lo


def within_logo(F, y, g, tag, n_perm=60):
    accs = {}
    for r in np.unique(g):
        tr = g != r
        clf = _clf(); clf.fit(F[tr], y[tr])
        accs[r] = float((clf.predict(F[~tr]) == y[~tr]).mean())
    m, lo = report(tag, accs, 1 / len(set(y)))
    p = group_perm_p(F, y, g, m, n_perm=n_perm)
    print(f"{tag} group-perm-p = {p:.3f} ({n_perm} perms)", flush=True)
    return accs


def run_pair(name, raw_src, raw_tgt, src, tgt, extra_progs=()):
    ys = np.array([r[1] for r in raw_src]); gs = np.array([r[2] for r in raw_src])
    yt = np.array([r[1] for r in raw_tgt]); gt = np.array([r[2] for r in raw_tgt])
    chance = 1 / len(set(ys))
    common = sorted(set(gs) & set(gt))
    # sanity: matched gids must carry the same label on both stations
    for r in common:
        assert ys[gs == r][0] == yt[gt == r][0], f"label mismatch at gid {r}"
    print(f"\n===== {name}: {src}({len(set(gs))} recs) <-> {tgt}({len(set(gt))} recs), "
          f"{len(common)} matched, chance={chance:.3f} =====", flush=True)
    Fs, Ft = lean_baseline(raw_src), lean_baseline(raw_tgt)

    within_logo(Ft, yt, gt, f"{tgt} within-LOGO (replication)")
    report(f"{src}->{tgt} cross-LOGO", cross_logo(Fs, ys, gs, Ft, yt, gt, common), chance)
    report(f"{tgt}->{src} cross-LOGO", cross_logo(Ft, yt, gt, Fs, ys, gs, common), chance)

    for pname in extra_progs:  # champion forge family, fixed (no selection bias)
        prog = {p[0]: p for p in programs()}.get(pname)
        if prog is None:
            print(f"  [prog {pname} not in grammar]", flush=True); continue
        Ps = np.array([[run_prog(prog, s)] for s, _, _ in raw_src])
        Pt = np.array([[run_prog(prog, s)] for s, _, _ in raw_tgt])
        report(f"{src}->{tgt} cross-LOGO [{pname}]",
               cross_logo(Ps, ys, gs, Pt, yt, gt, common), chance)


def seis():
    from retro_loaders import load_seismic_depth
    a = np.load(f"{BASE}/seismic/bank_depth.npz", allow_pickle=True)
    k = np.load(f"{BASE}/seismic/bank_depth_KONO.npz", allow_pickle=True)
    pa = [(m[0], m[1]) for m in a["meta"]]; pk = [(m[0], m[1]) for m in k["meta"]]
    assert pa == pk, "event order differs between stations"
    run_pair("SEIS-depth", load_seismic_depth(), load_seismic_depth(sta="KONO"),
             "ANMO", "KONO", extra_progs=["hent(clip(diff(x)))"])


def geomag():
    from harvest2_loaders import load_geomag, STORM_DAYS, QUIET_DAYS
    raw_bou = load_geomag()
    raw_frd = load_geomag(obs="FRD")
    # align gids by day: BOU cache predates the days field -> BOU got all 16
    # days in list order (16 recs verified); FRD npz stores its day list.
    frd_days = list(np.load(f"{BASE}/geomag/frd_storm_quiet.npz",
                            allow_pickle=True)["days"])
    bou_days = STORM_DAYS + QUIET_DAYS
    assert len(set(r[2] for r in raw_bou)) == len(bou_days), "BOU day/gid drift"
    day2bou = {d: i for i, d in enumerate(bou_days)}
    raw_frd = [(w, lab, day2bou[frd_days[gid]]) for w, lab, gid in raw_frd]
    run_pair("GEOMAG-storm", raw_bou, raw_frd, "BOU", "FRD")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("seis", "both"): seis()
    if which in ("geomag", "both"): geomag()
