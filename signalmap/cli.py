"""signalmap — unified CLI for the open signal-intelligence platform.

    signalmap plugins                       list everything pluggable
    signalmap run --source sim --sink stdout --limit 20
    signalmap run --source replay --dataset data/x.parquet --weights artifacts/model.pt --sink parquet
    signalmap train --synthetic 2000 --epochs 30
    signalmap map --dataset data/x.parquet --model artifacts/model.pt
    signalmap universal                     full cross-domain proof
"""
from __future__ import annotations

import argparse

# importing the adapter modules registers all built-in plugins
from . import models, sinks, sources, transforms  # noqa: F401
from .core import Pipeline, available, get


def _build_source(args):
    name = args.source
    cls = get("source", name)
    if name == "sim":
        return cls(count=args.count)
    if name == "replay":
        if not args.dataset:
            raise SystemExit("--dataset required for replay source")
        return cls(args.dataset)
    if name == "mic":
        return cls(count=args.count)
    if name == "mqtt":
        return cls(broker=args.broker, count=args.count)
    return cls()


def cmd_run(args):
    source = _build_source(args)
    transform = get("transform", args.transform)(n_bins=args.n_bins)
    model = get("model", args.model)(n_bins=args.n_bins, weights=args.weights)
    sink_objs = [get("sink", s)() for s in (args.sink or ["stdout"])]
    n = Pipeline(source, transform, model, sink_objs).run(limit=args.limit)
    print(f"done: {n} frames")


def cmd_plugins(_args):
    for kind in ("source", "transform", "model", "sink"):
        print(f"{kind:9s}: {', '.join(available(kind)) or '(none)'}")


def cmd_train(args):
    from .train import load_dataset, synthetic_dataset, train
    if args.dataset:
        feats = load_dataset(args.dataset)
    elif args.synthetic:
        feats = synthetic_dataset(args.synthetic)
    else:
        raise SystemExit("provide --dataset PATH or --synthetic N")
    train(feats, args.epochs, args.out)


def cmd_map(args):
    from .visualize import build
    build(args.dataset, args.model, args.out)


def cmd_universal(_args):
    from .simulate_universal import main as u
    u()


def _load_prove_dataset(path):
    """Strictly validate an NPZ bank of X/y/groups before scoring.

    allow_pickle stays False (an .npz is a trust boundary like any model file),
    and shape/dtype/length are checked so a malformed bank fails loudly instead
    of producing a silently wrong receipt.
    """
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in ("X", "y", "groups") if k not in data.files]
        if missing:
            raise ValueError(f"dataset {path} is missing arrays: {missing}")
        X = np.asarray(data["X"])
        y = np.asarray(data["y"])
        groups = np.asarray(data["groups"])
    if X.ndim not in (2, 3):
        raise ValueError(f"X must be (n,time) or (n,channels,time), got shape {X.shape}")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"X must be numeric, got dtype {X.dtype}")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError(f"X/y/groups length mismatch: {len(X)}/{len(y)}/{len(groups)}")
    if len(X) == 0:
        raise ValueError("dataset is empty")
    return X, y, groups


SYNTHETIC_PROVE_PER_GROUP = 6
SYNTHETIC_PROVE_MIN_GROUPS = 20


def synthetic_prove_groups(n_windows: int) -> int:
    """Group count of the synthetic prove demo.

    The floor is 20, not 12: the discovery/replication split leaves ~40 % of
    the groups for confirmation, and a permutation test over 6 replication
    groups cannot report a p below 0.118 (C(6,3)=20 assignments, identity and
    swap both perfect) — so no planted effect, however clean, could ever be
    confirmed. Measured 2026-08-05, see tests/test_prove_resolution.py.
    """
    return max(SYNTHETIC_PROVE_MIN_GROUPS, n_windows // SYNTHETIC_PROVE_PER_GROUP)


def cmd_prove(args):
    import numpy as np
    from .composer import compose, write_receipt

    if args.dataset:
        windows, labels, groups = _load_prove_dataset(args.dataset)
    else:
        rng = np.random.default_rng(args.seed)
        per = SYNTHETIC_PROVE_PER_GROUP
        n_groups = synthetic_prove_groups(args.synthetic)
        n = n_groups * per
        groups = np.repeat(np.arange(n_groups), per)
        labels = np.repeat(np.arange(n_groups) % 2, per)
        windows = rng.normal(size=(n, 2, 128))
        # Planted mixed measurement: channel 0 is stable; channel 1 changes.
        # The baseline (channel 0 only) therefore has no label signal.
        windows[:, 1, :] *= np.where(labels == 1, 2.0, 0.5)[:, None]
    receipt = compose(windows, labels, groups, budget=args.budget,
                      seed=args.seed, permutations=args.perms)
    write_receipt(receipt, args.out)
    best = receipt.get("best") or {}
    print(f"composer: {len(receipt['candidates'])} candidates; "
          f"best={best.get('verdict', 'none')} delta={best.get('delta', 0):.3f}; "
          f"receipt={args.out}")
    if best.get("verdict") != "supported":
        res = receipt["resolution"]
        print(f"  permutation floor: discovery {res['discovery']['expected_floor']:.4f} "
              f"({res['discovery']['groups']} groups) · replication "
              f"{res['replication']['expected_floor']:.4f} "
              f"({res['replication']['groups']} groups) — a p below the floor "
              f"is unreachable at this group count")


def cmd_benchmark(args):
    from .benchmark import run
    run(args.dataset, args.epochs, args.anomaly_label)


def cmd_ingest(args):
    from .ingest import ingest
    ingest(args.path, args.label, args.out, args.sr, args.column, args.sensor_class)


def cmd_discover(args):
    from .discover import run
    run(confounds=args.confound, naive=args.naive, n=args.n)


def cmd_causal(args):
    from .causal_discover import run, run_file
    if args.csv:
        run_file(args.csv, columns=args.column)
    else:
        run(n=args.n, seed=args.seed)


def cmd_fit(args):
    if args.spec:
        if not args.bank:
            raise SystemExit("fit --spec needs --bank (directory of healthy "
                             "recordings, same layout as distill --bank)")
        from .monitor import fit_spec_backend
        from .receipt import emit_signed_receipt, hash_bank, hash_file
        det = fit_spec_backend(args.spec, args.bank, args.out)
        emit_signed_receipt(
            claim=f"detector fitted on healthy bank {args.bank} via spec {args.spec}",
            verdict="PASS",
            evidence={"threshold": float(det.threshold)},
            input_hashes={"bank": hash_bank(args.bank),
                          "spec": hash_file(args.spec)},
            out_path=args.out + ".receipt.json")
        return
    if not args.dataset:
        raise SystemExit("fit needs --dataset (parquet) or --spec + --bank")
    from .monitor import fit_from_dataset
    fit_from_dataset(args.dataset, args.out, healthy_label=args.healthy_label,
                     epochs=args.epochs, threshold=args.threshold)


def cmd_monitor(args):
    if args.bank:
        from .monitor import monitor_spec_backend
        from .receipt import emit_signed_receipt, hash_bank, hash_file
        summary = monitor_spec_backend(args.detector, args.bank, quiet=args.quiet)
        emit_signed_receipt(
            claim=f"monitor run of {args.bank} against detector {args.detector}",
            verdict="PASS",
            evidence=summary,
            input_hashes={"bank": hash_bank(args.bank),
                          "detector": hash_file(args.detector)},
            out_path=args.detector + ".monitor.receipt.json")
        return
    from .detector import Detector
    from .monitor import run
    det = Detector.load(args.detector)
    run(det, _build_source(args).frames(), quiet=args.quiet)


def cmd_distill(args):
    from .distill import run_cli
    family_families = None
    if args.registry:
        if not all((args.source_id, args.device_id, args.setup_id)):
            raise SystemExit("--registry needs --source-id, --device-id, and --setup-id")
        from .distill import load_bank
        from .qualification import MethodRegistry, profile_bank
        bank = load_bank(args.bank, args.label_by, args.column, args.pattern,
                         args.multichannel, args.channel_axis)
        profile = profile_bank(bank, source_id=args.source_id,
                               device_id=args.device_id, setup_id=args.setup_id)
        registry = MethodRegistry.load(args.registry)
        status = registry.status(profile)
        if status != "confirmed":
            raise SystemExit(
                f"source qualification status is {status}; refusing routed distill")
        family_families = {
            name for name, record in registry.entry(profile)["families"].items()
            if record.get("state") == "confirmed"
        }
    run_cli(args.bank, args.label_by, args.budget_c, args.out,
            report_out=args.report, pattern=args.pattern, column=args.column,
            n_perm=args.perms, trees=args.trees,
            premium=tuple(args.premium.split(",")) if args.premium else (),
            multichannel=args.multichannel, channel_axis=args.channel_axis,
            family_families=family_families)


def cmd_corpus(args):
    from . import corpus
    from pathlib import Path
    import json
    if corpus.sources_present():
        paths = corpus.build_corpus(out_dir=args.out)
        if args.out is None:
            # Keep the copy that ships in the wheel in step with the repo one,
            # so a pip user never sees a different verdict list than a clone.
            corpus.sync_packaged(paths)
    else:
        # Installed from a wheel: no reports to rebuild from, so list the
        # receipts that shipped with the package. They verify all the same.
        paths = corpus.packaged_receipts(out_dir=args.out)
    receipts = [json.loads(p.read_text()) for p in paths]
    cwd = Path.cwd().resolve()
    for p, r in zip(paths, receipts):
        # The default out dir is absolute (repo root). Show it relative where
        # that is unambiguous, so a pasted listing carries no home directory.
        try:
            shown = Path(p).resolve().relative_to(cwd)
        except ValueError:
            shown = p
        print(f"{r['verdict']:<8} {r['evidence']['bank']:<11} "
              f"{r['evidence']['family']:<9} {shown}")
    print(corpus.traction_line(receipts))



def cmd_checkup(args) -> int:
    """Answer the first question a newcomer actually has, with a verdict."""
    from .checkup import run_cli
    res = run_cli(args.bank, label_by=args.label_by, pattern=args.pattern,
                  column=args.column, healthy_class=args.healthy_class,
                  out=args.out)
    return 0 if res.get("verdict") == "SEPARATES" else 1

def cmd_qualify(args):
    from .distill import load_bank
    from .qualification import MethodRegistry, profile_bank, route_method_families
    from .methods import MethodCatalog, route_capabilities
    import json

    bank = load_bank(args.bank, label_by=args.label_by, column=args.column,
                     pattern=args.pattern, multichannel=args.multichannel,
                     channel_axis=args.channel_axis)
    profile = profile_bank(bank, source_id=args.source_id,
                           device_id=args.device_id, setup_id=args.setup_id)
    routing = route_method_families(profile)
    import os
    registry_path = args.register or args.registry
    registry = (MethodRegistry.load(registry_path)
                if registry_path and os.path.exists(registry_path)
                else MethodRegistry())
    status = registry.status(profile)
    if args.register:
        if not args.receipts:
            raise SystemExit("--register needs --receipts RECEIPTS.json")
        with open(args.receipts, encoding="utf-8") as fh:
            receipt_payload = json.load(fh)
        receipts = receipt_payload.get("families", receipt_payload)
        if not isinstance(receipts, dict):
            raise SystemExit("receipts JSON must be an object or contain a families object")
        registry.register(profile, routing, receipts=receipts)
        registry.save(args.register)
    payload = {
        "status": status,
        "profile": profile.to_dict(),
        "routing": routing.to_dict(),
    }
    if args.register:
        payload["registry"] = args.register
    if args.catalog:
        with open(args.catalog, encoding="utf-8") as fh:
            catalog = MethodCatalog.from_dict(json.load(fh))
        payload["capabilities"] = route_capabilities(profile, catalog).to_dict()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"qualification: {payload['status']} ({len(routing.compatible)} compatible families)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="signalmap")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a Source->Transform->Model->Sink pipeline")
    r.add_argument("--source", default="sim")
    r.add_argument("--transform", default="fft")
    r.add_argument("--model", default="autoencoder")
    r.add_argument("--sink", action="append", help="repeatable; default stdout")
    r.add_argument("--dataset")
    r.add_argument("--weights")
    r.add_argument("--broker", default="localhost")
    r.add_argument("--count", type=int, default=200)
    r.add_argument("--limit", type=int)
    r.add_argument("--n-bins", type=int, default=256)
    r.set_defaults(func=cmd_run)

    sub.add_parser("plugins", help="list registered plugins").set_defaults(func=cmd_plugins)

    t = sub.add_parser("train", help="train the Conv-AE")
    t.add_argument("--dataset")
    t.add_argument("--synthetic", type=int)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--out", default="artifacts/model.pt")
    t.set_defaults(func=cmd_train)

    m = sub.add_parser("map", help="render latent-map HTML")
    m.add_argument("--dataset", required=True)
    m.add_argument("--model", default="artifacts/model.pt")
    m.add_argument("--out", default="artifacts/latent_map.html")
    m.set_defaults(func=cmd_map)

    sub.add_parser("universal", help="cross-domain platform proof").set_defaults(func=cmd_universal)

    pr = sub.add_parser("prove", help="bounded compositional search with null/holdout receipt")
    pr.add_argument("--dataset", help="NPZ with X, y, groups arrays")
    pr.add_argument("--synthetic", type=int, default=120,
                    help="synthetic window count when --dataset is omitted "
                         "(>=120 keeps both splits able to resolve p<0.05)")
    pr.add_argument("--budget", type=int, default=24)
    pr.add_argument("--perms", type=int, default=200,
                    help="permutation draws; 50 caps the reportable "
                         "p at 1/51 and cannot confirm a small split")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", default="artifacts/composer_receipt.json")
    pr.set_defaults(func=cmd_prove)

    b = sub.add_parser("benchmark", help="ROC-AUC anomaly benchmark (synthetic or real)")
    b.add_argument("--dataset", help="Parquet of raw frames (default: synthetic)")
    b.add_argument("--epochs", type=int, default=40)
    b.add_argument("--anomaly-label", default="ANOMALY")
    b.set_defaults(func=cmd_benchmark)

    g = sub.add_parser("ingest-file", help="convert a WAV/CSV/NPY recording into frames")
    g.add_argument("path")
    g.add_argument("--label", required=True)
    g.add_argument("--out", default="data/real.parquet")
    g.add_argument("--sr", type=int, default=16000)
    g.add_argument("--column", type=int, default=0)
    g.add_argument("--sensor-class", type=int, default=0)
    g.set_defaults(func=cmd_ingest)

    d = sub.add_parser("discover", help="cross-modal coupling discovery (confound-adjusted)")
    d.add_argument("--confound", action="append", help="channel(s) to treat as confound")
    d.add_argument("--naive", action="store_true", help="no confound removal (shows the trap)")
    d.add_argument("--n", type=int, default=2000)
    d.set_defaults(func=cmd_discover)

    cz = sub.add_parser("causal", help="directed causal-graph discovery + root-cause (CCM/EDM)")
    cz.add_argument("--csv", help="real recording (CSV/NPY); omit for synthetic demo")
    cz.add_argument("--column", action="append", help="repeatable; select channels")
    cz.add_argument("--n", type=int, default=1500)
    cz.add_argument("--seed", type=int, default=0)
    cz.set_defaults(func=cmd_causal)

    f = sub.add_parser("fit", help="fit an anomaly detector on healthy data (no labels)")
    f.add_argument("--dataset", help="parquet of healthy frames (spectral-AE backend)")
    f.add_argument("--spec", help="distilled spec.json backend (from `signalmap "
                                  "distill`); fit on --bank recordings instead "
                                  "of --dataset, threshold self-calibrates")
    f.add_argument("--bank", help="directory of healthy recordings (with --spec)")
    f.add_argument("--out", default="artifacts/detector.pt")
    f.add_argument("--healthy-label", default="", help="substring selecting healthy rows (default: all)")
    f.add_argument("--epochs", type=int, default=40)
    f.add_argument("--threshold", type=float, default=4.0, help="alert z-score threshold")
    f.set_defaults(func=cmd_fit)

    mo = sub.add_parser("monitor", help="monitor a source with a fitted detector")
    mo.add_argument("--detector", default="artifacts/detector.pt")
    mo.add_argument("--source", default="replay")
    mo.add_argument("--dataset")
    mo.add_argument("--bank", help="score a directory of recordings offline "
                                   "with a distilled detector.json")
    mo.add_argument("--broker", default="localhost")
    mo.add_argument("--count", type=int, default=200)
    mo.add_argument("--quiet", action="store_true")
    mo.set_defaults(func=cmd_monitor)

    ds = sub.add_parser("distill", help="distill per-domain features with a capacity gate + receipt")
    ds.add_argument("--bank", required=True, help="directory of recordings (one file per recording)")
    ds.add_argument("--label-by", default="prefix", dest="label_by",
                    help="labeling rule: prefix (before first _), stem, or parent dir")
    ds.add_argument("--budget-c", type=int, default=50, dest="budget_c",
                    help="capacity constant C: budget=min(len(grammar), C*n_recordings)")
    ds.add_argument("--out", default="artifacts/spec.json")
    ds.add_argument("--report", help="report .md path (default: <out>_report.md)")
    ds.add_argument("--pattern", default="*", help="glob for recording files within --bank")
    ds.add_argument("--column", type=int, default=0, help="CSV column index for the signal")
    ds.add_argument("--perms", type=int, default=60, help="group-permutation count")
    ds.add_argument("--trees", type=int, default=100)
    ds.add_argument("--premium", default="",
                    help="comma-separated premium families to challenge with "
                         "(e.g. rqa,coherence); cost-receipted, spec-included "
                         "only on a paired-CI-solid win")
    ds.add_argument("--multichannel", action="store_true",
                    help="recordings are multi-channel (.csv with a header of "
                         "channel names, or 2-D .npy); base grammar sees "
                         "channel 0, coherence-style families see all channels")
    ds.add_argument("--channel-axis", type=int, default=None, dest="channel_axis",
                    help=".npy axis holding the channels (default heuristic: "
                         "the longer axis is time)")
    ds.add_argument("--registry", help="confirmed MethodRegistry JSON for source-aware routing")
    ds.add_argument("--source-id", dest="source_id")
    ds.add_argument("--device-id", dest="device_id")
    ds.add_argument("--setup-id", dest="setup_id")
    ds.set_defaults(func=cmd_distill)

    co = sub.add_parser("corpus", help="(re)sign the archived premium verdicts "
                                       "and print the traction line")
    co.add_argument("--out", default=None,
                    help="output directory (default research/factory/receipts)")
    co.set_defaults(func=cmd_corpus)

    ck = sub.add_parser("checkup",
                        help="does this method work on YOUR recordings? "
                             "verdict + direction + whether an alarm would fire")
    ck.add_argument("--bank", required=True,
                    help="directory of recordings, both healthy and known-bad")
    ck.add_argument("--label-by", default="prefix", dest="label_by",
                    help="how the filename encodes the class (default: prefix)")
    ck.add_argument("--pattern", default="*")
    ck.add_argument("--column", type=int, default=0)
    ck.add_argument("--healthy-class", default=None, dest="healthy_class",
                    help="name of the healthy class if it cannot be guessed")
    ck.add_argument("--out", default=None, help="also write the verdict as JSON")
    ck.set_defaults(func=cmd_checkup)

    q = sub.add_parser("qualify", help="profile a source and route compatible method families")
    q.add_argument("--bank", required=True)
    q.add_argument("--source-id", required=True, dest="source_id")
    q.add_argument("--device-id", required=True, dest="device_id")
    q.add_argument("--setup-id", required=True, dest="setup_id")
    q.add_argument("--out", required=True)
    q.add_argument("--registry", help="existing MethodRegistry JSON for status lookup")
    q.add_argument("--register", help="write/update a MethodRegistry after receipt validation")
    q.add_argument("--receipts", help="JSON object or {\"families\": {...}} from validation")
    q.add_argument("--catalog", help="optional MethodCatalog JSON for modular capability routing")
    q.add_argument("--label-by", default="prefix", dest="label_by")
    q.add_argument("--pattern", default="*")
    q.add_argument("--column", type=int, default=0)
    q.add_argument("--multichannel", action="store_true")
    q.add_argument("--channel-axis", type=int, default=None, dest="channel_axis")
    q.set_defaults(func=cmd_qualify)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
