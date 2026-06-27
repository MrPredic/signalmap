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


def cmd_benchmark(args):
    from .benchmark import run
    run(args.dataset, args.epochs, args.anomaly_label)


def cmd_ingest(args):
    from .ingest import ingest
    ingest(args.path, args.label, args.out, args.sr, args.column, args.sensor_class)


def cmd_discover(args):
    from .discover import run
    run(confounds=args.confound, naive=args.naive, n=args.n)


def cmd_fit(args):
    from .monitor import fit_from_dataset
    fit_from_dataset(args.dataset, args.out, healthy_label=args.healthy_label,
                     epochs=args.epochs, threshold=args.threshold)


def cmd_monitor(args):
    from .detector import Detector
    from .monitor import run
    det = Detector.load(args.detector)
    run(det, _build_source(args).frames(), quiet=args.quiet)


def main() -> None:
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

    f = sub.add_parser("fit", help="fit an anomaly detector on healthy data (no labels)")
    f.add_argument("--dataset", required=True)
    f.add_argument("--out", default="artifacts/detector.pt")
    f.add_argument("--healthy-label", default="", help="substring selecting healthy rows (default: all)")
    f.add_argument("--epochs", type=int, default=40)
    f.add_argument("--threshold", type=float, default=4.0, help="alert z-score threshold")
    f.set_defaults(func=cmd_fit)

    mo = sub.add_parser("monitor", help="monitor a source with a fitted detector")
    mo.add_argument("--detector", default="artifacts/detector.pt")
    mo.add_argument("--source", default="replay")
    mo.add_argument("--dataset")
    mo.add_argument("--broker", default="localhost")
    mo.add_argument("--count", type=int, default=200)
    mo.add_argument("--quiet", action="store_true")
    mo.set_defaults(func=cmd_monitor)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
