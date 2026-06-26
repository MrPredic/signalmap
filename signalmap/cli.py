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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
