"""MQTT ingestor: subscribe -> decode -> DSP -> embed -> (score, sink).

Vertical slice. Persistence (QuestDB) and Qdrant upsert are wired as optional
hooks so the pipeline runs end-to-end with zero infra for a first test
(`--dry-run` prints scores to stdout).
"""
from __future__ import annotations

import argparse
import asyncio

import numpy as np

from .dsp import raw_to_features
from .embed import Embedder, anomaly_score
from .frame import GapTracker, decode
from .model import SpectralAutoencoder


async def run(broker: str, topic: str, dry_run: bool) -> None:
    import aiomqtt  # imported here so --help works without the dep

    model = SpectralAutoencoder(n_bins=256, latent_dim=32)
    embedder = Embedder(model)
    gaps = GapTracker()

    print(f"signalmap-ingestor: broker={broker} topic={topic} dry_run={dry_run}")
    async with aiomqtt.Client(broker) as client:
        await client.subscribe(topic)
        async for msg in client.messages:
            try:
                frame = decode(bytes(msg.payload))
            except ValueError as e:
                print(f"  drop bad frame: {e}")
                continue
            if frame.is_spectrum:
                continue  # raw is the source of truth in the vertical slice

            dropped = gaps.check(frame.node_id, frame.seq)
            if dropped:
                print(f"  node={frame.node_id} GAP: {dropped} frames lost")

            feat = raw_to_features(frame.payload.astype(np.float32), frame.sr_hz)
            emb, energy_z = embedder.embed(feat, frame.node_id, frame.seq, frame.ts_us)

            # kNN distance needs a populated index; in the dry slice we use
            # energy_z as the novelty proxy until Qdrant is connected.
            score = anomaly_score(emb.recon_error, knn_distance=1.0, energy_z=energy_z)

            if dry_run:
                print(
                    f"  node={emb.node_id} seq={emb.seq} "
                    f"recon={emb.recon_error:.4f} e_rms={emb.energy_rms:.1f} "
                    f"e_z={energy_z:.2f} score={score:.4f}"
                )
            # else: questdb_write(...) + qdrant_upsert(...)


def main() -> None:
    p = argparse.ArgumentParser(description="SignalMap MQTT ingestor")
    p.add_argument("--broker", default="localhost")
    p.add_argument("--topic", default="signals/+/raw")
    p.add_argument("--dry-run", action="store_true", help="print scores, no infra")
    args = p.parse_args()
    asyncio.run(run(args.broker, args.topic, args.dry_run))


if __name__ == "__main__":
    main()
