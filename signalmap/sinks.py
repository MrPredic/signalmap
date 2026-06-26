"""Built-in Sinks — consume pipeline Results. All degrade gracefully."""
from __future__ import annotations

import numpy as np

from .core import Result, register


@register("sink", "stdout")
class StdoutSink:
    def __init__(self, every: int = 1):
        self.every = every
        self.n = 0

    def emit(self, r: Result) -> None:
        self.n += 1
        if self.n % self.every == 0:
            print(f"  node={r.meta['node_id']} seq={r.meta['seq']:>5} "
                  f"cls={r.meta['sensor_class']:>3} score={r.score:.5f}")

    def close(self) -> None:
        print(f"  [stdout] {self.n} frames processed")


@register("sink", "parquet")
class ParquetSink:
    """Persist embeddings + scores for offline mapping/clustering."""

    def __init__(self, path: str = "data/embeddings.parquet"):
        self.path = path
        self.rows: list[dict] = []

    def emit(self, r: Result) -> None:
        self.rows.append({
            "node_id": r.meta["node_id"], "seq": r.meta["seq"],
            "sensor_class": r.meta["sensor_class"], "score": r.score,
            "embedding": np.asarray(r.embedding, dtype=np.float32).tobytes(),
        })

    def close(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        cols = {k: [row[k] for row in self.rows] for k in self.rows[0]}
        pq.write_table(pa.table(cols), self.path)
        print(f"  [parquet] wrote {len(self.rows)} embeddings -> {self.path}")


@register("sink", "questdb")
class QuestDBSink:
    def __init__(self, host: str = "localhost", port: int = 9009):
        from .store import QuestDBWriter
        self.w = QuestDBWriter(host, port)

    def emit(self, r: Result) -> None:
        self.w.write_measurement(r.meta["node_id"], r.frame.ts_us,
                                 energy_rms=0.0, recon_error=r.score, anomaly_score=r.score)

    def close(self) -> None:
        pass


@register("sink", "qdrant")
class QdrantSink:
    def __init__(self, collection: str = "signalmap", dim: int = 32):
        from .store import QdrantNovelty
        self.q = QdrantNovelty(collection=collection, dim=dim)

    def emit(self, r: Result) -> None:
        pid = (r.meta["node_id"] << 32) | (r.meta["seq"] & 0xFFFF_FFFF)
        self.q.upsert(pid, np.asarray(r.embedding), {
            "node_id": r.meta["node_id"], "seq": r.meta["seq"],
            "sensor_class": r.meta["sensor_class"], "score": r.score,
        })

    def close(self) -> None:
        pass
