"""Persistence sinks for the kept Docker stack (QuestDB + Qdrant).

Both sinks degrade gracefully: if the server is down, they warn once and become
no-ops so the pipeline/demo never hard-fails during bootstrap.

  * QuestDBWriter  -> raw signal energy + embeddings via InfluxDB line protocol
                      (ILP) over TCP :9009.
  * QdrantNovelty  -> upsert embeddings, query kNN distance = novelty signal
                      that replaces the energy-only proxy in the ingestor.
"""
from __future__ import annotations

import socket

import numpy as np


class QuestDBWriter:
    def __init__(self, host: str = "localhost", port: int = 9009) -> None:
        self.addr = (host, port)
        self.sock: socket.socket | None = None
        try:
            self.sock = socket.create_connection(self.addr, timeout=2)
        except OSError as e:
            print(f"  [questdb] offline ({e}); writes are no-ops")

    def write_measurement(self, node_id: int, ts_us: int, energy_rms: float,
                          recon_error: float, anomaly_score: float) -> None:
        if not self.sock:
            return
        line = (
            f"signal,node={node_id} "
            f"energy_rms={energy_rms},recon_error={recon_error},"
            f"anomaly_score={anomaly_score} {ts_us * 1000}\n"  # ILP wants ns
        )
        try:
            self.sock.sendall(line.encode())
        except OSError as e:
            print(f"  [questdb] write failed: {e}")
            self.sock = None


class QdrantNovelty:
    """Embedding store + kNN-distance novelty. Brute distance until the index
    is warm; that is fine for the thousands of points we have pre-investment."""

    def __init__(self, collection: str = "signalmap", dim: int = 32,
                 host: str = "localhost", port: int = 6333) -> None:
        self.collection = collection
        self.dim = dim
        self.client = None
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.client = QdrantClient(host=host, port=port, timeout=2)
            existing = {c.name for c in self.client.get_collections().collections}
            if collection not in existing:
                self.client.create_collection(
                    collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        except Exception as e:  # connection or import
            print(f"  [qdrant] offline ({e}); novelty falls back to 1.0")
            self.client = None

    def novelty(self, vector: np.ndarray, k: int = 5) -> float:
        """Mean distance to k nearest existing points. Higher = more novel."""
        if not self.client:
            return 1.0
        try:
            hits = self.client.search(
                self.collection, query_vector=vector.tolist(), limit=k
            )
        except Exception:
            return 1.0
        if not hits:
            return 1.0  # empty index -> maximally novel
        # cosine score in [-1,1]; distance = 1 - mean(score)
        return float(1.0 - np.mean([h.score for h in hits]))

    def upsert(self, point_id: int, vector: np.ndarray, payload: dict) -> None:
        if not self.client:
            return
        from qdrant_client.models import PointStruct

        try:
            self.client.upsert(
                self.collection,
                points=[PointStruct(id=point_id, vector=vector.tolist(), payload=payload)],
            )
        except Exception as e:
            print(f"  [qdrant] upsert failed: {e}")
