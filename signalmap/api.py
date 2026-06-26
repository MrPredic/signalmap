"""FastAPI read API over the latent landscape.

Endpoints:
  GET /health                  liveness
  GET /anomalies?limit=20      top high-performance outliers
  GET /map                     2D UMAP projection of the latent space
  GET /node/{node_id}          recent points for one node

Backed by Qdrant in production; here the store is pluggable so the slice runs
in-memory. UMAP is computed lazily and cached.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from fastapi import FastAPI


class PointStore(Protocol):
    def all_points(self) -> list[dict]: ...
    def top_anomalies(self, limit: int) -> list[dict]: ...
    def node_points(self, node_id: int) -> list[dict]: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._pts: list[dict] = []

    def add(self, vector: np.ndarray, payload: dict) -> None:
        self._pts.append({"vector": vector, **payload})

    def all_points(self) -> list[dict]:
        return self._pts

    def top_anomalies(self, limit: int) -> list[dict]:
        ordered = sorted(self._pts, key=lambda p: p.get("anomaly_score", 0.0), reverse=True)
        return [{k: v for k, v in p.items() if k != "vector"} for p in ordered[:limit]]

    def node_points(self, node_id: int) -> list[dict]:
        return [
            {k: v for k, v in p.items() if k != "vector"}
            for p in self._pts
            if p.get("node_id") == node_id
        ]


def create_app(store: PointStore) -> FastAPI:
    app = FastAPI(title="SignalMap", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "points": len(store.all_points())}

    @app.get("/anomalies")
    def anomalies(limit: int = 20) -> list[dict]:
        return store.top_anomalies(limit)

    @app.get("/node/{node_id}")
    def node(node_id: int) -> list[dict]:
        return store.node_points(node_id)

    @app.get("/map")
    def latent_map() -> list[dict]:
        pts = store.all_points()
        if len(pts) < 3:
            return []
        import umap  # lazy: heavy dependency

        vecs = np.array([p["vector"] for p in pts])
        coords = umap.UMAP(n_components=2, n_neighbors=min(15, len(pts) - 1)).fit_transform(vecs)
        return [
            {
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "node_id": pts[i].get("node_id"),
                "anomaly_score": pts[i].get("anomaly_score"),
            }
            for i in range(len(pts))
        ]

    return app
