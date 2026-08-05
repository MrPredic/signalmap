"""``qdrant_client.search`` is deprecated in favor
of ``query_points``; a bare ``except`` used to swallow that silently and
report "no novelty" (1.0) forever. The store must (a) prefer the modern API,
(b) keep working against old clients that only have ``search``, and (c) be
LOUD when the query path breaks instead of degrading silently.

Unit-level: clients are injected stubs — no server, no qdrant install needed.
"""
import numpy as np

from signalmap.store import QdrantNovelty


def _bare(client):
    q = QdrantNovelty.__new__(QdrantNovelty)
    q.collection = "c"
    q.dim = 2
    q.client = client
    return q


class _Hit:
    def __init__(self, score):
        self.score = score


class _ModernClient:
    """Post-deprecation client: only query_points exists."""

    def __init__(self):
        self.calls = []

    def query_points(self, collection, query, limit):
        self.calls.append((collection, tuple(query), limit))
        class R:  # QueryResponse shape
            points = [_Hit(0.9), _Hit(0.7)]
        return R()


class _LegacyClient:
    """Old client: only search exists."""

    def search(self, collection, query_vector, limit):
        return [_Hit(0.5)]


class _BrokenClient:
    def query_points(self, collection, query, limit):
        raise RuntimeError("boom")


def test_novelty_uses_query_points_on_modern_client():
    client = _ModernClient()
    q = _bare(client)
    val = q.novelty(np.array([1.0, 0.0]), k=2)
    assert client.calls == [("c", (1.0, 0.0), 2)]
    assert val == np.float64(1.0 - 0.8)


def test_novelty_falls_back_to_search_on_legacy_client():
    q = _bare(_LegacyClient())
    assert q.novelty(np.array([1.0, 0.0])) == 0.5


def test_novelty_degrades_loudly_not_silently(capsys):
    q = _bare(_BrokenClient())
    assert q.novelty(np.array([1.0, 0.0])) == 1.0
    out = capsys.readouterr().out
    assert "qdrant" in out and "boom" in out, (
        "query failure must be reported, not swallowed")
