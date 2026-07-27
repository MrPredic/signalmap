"""Transport + provenance adapters: make discovery results MCP/JSON-safe and
auditable. ``jsonable`` normalizes tuple-keyed edge dicts and numpy scalars;
``run_manifest`` wraps any result in a self-describing audit envelope
(version, method, params, seed, input hash) so a claim can be traced to the
exact code and data that produced it — the "receipt" every result must carry.
"""
from __future__ import annotations

import hashlib

import numpy as np

__version__ = "0.1.0"


def jsonable(obj):
    """Recursively convert a result into plain JSON-encodable types.

    Tuple dict-keys become "A->B" strings; numpy scalars/arrays become Python
    numbers/lists; sets become sorted lists.
    """
    if isinstance(obj, dict):
        return {("->".join(map(str, k)) if isinstance(k, tuple) else str(k)
                 if not isinstance(k, (str, int, float, bool)) and k is not None else k):
                jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(jsonable(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _hash_inputs(inputs) -> str:
    h = hashlib.sha256()
    if isinstance(inputs, dict):
        for k in sorted(inputs):
            h.update(str(k).encode())
            h.update(np.ascontiguousarray(np.asarray(inputs[k], float)).tobytes())
    else:
        h.update(np.ascontiguousarray(np.asarray(inputs, float)).tobytes())
    return h.hexdigest()


def run_manifest(result, method: str, params: dict, seed: int | None,
                 inputs) -> dict:
    """Wrap ``result`` in an auditable, JSON-transportable envelope."""
    return {
        "signalmap_version": __version__,
        "method": method,
        "params": jsonable(params),
        "seed": seed,
        "input_hash": _hash_inputs(inputs),
        "result": jsonable(result),
    }
