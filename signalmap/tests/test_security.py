"""Regression guards for model-file trust boundaries."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


def test_product_torch_loads_require_weights_only():
    """Every product deserialization call must reject arbitrary pickle objects."""
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "load":
                owner = fn.value
                if isinstance(owner, ast.Name) and owner.id == "torch":
                    kw = {k.arg: k.value for k in node.keywords if k.arg}
                    safe = isinstance(kw.get("weights_only"), ast.Constant) and \
                        kw["weights_only"].value is True
                    if not safe:
                        offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, "unsafe torch.load call(s): " + ", ".join(offenders)


def test_detector_rejects_executable_pickle_payload(tmp_path):
    import torch

    from signalmap.detector import Detector

    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return (Path.touch, (marker,))

    artifact = tmp_path / "malicious.pt"
    torch.save(Payload(), artifact)
    with pytest.raises(Exception):
        Detector.load(str(artifact))
    assert not marker.exists(), "untrusted pickle payload executed"
