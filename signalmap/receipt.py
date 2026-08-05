"""Receipt v1: versioned, Ed25519-signed verdict receipts.

Every verdict-producing command wraps its claim in a self-describing JSON
receipt (schema, claim, verdict, evidence, input hashes, tool+version,
timestamp, countersignatures) and signs it. The private key lives in
``$SIGNALMAP_HOME/signing_key`` (default ``~/.signalmap``, mode 0600) and
never enters the repo; only the raw public key travels inside the receipt,
so a third party can check integrity offline and pin the key out of band.

Signing pattern reused from ``indelible``: canonical JSON bytes (sorted
keys, tight separators, signature field excluded) signed with raw Ed25519.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_V1 = "signalmap.receipt/1"
VERDICTS = ("INCLUDED", "EXCLUDED", "PASS", "REFUSED")


def _tool_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("signalmap")
    except PackageNotFoundError:
        return "0+unknown"


def build_receipt(*, claim: str, verdict: str, evidence: dict,
                  input_hashes: dict) -> dict:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    return {
        "schema": SCHEMA_V1,
        "claim": claim,
        "verdict": verdict,
        "evidence": evidence,
        "input_hashes": input_hashes,
        "tool": "signalmap",
        "tool_version": _tool_version(),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "countersignatures": [],
    }


def canonical_bytes(receipt: dict) -> bytes:
    """Deterministic signing payload: everything except the signature."""
    body = {k: v for k, v in receipt.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode()


def default_key_path() -> Path:
    home = os.environ.get("SIGNALMAP_HOME") or os.path.expanduser("~/.signalmap")
    return Path(home) / "signing_key"


def load_or_create_key(path=None):
    """Load the Ed25519 signing key, creating it (0600) on first use."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        load_pem_private_key,
    )
    p = Path(path) if path is not None else default_key_path()
    if p.exists():
        key = load_pem_private_key(p.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError(f"{p} is not an Ed25519 key")
        return key
    key = Ed25519PrivateKey.generate()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    return key


def sign_receipt(receipt: dict, private_key) -> dict:
    """Return a copy with embedded pubkey + signature (both hex)."""
    signed = dict(receipt)
    signed["pubkey"] = private_key.public_key().public_bytes_raw().hex()
    signed["signature"] = private_key.sign(canonical_bytes(signed)).hex()
    return signed


def verify_receipt(receipt: dict) -> bool:
    """True iff the signature matches the receipt body under its own pubkey.

    Integrity only — authenticity additionally needs the pubkey pinned out
    of band (standalone verifier, V2).
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(receipt["pubkey"]))
        pub.verify(bytes.fromhex(receipt["signature"]), canonical_bytes(receipt))
        return True
    except (InvalidSignature, KeyError, ValueError, TypeError):
        return False


def write_receipt(receipt: dict, path) -> None:
    Path(path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def hash_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_bank(bank_dir) -> str:
    """Order-independent digest over (relative name, content hash) of every
    file in the bank directory tree."""
    root = Path(bank_dir)
    entries = sorted((str(p.relative_to(root)), hash_file(p))
                     for p in root.rglob("*") if p.is_file())
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def emit_signed_receipt(*, claim: str, verdict: str, evidence: dict,
                        input_hashes: dict, out_path) -> dict:
    """Build, sign with the persistent key, and write — the one-call surface
    for CLI commands."""
    receipt = build_receipt(claim=claim, verdict=verdict, evidence=evidence,
                            input_hashes=input_hashes)
    signed = sign_receipt(receipt, load_or_create_key())
    write_receipt(signed, out_path)
    return signed
