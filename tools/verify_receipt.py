#!/usr/bin/env python3
"""Standalone signalmap receipt verifier — imports NOTHING from signalmap.

Anyone can check a receipt offline with just this file, Python, and the
``cryptography`` package:

    python tools/verify_receipt.py receipt.json [--pubkey HEX]

Checks, in order:
  1. schema is ``signalmap.receipt/1`` and all required fields are present
  2. verdict is one of INCLUDED / EXCLUDED / PASS / REFUSED
  3. internal consistency: a REFUSED verdict carries no deploy-spec entry
  4. Ed25519 signature over the canonical body matches the embedded pubkey
  5. with ``--pubkey``: the embedded pubkey equals the pinned one
     (without a pin, the check proves integrity, not authenticity)

Exit 0 and print PASS, or exit 1 and print FAIL with every violated check.
"""
import argparse
import json
import sys

SCHEMA = "signalmap.receipt/1"
VERDICTS = ("INCLUDED", "EXCLUDED", "PASS", "REFUSED")
REQUIRED = ("schema", "claim", "verdict", "evidence", "input_hashes", "tool",
            "tool_version", "created_at", "countersignatures", "pubkey",
            "signature")


def canonical_bytes(receipt):
    body = {k: v for k, v in receipt.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode()


def check(receipt, pinned_pubkey=None):
    """Return a list of violated-check descriptions (empty == PASS)."""
    errors = []
    for field in REQUIRED:
        if field not in receipt:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors

    if receipt["schema"] != SCHEMA:
        errors.append(f"unsupported schema: {receipt['schema']!r} "
                      f"(expected {SCHEMA})")
    if receipt["verdict"] not in VERDICTS:
        errors.append(f"unknown verdict: {receipt['verdict']!r}")
    if (receipt["verdict"] == "REFUSED"
            and isinstance(receipt["evidence"], dict)
            and "deploy_spec" in receipt["evidence"]):
        errors.append("inconsistent: REFUSED verdict carries a deploy_spec")

    prov = (receipt["evidence"].get("provenance")
            if isinstance(receipt["evidence"], dict) else None)
    if isinstance(prov, dict) and prov.get("mode") == "archive_signature":
        if prov.get("rerun") is not False:
            errors.append("inconsistent: archive_signature must declare "
                          "rerun: false (it was not re-executed)")
        if not prov.get("source_sha256"):
            errors.append("inconsistent: archive_signature lacks "
                          "source_sha256 of the report it transcribes")
        if not prov.get("source_report"):
            errors.append("inconsistent: archive_signature lacks "
                          "source_report")

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError:
        # This script is meant to run in an environment that has nothing to do
        # with signalmap, so say what is missing instead of raising a traceback.
        raise SystemExit(
            "verify_receipt needs Ed25519 verification: pip install cryptography")
    try:
        pub = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(receipt["pubkey"]))
        pub.verify(bytes.fromhex(receipt["signature"]),
                   canonical_bytes(receipt))
    except (InvalidSignature, ValueError, TypeError):
        errors.append("signature does not match receipt body")

    if pinned_pubkey is not None and receipt["pubkey"] != pinned_pubkey:
        errors.append("pubkey mismatch: receipt is not signed by the "
                      "pinned key")
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("receipt", help="path to a receipt JSON file")
    ap.add_argument("--pubkey", help="pinned Ed25519 public key (hex); "
                    "required for authenticity, not just integrity")
    args = ap.parse_args(argv)

    try:
        with open(args.receipt, encoding="utf-8") as fh:
            receipt = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read receipt: {exc}")
        return 1
    if not isinstance(receipt, dict):
        print("FAIL: receipt is not a JSON object")
        return 1

    errors = check(receipt, pinned_pubkey=args.pubkey)
    if errors:
        print(f"FAIL: {args.receipt}")
        for e in errors:
            print(f"  - {e}")
        return 1
    scope = "authentic (pinned key)" if args.pubkey else "integrity only"
    print(f"PASS: {args.receipt} — verdict {receipt['verdict']}, {scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
