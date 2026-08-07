#!/usr/bin/env python3
"""Standalone signalmap receipt verifier — imports NOTHING from signalmap.

Anyone can check a receipt offline with just this file, Python, and the
``cryptography`` package:

    python tools/verify_receipt.py receipt.json [--pubkey HEX]

Checks, in order:
  1. the file is strict RFC 8259 JSON: UTF-8, no duplicate keys, no
     ``NaN``/``Infinity``, no unpaired surrogates, no numbers or nesting
     outside what every JSON host can represent — so the bytes that were
     signed are the bytes a reader (in any language) sees
  2. schema is ``signalmap.receipt/1``, all required fields are present and
     of the right type
  3. verdict is one of INCLUDED / EXCLUDED / PASS / REFUSED
  4. internal consistency: a REFUSED verdict endorses nothing — no
     ``deploy_spec`` and no nested PASS/INCLUDED verdict anywhere in its
     evidence; an ``archive_signature`` provenance declares what it
     transcribes; countersignatures are empty (schema/1 defines no format
     for them, so a non-empty list cannot be verified here)
  5. Ed25519 signature over the canonical body matches the embedded pubkey
  6. with ``--pubkey``: the embedded key equals the pinned one
     (without a pin, the check proves integrity, not authenticity)

Exit 0 and print PASS, or exit 1 and print FAIL with every violated check.
"""
import argparse
import hmac
import json
import math
import sys

SCHEMA = "signalmap.receipt/1"
VERDICTS = ("INCLUDED", "EXCLUDED", "PASS", "REFUSED")
ENDORSING = ("INCLUDED", "PASS")
FIELD_TYPES = {"schema": str, "claim": str, "verdict": str, "evidence": dict,
               "input_hashes": dict, "tool": str, "tool_version": str,
               "created_at": str, "countersignatures": list, "pubkey": str,
               "signature": str}
REQUIRED = tuple(FIELD_TYPES)
MAX_DEPTH = 64          # real receipts nest 3-4 deep
MAX_INT = 2 ** 53 - 1   # exactly representable in every JSON host


class ReceiptError(ValueError):
    """A receipt this verifier refuses to interpret — reported, never raised
    past ``main`` (a traceback is not a verdict)."""


def _typename(value):
    return {dict: "object", list: "array", str: "string", bool: "boolean",
            int: "number", float: "number", type(None): "null"}.get(
                type(value), type(value).__name__)


def _no_duplicate_keys(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ReceiptError(
                f"duplicate JSON key {key!r}: the signed bytes would keep one "
                "value while a reader sees another")
        seen.add(key)
    return dict(pairs)


def _reject_constant(token):
    raise ReceiptError(f"{token} is a Python extension, not JSON — a receipt "
                       "must parse in every host")


def _audit(root):
    """Reject anything whose meaning is not identical in every JSON host.

    Iterative on purpose: a hostile file must not blow the Python stack.
    """
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise ReceiptError(f"JSON nested deeper than {MAX_DEPTH} levels")
        if isinstance(node, dict):
            for key, value in node.items():
                stack.append((key, depth))
                stack.append((value, depth + 1))
        elif isinstance(node, list):
            for value in node:
                stack.append((value, depth + 1))
        elif isinstance(node, str):
            try:
                node.encode("utf-8")
            except UnicodeEncodeError:
                raise ReceiptError(
                    "string contains an unpaired surrogate and cannot be "
                    "encoded as UTF-8") from None
        elif isinstance(node, float):
            if not math.isfinite(node):
                raise ReceiptError("number is not finite (NaN or Infinity); "
                                   "note that literals like 1e999 overflow "
                                   "to Infinity")
        elif isinstance(node, int) and not isinstance(node, bool):
            if abs(node) > MAX_INT:
                raise ReceiptError(
                    f"integer {node!r} exceeds 2**53-1 and would be read as a "
                    "different number by other JSON hosts")


def load_strict(raw):
    """Parse receipt bytes under the rules the signature relies on."""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReceiptError(f"not valid UTF-8: {exc}") from None
    try:
        receipt = json.loads(raw, object_pairs_hook=_no_duplicate_keys,
                             parse_constant=_reject_constant)
    except RecursionError:
        raise ReceiptError("JSON is nested too deeply to parse") from None
    except ReceiptError:
        raise
    except ValueError as exc:  # JSONDecodeError, int-conversion limits, ...
        raise ReceiptError(f"not parseable as JSON: {exc}") from None
    if not isinstance(receipt, dict):
        raise ReceiptError("receipt is not a JSON object")
    _audit(receipt)
    return receipt


def canonical_bytes(receipt):
    body = {k: v for k, v in receipt.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _walk(node):
    """Yield every (key, value) pair anywhere inside a parsed structure."""
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for key, value in cur.items():
                yield key, value
                stack.append(value)
        elif isinstance(cur, list):
            stack.extend(cur)


def _consistency_errors(receipt):
    errors = []
    evidence = receipt["evidence"]
    if receipt["verdict"] == "REFUSED":
        for key, value in _walk(evidence):
            if key == "deploy_spec":
                errors.append("inconsistent: REFUSED verdict carries a "
                              "deploy_spec")
                break
        for key, value in _walk(evidence):
            if key == "verdict" and value in ENDORSING:
                errors.append(f"inconsistent: REFUSED verdict carries a nested "
                              f"{value} verdict — a refusal endorses nothing")
                break

    prov = evidence.get("provenance")
    if prov is not None and not isinstance(prov, dict):
        errors.append(f"evidence.provenance must be an object, got "
                      f"{_typename(prov)}")
    elif isinstance(prov, dict) and prov.get("mode") == "archive_signature":
        if prov.get("rerun") is not False:
            errors.append("inconsistent: archive_signature must declare "
                          "rerun: false (it was not re-executed)")
        if not prov.get("source_sha256"):
            errors.append("inconsistent: archive_signature lacks "
                          "source_sha256 of the report it transcribes")
        if not prov.get("source_report"):
            errors.append("inconsistent: archive_signature lacks "
                          "source_report")

    if receipt["countersignatures"]:
        errors.append(
            "countersignatures present, but signalmap.receipt/1 defines no "
            "format for them — this verifier cannot check them, so it will "
            "not vouch for the receipt")
    return errors


def check(receipt, pinned_pubkey=None):
    """Return a list of violated-check descriptions (empty == PASS)."""
    errors = []
    for field in REQUIRED:
        if field not in receipt:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors
    for field, expected in FIELD_TYPES.items():
        value = receipt[field]
        if not isinstance(value, expected) or isinstance(value, bool):
            errors.append(f"field {field} must be a "
                          f"{_typename(expected())}, got {_typename(value)}")
    if errors:
        return errors

    if receipt["schema"] != SCHEMA:
        errors.append(f"unsupported schema: {receipt['schema']!r} "
                      f"(expected {SCHEMA})")
    if receipt["verdict"] not in VERDICTS:
        errors.append(f"unknown verdict: {receipt['verdict']!r}")
    errors.extend(_consistency_errors(receipt))

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
    # Parsing the key and checking the signature are separate questions, and
    # they must fail separately: a tampered body leaves the receipt's own
    # pubkey intact, so blaming the key would point a reader at a key
    # substitution that never happened.
    try:
        pub_bytes = bytes.fromhex(receipt["pubkey"])
        pub = Ed25519PublicKey.from_public_bytes(pub_bytes)
    except (ValueError, TypeError):
        pub_bytes, pub = None, None
        errors.append("pubkey is not a 32-byte hex Ed25519 public key")

    if pub is not None:
        try:
            pub.verify(bytes.fromhex(receipt["signature"]),
                       canonical_bytes(receipt))
        except (InvalidSignature, ValueError, TypeError):
            errors.append("signature does not match receipt body")

    if pinned_pubkey is not None:
        try:
            pin_bytes = bytes.fromhex(pinned_pubkey.strip())
            if len(pin_bytes) != 32:
                raise ValueError("an Ed25519 public key is 32 bytes")
        except ValueError as exc:
            errors.append(f"--pubkey is not a 32-byte hex Ed25519 key: {exc}")
        else:
            # Compare key bytes, not their transcription: the same key written
            # in a different hex case is the same key, and a different key
            # must not leak a match through a prefix or partial compare.
            if pub_bytes is None or not hmac.compare_digest(pub_bytes,
                                                            pin_bytes):
                errors.append("pubkey mismatch: receipt is not signed by the "
                              "pinned key")
    return errors


UNPINNED_WARNING = (
    "  WARNING: no --pubkey given, so this is an integrity check only. It\n"
    "  proves the receipt has not changed since it was signed by the key\n"
    "  embedded in the file itself. It does NOT prove who signed it: anyone\n"
    "  can mint a receipt with their own key. For authenticity, re-run with\n"
    "  --pubkey <hex> using a key you obtained out of band.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("receipt", help="path to a receipt JSON file")
    ap.add_argument("--pubkey", help="pinned Ed25519 public key (hex); "
                    "required for authenticity, not just integrity")
    args = ap.parse_args(argv)

    try:
        with open(args.receipt, "rb") as fh:
            receipt = load_strict(fh.read())
    except OSError as exc:
        print(f"FAIL: cannot read receipt: {exc}")
        return 1
    except ReceiptError as exc:
        print(f"FAIL: {args.receipt}\n  - {exc}")
        return 1

    errors = check(receipt, pinned_pubkey=args.pubkey)
    if errors:
        print(f"FAIL: {args.receipt}")
        for e in errors:
            print(f"  - {e}")
        return 1
    if args.pubkey:
        print(f"PASS: {args.receipt} — verdict {receipt['verdict']}, "
              f"authentic (pinned key)")
    else:
        print(f"PASS: {args.receipt} — verdict {receipt['verdict']}, "
              f"integrity only (NOT authenticity)")
        print(UNPINNED_WARNING)
    return 0


if __name__ == "__main__":
    sys.exit(main())
