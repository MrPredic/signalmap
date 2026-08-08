"""Ed25519 attestation core -- offline-verifiable signed receipts for the
research factory's premium-family verdicts and ledger tip.

WHY (see PRIO2_ATTESTATION_SPEC.md for the full spec): a signed receipt
turns "we ran this and got this verdict" into something a third party can
verify without trusting our infrastructure -- the EU AI Act Art. 15
transparency wedge for PdM/TinyML deployments. This module is deliberately
tiny and dependency-light (`cryptography`'s Ed25519 primitives only, no
network, no custom crypto): sign a canonical JSON receipt with an offline
private key, distribute the public key, anyone can verify offline.

Canonical serialization (`canonical_json`) is `json.dumps(sort_keys=True,
separators=(",", ":"))` -- the exact same convention `receipt_ledger.py`
already uses for its hash-chain (`_canon`), so signing is deterministic and
consistent with the existing ledger.

Usage (library):
    from attestation import generate_keypair, sign_receipt, verify_receipt
    priv, pub = generate_keypair()
    sig = sign_receipt(receipt_dict, priv)
    ok = verify_receipt(receipt_dict, sig, pub)

Usage (CLI, offline, no network):
    attestation.py keygen                              # prints privkey_hex/pubkey_hex
    attestation.py sign-tip <privkey_hex>                # signs {ledger tip hash, ts}
    attestation.py verify-tip <sig_hex> <pubkey_hex> <ledger_tip_hash> <ts>
"""
import json
import os
import sys
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, PublicFormat, NoEncryption,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import receipt_ledger  # noqa: E402


# --------------------------------------------------------- canonical form
def canonical_json(receipt_dict):
    """Deterministic byte-serialization: same dict (any key order, any
    dict-nesting) always produces the exact same bytes, so signing/verifying
    is stable across processes and re-serializations."""
    return json.dumps(receipt_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ------------------------------------------------------------- key mgmt
def generate_keypair():
    """Offline keygen. Returns (privkey_bytes, pubkey_bytes), both 32 raw
    bytes. The privkey MUST stay offline/private; the pubkey is the only
    thing ever distributed for verification."""
    priv = Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub_bytes = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return priv_bytes, pub_bytes


def pubkey_from_privkey(privkey_bytes):
    return Ed25519PrivateKey.from_private_bytes(privkey_bytes).public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw)


# --------------------------------------------------------- sign / verify
def sign_receipt(receipt_dict, privkey_bytes):
    """Sign the canonical form of receipt_dict. Returns raw sig bytes (64)."""
    priv = Ed25519PrivateKey.from_private_bytes(privkey_bytes)
    return priv.sign(canonical_json(receipt_dict))


def verify_receipt(receipt_dict, sig_bytes, pubkey_bytes):
    """True iff sig_bytes is a valid Ed25519 signature of receipt_dict's
    canonical form under pubkey_bytes. Any tampering with any field (or a
    wrong pubkey) returns False -- never raises for a bad signature."""
    pub = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
    try:
        pub.verify(sig_bytes, canonical_json(receipt_dict))
        return True
    except InvalidSignature:
        return False


# ------------------------------------------------------- receipt builder
def build_receipt(config_hash, data_provenance_sha, verdict, ci_low, ci_high,
                  ledger_tip_hash, timestamp=None):
    """Build the canonical attestation receipt per PRIO2_ATTESTATION_SPEC.md
    section "WHAT gets signed". `verdict` is "INCLUDED" or "EXCLUDED"."""
    if verdict not in ("INCLUDED", "EXCLUDED"):
        raise ValueError(f"verdict must be INCLUDED or EXCLUDED, got {verdict!r}")
    return {
        "config_hash": config_hash,
        "data_provenance_sha": data_provenance_sha,
        "verdict": verdict,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ledger_tip_hash": ledger_tip_hash,
        "ts": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --------------------------------------------------------------------- CLI
def _hex(b):
    return b.hex()


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: attestation.py [keygen|sign-tip <privkey_hex>|"
             "verify-tip <sig_hex> <pubkey_hex> <ledger_tip_hash> <ts>]")
        return 0
    cmd = argv[0]

    if cmd == "keygen":
        priv, pub = generate_keypair()
        print(json.dumps({"privkey_hex": _hex(priv), "pubkey_hex": _hex(pub)}))
        return 0

    if cmd == "sign-tip":
        privkey_hex = argv[1] if len(argv) > 1 else os.environ.get("ATTEST_PRIVKEY_HEX")
        if not privkey_hex:
            print("need privkey hex (arg or ATTEST_PRIVKEY_HEX env)")
            return 1
        receipt = {
            "ledger_tip_hash": receipt_ledger._last_hash(),
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        sig = sign_receipt(receipt, bytes.fromhex(privkey_hex))
        print(json.dumps({"receipt": receipt, "sig_hex": _hex(sig)}))
        return 0

    if cmd == "verify-tip":
        if len(argv) < 5:
            print("usage: attestation.py verify-tip <sig_hex> <pubkey_hex> "
                 "<ledger_tip_hash> <ts>")
            return 1
        sig_hex, pubkey_hex, tip_hash, ts = argv[1], argv[2], argv[3], argv[4]
        receipt = {"ledger_tip_hash": tip_hash, "ts": ts}
        ok = verify_receipt(receipt, bytes.fromhex(sig_hex), bytes.fromhex(pubkey_hex))
        print("VALID" if ok else "INVALID")
        return 0 if ok else 1

    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
