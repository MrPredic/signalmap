# PRIO2 Attestation Spec — Ed25519-signed premium-verdict receipts

Status: core implemented (`attestation.py`, TDD, 9/9 green). Wiring into the
distill pipeline's normal output path is a documented next step, not done
here (see "Next step" below) — this session ships the cryptographic core +
spec, not the full pipeline integration.

## Why (market wedge)

The competitive scan (`signalmap_competitive_landscape.md`) flagged
cryptographic attestation of a model/detector's provenance as an
**uncontested** wedge in the PdM/TinyML space — everyone competes on
detection accuracy, nobody ships a verifiable paper trail. The EU AI Act
Art. 15 ("transparency and provision of information to deployers") and
similar audit regimes increasingly require exactly this: a deployer must be
able to demonstrate, to a regulator or auditor who does **not** trust
SignalMap's infrastructure, that a specific detector/premium-claim traces to
a specific, reproducible experiment. A signed receipt is the minimal
mechanism that satisfies "verifiable without trust."

## WHAT gets signed

A canonical receipt (Python dict, JSON-serializable) binding exactly these
fields — see `attestation.build_receipt()`:

| field                 | meaning                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| `config_hash`          | SHA-256 of the exact experiment config (spec/params) that produced the verdict |
| `data_provenance_sha`  | SHA-256 identifying the dataset version used (e.g. CWRU mirror SHA, MIMII release) |
| `verdict`              | `"INCLUDED"` or `"EXCLUDED"` — the premium-family readout call |
| `ci_low` / `ci_high`   | the paired confidence interval bounds backing the verdict |
| `ledger_tip_hash`      | the `receipt_ledger.py` hash-chain tip at signing time — binds this receipt to the FULL prior audit trail, not just this one result |
| `ts`                   | UTC timestamp of signing |

Deliberately NOT included: raw data, model weights, anything proprietary —
the receipt is a compact provenance/verdict binding, not a data export.

## Canonical serialization (why signing is deterministic)

`attestation.canonical_json(d)` = `json.dumps(d, sort_keys=True,
separators=(",", ":")).encode("utf-8")` — byte-identical regardless of dict
construction order or nesting, so signing/verifying is reproducible across
processes and after a JSON round-trip (store to disk, reload, still
verifies). This is the exact convention `receipt_ledger._canon()` already
uses for the hash-chain, kept consistent on purpose — one canonicalization
rule for the whole factory, not two.

## Key management

- **Offline private key.** Generated once via `attestation.generate_keypair()`
  (or `attestation.py keygen` CLI), 32 raw bytes, Ed25519. Never touches
  network code in this repo. Store it outside the repo (e.g. in a local
  password manager or an air-gapped file) — `attestation.py` never reads or
  writes it to disk itself; every entry point takes it as an explicit
  argument/hex string, so there is no default key file to accidentally
  commit.
- **Distributable public key.** 32 raw bytes, safe to publish anywhere
  (README, a `pubkey.txt` in the repo, a website) — anyone who has it can
  verify any receipt claiming to be signed by us, fully offline.
- **Rotation**: not built (out of scope for a single-operator research repo
  today); the receipt format has room for a `key_id` field if/when rotation
  is needed — noted as a future extension, not built to avoid speculative
  complexity.

## Offline verify flow

```
receipt = {...}                     # exactly what was signed, reconstructed by the verifier
sig_bytes = bytes.fromhex(sig_hex)  # published alongside the receipt
pub_bytes = bytes.fromhex(pubkey_hex)  # published once, out of band
attestation.verify_receipt(receipt, sig_bytes, pub_bytes)  # -> True/False, no network call
```

A verifier needs only: the receipt (public), the signature (public), and
our published pubkey (public). No API call, no trust in SignalMap's servers
or continued existence — this is the whole point versus a "trust our
dashboard" attestation.

## How this extends `receipt_ledger.py`

`receipt_ledger.py` already gives a **tamper-evident, chained** log (every
entry commits to the previous via SHA-256; `verify()` walks the whole chain).
What it does NOT give: non-repudiation to a third party — anyone with
filesystem access could, in principle, regenerate a self-consistent chain
from scratch. Signing the chain **tip** with an offline Ed25519 key adds
exactly the missing property: a third party who only trusts the published
pubkey (not our disk, not our process) can verify that a specific,
particular tip hash was attested to at a specific time by the holder of the
private key. `attestation.py`'s CLI already exercises this today:
```
attestation.py sign-tip <privkey_hex>            # signs {ledger_tip_hash, ts}
attestation.py verify-tip <sig_hex> <pubkey_hex> <ledger_tip_hash> <ts>
```
Per-verdict receipts (`build_receipt(...)`) are the finer-grained sibling of
this: one signed receipt per premium-family call (CWRU coherence, envelope,
etc.), each referencing the ledger tip at the time it was made, so a single
signed artifact both proves the specific verdict AND anchors it into the
full audit history.

## Future step: hardware TEE / TPM extension (cited, not built)

The natural hardening beyond "offline software private key" is binding the
signing operation to a hardware root of trust — a TPM 2.0 (sign inside the
TPM, key never leaves it) or a TEE (SGX/SEV/ARM TrustZone) enclave, so even
compromise of the host OS cannot exfiltrate the signing key or forge
receipts for computations that didn't actually run inside the attested
environment. This would upgrade the claim from "the holder of this key
attests X" to "a specific, remotely-attestable hardware environment produced
X" — the stronger form regulators/auditors increasingly ask for in
high-assurance deployments. Not built in this session (no TEE/TPM hardware
target chosen yet); cited here as the documented next hardening step once
there's a concrete deployment target that has one.

## Dependency note

`cryptography` (Ed25519 support, PyCA) was already present in
`.venv-research` transitively but was **not** a declared project dependency
in `pyproject.toml`/`requirements.txt`. Added as `cryptography>=42` under a
new `attest` extra so `attestation.py` doesn't silently rely on an
undeclared transitive package.
