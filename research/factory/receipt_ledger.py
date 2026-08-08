"""Tamper-evident receipt ledger (hash chain, append-only JSONL).

Every gauntlet run appends one entry; each entry commits to the previous one
via SHA-256, so past results cannot be silently edited without breaking the
chain — the local, dependency-free version of what a blockchain would give us.
The `tip` command prints the latest chain hash: anchoring THAT single hash
externally (git tag, e-mail-to-self, or a Solana memo tx) timestamps the whole
history without revealing any private content. Chain stays fully offline;
anchoring is optional and out-of-band by design.

Usage:
  from receipt_ledger import log_receipt; log_receipt("BANK", {...})
  .venv-research/bin/python receipt_ledger.py verify   # walk + check chain
  .venv-research/bin/python receipt_ledger.py tip      # latest hash (anchor this)
"""
import hashlib, json, os, subprocess, sys, time

LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LEDGER.jsonl")
GENESIS = "signalmap-research-genesis"


def _canon(entry):
    return json.dumps(entry, sort_keys=True, separators=(",", ":"))


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=os.path.dirname(LEDGER)).stdout.strip() or "?"
    except Exception:
        return "?"


def _last_hash():
    if not os.path.exists(LEDGER):
        return GENESIS
    with open(LEDGER) as f:
        lines = [ln for ln in f if ln.strip()]
    return json.loads(lines[-1])["hash"] if lines else GENESIS


def log_receipt(bank, payload):
    """Append one chained entry. payload = jsonable dict (gauntlet result)."""
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "bank": bank, "git": _git_head(),
             "payload": json.loads(json.dumps(payload, default=str)),
             "prev": _last_hash()}
    entry["hash"] = hashlib.sha256(_canon(entry).encode()).hexdigest()
    with open(LEDGER, "a") as f:
        f.write(_canon(entry) + "\n")
    return entry["hash"]


def verify():
    if not os.path.exists(LEDGER):
        print("ledger empty"); return True
    prev = GENESIS
    with open(LEDGER) as f:
        for i, ln in enumerate(ln for ln in f if ln.strip()):
            e = json.loads(ln)
            h = e.pop("hash")
            if e.get("prev") != prev:
                print(f"BROKEN CHAIN at entry {i}: prev mismatch"); return False
            if hashlib.sha256(_canon(e).encode()).hexdigest() != h:
                print(f"BROKEN CHAIN at entry {i}: hash mismatch"); return False
            prev = h
    print(f"chain OK, tip = {prev}")
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "verify":
        sys.exit(0 if verify() else 1)
    elif cmd == "tip":
        print(_last_hash())
