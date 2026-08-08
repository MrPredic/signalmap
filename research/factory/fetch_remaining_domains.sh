#!/usr/bin/env bash
# Fetch the two remaining sign-study domains. Deterministic subsets, declared
# here so a reviewer can repeat the exact download:
#
#   paderborn_kat : healthy K001-K005, plus the first two outer-race (KA*) and
#                   first two inner-race (KI*) damage codes in sorted order.
#   mafaulda      : normal.zip (healthy) + imbalance.zip (one named fault
#                   family), both taken whole.
#
# Both stay well under the prereg's 4 GB per-domain cap (~1.5 GB and ~2.4 GB).
# Resumable (curl -C -), niced, and each file's exit status is logged so a
# partial download is visible rather than silently treated as complete.
set -u
ROOT="${SIGNALMAP_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}/raw_domains"
PB="$ROOT/paderborn"
MF="$ROOT/mafaulda"
mkdir -p "$PB" "$MF"
LOG="$ROOT/fetch.log"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

get() {  # url dest
  local url="$1" dest="$2" name
  name="$(basename "$dest")"
  if [ -s "$dest.done" ]; then log "$name already complete"; return 0; fi
  log "fetching $name"
  if nice -n 19 curl -sSL -C - --max-time 3600 --retry 3 -o "$dest" "$url"; then
    touch "$dest.done"
    log "$name ok ($(du -h "$dest" | cut -f1))"
  else
    log "$name FAILED (exit $?) — left partial for resume"
  fi
}

log "=== paderborn ==="
INDEX="$(nice -n 19 curl -sL --max-time 60 https://groups.uni-paderborn.de/kat/BearingDataCenter/)"
codes="$(echo "$INDEX" | grep -oE '[A-Z]+[0-9]+\.rar' | sort -u)"
healthy="$(echo "$codes" | grep -E '^K00[1-5]\.rar$')"
outer="$(echo "$codes" | grep -E '^KA[0-9]+\.rar$' | head -2)"
inner="$(echo "$codes" | grep -E '^KI[0-9]+\.rar$' | head -2)"
log "healthy: $(echo $healthy) | outer: $(echo $outer) | inner: $(echo $inner)"
for f in $healthy $outer $inner; do
  get "https://groups.uni-paderborn.de/kat/BearingDataCenter/$f" "$PB/$f"
done

log "=== mafaulda ==="
for f in normal.zip imbalance.zip; do
  get "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/$f" "$MF/$f"
done

log "=== done ==="
du -sh "$PB" "$MF" | tee -a "$LOG"
