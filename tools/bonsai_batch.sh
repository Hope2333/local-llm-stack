#!/usr/bin/env bash
# Sequentially fetch every manifest entry through strict_fetch.sh, skipping files
# that already exist with the exact size and matching sha256.
set -uo pipefail

MANIFEST=/tmp/bonsai_manifest.tsv
ROOT="$HOME/models/bonsai2"
LOG=/tmp/bonsai_batch.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

total=$(wc -l < "$MANIFEST")
idx=0
while IFS=$'\t' read -r rel size sha url; do
  idx=$((idx + 1))
  out="$ROOT/$rel"
  mkdir -p "$(dirname "$out")"

  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then
    if [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sha" ]; then
      log "[$idx/$total] SKIP $rel (verified)"
      continue
    fi
    log "[$idx/$total] $rel failed sha, deleting"
    rm -f "$out"
  fi

  log "[$idx/$total] FETCH $rel ($size bytes)"
  if bash /tmp/strict_fetch.sh "$url" "$out" "$size" "$sha" >> "$LOG" 2>&1; then
    log "[$idx/$total] DONE  $rel"
  else
    log "[$idx/$total] FAIL  $rel"
  fi
done < "$MANIFEST"

log "BATCH COMPLETE"
du -sh "$ROOT"/* | tee -a "$LOG"
