#!/usr/bin/env bash
# STRICT chunked downloader.
# Rules that prevent the corruption seen earlier:
#   1. every range request must return HTTP 206 (a 200 = full file => reject)
#   2. no append: each attempt writes a fresh .tmp, size-checked, then renamed
#   3. final file verified against the authoritative sha256
set -uo pipefail

LOG=/tmp/strict_fetch.log
nchunks=16
par=6

URL="$1"; OUT="$2"; TOTAL="$3"; SHA="$4"
DIR="$(dirname "$OUT")"; BASE="$(basename "$OUT")"
PARTS="$DIR/.strict/$BASE"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

mkdir -p "$PARTS"
CSIZE=$(( (TOTAL + nchunks - 1) / nchunks ))
log "START $BASE total=$TOTAL chunk=$CSIZE x$nchunks par=$par"

pids=()
for ((i=0; i<nchunks; i++)); do
  s=$(( i * CSIZE )); [ "$s" -ge "$TOTAL" ] && break
  e=$(( s + CSIZE - 1 )); [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
  want=$(( e - s + 1 ))
  part="$PARTS/$i"
  if [ -f "$part" ] && [ "$(stat -c %s "$part")" = "$want" ]; then
    log "skip chunk $i (complete)"; continue
  fi
  (
    for try in $(seq 1 20); do
      tmp="$part.$try.tmp"
      rm -f "$tmp"
      code=$(curl -sL --connect-timeout 20 --max-time 1800 --retry 2 --retry-delay 3 \
                  -r "$s-$e" -o "$tmp" -w '%{http_code}' "$URL")
      sz=$(stat -c %s "$tmp" 2>/dev/null || echo 0)
      if [ "$code" = "206" ] && [ "$sz" = "$want" ]; then
        mv -f "$tmp" "$part"; log "chunk $i OK (try $try)"; exit 0
      fi
      log "chunk $i retry: code=$code size=$sz want=$want (try $try)"
      rm -f "$tmp"
      sleep $(( try < 5 ? try*2 : 10 ))
    done
    log "chunk $i FAILED permanently"; exit 1
  ) &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do sleep 1; done
done

rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" -ne 0 ] && { log "ABORT: some chunks failed; rerun to resume"; exit 1; }

log "assembling..."
: > "$OUT.strict.tmp"
for ((i=0; i<nchunks; i++)); do cat "$PARTS/$i" >> "$OUT.strict.tmp"; done
sz=$(stat -c %s "$OUT.strict.tmp")
if [ "$sz" != "$TOTAL" ]; then log "ABORT size $sz != $TOTAL"; rm -f "$OUT.strict.tmp"; exit 1; fi
mv -f "$OUT.strict.tmp" "$OUT"

got=$(sha256sum "$OUT" | cut -d' ' -f1)
if [ "$got" = "$SHA" ]; then
  log "SUCCESS $BASE sha256 verified"
  rm -rf "$PARTS"
else
  log "FAILURE sha256 mismatch got=$got want=$SHA"
  exit 1
fi
