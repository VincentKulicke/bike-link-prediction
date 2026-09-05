#!/usr/bin/env bash
# Supervisor for the final HPO run.
#
# This laptop GPU throws the occasional transient cudaErrorIllegalInstruction
# under sustained load, and such a fault poisons the CUDA context for the rest
# of the process. A fresh process is the only reliable recovery, so we simply
# restart until the script reports it is finished. hpo_final.py appends every
# finished config and skips it on restart, so nothing is recomputed and nothing
# is lost.
set -u
PY="C:/Users/user/anaconda3/python.exe"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
LOG="${1:-/tmp/hpo_final.log}"
MAX_RESTARTS=40

export KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO" || exit 1

for (( i=1; i<=MAX_RESTARTS; i++ )); do
  echo "=== Versuch $i/$MAX_RESTARTS  ($(date '+%H:%M:%S')) ===" >> "$LOG"
  "$PY" -u ablation/hpo_final.py --model all >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== sauber beendet nach $i Versuch(en) ===" >> "$LOG"
    exit 0
  fi
  echo "=== Abbruch mit Code $rc, Neustart in 20 s ===" >> "$LOG"
  sleep 20          # let the driver settle before touching the GPU again
done

echo "=== $MAX_RESTARTS Versuche erschoepft, gebe auf ===" >> "$LOG"
exit 1
