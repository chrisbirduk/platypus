#!/usr/bin/env bash
# Run local harness jobs from a job file, in up to two lanes, resuming from
# checkpoints and skipping finished runs.
#
#   run_local.sh JOBS.tsv NAME [NAME ...]
#
# A job file has one run per line: name<TAB>{json overrides}.
# Logs go to runs/<name>.log, checkpoints to runs/<name>.ckpt; a run is
# finished once its log has a line starting "WALL".
#
# The harness machine has 2 CPUs and 8 GB: never more than two lanes, and no
# heavy analysis while two runs are going. Long jobs may be stopped by a time
# limit; running the same command again resumes them.
set -euo pipefail

JOBS="${1:?usage: run_local.sh JOBS.tsv NAME [NAME ...]}"; shift
DRIVER="${DRIVER:-v97_ckpt.py}"          # the checkpointing harness driver
LANES="${LANES:-2}"
mkdir -p runs

run_one() {
  local name="$1" ovr
  grep -q "^WALL" "runs/$name.log" 2>/dev/null && { echo "skip $name (finished)"; return; }
  ovr="$(awk -F'\t' -v n="$name" '$1==n{print $2}' "$JOBS")"
  [ -n "$ovr" ] || { echo "no job named $name in $JOBS" >&2; return 1; }
  [ -f "runs/$name.ckpt" ] || : > "runs/$name.log"
  echo "$(date +%H:%M) start $name"
  OVR="$ovr" CKPT="runs/$name.ckpt" python3 "$DRIVER" >> "runs/$name.log" 2>&1
  echo "$(date +%H:%M) end $name: $(grep -E 'MEAN FORGETTING' "runs/$name.log" | tail -1)"
}

# Split the names round-robin into lanes; each lane runs its names in order.
for ((lane=0; lane<LANES; lane++)); do
  (
    i=0
    for name in "$@"; do
      if (( i % LANES == lane )); then run_one "$name"; fi
      i=$((i+1))
    done
  ) &
  sleep 20   # stagger start-up so the two lanes do not load data at once
done
wait
echo "all done $(date +%H:%M)"
