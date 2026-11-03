#!/usr/bin/env bash
# Run one Platypus build on a GPU and keep a complete record of the run.
#
#   run_gpu.sh BUILD.py PRESET.env [SEED] [CONTROL.env ...]
#
# Writes runs/<build>_<preset>[_<control>]_s<seed>_<UTC time>/ with
#   run.log        everything the build prints
#   manifest.json  build SHA-256, preset and control files, every switch,
#                  the merged LAB_OVR, GPU, git commit, start and end times
#   readouts.md    the summary from extract_readouts.py
# and finishes with a secret scan of the run folder.
#
# Works with the current single-file builds. After the package split, set
# ENTRY="python3 -m platypus" and pass the preset instead of a build file.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${1:?usage: run_gpu.sh BUILD.py PRESET.env [SEED] [CONTROL.env ...]}"
PRESET="${2:?usage: run_gpu.sh BUILD.py PRESET.env [SEED] [CONTROL.env ...]}"
SEED="${3:-1}"
shift $(( $# >= 3 ? 3 : $# ))
CONTROLS=("$@")

[ -f "$BUILD" ] || { echo "no build file: $BUILD" >&2; exit 1; }
[ -f "$PRESET" ] || { echo "no preset: $PRESET" >&2; exit 1; }

# Load the preset, then any controls on top (later files win).
set -a
# shellcheck disable=SC1090
source "$PRESET"
for c in ${CONTROLS[@]+"${CONTROLS[@]}"}; do source "$c"; done
set +a
if [ -n "${PRESET_BUILD:-}" ] && [ "$(basename "$BUILD")" != "$PRESET_BUILD" ]; then
  echo "note: preset $PRESET is written for $PRESET_BUILD, running $(basename "$BUILD")" >&2
fi

# Seed: the builds take it from LAB_OVR ("seed"); the preset may also name an override file.
LAB_OVR="$(python3 "$HERE/merge_overrides.py" "$BUILD" ${LAB_OVR_FILE:+"$LAB_OVR_FILE"})"
LAB_OVR="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["seed"]=int(sys.argv[2]); print(json.dumps(d,separators=(",",":")))' "$LAB_OVR" "$SEED")"
export LAB_OVR

tag="$(basename "$BUILD" .py | sed -E 's/^threshold_lean_stagec9_7//; s/_conduit$//')_$(basename "$PRESET" .env)"
for c in ${CONTROLS[@]+"${CONTROLS[@]}"}; do tag="${tag}_$(basename "$c" .env)"; done
RUN_DIR="${RUNS_DIR:-runs}/${tag}_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"

python3 - "$BUILD" "$PRESET" "$SEED" "$RUN_DIR/manifest.json" ${CONTROLS[@]+"${CONTROLS[@]}"} <<'PY'
import hashlib, json, os, subprocess, sys, datetime
build, preset, seed, out, *controls = sys.argv[1:]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def git(*a):
    try: return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception: return None
def gpu():
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return {"name": p.name, "memory_gb": round(p.total_memory / 2**30, 1), "torch": torch.__version__, "cuda": torch.version.cuda}
        return {"name": None, "torch": torch.__version__}
    except Exception as e:
        return {"error": str(e)}
switches = {k: v for k, v in os.environ.items() if k.startswith("PLATYPUS_") or k in ("PILE_CACHE", "FLA_AUTO_INSTALL")}
m = {
    "build": os.path.basename(build), "build_sha256": sha(build),
    "preset": preset, "preset_sha256": sha(preset),
    "controls": [{"file": c, "sha256": sha(c)} for c in controls],
    "seed": int(seed), "switches": switches, "lab_ovr": json.loads(os.environ["LAB_OVR"]),
    "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
    "gpu": gpu(), "host": os.uname().nodename,
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
json.dump(m, open(out, "w"), indent=2)
print(f"manifest: {out}")
PY

echo "run: $RUN_DIR"
start=$(date +%s)
set +e
${ENTRY:-python3} "$BUILD" 2>&1 | tee "$RUN_DIR/run.log"
status=${PIPESTATUS[0]}
set -e

python3 - "$RUN_DIR/manifest.json" "$status" "$start" <<'PY'
import json, sys, time, datetime
p, status, start = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = json.load(open(p))
m.update(exit_status=status, wall_seconds=int(time.time()) - start,
         finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
json.dump(m, open(p, "w"), indent=2)
PY

python3 "$HERE/extract_readouts.py" "$RUN_DIR/run.log" --md > "$RUN_DIR/readouts.md" || true
python3 "$HERE/scan_secrets.py" "$RUN_DIR" || { echo "SECRET SCAN FAILED - do not share $RUN_DIR" >&2; exit 3; }
echo "done (exit $status): $RUN_DIR"
exit "$status"
