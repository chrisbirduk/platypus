#!/usr/bin/env bash
# Run one Platypus build on a GPU and keep a complete record of the run.
#
#   run_gpu.sh BUILD PRESET.env [SEED] [CONTROL.env ...]
#   run_gpu.sh --resume RUN_DIR      continue a stopped run from its last checkpoint
#
# BUILD is the platypus package directory (run as python3 -m platypus) or a
# single-file build (an older build, or scripts/export_single_file.py output).
#
# Writes runs/<build>_<preset>[_<control>]_s<seed>_<UTC time>/ with
#   run.log        everything the build prints
#   manifest.json  build SHA-256, preset and control files, every switch,
#                  the merged LAB_OVR, GPU, git commit, start and end times
#   readouts.md    the summary from extract_readouts.py
#   ckpt/          a checkpoint after every visit (platypus/checkpoint.py), deleted
#                  when the run finishes cleanly (PLATYPUS_CKPT_KEEP=1 keeps it;
#                  PLATYPUS_CKPT=0 turns checkpoints off)
# and finishes with a secret scan of the run folder.
#
# --resume reads the build, preset, seed and controls from RUN_DIR/manifest.json,
# appends to run.log and records the resume in the manifest. It refuses to start
# unless the checkout is at the manifest's git commit, and the build refuses a
# checkpoint written by different code or options.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESUME=""
if [ "${1:-}" = "--resume" ]; then
  RESUME="${2:?usage: run_gpu.sh --resume RUN_DIR}"; RESUME="${RESUME%/}"
  [ -f "$RESUME/manifest.json" ] || { echo "no manifest.json in $RESUME" >&2; exit 1; }
  mapfile -t _args < <(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m.get("build_path") or m["build"]); print(m["preset"]); print(m["seed"]); [print(c["file"]) for c in m["controls"]]' "$RESUME/manifest.json")
  set -- "${_args[@]}"
  # The checkpoint only fits the code that wrote it: resume from the commit the run started at.
  want="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("git_commit") or "")' "$RESUME/manifest.json")"
  have="$(git rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$want" ] && [ "$want" != "$have" ] && [ "${PLATYPUS_CKPT_FORCE:-0}" != 1 ]; then
    echo "this run started at commit $want, the checkout is at ${have:-no commit}." >&2
    echo "check out the run's commit first (git fetch origin $want && git checkout --detach $want; the notebook does this when RESUME is set)," >&2
    echo "or set PLATYPUS_CKPT_FORCE=1 to resume with this code anyway" >&2
    exit 1
  fi
fi
BUILD="${1:?usage: run_gpu.sh BUILD PRESET.env [SEED] [CONTROL.env ...]}"; BUILD="${BUILD%/}"
PRESET="${2:?usage: run_gpu.sh BUILD PRESET.env [SEED] [CONTROL.env ...]}"
SEED="${3:-1}"
shift $(( $# >= 3 ? 3 : $# ))
CONTROLS=("$@")

[ -f "$BUILD" ] || [ -f "$BUILD/__main__.py" ] || { echo "no build file or package: $BUILD" >&2; exit 1; }
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
if [ -n "$RESUME" ]; then RUN_DIR="$RESUME"; else RUN_DIR="${RUNS_DIR:-runs}/${tag}_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)"; fi
mkdir -p "$RUN_DIR"
# Checkpoints after every visit, inside the run folder, so a stopped run can be resumed.
if [ "${PLATYPUS_CKPT:-}" != "0" ]; then export PLATYPUS_CKPT="${PLATYPUS_CKPT:-$RUN_DIR/ckpt}"; fi

if [ -n "$RESUME" ]; then
python3 - "$RUN_DIR/manifest.json" "${PLATYPUS_CKPT:-}" <<'PY'
import json, os, sys, datetime
p, ck = sys.argv[1], sys.argv[2]
m = json.load(open(p))
info = json.load(open(os.path.join(ck, "ckpt.json"))) if ck and os.path.exists(os.path.join(ck, "ckpt.json")) else {}
m.setdefault("resumes", []).append({"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                    "after_visit": info.get("visit"), "host": os.uname().nodename})
json.dump(m, open(p, "w"), indent=2)
print(f"resume: {p} (checkpoint after visit {info.get('visit')})" if info else f"resume: {p} (no checkpoint found: the run starts again from the beginning)")
PY
else
python3 - "$BUILD" "$PRESET" "$SEED" "$RUN_DIR/manifest.json" ${CONTROLS[@]+"${CONTROLS[@]}"} <<'PY'
import hashlib, json, os, subprocess, sys, datetime
build, preset, seed, out, *controls = sys.argv[1:]
def sha(p):
    if os.path.isdir(p):                     # a package: every .py file, by relative path, in sorted order
        h = hashlib.sha256()
        for root, dirs, files in sorted(os.walk(p)):
            dirs.sort()
            for f in sorted(files):
                if f.endswith(".py"):
                    fp = os.path.join(root, f); h.update(os.path.relpath(fp, p).encode() + b"\0" + open(fp, "rb").read() + b"\0")
        return h.hexdigest()
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
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
    "build": os.path.basename(build), "build_path": build, "build_sha256": sha(build), "checkpoints": os.environ.get("PLATYPUS_CKPT") or None,
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
fi

echo "run: $RUN_DIR"
start=$(date +%s)
TEE=(tee); [ -n "$RESUME" ] && { TEE=(tee -a); echo "######## resumed $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" >> "$RUN_DIR/run.log"; }
set +e
if [ -d "$BUILD" ]; then
  PYTHONPATH="$(cd "$(dirname "$BUILD")" && pwd)${PYTHONPATH:+:$PYTHONPATH}" ${ENTRY:-python3} -m "$(basename "$BUILD")" 2>&1 | "${TEE[@]}" "$RUN_DIR/run.log"
else
  ${ENTRY:-python3} "$BUILD" 2>&1 | "${TEE[@]}" "$RUN_DIR/run.log"
fi
status=${PIPESTATUS[0]}
set -e

python3 - "$RUN_DIR/manifest.json" "$status" "$start" <<'PY'
import json, sys, time, datetime
p, status, start = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = json.load(open(p))
import re, os
fl_ = None
lg_ = os.path.join(os.path.dirname(p), "run.log")
if os.path.exists(lg_):
    for ln_ in open(lg_, errors="replace"):
        mm_ = re.match(r"FAILURES: (none|\d+)(.*)", ln_)
        if mm_: fl_ = {"count": 0 if mm_.group(1) == "none" else int(mm_.group(1)), "detail": mm_.group(2).strip(" -")[:4000]}
m.update(failures=fl_)   # issue #28: handlers that printed and continued during the run
seg = int(time.time()) - start                       # this process; a resumed run adds one segment per resume
m.setdefault("segments", []).append({"seconds": seg, "exit_status": status})
m.update(exit_status=status, wall_seconds=sum(s["seconds"] for s in m["segments"]),
         finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
json.dump(m, open(p, "w"), indent=2)
PY

if [ "$status" = 0 ] && [ -n "${PLATYPUS_CKPT:-}" ] && [ "${PLATYPUS_CKPT_KEEP:-0}" != 1 ] && [ "$(dirname "$PLATYPUS_CKPT")" = "$RUN_DIR" ]; then
  rm -rf "$PLATYPUS_CKPT"                           # finished cleanly: the checkpoints are no longer needed
fi
python3 "$HERE/extract_readouts.py" "$RUN_DIR/run.log" --md > "$RUN_DIR/readouts.md" || true
python3 "$HERE/scan_secrets.py" "$RUN_DIR" || { echo "SECRET SCAN FAILED - do not share $RUN_DIR" >&2; exit 3; }
echo "done (exit $status): $RUN_DIR"
exit "$status"
