# Platypus deployment

How to run the Platypus build (the `platypus` package) somewhere other than a laptop:
a GPU run on Colab or a cloud machine, the local CPU harness, the
checks a build passes before it reaches the GPU, and the record kept of
each run.

Copyright Christian Bird. Apache 2.0.

## Layout

```
(repository root)
  platypus/           the current build (cg26p on this branch), run with python -m platypus
  tools/refactor/     how the package was split from the single file, and the identity checks
  README.md           the project README
  DEPLOY.md           this file
  configs/            what a run is: switches and option overrides
    presets/          one file per build preset (cg26m, cg26n, cg26o, cg26p ...)
    controls/         the control runs (random dreams, no replay, edits off)
    local/            the five-domain CPU harness
  env/                Python requirements and the names of environment variables
  colab/              the Colab notebook used for GPU runs
  docker/             a container for a cloud GPU machine
  scripts/            run, summarise, scan and release
  ci/github/          the GitHub Actions workflow (move to .github/workflows/ to switch it on)
  runbooks/           step-by-step procedures: GPU run, release, secrets
  manifests/          the build history and the record written for each run
  fixtures/           short excerpts of real logs, used by the tests
  tests/              tests for the scripts
```

## How a run is described

A run is fully described by three things, and the run script records
all three:

1. **The build**: the `platypus` package (or a single-file build),
   identified by its SHA-256 (for the package: every `.py` file, by path).
2. **A preset**: `configs/presets/<build>.env` holds the switches the build
   reads from the environment (`PLATYPUS_CF`, `PLATYPUS_DREAMS`, ...) and
   may name a JSON file of option overrides (`LAB_OVR`).
3. **A seed** (the GPU runs so far all use seed 1).

`scripts/run_gpu.sh` writes these to `manifest.json` in the run folder
beside the log, so any result can be traced to exactly what produced it.

## Quick start

GPU, on Colab: open `colab/platypus_gpu.ipynb`, set the preset in
the first cell, run all. The log and manifest are saved to Drive.

GPU, on a cloud machine with Docker:

```
docker build -f docker/Dockerfile -t platypus:cg26p .
docker run --gpus all -v $PWD/runs:/work/runs -v $PWD/pile_cache:/work/pile_cache \
    platypus:cg26p platypus configs/presets/cg26p.env 1
```

## Checkpoints and resume

Checkpoints are **off by default**. With `PLATYPUS_CKPT=1`, `scripts/run_gpu.sh`
writes a checkpoint to `<run folder>/ckpt/` after every visit
(`platypus/checkpoint.py`); `PLATYPUS_CKPT=<folder>` writes it elsewhere. In the
Colab notebooks set `CKPT = True` in the first cell. If a run stops (a Colab session ends, a spot
machine is taken back), continue it with

```
scripts/run_gpu.sh --resume runs/<run>
```

The run goes on after the last saved visit and appends to the same `run.log`;
the manifest records each resume. A resumed run computes what the
uninterrupted run computes: `python tools/ckpt/equiv.py OUT 3 6` stops a small
CPU run after visits 3 and 6, resumes it, and compares the logs line for line.

- A checkpoint is refused if the code, the options, Python or torch differ
  from the run that wrote it.
- The checkpoints are deleted when the run finishes cleanly
  (`PLATYPUS_CKPT_KEEP=1` keeps them). `run_gpu.sh --resume` keeps writing
  them for the resumed run.
- Two checkpoints are kept (`ckpt.pt` and `ckpt.prev.pt`); each line
  `[checkpoint] saved after visit N` in the log gives the size and the time.
  A checkpoint holds the whole run state (weights, optimiser, the initial
  weights, the stores and inference state), so with the run folder on Google
  Drive expect two copies of about 9 GB each (kra57 on the GPU, 25 Nov:
  8,977-9,099 MB, 76 s a save); `PLATYPUS_CKPT_EVERY=2` halves the writes.
  Saving works on the GPU with CUDA graphs on; resuming is not yet tried there.
- `tools/ckpt/equiv.py` runs the build's defaults plus the preset's
  `LAB_OVR_FILE` (before 25 Nov `tools/ckpt/tiny_run.py` replaced the defaults
  with the tiny sizes alone, so earlier equivalence checks ran without the
  default options or the preset's overrides).
- A new local in `run()` that a later visit reads must be added to
  `CK_LOCALS` in `platypus/runner.py`; run `tools/ckpt/equiv.py` after changes
  to the run loop.

Summarise a log (the readouts that decide whether a build is kept):

```
python3 scripts/extract_readouts.py runs/<run>/run.log --md
```

Before anything is shared or pushed:

```
python3 scripts/scan_secrets.py <files or folders>
```

## Where this fits in the refactor

The refactor plan (snapshot branch, code map, package split with
`state.py`, equivalence check, config and presets with a CHANGELOG, CI,
docs) needs somewhere for runs, presets and checks to live. These files are
that place, and they work with the current single-file build now:

| Refactor step | What here supports it |
|---|---|
| 0 Snapshot branch | `manifests/builds.csv` records the freeze point (suggested cg26m) |
| 3 Package split | done (5 November): `platypus/`; `scripts/run_gpu.sh platypus ...` runs `python -m platypus` |
| 4 Equivalence check | `scripts/extract_readouts.py --json` gives the numbers to compare old and new code on the same seed |
| 5 Config and presets | `configs/` is the start of it; the presets move into the package's config later |
| 6 CI | `ci/github/workflows/ci.yml` |
| 7 Docs | `runbooks/` |

## Rules

- No secret goes in any file here. Keys live in Colab Secrets or the
  machine's secret store; `env/.env.example` lists names only.
- Every log, archive and output is scanned with `scan_secrets.py` before
  it is sent or pushed.
- A build reaches the GPU only after the local safety run
  (`runbooks/release_checklist.md`).
- Random dreams and no replay are controls, never defaults.
