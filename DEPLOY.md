# Platypus deployment

How to run the Platypus build (`platypus_test.py`) somewhere other than a laptop:
a GPU run on Colab or a cloud machine, the local CPU harness, the
checks a build passes before it reaches the GPU, and the record kept of
each run.

Copyright Christian Bird. Apache 2.0.

## Layout

```
(repository root)
  platypus_test.py    the current build (cg26n on this branch)
  README.md           the project README
  DEPLOY.md           this file
  configs/            what a run is: switches and option overrides
    presets/          one file per build preset (cg26m, cg26n ...)
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

1. **The build file**, e.g. `platypus_test.py`,
   identified by its SHA-256.
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
docker build -f docker/Dockerfile -t platypus:cg26n .
docker run --gpus all -v $PWD/runs:/work/runs -v $PWD/pile_cache:/work/pile_cache \
    platypus:cg26n platypus_test.py configs/presets/cg26n.env 1
```

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
| 3 Package split | `scripts/run_gpu.sh` takes any entry point; switch it to `python -m platypus` when the package exists |
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
