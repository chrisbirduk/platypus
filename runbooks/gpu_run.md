# GPU run

From a new build to its results.

## Before the run

- The build has passed the local safety run (`release_checklist.md`).
- The build file, README and commit log are in the release folder and the
  secret scan is clean.
- Know the reference to compare against: the previous build's pass-1 mean
  BPC and forgetting, weights-only two-hop by visit, and old-domain BPC.

## Start

Colab: `colab/platypus_gpu.ipynb`, first cell set to the build and preset.
Cloud machine: `docker run ... <build> <preset> 1` (see `docker/Dockerfile`).

A three-pass run takes four to five hours on one 95 GB-class GPU; each
sleep adds two to three minutes.

## Stop early on a clear failure

Stop the run and report as soon as any of these appear:

- a Traceback, an out-of-memory error, or the process killed;
- a routing line marked WRONG in pass 1;
- pass-1 BPC on the visited domain far above the previous build at the
  same visit (more than about 0.05);
- the brake rolling back the sleep at every visit from visit 5 on;
- with-store two-hop falling as visits go on in pass 1.

A single rolled-back closed-form edit early in a run is normal (cg26m:
visits 3 and 4); repeated rollbacks later are not.

## What to read while it runs

`python3 scripts/extract_readouts.py <run>/run.log --md` at any
time. The lines that matter most, in order:

1. `PASS 1 SUMMARY`: mean BPC and forgetting, and the per-domain
   post-train BPC (compare the mean with the previous build).
2. `[inference eval | visit N]`: recall from the weights alone.
3. `[inference eval | visit N | WITH THE FAST STORE]`: recall with the store.
4. `CLOSED-FORM CONSOLIDATION APPLIED`: kept or rolled back, cost, matrix
   size, persistence of earlier edits.
5. `CLS | RELEASE`, `CLS | SCHEMA SPEED`, `DECAY MODEL`.
6. At the end: `temporal gradient | END OF RUN`, `MEAN FORGETTING`.

## After the run

1. `extract_readouts.py <new log> <previous log> --md` for the comparison.
2. Scan anything you send: `scan_secrets.py <files>`.
3. Update the CLS tracker, the README results table and the commit log
   (`release_checklist.md`, step 6).
4. Add a line to `manifests/builds.csv`.
