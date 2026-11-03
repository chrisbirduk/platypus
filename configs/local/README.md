# Local CPU harness

The five-domain harness tests options before they reach the GPU: five
domains, two passes, a 2-block model of width 512, 128-byte windows and
40 probes. Seed noise is about 0.017 BPC, so local results are read
through the targeted readouts, not final BPC alone.

Today it lives outside the repository (the lab script `threshold_lab.py`,
the checkpointing driver `v97_ckpt.py`, a base override file and job
files with one line per run: `name<TAB>{json overrides}`).

To move here in the refactor:

- `base.json`: the harness's base overrides (`base_ovr.json`).
- `jobs/<topic>.tsv`: job files, one run per line.
- `scripts/run_local.sh` already runs a job file in up to two lanes with
  resume from checkpoints.

Reference results (mean forgetting / final BPC):

| Run | Result |
|---|---|
| cg26m, edits off | -0.012 / 2.073 |
| cg26m, edits on (uncapped) | -0.002 / 2.104 |
| cg26m, edits on + growth cap 1.5 | -0.008 / 2.079 |

Machine limits: 2 CPUs and 8 GB, so at most two runs at once and no
analysis beside two running jobs.
