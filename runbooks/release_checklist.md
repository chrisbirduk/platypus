# Release checklist: from a change to a GPU build

1. **Lab change.** Make the change in the lab script behind an option,
   off by default, with a readout that measures it.
2. **Local test.** Run it on the five-domain harness against the current
   baseline (`scripts/run_local.sh`). Seed noise is about 0.017 BPC, so
   judge by the targeted readout, and by final BPC only for harm.
   Stop early on a clear failure.
3. **Decide.** Keep it if the readout moves the right way and final BPC
   is within noise of the baseline. Record the decision and numbers in the
   decisions log.
4. **Build.** Produce the GPU build from the base with the builder, adding
   the option to the preset. Check:
   - `make_release.sh BUILD README COMMIT` passes (compiles, no apostrophe
     in NAME strings, secret scan clean);
   - the switch line printed at start-up names the new option.
5. **Local safety run of the full build.** Forgetting and final BPC within
   noise of the previous build's safety run, no new errors.
6. **Paperwork.**
   - README: results table, options, log lines, history.
   - Commit log: changes since the last commit, local results, GPU results.
   - CLS tracker: rows whose evidence changed, and the "if pending
     succeeds" figures.
   - `manifests/builds.csv`: one line.
   - `configs/presets/<build>.env` for the new build (and any new control).
   - The copies in the Platypus project on claude.ai: the repository root
     files are kept there under `deploy/...` (DEPLOY.md as `deploy/README.md`).
     Rewrite every file that changed and run the secret scan first, or sync
     the project from this repository instead.
7. **GPU.** `gpu_run.md`.
8. **Controls.** For a claim about dreams, the same build with
   `controls/random_dreams.env` (and `noreplay.env` where relevant).
   Controls are never defaults.
