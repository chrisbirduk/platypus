# Fixtures

Short excerpts of real logs, used by `tests/test_scripts.py`:

- `gpu_log_cg26m_excerpt.txt`: the readout lines of the cg26m GPU run to
  visit 11 (seed 1), taken from the full log with grep.
- `local_log_tail_excerpt.txt`: the end-of-run lines of the local growth-cap
  run (q_cfcap).

When a build changes a log line's wording, add an excerpt from the new
build and a test, rather than editing these.
