# Secrets

- API keys and tokens live only in **Colab Secrets** (or the cloud
  machine's secret store). They are read at run time with
  `google.colab.userdata.get(...)` or passed as `docker run -e NAME=...`.
- A key is never pasted into a chat, a notebook cell's text, a log, a
  commit, an env file in the repository, or a Docker image.
- `env/.env.example` lists the names of variables only.
- Every log, archive and output is scanned before it is sent or pushed:

  ```
  python3 scripts/scan_secrets.py <files or folders>
  ```

  `run_gpu.sh` and `make_release.sh` run the scan themselves and stop if it
  finds anything. CI scans the whole repository on every push.
- If a key is ever exposed: revoke it at the provider first, then remove
  it from wherever it appeared (rewriting git history if it was pushed),
  then issue a new one into Colab Secrets.
