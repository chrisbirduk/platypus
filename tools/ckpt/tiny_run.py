"""Tiny CPU run of the real entry point (python -m platypus), for checkpoint and resume tests.

usage: python tools/ckpt/tiny_run.py
  LAB_OVR and the PLATYPUS_* switches are read as in a GPU run; TINY_OVR (JSON) is applied on top
  of the tiny sizes below. PLATYPUS_CKPT names the checkpoint folder (see platypus/checkpoint.py).

The Pile is replaced by local text: each domain reads its own slice of the Python standard library's
source files, so no network and no Hugging Face download is needed.
"""
import json, os, sys, glob, hashlib, runpy, sysconfig

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

TINY = {
    "d_model": 64, "d_state": 64, "n_layers": 2, "seq_len": 64, "batch_size": 4, "steps_per_task": 6,
    "pass_shard_bytes": 120_000, "extra_domains": [], "dream_bank": 64, "dream_batch": 4,
}

_FILES = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "**", "*.py"), recursive=True))


def local_bytes(domain, seed, n):
    """deterministic stand-in text for (domain, seed, n): stdlib files chosen by a hash of the domain and seed"""
    h = int(hashlib.sha256(f"{domain}|{seed}".encode()).hexdigest(), 16)
    k = len(_FILES)
    buf = bytearray(); i = 0
    while len(buf) < n:
        fp = _FILES[(h + i * 7919) % k]; i += 1
        try: buf.extend(open(fp, "rb").read())
        except OSError: continue
        buf.extend(b"\n")
    return bytes(buf[:n])


def main():
    ovr = json.loads(os.environ.get("LAB_OVR", "{}") or "{}")
    ovr = {**TINY, **ovr, **json.loads(os.environ.get("TINY_OVR", "{}") or "{}")}
    os.environ["LAB_OVR"] = json.dumps(ovr)
    import torch; torch.set_num_threads(int(os.environ.get("TINY_THREADS", "2")))
    import platypus
    platypus.data.set_stream_source(local_bytes)
    runpy.run_module("platypus.__main__", run_name="__main__")


if __name__ == "__main__":
    main()
