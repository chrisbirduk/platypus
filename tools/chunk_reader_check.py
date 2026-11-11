"""Unit check of inf_reader_chunk (the fast sequence channel inside the store): local Wikipedia and the benchmark documents are counted
window by window as the store does, the benchmark sentences are read into S_['relc'], and the two readouts run ([READER vs GOLD] and
the relational module on real text). Should come near tools/chunk_reader.py (325 gold relations). Usage: python3 tools/chunk_reader_check.py"""
import os
import sys
import time
import types
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)
import platypus.config as C  # noqa: E402
from platypus.inference import units, probes, distill  # noqa: E402
cfg = C.Config(); cfg.inf_bench = 'embedded'; cfg.inf_reader_chunk = 1; cfg.inf_distill_relmod = 800; cfg.seed = 0
G = probes._bench(cfg); docs = [d.encode('utf-8', 'replace') for k in sorted(G['docs']) for d in G['docs'][k]]
raw = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin', 'rb').read().replace(b'\n', b' ')
S_ = {'it': []}; t0 = time.time()
stream = raw + b' ' + b' '.join(docs)
for i in range(0, len(stream), 1024): units._chk_count(cfg, S_, stream[i:i + 1024])
print(f"counted {S_['chk']['nwin']} windows in {time.time() - t0:.0f}s | frame words {len(S_['chk']['frame'])}, word types {len(S_['chk']['uni'])}", flush=True)
RC = S_.setdefault('relc', {}); ns = 0
for d in docs:
    for s in units._INF_SENT.split(d):
        S_['it'].append({'s': s}); i_ = len(S_['it']) - 1; ns += 1
        for a, r, b in units._chk_records(cfg, S_, s): RC.setdefault((a, r), {}).setdefault(b, []).append(i_)
print(f"read {ns} benchmark sentences -> {len(RC)} record keys", flush=True)
probes._reader_gold_readout(S_, cfg, 0)
model = types.SimpleNamespace(); distill._dw_relmod_real(model, cfg, S_, 0)
