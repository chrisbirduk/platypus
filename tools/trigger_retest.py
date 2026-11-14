"""WHY DID THE RETRACTION TRIGGER FAIL ON THE GPU? (14 Nov 2026; tools/trigger_retest.py) - offline, the package's own code path.
GPU cg27a_rret visit 4: the confirmation score (short-gap records + document PMI) AUC 0.517 on the 200 visit-2 writes, flagged wrong 0.38
vs 0.42; offline (tools/confirm_trigger.py, Re-DocRED writes, human labels) AUC 0.72-0.73. Two things differ: the TRUTH (distant Wikidata
labels vs Re-DocRED human labels) and the WRITES (pairs from held-out distant documents vs held-out Re-DocRED documents).
Part 1 reruns the GPU path exactly (platypus.inference.premises: _load_distant, _rcode_step at visit 2 with 2/10 of the distant set,
code + resource allocation, 512 dims, seed 4) and judges the writes on the next 2/10 - each signal's AUC against the distant labels, and
against what the store reads (any later sentence with both names; within 3 words).
Part 2 takes the Re-DocRED writes of tools/fact_compress.py FC_SAVE (human labels) and scores them against the DISTANT labels too: the same
writes, two truths.
Usage: python3 tools/trigger_retest.py <train_distant.json.gz> <ret_s0.json> [<ret_s1.json>]"""
import sys, json, types, math, collections, time
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.compress import _cg_auc
P = lambda *a: print(*a, flush=True); T0 = time.time()
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+ra',
                            inf_rcode_selreport=0, inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; LAB = pm._PM['LAB']
n1 = int(round(ND * 0.2)); n2 = int(round(ND * 0.4))
P(f"distant set: {ND} documents, {len(R)} records, {len(LAB)} labelled pairs; write at {n1} documents, judge on {n1}-{n2}  ({time.time() - T0:.0f}s)")
pm._PM['rti'] = {}
pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < n1], 1)
W = pm._PM['rti']; ks = list(W); y = np.array([bool(W[k]['truth']) for k in ks])
P(f"PART 1 (the GPU path): {len(ks)} written, labelled right {y.mean():.3f}  ({time.time() - T0:.0f}s)")
want = {frozenset(k) for k in ks}; nm = {n for k in ks for n in k}; df = collections.Counter(); dn = set(); byp = collections.defaultdict(lambda: [set(), 0, 0])
for r in R:
    if not (n1 <= r[0] < n2): continue
    for n in (r[1], r[2]):
        if n in nm and (r[0], n) not in dn: dn.add((r[0], n)); df[n] += 1
    fk = frozenset((r[1], r[2]))
    if fk in want: b = byp[fk]; b[0].add(r[0]); b[1] += int(r[5] <= 3); b[2] += 1
nd = n2 - n1
g = lambda k, i: (byp[frozenset(k)][i] if frozenset(k) in byp else (set() if i == 0 else 0))
docs = np.array([len(g(k, 0)) for k in ks], float); short = np.array([g(k, 1) for k in ks], float); recs = np.array([g(k, 2) for k in ks], float)
pmi = np.array([math.log(len(g(k, 0)) + 0.5) + math.log(nd) - math.log(df[k[0]] + 1) - math.log(df[k[1]] + 1) for k in ks])
pop = -np.array([math.log((df[k[0]] + 1) * (df[k[1]] + 1)) for k in ks]); rk = lambda x: np.argsort(np.argsort(x + 1e-9 * np.random.RandomState(7).rand(len(x)))) / (len(x) - 1)
sc = rk(short) + rk(pmi); code = np.array([W[k]['code'] for k in ks]); br = np.array([W[k]['bridges'] for k in ks], float)
P(f"   later reading: pairs stated again {np.mean(docs > 0):.2f} (right {np.mean(docs[y] > 0):.2f} / wrong {np.mean(docs[~y] > 0):.2f}); short-gap {np.mean(short > 0):.2f}")
for name, v in (('documents stating the pair', docs), ('records', recs), ('short-gap records', short), ('document PMI', pmi), ('short + PMI (the trigger)', sc),
                ('popularity of the names', pop), ('the write-time code', code), ('bridges at write', br)):
    f = rk(v) <= 0.2
    P(f"   {name:28s} AUC vs DISTANT labels {_cg_auc(v[y], v[~y]):.3f} | bottom 20% wrong {1 - y[f].mean():.2f} (overall {1 - y.mean():.2f})")
P("PART 2: the Re-DocRED writes (human labels) scored against the distant labels as well")
for path in sys.argv[2:]:
    Wr = json.load(open(path)); yh = np.array([w[2] for w in Wr], bool); yd = np.array([frozenset((w[0], w[1])) in LAB for w in Wr], bool)
    P(f"   {path.split('/')[-1]}: human-labelled right {yh.mean():.2f}, distant-labelled right {yd.mean():.2f}, agreement {np.mean(yh == yd):.2f} "
      f"(human right & distant right {np.mean(yh & yd):.2f}, human right only {np.mean(yh & ~yd):.2f}, distant right only {np.mean(~yh & yd):.2f})")
P(f"DONE ({time.time() - T0:.0f}s)")
