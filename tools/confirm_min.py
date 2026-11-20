"""TIGHTER CONFIRMATION (20 Nov 2026; tools/confirm_min.py) - offline, distant set, the package's own writes.
kra14 on the GPU consolidates its confirmed real-text inferences (taught 0.13 vs held back 0.03 open at visit 8) but confirmation passes
pairs right 0.72-0.76, so wrong ones are consolidated too (taught wrong recalled 0.08 vs 0.00-0.01). Here the package's code writes
(premises._rcode_step as kra14: 512 dims, the seeded reader's top half, kra14's write rule, CF_W writes from the documents read so far,
CF_FRAC) are confirmed on the documents read AFTER (the rest of the set, in CF_STEPS slices, as later visits) under rules (inf_rti_conf_min
and variants): >= n relational records of the pair (n 1 / 2 / 3), records from >= 2 different documents, and a relational record the seeded
reader puts in its top half. For each rule: how many writes it confirms by each slice, how many of them are right (distant labels), and how
many RIGHT writes it keeps (recall of the right ones).
Usage: python3 tools/confirm_min.py <distant_50k.json.gz>   env CF_FRAC (0.4), CF_W (2400), CF_STEPS (3)"""
import os, sys, types, time, collections
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time()
FR_ = float(os.environ.get('CF_FRAC', 0.4)); NW = int(os.environ.get('CF_W', 2400)); NS = int(os.environ.get('CF_STEPS', 3))
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa', inf_rcode_asym=1.0,
                            inf_rcode_selreport=0, inf_rti_max=NW, inf_rret=0, inf_rti_ctl=0.0, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rcode_reader=0.5, inf_premises_seed_ctx=1)
pm._load_distant(cfg); ND = pm._PM['ND']; CX = pm._PM.get('CTX') or {}; R = pm._PM['R']; n0 = int(round(ND * FR_))
y = np.array([bool(r[6]) for r in R]); hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(int(r[5]), 12) // 3} " + ' '.join('C_' + w for w in CX.get(id(r), '').split()) for r in R])
iRD = np.array([i for i, r in enumerate(R) if r[0] < n0]); sd = iRD[np.random.RandomState(4 + 707).permutation(len(iRD))[:20000]]
c = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); c.fit(X[sd], y[sd]); score = c.decision_function(X); del X
RD = [R[i] for i in iRD]; pm._PM['rdscore'] = score[iRD]
pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], RD, 3)
Wr = {frozenset(k): bool(w['truth']) for k, w in pm._PM['rti'].items()}
thr = float(np.median(score[iRD]))
P(f"{len(RD)} records read ({n0} of {ND} documents); {len(Wr)} written (right {np.mean(list(Wr.values())):.3f}); reader top-half threshold {thr:.3f}  ({time.time() - T0:.0f}s)")
ev = collections.defaultdict(list)                                           # every later RELATIONAL record of a written pair: (document, reader score)
for i, r in enumerate(R):
    if r[0] >= n0:
        fk = frozenset((r[1], r[2]))
        if fk in Wr and 1 <= r[5] <= 8 and not pm._rret_listlike(r): ev[fk].append((r[0], float(score[i])))
RULES = {'>= 1 record (kra14)': lambda L: len(L) >= 1, '>= 2 records': lambda L: len(L) >= 2, '>= 3 records': lambda L: len(L) >= 3,
         '>= 2 documents': lambda L: len({d for d, _ in L}) >= 2, '1 record the reader rates (top half)': lambda L: any(s >= thr for _, s in L),
         '2 rated records': lambda L: sum(s >= thr for _, s in L) >= 2}
nR = sum(Wr.values())
for k in range(1, NS + 1):
    cut = n0 + (ND - n0) * k // NS
    P(f"READ UP TO {cut} documents ({cut / ND:.0%}):")
    for nm, f in RULES.items():
        cf = [Wr[p] for p in Wr if f([e for e in ev.get(p, []) if e[0] < cut])]
        P(f"   {nm:38s} confirmed {len(cf):5d} ({len(cf) / len(Wr):.2f} of writes) | right {np.mean(cf) if cf else float('nan'):.3f} | right ones kept {sum(cf) / max(nR, 1):.2f}")
P(f"DONE ({time.time() - T0:.0f}s)")
