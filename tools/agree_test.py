"""AGREEMENT TEST (18 Nov 2026; tools/agree_test.py) - offline, distant set, label-free (distant labels for evaluation only). Does a second,
independent vote from the LEARNED RELATION MAPS (tools/latent_confirm.py: label-free coarse-frame maps on name keys) raise the precision of
the package's own inference writer (premises._rcode_step: the compressed fact table, code + Adamic-Adar), which writes pairs never stated
together right 0.6-0.7 of the time at the top?
1. The package's writer on the records read by visit AG_VIS (as tools/adapt_check.py), with its candidates saved (save_cand).
2. Name keys (the stand-in trunk), label-free maps per coarse frame key fitted on the read records (50+ records).
3. For each of the writer's top AG_TOP candidates (A, C): every bridge B (a record A-B and a record B-C in the store), the two maps COMPOSED
   predict C from A (and A from C the other way) - pure latent composition, since predicting C from B would only recall a fitted record: MAP RANK = C's rank among AG_NEG random names of C's type (+ C), best over
   bridges; the vote = rank 1 (or top 3).
4. Precision by distant labels of the writer's top N, of the top N among the map-agreeing candidates, and of the disagreeing ones.
Usage: TG_D=192 TG_CKPT=<model> python3 tools/agree_test.py <distant.json.gz>   env AG_VIS (5) AG_TOP (3000) AG_NEG (50)"""
import os, sys, time, types, collections
import numpy as np
import torch
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(here))
import platypus.inference.premises as pm
T0 = time.time(); P = lambda s: print(s, flush=True); DIST = sys.argv[1]
VIS = int(os.environ.get('AG_VIS', 5)); TOP = int(os.environ.get('AG_TOP', 3000)); NEG = int(os.environ.get('AG_NEG', 50))
cfg = types.SimpleNamespace(inf_premises_path=DIST, inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa', inf_rcode_asym=1.0,
                            inf_rcode_selreport=0, inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rti_typed=0, inf_rret_list2=0)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; LAB = pm._PM['LAB']; TY = pm._PM['TY']
h = int(round(ND * min(1.0, (VIS + 1) / 10))); RD = [r for r in R if r[0] < h]
pm._PM['rti'] = {}; pm._PM['save_cand'] = True
pm._rcode_step(None, cfg, lambda d: 0, [1], RD, VIS)
C = pm._PM['cand']; rule = C['rule']; o = np.argsort(-rule); o = o[np.isfinite(rule[o])][:TOP]
pairs = [C['pairs'][i] for i in o]; y = C['y'][o]
P(f"writer: {len(C['pairs'])} candidates never stated together at visit {VIS + 1} ({h} docs read); its top {len(o)}: labelled {y.mean():.3f}; top 200 {y[:200].mean():.3f}  ({time.time() - T0:.0f}s)")
GL = {'and', 'or', 'the', 'a', 'an', 'with', 'to'}
nb = collections.defaultdict(set); recs = collections.defaultdict(set)
for r in RD:
    if r[5] > 12 or r[1] == r[2]: continue
    nb[r[1]].add(r[2]); nb[r[2]].add(r[1])
    recs[(r[1], r[2])].add((TY.get(r[1], 'X'), r[4], TY.get(r[2], 'X'), 'f')); recs[(r[2], r[1])].add((TY.get(r[2], 'X'), r[4], TY.get(r[1], 'X'), 'r'))
sys.argv = [sys.argv[0], '6000', '5']
src = open(os.path.join(here, 'trunk_guides.py')).read(); g = {}
exec(compile(src[:src.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'), g)
lm, run = g['lm'], g['run']; torch.set_num_threads(2)
names = sorted({x for r in RD for x in (r[1], r[2])}); NI = {n: i for i, n in enumerate(names)}
K = np.zeros((len(names), lm.o.in_features), np.float32)
for i, n in enumerate(names):
    _, H = run(lm, (' ' + n).encode('utf-8', 'replace')); K[i] = H[1:].mean(0)
    if (i + 1) % 50000 == 0: P(f"   name keys {i + 1} of {len(names)}  ({time.time() - T0:.0f}s)")
K -= K.mean(0); K /= np.linalg.norm(K, axis=1, keepdims=True) + 1e-8; D_ = K.shape[1]
XY = collections.defaultdict(list)
for (a, b), ks in recs.items():
    for k in ks: XY[k].append((NI[a], NI[b]))
W = {}
for k, v in XY.items():
    if len(v) < 50: continue
    X = K[[a for a, _ in v]]; Y = K[[b for _, b in v]]; W[k] = np.linalg.solve(X.T @ X + np.eye(D_), X.T @ Y)
byty = collections.defaultdict(list)
for n in names: byty[TY.get(n, 'X')].append(NI[n])
P(f"{len(names)} name keys, {len(W)} label-free maps  ({time.time() - T0:.0f}s)")
rs = np.random.RandomState(0); best = []
for (A, Cn) in pairs:
    rk = 10 ** 9
    for X_, Z_ in ((A, Cn), (Cn, A)):                                          # predict either end from the bridge
        if X_ not in NI or Z_ not in NI: continue
        L = byty[TY.get(Z_, 'X')]; negs = [L[i] for i in rs.randint(0, len(L), NEG)]
        for B in sorted(nb[X_] & nb[Z_])[:8]:                                 # PURE LATENT composition from X (the B-Z record itself is a stated fact the
            for k1 in recs.get((X_, B), ()):                                   #   map was fitted on, so predicting Z from B would only recall it)
              for k in recs.get((B, Z_), ()):
                if k not in W or k1 not in W: continue
                v = K[NI[X_]] @ W[k1] @ W[k]; v /= np.linalg.norm(v) + 1e-8
                s_true = float(K[NI[Z_]] @ v); s_neg = K[negs] @ v
                rk = min(rk, 1 + int((s_neg > s_true).sum()))
    best.append(rk)
best = np.array(best); has = best < 10 ** 9
def pr(m, n=None):
    idx = np.where(m)[0][:n] if n else np.where(m)[0]
    return f"{y[idx].mean() if len(idx) else float('nan'):.3f} ({len(idx)})"
P(f"MAP VOTE on the writer's top {len(pairs)}: with a usable bridge map {has.mean():.3f}; rank 1 {np.mean(best == 1):.3f}, top 3 {np.mean(best <= 3):.3f} (chance {1 / (NEG + 1):.3f} / {3 / (NEG + 1):.3f})")
ar = np.arange(len(y))
for n in (100, 200, 400, 1000, 3000):                                          # WITHIN the writer's top n: split by the map's vote
    t_ = ar < n
    P(f"   within the writer's top {n:4d} ({pr(t_)}): map rank 1 {pr(t_ & (best == 1))} | map top 3 {pr(t_ & (best <= 3))} | "
      f"map rank > 10 {pr(t_ & has & (best > 10))} | no usable map {pr(t_ & ~has)}")
P(f"DONE ({time.time() - T0:.0f}s)")
