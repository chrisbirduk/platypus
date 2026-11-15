"""LEARNED CONTROL OVER WHICH INFERENCES ARE FORMED (15 Nov 2026; tools/learned_selector.py) - offline, the GPU code path, label-free loop.
The judge's verdict is the one label-free feedback that works (relational records of the pair since the write: AUC 0.69-0.70). Here it is
fed BACK: a selector learns from the judged outcomes of earlier rounds which write-time features predict 'later confirmed by relational
records', and chooses the next round's writes. Rounds as on the GPU: candidates and their features from premises._rcode_step (the package's
own write step) at documents 2/10, 4/10, 6/10 (SL_STAGE=cand saves each round in its own process: memory); the self-label of a candidate
= relational records of the pair in the documents read by two visits later (no Wikidata); features at WRITE time: the code, Adamic-Adar,
resource allocation, both degrees and their asymmetry, the names' entity types, and the pair's own co-mention in the held-out documents it
came from - relational records vs list-only records (the error class).
Round 1 writes by the rule (code + AA under the asymmetry filter, as the GPU); round k > 1: a logistic selector trained on all candidates of
the earlier rounds with their self-labels (the store can check any pair later, written or not) picks the top 200. Scored with the distant
labels (evaluation only): precision of the 200 written each round - learned vs the rule - and what the judge then passes to the trunk.
Usage: SL_STAGE=cand SL_ROUND=k python3 tools/learned_selector.py <train_distant.json.gz> (k = 1, 2, 3), then SL_STAGE=learn"""
import sys, os, zlib, types, time, collections
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time()
S = os.environ.get('SL_OUT', 'sl'); ST = os.environ.get('SL_STAGE', 'cand')
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; TY = pm._PM['TY']
h = lambda v: int(round(ND * min(1.0, (v + 1) / 10)))
VIS = {1: 1, 2: 3, 3: 5}                                                       # rounds -> write visits (2, 4, 6); judged two visits later
listlike = pm._rret_listlike
if ST == 'cand':
    k = int(os.environ['SL_ROUND']); v = VIS[k]; pm._PM['rti'] = {}; pm._PM['save_cand'] = True
    pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(v)], v); C = pm._PM['cand']
    want = {frozenset(p): i for i, p in enumerate(C['pairs'])}; n = len(want)
    hold = lambda d: zlib.crc32(b'rcode%d' % int(d)) % 1000 < 100
    relW, lstW, relL = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
    for r in R:
        i = want.get(frozenset((r[1], r[2])))
        if i is None: continue
        rel = 1 <= r[5] <= 8 and not listlike(r)
        if r[0] < h(v) and hold(r[0]): relW[i] += rel; lstW[i] += not rel      # write time: the held-out document(s) the candidate came from
        if r[0] < h(v + 2) and rel: relL[i] += 1                               # the self-label: relational records by two visits later
    ET = ['PER', 'LOC', 'ORG', 'TIME', 'NUM', 'MISC']
    tA = np.array([ET.index(TY.get(a)) if TY.get(a) in ET else 6 for a, b in C['pairs']]); tB = np.array([ET.index(TY.get(b)) if TY.get(b) in ET else 6 for a, b in C['pairs']])
    np.savez(f"{S}_r{k}.npz", code=C['code'], aa=C['aa'], ra=C['ra'], dA=C['dA'], dB=C['dB'], y=C['y'], rule=C['rule'], relW=relW, lstW=lstW, relL=relL, tA=tA, tB=tB)
    P(f"round {k} (written at visit {v + 1}): {n} candidates, labelled right {C['y'].mean():.3f}; with a relational co-mention at write time {np.mean(relW > 0):.3f}; "
      f"self-label (relational records by visit {v + 3}) {np.mean(relL > 0):.3f}  ({time.time() - T0:.0f}s)"); sys.exit(0)
# learn: rounds in order
from sklearn.linear_model import LogisticRegression
from platypus.inference.compress import _cg_auc
def feats(D):
    ld = np.log1p(np.stack([D['dA'], D['dB']], 1)); asy = np.abs(ld[:, 0] - ld[:, 1])[:, None]
    oh = lambda t: np.eye(7)[t]
    return np.concatenate([D['code'][:, None], np.log1p(D['aa'])[:, None], np.log1p(D['ra'] * 100)[:, None], ld, asy, np.log1p(D['relW'])[:, None], np.log1p(D['lstW'])[:, None],
                           ((D['relW'] == 0) & (D['lstW'] > 0)).astype(np.float32)[:, None], oh(D['tA']), oh(D['tB'])], 1)
RD_ = [dict(np.load(f"{S}_r{k}.npz")) for k in (1, 2, 3)]
judged = lambda D, sel: (D['relL'][sel] > 0)                                   # the judge's verdict on a written batch (relational records later)
Xtr, ytr = [], []
for k, D in enumerate(RD_, 1):
    y = D['y'].astype(bool); X = feats(D); rule = np.where(np.isfinite(D['rule']), D['rule'], -1e9)
    top_rule = np.argsort(-rule)[:200]; msg = f"ROUND {k}: {len(y)} candidates (right {y.mean():.3f}) | RULE top 200 right {y[top_rule].mean():.3f}"
    if Xtr:
        Xa, ya = np.concatenate(Xtr), np.concatenate(ytr); mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced').fit((Xa - mu) / sd, ya); sc = clf.decision_function((X - mu) / sd)
        top_l = np.argsort(-sc)[:200]; both = np.argsort(-(np.argsort(np.argsort(sc)) + np.argsort(np.argsort(rule))))[:200]
        jl = judged(D, top_l); jr = judged(D, top_rule)
        msg += (f" | LEARNED top 200 right {y[top_l].mean():.3f} (self-label AUC on this round {_cg_auc(sc[D['relL'] > 0], sc[D['relL'] == 0]):.3f}, true-label AUC {_cg_auc(sc[y], sc[~y]):.3f})"
                f" | RULE + LEARNED rank sum {y[both].mean():.3f} || AFTER THE JUDGE (kept = relational records later): rule {y[top_rule][jr].mean() if jr.any() else float('nan'):.3f} "
                f"({jr.sum()} kept), learned {y[top_l][jl].mean() if jl.any() else float('nan'):.3f} ({jl.sum()} kept)")
        coef = clf.coef_[0]; nm = ['code', 'AA', 'RA', 'log deg A', 'log deg C', 'asymmetry', 'REL co-mention', 'LIST co-mention', 'list-only'] + [f'A:{t}' for t in ['PER', 'LOC', 'ORG', 'TIME', 'NUM', 'MISC', 'X']] + [f'C:{t}' for t in ['PER', 'LOC', 'ORG', 'TIME', 'NUM', 'MISC', 'X']]
        msg += " | strongest features: " + ", ".join(f"{nm[i]} {coef[i]:+.2f}" for i in np.argsort(-np.abs(coef))[:6])
    P(msg)
    Xtr.append(X); ytr.append(D['relL'] > 0)                                   # the system learns from its own judged outcomes, not from labels
P(f"DONE ({time.time() - T0:.0f}s)")
