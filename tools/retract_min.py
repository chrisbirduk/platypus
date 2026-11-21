"""CONFIRM, TEACH, RETRACT (21 Nov 2026; tools/retract_min.py) - offline, distant set, the package's own writes. Chooses kra21's two rules.
kra20 on the GPU: every confirmed inference now trains the trunk and is recalled (taught right 0.96, taught WRONG 0.95): what the trunk holds is
as right as what confirmation passes (0.80), and a confirmed inference is never judged again (the retraction trigger does not run under the
confirm pipeline). Here, as tools/confirm_min.py: the package's code writes (premises._rcode_step, kra14's rule, the seeded reader's top half)
from the first RT_FRAC of the documents; CONFIRMATION on the next slice (to RT_CONF), by rules from kra20's (one record the reader rates in its
top half) to stricter ones; then RETRACTION of confirmed inferences on the documents after (to the end) by CONTRADICTION: head-first relational
records of the inference's head naming ANOTHER tail (not one the store linked to the inference's tail before the write: a container is not a
contradiction), optionally only in the inference's own typed wording, against its confirmations (relational records of the pair).
For each confirmation rule: confirmed (right); for each retraction rule on top: retracted (right among them), wrong removed, right lost,
and what the trunk would keep (right share, count right).
Usage: python3 tools/retract_min.py <distant_50k.json.gz>   env RT_FRAC (0.4), RT_CONF (0.6), RT_END (1.0: the retraction window ends there), RT_W (2400)"""
import os, sys, types, time, collections
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time()
FR_ = float(os.environ.get('RT_FRAC', 0.4)); FC_ = float(os.environ.get('RT_CONF', 0.6)); FE_ = float(os.environ.get('RT_END', 1.0)); NW = int(os.environ.get('RT_W', 2400))
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa', inf_rcode_asym=1.0,
                            inf_rcode_selreport=0, inf_rti_max=NW, inf_rret=0, inf_rti_ctl=0.0, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rcode_reader=0.5, inf_premises_seed_ctx=1, inf_rti_typed=1)
pm._load_distant(cfg); ND = pm._PM['ND']; CX = pm._PM.get('CTX') or {}; R = pm._PM['R']; n0 = int(round(ND * FR_)); n1 = int(round(ND * FC_))
y = np.array([bool(r[6]) for r in R]); hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
wtx = lambda r: f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(int(r[5]), 12) // 3} " + ' '.join('C_' + w for w in CX.get(id(r), '').split())
iRD = np.array([i for i, r in enumerate(R) if r[0] < n0]); sd = iRD[np.random.RandomState(4 + 707).permutation(len(iRD))[:20000]]
c = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); c.fit(hv.transform([wtx(R[i]) for i in sd]), y[sd])
score = np.concatenate([c.decision_function(hv.transform([wtx(R[i]) for i in range(j, min(len(R), j + 200000))])) for j in range(0, len(R), 200000)])
RD = [R[i] for i in iRD]; pm._PM['rdscore'] = score[iRD]; pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], RD, 3)
W = pm._PM['rti']; K = list(W); T = {k: bool(W[k]['truth']) for k in K}
q50, q75, q90 = (float(np.quantile(score[iRD], q)) for q in (0.5, 0.75, 0.9))
P(f"{len(RD)} records read ({n0} of {ND} documents); {len(K)} written (right {np.mean(list(T.values())):.3f}); confirm on documents {n0}-{n1}, retract on {n1}-{ND}  ({time.time() - T0:.0f}s)")
FK = {frozenset(k): k for k in K}; HD = collections.defaultdict(list)
for k in K: HD[W[k].get('th') or k[0]].append(k)
TL = {W[k].get('tt') or k[1] for k in K}
EVc = collections.defaultdict(list); EVr = collections.defaultdict(list); CT = collections.defaultdict(list); LK = set()
for i, r in enumerate(R):
    rel = 1 <= r[5] <= 8 and not pm._rret_listlike(r)
    if r[0] < n0:
        if rel and (r[1] in TL or r[2] in TL): LK.add(frozenset((r[1], r[2])))      # pairs the store linked before the write (containers of the tail)
        continue
    if not rel or r[0] >= int(round(ND * FE_)): continue
    fk = frozenset((r[1], r[2]))
    if fk in FK: (EVc if r[0] < n1 else EVr)[FK[fk]].append(float(score[i]))          # relational records of the pair: confirming window / later
    if r[0] >= n1 and r[1] in HD:                                                    # head-first records of an inference's head, later window
        for k in HD[r[1]]:
            tt = W[k].get('tt') or k[1]
            if r[2] != tt and frozenset((r[2], tt)) not in LK: CT[k].append((pm._rnorm(r[3]), float(score[i])))
CONF = {'any record (kra14)': lambda L: len(L) >= 1, '1 rated, top half (kra17-kra20)': lambda L: any(s >= q50 for s in L),
        '1 rated, top quarter': lambda L: any(s >= q75 for s in L), '1 rated, top tenth': lambda L: any(s >= q90 for s in L),
        '2 rated, top half': lambda L: sum(s >= q50 for s in L) >= 2, '1 top quarter or 2 top half': lambda L: any(s >= q75 for s in L) or sum(s >= q50 for s in L) >= 2}
def ctr(k, mode, rated):
    tw = W[k].get('tw'); L = CT.get(k, [])
    return sum(1 for w, s in L if (mode == 'any' or w == tw) and (not rated or s >= q50))
RET = [(f"{m} contradictions, {'own wording' if md == 'own' else 'any wording'}{', rated' if rt else ''}, > {f}x confirmations", m, md, rt, f)
       for md in ('own', 'any') for rt in (0, 1) for m in (1, 2, 3) for f in (1, 2)]
nR = sum(T.values())
for cn, cf in CONF.items():
    cs = [k for k in K if cf(EVc.get(k, []))]; nr = sum(T[k] for k in cs)
    P(f"CONFIRM {cn:34s}: {len(cs):4d} confirmed, right {np.mean([T[k] for k in cs]) if cs else float('nan'):.3f} ({nr} right = {nr / max(nR, 1):.2f} of the right writes)")
    if cn not in ('1 rated, top half (kra17-kra20)', '1 rated, top quarter'): continue
    nw = len(cs) - nr; out = []
    for rn, m, md, rt, f in RET:
        rr = [k for k in cs if (lambda c_: c_ >= m and c_ > f * len(EVr.get(k, [])))(ctr(k, md, rt))]; kp = [k for k in cs if k not in set(rr)]
        rw = sum(not T[k] for k in rr); rg = sum(T[k] for k in rr)
        out.append((np.mean([T[k] for k in kp]) if kp else 0, f"   RETRACT {rn:62s}: {len(rr):4d} retracted (right {rg / max(len(rr), 1):.2f}) | wrong removed {rw / max(nw, 1):.2f} | right lost "
                    f"{rg / max(nr, 1):.2f} | KEPT {len(kp)} right {np.mean([T[k] for k in kp]) if kp else float('nan'):.3f} ({sum(T[k] for k in kp)} right)"))
    for rn, f_ in (('LATER records of the pair (>= 2) all rated below the median', lambda L: len(L) >= 2 and max(L) < q50),
                   ('LATER records of the pair (>= 1) all rated below the median', lambda L: len(L) >= 1 and max(L) < q50),
                   ('LATER mean rating below the median (>= 2 records)', lambda L: len(L) >= 2 and np.mean(L) < q50),
                   ('NO later record of the pair', lambda L: len(L) == 0),
                   ('own-wording contradiction (1, > 2x) OR later records all below the median (>= 2)', None)):
        if f_ is None: rr = [k for k in cs if (ctr(k, 'own', 0) >= 1 and ctr(k, 'own', 0) > 2 * len(EVr.get(k, []))) or (len(EVr.get(k, [])) >= 2 and max(EVr[k]) < q50)]
        else: rr = [k for k in cs if f_(EVr.get(k, []))]
        kp = [k for k in cs if k not in set(rr)]; rw = sum(not T[k] for k in rr); rg = sum(T[k] for k in rr)
        out.append((np.mean([T[k] for k in kp]) if kp else 0, f"   RETRACT {rn:62s}: {len(rr):4d} retracted (right {rg / max(len(rr), 1):.2f}) | wrong removed {rw / max(nw, 1):.2f} | right lost "
                    f"{rg / max(nr, 1):.2f} | KEPT {len(kp)} right {np.mean([T[k] for k in kp]) if kp else float('nan'):.3f} ({sum(T[k] for k in kp)} right)"))
    for _, s in sorted(out, key=lambda x: -x[0])[:14]: P(s)
P(f"DONE ({time.time() - T0:.0f}s)")
