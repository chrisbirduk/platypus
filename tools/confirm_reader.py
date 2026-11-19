"""CONFIRMATION-TRAINED READER (19 Nov 2026; tools/confirm_reader.py) - offline, distant set, LABEL-FREE (own-document labels for evaluation).
The user's design: the reader is not precise at first; it writes TENTATIVE records, later reading CONFIRMS some (reinforced) and the rest FADE;
the reader learns from what was confirmed and what faded, so it improves by learning. Documents arrive in blocks (CR_BLOCK, default 5000).
  block 0: the reader starts as the wording reader self-trained from usefulness on block 0 (as tools/wording_reader.py).
  each later block t: (1) the current reader is scored on block t BEFORE it learns from it (AUC on the unread block: the learning curve);
  (2) it writes its top CR_TOP of block t's records as TENTATIVE pairs; (3) a tentative pair from an earlier block is CONFIRMED when block t has
  the same pair in ANOTHER document, with a different record text (no copied boilerplate) and a RELATIONAL wording (the reader's score on that
  record in the top CR_RELQ of block t); a plain-recurrence control confirms on any recurrence; (4) tentative pairs unconfirmed after CR_K blocks
  FADE; (5) the reader is refitted: confirmed records positive, faded records negative (plus the block-0 usefulness pseudo-labels).
Arms: CONFIRM (the design), RECUR (confirmation by plain recurrence), STATIC (never retrained), USEFUL (refitted each block on usefulness
pseudo-labels from everything read: the flat curve of tools/grow_reader.py).
Precision is the original record's OWN-document label (no bias toward pairs seen in many documents).
Usage: python3 tools/confirm_reader.py <distant_50k.json.gz>   env CR_BLOCK (5000), CR_TOP (0.05), CR_RELQ (0.3), CR_K (2)"""
import os, sys, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, ND = pm._PM['R'], pm._PM['ND']; BL = int(os.environ.get('CR_BLOCK', 5000)); TOP = float(os.environ.get('CR_TOP', 0.05))
RQ = float(os.environ.get('CR_RELQ', 0.3)); KF = int(os.environ.get('CR_K', 2)); NB = ND // BL
y = np.array([bool(r[6]) for r in R], float); doc = np.array([r[0] for r in R]); blk = doc // BL
key = [frozenset((r[1], r[2])) for r in R]; txt = [r[7] for r in R]
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3}" for r in R])
byb = [np.where(blk == b)[0] for b in range(NB)]
P(f"{ND} documents in {NB} blocks of {BL}, {len(R)} records; tentative top {TOP:.0%} a block, relational confirmation in the top {RQ:.0%}, fade after {KF} blocks  ({time.time() - T0:.0f}s)")
def fit(pos, neg):
    c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); c.fit(X[np.r_[pos, neg]], np.r_[np.ones(len(pos)), np.zeros(len(neg))]); return c
def useful_labels(upto):                                                       # usefulness pseudo-labels over everything read in blocks < upto
    m = blk < upto; pd, nd = collections.defaultdict(set), collections.defaultdict(set); I = np.where(m)[0]
    for i in I: r = R[i]; pd[key[i]].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0])
    Nd = upto * BL; U = []
    for i in I:
        r = R[i]; a_ = len(pd[key[i]]) - 1; nA, nB = len(nd[r[1]]) - 1, len(nd[r[2]]) - 1
        if not a_: U.append(np.nan); continue
        hA, hB = a_ / (nA + 2), a_ / (nB + 2); U.append(max(hA * np.log(Nd / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(Nd / (nA + 2)) - (1 - hB) * 0.5))
    U = np.array(U); ok = ~np.isnan(U); hi, lo = np.quantile(U[ok], 0.8), np.quantile(U[ok], 0.5)
    return I[ok & (U >= hi)], I[ok & (U <= lo)]
p0, n0 = useful_labels(1)
P(f"block 0: usefulness pseudo-labels {len(p0)} / {len(n0)} (true related {y[p0].mean():.3f} / {y[n0].mean():.3f})")
ARMS = os.environ.get('CR_ARMS', 'CONFIRM,RECUR,STATIC,USEFUL').split(','); rsu = np.random.RandomState(5)
arms = {nm: dict(clf=fit(p0, n0), tent={}, pos=[], neg=[], curve=[], log=[]) for nm in ARMS}
def auc_top(s, I):
    yy = y[I]; o = np.argsort(-s); return roc_auc_score(yy, s), yy[o[:max(1, len(o) // 20)]].mean()
for t in range(1, NB):
    I = byb[t]
    for nm, A in arms.items():
        s = A['clf'].decision_function(X[I]); au, p5 = auc_top(s, I); A['curve'].append((t, au, p5))
        if nm in ('STATIC',): continue
        if nm == 'USEFUL': A['clf'] = fit(*useful_labels(t + 1)); continue
        if nm == 'USEFUL+CONF': pu_, nu_ = useful_labels(t + 1)                # usefulness from everything read, plus confirmed records as extra positives
        thr = np.quantile(s, 1 - RQ); rel = {i for i, v in zip(I, s) if v >= thr}; seenb = collections.defaultdict(list)
        for i in I: seenb[key[i]].append(i)
        conf, fade = [], []
        for k, (i0, t0, d0) in list(A['tent'].items()):                       # confirmation of earlier tentative pairs by this block
            hit = [j for j in seenb.get(k, []) if doc[j] != d0 and txt[j] != txt[i0] and (nm == 'RECUR' or j in rel)]
            if hit: conf.append(i0); del A['tent'][k]
            elif t - t0 >= KF: fade.append(i0); del A['tent'][k]
        top = I[np.argsort(-s)[:int(len(I) * TOP)]]
        for i in top:
            if key[i] not in A['tent']: A['tent'][key[i]] = (i, t, doc[i])
        A['pos'] += conf; A['neg'] += fade
        A['log'].append((t, len(top), y[top].mean(), len(conf), y[conf].mean() if conf else np.nan, len(fade), y[fade].mean() if fade else np.nan))
        PO = np.array(A['pos'], int)
        if nm in ('CONFIRM_PU', 'USEFUL+CONF') and len(PO):                    # 19 Nov: unconfirmed is NOT false (79% of pairs never recur): PU learning - confirmed vs a
            rd = np.where(blk <= t)[0]; UN = rd[rsu.randint(len(rd), size=2 * len(PO))]   #   RANDOM sample of what was read (unlabelled), not the faded top picks
            A['clf'] = fit(np.r_[p0, PO], np.r_[n0, UN]) if nm == 'CONFIRM_PU' else fit(np.r_[pu_, PO], np.r_[nu_, UN])
        elif nm == 'USEFUL+CONF': A['clf'] = fit(pu_, nu_)
        elif A['pos'] and A['neg']: A['clf'] = fit(np.r_[p0, PO], np.r_[n0, np.array(A['neg'], int)])
    P(f"   block {t:2d} (unread, {len(I)} records, base {y[I].mean():.3f}): " + " | ".join(f"{nm} AUC {A['curve'][-1][1]:.3f} top5% {A['curve'][-1][2]:.2f}" for nm, A in arms.items())
      + "  || " + " | ".join(f"{nm}: tentative {L[1]} (right {L[2]:.2f}), confirmed {L[3]} (right {L[4]:.2f}), faded {L[5]} (right {L[6]:.2f})"
                            for nm, A in arms.items() if A['log'] for L in [A['log'][-1]]) + f"  ({time.time() - T0:.0f}s)")
P("TOTALS (all blocks): " + " | ".join(f"{nm}: confirmed {len(A['pos'])} right {y[A['pos']].mean() if A['pos'] else float('nan'):.3f}, faded {len(A['neg'])} right "
                                         f"{y[A['neg']].mean() if A['neg'] else float('nan'):.3f}" for nm, A in arms.items() if A['log']))
P("CURVES (AUC on each unread block): " + " || ".join(f"{nm}: " + ' '.join(f"{a:.3f}" for _, a, _ in A['curve']) for nm, A in arms.items()))
P(f"DONE ({time.time() - T0:.0f}s)")
