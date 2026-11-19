"""TEACHING THE READER MEANING, NOT FORM (19 Nov 2026; tools/teach_reader.py) - offline, distant set, LABEL-FREE (own-document labels and gold
relations for evaluation only). The reader learns commas and brackets ('Paris, France') and misses relations stated by phrases ('born in',
'president of'): its teacher (usefulness) rewards what recurs, and comma pairs recur most. Arms, rebuilt after each block and scored on the
NEXT, unread block (blocks of TC_BLOCK documents):
  USEFUL      usefulness pseudo-labels (top 20% / bottom half), as now
  BALANCED    the same, chosen WITHIN each wording form (comma / bracket / list, short phrase 1-3 words, longer phrase 4-12 words), so that
              form alone cannot separate positives from negatives
  KNOWLEDGE   the reader's top 5% of each block are tentative; a pair is confirmed when a later block states it in another document in the
              reader's top 30%; then EVERY record of a confirmed pair, whatever its wording, is a positive (distant supervision from the
              system's own confirmed facts), with the usefulness labels
  KNOWLEDGE + BALANCED
Measured on each unread block: AUC; precision of the top 10%; RECALL of related records stated by a PHRASE (2-12 words between, not a list
or bracket) in the top 10% - now 0.11-0.19 - and of person-place relations.
Usage: python3 tools/teach_reader.py <distant_50k.json.gz>   env TC_BLOCK (5000)"""
import os, sys, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=1))
R, ND, TY = pm._PM['R'], pm._PM['ND'], pm._PM['TY']; BL = int(os.environ.get('TC_BLOCK', 5000)); NB = ND // BL
y = np.array([bool(r[6]) for r in R]); doc = np.array([r[0] for r in R]); blk = doc // BL; key = [frozenset((r[1], r[2])) for r in R]
lst = np.array([pm._rret_listlike(r) for r in R]); dist = np.array([r[5] for r in R]); par = np.array(['(' in r[3] for r in R])
form = np.where(lst | par | (dist <= 1), 0, np.where(dist <= 3, 1, 2))     # 0 comma / bracket / list / adjacent, 1 short phrase, 2 longer phrase
phrase = (form > 0); perloc = np.array([{TY.get(r[1]), TY.get(r[2])} == {'PER', 'LOC'} for r in R])
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3}" for r in R])
P(f"{ND} documents, {NB} blocks, {len(R)} records; forms: comma/bracket/list {np.mean(form == 0):.2f} (related {y[form == 0].mean():.3f}), short phrase "
  f"{np.mean(form == 1):.2f} ({y[form == 1].mean():.3f}), longer phrase {np.mean(form == 2):.2f} ({y[form == 2].mean():.3f})  ({time.time() - T0:.0f}s)")
def useful(upto):
    I = np.where(blk < upto)[0]; pd, nd = collections.defaultdict(set), collections.defaultdict(set)
    for i in I: r = R[i]; pd[key[i]].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0])
    U = np.full(len(I), np.nan); Nd = upto * BL
    for n_, i in enumerate(I):
        r = R[i]; a_ = len(pd[key[i]]) - 1
        if a_:
            nA, nB = len(nd[r[1]]) - 1, len(nd[r[2]]) - 1; hA, hB = a_ / (nA + 2), a_ / (nB + 2)
            U[n_] = max(hA * np.log(Nd / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(Nd / (nA + 2)) - (1 - hB) * 0.5)
    return I, U
def labels(I, U, balanced):
    ok = ~np.isnan(U); pos, neg = [], []
    groups = [np.ones(len(I), bool)] if not balanced else [form[I] == f for f in (0, 1, 2)]
    for g in groups:
        m = ok & g
        if m.sum() < 20: continue
        hi, lo = np.quantile(U[m], 0.8), np.quantile(U[m], 0.5); pos.append(I[m & (U >= hi)]); neg.append(I[m & (U <= lo)])
    if balanced:                                                               # equal numbers from each form, so form is not a shortcut
        k = min(len(p) for p in pos); pos = [p[np.random.RandomState(1).permutation(len(p))[:k]] for p in pos]
        k = min(len(n) for n in neg); neg = [n[np.random.RandomState(2).permutation(len(n))[:k]] for n in neg]
    return np.concatenate(pos), np.concatenate(neg)
def fit(pos, neg):
    c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); c.fit(X[np.r_[pos, neg]], np.r_[np.ones(len(pos)), np.zeros(len(neg))]); return c
ARMS = ['USEFUL', 'BALANCED', 'KNOWLEDGE', 'KNOWLEDGE + BALANCED']; st = {a: dict(clf=None, tent={}, conf=set()) for a in ARMS}; curves = collections.defaultdict(list)
byk_all = collections.defaultdict(list)
for i, k in enumerate(key): byk_all[k].append(i)
for t in range(1, NB):
    I, U = useful(t); Inext = np.where(blk == t)[0]; line = []
    for a in ARMS:
        A = st[a]; pos, neg = labels(I, U, 'BALANCED' in a)
        if 'KNOWLEDGE' in a and A['clf'] is not None:                         # tentative top 5% of the last block; confirmed by the block just read (t - 1)
            Ib = np.where(blk == t - 1)[0]; sb = A['clf'].decision_function(X[Ib]); thr = np.quantile(sb, 0.7); sc = dict(zip(Ib, sb))
            for k_, (i0, d0) in list(A['tent'].items()):
                if k_ in A['conf']: continue
                if any(blk[j] == t - 1 and doc[j] != d0 and R[j][7] != R[i0][7] and sc[j] >= thr for j in byk_all[k_]): A['conf'].add(k_)
            for i in Ib[np.argsort(-sb)[:len(Ib) // 20]]:
                if key[i] not in A['tent']: A['tent'][key[i]] = (i, doc[i])
            kp = np.array([j for k_ in A['conf'] for j in byk_all[k_] if blk[j] < t], int)   # every record of a confirmed pair read so far, any wording
            if len(kp): pos = np.unique(np.r_[pos, kp]); neg = np.setdiff1d(neg, kp)
        A['clf'] = fit(pos, neg); s = A['clf'].decision_function(X[Inext]); o = np.argsort(-s); top = np.zeros(len(Inext), bool); top[o[:len(o) // 10]] = True
        yy = y[Inext]; ph = phrase[Inext]; pl = perloc[Inext]; au = roc_auc_score(yy, s); curves[a].append(au)
        line.append(f"{a}: AUC {au:.3f}, top10% {yy[top].mean():.3f}, phrase recall {top[yy & ph].mean():.3f}, person-place {top[yy & pl].mean():.3f}"
                    + (f" (confirmed pairs {len(A['conf'])}, {np.mean([y[byk_all[k_][0]] for k_ in A['conf']]) if A['conf'] else float('nan'):.2f} right; positives {len(pos)}, phrase share {phrase[pos].mean():.2f})" if 'KNOWLEDGE' in a else f" (positives {len(pos)}, phrase share {phrase[pos].mean():.2f})"))
    P(f"   block {t} (read {t * BL}): " + " || ".join(line) + f"  ({time.time() - T0:.0f}s)")
P("CURVES (AUC on each unread block): " + " || ".join(f"{a}: " + " ".join(f"{x:.3f}" for x in v) for a, v in curves.items()))
P(f"DONE ({time.time() - T0:.0f}s)")
