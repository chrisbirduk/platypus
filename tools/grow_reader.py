"""GROWING READER (19 Nov 2026; tools/grow_reader.py) - offline, distant set, LABEL-FREE (own-document labels for evaluation only).
Does the reader improve as it reads more? Documents arrive in blocks (GR_BLOCK, default 10,000). After each block the reader is rebuilt from
EVERYTHING read so far and scored on the NEXT block (never read yet), as the package would read it online:
  usefulness over time (route 2; cross-document, from the documents read so far)
  WORDING reader self-trained from usefulness pseudo-labels (tools/wording_reader.py)
  CO-TRAINING (GR_ROUNDS rounds): the wording reader's most confident records on pairs seen ONCE join the positives, its least confident the
     negatives, and it is refitted - the two views teach each other; the risk is drift, which the next block's score shows.
Usage: python3 tools/grow_reader.py <distant_50k.json.gz>   env GR_BLOCK (10000), GR_ROUNDS (2), GR_ADD (0.02)"""
import os, sys, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, ND = pm._PM['R'], pm._PM['ND']; BL = int(os.environ.get('GR_BLOCK', 10000)); RO = int(os.environ.get('GR_ROUNDS', 2)); AD = float(os.environ.get('GR_ADD', 0.02))
y = np.array([bool(r[6]) for r in R], float); doc = np.array([r[0] for r in R])
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3}" for r in R])
P(f"{ND} documents, {len(R)} records; blocks of {BL}  ({time.time() - T0:.0f}s)")
def auc_p(s, m):
    yy, ss = y[m], s[m]; o = np.argsort(-ss + 1e-9 * np.random.RandomState(0).rand(len(ss)))
    return f"AUC {roc_auc_score(yy, ss):.3f} top 1% {yy[o[:max(1, len(o) // 100)]].mean():.2f} 5% {yy[o[:max(1, len(o) // 20)]].mean():.2f}"
for nb in range(1, ND // BL):
    rd = doc < nb * BL; te = (doc >= nb * BL) & (doc < (nb + 1) * BL)
    pd, nd = collections.defaultdict(set), collections.defaultdict(set)
    for i in np.where(rd | te)[0]:
        r = R[i]; pd[frozenset((r[1], r[2]))].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0])
    NDs = (nb + 1) * BL
    def use_(i):                                                               # read-so-far documents only (plus the record's own block for the test records:
        r = R[i]; own = r[0]                                                   #   a test record sees what was read before it and its own block, never later blocks)
        a_ = len(pd[frozenset((r[1], r[2]))] - {own}); nA, nB = len(nd[r[1]] - {own}), len(nd[r[2]] - {own})
        if not a_: return np.nan
        hA, hB = a_ / (nA + 2), a_ / (nB + 2); sA, sB = np.log(NDs / (nA + 2)), np.log(NDs / (nB + 2))
        return max(hA * sB - (1 - hA) * 0.5, hB * sA - (1 - hB) * 0.5)
    ri = np.where(rd)[0]; U = np.full(len(R), np.nan); U[ri] = [use_(i) for i in ri]; ti = np.where(te)[0]; U[ti] = [use_(i) for i in ti]
    sn = ri[~np.isnan(U[ri])]; hi, lo = np.quantile(U[sn], 0.8), np.quantile(U[sn], 0.5)
    pos, neg = sn[U[sn] >= hi], sn[U[sn] <= lo]
    def fit(p_, n_):
        c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); c.fit(X[np.r_[p_, n_]], np.r_[np.ones(len(p_)), np.zeros(len(n_))]); return c.decision_function(X)
    sW = fit(pos, neg); once = te & np.isnan(U)
    line = [f"read {nb * BL:5d} docs ({rd.sum()} records; pairs seen elsewhere {np.mean(~np.isnan(U[ri])):.3f}) -> next block {te.sum()} records:",
            f"usefulness {auc_p(np.nan_to_num(U, nan=-9.0), te)}", f"WORDING {auc_p(sW, te)} | seen once {auc_p(sW, once)}"]
    s = sW; ro_ = np.where(rd & np.isnan(U))[0]                                # co-training pool: records read so far whose pair is seen once
    for k in range(RO):
        o = np.argsort(-s[ro_]); kk = int(len(ro_) * AD); p2, n2 = ro_[o[:kk]], ro_[o[-kk * 2:]]
        s = fit(np.r_[pos, p2], np.r_[neg, n2])
        line.append(f"CO-TRAIN round {k + 1} (+{kk} pos / +{2 * kk} neg; their true related {y[p2].mean():.2f} / {y[n2].mean():.2f}) {auc_p(s, te)} | seen once {auc_p(s, once)}")
    P("   " + " || ".join(line) + f"  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
