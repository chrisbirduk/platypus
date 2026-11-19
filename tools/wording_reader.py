"""WORDING READER, SELF-TRAINED (19 Nov 2026; tools/wording_reader.py) - offline, distant set, LABEL-FREE (labels for evaluation only).
Causes 1-3 of the reader's limit: co-occurrence signals measure association, not what the sentence says, and 79% of pairs are seen once.
Here the cross-document signal (route 2, usefulness over time) TEACHES a wording model: on training documents, records of pairs seen in other
documents get a pseudo-label (top WR_POS share by usefulness = 1, bottom half = 0); a logistic regression on the words between the names
(unigrams + bigrams, the last word, distance bucket, and the two names' learned-type-free shapes) learns which wordings state a relation.
It is applied to held-out records, INCLUDING pairs never seen elsewhere (where co-occurrence says nothing).
Reference (evaluation only, not label-free): the same model trained on the true labels = the ceiling of these wording features.
Usage: python3 tools/wording_reader.py <distant.json.gz>   env PR_Y (doc|any), WR_POS (0.2)"""
import os, sys, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, LAB, ND = pm._PM['R'], pm._PM['LAB'], pm._PM['ND']
y = np.array([bool(r[6]) for r in R], float) if os.environ.get('PR_Y', 'doc') == 'doc' else np.array([frozenset((r[1], r[2])) in LAB for r in R], float)
doc = np.array([r[0] for r in R]); test = doc >= int(ND * 0.8); train = ~test
pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for r in R: k = frozenset((r[1], r[2])); pd[k].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0])
def use_(r):
    a_ = len(pd[frozenset((r[1], r[2]))]) - 1; nA, nB = len(nd[r[1]]) - 1, len(nd[r[2]]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); sA, sB = np.log(ND / (nA + 2)), np.log(ND / (nB + 2))
    return max(hA * sB - (1 - hA) * 0.5, hB * sA - (1 - hB) * 0.5)
U = np.array([use_(r) for r in R]); seen = ~np.isnan(U)
def shape(n): return ('cap' if n[:1].isupper() else 'low') + ('_num' if any(c.isdigit() for c in n) else '') + f"_w{min(len(n.split()), 3)}"
def text(r): return f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3} SA_{shape(r[1])} SB_{shape(r[2])}"
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([text(r) for r in R])
P(f"{ND} documents, {len(R)} records ({train.sum()} training, {test.sum()} held out; base rate held out {y[test].mean():.3f}); pairs seen in another document: "
  f"training {seen[train].mean():.3f}, held out {seen[test].mean():.3f}  ({time.time() - T0:.0f}s)")
pos_q = float(os.environ.get('WR_POS', 0.2)); tr_seen = np.where(train & seen)[0]; hi = np.quantile(U[tr_seen], 1 - pos_q); lo = np.quantile(U[tr_seen], 0.5)
ps = tr_seen[U[tr_seen] >= hi]; ng = tr_seen[U[tr_seen] <= lo]; ix = np.r_[ps, ng]; yl = np.r_[np.ones(len(ps)), np.zeros(len(ng))]
P(f"PSEUDO-LABELS from usefulness (training records seen elsewhere): {len(ps)} positive (top {pos_q:.0%}), {len(ng)} negative (bottom half); "
  f"true related share among them (evaluation only): positive {y[ps].mean():.3f}, negative {y[ng].mean():.3f}")
clf = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); clf.fit(X[ix], yl); sW = clf.decision_function(X)
ref = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); ref.fit(X[train], y[train]); sR = ref.decision_function(X)
P(f"   trained  ({time.time() - T0:.0f}s)")
def rep(nm, s, m):
    yy, ss = y[m], s[m]; o = np.argsort(-ss + 1e-9 * np.random.RandomState(0).rand(len(ss)))
    pr = ' '.join(f"{q}% {yy[o[:max(1, len(o) * q // 100)]].mean():.3f}" for q in (1, 2, 5, 10, 20))
    P(f"   {nm:56s} AUC {roc_auc_score(yy, ss):.3f} | precision top {pr}  (n {m.sum()}, base {yy.mean():.3f})")
rk = lambda v: np.argsort(np.argsort(np.nan_to_num(v, nan=-1e9) + 1e-9 * np.random.RandomState(1).rand(len(v)))).astype(float)
dist = -np.array([r[5] for r in R], float)
for lab, m in (('ALL held-out records', test), ('held-out pairs NEVER seen in another document', test & ~seen), ('held-out pairs seen elsewhere', test & seen)):
    P(f"{lab}:")
    rep('distance only', dist, m)
    if (m & seen).any() and lab != 'held-out pairs NEVER seen in another document': rep('usefulness (route 2)', np.nan_to_num(U, nan=-9.0), m)
    rep('WORDING READER, self-trained from usefulness (label-free)', sW, m)
    if lab == 'ALL held-out records': rep('wording reader + usefulness (ranks; unseen pairs by wording only)', np.where(seen, rk(sW) + rk(U), rk(sW) * 0.5), m)
    rep('reference: same features trained on TRUE labels (ceiling)', sR, m)
co = clf.coef_[0]; vocab = collections.Counter()
for r in np.random.RandomState(2).choice(np.where(train)[0], 50000, replace=False):
    for t in R[r][3].split(): vocab[t] += 1
ws = [w for w, c in vocab.items() if c >= 20]; wv = hv.transform(ws); sc = wv @ co
o = np.argsort(-sc); P(f"   wordings the self-trained reader trusts most: {', '.join(repr(ws[i]) for i in o[:20])}")
P(f"   wordings it trusts least: {', '.join(repr(ws[i]) for i in o[-15:])}")
P(f"DONE ({time.time() - T0:.0f}s)")
