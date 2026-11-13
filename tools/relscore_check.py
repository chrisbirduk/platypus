"""WHY IS THE IN-STORE RELATIONAL SCORER WEAKER? (13 Nov 2026; tools/relscore_check.py) - offline, no effect on the package.
The premises store's relational-wording scorer (platypus/inference/premises.py _typed_readout) reached top 5% 0.29-0.31 (base 0.20) on the
benchmark gold, against 0.55-0.60 (base 0.24) for the offline consistency learner (tools/consistency_teacher.py). Scored on the SAME
held-out Re-DocRED records (gold, evaluation only), variants of the in-store scorer:
  A  as in the package: teachers over the whole store (T1 corroboration, T2 functional, T3 selective), learner on hashed words between + ends
     + name shapes, trained on 200k records of the whole store
  B  as A, trained on Re-DocRED training records only (the store's own Wikipedia-like text)
  C  as B, teachers counted on Re-DocRED training records only
  D  as C + the offline learner's dense features (length, comma, brackets, digits, capitals between, wording frequency, empty)
  REF the offline consistency learner (teachers T1-T4 incl. counting-miner rules)
Usage: RM_EXTRA=<distant documents> python3 tools/relscore_check.py"""
import os
import sys
import math
import pickle
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('RM_K', '4'); os.environ.setdefault('RM_M', '8'); os.environ.setdefault('RM_R', '1')
_rm = open(os.path.join(HERE, 'record_mdl.py')).read()
exec(compile(_rm[:_rm.index('KBEST = min(FITS')], 'record_mdl', 'exec'))
from sklearn.feature_extraction.text import HashingVectorizer
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
EXTRA = int(os.environ.get('RM_EXTRA', 0)); DR = []
if EXTRA:
    _dr = pickle.load(open(SCR + '/distant/records.pkl', 'rb')); DR = [r for r in _dr['rec'] if r[0] < EXTRA]; del _dr
# the store: Re-DocRED records (C) first, then distant records
mid = [c['f'] for c in C] + [r[3] for r in DR]; An = [c['cA'] for c in C] + [r[1] for r in DR]; Bn = [c['cB'] for c in C] + [r[2] for r in DR]
dk = [c['di'] for c in C] + [100000 + r[0] for r in DR]; NS = len(mid)
def shp(n):
    if any(ch.isdigit() for ch in n): return 'num'
    k = sum(t[:1].isupper() for t in n.split()); return f"cap{min(k, 3)}" if k else 'low'
def teachers(ix):
    trip = collections.defaultdict(set); fs = collections.defaultdict(lambda: collections.defaultdict(set)); fA = collections.defaultdict(collections.Counter); fB = collections.defaultdict(collections.Counter)
    for i in ix: trip[(An[i], mid[i], Bn[i])].add(dk[i]); fs[mid[i]][An[i]].add(Bn[i]); fA[mid[i]][shp(An[i])] += 1; fB[mid[i]][shp(Bn[i])] += 1
    FUN = {f: float(np.mean([len(v) == 1 for v in d.values()])) for f, d in fs.items() if len(d) >= 3}
    ent = lambda c_: -sum(v / sum(c_.values()) * math.log2(v / sum(c_.values())) for v in c_.values())
    SEL = {f: -(ent(fA[f]) + ent(fB[f])) for f in fA if sum(fA[f].values()) >= 5}
    return trip, FUN, SEL
def score_of(trip, FUN, SEL, ix):
    t1 = np.array([float(len(trip.get((An[i], mid[i], Bn[i]), set()) - {dk[i]}) > 0) for i in ix])
    t2 = np.array([FUN.get(mid[i], np.nan) for i in ix]); t3 = np.array([SEL.get(mid[i], np.nan) for i in ix])
    rk_ = lambda x: (lambda z: np.argsort(np.argsort(z, kind='stable'), kind='stable') / max(1, len(z) - 1))(np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(x).any() else 0.0, x))
    return rk_(t2) + rk_(t3) + 2 * t1
HV = HashingVectorizer(n_features=2 ** 16, ngram_range=(1, 2), alternate_sign=False, token_pattern=r"[A-Za-z]+|[^\sA-Za-z]", norm='l2')
txt = lambda i: mid[i] + ' FIRST_' + (mid[i].split() or ['none'])[0] + ' LAST_' + (mid[i].split() or ['none'])[-1] + ' SA_' + shp(An[i]) + ' SB_' + shp(Bn[i])
fqm = collections.Counter(mid)
def dense(ix):
    D = np.array([[len(mid[i].split()), float(',' in mid[i]), float('(' in mid[i] or ')' in mid[i]), float(any(ch.isdigit() for ch in An[i] + Bn[i])),
                   float(any(w_[:1].isupper() for w_ in mid[i].split())), np.log1p(fqm[mid[i]]), float(mid[i] == '')] for i in ix], float)
    return csr_matrix((D - D.mean(0)) / (D.std(0) + 1e-6))
def learner(tr_ix, lab, dn=False):
    X = HV.transform([txt(i) for i in tr_ix]); X = hstack([X, dense(tr_ix)]).tocsr() if dn else X
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X, lab >= np.percentile(lab, 75))
    Xt = HV.transform([txt(i) for i in TE]); Xt = hstack([Xt, dense(TE)]).tocsr() if dn else Xt
    return m.decision_function(Xt)
def show(tag, sc):
    yy = y[TE]; o = np.argsort(-sc + 1e-9 * np.random.RandomState(5).rand(len(sc)))
    P(f"   {tag:62s} AUC {auc(sc[yy], sc[~yy]):.3f} | precision top 5% {yy[o[:len(o) // 20]].mean():.3f}, 10% {yy[o[:len(o) // 10]].mean():.3f}, 25% {yy[o[:len(o) // 4]].mean():.3f}")
P(f"store: {len(C)} Re-DocRED + {len(DR)} distant records; held-out Re-DocRED records {len(TE)} (related {y[TE].mean():.3f})  ({time.time() - T0:.0f}s)")
rs9 = np.random.RandomState(0); ALLI = np.arange(NS); trA = rs9.permutation(NS)[:200000]
tA = teachers(ALLI); labA = score_of(*tA, trA); show('A  as in the package (whole store, 200k sample)', learner(trA, labA))
labB = score_of(*tA, TR); show('B  trained on Re-DocRED training records (teachers over the store)', learner(TR, labB))
tC = teachers(TR); labC = score_of(*tC, TR); show('C  teachers AND training on Re-DocRED training records', learner(TR, labC))
show('D  C + dense features', learner(TR, labC, dn=True))
show('REF offline consistency learner (T1-T4)', s_all[TE])
P(f"DONE ({time.time() - T0:.0f}s)")
