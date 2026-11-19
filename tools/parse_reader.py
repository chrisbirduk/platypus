"""PARSER READER TEST (19 Nov 2026; tools/parse_reader.py) - offline, distant set, LABEL-FREE (labels for evaluation only).
Does sentence structure help the reader? Records = name pairs within 12 words of one sentence (as the package's reader). Each record's
sentence is parsed with spaCy (en_core_web_sm, on the given tokens) and the dependency path between the two names is taken: the path's
labels, the word where the two branches meet, its length, and whether it is pure coordination (a list). Compared, all self-trained from the
usefulness pseudo-labels exactly as tools/wording_reader.py (no labels):
  WORDING (words between the names)  vs  PATH ONLY  vs  WORDING + PATH;  plus two label-free rules (not a coordination path; path length).
Reference (evaluation only): WORDING + PATH trained on the true labels. Labels: the pair labelled in its OWN document.
Usage: python3 tools/parse_reader.py <distant.json.gz>   env PA_DOCS (all), PA_THREADS (1)"""
import os, sys, gzip, json, time, collections
import numpy as np
import spacy
from spacy.tokens import Doc
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
DOCS = json.load(gzip.open(sys.argv[1])); ND = min(len(DOCS), int(os.environ.get('PA_DOCS', len(DOCS))))
nlp = spacy.load('en_core_web_sm', disable=['ner'])
recs, todo = [], {}
for di in range(ND):
    d = DOCS[di]; cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    lab = {(l['h'], l['t']) for l in d['labels']}; bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms)
        for x in range(len(ms)):
            for y_ in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y_]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb] or b0 - a1 > 12: continue
                recs.append((di, si, a0, a1, b0, b1, cn[ea], cn[eb], (ea, eb) in lab or (eb, ea) in lab)); todo[(di, si)] = d['sents'][si]
P(f"{ND} documents, {len(recs)} records, {len(todo)} sentences to parse  ({time.time() - T0:.0f}s)")
keys = list(todo); parsed = {}
for k, dd in zip(keys, nlp.pipe((Doc(nlp.vocab, words=[w if w.strip() else '_' for w in todo[k]]) for k in keys), batch_size=256)): parsed[k] = dd
P(f"   parsed  ({time.time() - T0:.0f}s)")
def root(dd, s, e):
    for t in dd[s:e]:
        if not (s <= t.head.i < e) or t.head.i == t.i: return t
    return dd[s]
COORD = {'conj', 'cc', 'punct', 'appos', 'compound', 'npadvmod'}
def path(dd, a0, a1, b0, b1):
    i, j = root(dd, a0, a1), root(dd, b0, b1); anc = {}
    t, n = i, 0
    while True:
        anc[t.i] = n
        if t.head.i == t.i or n > 40: break
        t = t.head; n += 1
    up, t, dn = [], j, []
    while t.i not in anc and t.head.i != t.i and len(dn) < 40: dn.append(t.dep_); t = t.head
    if t.i not in anc: return 'NOPATH', 'none', 99, False
    lca = t; t = i
    while t.i != lca.i: up.append(t.dep_); t = t.head
    deps = up + dn[::-1]
    return '^'.join(up) + '|' + lca.lemma_.lower() + '|' + 'v'.join(dn[::-1]), lca.lemma_.lower(), len(deps), all(x in COORD for x in deps) and len(deps) > 0
F = []
for r in recs:
    dd = parsed[(r[0], r[1])]; ps, lc, ln, co = path(dd, r[2], r[3], r[4], r[5])
    mid = ' '.join(DOCS[r[0]]['sents'][r[1]][r[3]:r[4]]).lower(); F.append((mid, ps, lc, ln, co))
y = np.array([r[8] for r in recs], float); doc = np.array([r[0] for r in recs]); test = doc >= int(ND * 0.8); train = ~test
pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for r in recs: pd[frozenset((r[6], r[7]))].add(r[0]); nd[r[6]].add(r[0]); nd[r[7]].add(r[0])
def use_(r):
    a_ = len(pd[frozenset((r[6], r[7]))]) - 1; nA, nB = len(nd[r[6]]) - 1, len(nd[r[7]]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); sA, sB = np.log(ND / (nA + 2)), np.log(ND / (nB + 2))
    return max(hA * sB - (1 - hA) * 0.5, hB * sA - (1 - hB) * 0.5)
U = np.array([use_(r) for r in recs]); seen = ~np.isnan(U); dist = -np.array([r[4] - r[3] for r in recs], float)
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
tw = lambda f, r: f"{f[0]} LAST_{f[0].split()[-1] if f[0] else 'none'} DIST_{min(r[4] - r[3], 12) // 3}"
tp = lambda f: f"PATH_{f[1].replace(' ', '_')} LCA_{f[2]} PLEN_{min(f[3], 8)} COORD_{int(f[4])}"
XW = hv.transform([tw(f, r) for f, r in zip(F, recs)]); XP = hv.transform([tp(f) for f in F]); XB = hv.transform([tw(f, r) + ' ' + tp(f) for f, r in zip(F, recs)])
tr_seen = np.where(train & seen)[0]; hi = np.quantile(U[tr_seen], 0.8); lo = np.quantile(U[tr_seen], 0.5)
ps_ = tr_seen[U[tr_seen] >= hi]; ng_ = tr_seen[U[tr_seen] <= lo]; ix = np.r_[ps_, ng_]; yl = np.r_[np.ones(len(ps_)), np.zeros(len(ng_))]
P(f"held out {test.sum()} (base {y[test].mean():.3f}); pseudo-labels {len(ps_)} positive / {len(ng_)} negative (true related {y[ps_].mean():.3f} / {y[ng_].mean():.3f}); "
  f"coordination paths {np.mean([f[4] for f in F]):.3f} of records, related among them {y[np.array([f[4] for f in F])].mean():.3f} vs {y[~np.array([f[4] for f in F])].mean():.3f}")
def fit(X, ii, yy): c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); c.fit(X[ii], yy); return c.decision_function(X)
sW, sP, sB = fit(XW, ix, yl), fit(XP, ix, yl), fit(XB, ix, yl); sR = fit(XB, np.where(train)[0], y[train])
def rep(nm, s, m):
    yy, ss = y[m], s[m]; o = np.argsort(-ss + 1e-9 * np.random.RandomState(0).rand(len(ss)))
    pr = ' '.join(f"{q}% {yy[o[:max(1, len(o) * q // 100)]].mean():.3f}" for q in (1, 2, 5, 10))
    P(f"   {nm:52s} AUC {roc_auc_score(yy, ss):.3f} | precision top {pr}  (n {m.sum()}, base {yy.mean():.3f})")
co_ = np.array([f[4] for f in F]); pl = np.array([f[3] for f in F], float)
for lab, m in (('ALL held-out records', test), ('held-out pairs NEVER seen in another document', test & ~seen)):
    P(f"{lab}:")
    rep('distance only', dist, m); rep('RULE: not a coordination path (then nearer)', (~co_) * 100 + dist, m); rep('RULE: shorter dependency path', -pl, m)
    rep('WORDING, self-trained (label-free)', sW, m); rep('PATH ONLY, self-trained (label-free)', sP, m); rep('WORDING + PATH, self-trained (label-free)', sB, m)
    rep('reference: WORDING + PATH on TRUE labels (ceiling)', sR, m)
P(f"DONE ({time.time() - T0:.0f}s)")
