"""INFERENCE AT READING TIME, WITHOUT RULES (13 Nov 2026; tools/integrate_test.py) - offline, no effect on the package.
Brain research (integrative encoding: Shohamy & Wagner 2008; Zeithamova, Schlichting & Preston 2012; REMERGE, Kumaran & McClelland 2012):
reading B-C reactivates the stored A-B and binds A-C - no rule, no relation label. Here the store holds every name pair in a sentence of the
Re-DocRED train + dev documents (gold mentions as names); when a sentence of one document joins B and C, the store's records joining A and B
from OTHER documents are recalled and A-C is proposed. Checked against gold: A-C is a gold relation pair (any type) where Re-DocRED annotates
it (A and C named in one document; elsewhere unverifiable). By what the store can know, without labels:
  number of distinct bridges B supporting A-C | how rare the bridge is (documents naming it) | how close the names sit in both records |
  the learned extractor's score of both records (tools/consistency_teacher.py, label-free) | CEILING: both premise records gold-related
Base rate: name pairs that co-occur in a document, gold-related.
Usage: python3 tools/integrate_test.py"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_ct = open(os.path.join(HERE, 'consistency_teacher.py')).read()
exec(compile(_ct[:_ct.index('# ---- SELF-TRAINING ROUNDS')], 'consistency_teacher', 'exec'))
lr_ = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, teacher_tr >= np.percentile(teacher_tr, 75))
s_all = lr_.decision_function(feats(np.arange(len(C))))
GOLDP = set(); ANN = set(); dfreq = collections.Counter()
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    for l in d['labels']: GOLDP.add(frozenset((cn[l['h']], cn[l['t']])))
    ns = set(cn); ANN |= {frozenset((a, b)) for a in ns for b in ns if a != b}
    for n in ns: dfreq[n] += 1
base = np.mean([p in GOLDP for p in ANN])
P(f"base rate: name pairs co-occurring in a document that are gold-related {base:.3f} ({len(ANN)} pairs)")
# the store: for each name, its records (partner, document, record index)
nb = collections.defaultdict(list)
for i, c in enumerate(C):
    nb[c['cA']].append((c['cB'], c['di'], i)); nb[c['cB']].append((c['cA'], c['di'], i))
# reading: each record B-C recalls the A-B records of OTHER documents through the shared name B (both orientations)
PROP = collections.defaultdict(lambda: dict(bridges=set(), rec=[], docs=set()))
for i, c in enumerate(C):
    for B, Cn in ((c['cA'], c['cB']), (c['cB'], c['cA'])):
        if dfreq[B] > 200: continue                                           # hub names (countries, years) recalled thousands of times: skipped
        for A, dA, j in nb[B]:
            if dA == c['di'] or A == Cn: continue
            k = frozenset((A, Cn)); p = PROP[k]; p['bridges'].add(B); p['rec'].append((j, i)); p['docs'] |= {dA, c['di']}
keys = [k for k in PROP if k in ANN]                                          # verifiable: Re-DocRED annotates the pair somewhere
yk = np.array([k in GOLDP for k in keys])
P(f"proposals A-C through a shared name in another document: {len(PROP)} | verifiable {len(keys)} | gold {yk.mean():.3f} (base {base:.3f})")
def rep(tag, sc):
    o = np.argsort(-sc + 1e-9 * np.random.RandomState(5).rand(len(sc))); k1, k2 = max(1, len(sc) // 10), max(1, len(sc) // 4)
    P(f"   {tag:58s} AUC {auc(sc[yk], sc[~yk]):.3f} | top 10%: gold {yk[o[:k1]].mean():.3f} | top 25%: gold {yk[o[:k2]].mean():.3f}")
nbr = np.array([len(PROP[k]['bridges']) for k in keys], float)
rare = np.array([-min(dfreq[b] for b in PROP[k]['bridges']) for k in keys], float)
close = np.array([-min(len(C[j]['mid']) + len(C[i]['mid']) for j, i in PROP[k]['rec']) for k in keys], float)
ext = np.array([max(min(s_all[j], s_all[i]) for j, i in PROP[k]['rec']) for k in keys])
both = np.array([float(any(y[j] and y[i] for j, i in PROP[k]['rec'])) for k in keys])
rep('number of distinct bridges', nbr); rep('rarest bridge (fewest documents naming it)', rare); rep('closeness in both records', close)
rep('learned extractor, both records (label-free)', ext)
rk = lambda x: np.argsort(np.argsort(x)) / len(x)
rep('bridges + rare bridge + extractor (rank sum)', rk(nbr) + rk(rare) + rk(ext))
P(f"   CEILING - both premise records gold-related: {int(both.sum())} proposals, gold {yk[both > 0].mean() if both.any() else float('nan'):.3f}; "
  f"neither: gold {yk[both == 0].mean():.3f}")
for nmin in (2, 3, 5):
    m = nbr >= nmin; P(f"   proposals with >= {nmin} distinct bridges: {int(m.sum())}, gold {yk[m].mean() if m.any() else float('nan'):.3f}")
m = (nbr >= 2) & (ext >= np.percentile(ext, 50)); P(f"   >= 2 bridges AND extractor in the top half: {int(m.sum())}, gold {yk[m].mean() if m.any() else float('nan'):.3f}")
P(f"DONE ({time.time() - T0:.0f}s)")
