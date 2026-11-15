"""TYPED ENTAILMENT GRAPH FROM THE RECORDS' NAME PAIRS (15 Nov 2026; tools/ent_graph.py) - offline, label-free; labels only to score.
After Hosseini et al. (TACL 2018, 'Learning Typed Entailment Graphs'): a TYPED PREDICATE = a wording between two names with the names'
entity types; p ENTAILS q when the name pairs seen with p are mostly included in those seen with q (distributional inclusion; Weeds
precision on name pairs); edges both ways = PARAPHRASES, whose cliques are relation types. Here over the premises records of documents
0 -> 2/10 (1-8 words apart; a predicate needs 5+ distinct name pairs):
  TYPES       connected groups of predicates that include each other both ways (>= TAU), each record typed by its predicate's group;
              purity / NMI on the same 30,000-record sample as tools/rel_mtb.py (surface wordings 0.343 / 0.290; MTB key 0.574 / 0.568)
  ENTAILMENT  edges p -> q (inclusion >= TAU one way only); NEW TYPED FACTS = q(a, b) for pairs stated only with p; scored: q's commonest
              Wikidata relation (evaluation only) holds for (a, b) in that direction - against p's own (the fact already stated)
Usage: python3 tools/ent_graph.py <train_distant.json.gz>   env EG_TAU (0.3,0.5), EG_MIN (5)"""
import os, sys, types, time, collections
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import normalized_mutual_info_score
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time()
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10)); R = pm._PM['R']; ND = pm._PM['ND']; TY = pm._PM['TY']; PR = pm._PM.get('PRD') or {}
n1 = int(round(ND * 0.2)); MINP = int(os.environ.get('EG_MIN', 5)); g = np.random.RandomState(0)
early = [r for r in R if r[0] < n1 and 1 <= r[5] <= 8 and r[1] != r[2] and len(r[1]) < 40 and len(r[2]) < 40]
pred = lambda r: (r[3], TY.get(r[1], 'X'), TY.get(r[2], 'X'))
PP = collections.defaultdict(set)
for r in early: PP[pred(r)].add((r[1], r[2]))
PRED = [p for p, s in PP.items() if len(s) >= MINP]; PI = {p: i for i, p in enumerate(PRED)}
pairs = {}; ri, ci = [], []
for p in PRED:
    for ab in PP[p]: ri.append(PI[p]); ci.append(pairs.setdefault(ab, len(pairs)))
M = csr_matrix((np.ones(len(ri), np.float32), (ri, ci)), shape=(len(PRED), len(pairs))); sz = np.asarray(M.sum(1)).ravel()
O = (M @ M.T).tocoo(); keep = O.row != O.col; O = (O.row[keep], O.col[keep], O.data[keep])
inc = O[2] / sz[O[0]]                                                          # inclusion of p (row) in q (col): |P(p) & P(q)| / |P(p)|
P(f"documents < {n1}: {len(early)} records; {len(PRED)} typed predicates with {MINP}+ name pairs ({len(pairs)} pairs); {len(inc)} predicate pairs share a name pair  ({time.time() - T0:.0f}s)")
# the evaluation sample of tools/rel_mtb.py
nc = collections.Counter(n for r in early for n in (r[1], r[2])); pool = [r for r in early if nc[r[1]] >= 2 and nc[r[2]] >= 2]
S = [pool[i] for i in g.choice(len(pool), min(30000, len(pool)), replace=False)]
lab = np.array([next(iter(PR[(r[1], r[2])])) if len(PR.get((r[1], r[2]), ())) == 1 else '' for r in S]); has = lab != ''
def purity(c):
    c = np.array(c); m = has & (c >= 0); ct = collections.defaultdict(collections.Counter)
    for a, b in zip(c[m], lab[m]): ct[a][b] += 1
    return sum(v.most_common(1)[0][1] for v in ct.values()) / m.sum(), normalized_mutual_info_score(lab[m], c[m]), m.sum() / has.sum()
PREL = collections.defaultdict(collections.Counter)                           # evaluation only: a predicate's commonest Wikidata relation
for p in PRED:
    for a, b in PP[p]:
        for rr in PR.get((a, b), ()): PREL[p][rr] += 1
for tau in [float(t) for t in os.environ.get('EG_TAU', '0.3,0.5').split(',')]:
    D = {(int(a), int(b)): v for a, b, v in zip(O[0], O[1], inc)}
    par = list(range(len(PRED)))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for (a, b), v in D.items():
        if a < b and v >= tau and D.get((b, a), 0) >= tau: par[find(a)] = find(b)
    grp = [find(i) for i in range(len(PRED))]; gs = collections.Counter(grp); multi = sum(1 for c in gs.values() if c > 1)
    c = [grp[PI[pred(r)]] if pred(r) in PI else -1 for r in S]; pu, nmi, cov = purity(c)
    big = sorted(((n, k) for k, n in gs.items() if n > 1), reverse=True)[:3]
    ex = " | ".join(", ".join(f"'{PRED[i][0]}'" for i in range(len(PRED)) if grp[i] == k)[:120] for n, k in big)
    P(f"TAU {tau}: {multi} paraphrase groups of 2+ predicates ({sum(c for c in gs.values() if c > 1)} predicates), {len(gs)} types in all | TYPES on the sample: purity {pu:.3f}, NMI {nmi:.3f} "
      f"(records whose predicate has {MINP}+ pairs: {cov:.2f} of the labelled) | e.g. {ex}")
    ent = [(a, b, v) for (a, b), v in D.items() if v >= tau and D.get((b, a), 0) < tau]
    new = {}
    for a, b, v in ent:
        for ab in PP[PRED[a]] - PP[PRED[b]]:
            if ab not in new or new[ab][1] < v: new[ab] = (b, v, a)
    ok = []; base = []
    for (x, z), (q, v, p) in new.items():
        if PREL[PRED[q]]: ok.append(int(PREL[PRED[q]].most_common(1)[0][0] in PR.get((x, z), ())))
        if PREL[PRED[p]]: base.append(int(PREL[PRED[p]].most_common(1)[0][0] in PR.get((x, z), ())))
    exe = "; ".join(f"{PRED[a][1]} '{PRED[a][0]}' {PRED[a][2]} => '{PRED[b][0]}' ({v:.2f})" for a, b, v in sorted(ent, key=lambda t: -t[2])[:5])
    P(f"   ENTAILMENT edges (one way, >= {tau}): {len(ent)}; NEW TYPED FACTS q(a, b) for pairs stated only with p: {len(new)} - right (q's relation holds) "
      f"{np.mean(ok) if ok else float('nan'):.3f} (n {len(ok)}), against p's own relation on the same pairs {np.mean(base) if base else float('nan'):.3f} | e.g. {exe}")
P(f"DONE ({time.time() - T0:.0f}s)")
