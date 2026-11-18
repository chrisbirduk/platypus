"""COMPOSE, THEN CONFIRM (17 Nov 2026; tools/compose_confirm.py) - offline, distant set, LABEL-FREE throughout. No label-free reader yet
gives premises clean enough to compose over (tools/reader_diag.py: AUC 0.62 at best, composition 0.20-0.22). Instead of cleaning the
premises, compose from every co-occurrence and keep only the conclusions that LATER READING confirms (the package's predict-then-confirm and
typed-confirmation judges): does confirmation lift composed conclusions to the precision real-text inference needs?
Records of a document (mention pairs within 12 tokens in one sentence) are the premises; rules over keys (type pair + order, or type pair +
last word) are mined on the first TRAIN share of documents (chains A k1 B, B k2 C whose A-C pair is also a record there; support >= 10,
Wilson LB >= MINLB); on the later documents every chain with a kept rule whose A-C pair is NOT a record in its document is a composed
conclusion. CONFIRMED = the A-C pair is a record in some LATER document (any wording), or with a relational last word (TYPED). Precision by
the distant labels (Wikidata; evaluation only), against: every within-document pair never recorded together (base), and two-step chains
without a rule.
Usage: python3 tools/compose_confirm.py <distant.json.gz>   env CC_TRAIN (0.5) CC_MINLB (0.3)"""
import os, sys, math, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
T0 = time.time(); P = lambda s: print(s, flush=True)
TRS = float(os.environ.get('CC_TRAIN', 0.5)); MINLB = float(os.environ.get('CC_MINLB', 0.3))
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, TY, LAB, ND = pm._PM['R'], pm._PM['TY'], pm._PM['LAB'], pm._PM['ND']
cut = int(ND * TRS); GLUE = {'and', 'or', 'the', 'a', 'an', 'with', 'to'}
by_doc = collections.defaultdict(list)
for r in R: by_doc[r[0]].append(r)
first_seen = collections.defaultdict(lambda: 10 ** 9); first_rel = collections.defaultdict(lambda: 10 ** 9)   # first document stating each pair
for r in R:
    k = frozenset((r[1], r[2])); first_seen[k] = min(first_seen[k], r[0])
seen_docs = collections.defaultdict(set); rel_docs = collections.defaultdict(set)
for r in R:
    k = frozenset((r[1], r[2])); seen_docs[k].add(r[0])
    if r[4].isalpha() and r[4] not in GLUE: rel_docs[k].add(r[0])
def keys(kind, r, rev=False):
    a, b = (r[2], r[1]) if rev else (r[1], r[2])
    if kind == 'TYPES': return (TY.get(a, 'X'), TY.get(b, 'X'), 'r' if rev else 'f')
    return (TY.get(a, 'X'), r[4], TY.get(b, 'X'), 'r' if rev else 'f')
def prem(d, kind):
    out = collections.defaultdict(set)
    for r in by_doc[d]:
        if r[5] > 12: continue
        out[(r[1], r[2])].add(keys(kind, r)); out[(r[2], r[1])].add(keys(kind, r, True))
    return out
def chains(pr):
    byh = collections.defaultdict(list)
    for (a, b), ks in pr.items():
        for k in ks: byh[a].append((k, b))
    for (a, b), ks in pr.items():
        for k1 in ks:
            for k2, c in byh.get(b, ()):
                if c != a and c != b: yield a, k1, b, k2, c
def wlb(k, n, z=1.96):
    p = k / n; dd = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / dd
P(f"distant set {ND} docs, {len(R)} records; rules mined on docs < {cut}  ({time.time() - T0:.0f}s)")
for kind in ('TYPES', 'COARSE'):
    cnt = collections.Counter(); hit = collections.Counter()
    for d in range(cut):
        pr = prem(d, kind); seen = set()
        for a, k1, b, k2, c in chains(pr):
            if (a, k1, k2, c) in seen: continue
            seen.add((a, k1, k2, c)); cnt[(k1, k2)] += 1; hit[(k1, k2)] += bool(pr.get((a, c)))
    rules = {k for k, n in cnt.items() if hit[k] >= 10 and wlb(hit[k], n) >= MINLB}
    res = collections.defaultdict(list)
    for d in range(cut, ND):
        pr = prem(d, kind); ents = {x for p in pr for x in p}; out = set(); allc = set()
        for a, k1, b, k2, c in chains(pr):
            if (a, c) in pr: continue
            p_ = frozenset((a, c)); allc.add(p_)
            if (k1, k2) in rules: out.add(p_)
        for p_ in allc:
            y = p_ in LAB; later = any(x > d for x in seen_docs.get(p_, ())); later_rel = any(x > d for x in rel_docs.get(p_, ()))
            res['chain, no rule'].append(y)
            if later: res['chain, no rule, CONFIRMED later'].append(y)
            if p_ in out:
                res['COMPOSED by a rule'].append(y)
                if later: res['COMPOSED + CONFIRMED later (any wording)'].append(y)
                if later_rel: res['COMPOSED + CONFIRMED later (relational wording)'].append(y)
        el = sorted(ents); rs = np.random.RandomState(d)
        for _ in range(min(20, len(el))):
            i, j = rs.randint(len(el)), rs.randint(len(el))
            if i != j and (el[i], el[j]) not in pr: res['base: random pair in the document, never recorded together'].append(frozenset((el[i], el[j])) in LAB)
    P(f"{kind}: {len(rules)} rules")
    for k in ('base: random pair in the document, never recorded together', 'chain, no rule', 'chain, no rule, CONFIRMED later', 'COMPOSED by a rule',
              'COMPOSED + CONFIRMED later (any wording)', 'COMPOSED + CONFIRMED later (relational wording)'):
        v = res[k]; P(f"   {k:52s} precision {np.mean(v) if v else float('nan'):.3f} (n {len(v)})")
P(f"DONE ({time.time() - T0:.0f}s)")
