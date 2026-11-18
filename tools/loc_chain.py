"""LOCATION CHAINS ON REAL TEXT (18 Nov 2026; tools/loc_chain.py) - offline, distant set, label-free (distant labels for evaluation only).
Route 1 to real-text inference: the commonest real composition is containment - 'A ... in B' and 'B , C' / 'B ... in C' with B and C places
give 'A is in C'. Records are mention pairs within LC_DIST tokens in one sentence (the package's distant reader). A record is a CONTAINMENT
record when its object is typed LOC and its link is 'in' (last word) or a bare comma ('Paris , France'). Chains A -> B -> C over the records
of ALL documents read so far (the fact table), whose A-C pair is never recorded together anywhere: precision by the distant labels against
(a) random A-C pairs of the same types never recorded together, (b) every two-record chain through a shared name (no rule), (c) the same with
B and C places but any link. Also by A's type and with the A-B link confirmed by 2+ documents.
Usage: python3 tools/loc_chain.py <distant.json.gz>   env LC_DIST (12)"""
import os, sys, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, TY, LAB, ND = pm._PM['R'], pm._PM['TY'], pm._PM['LAB'], pm._PM['ND']
DMAX = int(os.environ.get('LC_DIST', 12))
seen = set(); cont = collections.defaultdict(lambda: collections.defaultdict(set)); anyl = collections.defaultdict(set)
for r in R:
    if r[5] > DMAX or r[1] == r[2]: continue
    a, b = r[1], r[2]; seen.add(frozenset((a, b))); anyl[a].add(b); anyl[b].add(a)
    if TY.get(b) == 'LOC' and (r[4] == 'in' or r[3].strip() == ','): cont[a][b].add(r[0])
P(f"{ND} documents, {len(R)} records; containment records: {sum(len(v) for v in cont.values())} from {len(cont)} subjects  ({time.time() - T0:.0f}s)")
def prec(pairs):
    pairs = list(pairs); return (np.mean([p in LAB for p in pairs]) if pairs else float('nan')), len(pairs)
comp, comp2 = set(), set(); byty = collections.defaultdict(set)
for a, bs in cont.items():
    for b, docs_ab in bs.items():
        for c, docs_bc in cont.get(b, {}).items():
            if c in (a, b): continue
            p_ = frozenset((a, c))
            if p_ in seen: continue
            comp.add(p_); byty[TY.get(a, 'X')].add(p_)
            if len(docs_ab) >= 2: comp2.add(p_)
rs = np.random.RandomState(0); names_loc = [n for n, t in TY.items() if t == 'LOC']; subj = list(cont)
base = set()
while len(base) < 20000:
    p_ = frozenset((subj[rs.randint(len(subj))], names_loc[rs.randint(len(names_loc))]))
    if len(p_) == 2 and p_ not in seen: base.add(p_)
anyc = set()
for a in subj[:20000]:
    for b in anyl[a]:
        for c in list(anyl[b])[:50]:
            if c != a and frozenset((a, c)) not in seen: anyc.add(frozenset((a, c)))
loc2 = set()
for a in subj[:20000]:
    for b in anyl[a]:
        if TY.get(b) != 'LOC': continue
        for c in anyl[b]:
            if c != a and TY.get(c) == 'LOC' and frozenset((a, c)) not in seen: loc2.add(frozenset((a, c)))
show = lambda nm, v: P(f"   {nm:70s} precision {v[0]:.3f} (n {v[1]})")
show('base: random subject - place pair never recorded together', prec(base))
show('every two-record chain through a shared name (no rule)', prec(anyc))
show('chains through a place to a place, any link', prec(loc2))
show('CONTAINMENT CHAINS (A in/, B, B in/, C; B and C places)', prec(comp))
show('... and the A-B link in 2+ documents', prec(comp2))
for t, v in sorted(byty.items(), key=lambda x: -len(x[1])): show(f'   containment chains, A typed {t}', prec(v))
# WITHIN ONE DOCUMENT (the distant labels annotate pairs of the same document): A, B, C all in the document, the A-C pair not a record of it
byd = collections.defaultdict(list)
for r in R:
    if r[5] <= DMAX and r[1] != r[2]: byd[r[0]].append(r)
W = collections.defaultdict(list)
for d, L_ in byd.items():
    rec = {frozenset((r[1], r[2])) for r in L_}; ct = collections.defaultdict(set); al = collections.defaultdict(set); ents = {x for r in L_ for x in (r[1], r[2])}
    for r in L_:
        al[r[1]].add(r[2]); al[r[2]].add(r[1])
        if TY.get(r[2]) == 'LOC' and (r[4] == 'in' or r[3].strip() == ','): ct[r[1]].add(r[2])
    out = set(); ch = set(); rnd = set()
    for a, bs in ct.items():
        for b in bs:
            for c in ct.get(b, ()):
                if c != a and frozenset((a, c)) not in rec: out.add(frozenset((a, c)))
    for a in al:
        for b in al[a]:
            for c in al[b]:
                if c != a and frozenset((a, c)) not in rec: ch.add(frozenset((a, c)))
    el = sorted(ents); rr = np.random.RandomState(d)
    for _ in range(min(20, len(el))):
        i, j = rr.randint(len(el)), rr.randint(len(el))
        if i != j and frozenset((el[i], el[j])) not in rec: rnd.add(frozenset((el[i], el[j])))
    W['base'] += [p_ in LAB for p_ in rnd]; W['chain'] += [p_ in LAB for p_ in ch]; W['cont'] += [p_ in LAB for p_ in out]
P("WITHIN ONE DOCUMENT (the labels annotate a document's own pairs):")
for k, nm in (('base', 'base: random pair of the document never recorded together'), ('chain', 'every two-record chain in the document (no rule)'), ('cont', 'CONTAINMENT CHAINS in the document')):
    P(f"   {nm:70s} precision {np.mean(W[k]) if W[k] else float('nan'):.3f} (n {len(W[k])})")
P(f"DONE ({time.time() - T0:.0f}s)")
