"""ENTITY IDENTITY CHANNEL (12 Nov 2026; tools/entity_identity.py): do real-text facts form chains once name variants are merged?

tools/rule_diag.py: of 2,009 candidate rules on the fast channel's real-text records, 1,491 had no observed two-hop chain - each filler
sits in few records and one entity appears under several names ('Marie Curie', 'Curie'; 'Fredrik Laurentz Ekman', 'Ekman'). Merging:
  DOC     within a document, a filler whose words all occur, in order, at the start or end of a LONGER filler of the same document is that
          filler ('Curie' -> 'Marie Curie') - head matching, the commonest coreference in encyclopedic text
  GLOBAL  + across the corpus, a one-word filler is the multi-word filler that most of its multi-word occurrences belong to (3+ of them, 60%+)
Measured before and after: fillers, records per filler, MIDDLE fillers (both a subject and an object - the joints of two-hop chains),
two-hop paths, gold coverage, and the relational module's rule diagnosis (as tools/rule_diag.py). Usage: python3 tools/entity_identity.py"""
import os
import sys
import collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'reader_tests.py')).read()
exec(compile(_src[:_src.index('# ---------------- T1 REDUNDANCY')], 'reader_tests', 'exec'))
P = lambda *a: print(*a, flush=True)
REC = [(a, f, b, pv, k) for a, f, b, pv, k in BR + WR if k in (0, 1)]       # adjacent and centre records (no two-apart records)


def sub_of(s, L):
    n = len(s)
    return n < len(L) and (L[:n] == s or L[-n:] == s)


def doc_map(recs):
    by = collections.defaultdict(collections.Counter)
    for a, f, b, pv, k in recs: by[pv[:2]][a] += 1; by[pv[:2]][b] += 1
    M = {}
    for d, cnt in by.items():
        longs = sorted((x for x in cnt if len(x) >= 2), key=lambda x: -cnt[x])
        for x in cnt:
            c = [L for L in longs if sub_of(x, L)]
            if c: M[(d, x)] = c[0]
    return M


def glob_map(recs):
    occ = collections.Counter(); multi = collections.defaultdict(collections.Counter)
    for a, f, b, pv, k in recs:
        for x in (a, b):
            occ[x] += 1
            if len(x) >= 2:
                for w in (x[0], x[-1]): multi[(w,)][x] += 1
    M = {}
    for w, c in multi.items():
        tot = sum(c.values()); top, n = c.most_common(1)[0]
        if n >= 3 and n >= 0.6 * tot: M[w] = top
    return M


def apply(recs, dm=None, gm=None):
    out = []
    for a, f, b, pv, k in recs:
        a2 = dm.get((pv[:2], a), a) if dm else a; b2 = dm.get((pv[:2], b), b) if dm else b
        if gm: a2 = gm.get(a2, a2); b2 = gm.get(b2, b2)
        if a2 != b2: out.append((a2, f, b2, pv, k))
    return out


def stats(recs, tag):
    subj = collections.Counter(a for a, *_ in recs); obj = collections.Counter(b for _, _, b, *_ in recs)
    nodes = set(subj) | set(obj); mid = [x for x in nodes if subj[x] and obj[x]]
    paths = sum(obj[x] * subj[x] for x in mid)
    bm = [(a, f, b) for a, f, b, pv, k in recs if pv[0] == 'B']
    P(f"{tag}: {len(recs)} records, {len(nodes)} fillers, records per filler {2 * len(recs) / max(1, len(nodes)):.2f}, fillers in 2+ records "
      f"{sum(1 for x in nodes if subj[x] + obj[x] >= 2) / max(1, len(nodes)):.2f} | MIDDLE fillers (subject and object) {len(mid)} | two-hop paths {paths}")
    score(bm, "   benchmark records")


def mine(recs, tag):
    fc = collections.Counter(f for _, f, _, _, _ in recs); top = set(f for f, _ in fc.most_common(60)); deg = collections.Counter()
    for a, f, b, pv, k in recs:
        if f in top: deg[a] += 1; deg[b] += 1
    keep = set(n for n, _ in deg.most_common(4000)); pairs = {}; nf = 0
    for a, f, b, pv, k in recs:
        if f in top and a in keep and b in keep and a != b and nf < 20000: pairs.setdefault(f, set()).add((' '.join(a), ' '.join(b))); nf += 1
    R = _rm_mine(pairs, 800, 925); D = R.get('cdiag', [])
    g = sum(1 for x in D if x[6] == -1); lt3 = sum(1 for x in D if 0 <= x[6] < 3); ok = [x for x in D if x[6] >= 3]
    def isg(a, c):
        gg = gp(tuple(a.split()), tuple(c.split())); return None if gg is None else bool(gg)
    pr = [(p[0], p[1], c) for p in R['props'] for c in (p[4] or [p[2]])]; g1 = [x for x in (isg(a, c) for a, r, c in pr) if x is not None]
    P(f"{tag} RULE DIAGNOSIS: {nf} facts, {R['N']} fillers | candidates {R['cand']}: no observed chains {g}, under 3 testable {lt3}, testable {len(ok)} "
      f"(functional accuracy median {np.median([x[5] for x in ok]) if ok else float('nan'):.2f}, >= 0.5: {sum(1 for x in ok if x[5] >= 0.5)}) | RULES {len(R['rules'])}"
      + "".join(f"\n     '{t[0]}' o '{t[1]}' -> '{t[2]}' predicts {t[6]:.2f} of {t[7]}" for t in R['rules'][:8])
      + f"\n   proposals {len(pr)}, on benchmark entities {len(g1)}, gold {np.mean(g1) if g1 else float('nan'):.2f}")
    for x in sorted(ok, key=lambda x: -x[5])[:6]: P(f"     best testable: '{x[0]}' o '{x[1]}' -> '{x[2]}' predicts {x[5]:.2f} of {x[6]}")


stats(REC, "AS READ")
dm = doc_map(REC); R1 = apply(REC, dm); P(f"DOC merges: {len(dm)} (filler, document) mapped, e.g. {[(k[1], v) for k, v in list(dm.items())[:5]]}")
stats(R1, "DOC")
gm = glob_map(R1); R2 = apply(R1, None, gm); P(f"GLOBAL merges: {len(gm)}, e.g. {list(gm.items())[:6]}")
stats(R2, "DOC + GLOBAL")
mine(R2, "DOC + GLOBAL")
P("DONE")
