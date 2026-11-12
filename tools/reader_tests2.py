"""READER TESTS 2 (12 Nov 2026; tools/reader_tests2.py) - from the maths of what worked, against the Re-DocRED gold (reuses reader_tests.py):
  M1 INFORMATIVE FRAMES   rank frames by SELECTIONAL PREFERENCE STRENGTH (Resnik 1996): KL of the subject- and object-type distributions
                          given the frame from their overall distributions (types: fillers co-clustered by frame sides, k 24). Glue
                          frames ('and', ',', ':') join any types; relational frames ('was born in') constrain them. Rule mining among the
                          60 most informative frames (support 30+) against the 60 most frequent.
  M2 SUBJECT-ANCHORED     every sentence about the document's title (the title is its first filler, or it opens with frame material) gives
                          (title, frame up to each of its first 3 fillers, filler) - centre records were 0.55 precise by hand vs 0.15
  M3 LISTS AND DATES      records inside a run of 3+ fillers joined by one repeated frame are a LIST (exchangeable, not a relation): dropped;
                          a filler of digits is a date / number: kept only on title-anchored records (a date belongs to its event)
Usage: python3 tools/reader_tests2.py"""
import os
import sys
import time
import collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'reader_tests.py')).read()
exec(compile(_src[:_src.index('# ---------------- T1 REDUNDANCY')], 'reader_tests', 'exec'))
P = lambda *a: print(*a, flush=True); T0 = time.time()
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

ADJ = [(a, f, b, pv, k) for a, f, b, pv, k in BR + WR if k in (0, 1)]
# ---- types (as T2a, k 24) ----
feat = collections.Counter(); occ = collections.Counter()
for a, f, b, pv, k in ADJ: feat[('A', f)] += 1; feat[('B', f)] += 1; occ[a] += 1; occ[b] += 1
F = {x: i for i, (x, _) in enumerate(feat.most_common(3000))}; Fl = [x for x in occ if occ[x] >= 3]; Fi = {x: i for i, x in enumerate(Fl)}
cnt = collections.Counter()
for a, f, b, pv, k in ADJ:
    if a in Fi and ('A', f) in F: cnt[(Fi[a], F[('A', f)])] += 1
    if b in Fi and ('B', f) in F: cnt[(Fi[b], F[('B', f)])] += 1
r_, c_ = zip(*cnt.keys()); v = np.array(list(cnt.values()), float)
M = csr_matrix((v, (r_, c_)), shape=(len(Fl), len(F))); tot = M.sum(); rs_ = np.asarray(M.sum(1)).ravel(); cs_ = np.asarray(M.sum(0)).ravel()
M = M.tocoo(); pmi = np.log(M.data * tot / (rs_[M.row] * cs_[M.col])); kp = pmi > 0
Z = TruncatedSVD(50, random_state=0).fit_transform(csr_matrix((pmi[kp], (M.row[kp], M.col[kp])), shape=M.shape)); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
lab = KMeans(24, n_init=4, random_state=0).fit_predict(Z); TYPE = {x: f"t{lab[i]}" for x, i in Fi.items()}
ty = lambda x: 'tD' if all(w[:1].isdigit() for w in x) else TYPE.get(x, 'tU')
P(f"types ready ({time.time() - T0:.0f}s)")


def mine_report(recs, frames, tag):
    deg = collections.Counter()
    for a, f, b, *_ in recs:
        if f in frames: deg[a] += 1; deg[b] += 1
    keep = set(n for n, _ in deg.most_common(4000)); pairs = {}; nf = 0
    for a, f, b, *_ in recs:
        if f in frames and a in keep and b in keep and a != b and nf < 20000: pairs.setdefault(f, set()).add((' '.join(a), ' '.join(b))); nf += 1
    R = _rm_mine(pairs, 800, 927); D = R.get('cdiag', [])
    g = sum(1 for x in D if x[6] == -1); ok = [x for x in D if x[6] >= 3]
    def isg(a, c):
        gg = gp(tuple(a.split()), tuple(c.split())); return None if gg is None else bool(gg)
    pr = [(p[0], p[1], c) for p in R['props'] for c in (p[4] or [p[2]])]; g1 = [x for x in (isg(a, c) for a, r, c in pr) if x is not None]
    wr = [(p[0], p[1], c) for p in R['props'] if p[6] for c in p[4]]; g2 = [x for x in (isg(a, c) for a, r, c in wr) if x is not None]
    P(f"{tag}: {nf} facts, {len(R['rels'])} frames | candidates {R['cand']}: no observed chains {g}, testable {len(ok)} (functional accuracy median "
      f"{np.median([x[5] for x in ok]) if ok else float('nan'):.2f}, >= 0.5: {sum(1 for x in ok if x[5] >= 0.5)}) | RULES {len(R['rules'])}"
      + "".join(f"\n     '{t[0]}' o '{t[1]}' -> '{t[2]}' predicts {t[6]:.2f} of {t[7]}" for t in R['rules'][:10])
      + f"\n   proposals {len(pr)}, on benchmark entities {len(g1)}, gold {np.mean(g1) if g1 else float('nan'):.2f} | would be written {len(wr)}, gold "
      f"{np.mean(g2) if g2 else float('nan'):.2f} (n {len(g2)})")
    for x in sorted(ok, key=lambda x: -x[5])[:5]: P(f"     best testable: '{x[0]}' o '{x[1]}' -> '{x[2]}' predicts {x[5]:.2f} of {x[6]}")


# ---------------- M2 + M3: readers, scored against gold ----------------
def anchored(sents):
    out = []; title = None
    for si, toks in enumerate(sents):
        R = segment(toks, 300, 'B')
        if title is None and R: title = tuple(toks[R[0][0]:R[0][1]])
        if title is None: continue
        first = tuple(toks[R[0][0]:R[0][1]]) if R else None
        about = (R and R[0][0] <= 1 and (first == title or (len(first) == 1 and first[0] in title))) or (not R or R[0][0] >= 2)
        if not about: continue
        start = R[0][1] if (R and R[0][0] <= 1) else 1; js = [j for j, x in enumerate(R) if x[0] >= start][:3]
        for n, j in enumerate(js):
            b0, b1 = R[j]
            if b0 - start > 12: break
            g = fstr(toks, start, b0, [R[i] for i in js[:n]], True)
            if g: out.append((title, g, tuple(toks[b0:b1]), si, 0))
    return out


def list_runs(toks):
    """fillers inside a run of 3+ joined by one repeated frame string (a list)"""
    R = segment(toks, 300, 'B'); bad = set()
    for i in range(len(R) - 2):
        f1 = ' '.join(toks[R[i][1]:R[i + 1][0]]); f2 = ' '.join(toks[R[i + 1][1]:R[i + 2][0]])
        if f1 == f2 or (f1 in (',', 'and', ', and') and f2 in (',', 'and', ', and')): bad.update([tuple(toks[R[i][0]:R[i][1]]), tuple(toks[R[i + 1][0]:R[i + 1][1]]), tuple(toks[R[i + 2][0]:R[i + 2][1]])])
    return bad


base = [(a, f, b) for a, f, b, pv, k in BR if k in (0, 1)]
score(base, "BASELINE (adjacent + pronoun-centre records)")
anc = [(a, f, b) for d in bdocs for a, f, b, si, k in anchored(d)]
score(anc, "M2 SUBJECT-ANCHORED records only")
score(base + anc, "M2 baseline + subject-anchored")
clean = []
for di, d in enumerate(bdocs):
    lr = {si: list_runs(t) for si, t in enumerate(d)}
    for a, f, b, si, k in recs_full(d):
        if k not in (0, 1): continue
        if k == 1 and (a in lr[si] and b in lr[si]): continue                 # inside a list
        if k == 1 and (ty(a) == 'tD' or ty(b) == 'tD'): continue              # a date / number pairs with its event (the anchored record), not a neighbour
        clean.append((a, f, b))
score(clean, "M3 lists and neighbour dates dropped")
score(clean + anc, "M2 + M3 (anchored + cleaned adjacent)")

# ---------------- M1: informative frames ----------------
ALL = [(a, f, b) for a, f, b, pv, k in ADJ]
fa = collections.defaultdict(collections.Counter); fb = collections.defaultdict(collections.Counter); TA = collections.Counter(); TB = collections.Counter()
for a, f, b in ALL:
    ta, tb = ty(a), ty(b); fa[f][ta] += 1; fb[f][tb] += 1; TA[ta] += 1; TB[tb] += 1
na, nb = sum(TA.values()), sum(TB.values())
def kl(c, T, n):
    s = sum(c.values()); return sum(v / s * np.log((v / s) / (T[t] / n)) for t, v in c.items())
fc = collections.Counter(f for _, f, _ in ALL)
sps = {f: kl(fa[f], TA, na) + kl(fb[f], TB, nb) for f, c in fc.items() if c >= 30}
top_inf = [f for f in sorted(sps, key=lambda f: -sps[f])][:60]; top_freq = [f for f, _ in fc.most_common(60)]
P(f"M1 frames with 30+ records: {len(sps)} | MOST INFORMATIVE: {top_inf[:15]} | most frequent: {top_freq[:10]} | SPS of 'and' {sps.get('and', 0):.2f}, ',' {sps.get(',', 0):.2f}, "
  f"'was born in' {sps.get('was born in', float('nan')):.2f}")
mine_report(ALL, set(top_inf), "M1 RULES among the 60 MOST INFORMATIVE frames")
P(f"DONE in {time.time() - T0:.0f}s")
