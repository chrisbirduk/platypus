"""READER TESTS T1-T6 (12 Nov 2026; tools/reader_tests.py) - the fast sequence channel, scored offline against the Re-DocRED gold relations
and gold entity types, testing what the structure of language says should raise its accuracy (reuses tools/chunk_reader2.py):
  T1  REDUNDANCY     the gold rate of a filler pair against its number of INDEPENDENT sightings (distinct sentences, frames, documents) -
                     the 'urns' model: a pair stated in several places, in several ways, is more likely a real relation
  T2a TYPES          fillers co-clustered by the frames they stand in (PPMI of the filler x frame-side matrix, SVD, k-means) - purity and
                     NMI against the gold entity types (PER, LOC, ORG, TIME, MISC); selectional preferences
  T2b TYPED FRAMES   is a gold-pair record's nearest other record of the same gold relation type: frame words alone vs + the two filler types
  T5  FUNCTIONAL     frames that are single-valued (80%+ of 20+ subjects have one object): conflicting records dropped but the most sighted
  T3  DOCUMENT       the reference channel with types: the centre is the latest filler of the TITLE'S TYPE (vs the title: 444 gold)
  T6  RULES          the relational module on TYPED frames ('LOC , LOC'), adjacent and centre records only (no two-apart records - the
                     reader's own list structure), proposals scored against gold
Usage: python3 tools/reader_tests.py   (about an hour on two CPUs; prints as it goes)"""
import os
import sys
import time
import json
import collections
import traceback
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'chunk_reader2.py')).read()
exec(_src[:_src.index('\nbdocs = ')])                                       # counts, segment(), score(), ents_of, gold, fstr, _rm_mine, doc_records
T0 = time.time(); SUMMARY = {}
P = lambda *a: print(*a, flush=True)


def recs_full(sents, ref='doc', canon=True, K=300, skip=3, maxgap=12, centre_fn=None):
    """records with provenance: (a, frame, b, sentence index, k) - k 1 adjacent, 2-3 fillers that far apart, 0 via the document centre"""
    out = []; centre = None; title = None; hist = []
    for si, toks in enumerate(sents):
        R = segment(toks, K, 'B')
        if title is None and R: title = tuple(toks[R[0][0]:R[0][1]])
        c = centre_fn(title, hist) if centre_fn else title
        for k in range(1, skip + 1):
            for i in range(len(R) - k):
                (a0, a1), (b0, b1) = R[i], R[i + k]
                if 1 <= b0 - a1 <= maxgap: out.append((tuple(toks[a0:a1]), fstr(toks, a1, b0, R[i + 1:i + k], canon), tuple(toks[b0:b1]), si, k))
        if ref and c is not None and (not R or R[0][0] >= 2):
            for j in range(min(skip, len(R))):
                b0, b1 = R[j]
                if b0 - 1 > maxgap: break
                g = fstr(toks, 1, b0, R[:j], canon)
                if g and tuple(toks[b0:b1]) != c: out.append((c, g, tuple(toks[b0:b1]), si, 0))
        hist.extend(tuple(toks[x0:x1]) for x0, x1 in R)
    return out


def gp(a, b):
    ea, eb = ents_of(a), ents_of(b)
    if not ea or not eb: return None
    return [p for x in ea for y in eb if x != y and (min(x, y), max(x, y)) in gold for p in [(min(x, y), max(x, y))]]


bdocs = [[TOK.findall(s) for s in SPL.split(d)] for d in docs]
wdocs = [[TOK.findall(s) for s in SPL.split(a)] for a in re.split(r'\n = [^=\n][^\n]*? = \n', raw)]
if os.environ.get('RT_SMALL'): wdocs = wdocs[:int(os.environ['RT_SMALL'])]; bdocs = bdocs[:80]   # smoke test
BR = [(a, f, b, ('B', di, si), k) for di, d in enumerate(bdocs) for a, f, b, si, k in recs_full(d)]
WR = [(a, f, b, ('W', di, si), k) for di, d in enumerate(wdocs) for a, f, b, si, k in recs_full(d)]
P(f"records: benchmark {len(BR)}, Wikipedia {len(WR)} ({time.time() - T0:.0f}s)")

# ---------------- T1 REDUNDANCY ----------------
try:
    sight = collections.defaultdict(set); frs = collections.defaultdict(set); dcs = collections.defaultdict(set)
    for a, f, b, pv, k in BR + WR:
        key = (min(a, b), max(a, b)); sight[key].add(pv[:3]); frs[key].add(f); dcs[key].add(pv[:2])
    rows = {}
    for key in sight:
        g = gp(*key)
        if g is None: continue
        for tag, n in (('sentences', len(sight[key])), ('frames', len(frs[key])), ('documents', len(dcs[key]))):
            bk = '1' if n == 1 else '2' if n == 2 else '3-4' if n <= 4 else '5+'
            rows.setdefault(tag, collections.defaultdict(lambda: [0, 0]))[bk][0] += bool(g); rows[tag][bk][1] += 1
    P("T1 REDUNDANCY - gold rate of benchmark-entity filler pairs by independent sightings:")
    for tag in ('sentences', 'frames', 'documents'):
        P(f"   by {tag}: " + " | ".join(f"{bk}: {rows[tag][bk][0] / max(1, rows[tag][bk][1]):.2f} (n {rows[tag][bk][1]})" for bk in ('1', '2', '3-4', '5+') if rows[tag][bk][1]))
    SUMMARY['T1'] = {tag: {bk: [v[0], v[1]] for bk, v in rows[tag].items()} for tag in rows}
except Exception: P("T1 failed:\n" + traceback.format_exc())

# ---------------- T2a TYPES ----------------
TYPE = {}
try:
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score
    feat = collections.Counter(); occ = collections.Counter()
    for a, f, b, pv, k in BR + WR:
        if k != 1: continue
        feat[('A', f)] += 1; feat[('B', f)] += 1; occ[a] += 1; occ[b] += 1
    F = {x: i for i, (x, _) in enumerate(feat.most_common(3000))}; Fl = [x for x in occ if occ[x] >= 3]; Fi = {x: i for i, x in enumerate(Fl)}
    cnt = collections.Counter()
    for a, f, b, pv, k in BR + WR:
        if k != 1: continue
        if a in Fi and ('A', f) in F: cnt[(Fi[a], F[('A', f)])] += 1
        if b in Fi and ('B', f) in F: cnt[(Fi[b], F[('B', f)])] += 1
    r_, c_ = zip(*cnt.keys()); v = np.array(list(cnt.values()), float)
    M = csr_matrix((v, (r_, c_)), shape=(len(Fl), len(F))); tot = M.sum(); rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel()
    M = M.tocoo(); pmi = np.log(M.data * tot / (rs[M.row] * cs[M.col])); keep = pmi > 0
    PP = csr_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=M.shape)
    Z = TruncatedSVD(50, random_state=0).fit_transform(PP); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    gt = {}
    for nm, i in Fi.items():
        es = ents_of(nm)
        if len(es) == 1:
            e = next(iter(es))
            if G['ent_name'].get(str(e)) == ' '.join(nm) and G['ent_type'].get(str(e)) in ('PER', 'LOC', 'ORG', 'TIME', 'MISC'): gt[i] = G['ent_type'][str(e)]
    idx = sorted(gt); ys = [gt[i] for i in idx]; maj = collections.Counter(ys).most_common(1)[0][1] / len(ys)
    P(f"T2a TYPES: {len(Fl)} fillers seen 3+ times, {len(F)} frame-side features; {len(idx)} fillers that name exactly one benchmark entity | majority type {maj:.2f}")
    best = None
    for kk in (6, 12, 24):
        lab = KMeans(kk, n_init=4, random_state=0).fit_predict(Z); L = lab[idx]
        pur = sum(collections.Counter(np.array(ys)[L == c]).most_common(1)[0][1] for c in set(L)) / len(ys)
        nmi = normalized_mutual_info_score(ys, L)
        rl = np.random.RandomState(0).permutation(L); rn = normalized_mutual_info_score(ys, rl)
        P(f"   k {kk}: PURITY {pur:.2f} | NMI {nmi:.2f} (shuffled clusters {rn:.2f})")
        comp = collections.defaultdict(collections.Counter)
        for i, y in zip(idx, ys): comp[lab[i]][y] += 1
        P("      clusters (gold types): " + " | ".join(f"{c}: " + ", ".join(f"{t} {n}" for t, n in comp[c].most_common(2)) for c in sorted(comp)[:12]))
        SUMMARY.setdefault('T2a', {})[kk] = [pur, nmi, rn]
        if kk == 12: TYPE = {x: f"t{lab[i]}" for x, i in Fi.items()}
    SUMMARY['T2a_majority'] = maj
except Exception: P("T2a failed:\n" + traceback.format_exc())
ty = lambda x: TYPE.get(x, 'tU')

# ---------------- T2b TYPED FRAMES ----------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    rel = {}
    for h, t, fl, ps in G['rel']: rel.setdefault((min(h, t), max(h, t)), set()).update(ps)
    ex = []
    for a, f, b, pv, k in BR:
        if k not in (0, 1): continue
        g = gp(a, b)
        if g: ex.append((f, ty(a), ty(b), set().union(*[rel[x] for x in g])))
    def knn(X):
        S = X @ X.T; np.fill_diagonal(S, -9); nn = S.argmax(1); return float(np.mean([bool(ex[i][3] & ex[j][3]) for i, j in enumerate(nn)]))
    bow = TfidfVectorizer(token_pattern=r'[^ ]+').fit_transform([e[0] for e in ex]).toarray(); bow /= np.linalg.norm(bow, axis=1, keepdims=True) + 1e-9
    tl = sorted({e[1] for e in ex} | {e[2] for e in ex}); T = np.zeros((len(ex), 2 * len(tl)))
    for i, e in enumerate(ex): T[i, tl.index(e[1])] = 1; T[i, len(tl) + tl.index(e[2])] = 1
    T /= np.sqrt(2); both = np.concatenate([bow, T], 1); both /= np.linalg.norm(both, axis=1, keepdims=True)
    Pc = collections.Counter(p for e in ex for p in e[3]); maj = max(np.mean([p in e[3] for e in ex]) for p in Pc)
    a1, a2, a3 = knn(bow), knn(T + 1e-6 * np.random.RandomState(0).rand(*T.shape)), knn(both)
    P(f"T2b TYPED FRAMES ({len(ex)} gold-pair records): nearest other record shares a gold relation type: majority {maj:.2f} | frame words {a1:.2f} | "
      f"filler types only {a2:.2f} | FRAME WORDS + TYPES {a3:.2f}  (frame probe: connector words 0.44)")
    SUMMARY['T2b'] = [maj, a1, a2, a3, len(ex)]
except Exception: P("T2b failed:\n" + traceback.format_exc())

# ---------------- T5 FUNCTIONAL FILTER ----------------
try:
    so = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for a, f, b, pv, k in BR + WR:
        if k == 1: so[f][a][b] += 1
    func = set()                                                               # judged on subjects seen 2+ times in the frame (most fillers occur
    for f, d in so.items():                                                    #   once, which would make every frame look single-valued)
        m = [len(v) == 1 for v in d.values() if sum(v.values()) >= 2]
        if len(m) >= 20 and np.mean(m) >= 0.8: func.add(f)
    def kept(a, f, b): return f not in func or len(so[f][a]) <= 1 or so[f][a].most_common(1)[0][0] == b
    base = [(a, f, b) for a, f, b, pv, k in BR]; filt = [(a, f, b) for a, f, b, pv, k in BR if kept(a, f, b)]
    P(f"T5 FUNCTIONAL FILTER: {len(func)} single-valued frames (e.g. {sorted(func, key=lambda f: -len(so[f]))[:6]}); benchmark records {len(base)} -> {len(filt)}")
    n0 = score(base, "   before"); n1 = score(filt, "   after ")
    SUMMARY['T5'] = [list(n0), list(n1), len(func)]
except Exception: P("T5 failed:\n" + traceback.format_exc())

# ---------------- T3 TYPED DOCUMENT CHANNEL ----------------
try:
    def same_type(title, hist):
        if title is None: return None
        t = ty(title)
        for x in reversed(hist):
            if t != 'tU' and ty(x) == t: return x
        return title
    for tag, fn in (('title (444 before)', None), ('latest filler of the title type', same_type)):
        recs = [(a, f, b) for d in bdocs for a, f, b, si, k in recs_full(d, centre_fn=fn)]
        SUMMARY.setdefault('T3', {})[tag] = list(score(recs, f"T3 DOCUMENT CENTRE = {tag}"))
except Exception: P("T3 failed:\n" + traceback.format_exc())

# ---------------- T6 RULES ON TYPED FRAMES ----------------
try:
    for tag, red in (('typed frames, adjacent + centre records', False), ('the same, pairs sighted in 2+ sentences', True)):
        t1 = time.time(); allr = [(a, f, b) for a, f, b, pv, k in BR + WR if k in (0, 1) and ty(a) != 'tU' and ty(b) != 'tU'
                                  and (not red or len(sight[(min(a, b), max(a, b))]) >= 2)]
        tf = [(a, f"{ty(a)} {f} {ty(b)}", b) for a, f, b in allr]
        fc = collections.Counter(f for _, f, _ in tf); top = set(f for f, _ in fc.most_common(60)); deg = collections.Counter()
        for a, f, b in tf:
            if f in top: deg[a] += 1; deg[b] += 1
        keep = set(n for n, _ in deg.most_common(4000)); pairs = {}; nf = 0
        for a, f, b in tf:
            if f in top and a in keep and b in keep and a != b and nf < 20000: pairs.setdefault(f, set()).add((' '.join(a), ' '.join(b))); nf += 1
        R = _rm_mine(pairs, 800, 921)
        def isg(a, c):
            g = gp(tuple(a.split()), tuple(c.split())); return None if g is None else bool(g)
        pr = [(p[0], p[1], c) for p in R['props'] for c in (p[4] or [p[2]])]; g1 = [g for g in (isg(a, c) for a, r, c in pr) if g is not None]
        wr = [(p[0], p[1], c) for p in R['props'] if p[6] for c in p[4]]; g2 = [g for g in (isg(a, c) for a, r, c in wr) if g is not None]
        P(f"T6 RULES ({tag}): {nf} facts, {R['N']} fillers, {len(R['rels'])} typed frames | {R['cand']} beat the null | RULES {len(R['rules'])}" +
          "".join(f"\n     '{t[0]}' o '{t[1]}' -> '{t[2]}' predicts {t[6]:.2f} of {t[7]}" for t in R['rules'][:10]) +
          f"\n   PROPOSALS {len(pr)}, on benchmark entities {len(g1)}, gold {np.mean(g1) if g1 else float('nan'):.2f} | would be written {len(wr)}, "
          f"on benchmark entities {len(g2)}, gold {np.mean(g2) if g2 else float('nan'):.2f} | {time.time() - t1:.0f}s")
        SUMMARY.setdefault('T6', {})[tag] = dict(rules=[list(t[:3]) + [t[6], t[7]] for t in R['rules']], props=len(pr), pg=[len(g1), float(np.mean(g1)) if g1 else None],
                                               wr=len(wr), wg=[len(g2), float(np.mean(g2)) if g2 else None])
except Exception: P("T6 failed:\n" + traceback.format_exc())

json.dump(SUMMARY, open(os.environ.get('READER_TESTS_JSON', '/tmp/reader_tests.json'), 'w'), default=str, indent=1)
P(f"ALL DONE in {time.time() - T0:.0f}s")
