"""READER TESTS 3 (12 Nov 2026; tools/reader_tests3.py) - tests 4-6 from the maths of what worked, on the SUBJECT-ANCHORED reader's records
(M2, 0.46 gold rate) plus the cleaned adjacent records (M3), local Wikipedia + the benchmark, scored against the Re-DocRED gold:
  M5 RELATION PROPERTIES AS FILTERS  each frame's TYPE SIGNATURE (its filler-type pairs; a record whose type pair has < 5% of its frame's
                          mass is dropped), FUNCTIONAL frames (one object per subject: keep the most sighted), and IRREFLEXIVITY
                          (a filler that is a fragment of the other names the same thing: 'Margaret Thatcher' / 'Thatcher')
  M4 TENSOR FACTORISATION a code per filler and an operator per frame (the relational module's fit, _rm_fit); each record scored by how
                          well the fit explains it (rank of its object among its document's fillers): a FILTER (keep the best-explained
                          records) and COMPLETION (each document title's best new object per frame, among the document's fillers);
                          without constraints, with RULE CONSTRAINTS mined from the data (inverse frames: 'A f B' <=> 'B g A'; paraphrase
                          frames: the same pairs), and with frames TIED to their pooled cluster (M6). Baselines: random drop / random filler
  M6 RULE POOLING         frames pooled into clusters (frame words + filler-type pairs + shared filler pairs, k-means 40); rules mined
                          (_rm_mine, every guard) on the anchored frames as read, on the pooled clusters, and on cluster x type pair
Usage: python3 tools/reader_tests3.py"""
import os
import sys
import time
import collections
import numpy as np
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'reader_tests2.py')).read()
exec(compile(_src[:_src.index('# ---------------- M1: informative frames')], 'reader_tests2', 'exec'))   # types (ty), anchored(), list_runs(), M2/M3 lines
import torch.nn.functional as F  # noqa: E402 (reader_tests2 uses F for its feature index)
torch.set_num_threads(1)
P = lambda *a: print(*a, flush=True)
T0 = time.time()


def doc_recs(d):
    """anchored records + cleaned adjacent records of one document, and its fillers"""
    lr = {si: list_runs(t) for si, t in enumerate(d)}; out = [(a, f, b, si, 'anc') for a, f, b, si, k in anchored(d)]
    for a, f, b, si, k in recs_full(d):
        if k not in (0, 1): continue
        if k == 1 and (a in lr[si] and b in lr[si]): continue
        if k == 1 and (ty(a) == 'tD' or ty(b) == 'tD'): continue
        out.append((a, f, b, si, 'adj' if k == 1 else 'ctr'))
    fil = set()
    for t in d:
        for x0, x1 in segment(t, 300, 'B'): fil.add(tuple(t[x0:x1]))
    title = next((a for a, f, b, si, k in out if k == 'anc'), None)
    return out, fil, title


DOCS = []                                                                  # (source, doc index, records, fillers, title)
for src, dd in (('B', bdocs), ('W', wdocs)):
    for di, d in enumerate(dd):
        r, fl, tt = doc_recs(d); DOCS.append((src, di, r, fl, tt))
REC = [(a, f, b, (s, di, si), k) for s, di, r, fl, tt in DOCS for a, f, b, si, k in r if a != b]
BREC = [x for x in REC if x[3][0] == 'B']
P(f"M2 + M3 records: {len(REC)} (benchmark {len(BREC)}) from {len(DOCS)} documents ({time.time() - T0:.0f}s)")
n0 = score([(a, f, b) for a, f, b, pv, k in BREC], "REFERENCE: M2 + M3 benchmark records (all frames)")


def gold_of(recs):
    """per record: None (not both benchmark entities) / True / False"""
    out = []
    for a, f, b in recs:
        g = gp(a, b); out.append(None if g is None else bool(g))
    return out


def rate(gs, keep):
    k = [g for g, m in zip(gs, keep) if m and g is not None]; return (np.mean(k) if k else float('nan')), len(k)


# ---------------- M5 RELATION PROPERTIES AS FILTERS ----------------
fsig = collections.defaultdict(collections.Counter); fn = collections.Counter()
for a, f, b, pv, k in REC: fsig[f][(ty(a), ty(b))] += 1; fn[f] += 1
def sig_ok(a, f, b):
    if fn[f] < 20: return True
    return fsig[f][(ty(a), ty(b))] / fn[f] >= 0.05
so = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for a, f, b, pv, k in REC: so[f][a][b] += 1
func = set()
for f, d in so.items():
    m = [len(v) == 1 for v in d.values() if sum(v.values()) >= 2]
    if len(m) >= 10 and np.mean(m) >= 0.7: func.add(f)
def func_ok(a, f, b): return f not in func or len(so[f][a]) <= 1 or so[f][a].most_common(1)[0][0] == b
def irr_ok(a, f, b): return not (set(a) <= set(b) or set(b) <= set(a))
P(f"M5: frames with 20+ records {sum(1 for f in fn if fn[f] >= 20)}; FUNCTIONAL frames {len(func)} (e.g. {sorted(func, key=lambda f: -len(so[f]))[:8]})")
B3 = [(a, f, b) for a, f, b, pv, k in BREC]; GB = gold_of(B3)
for tag, fnc in (('type signature', lambda r: sig_ok(*r)), ('functional', lambda r: func_ok(*r)), ('irreflexive (name fragments)', lambda r: irr_ok(*r)),
                 ('ALL THREE', lambda r: sig_ok(*r) and func_ok(*r) and irr_ok(*r))):
    keep = [fnc(r) for r in B3]; score([r for r, m in zip(B3, keep) if m], f"M5 {tag}: kept {sum(keep)} of {len(B3)}")
    rr = np.random.RandomState(0).rand(len(B3)) < np.mean(keep); P(f"      random drop of the same share: gold rate {rate(GB, rr)[0]:.2f}")
KEEP5 = [sig_ok(a, f, b) and func_ok(a, f, b) and irr_ok(a, f, b) for a, f, b, pv, k in REC]

# ---------------- frame clusters (for M6 and M4's tying) ----------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
FR = [f for f, c in fn.most_common() if c >= 5]; Fi = {f: i for i, f in enumerate(FR)}
Xw = TfidfVectorizer(token_pattern=r'[^ ]+').fit_transform(FR).toarray(); Xw /= np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-9
tps = sorted({t for f in FR for t in [x for p in fsig[f] for x in p]}); Xt = np.zeros((len(FR), len(tps) * 2))
for f in FR:
    for (ta, tb), c in fsig[f].items(): Xt[Fi[f], tps.index(ta)] += c; Xt[Fi[f], len(tps) + tps.index(tb)] += c
Xt /= np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-9
X = np.concatenate([Xw, 0.7 * Xt], 1); X /= np.linalg.norm(X, axis=1, keepdims=True)
KC = min(40, max(2, len(FR) // 3)); CL = KMeans(KC, n_init=4, random_state=0).fit_predict(X); CLU = {f: int(CL[Fi[f]]) for f in FR}
cm = collections.defaultdict(list)
for f in FR: cm[CLU[f]].append(f)
P(f"frame clusters: {len(FR)} frames with 5+ records into {KC} | e.g. " + " || ".join(", ".join(repr(x) for x in cm[c][:4]) for c in sorted(cm, key=lambda c: -len(cm[c]))[:6]))

# ---------------- M4 TENSOR FACTORISATION ----------------
names = sorted({a for a, f, b, pv, k in REC} | {b for a, f, b, pv, k in REC}); Ni = {n: i for i, n in enumerate(names)}
RF = FR + ['<rare>']; Ri = {f: i for i, f in enumerate(RF)}
rid = lambda f: Ri.get(f, len(RF) - 1)
rows = [(rid(f), Ni[a], Ni[b]) for a, f, b, pv, k in REC]
pairs = collections.defaultdict(set)
for a, f, b, pv, k in REC:
    if f in Fi: pairs[f].add((a, b))
INV, PARA = [], []                                                         # constraints mined from the data (length-1 rules)
fl = [f for f in FR if len(pairs[f]) >= 3]
for i, f in enumerate(fl):
    for g in fl[i + 1:]:
        s1 = len(pairs[f] & pairs[g]); s2 = len(pairs[f] & {(b, a) for a, b in pairs[g]}); mn = min(len(pairs[f]), len(pairs[g]))
        if s1 >= 3 and s1 / mn >= 0.3: PARA.append((Ri[f], Ri[g]))
        if s2 >= 3 and s2 / mn >= 0.3: INV.append((Ri[f], Ri[g]))
P(f"M4: {len(names)} fillers, {len(RF)} frames, {len(rows)} records | constraints mined: paraphrase pairs {len(PARA)}, inverse pairs {len(INV)} "
  f"(e.g. {[(RF[x], RF[y]) for x, y in INV[:4]]}, {[(RF[x], RF[y]) for x, y in PARA[:4]]})")
subj_of = collections.defaultdict(list)
for r, a, b in rows: subj_of[r].append(a)


def fit(steps, seed, D=64, cons=False, tie=False, w=0.5):
    g = torch.Generator().manual_seed(seed); rr = np.random.RandomState(seed); N, R = len(names), len(RF)
    E = torch.nn.Parameter(torch.randn(N, D, generator=g) * 0.1)
    Pm = torch.nn.Parameter(torch.eye(D).repeat(R, 1, 1) + 0.01 * torch.randn(R, D, D, generator=g))
    opt = torch.optim.Adam([E, Pm], lr=3e-3); T = torch.tensor(rows, dtype=torch.long)
    cl = torch.tensor([CLU.get(f, KC) for f in RF]); I = torch.eye(D)
    for st in range(steps):
        b = T[torch.from_numpy(rr.randint(len(rows), size=256))]; z = F.normalize(E, dim=-1)
        pred = F.normalize(torch.einsum('bd,bde->be', z[b[:, 1]], Pm[b[:, 0]]), dim=-1)
        zc = z - z.mean(0); cov = zc.t() @ zc / N
        loss = F.cross_entropy(pred @ z.t() / 0.1, b[:, 2]) + 0.1 * ((cov * N / D - I) ** 2).mean()
        if cons and (INV or PARA):                                         # rules on the subjects' codes: f then g returns to the subject (inverse);
            L = []                                                         #   f and g map a subject to the same place (paraphrase)
            for (f1, g1), kind in [(x, 'i') for x in INV] + [(x, 'p') for x in PARA]:
                s = torch.tensor(rr.choice(subj_of[f1], size=min(16, len(subj_of[f1]))), dtype=torch.long); zs = z[s]
                p1 = F.normalize(zs @ Pm[f1], dim=-1)
                if kind == 'i': L.append(1 - F.cosine_similarity(F.normalize(p1 @ Pm[g1], dim=-1), zs, dim=-1).mean())
                else: L.append(1 - F.cosine_similarity(p1, F.normalize(zs @ Pm[g1], dim=-1), dim=-1).mean())
            loss = loss + w * torch.stack(L).mean()
        if tie:                                                            # pooling: each frame's operator pulled toward its cluster's mean
            M = torch.zeros(KC + 1, D, D).index_add_(0, cl, Pm) / torch.bincount(cl, minlength=KC + 1).clamp(min=1).view(-1, 1, 1).float()
            loss = loss + w * ((Pm[:-1] - M[cl[:-1]]) ** 2).mean() * D
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 500 == 0: P(f"      step {st}: loss {float(loss.detach()):.3f}")
    with torch.no_grad(): return F.normalize(E.detach(), dim=-1), Pm.detach()


DOCF = {(s, di): [Ni[x] for x in fl_ if x in Ni] for s, di, r, fl_, tt in DOCS}
rB = [(rid(f), Ni[a], Ni[b], pv) for a, f, b, pv, k in BREC]
GBR = gold_of([(a, f, b) for a, f, b, pv, k in BREC])
for tag, cons, tie in (('no constraints', False, False), ('+ RULE CONSTRAINTS (inverse, paraphrase)', True, False), ('+ constraints + TIED TO CLUSTERS', True, True)):
    t1 = time.time(); P(f"M4 fit, {tag}:"); Z, PM = fit(int(os.environ.get('RT_STEPS', 2000)), 0, cons=cons, tie=tie)
    # filter: rank of the record's object among its document's fillers
    rk = []
    with torch.no_grad():
        for r, a, b, pv in rB:
            c = DOCF[pv[:2]]; c = [x for x in c if x != a] or [b]
            if b not in c: c = c + [b]
            s = F.normalize(Z[a] @ PM[r], dim=-1) @ Z[torch.tensor(c)].t(); rk.append(float((s > s[c.index(b)]).float().mean()))
    rk = np.array(rk)
    for q in (0.75, 0.5, 0.25):
        keep = rk <= np.quantile(rk, q); score([(a, f, b) for (a, f, b, pv, k), m in zip(BREC, keep) if m], f"   FILTER keep best-explained {q:.0%}")
    # completion: each title's best NEW object per frame among the document's fillers
    seen = set((a, b) for r, a, b, pv in rB) | set((b, a) for r, a, b, pv in rB)
    tf = collections.defaultdict(set)
    for (r, a, b, pv), (a_, f_, b_, pv_, k_) in zip(rB, BREC):
        if k_ == 'anc' and f_ in Fi and fn[f_] >= 20: tf[(pv[:2], a)].add(r)
    new, rnd = [], []; rs = np.random.RandomState(1)
    with torch.no_grad():
        for (dk, a), rs_ in tf.items():
            c = [x for x in DOCF[dk] if x != a and (a, x) not in seen]
            if not c: continue
            for r in rs_:
                s = F.normalize(Z[a] @ PM[r], dim=-1) @ Z[torch.tensor(c)].t(); x = c[int(s.argmax())]
                new.append((names[a], RF[r], names[x])); rnd.append((names[a], RF[r], names[c[rs.randint(len(c))]]))
    new = list({(a, b): (a, f, b) for a, f, b in new}.values()); rnd = list({(a, b): (a, f, b) for a, f, b in rnd}.values())
    score(new, f"   COMPLETION, the fit's best new object per (title, frame)")
    score(rnd, f"   completion baseline, a random new document filler")
    score([(a, f, b) for a, f, b, pv, k in BREC] + new, f"   M2 + M3 + completion")
    P(f"   ({time.time() - t1:.0f}s)")

# ---------------- M6 RULE POOLING ----------------
A3 = [(a, f, b) for a, f, b, pv, k in REC]
mine_report(A3, set(f for f, _ in fn.most_common(60)), "M6a RULES, anchored + cleaned records, the 60 commonest frames as read")
C3 = [(a, f"c{CLU[f]}", b) for a, f, b in A3 if f in CLU]
mine_report(C3, set(f"c{c}" for c in range(KC)), "M6b RULES, frames POOLED into 40 clusters")
CT = [(a, f"c{CLU[f]} {ty(a)} {ty(b)}", b) for a, f, b in A3 if f in CLU and ty(a) not in ('tU',) and ty(b) not in ('tU',)]
mine_report(CT, set(f for f, _ in collections.Counter(f for _, f, _ in CT).most_common(60)), "M6c RULES, cluster x type pair, 60 commonest")
P(f"DONE in {time.time() - T0:.0f}s")
