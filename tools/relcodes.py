"""RELATION CODES FROM A TRUNK'S STATES, AND RULES OVER THEM (12 Nov 2026; tools/relcodes.py) - option B, stages 1 and 2, offline.
STAGE 1: every record of the fast channel (subject-anchored + cleaned adjacent records, local Wikipedia + the benchmark) gets a RELATION
STATE: a byte language model (trained here on the same text; the toy's architecture) reads 'Someone <frame> ' (the subject masked, inner
fillers as 'something') and its last hidden state is taken where the object would begin. A small projector is trained contrastively
(InfoNCE) so that records linking the SAME PAIR of names in different sentences get the same code (Matching-the-Blanks on the model's
states; no labels). Scored as reader_tests T2b: on benchmark records of gold pairs, is the nearest other record of the same gold relation
type - frame words (tf-idf), the raw state, the trained code, code + words - and cluster purity against gold types.
STAGE 2: records re-labelled by their code's cluster (k-means); the relational module (_rm_mine, every guard) mines rules over the
clusters; rules, proposals and their gold rate on benchmark entities - against the same mining over frame strings (0 rules so far).
Usage: python3 tools/relcodes.py [lm steps] [projector epochs] [clusters]"""
import os
import sys
import time
import collections
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
LMS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; EP = int(sys.argv[2]) if len(sys.argv) > 2 else 6; KC = int(sys.argv[3]) if len(sys.argv) > 3 else 48
_r = open(os.path.join(HERE, 'reader_tests2.py')).read()
exec(compile(_r[:_r.index('# ---------------- M1: informative frames')], 'reader_tests2', 'exec'))   # anchored, list_runs, recs_full, ty, gp, G, mine_report
_t = open(os.path.join(HERE, 'toy_jname.py')).read(); _argv = sys.argv; sys.argv = [sys.argv[0], '0', '10']
exec(compile(_t[:_t.index('# ---------------------------------------------------------------- readouts')], 'toy_jname', 'exec'))   # LM, train_lm, Tr (F = torch functional again)
sys.argv = _argv
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2)
# ---- records with provenance
REC = []                                                                    # (a, frame, b, doc key, kind)
for src, dd in (('B', bdocs), ('W', wdocs)):
    for di, d in enumerate(dd):
        lr = {si: list_runs(t) for si, t in enumerate(d)}
        for a, f, b, si, k in anchored(d): REC.append((a, f, b, (src, di, si), 'anc'))
        for a, f, b, si, k in recs_full(d):
            if k != 1 or (a in lr[si] and b in lr[si]) or ty(a) == 'tD' or ty(b) == 'tD' or a == b: continue
            REC.append((a, f, b, (src, di, si), 'adj'))
P(f"records {len(REC)} ({sum(1 for r in REC if r[3][0] == 'B')} benchmark)  ({time.time() - T0:.0f}s)")
# ---- the byte LM on the same text
raw_t = (raw.replace('\n', ' ') + ' ' + ' '.join(docs)).encode('latin-1', 'replace'); rsx = np.random.RandomState(0)
def rbatch(n):
    ix = rsx.randint(0, len(raw_t) - L - 1, n); return torch.tensor([list(raw_t[i:i + L]) for i in ix])
batch = lambda n, *a, **k: rbatch(n)
lm = train_lm(LMS)
frames = sorted({r[1] for r in REC}); Fi = {f: i for i, f in enumerate(frames)}
@torch.no_grad()
def states(fs, bs=256):
    out = []
    for i in range(0, len(fs), bs):
        ps = [('Someone ' + f.replace('[F]', 'something') + ' ').encode('latin-1', 'replace')[:L - 2] for f in fs[i:i + bs]]
        X = torch.full((len(ps), L), PAD, dtype=torch.long)
        for j, p in enumerate(ps): X[j, 1:1 + len(p)] = torch.tensor(list(p))
        X[:, 0] = PAD; H = lm.t(X)                                                # input = PAD + prefix: the state at index len(prefix) has read it all
        out.append(torch.stack([H[j, len(p)] for j, p in enumerate(ps)]))
    return torch.cat(out)
S = states(frames); P(f"relation states for {len(frames)} distinct frames  ({time.time() - T0:.0f}s)")
# ---- stage 1: contrastive projector over same-pair records in different sentences
grp = collections.defaultdict(list)
for a, f, b, pv, k in REC: grp[(min(a, b), max(a, b))].append((Fi[f], pv))
pos = []
for key, L_ in grp.items():
    seen = {}
    for fi, pv in L_: seen.setdefault(pv, fi)
    fs = sorted(set(seen.values()))
    for i in range(len(fs)):
        for j in range(i + 1, min(len(fs), i + 4)): pos.append((fs[i], fs[j]))
P(f"positive pairs (one name pair, two different frames): {len(pos)}")
proj = nn.Sequential(nn.Linear(S.shape[1], 256), nn.GELU(), nn.Linear(256, 64)); opt = torch.optim.Adam(proj.parameters(), lr=1e-3); rs8 = np.random.RandomState(0)
for ep in range(EP):
    idx = rs8.permutation(len(pos)); ls = []
    for i in range(0, len(idx) - 255, 256):
        bt = [pos[t] for t in idx[i:i + 256]]; z1 = F.normalize(proj(S[[x for x, y in bt]]), dim=-1); z2 = F.normalize(proj(S[[y for x, y in bt]]), dim=-1)
        lg = z1 @ z2.t() / 0.1; y = torch.arange(len(bt)); loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.t(), y)) / 2
        opt.zero_grad(); loss.backward(); opt.step(); ls.append(float(loss))
    P(f"   epoch {ep + 1}: loss {np.mean(ls[:20]) if ls else float('nan'):.2f} -> {np.mean(ls[-20:]) if ls else float('nan'):.2f}")
with torch.no_grad(): Z = F.normalize(proj(S), dim=-1).numpy()
# ---- stage 1 scoring (as reader_tests T2b)
rel = {}
for h, t, fl, ps in G['rel']: rel.setdefault((min(h, t), max(h, t)), set()).update(ps)
ex = []
for a, f, b, pv, k in REC:
    if pv[0] != 'B': continue
    g = gp(a, b)
    if g: ex.append((Fi[f], set().union(*[rel[x] for x in g])))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
def knn(X):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); Sm = X @ X.T; np.fill_diagonal(Sm, -9); nn_ = Sm.argmax(1)
    return float(np.mean([bool(ex[i][1] & ex[j][1]) for i, j in enumerate(nn_)]))
fi_ex = [e[0] for e in ex]
W = TfidfVectorizer(token_pattern=r'[^ ]+').fit_transform([frames[i] for i in fi_ex]).toarray()
Sraw = S.numpy()[fi_ex]; Zc = Z[fi_ex]
both = np.concatenate([Zc / np.linalg.norm(Zc, axis=1, keepdims=True), W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)], 1)
Pc = collections.Counter(p for e in ex for p in e[1]); maj = max(np.mean([p in e[1] for e in ex]) for p in Pc)
P(f"STAGE 1 ({len(ex)} benchmark gold-pair records): nearest other record shares a gold relation type - majority {maj:.2f} | frame words {knn(W):.2f} | "
  f"raw LM state {knn(Sraw):.2f} | TRAINED RELATION CODE {knn(Zc):.2f} | code + words {knn(both):.2f}   (earlier: frame words 0.44, + filler types 0.47)")
lab = KMeans(KC, n_init=4, random_state=0).fit_predict(Z); CL = {frames[i]: int(lab[i]) for i in range(len(frames))}
pur = []
for c in range(KC):
    ps_ = [p for e in ex if CL[frames[e[0]]] == c for p in e[1]]
    if ps_: pur.append((collections.Counter(ps_).most_common(1)[0][1], len([e for e in ex if CL[frames[e[0]]] == c])))
P(f"   clusters ({KC}): gold-type purity of benchmark gold-pair records {sum(a for a, b in pur) / max(1, sum(b for a, b in pur)):.2f}")
cm = collections.defaultdict(collections.Counter)
for a, f, b, pv, k in REC: cm[CL[f]][f] += 1
for c in sorted(cm, key=lambda c: -sum(cm[c].values()))[:8]: P(f"   cluster {c}: " + ", ".join(repr(f) for f, _ in cm[c].most_common(6)))
# ---- stage 2: rules over the clusters
A3 = [(a, f"c{CL[f]}", b) for a, f, b, pv, k in REC]
mine_report(A3, set(f"c{c}" for c in range(KC)), f"STAGE 2 RULES over {KC} learned relation clusters")
mine_report([(a, f, b) for a, f, b, pv, k in REC], set(f for f, _ in collections.Counter(r[1] for r in REC).most_common(60)), "   comparison: the 60 commonest frame strings")
P(f"DONE ({time.time() - T0:.0f}s)")
