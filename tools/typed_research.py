"""TYPED INFERENCE FROM THE LITERATURE, TESTED ON THE STORE (14 Nov 2026; tools/typed_research.py) - offline, no effect on the package.
Store as tools/compress_typing.py (Re-DocRED + RM_EXTRA distant documents; held-out documents; gold only to score, and to map label-free
classes to gold relations on kept documents - a probe). Two ideas:
 T1 TYPED PREDICATES + ENTAILMENT BACK-OFF (after Berant et al. 2011, Hosseini et al. 2018): a record's predicate is TYPED by the
    compressed classes of its two names: (class A, wording, class B). Sparse typed predicates back off to the predicate that most INCLUDES
    their argument pairs (directional inclusion, the entailment-graph score) among frequent ones. Typing a held-out record (a pair READ in
    a held-out document): wording alone / latent wording cluster / typed predicate / typed predicate + entailment back-off.
 T2 RELATION DISCOVERY BY RECONSTRUCTING ARGUMENTS (after Marcheggiani & Titov 2016): an encoder (wording + class pair) gives a
    distribution over K latent relations; a factorisation decoder (DistMult over name embeddings) must reconstruct B from A and the relation;
    trained jointly on all kept records, no labels. Scored on held-out records (encoder) and on held-out pairs NEVER STATED in the store
    (decoder alone: the relation that best links A and C - typing an inference).
Scores: exact relation (undirected gold relation set) and family; accuracy of the typed share, and COVERAGE AT 0.8 PRECISION.
Usage: RM_EXTRA=<docs> python3 tools/typed_research.py   env FC_NT (400), FC_SEED (0), TR_K (64), TR_EP (6)"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_rm = open(os.path.join(HERE, 'record_mdl.py')).read()
exec(compile(_rm[:_rm.index('FITS = {}')], 'record_mdl', 'exec'))
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
torch.set_num_threads(1)
NT = int(os.environ.get('FC_NT', 400)); FS = int(os.environ.get('FC_SEED', 0)); KR = int(os.environ.get('TR_K', 64)); EP = int(os.environ.get('TR_EP', 6))
rh = np.random.RandomState(100 + FS); HOLD = set(rh.permutation(len(docs))[:NT].tolist())
FAM = {}
for r in 'P131 P17 P150 P361 P527 P706 P276 P30 P36 P1376 P159 P740 P205 P206 P403 P551 P937 P1001 P291 P840'.split(): FAM[r] = 'LOC'
for r in 'P569 P570 P571 P576 P577 P580 P582 P585 P1198'.split(): FAM[r] = 'TIME'
for r in 'P19 P20 P27 P26 P22 P25 P40 P3373 P69 P108 P140 P1412 P103 P39 P166 P241 P1344'.split(): FAM[r] = 'PERSON'
for r in 'P463 P102 P54 P118 P127 P749 P355 P112 P488 P35 P6 P194 P710 P1441 P279 P31'.split(): FAM[r] = 'ORG'
for r in 'P50 P57 P58 P86 P161 P162 P175 P176 P178 P272 P264 P123 P449 P495 P364 P400 P136 P170 P674 P676 P179 P155 P156 P1056'.split(): FAM[r] = 'WORK'
fam = lambda r: FAM.get(r, 'OTHER')
cnz = lambda d: [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
GREL = collections.defaultdict(set); CAND = set()
for di, d in enumerate(docs):
    cn = cnz(d)
    for l in d['labels']:
        if cn[l['h']] != cn[l['t']]:
            GREL[frozenset((cn[l['h']], cn[l['t']]))].add(l['r'])
            if di in HOLD: CAND.add(frozenset((cn[l['h']], cn[l['t']])))
dvec = np.array([c['di'] for c in C] + [-1] * (len(Af) - N))
pk = [frozenset((NAMES[x], NAMES[z])) for x, z in zip(Af, Bf)]
keep = np.array([(dvec[i] not in HOLD) and (pk[i] not in CAND) and Af[i] != Bf[i] for i in range(len(Af))])
hrec = np.array([i for i in range(N) if dvec[i] in HOLD and Af[i] != Bf[i] and pk[i] in GREL])          # held-out records of gold pairs (read)
krec = np.array([i for i in range(N) if keep[i] and pk[i] in GREL])                                       # kept gold records (probe fit)
P(f"held-out documents {NT}: held-out gold records {len(hrec)}, kept gold records {len(krec)}, never-stated gold pairs {len(CAND)} | store records kept {int(keep.sum())}  ({time.time() - T0:.0f}s)")
# ---- name classes (compressed codes) and latent wording clusters (universal schema)
G1 = csr_matrix((np.ones(int(keep.sum())), (Af[keep], Bf[keep])), shape=(NN, NN)); G1 = G1 + G1.T; G1.data[:] = 1.0; deg = np.asarray(G1.sum(1)).ravel()
Gc = G1.tocoo(); v = np.log(G1.sum() * Gc.data / (deg[Gc.row] * deg[Gc.col]) + 1e-12); m = v > 0
U = normalize(TruncatedSVD(128, random_state=0).fit_transform(csr_matrix((v[m].astype(np.float32), (Gc.row[m], Gc.col[m])), shape=G1.shape)).astype(np.float32))
ncl = np.full(NN, -1); hs = deg > 0; ncl[hs] = MiniBatchKMeans(32, random_state=0, batch_size=4096, n_init=3).fit_predict(U[hs])
PI = {}; pid = np.array([PI.setdefault((int(x), int(z)), len(PI)) for x, z in zip(Af[keep], Bf[keep])])
M_ = csr_matrix((np.ones(len(pid)), (Wf[keep], pid)), shape=(NW, len(PI))); M_.sum_duplicates(); M_.data[:] = 1.0
rs_ = np.asarray(M_.sum(1)).ravel(); cs_ = np.asarray(M_.sum(0)).ravel(); tot = M_.sum(); Mc = M_.tocoo()
pm = np.log(tot / (rs_[Mc.row] * cs_[Mc.col]) + 1e-12); kp = pm > 0
lrel = MiniBatchKMeans(KR, random_state=0, batch_size=4096, n_init=3).fit_predict(normalize(TruncatedSVD(64, random_state=0).fit_transform(csr_matrix((pm[kp], (Mc.row[kp], Mc.col[kp])), shape=M_.shape))))
P(f"name classes 32, latent wording clusters {KR}  ({time.time() - T0:.0f}s)")
# ---- probe machinery
def probe(keys_tr, keys_te, ys_tr, ys_te, name):
    tab = collections.defaultdict(collections.Counter)
    for ks, ys in zip(keys_tr, ys_tr):
        for k in ks:
            for y in ys: tab[k][y] += 1
    ok, conf = [], []
    for ks, ys in zip(keys_te, ys_te):
        vt = collections.Counter()
        for k in ks:
            c = tab.get(k)
            if c and sum(c.values()) >= 3: y0, n0 = c.most_common(1)[0]; vt[y0] += n0 / sum(c.values())
        if not vt: continue
        y0, s0 = vt.most_common(1)[0]; ok.append(y0 in ys); conf.append(s0 / max(1e-9, sum(vt.values())) * min(1.0, s0))
    if not ok: return f"{name}: none typed"
    ok = np.array(ok); o = np.argsort(-np.array(conf)); cp = np.cumsum(ok[o]) / np.arange(1, len(o) + 1)
    c80 = (np.where(cp >= 0.8)[0].max() + 1) / len(keys_te) if (cp >= 0.8).any() else 0.0
    return f"{name}: typed {len(ok) / len(keys_te):.2f}, right {ok.mean():.3f} | coverage at 0.8 precision {c80:.3f}"
yr_tr = [GREL[pk[i]] for i in krec]; yr_te = [GREL[pk[i]] for i in hrec]
yf_tr = [{fam(r) for r in s} for s in yr_tr]; yf_te = [{fam(r) for r in s} for s in yr_te]
cr = collections.Counter(r for s in yr_te for r in s).most_common(1)[0][0]; cf = collections.Counter(f for s in yf_te for f in s).most_common(1)[0][0]
P(f"T1 TYPED PREDICATES on {len(hrec)} held-out read records (baselines: commonest relation {cr} {np.mean([cr in s for s in yr_te]):.3f}, commonest family {cf} "
  f"{np.mean([cf in s for s in yf_te]):.3f})")
tp = lambda i: (int(ncl[Af[i]]), int(Wf[i]), int(ncl[Bf[i]]))
args = collections.defaultdict(set)
for i in np.where(keep)[0]: args[tp(i)].add((int(Af[i]), int(Bf[i])))
freq = {p for p, s in args.items() if len(s) >= 3}
byarg = collections.defaultdict(set)
for p in freq:
    for ab in args[p]: byarg[ab].add(p)
wargs = collections.defaultdict(set)                                              # a WORDING's argument pairs over all classes (kept records)
for i in np.where(keep)[0]: wargs[int(Wf[i])].add((int(Af[i]), int(Bf[i])))
bycp = collections.defaultdict(list)
for q in freq: bycp[(q[0], q[2])].append(q)
_bo = {}
def backoff(p):                                                                   # entailment: the frequent typed predicate of the same class pair that
    if p in freq: return p                                                        #   most INCLUDES p's argument pairs - p's own if it was ever read in the
    if p in _bo: return _bo[p]                                                    #   store, else its wording's over all classes (directional inclusion,
    src = args.get(p) or wargs.get(p[1], set()); best, bs = None, 0.0             #   the entailment-graph score)
    if src:
        for q in bycp.get((p[0], p[2]), ()):
            sc = len(src & args[q]) / len(src)
            if sc > bs: best, bs = q, sc
    _bo[p] = best if bs >= 0.05 else None; return _bo[p]
for nm, kf in (('wording alone', lambda i: [('w', int(Wf[i]))]), ('latent wording cluster', lambda i: [('l', int(lrel[Wf[i]]))]),
               ('class pair alone', lambda i: [('c', int(ncl[Af[i]]), int(ncl[Bf[i]]))]),
               ('TYPED PREDICATE (class A, wording, class B)', lambda i: [('t',) + tp(i)]),
               ('typed predicate, else class pair + cluster', lambda i: [('t',) + tp(i)] if tp(i) in freq else [('cl', int(ncl[Af[i]]), int(lrel[Wf[i]]), int(ncl[Bf[i]]))]),
               ('TYPED PREDICATE + ENTAILMENT BACK-OFF', lambda i: ([('t',) + backoff(tp(i))] if backoff(tp(i)) else [('cl', int(ncl[Af[i]]), int(lrel[Wf[i]]), int(ncl[Bf[i]]))]))):
    ktr = [kf(i) for i in krec]; kte = [kf(i) for i in hrec]
    P(f"   EXACT  {probe(ktr, kte, yr_tr, yr_te, nm)}"); P(f"   FAMILY {probe(ktr, kte, yf_tr, yf_te, nm)}")
nb = sum(1 for i in hrec if tp(i) not in freq and backoff(tp(i)) is not None); nf_ = sum(1 for i in hrec if tp(i) in freq)
P(f"   held-out records with a frequent typed predicate {nf_ / len(hrec):.2f}, rescued by entailment back-off {nb / len(hrec):.2f}  ({time.time() - T0:.0f}s)")
# ---- T2 joint relation discovery: encoder q(r | wording, class pair), DistMult decoder reconstructs B
P(f"T2 RELATION DISCOVERY BY RECONSTRUCTING ARGUMENTS (K {KR}, {EP} epochs over {int(keep.sum())} kept records)")
torch.manual_seed(FS); D_ = 64
encW = torch.nn.Embedding(NW, KR); encC = torch.nn.Embedding(33 * 33, KR); emb = torch.nn.Embedding(NN, D_); rop = torch.nn.Embedding(KR, D_)
torch.nn.init.normal_(encW.weight, std=0.1); torch.nn.init.zeros_(encC.weight); torch.nn.init.normal_(emb.weight, std=0.1); torch.nn.init.normal_(rop.weight, mean=1.0, std=0.1)
op = torch.optim.Adam(list(encW.parameters()) + list(encC.parameters()) + list(emb.parameters()) + list(rop.parameters()), lr=0.01)
ki = np.where(keep)[0]; TA = torch.tensor(Af[ki]); TB = torch.tensor(Bf[ki]); TW = torch.tensor(Wf[ki]); TC = torch.tensor((ncl[Af[ki]] + 1) * 33 + ncl[Bf[ki]] + 1)
for ep in range(EP):
    perm = torch.randperm(len(ki)); ls = []
    for s0 in range(0, len(ki), 4096):
        ix = perm[s0:s0 + 4096]; q = F.softmax(encW(TW[ix]) + encC(TC[ix]), -1)                       # (b, K)
        a = emb(TA[ix]); b = emb(TB[ix]); neg = emb(torch.randint(0, NN, (len(ix), 8)))
        pos = torch.einsum('bd,kd,bd->bk', a, rop.weight, b); ng = torch.einsum('bd,kd,bnd->bkn', a, rop.weight, neg)
        lp = pos - torch.logsumexp(torch.cat([pos[..., None], ng], -1), -1)                          # log p(B | A, r) against negatives
        l = -(q * lp).sum(-1).mean() - 0.05 * (-(q * q.clamp(min=1e-9).log()).sum(-1)).mean() * 0 + 0.1 * (q.mean(0) * (q.mean(0).clamp(min=1e-9).log() + math.log(KR))).sum()
        op.zero_grad(); l.backward(); op.step(); ls.append(float(l))
    P(f"   epoch {ep + 1}: loss {np.mean(ls):.4f}  ({time.time() - T0:.0f}s)")
with torch.no_grad():
    def enc_r(ix): return F.softmax(encW(torch.tensor(Wf[ix])) + encC(torch.tensor((ncl[Af[ix]] + 1) * 33 + ncl[Bf[ix]] + 1)), -1).argmax(-1).numpy()
    rk_tr = enc_r(krec); rk_te = enc_r(hrec)
P(f"   EXACT  {probe([[('z', int(r))] for r in rk_tr], [[('z', int(r))] for r in rk_te], yr_tr, yr_te, 'ENCODER relation of a held-out read record')}")
P(f"   FAMILY {probe([[('z', int(r))] for r in rk_tr], [[('z', int(r))] for r in rk_te], yf_tr, yf_te, 'ENCODER relation of a held-out read record')}")
cp_ = [k for k in CAND if all(NI.get(n) is not None and deg[NI[n]] > 0 for n in k)]
XA = torch.tensor([NI[sorted(k)[0]] for k in cp_]); XB = torch.tensor([NI[sorted(k)[1]] for k in cp_])
trp = [k for k in GREL if k not in CAND and all(NI.get(n) is not None and deg[NI[n]] > 0 for n in k)]; rt = np.random.RandomState(4); rt.shuffle(trp); trp = trp[:20000]
YA = torch.tensor([NI[sorted(k)[0]] for k in trp]); YB = torch.tensor([NI[sorted(k)[1]] for k in trp])
with torch.no_grad():
    dec = lambda X1, X2: torch.maximum(torch.einsum('bd,kd,bd->bk', emb(X1), rop.weight, emb(X2)), torch.einsum('bd,kd,bd->bk', emb(X2), rop.weight, emb(X1))).argmax(-1).numpy()
    rd_te = dec(XA, XB); rd_tr = dec(YA, YB)
yc_te = [GREL[k] for k in cp_]; yc_tr = [GREL[k] for k in trp]
P(f"   never-stated held-out pairs {len(cp_)} (commonest relation {np.mean([cr in s for s in yc_te]):.3f}, commonest family {np.mean([cf in {fam(r) for r in s} for s in yc_te]):.3f}); "
  f"class pair alone: EXACT {probe([[('c', int(ncl[NI[sorted(k)[0]]]), int(ncl[NI[sorted(k)[1]]]))] for k in trp], [[('c', int(ncl[NI[sorted(k)[0]]]), int(ncl[NI[sorted(k)[1]]]))] for k in cp_], yc_tr, yc_te, 'class pair').split(': ', 1)[1]}")
P(f"   EXACT  {probe([[('d', int(r))] for r in rd_tr], [[('d', int(r))] for r in rd_te], yc_tr, yc_te, 'DECODER relation of a NEVER-STATED pair (typing an inference)')}")
P(f"   FAMILY {probe([[('d', int(r))] for r in rd_tr], [[('d', int(r))] for r in rd_te], [{fam(r) for r in s} for s in yc_tr], [{fam(r) for r in s} for s in yc_te], 'DECODER relation of a NEVER-STATED pair')}")
P(f"DONE ({time.time() - T0:.0f}s)")
