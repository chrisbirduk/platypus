"""CAN COMPRESSION STATISTICS TYPE AN INFERENCE? (14 Nov 2026; tools/compress_typing.py) - offline, no effect on the package.
Retraction needs claims a later sentence can contradict, i.e. TYPED inferences ('A is located in C', not 'A is connected to C';
tools/trigger_retest.py). Here: label-free statistics of the store (Re-DocRED + RM_EXTRA distant documents, as tools/fact_compress.py)
for pairs the store might infer - gold-related pairs of held-out documents, every record pairing the two names removed (never stated) -
and how well each says WHICH relation. Gold (Re-DocRED) is used only to score, and to map label-free classes to gold relations on the
KEPT documents (a probe of how much a signal determines the relation, as tools/rel_codes.py).
SIGNALS (no labels):
  S1 WORDING MDL       wordings clustered into latent relations (universal schema: PPMI wording x pair, SVD, k-means); a pair's contexts
                       coded under each latent relation's wording counts, the cheapest wins. PATH: the wordings of its bridge records
                       (A-B, B-C); DIRECT: the held-out document's own records of the pair (what later reading would supply)
  S2 NAME CLASSES      k-means classes of the names' compressed codes (PPMI-SVD of the name graph); the class pair
  S3 ASYMMETRY         store degree of A vs C (which name is general); predicts the direction of containment-like relations
  S4 FUNCTIONALITY     the share of a wording's subjects with a single partner (max over the pair's wordings)
  S5 PATH COMPOSITION  (latent relation of A-B, latent relation of B-C) -> the gold relation it most often yields on kept pairs
TARGETS: exact gold relation; FAMILY (location/containment, time, person attribute, organisation/membership, work, other); FUNCTIONAL or
not; DIRECTION. Baselines: the commonest relation / family. For each signal and the combination: accuracy, and COVERAGE AT 0.8
PRECISION (the share of pairs that can be typed with the signal's most confident predictions at 80% right).
Usage: RM_EXTRA=<distant documents> python3 tools/compress_typing.py   env FC_NT (400), FC_SEED (0), CT_K (64 latent relations), CT_NK (32)"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_rm = open(os.path.join(HERE, 'record_mdl.py')).read()
exec(compile(_rm[:_rm.index('FITS = {}')], 'record_mdl', 'exec'))
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
NT = int(os.environ.get('FC_NT', 400)); FS = int(os.environ.get('FC_SEED', 0)); KR = int(os.environ.get('CT_K', 64)); NK = int(os.environ.get('CT_NK', 32))
rh = np.random.RandomState(100 + FS); HOLD = set(rh.permutation(len(docs))[:NT].tolist())
FAM = {}
for r in 'P131 P17 P150 P361 P527 P706 P276 P30 P36 P1376 P159 P740 P205 P206 P403 P551 P937 P1001 P291 P840'.split(): FAM[r] = 'LOC'
for r in 'P569 P570 P571 P576 P577 P580 P582 P585 P1198'.split(): FAM[r] = 'TIME'
for r in 'P19 P20 P27 P26 P22 P25 P40 P3373 P69 P108 P140 P1412 P103 P39 P166 P241 P1344'.split(): FAM[r] = 'PERSON'
for r in 'P463 P102 P54 P118 P127 P749 P355 P112 P488 P35 P6 P194 P710 P1441 P279 P31'.split(): FAM[r] = 'ORG'
for r in 'P50 P57 P58 P86 P161 P162 P175 P176 P178 P272 P264 P123 P449 P495 P364 P400 P136 P170 P674 P676 P179 P155 P156 P1056'.split(): FAM[r] = 'WORK'
FUNC = set('P19 P20 P17 P131 P36 P1376 P159 P569 P570 P571 P576 P577 P740 P30 P495 P50 P57 P112 P26 P22 P25 P1412 P364'.split())
fam = lambda r: FAM.get(r, 'OTHER')
cnz = lambda d: [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
CANDL = {}; GOLDP = collections.defaultdict(set)                                    # (A, C) directed -> gold relations; all docs
for di, d in enumerate(docs):
    cn = cnz(d)
    for l in d['labels']:
        A, C_ = cn[l['h']], cn[l['t']]
        if A == C_: continue
        GOLDP[(A, C_)].add(l['r'])
        if di in HOLD: CANDL.setdefault((A, C_), set()).add(l['r'])
cand_pairs = {frozenset(k) for k in CANDL}
dvec = np.array([c['di'] for c in C] + [-1] * (len(Af) - N))
pk = [frozenset((NAMES[x], NAMES[z])) for x, z in zip(Af, Bf)]
keep = np.array([(dvec[i] not in HOLD) and (pk[i] not in cand_pairs) and Af[i] != Bf[i] for i in range(len(Af))])
P(f"held-out documents {NT}: candidate directed gold pairs {len(CANDL)} | store records kept {int(keep.sum())} of {len(Af)}  ({time.time() - T0:.0f}s)")
# ---- S1 latent relations of wordings (universal schema over kept records)
PI = {}; pid = np.array([PI.setdefault((int(x), int(z)), len(PI)) for x, z in zip(Af[keep], Bf[keep])])
M_ = csr_matrix((np.ones(len(pid)), (Wf[keep], pid)), shape=(NW, len(PI))); M_.sum_duplicates(); M_.data[:] = 1.0
rs_ = np.asarray(M_.sum(1)).ravel(); cs_ = np.asarray(M_.sum(0)).ravel(); tot = M_.sum(); Mc = M_.tocoo()
pm = np.log(tot / (rs_[Mc.row] * cs_[Mc.col]) + 1e-12); kp = pm > 0
E = normalize(TruncatedSVD(64, random_state=0).fit_transform(csr_matrix((pm[kp], (Mc.row[kp], Mc.col[kp])), shape=M_.shape)))
lrel = MiniBatchKMeans(KR, random_state=0, batch_size=4096, n_init=3).fit_predict(E)
wc = collections.Counter(Wf[keep].tolist())
# ---- S2 name classes (codes of the kept name graph)
G1 = csr_matrix((np.ones(int(keep.sum())), (Af[keep], Bf[keep])), shape=(NN, NN)); G1 = G1 + G1.T; G1.data[:] = 1.0; deg = np.asarray(G1.sum(1)).ravel()
Gc = G1.tocoo(); v = np.log(G1.sum() * Gc.data / (deg[Gc.row] * deg[Gc.col]) + 1e-12); m = v > 0
U = normalize(TruncatedSVD(128, random_state=0).fit_transform(csr_matrix((v[m].astype(np.float32), (Gc.row[m], Gc.col[m])), shape=G1.shape)).astype(np.float32))
ncl = np.full(NN, -1); has = deg > 0; ncl[has] = MiniBatchKMeans(NK, random_state=0, batch_size=4096, n_init=3).fit_predict(U[has])
P(f"latent relations {KR}, name classes {NK}  ({time.time() - T0:.0f}s)")
# ---- S4 functionality of wordings (kept records, directed A -> B)
subj = collections.defaultdict(lambda: collections.defaultdict(set))
for a_, w_, b_ in zip(Af[keep], Wf[keep], Bf[keep]): subj[int(w_)][int(a_)].add(int(b_))
fun = {w: float(np.mean([1.0 / len(s) for s in d.values()])) for w, d in subj.items() if sum(len(s) for s in d.values()) >= 5}
# ---- neighbours with wordings (kept), for PATH signals
nbw = collections.defaultdict(list)
for a_, w_, b_ in zip(Af[keep], Wf[keep], Bf[keep]): nbw[int(a_)].append((int(b_), int(w_), 0)); nbw[int(b_)].append((int(a_), int(w_), 1))
# ---- the held-out documents' own records of each candidate pair (DIRECT, 'later reading')
dirw = collections.defaultdict(list)
for i in range(N):
    if dvec[i] in HOLD and Af[i] != Bf[i]: dirw[pk[i]].append(int(Wf[i]))
# ---- training pairs (kept documents, gold-related, both names in the graph): fit the probes' maps
def feats(A, C_):
    a, c = NI.get(A), NI.get(C_)
    if a is None or c is None or deg[a] == 0 or deg[c] == 0: return None
    na = {}; [na.setdefault(b, []).append((w, s)) for b, w, s in nbw[a]]
    path = [(w1, w2) for b, w2, s2 in nbw[c] if b in na for w1, s1 in na[b]][:200]
    dw = dirw.get(frozenset((A, C_)), [])
    fn = max([fun.get(w, 0.0) for w1, w2 in path for w in (w1, w2)] + [fun.get(w, 0.0) for w in dw] + [0.0])
    return dict(cp=(int(ncl[a]), int(ncl[c])), path=[(int(lrel[w1]), int(lrel[w2])) for w1, w2 in path], pathw=[int(lrel[w]) for p_ in path for w in p_],
                dw=[int(lrel[w]) for w in dw], asym=float(np.log((deg[a] + 1) / (deg[c] + 1))), fun=fn)
TRP = [(k, v_) for k, v_ in GOLDP.items() if k not in CANDL and frozenset(k) not in cand_pairs]
rt = np.random.RandomState(3); rt.shuffle(TRP); TRP = TRP[:30000]
FTR = [(k, rr, feats(*k)) for k, rr in TRP]; FTR = [(k, rr, f) for k, rr, f in FTR if f is not None]
FTE = [(k, rr, feats(*k)) for k, rr in CANDL.items()]; FTE = [(k, rr, f) for k, rr, f in FTE if f is not None]
P(f"probe-fit pairs (kept documents) {len(FTR)}, test pairs (held out, never stated in the store) {len(FTE)}  ({time.time() - T0:.0f}s)")
def table(keyf, target):                                                          # label-free key -> majority target on training pairs
    t = collections.defaultdict(collections.Counter)
    for k, rr, f in FTR:
        for key in keyf(f):
            for r in rr: t[key][target(r)] += 1
    return {key: (c.most_common(1)[0][0], c.most_common(1)[0][1] / sum(c.values()), sum(c.values())) for key, c in t.items()}
def evaluate(name, keyf, target, tgt_set):
    tb = table(keyf, target); ok = []; conf = []; cov = 0
    for k, rr, f in FTE:
        votes = collections.Counter()
        for key in keyf(f):
            if key in tb and tb[key][2] >= 3: votes[tb[key][0]] += tb[key][1]
        if not votes: continue
        pred, s = votes.most_common(1)[0]; ok.append(pred in tgt_set(rr)); conf.append(s / sum(votes.values()) * min(1.0, sum(votes.values())))
        cov += 1
    if not ok: return f"{name}: no coverage"
    ok = np.array(ok); conf = np.array(conf); o = np.argsort(-conf); cp = np.cumsum(ok[o]) / np.arange(1, len(o) + 1)
    c80 = (np.where(cp >= 0.8)[0].max() + 1) / len(FTE) if (cp >= 0.8).any() else 0.0
    return f"{name}: typed {cov / len(FTE):.2f} of pairs, right {ok.mean():.3f} | coverage at 0.8 precision {c80:.3f}"
gold_r = lambda rr: rr; gold_f = lambda rr: {fam(r) for r in rr}; gold_fn = lambda rr: {r in FUNC for r in rr}
commonest = collections.Counter(r for k, rr, f in FTR for r in rr).most_common(1)[0][0]
commonf = collections.Counter(fam(r) for k, rr, f in FTR for r in rr).most_common(1)[0][0]
P(f"BASELINES on the test pairs: commonest relation ({commonest}) {np.mean([commonest in rr for k, rr, f in FTE]):.3f} | commonest family ({commonf}) "
  f"{np.mean([commonf in gold_f(rr) for k, rr, f in FTE]):.3f} | functional share {np.mean([any(r in FUNC for r in rr) for k, rr, f in FTE]):.3f} | test families "
  + ", ".join(f"{a} {b}" for a, b in collections.Counter(fam(next(iter(sorted(rr)))) for k, rr, f in FTE).most_common()))
SIG = [('S2 name-class pair', lambda f: [('cp',) + f['cp']]),
       ('S1 PATH wording (latent relations of the bridge records)', lambda f: [('pw', w) for w in f['pathw'][:40]]),
       ('S5 PATH composition (latent A-B, latent B-C)', lambda f: [('pc',) + p for p in f['path'][:40]]),
       ('S1 DIRECT wording (the pair read later)', lambda f: [('dw', w) for w in f['dw']]),
       ('S3 asymmetry (A more specific than C: quartile)', lambda f: [('as', int(np.digitize(f['asym'], [-1.5, -0.5, 0.5, 1.5])))]),
       ('S2 + S3 (class pair and asymmetry)', lambda f: [('ca',) + f['cp'] + (int(np.digitize(f['asym'], [-1.5, -0.5, 0.5, 1.5])),)]),
       ('ALL (class pair + asymmetry, path composition, direct wording)', lambda f: [('ca',) + f['cp'] + (int(np.digitize(f['asym'], [-1.5, -0.5, 0.5, 1.5])),)]
                                                                                   + [('pc',) + p for p in f['path'][:40]] + [('dw', w) for w in f['dw']])]
for tname, target, tset in (('EXACT RELATION', lambda r: r, gold_r), ('FAMILY', fam, gold_f), ('FUNCTIONAL', lambda r: r in FUNC, gold_fn)):
    P(f"{tname}:")
    for sname, kf in SIG: P("   " + evaluate(sname, kf, target, tset))
# ---- S4 functionality as a direct test of 'functional'; S3 direction for LOC-family pairs
yfn = np.array([any(r in FUNC for r in rr) for k, rr, f in FTE]); fnv = np.array([f['fun'] for k, rr, f in FTE])
P(f"S4 FUNCTIONALITY of the pair's wordings vs gold functional: AUC {auc(fnv[yfn], fnv[~yfn]):.3f}")
loc = [(k, rr, f) for k, rr, f in FTE if any(fam(r) == 'LOC' for r in rr)]
if loc:
    P(f"S3 DIRECTION, location-family pairs ({len(loc)}): the head is the more specific name (lower store degree) {np.mean([f['asym'] < 0 for k, rr, f in loc]):.3f} "
      f"(gold head = the pair's first name; 0.5 = no signal)")
P(f"DONE ({time.time() - T0:.0f}s)")
