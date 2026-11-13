"""RECORD-LEVEL COMPRESSION: CAN TYPES BE DISCOVERED BY COMPRESSING THE STORE? (13 Nov 2026; tools/record_mdl.py) - offline, no effect on the package.
The store's records (name A, wording between, name B) from Re-DocRED train + dev (tools/consistency_teacher.py candidates; no labels used)
are compressed by a three-way block model: every name gets one NAME CLASS, every wording one WORDING CLASS, and a record costs
   -log2 p(class A, wording class, class B) - log2 p(A | its class) - log2 p(wording | its class) - log2 p(B | its class)
Fitted by information-theoretic co-clustering (Dhillon et al. 2003, three-way): names reassigned to the class whose (wording class, partner
class) profile best predicts theirs, wordings likewise, alternating; 3 random restarts. Total description length (data + block table at
0.5 log2 N bits a cell + class assignments) picks the number of classes (MDL). Then, per configuration:
  TYPES      name classes against gold entity types (purity, NMI; weighted by records)
  RELATION   held-out documents, gold-related records: 600 queries x 30 candidates from other documents; is the same gold relation found by
             equal name-class pair / wording class / block (as tools/relation_free.py; gold entity-type pair is the reference 0.83)
  PREMISES   label-free record scores from the block model (how strongly the name-class pair predicts the wording class) on held-out
             records: AUC and precision at the top 5 / 10 / 25 %, alone and with the consistency-teacher learner; the target from
             tools/premise_threshold.py is premise precision 0.6-0.7
  INFERENCE  A-C proposals (tools/integrate_test.py) from the store kept by the best label-free score: gold rate
Writes the name classes to RM_OUT ({k: [...], types: {name: [class per configuration]}}) for tools/relation_free.py RF_TYPES.
RM_INIT=split grows each configuration from the largest smaller one (classes split at random); nt:<json>:<column> starts name classes
from tools/name_types.py context types.
RM_SUB=<fraction>: the block model is fitted on the consistency-teacher learner's top records only (a cleaner store); every other name and
wording is then classed by all its records under the fitted table (the extractor cleans the store, compression finds classes on it).
Usage: python3 tools/record_mdl.py   env RM_SUB, RM_INIT, RM_K (name classes, default 4,8,16,32), RM_M (wording classes, default 8,32,128), RM_OUT"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_ct = open(os.path.join(HERE, 'consistency_teacher.py')).read()
exec(compile(_ct[:_ct.index('# ---- SELF-TRAINING ROUNDS')], 'consistency_teacher', 'exec'))
from scipy.sparse import csr_matrix
from sklearn.metrics import normalized_mutual_info_score
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
OUT = os.environ.get('RM_OUT', SCR + '/record_mdl.json')
KS = [int(x) for x in os.environ.get('RM_K', '4,8,16,32').split(',')]; MS = [int(x) for x in os.environ.get('RM_M', '8,32,128').split(',')]
N = len(C)
# ---- records: name ids, wording ids (wordings seen < 3 times backed off to their last word, then to their length)
fc = collections.Counter(c['f'] for c in C); lw = lambda c: '~' + (c['mid'][-1].lower() if c['mid'] else '')
lc = collections.Counter(lw(c) for c in C)
def wk(c):
    if fc[c['f']] >= 3: return c['f']
    k = lw(c); return k if lc[k] >= 3 else '~?%d' % min(len(c['mid']), 6)
NI = {}; WI = {}
a = np.array([NI.setdefault(c['cA'], len(NI)) for c in C]); b = np.array([NI.setdefault(c['cB'], len(NI)) for c in C])
w = np.array([WI.setdefault(wk(c), len(WI)) for c in C]); NN, NW = len(NI), len(WI); NAMES = list(NI)
P(f"records {N}, names {NN}, wordings {NW} (exact {sum(1 for k in WI if not k.startswith('~'))}, backed off {sum(1 for k in WI if k.startswith('~'))})")
# gold entity type per name (evaluation only)
gtc = collections.defaultdict(collections.Counter)
for d in docs:
    for vs in d['vertexSet']:
        cn_ = collections.Counter(m['name'] for m in vs).most_common(1)[0][0]
        for m in vs: gtc[cn_][m.get('type', 'X')] += 1
GT = np.array([gtc[n].most_common(1)[0][0] if gtc[n] else 'X' for n in NAMES])
SUB = float(os.environ.get('RM_SUB', 0))                                        # 13 Nov 18:30: fit on the extractor's top records only
lr_ = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, teacher_tr >= np.percentile(teacher_tr, 75))
XA = feats(np.arange(N)); s_all = lr_.decision_function(XA)
FI = np.argsort(-s_all)[:int(N * SUB)] if SUB else np.arange(N)
fa, fb, fw = a[FI], b[FI], w[FI]; NF = len(FI); one = np.ones(NF)
if SUB: P(f"FITTED ON the consistency-teacher learner's top {SUB:.0%} of records: {NF}, related {y[FI].mean():.3f}")


def table(gn, gw, K, M, al=0.0):
    T = np.zeros((K, M, K)); np.add.at(T, (gn[fa], gw[fw], gn[fb]), 1); return T + al


def fit(K, M, seed, iters=30, al=0.1, init=None):
    r = np.random.RandomState(seed); gn = r.randint(K, size=NN); gw = r.randint(M, size=NW)
    if init is not None: gn, gw = init(r)
    for it in range(iters):
        T = table(gn, gw, K, M, al)
        PL = np.log(T / T.sum((1, 2), keepdims=True)).reshape(K, M * K)                       # p(wording class, B class | A class)
        PR = np.log(T / T.sum((0, 1), keepdims=True)).transpose(2, 0, 1).reshape(K, K * M)     # p(A class, wording class | B class)
        Lm = csr_matrix((one, (fa, gw[fw] * K + gn[fb])), shape=(NN, M * K)); Rm = csr_matrix((one, (fb, gn[fa] * M + gw[fw])), shape=(NN, K * M))
        g2 = np.asarray(Lm @ PL.T + Rm @ PR.T).argmax(1)
        for e in set(range(K)) - set(g2.tolist()): g2[r.randint(NN)] = e                     # no empty class
        ch = (g2 != gn).mean(); gn = g2
        T = table(gn, gw, K, M, al); PW = np.log(T / T.sum((0, 2), keepdims=True)).transpose(1, 0, 2).reshape(M, K * K)   # p(A cl, B cl | w cl)
        Wm = csr_matrix((one, (fw, gn[fa] * K + gn[fb])), shape=(NW, K * K)); h2 = np.asarray(Wm @ PW.T).argmax(1)
        for e in set(range(M)) - set(h2.tolist()): h2[r.randint(NW)] = e
        ch2 = (h2 != gw).mean(); gw = h2
        if ch < 0.002 and ch2 < 0.002: break
    if SUB:                                                                     # names / wordings outside the fitted records: classed by all
        T = table(gn, gw, K, M, al); o1 = np.ones(N)                            #   their records under the fitted block table
        PL = np.log(T / T.sum((1, 2), keepdims=True)).reshape(K, M * K); PR = np.log(T / T.sum((0, 1), keepdims=True)).transpose(2, 0, 1).reshape(K, K * M)
        Lm = csr_matrix((o1, (a, gw[w] * K + gn[b])), shape=(NN, M * K)); Rm = csr_matrix((o1, (b, gn[a] * M + gw[w])), shape=(NN, K * M))
        out = np.ones(NN, bool); out[fa] = False; out[fb] = False; gn = gn.copy(); gn[out] = np.asarray(Lm @ PL.T + Rm @ PR.T)[out].argmax(1)
        PW = np.log(T / T.sum((0, 2), keepdims=True)).transpose(1, 0, 2).reshape(M, K * K)
        Wm = csr_matrix((o1, (w, gn[a] * K + gn[b])), shape=(NW, K * K)); out = np.ones(NW, bool); out[fw] = False; gw = gw.copy()
        gw[out] = np.asarray(Wm @ PW.T)[out].argmax(1)
    return gn, gw


def dl(gn, gw, K, M):
    """total description length in bits: records given the classes + block table + class assignments"""
    T = table(gn, gw, K, M); d = -np.log2(T[gn[fa], gw[fw], gn[fb]] / NF).sum()
    for ids, g in ((fa, gn[fa]), (fb, gn[fb]), (fw, gw[fw])):
        ni = np.bincount(ids)[ids]; ng = np.bincount(g)[g]; d += -np.log2(ni / ng).sum()
    model = 0.5 * K * M * K * np.log2(NF) + len(set(fa) | set(fb)) * np.log2(max(K, 1)) + len(set(fw)) * np.log2(max(M, 1))
    tc = (T / NF * np.log2(np.where(T > 0, T * NF * NF / (T.sum((1, 2), keepdims=True) * T.sum((0, 2), keepdims=True) * T.sum((0, 1), keepdims=True)), 1))).sum()
    return d + model, d, model, tc


def purity(gn, wts):
    tot = 0.0
    for g in np.unique(gn):
        m = gn == g; tot += max(wts[m & (GT == t)].sum() for t in np.unique(GT[m]))
    return tot / wts.sum()


nocc = np.bincount(a, minlength=NN) + np.bincount(b, minlength=NN)
d0 = dl(np.zeros(NN, int), np.zeros(NW, int), 1, 1)
P(f"NO CLASSES: {d0[0] / NF:.2f} bits a record  ({time.time() - T0:.0f}s)")
P(f"gold types by records: " + ", ".join(f"{t} {nocc[GT == t].sum() / nocc.sum():.2f}" for t in np.unique(GT)))
FITS = {}
INIT = os.environ.get('RM_INIT', 'random')                                      # 13 Nov 18:20: random | split | nt:<json>:<column>
if INIT.startswith('nt:'):                                                      #   name classes started from context types (tools/name_types.py)
    _p, _c = INIT[3:].rsplit(':', 1); _nt = json.load(open(_p))['types']; NT0 = np.array([_nt.get(n, [0] * 9)[int(_c)] for n in NAMES])
for K in KS:
    for M in MS:
        best = None; prev = [km for km in FITS if km[0] <= K and km[1] <= M]
        for s in range(3):
            ini = None
            if INIT == 'split' and prev:                                       # grow from the largest smaller fit: each class split at random
                K0, M0 = max(prev, key=lambda km: km[0] * km[1]); g0, h0 = FITS[K0, M0]
                ini = lambda r, g0=g0, h0=h0, K0=K0, M0=M0: (g0 + K0 * r.randint(K // K0, size=NN), h0 + M0 * r.randint(M // M0, size=NW))
            elif INIT.startswith('nt:'):
                ini = lambda r: (NT0 % K, r.randint(M, size=NW))
            gn, gw = fit(K, M, s, init=ini); D = dl(gn, gw, K, M)
            if best is None or D[0] < best[2][0]: best = (gn, gw, D)
        gn, gw, D = best; FITS[(K, M)] = (gn, gw)
        P(f"   K {K:2d} name classes, M {M:3d} wording classes: total {D[0] / NF:.2f} bits a record (data {D[1] / NF:.2f} + model {D[2] / NF:.2f}) | "
          f"class triple shares {D[3]:.3f} bits | name classes vs gold types: purity {purity(gn, nocc):.3f}, NMI "
          f"{normalized_mutual_info_score(np.repeat(GT, nocc), np.repeat(gn, nocc)):.3f}  ({time.time() - T0:.0f}s)")
KBEST = min(FITS, key=lambda km: dl(*FITS[km], *km)[0]); P(f"MDL choice: K {KBEST[0]}, M {KBEST[1]}")
json.dump(dict(k=[f'{K}x{M}' for K, M in FITS], types={n: [int(FITS[km][0][i]) for km in FITS] for i, n in enumerate(NAMES)}), open(OUT, 'w'))
gn, gw = FITS[KBEST]
for g in range(KBEST[0]):
    m = gn == g; tc_ = collections.Counter(dict((t, int(nocc[m & (GT == t)].sum())) for t in np.unique(GT[m])))
    ex = [NAMES[i] for i in np.argsort(-nocc * m)[:5]]
    P(f"   name class {g}: {int(m.sum())} names, {int(nocc[m].sum())} records | gold {dict(tc_.most_common(3))} | e.g. {ex}")
for h in np.argsort(-np.bincount(gw[w], minlength=KBEST[1]))[:12]:
    m = gw[w] == h; ex = [k for k, _ in collections.Counter(wk(C[i]) for i in np.where(m)[0]).most_common(6)]
    P(f"   wording class {h}: {int(m.sum())} records, related {y[m].mean():.2f} | e.g. {ex}")
# ---- RELATION identity on held-out gold-related records
RM = {}
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]; R = collections.defaultdict(set)
    for l in d['labels']: R[(cn[l['h']], cn[l['t']])].add(l['r']); R[(cn[l['t']], cn[l['h']])].add(l['r'] + '^-1')
    RM[di] = R
HR = [i for i in TE if y[i]]; RS = [RM[C[i]['di']][(C[i]['cA'], C[i]['cB'])] for i in HR]
rq = np.random.RandomState(1); qs = rq.choice(len(HR), 600, replace=False); PAIRS = []
for q in qs:
    cs = [int(c) for c in rq.choice(len(HR), 40, replace=False) if C[HR[int(c)]]['di'] != C[HR[q]]['di']][:30]; PAIRS += [(int(q), c) for c in cs]
same = np.array([bool(RS[i] & RS[j]) for i, j in PAIRS])
def report(nm_, sim):
    top, ch = [], []
    for n0 in range(0, len(PAIRS), 30):
        blk = list(range(n0, min(n0 + 30, len(PAIRS)))); bb = max(blk, key=lambda n: sim[n] + 1e-9 * rq.rand()); top.append(same[bb]); ch.append(np.mean(same[blk]))
    P(f"   {nm_:52s} top-1 same relation {np.mean(top):.3f} (chance {np.mean(ch):.3f}) | AUC {auc(sim[same], sim[~same]):.3f}")
P(f"RELATION IDENTITY: {len(qs)} held-out queries x 30 candidates (held-out gold-related records {len(HR)})")
gtp = lambda i: (GT[a[HR[i]]], GT[b[HR[i]]])
eq = lambda f: np.array([float(f(i) == f(j)) for i, j in PAIRS])
report('GOLD entity-type pair (reference)', eq(gtp)); report('exact wording (backed off)', eq(lambda i: w[HR[i]]))
for km, (gn, gw) in FITS.items():
    if km[1] != KBEST[1] and km != KBEST and km[0] != KBEST[0]: continue
    ncp = eq(lambda i: (gn[a[HR[i]]], gn[b[HR[i]]])); wcl = eq(lambda i: gw[w[HR[i]]])
    report(f'K {km[0]} M {km[1]}: name-class pair', ncp); report(f'K {km[0]} M {km[1]}: wording class', wcl)
    report(f'K {km[0]} M {km[1]}: name-class pair + wording class', ncp + wcl)
# ---- PREMISES: label-free record scores on held-out records
rk = lambda x: np.argsort(np.argsort(x)) / len(x)
def prec(tag, sc, ix=TE):
    yy = y[ix]; s = sc[ix]; o = np.argsort(-s + 1e-9 * rq.rand(len(s)))
    P(f"   {tag:60s} AUC {auc(s[yy], s[~yy]):.3f} | precision top 5% {yy[o[:len(s) // 20]].mean():.3f}, 10% {yy[o[:len(s) // 10]].mean():.3f}, "
      f"25% {yy[o[:len(s) // 4]].mean():.3f}")
P(f"PREMISES (held-out records {len(TE)}, related {yt.mean():.3f}; target precision 0.6-0.7):")
prec('consistency-teacher learner (label-free, before)', s_all)
SC = {}
for km, (gn, gw) in FITS.items():
    K, M = km; T = table(gn, gw, K, M, 0.1); pg = T.sum((0, 2)) / T.sum()
    s_pmi = np.log(T[gn[a], gw[w], gn[b]] / T[gn[a], :, gn[b]].sum(-1)) - np.log(pg[gw[w]])           # log p(w cl | A cl, B cl) / p(w cl)
    Hc = T / T.sum((0, 2), keepdims=True); Hw = -(Hc * np.log(Hc)).sum((0, 2))                          # how selective the wording class is
    s_sel = -Hw[gw[w]]
    SC[km] = (s_pmi, s_sel)
    if km == KBEST or km == (max(KS), max(MS)):
        prec(f'K {K} M {M}: block association p(w cl | name classes) / p(w cl)', s_pmi); prec(f'K {K} M {M}: wording-class selectivity', s_sel)
        prec(f'K {K} M {M}: association + selectivity', rk(s_pmi) + rk(s_sel))
        prec(f'K {K} M {M}: association + learner (rank sum)', rk(s_pmi) + rk(s_all))
        tch = rk(s_pmi[TR]) + rk(s_sel[TR]) + rk(teacher_tr)
        s_l = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, tch >= np.percentile(tch, 75)).decision_function(XA)
        prec(f'K {K} M {M}: LEARNER from teachers + block scores', s_l); SC[km] = (s_pmi, s_sel, s_l)
        prec(f'K {K} M {M}: that learner + association (rank sum)', rk(s_l) + rk(s_pmi))
# ---- INFERENCE: A-C from the store kept by the best label-free score
GOLDP = set(); ANN = set(); dfreq = collections.Counter()
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    for l in d['labels']: GOLDP.add(frozenset((cn[l['h']], cn[l['t']])))
    ns = set(cn); ANN |= {frozenset((x, z)) for x in ns for z in ns if x != z}
    for n in ns: dfreq[n] += 1
def infer(tag, sc, frac):
    keep = np.zeros(N, bool); keep[np.argsort(-sc)[:int(N * frac)]] = True; nb = collections.defaultdict(list); PR = collections.defaultdict(set)
    for i in np.where(keep)[0]: c = C[i]; nb[c['cA']].append((c['cB'], c['di'])); nb[c['cB']].append((c['cA'], c['di']))
    for i in np.where(keep)[0]:
        c = C[i]
        for B_, Cn in ((c['cA'], c['cB']), (c['cB'], c['cA'])):
            if dfreq[B_] > 200: continue
            for A_, dA in nb[B_]:
                if dA != c['di'] and A_ != Cn: PR[frozenset((A_, Cn))].add(B_)
    ks = [k for k in PR if k in ANN]; g = np.array([k in GOLDP for k in ks]); b2 = np.array([len(PR[k]) >= 2 for k in ks])
    P(f"   {tag:44s} keep {frac:.0%}: premise precision {y[keep].mean():.3f} | A-C proposals {len(ks)}, gold {g.mean():.3f} | 2+ bridges {int(b2.sum())}, "
      f"gold {g[b2].mean() if b2.any() else float('nan'):.3f}")
P("INFERENCE (A-C through a shared name, store kept by a label-free score; all records):")
sb = SC[KBEST]
for frac in (0.25, 0.10, 0.05):
    infer('consistency-teacher learner', s_all, frac); infer(f'block association (K {KBEST[0]} M {KBEST[1]})', sb[0], frac)
    infer('learner + association', rk(s_all) + rk(sb[0]), frac)
    if len(sb) > 2: infer('learner from teachers + block scores', sb[2], frac)
P(f"DONE ({time.time() - T0:.0f}s)")
