"""22 Nov RELATION MERGING BY JEPA OPERATORS (offline, real records) - gap 3: raw connecting words ('in', 'on', 'of', 'until') are too
ambiguous to compose (realrules: one rule above the shuffled null), and real records are about one fact per name per wording.

Route J (name codes + one JEPA-style predictor per raw wording, InfoNCE over the names) is fitted on the records. Two wordings are the
SAME RELATION when their operators predict the same codes from the same heads (operator agreement: mean cosine of the two predictions over
the heads of either wording) - a label-free, latent criterion. Wordings are merged by average-linkage clustering at a threshold.
Compared with (a) no merging, (b) a symbolic merge (Jaccard of the wordings' (head, tail) pairs). Evaluation:
  1. PURITY against Wikidata: each wording's dominant gold property (over its gold-labelled facts); a cluster is pure when its wordings
     share it (fact-weighted).
  2. RULES r1 o r2 => r3 above a shuffled null, over raw wordings vs merged relations.
  3. ROUTE J HELD-OUT one-hop (never-trained facts), raw vs merged relations (merged = more facts per operator), all heads and heads
     with other training facts.
Usage: python3 tools/relmerge.py <distant_set.json.gz>  (env RM_NDOC, RM_K wordings, RM_STEPS, RM_TH merge threshold)"""
import os, sys, types, collections, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from platypus.inference import premises as pm
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2); T0 = time.time()
P = lambda s: print(s, flush=True)
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0, inf_premises_seed_ctx=0, inf_premises_wide=0)
pm._load_distant(cfg); R = pm._PM['R']; PRD = pm._PM.get('PRD') or {}
NDOC = int(os.environ.get('RM_NDOC', 20000)); K = int(os.environ.get('RM_K', 40)); STEPS = int(os.environ.get('RM_STEPS', 3000))
TH = float(os.environ.get('RM_TH', 0.5)); D = 64
LISTW = set(getattr(pm, '_RRET_LISTW', ())) | {'and', 'or', 'with', 'to', ',', 'the', 'of the', 'from', 'for'}
fc = collections.Counter()
for r in R:
    if r[0] >= NDOC or not (1 <= r[5] <= 8) or r[1] == r[2]: continue
    w = pm._rnorm(r[3])
    if not w or w in LISTW or not all(t.isalpha() for t in w.split()) or len(w.split()) > 4: continue
    fc[(r[1], w, r[2])] += 1
wc = collections.Counter(w for (_, w, _) in fc); W = [w for w, _ in wc.most_common(K)]; WS = set(W); WI = {w: j for j, w in enumerate(W)}
FACT = sorted(k for k in fc if k[1] in WS); NM = sorted({a for a, w, b in FACT} | {b for a, w, b in FACT}); NI = {n: i for i, n in enumerate(NM)}
gold = lambda a, b: PRD.get((a, b)) or set()
P(f"{len(FACT)} facts in the {K} commonest wordings from {NDOC} documents, {len(NM)} names ({time.time() - T0:.0f}s)")
rs = np.random.RandomState(7); hold = set(int(i) for i in rs.choice(len(FACT), len(FACT) // 10, replace=False))
TR = [f for i, f in enumerate(FACT) if i not in hold]; HO = [f for i, f in enumerate(FACT) if i in hold]
def fit(rel_of, facts, steps, seed=0):
    """route J with one operator per relation id (rel_of: wording -> relation id)"""
    torch.manual_seed(seed); nr = max(rel_of.values()) + 1
    E = torch.nn.Parameter(torch.randn(len(NM), D) * 0.1)
    OPS = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.GELU(), torch.nn.Linear(256, D)) for _ in range(nr))
    opt = torch.optim.Adam([E] + list(OPS.parameters()), lr=3e-3)
    A_ = torch.tensor([NI[a] for a, w, b in facts]); R_ = torch.tensor([rel_of[w] for a, w, b in facts]); B_ = torch.tensor([NI[b] for a, w, b in facts])
    for st in range(steps):
        k = torch.randint(0, len(facts), (512,)); l = F.cross_entropy(pred(E, OPS, A_[k], R_[k]) @ F.normalize(E, dim=-1).T / 0.05, B_[k]); opt.zero_grad(); l.backward(); opt.step()
    return E.detach(), OPS.eval()
def pred(E, OPS, ai, ri, z=None):
    z = F.normalize(E[ai], dim=-1) if z is None else z; out = torch.zeros_like(z)
    for j in ri.unique().tolist():
        m = ri == j; out[m] = OPS[j](z[m])
    return F.normalize(out, dim=-1)
@torch.no_grad()
def heldout(E, OPS, rel_of):
    hd = collections.Counter(a for a, w, b in TR); EN = F.normalize(E, dim=-1); ok = []; cv = []
    for i in range(0, len(HO), 1024):
        ch = HO[i:i + 1024]; a = torch.tensor([NI[x[0]] for x in ch]); r = torch.tensor([rel_of[x[1]] for x in ch])
        t = (pred(E, OPS, a, r) @ EN.T).argmax(-1).tolist(); ok += [NM[j] == x[2] for j, x in zip(t, ch)]; cv += [hd[x[0]] > 0 for x in ch]
    ok = np.array(ok); cv = np.array(cv); return ok.mean(), ok[cv].mean() if cv.any() else float('nan'), int(cv.sum())
# 1. operators per raw wording -> agreement matrix
E0, OPS0 = fit({w: WI[w] for w in W}, TR, STEPS)
h0 = heldout(E0, OPS0, {w: WI[w] for w in W})
heads = {w: sorted({a for a, w_, b in TR if w_ == w}) for w in W}
AG = np.eye(len(W))
with torch.no_grad():
    for i, w1 in enumerate(W):
        for j in range(i + 1, len(W)):
            w2 = W[j]; hs = (heads[w1][:300] + heads[w2][:300]); a = torch.tensor([NI[h] for h in hs])
            p1 = pred(E0, OPS0, a, torch.full_like(a, i)); p2 = pred(E0, OPS0, a, torch.full_like(a, j)); AG[i, j] = AG[j, i] = float((p1 * p2).sum(-1).mean())
pairs = {w: {(a, b) for a, w_, b in TR if w_ == w} for w in W}
JA = np.eye(len(W))
for i, w1 in enumerate(W):
    for j in range(i + 1, len(W)):
        u = len(pairs[w1] | pairs[W[j]]); JA[i, j] = JA[j, i] = len(pairs[w1] & pairs[W[j]]) / u if u else 0.0
def cluster(S, th=None, n=None):
    """average-linkage agglomerative clustering on a similarity matrix, merging while the best pair's average similarity >= th
    (or until n clusters remain)"""
    cl = [[i] for i in range(len(S))]
    while True:
        if n is not None and len(cl) <= n: return cl
        best = None
        for x in range(len(cl)):
            for y in range(x + 1, len(cl)):
                s = np.mean([S[i, j] for i in cl[x] for j in cl[y]])
                if best is None or s > best[0]: best = (s, x, y)
        if best is None or (n is None and best[0] < th): return cl
        s, x, y = best; cl[x] = cl[x] + cl[y]; del cl[y]
# 2. purity against Wikidata
gw = {}
for w in W:
    c = collections.Counter(p for a, w_, b in TR if w_ == w for p in gold(a, b)); gw[w] = c.most_common(1)[0][0] if c else None
def purity(cl):
    tot = good = 0
    for c in cl:
        ws = [W[i] for i in c]; dom = collections.Counter(gw[w] for w in ws if gw[w]).most_common(1)
        for w in ws:
            if gw[w]: n = wc[w]; tot += n; good += n * (gw[w] == dom[0][0])
    return good / tot if tot else float('nan')
def rules(rel_of):
    by = collections.defaultdict(lambda: collections.defaultdict(set))
    for a, w, b in TR: by[rel_of[w]][a].add(b)
    rels = sorted(by)
    def stats(shuffle):
        out = []
        for r2 in rels:
            m2 = by[r2]
            if shuffle:
                ks = list(m2); vs = [m2[k] for k in ks]; np.random.RandomState(3).shuffle(vs); m2 = dict(zip(ks, vs))
            for r1 in rels:
                ch = collections.defaultdict(set)
                for a, bs in by[r1].items():
                    for b in bs:
                        for c in m2.get(b, ()):
                            if c != a: ch[a].add(c)
                for r3 in rels:
                    sup = den = 0
                    for a, cs in ch.items():
                        if a in by[r3]: den += len(cs); sup += len(cs & by[r3][a])
                    if den >= 20: out.append((sup / den, sup, den, r1, r2, r3))
        return out
    rs_ = stats(False); nm = max([x[0] for x in stats(True)] + [0.0]); ab = sorted([x for x in rs_ if x[0] > nm], reverse=True)
    return nm, ab
P(f"1. RAW WORDINGS ({len(W)}): route J HELD-OUT one-hop {h0[0]:.4f} | heads with other training facts ({h0[2]}) {h0[1]:.4f} | gold purity (each wording alone) {purity([[i] for i in range(len(W))]):.3f}")
nm0, ab0 = rules({w: WI[w] for w in W}); P(f"   RULES above the shuffled null ({nm0:.3f}): {len(ab0)}")
for th in (TH, TH - 0.15):
    ncl = len(cluster(AG, th))
    for nam, S in (('JEPA OPERATOR AGREEMENT', AG), ('SYMBOLIC (Jaccard of pairs), same number of relations', JA)):
        cl = cluster(S, th) if nam.startswith('JEPA') else cluster(S, n=ncl); th_ = th; rel_of = {W[i]: k for k, c in enumerate(cl) for i in c}
        big = sorted(cl, key=lambda c: -sum(wc[W[i]] for i in c))[:4]
        E1, OPS1 = fit(rel_of, TR, STEPS); h1 = heldout(E1, OPS1, rel_of); nm1, ab1 = rules(rel_of)
        P(f"2. MERGED by {nam}, JEPA threshold {th_:.2f}: {len(cl)} relations | gold purity {purity(cl):.3f} | route J HELD-OUT {h1[0]:.4f}, heads with other facts {h1[1]:.4f} | "
          f"RULES above the null ({nm1:.3f}) {len(ab1)} | largest: " + ' ; '.join('{' + ', '.join(W[i] for i in c[:6]) + '}' for c in big) + f"  ({time.time() - T0:.0f}s)")
        for c_, s_, d_, r1, r2, r3 in ab1[:3]:
            nmr = lambda r: '/'.join(W[i] for i in cl[r][:3])
            P(f"      rule '{nmr(r1)}' o '{nmr(r2)}' => '{nmr(r3)}': {c_:.3f} ({s_} / {d_})")
P(f"DONE ({time.time() - T0:.0f}s)")
