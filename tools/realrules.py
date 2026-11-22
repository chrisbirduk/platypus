"""22 Nov: ROUTE J ON REAL TEXT (offline, no trunk) and a RULE-CONSISTENCY retraction signal.

Records of the distant Wikipedia set (premises.py's record store: 'A <words> B' in one sentence, 1-8 words apart), wordings normalised.
1. RULES, symbolic (AMIE-style): r1 o r2 => r3 among the frequent wordings - support and PCA confidence, against a null with r2's tails shuffled.
2. ROUTE J, latent: name codes + one JEPA-style predictor per wording (MLP, InfoNCE over the names, as route J), fitted on 90% of the facts;
   held-out one-hop, and composites A -> C through the best rules scored against (a) held-out recorded r3 facts, (b) GOLD Wikidata containment.
3. RETRACTION SIGNAL: held-out recorded facts scored by the predictor's probability of their tail (how well the rest of the knowledge
   predicts them); AUC for gold-labelled facts vs facts with no gold label (a proxy for wrong records), against the number-of-documents baseline.
Usage: python3 tools/realrules.py <distant_set.json.gz>   (env RR_NDOC documents, RR_K wordings, RR_STEPS predictor steps)"""
import os, sys, types, collections, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from platypus.inference import premises as pm
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2); T0 = time.time()
P = lambda s: print(s, flush=True)
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0, inf_premises_seed_ctx=0, inf_premises_wide=0)
pm._load_distant(cfg); R = pm._PM['R']; PRD = pm._PM.get('PRD') or {}; ND = pm._PM['ND']
NDOC = int(os.environ.get('RR_NDOC', ND)); K = int(os.environ.get('RR_K', 30)); STEPS = int(os.environ.get('RR_STEPS', 3000))
LISTW = set(getattr(pm, '_RRET_LISTW', ())) | {'and', 'or', 'with', 'to', ',', 'the', 'of the', 'from', 'for'}
GC = {'P131', 'P17', 'P30', 'P361', 'P276', 'P150', 'P706'}
fc = collections.Counter(); docs = collections.defaultdict(set); lab = {}
for r in R:
    if r[0] >= NDOC or not (1 <= r[5] <= 8) or r[1] == r[2]: continue
    w = pm._rnorm(r[3])
    if not w or w in LISTW or not all(t.isalpha() for t in w.split()) or len(w.split()) > 4: continue
    k = (r[1], w, r[2]); fc[k] += 1; docs[k].add(r[0]); lab[k] = lab.get(k, False) or bool(r[6])
wc = collections.Counter(w for (_, w, _) in fc); W = [w for w, _ in wc.most_common(K)]; WS = set(W)
FACT = [k for k in fc if k[1] in WS]
P(f"records read from {NDOC} documents: {len(fc)} distinct facts, {len(FACT)} in the {K} commonest wordings ({time.time() - T0:.0f} s): {', '.join(W[:K])}")
# 1. RULES (symbolic)
by = {w: collections.defaultdict(set) for w in W}
for (a, w, b) in FACT: by[w][a].add(b)
def rule_stats(shuffle=False):
    out = []
    for r2 in W:
        m2 = by[r2]
        if shuffle:
            ks = list(m2); vs = [m2[k] for k in ks]; np.random.RandomState(3).shuffle(vs); m2 = dict(zip(ks, vs))
        for r1 in W:
            ch = collections.defaultdict(set)
            for a, bs in by[r1].items():
                for b in bs:
                    for c in m2.get(b, ()):
                        if c != a: ch[a].add(c)
            if not ch: continue
            for r3 in W:
                m3 = by[r3]; sup = 0; den = 0
                for a, cs in ch.items():
                    if a in m3: den += len(cs); sup += len(cs & m3[a])
                if den >= 20: out.append((sup / den, sup, den, r1, r2, r3))
    return out
RS = rule_stats(); RN = rule_stats(True); nullmax = max([x[0] for x in RN] + [0.0])
RS = sorted(RS, reverse=True)
P(f"1. RULES r1 o r2 => r3 (PCA confidence = chains whose conclusion is recorded as r3 / chains from heads with any r3 fact; >= 20 chains); "
  f"NULL (r2's tails shuffled) best {nullmax:.3f}")
for c, s, d, r1, r2, r3 in RS[:12]: P(f"   '{r1}' o '{r2}' => '{r3}': {c:.3f} ({s} / {d}){'  ABOVE NULL' if c > nullmax else ''}")
RULES = [(r1, r2, r3) for c, s, d, r1, r2, r3 in RS if c > nullmax][:5]
# 2. ROUTE J (latent)
rs = np.random.RandomState(7); hold = set(int(i) for i in rs.choice(len(FACT), len(FACT) // 10, replace=False))
TR = [f for i, f in enumerate(FACT) if i not in hold]; HO = [f for i, f in enumerate(FACT) if i in hold]
NM = sorted({a for a, w, b in FACT} | {b for a, w, b in FACT}); NI = {n: i for i, n in enumerate(NM)}; D = 64
E = torch.nn.Parameter(torch.randn(len(NM), D) * 0.1)
OPS = torch.nn.ModuleDict({str(j): torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.GELU(), torch.nn.Linear(256, D)) for j in range(len(W))})
WI = {w: j for j, w in enumerate(W)}; opt = torch.optim.Adam([E] + list(OPS.parameters()), lr=3e-3)
A_ = torch.tensor([NI[a] for a, w, b in TR]); R_ = torch.tensor([WI[w] for a, w, b in TR]); B_ = torch.tensor([NI[b] for a, w, b in TR])
def pred(ai, ri):
    z = F.normalize(E[ai], dim=-1); out = torch.zeros_like(z)
    for j in ri.unique().tolist():
        m = ri == j; out[m] = OPS[str(j)](z[m])
    return F.normalize(out, dim=-1)
for st in range(STEPS):
    k = torch.randint(0, len(TR), (512,)); p_ = pred(A_[k], R_[k])
    l = F.cross_entropy(p_ @ F.normalize(E, dim=-1).T / 0.05, B_[k]); opt.zero_grad(); l.backward(); opt.step()
    if st % 1000 == 0: P(f"   predictor step {st}: loss {float(l):.3f}")
with torch.no_grad():
    EN = F.normalize(E, dim=-1)
    def top(ai, ri):                                                           # top-1 name, in chunks (a names x heads matrix is large)
        return torch.cat([(pred(ai[i:i + 512], ri[i:i + 512]) @ EN.T).argmax(-1, keepdim=True) for i in range(0, len(ai), 512)])
    def hits(L):
        if not L: return float('nan')
        a = torch.tensor([NI[x[0]] for x in L]); r = torch.tensor([WI[x[1]] for x in L]); b = torch.tensor([NI[x[2]] for x in L])
        return float((top(a, r)[:, 0] == b).float().mean())
    P(f"2. ROUTE J on real records: names {len(NM)}, facts trained {len(TR)}, held out {len(HO)} | one-hop TRAIN {hits(TR[:3000]):.3f} | HELD-OUT (never trained) {hits(HO):.3f} (chance {1 / len(NM):.5f})")
    HOS = set(HO)
    for (r1, r2, r3) in RULES:
        heads = [a for a in by[r1] if by[r1][a]][:3000]
        if not heads: continue
        a = torch.tensor([NI[x] for x in heads]); b1 = top(a, torch.full_like(a, WI[r1]))[:, 0]; c1 = top(b1, torch.full_like(a, WI[r2]))[:, 0]
        comp = [(h, NM[int(c)]) for h, c in zip(heads, c1) if NM[int(c)] != h]
        held3 = [(h, c) for h, c in comp if any((h, r3, x) in HOS for x in by[r3].get(h, ()))]
        rec3 = [c in by[r3].get(h, ()) for h, c in held3]
        g = [(h, c) for h, c in comp if (PRD.get((h, c)) or set()) | (PRD.get((c, h)) or set())]
        gt = [bool((PRD.get((h, c)) or set()) & GC) for h, c in g]
        P(f"   COMPOSITES by '{r1}' o '{r2}' => '{r3}' (latent, two operator hops, snapped): {len(comp)} | heads with a HELD-OUT '{r3}' fact: recorded tail reached {np.mean(rec3) if rec3 else float('nan'):.3f} "
          f"(n {len(rec3)}) | pairs with any gold relation {len(g)}: gold CONTAINMENT {np.mean(gt) if gt else float('nan'):.3f} | e.g. " + '; '.join(f"{h} -> {c}" for h, c in comp[:4]))
    # 3. RETRACTION SIGNAL
    if HO:
        a = torch.tensor([NI[x[0]] for x in HO]); r = torch.tensor([WI[x[1]] for x in HO]); b = torch.tensor([NI[x[2]] for x in HO])
        lp = np.concatenate([F.log_softmax(pred(a[i:i + 512], r[i:i + 512]) @ EN.T / 0.05, -1).gather(-1, b[i:i + 512, None])[:, 0].numpy() for i in range(0, len(a), 512)])
        y = np.array([lab.get(f, False) for f in HO]); nd = np.array([len(docs[f]) for f in HO])
        from sklearn.metrics import roc_auc_score
        if 0 < y.sum() < len(y):
            P(f"3. RETRACTION SIGNAL on held-out records ({len(HO)}; gold-labelled {y.mean():.3f}): AUC gold vs unlabelled - rest-of-knowledge prediction (log p of the tail) "
              f"{roc_auc_score(y, lp):.3f} | number of documents {roc_auc_score(y, nd):.3f} | both (rank sum) {roc_auc_score(y, lp.argsort().argsort() + nd.argsort().argsort()):.3f}")
P(f"DONE ({time.time() - T0:.0f} s)")
