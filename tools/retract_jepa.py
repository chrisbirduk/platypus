"""22 Nov RETRACTION BY JEPA SURPRISE (offline, real records; after Garrido, LeCun et al. 2025 - surprise as latent prediction error;
Hristov et al. 2026 - controls for confounds; Causal-JEPA - mask one entity, infer it from the rest; CCA (AAAI 2024) - two-view consistency;
VJEPA - uncertainty, so 'unsure' is not 'contradicted').

Records of the distant Wikipedia set as in tools/realrules.py ('A <words> B', commonest K wordings). A known fraction of the facts is
CORRUPTED (the tail replaced by the tail of another random fact of the same wording - frequency-matched, so a rare tail is no giveaway);
the corrupted fact keeps its source's document count. Route J (name codes + one JEPA-style predictor per wording, InfoNCE over the names)
is trained K-fold: each fact is scored by models that never saw it (leave-fold-out = the masked entity inferred from the rest), with
several seeds. Signals (higher = more suspect):
  S1 SURPRISE        - log p of the recorded tail (negated)
  S2 SURPRISE, NORMALISED - S1 minus the tail's own prior under the wording (the 'target difficulty' control)
  S3 PATH CONSISTENCY - (a) latent: the best rule's two-hop composite (route J, bridge snapped) log p of the tail; (b) symbolic: 2-hop
                       paths a -> x -> tail in the training graph
  S4 CONFIDENT CONTRADICTION - the seeds agree on another tail with high probability (uncertainty-gated)
Baselines: documents, tail prior, head degree. AUC corrupted vs clean (all facts; facts whose head has other training facts), precision at
the noise rate; and the real-data proxy (gold-labelled vs unlabelled clean facts).
Usage: python3 tools/retract_jepa.py <distant_set.json.gz>  (env RJ_NDOC, RJ_K, RJ_NOISE, RJ_FOLDS, RJ_SEEDS, RJ_STEPS)"""
import os, sys, types, collections, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from platypus.inference import premises as pm
from sklearn.metrics import roc_auc_score
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2); T0 = time.time()
P = lambda s: print(s, flush=True)
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0, inf_premises_seed_ctx=0, inf_premises_wide=0)
pm._load_distant(cfg); R = pm._PM['R']; ND = pm._PM['ND']
NDOC = int(os.environ.get('RJ_NDOC', 20000)); K = int(os.environ.get('RJ_K', 30)); NOISE = float(os.environ.get('RJ_NOISE', 0.05))
FO = int(os.environ.get('RJ_FOLDS', 4)); SE = int(os.environ.get('RJ_SEEDS', 2)); STEPS = int(os.environ.get('RJ_STEPS', 1000)); D = 64
LISTW = set(getattr(pm, '_RRET_LISTW', ())) | {'and', 'or', 'with', 'to', ',', 'the', 'of the', 'from', 'for'}
fc = collections.Counter(); docs = collections.defaultdict(set); lab = {}
for r in R:
    if r[0] >= NDOC or not (1 <= r[5] <= 8) or r[1] == r[2]: continue
    w = pm._rnorm(r[3])
    if not w or w in LISTW or not all(t.isalpha() for t in w.split()) or len(w.split()) > 4: continue
    k = (r[1], w, r[2]); fc[k] += 1; docs[k].add(r[0]); lab[k] = lab.get(k, False) or bool(r[6])
wc = collections.Counter(w for (_, w, _) in fc); W = [w for w, _ in wc.most_common(K)]; WS = set(W); WI = {w: j for j, w in enumerate(W)}
FACT = sorted(k for k in fc if k[1] in WS)
# CORRUPTION (frequency-matched tails)
rs = np.random.RandomState(1); TRUE = set(FACT); byw = collections.defaultdict(list)
for f in FACT: byw[f[1]].append(f)
cidx = set(int(i) for i in rs.choice(len(FACT), int(NOISE * len(FACT)), replace=False)); F_ = []
for i, (a, w, b) in enumerate(FACT):
    nd = len(docs[(a, w, b)]); gl = bool(lab.get((a, w, b)))
    if i in cidx:
        for _ in range(30):
            b2 = byw[w][rs.randint(len(byw[w]))][2]
            if b2 not in (a, b) and (a, w, b2) not in TRUE: F_.append((a, w, b2, 1, nd, False)); break
        continue
    F_.append((a, w, b, 0, nd, gl))
y = np.array([f[3] for f in F_]); NM = sorted({f[0] for f in F_} | {f[2] for f in F_}); NI = {n: i for i, n in enumerate(NM)}
P(f"facts from {NDOC} documents in the {K} commonest wordings: {len(F_)} ({int(y.sum())} CORRUPTED, {y.mean():.3f}), names {len(NM)} ({time.time() - T0:.0f}s)")
A_ = torch.tensor([NI[f[0]] for f in F_]); R_ = torch.tensor([WI[f[1]] for f in F_]); B_ = torch.tensor([NI[f[2]] for f in F_])
fold = np.random.RandomState(2).randint(0, FO, len(F_))
def train(tr, seed):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    E = torch.nn.Parameter(torch.randn(len(NM), D, generator=g) * 0.1)
    OPS = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.GELU(), torch.nn.Linear(256, D)) for _ in W)
    opt = torch.optim.Adam([E] + list(OPS.parameters()), lr=3e-3); tr = torch.tensor(tr)
    for st in range(STEPS):
        k = tr[torch.randint(0, len(tr), (512,), generator=g)]; p_ = pred(E, OPS, A_[k], R_[k])
        l = F.cross_entropy(p_ @ F.normalize(E, dim=-1).T / 0.05, B_[k]); opt.zero_grad(); l.backward(); opt.step()
    return E.detach(), OPS.eval()
def pred(E, OPS, ai, ri, z=None):
    z = F.normalize(E[ai], dim=-1) if z is None else z; out = torch.zeros_like(z)
    for j in ri.unique().tolist():
        m = ri == j; out[m] = OPS[j](z[m])
    return F.normalize(out, dim=-1)
# rules for the path signal (symbolic, PCA confidence on the clean+corrupted graph; best per target wording)
by = {w: collections.defaultdict(set) for w in W}
for (a, w, b, c, nd, gl) in F_: by[w][a].add(b)
best = {}
for r1 in W:
    for r2 in W:
        ch = collections.defaultdict(set)
        for a, bs in by[r1].items():
            for b in bs:
                for c in by[r2].get(b, ()):
                    if c != a: ch[a].add(c)
        for r3 in W:
            sup = den = 0
            for a, cs in ch.items():
                if a in by[r3]: den += len(cs); sup += len(cs & by[r3][a])
            if den >= 20 and (r3 not in best or sup / den > best[r3][0]): best[r3] = (sup / den, r1, r2)
P("   best rule per wording (PCA confidence): " + '; '.join(f"'{r1}' o '{r2}' => '{r3}' {c:.2f}" for r3, (c, r1, r2) in sorted(best.items(), key=lambda x: -x[1][0])[:8]))
LP = np.zeros((SE, len(F_))); PT = np.zeros((SE, len(F_))); TT = np.zeros((SE, len(F_)), dtype=int); LC = np.full((SE, len(F_)), np.nan)
for s in range(SE):
    for f_ in range(FO):
        te = np.where(fold == f_)[0]; E, OPS = train(np.where(fold != f_)[0].tolist(), 100 * s + f_)
        with torch.no_grad():
            EN = F.normalize(E, dim=-1)
            for i in range(0, len(te), 1024):
                ix = torch.tensor(te[i:i + 1024]); lg = F.log_softmax(pred(E, OPS, A_[ix], R_[ix]) @ EN.T / 0.05, -1)
                LP[s, ix.numpy()] = lg.gather(-1, B_[ix, None])[:, 0].numpy(); mx = lg.max(-1); PT[s, ix.numpy()] = mx.values.exp().numpy(); TT[s, ix.numpy()] = mx.indices.numpy()
                grp = collections.defaultdict(list)                            # latent two-hop composite through the wording's best rule
                for j, r in enumerate(R_[ix].tolist()):
                    if W[r] in best: grp[best[W[r]][1:]].append(j)
                for (r1, r2), js in grp.items():
                    jj = ix[torch.tensor(js)]; v = pred(E, OPS, A_[jj], torch.full((len(js),), WI[r1])); bi = (v @ EN.T).argmax(-1)
                    v2 = pred(E, OPS, bi, torch.full((len(js),), WI[r2])); LC[s, jj.numpy()] = F.log_softmax(v2 @ EN.T / 0.05, -1).gather(-1, B_[jj, None])[:, 0].numpy()
        P(f"   seed {s} fold {f_}: trained on {int((fold != f_).sum())}, scored {len(te)} ({time.time() - T0:.0f}s)")
lp = LP.mean(0); nd = np.array([f[4] for f in F_], dtype=float)
tc = collections.Counter((f[1], f[2]) for f in F_); nw = collections.Counter(f[1] for f in F_); ntw = collections.Counter(w_ for (w_, b_) in set((f[1], f[2]) for f in F_))
prior = np.array([np.log((tc[(f[1], f[2])] + 1) / (nw[f[1]] + ntw[f[1]])) for f in F_])
hdeg = collections.Counter(f[0] for f in F_); deg = np.array([hdeg[f[0]] for f in F_], dtype=float)
out = collections.defaultdict(set)
for f in F_: out[f[0]].add(f[2])
paths = np.array([sum(1 for x in out[f[0]] if x != f[2] and f[2] in out.get(x, ())) for f in F_], dtype=float)
top = np.array([np.bincount(TT[:, i]).argmax() for i in range(len(F_))]); agree = (TT == top[None]).mean(0)
other = top != B_.numpy(); s4 = np.where(other, PT.mean(0) * agree, 0.0) - np.exp(lp)
lc = np.nanmean(LC, 0) if np.isfinite(LC).any() else np.full(len(F_), np.nan)
SIG = {'S1 surprise (-log p, leave-fold-out)': -lp, 'S2 surprise normalised (- (log p - log prior))': -(lp - prior),
       'S3a path, latent composite (-log p)': -lc, 'S3b path, symbolic (-2-hop paths)': -paths, 'S4 confident contradiction (seeds agree on another tail)': s4,
       'S1+S2+S4 (rank sum)': None, 'BASE documents (fewer = suspect)': -nd, 'BASE tail prior (rarer = suspect)': -prior, 'BASE head degree (lower = suspect)': -deg}
rk = lambda v: np.argsort(np.argsort(v))
SIG['S1+S2+S4 (rank sum)'] = rk(-lp) + rk(-(lp - prior)) + rk(s4)
cov = deg > 1
def pat(v, m):
    v = v[m]; yy = y[m]; n = max(1, int(round(yy.mean() * len(yy)))); return yy[np.argsort(-v)[:n]].mean()
P(f"RETRACTION ON PLANTED ERRORS ({len(F_)} facts, {NOISE:.0%} corrupted, frequency-matched tails; {FO}-fold x {SE} seeds, {STEPS} steps): AUC corrupted vs clean | "
  f"heads with other facts ({int(cov.sum())}) | precision at the noise rate")
for k_, v in SIG.items():
    m = np.isfinite(v); mc = m & cov
    P(f"   {k_}: AUC {roc_auc_score(y[m], v[m]) if 0 < y[m].sum() < m.sum() else float('nan'):.3f} (n {int(m.sum())}) | covered {roc_auc_score(y[mc], v[mc]) if 0 < y[mc].sum() < mc.sum() else float('nan'):.3f} | P@noise {pat(v, m):.3f}")
cl = (y == 0); g = np.array([f[5] for f in F_])
P(f"REAL-DATA PROXY (clean facts only; gold-labelled {g[cl].mean():.3f} vs unlabelled - unlabelled more suspect):")
for k_, v in SIG.items():
    m = np.isfinite(v) & cl
    if 0 < g[m].sum() < m.sum(): P(f"   {k_}: AUC {roc_auc_score(~g[m], v[m]):.3f}")
P(f"DONE ({time.time() - T0:.0f}s)")
