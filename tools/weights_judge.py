"""DO THE EXISTING WEIGHTS SAY AN INFERENCE IS FALSE? (15 Nov 2026; tools/weights_judge.py) - offline, stand-in trunk, label-free signals.
The trunk's LIKELIHOOD of a written inference does not separate right from wrong (GPU: its loss at first AUC 0.48-0.50). Two other readings
of the weights, scored on the GPU path's 200 written inferences (tools/select_judge.py, code + AA, 0.74 right; distant labels score only):
  1 TRUTH DIRECTION   (Marks & Tegmark 2023, 'The Geometry of Truth'; Burns et al. 2022, contrast-consistent search): statements the
                      system has READ ('A <words> B' spans of documents 0 -> 2/10) against the same spans with B swapped for another name of
                      B's entity type - a logistic probe on the trunk's pooled states (middle + last block), trained on spans naming no
                      written name. An inference A-C is asked in the CONTEXTS the system read about A: each span 'A <words> X' with X
                      replaced by C (and each 'Y <words> C' with Y replaced by A); signal = the probe's mean / max probability of TRUE
  2 INTERFERENCE      learning the inference (a few SGD steps on 'A is connected to C.' and the substituted statements, on a copy) and the
                      change of the trunk's loss on what it read about A and C (their spans): consistent knowledge should not suffer;
                      signal = minus the loss rise; also the cosine between the inference's gradient and the known spans' gradient
AUC right vs wrong for each (the current judge 0.64; likelihood alone ~0.50), and the trunk's own likelihood of the statements as the baseline.
Usage: TG_D=192 TG_CKPT=<model> python3 tools/weights_judge.py <train_distant.json.gz> <writes .json>   env WJ_N (200), WJ_LR (3e-4), WJ_STEPS (3)"""
import os, sys, json, time, types, copy, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '.')
from platypus.inference.compress import _cg_auc
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(int(os.environ.get('WJ_THREADS', 2)))
DW = int(os.environ.get('TG_D', 192)); CK = os.environ['TG_CKPT']; L = 128; PAD = 256; V = 257
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        h = s.n1(x); x = x + s.a(h, h, h, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class LM(nn.Module):
    def __init__(s, d=DW, nl=3):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(L, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl))
        s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V)
    def states(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]; hs = []
        for b in s.b: x = b(x, am); hs.append(x)
        return hs
    def forward(s, X): return s.o(s.n(s.states(X)[-1]))
lm = LM(); lm.load_state_dict(torch.load(CK)); lm.eval()
def tens(texts):
    ch = [t.encode('latin-1', 'replace')[:L] for t in texts]; T = max(len(b) for b in ch); X = torch.full((len(ch), T), PAD, dtype=torch.long)
    for k, b in enumerate(ch): X[k, :len(b)] = torch.tensor(list(b))
    return X, torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1)
@torch.no_grad()
def pooled(texts):
    out = []
    for i0 in range(0, len(texts), 128):
        X, Xi = tens(texts[i0:i0 + 128]); hs = lm.states(Xi); m = (X != PAD).float()[:, :, None]
        out.append(torch.cat([(h * m).sum(1) / m.sum(1) for h in (hs[1], hs[2])], -1).numpy())
    return np.concatenate(out)
def loss(m, texts, red='mean'):
    X, Xi = tens(texts); lo = F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1), reduction='none').view(X.shape); mk = (X != PAD).float()
    return (lo * mk).sum() / mk.sum() if red == 'mean' else (lo * mk).sum(1) / mk.sum(1)
import platypus.inference.premises as pm
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10)); R = pm._PM['R']; ND = pm._PM['ND']; TYP = pm._PM['TY']
n1 = int(round(ND * 0.2)); g = np.random.RandomState(0)
WR = [(tuple(k), bool(t)) for k, t in json.load(open(sys.argv[2]))][:int(os.environ.get('WJ_N', 200))]; WN = {n for k, _ in WR for n in k}
spans = [r for r in R if r[0] < n1 and r[1] != r[2] and r[7].startswith(r[1]) and r[7].endswith(r[2]) and len(r[7]) <= 60]
byT = collections.defaultdict(list)
for r in spans: byT[TYP.get(r[2], 'X')].append(r[2])
about = collections.defaultdict(list)
for r in spans:
    if r[1] in WN: about[r[1]].append(r)
    if r[2] in WN: about[r[2]].append(r)
# 1 TRUTH DIRECTION: probe on read spans vs the same spans with the second name swapped (spans naming no written name)
tr = [r for r in spans if r[1] not in WN and r[2] not in WN]; tr = [tr[i] for i in g.choice(len(tr), min(6000, len(tr)), replace=False)]
fake = lambda r: r[7][:len(r[7]) - len(r[2])] + (lambda c: c[g.randint(len(c))])([x for x in byT[TYP.get(r[2], 'X')][:5000] if x != r[2]] or [r[2]])
Xt = pooled([r[7] + '.' for r in tr]); Xf = pooled([fake(r) + '.' for r in tr])
Xa = np.concatenate([Xt, Xf]); ya = np.r_[np.ones(len(Xt)), np.zeros(len(Xf))]; mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
idx = g.permutation(len(Xa)); cut = int(0.8 * len(idx)); clf = LogisticRegression(max_iter=2000, C=0.1).fit(((Xa - mu) / sd)[idx[:cut]], ya[idx[:cut]])
acc = clf.score(((Xa - mu) / sd)[idx[cut:]], ya[idx[cut:]])
P(f"TRUTH PROBE: {len(tr)} read spans vs the same with the second name swapped (same entity type); held-out accuracy {acc:.3f} (chance 0.50)  ({time.time() - T0:.0f}s)")
def stmts(a, c, cap=6):
    out = []
    for r in about.get(a, [])[:cap]:
        if r[1] == a: out.append(r[7][:len(r[7]) - len(r[2])] + c)
    for r in about.get(c, [])[:cap]:
        if r[2] == c and r[7].startswith(r[1]): out.append(a + r[7][len(r[1]):])
    return out
y = np.array([t for _, t in WR]); pm_, px_, lk_, hasctx = [], [], [], []
for (a, c), t in WR:
    S_ = stmts(a, c) or [f"{a} is connected to {c}"]; hasctx.append(len(stmts(a, c)) > 0)
    p = clf.predict_proba((pooled([s + '.' for s in S_]) - mu) / sd)[:, 1]; pm_.append(p.mean()); px_.append(p.max())
    with torch.no_grad(): lk_.append(-float(loss(lm, [s + '.' for s in S_])))
pm_, px_, lk_, hasctx = map(np.array, (pm_, px_, lk_, hasctx))
au = lambda s, m=None: (lambda s_, y_: _cg_auc(s_[y_], s_[~y_]) if y_.any() and (~y_).any() else float('nan'))(s if m is None else s[m], y if m is None else y[m])
P(f"   on the {len(WR)} written inferences (right {y.mean():.3f}; {hasctx.mean():.2f} with read contexts): TRUTH PROBE mean AUC {au(pm_):.3f}, max {au(px_):.3f} | "
  f"the trunk's LIKELIHOOD of the same statements {au(lk_):.3f} || with contexts only (n {hasctx.sum()}): probe {au(pm_, hasctx):.3f}, likelihood {au(lk_, hasctx):.3f}  ({time.time() - T0:.0f}s)")
# 2 INTERFERENCE: a few SGD steps on the inference's statements; the loss change on what was read about A and C
LR, NS = float(os.environ.get('WJ_LR', 3e-4)), int(os.environ.get('WJ_STEPS', 3)); base = copy.deepcopy(lm.state_dict()); dl, cs = [], []
params = [p for n, p in lm.named_parameters() if '.m.' in n]                     # the MLPs (where facts are thought to live)
for (a, c), t in WR:
    known = [r[7] + '.' for r in (about.get(a, [])[:8] + about.get(c, [])[:8])]
    S_ = [s + '.' for s in (stmts(a, c) or [f"{a} is connected to {c}"])] + [f"{a} is connected to {c}."]
    if not known: dl.append(np.nan); cs.append(np.nan); continue
    lm.train(); g1 = torch.autograd.grad(loss(lm, S_), params); g2 = torch.autograd.grad(loss(lm, known), params)
    cs.append(float(sum((x * z).sum() for x, z in zip(g1, g2)) / (torch.sqrt(sum((x * x).sum() for x in g1)) * torch.sqrt(sum((z * z).sum() for z in g2)) + 1e-12)))
    with torch.no_grad(): l0 = float(loss(lm, known))
    opt = torch.optim.SGD(params, lr=LR * 100)
    for _ in range(NS): opt.zero_grad(); loss(lm, S_).backward(); opt.step()
    with torch.no_grad(): dl.append(float(loss(lm, known)) - l0)
    lm.load_state_dict(base); lm.eval()
dl, cs = np.array(dl), np.array(cs); m = ~np.isnan(dl)
P(f"INTERFERENCE ({NS} SGD steps on the inference, lr {LR * 100:g}, MLPs only; {m.sum()} inferences with read spans about A or C): loss rise on what was read "
  f"about A and C - AUC of minus the rise {au(-dl, m):.3f} (right: {np.nanmean(dl[y]):+.4f}, wrong: {np.nanmean(dl[~y]):+.4f} nats) | gradient cosine with the "
  f"known spans {au(cs, m):.3f} (right {np.nanmean(cs[y]):+.3f}, wrong {np.nanmean(cs[~y]):+.3f})  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
