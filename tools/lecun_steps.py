"""22 Nov THREE STEPS TOWARD LECUN'S DESIGN (offline, synthetic; the five-level world of tools/latent_search.py, same seed; no trunk:
route J's name codes are learned jointly with its operators, which route J's random-code control showed is enough).
A. NON-CONTRASTIVE TRAINING of route J: InfoNCE over the names (current) vs regression of the operator's prediction onto the tail's code
   with (a) VICReg variance + covariance terms, (b) SIGReg (LeJEPA, Epps-Pulley), (c) SVReg (variance floor on random projections),
   (d) nothing (the collapse control). Each judged by search over operator programs: 3 hops ('citizen of') and 4 hops ('continent'),
   held-out people in cities in no taught example, the margin gate and never-read names.
B. ENSEMBLE UNCERTAINTY (after VJEPA: a distribution, not a point): 5 seeds. Known questions vs never-read names (does disagreement
   flag the unknown better than one model's margin?) and TRUE vs FALSE claims about known people ('X is a citizen of Y'): is a false
   claim CONTRADICTED (seeds agree on another answer) while an unknown name is merely UNCERTAIN (seeds disagree)?
C. HIERARCHY BY CACHED PROGRAMS: the program search finds for 'citizen of' is kept as a new operator; 'continent' is then searched with
   programs of at most 2 steps over the relations + the cached one (without the cache a 4-hop relation is out of reach at that budget);
   and a distilled macro (one MLP fitted on the cached program's endpoints) against the exact cached composition.
Usage: python3 tools/lecun_steps.py  (env LC_STEPS, LC_SEEDS)"""
import os, time, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
torch.set_num_threads(2); SEED = int(os.environ.get('LS_SEED', 0)); torch.manual_seed(SEED); rs = np.random.RandomState(SEED); T0 = time.time()
P = lambda s: print(s, flush=True)
SY = ['ka', 'lo', 'mi', 'ra', 'zo', 've', 'tu', 'ni', 'sa', 'po', 'el', 'du', 'fa', 'gi', 'ho', 'ju', 'ke', 'lu', 'me', 'no', 'ri', 'su', 'ta', 'vi']
used = set()
def name(k):
    while True:
        n = ''.join(rs.choice(SY) for _ in range(k)).capitalize()
        if n not in used: used.add(n); return n
NCT, NK, NR, NCI, NTR, NHE = 5, 20, 60, 300, 400, 100
CONT = [name(3) for _ in range(NCT)]; CTY = [(name(3), rs.randint(NCT)) for _ in range(NK)]; REG = [(name(3), rs.randint(NK)) for _ in range(NR)]
CIT = [(name(3), rs.randint(NR)) for _ in range(NCI)]
def person():
    while True:
        n = name(2) + ' ' + name(3)
        if n not in used: used.add(n); return n
TRP = [(person(), rs.randint(NCI)) for _ in range(NTR)]; HEP = [(person(), rs.randint(NCI)) for _ in range(NHE)]
NOV = [person() for _ in range(60)]                                            # never-read names
reg = lambda ci: CIT[ci][1]; cty = lambda ci: REG[reg(ci)][1]; cnt = lambda ci: CTY[cty(ci)][1]
W = {'lives': ["{a} lives in {b}.", "{a} has a home in {b}."], 'city_in': ["{a} lies in {b}.", "The city of {a} is in {b}."],
     'region_of': ["{a} is part of {b}.", "The region {a} belongs to {b}."], 'country_in': ["{a} is in {b}.", "The country {a} lies in {b}."]}
FACTS = {'lives': [(a, CIT[ci][0]) for a, ci in TRP + HEP], 'city_in': [(c, REG[r][0]) for c, r in CIT], 'region_of': [(r, CTY[k][0]) for r, k in REG],
         'country_in': [(k, CONT[c]) for k, c in CTY]}
NAMES = sorted({x for fs in FACTS.values() for p_ in fs for x in p_} | set(NOV)); NI = {n: i for i, n in enumerate(NAMES)}
REL = list(FACTS); RI = {r: j for j, r in enumerate(REL)}; D = 64; STEPS = int(os.environ.get('LC_STEPS', 3000)); NSE = int(os.environ.get('LC_SEEDS', 5))
TR = [(a, r, b) for r, fs in FACTS.items() for a, b in fs]
A_ = torch.tensor([NI[a] for a, r, b in TR]); R_ = torch.tensor([RI[r] for a, r, b in TR]); B_ = torch.tensor([NI[b] for a, r, b in TR])
def sigreg(z, ns=128, nk=17, tm=5.0):
    U = F.normalize(torch.randn(z.shape[1], ns), dim=0); x = z @ U; t = torch.linspace(0.0, tm, (nk + 1) // 2); tx = x[:, :, None] * t[None, None, :]
    re = torch.cos(tx).mean(0); im = torch.sin(tx).mean(0); cf = torch.exp(-t ** 2 / 2)
    return z.shape[0] * 2.0 * torch.trapezoid(((re - cf[None, :]) ** 2 + im ** 2) * cf[None, :], t, dim=-1).mean()
def svreg(z, ns=128):
    U = F.normalize(torch.randn(z.shape[1], ns), dim=0); return F.relu(1 - (z @ U).std(0)).pow(2).mean()
def vicreg(z):
    zc = z - z.mean(0); std = torch.sqrt(zc.var(0) + 1e-4); c = (zc.T @ zc) / (len(z) - 1); off = c - torch.diag(torch.diag(c))
    return F.relu(1 - std).mean() + off.pow(2).sum() / z.shape[1]
def train(obj, seed):
    torch.manual_seed(seed); E = nn.Parameter(torch.randn(len(NAMES), D))
    OPS = nn.ModuleList(nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D)) for _ in REL)
    opt = torch.optim.Adam([E] + list(OPS.parameters()), lr=1e-3)
    for st in range(STEPS):
        k = torch.randint(0, len(TR), (256,)); za = E[A_[k]]; out = torch.zeros_like(za)
        for j in R_[k].unique().tolist():
            m = R_[k] == j; out[m] = OPS[j](za[m])
        if obj == 'nce': l = F.cross_entropy(F.normalize(out, dim=-1) @ F.normalize(E, dim=-1).T / 0.05, B_[k])
        else:
            l = F.mse_loss(out, E[B_[k]]); zb = E[torch.randint(0, len(NAMES), (256,))]
            if obj == 'vicreg': l = l + vicreg(zb)
            elif obj == 'sigreg': l = l + 0.05 * sigreg(zb)
            elif obj == 'svreg': l = l + svreg(zb)
        opt.zero_grad(); l.backward(); opt.step()
    return E.detach(), OPS.eval()
def runner(E, OPS, obj):
    Z = F.normalize(E, dim=-1)
    @torch.no_grad()
    def run(prog, heads, return_probs=False):
        v = E[torch.tensor([NI[h] for h in heads])]; mg = torch.full((len(heads),), 9.0)
        for r in prog:
            p = OPS[RI[r]](v) if r in RI else CACHE[r](v)
            s = F.normalize(p, dim=-1) @ Z.T; t2 = s.topk(2, -1); mg = torch.minimum(mg, t2.values[:, 0] - t2.values[:, 1]); v = E[t2.indices[:, 0]]
        out = [NAMES[i] for i in t2.indices[:, 0].tolist()]
        return (out, mg, F.softmax(s / 0.05, -1)) if return_probs else (out, mg)
    return run
CACHE = {}
NT = 100; TPOOL = [p_ for p_ in TRP if p_[1] % 2 == 0][:NT]; TC = {ci for a, ci in TPOOL}; HN = [p_ for p_ in HEP if p_[1] not in TC]
def search(run, ex, rels, maxlen):
    hs = [h for h, a in ex]; ans = [a for h, a in ex]; sh = list(np.random.RandomState(3).permutation(ans)); best, bn = None, 0.0; n = 0
    for p_ in (p for k in range(1, maxlen + 1) for p in itertools.product(rels, repeat=k)):
        out, mg = run(p_, hs); n += 1; acc = np.mean([o == a for o, a in zip(out, ans)]); bn = max(bn, np.mean([o == a for o, a in zip(out, sh)]))
        if best is None or (acc, float(mg.mean())) > best[0]: best = ((acc, float(mg.mean())), p_)
    return best[1], best[0][0], bn, n
def evaluate(run, tag):
    pc, acc, nul, n = search(run, [(a, CTY[cty(ci)][0]) for a, ci in TPOOL], REL, 4)
    out, mg = run(pc, [a for a, ci in HN]); rt = np.array([o == CTY[cty(ci)][0] for o, (a, ci) in zip(out, HN)])
    to, tm = run(pc, [a for a, ci in TPOOL]); tr = np.array([o == CTY[cty(ci)][0] for o, (a, ci) in zip(to, TPOOL)])
    th = float(np.quantile(tm[torch.tensor(tr)].numpy(), 0.25)) if tr.any() else 0.0; g = (mg >= th).numpy(); _, nm = run(pc, NOV)
    pk, acck, nulk, _ = search(run, [(a, CONT[cnt(ci)]) for a, ci in TPOOL[:20]], REL, 4)
    o4, _ = run(pk, [a for a, ci in HN])
    P(f"   {tag}: 3 hops found {' o '.join(pc)} (right {acc:.2f}, null {nul:.2f}) | HELD-OUT {rt.mean():.2f} | gated answered {g.mean():.2f}, right {rt[g].mean() if g.any() else float('nan'):.2f} | "
      f"never-read answered {float((nm >= th).float().mean()):.2f} || 4 hops found {' o '.join(pk)} ({acck:.2f}, null {nulk:.2f}), held-out {np.mean([o == CONT[cnt(ci)] for o, (a, ci) in zip(o4, HN)]):.2f}  ({time.time() - T0:.0f}s)")
    return pc
P(f"A. ROUTE J OBJECTIVES ({len(NAMES)} names, {len(TR)} premises, codes learned jointly, {STEPS} steps)")
for obj in ('nce', 'vicreg', 'sigreg', 'svreg', 'none'):
    E, OPS = train(obj, 0); run = runner(E, OPS, obj)
    with torch.no_grad():
        z = E; zc = z - z.mean(0); ev = torch.linalg.eigvalsh(torch.cov(zc.T)).clamp(min=0); er = float(ev.sum() ** 2 / (ev ** 2).sum())
    evaluate(run, f"{obj:7s} (code effective rank {er:.1f} of {D})")
P(f"B. ENSEMBLE UNCERTAINTY ({NSE} seeds, InfoNCE)")
M = [train('nce', s) for s in range(NSE)]; RUNS = [runner(E, O, 'nce') for E, O in M]
pc = search(RUNS[0], [(a, CTY[cty(ci)][0]) for a, ci in TPOOL], REL, 3)[0]
def ens(heads):
    outs, mgs, prs = zip(*[r(pc, heads, True) for r in RUNS])
    top = [max(set(col), key=col.count) for col in zip(*outs)]; agree = np.array([np.mean([o[i] == top[i] for o in outs]) for i in range(len(heads))])
    return top, agree, np.mean([m.numpy() for m in mgs], 0), mgs[0].numpy(), prs
kh = [a for a, ci in HN]; ktop, kag, kmg, kmg0, kpr = ens(kh); utop, uag, umg, umg0, upr = ens(NOV)
y = np.r_[np.zeros(len(kh)), np.ones(len(NOV))]
P(f"   known (held-out people, 3 hops) right {np.mean([t == CTY[cty(ci)][0] for t, (a, ci) in zip(ktop, HN)]):.2f}, seeds agree {kag.mean():.2f} | never-read names: seeds agree {uag.mean():.2f}")
P(f"   UNKNOWN vs KNOWN, AUC: ensemble disagreement {roc_auc_score(y, -np.r_[kag, uag]):.3f} | one model's margin {roc_auc_score(y, -np.r_[kmg0, umg0]):.3f} | ensemble mean margin {roc_auc_score(y, -np.r_[kmg, umg]):.3f}")
rs2 = np.random.RandomState(9); truec = [CTY[cty(ci)][0] for a, ci in HN]; falsec = [CTY[(cty(ci) + 1 + rs2.randint(NK - 1)) % NK][0] for a, ci in HN]
def claim_p(prs, claims): return np.mean([[float(p[i, NI[c]]) for i, c in enumerate(claims)] for p in prs], 0)
pt, pf = claim_p(kpr, truec), claim_p(kpr, falsec)
contra_f = (kag >= 0.8) & (np.array(ktop) != np.array(falsec)); contra_t = (kag >= 0.8) & (np.array(ktop) != np.array(truec))
P(f"   CLAIMS about known people: TRUE vs FALSE AUC by the ensemble's probability of the claim {roc_auc_score(np.r_[np.ones(len(pt)), np.zeros(len(pf))], np.r_[pt, pf]):.3f} | "
  f"CONTRADICTED (seeds agree >= 0.8 on another answer): false claims {contra_f.mean():.2f}, true claims {contra_t.mean():.2f} | never-read names called contradicted {np.mean(uag >= 0.8):.2f} (should be low: unknown, not contradicted)")
P("C. HIERARCHY BY CACHED PROGRAMS (InfoNCE, seed 0)")
E, OPS = M[0]; run = runner(E, OPS, 'nce'); pc0 = search(run, [(a, CTY[cty(ci)][0]) for a, ci in TPOOL], REL, 4)[0]
ex4 = [(a, CONT[cnt(ci)]) for a, ci in TPOOL[:20]]
p2, a2, n2, k2 = search(run, ex4, REL, 2); P(f"   continent, budget 2 steps, relations only: best {' o '.join(p2)} right {a2:.2f} (null {n2:.2f}; {k2} programs)")
CACHE['citizen*'] = lambda v, _p=pc0: _cached(v, _p)
@torch.no_grad()
def _cached(v, prog):
    Z = F.normalize(E, dim=-1)
    for r in prog: s = F.normalize(OPS[RI[r]](v), dim=-1) @ Z.T; v = E[s.argmax(-1)]
    return v
p3, a3, n3, k3 = search(run, ex4, REL + ['citizen*'], 2); o3, _ = run(p3, [a for a, ci in HN])
P(f"   continent, budget 2 steps, relations + the cached 'citizen of' ({' o '.join(pc0)}): best {' o '.join(p3)} right {a3:.2f} (null {n3:.2f}; {k3} programs) | held-out {np.mean([o == CONT[cnt(ci)] for o, (a, ci) in zip(o3, HN)]):.2f}")
torch.manual_seed(1); mac = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, D)); om = torch.optim.Adam(mac.parameters(), lr=1e-3)
ha = torch.tensor([NI[a] for a, ci in TPOOL]); hb = torch.tensor([NI[CTY[cty(ci)][0]] for a, ci in TPOOL]); Zn = F.normalize(E, dim=-1)
for st in range(1500):
    l = F.cross_entropy(F.normalize(mac(E[ha]), dim=-1) @ Zn.T / 0.05, hb); om.zero_grad(); l.backward(); om.step()
with torch.no_grad():
    hh = torch.tensor([NI[a] for a, ci in HN]); mo = (F.normalize(mac(E[hh]), dim=-1) @ Zn.T).argmax(-1).tolist()
P(f"   distilled macro (one MLP on the cached program's {len(TPOOL)} endpoints): held-out {np.mean([NAMES[i] == CTY[cty(ci)][0] for i, (a, ci) in zip(mo, HN)]):.2f} vs the exact cached composition (above) - a fitted shortcut is a table, the cached program generalises")
P(f"DONE ({time.time() - T0:.0f}s)")
