"""22 Nov RETRACTION IN A DENSE WORLD (offline, synthetic) - tools/retract_jepa.py found no signal on real records (about one fact per
name: nothing contradicts a wrong fact). Here the same JEPA signals in a five-level world like tools/latent_search.py's, where a
redundant view exists: some people's COUNTRY is also recorded ('citizen'). Wrong facts are planted in 'lives' (person -> wrong city) and
'city_in' (city -> wrong region); the redundant view is varied (0 / 0.5 / 1.0 of people have a recorded country) to show whether
density is what makes retraction possible.
Route J (name codes + one JEPA-style predictor per relation, InfoNCE over names) trained K-fold; each fact scored by models that never
saw it. Signals (higher = more suspect): S1 surprise (-log p of the recorded tail), S4 confident contradiction (seeds agree on another
tail), S5 PATH CONTRADICTION for a 'lives' fact (person, city): the latent composite city -> region -> country disagrees with the
person's recorded country (and its symbolic twin with the recorded facts); for a 'city_in' fact (city, region): the fraction of its
residents whose recorded country disagrees with region -> country. BLAME: when a person's chain contradicts the recorded country, is the
planted fact the one with the highest surprise?
Usage: python3 tools/retract_dense.py  (env RD_NOISE, RD_FOLDS, RD_SEEDS, RD_STEPS)"""
import os, time, collections
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
PEO = [(person(), rs.randint(NCI)) for _ in range(NTR + NHE)]
NOISE = float(os.environ.get('RD_NOISE', 0.05)); FO = int(os.environ.get('RD_FOLDS', 4)); SE = int(os.environ.get('RD_SEEDS', 2)); STEPS = int(os.environ.get('RD_STEPS', 1500)); D = 64
reg_of = {c: REG[r][0] for c, r in CIT}; cty_of = {r: CTY[k][0] for r, k in REG}
def run(frac, seed0=0):
    r_ = np.random.RandomState(100 + int(frac * 10))
    facts = [('lives', a, CIT[ci][0], 0) for a, ci in PEO] + [('city_in', c, REG[r][0], 0) for c, r in CIT] + [('region_of', r, CTY[k][0], 0) for r, k in REG] + [('country_in', k, CONT[c], 0) for k, c in CTY]
    for i, (rel, a, b, _) in enumerate(facts):                                 # plant errors in 'lives' and 'city_in' (frequency-matched tails)
        if rel in ('lives', 'city_in') and r_.rand() < NOISE:
            pool = [f[2] for f in facts if f[0] == rel and f[2] != b]; facts[i] = (rel, a, pool[r_.randint(len(pool))], 1)
    true_city = {a: CIT[ci][0] for a, ci in PEO}
    rec = [p for p, ci in PEO if r_.rand() < frac]                             # the redundant view: recorded country of some people (true)
    facts += [('citizen', p, cty_of[reg_of[true_city[p]]], 0) for p in rec]
    REL = sorted({f[0] for f in facts}); RI = {r: j for j, r in enumerate(REL)}
    NM = sorted({f[1] for f in facts} | {f[2] for f in facts}); NI = {n: i for i, n in enumerate(NM)}
    A_ = torch.tensor([NI[f[1]] for f in facts]); R_ = torch.tensor([RI[f[0]] for f in facts]); B_ = torch.tensor([NI[f[2]] for f in facts]); y = np.array([f[3] for f in facts])
    def pred(E, OPS, ai, ri):
        z = F.normalize(E[ai], dim=-1); out = torch.zeros_like(z)
        for j in ri.unique().tolist():
            m = ri == j; out[m] = OPS[j](z[m])
        return F.normalize(out, dim=-1)
    fold = np.random.RandomState(2).randint(0, FO, len(facts)); LP = np.zeros((SE, len(facts))); TT = np.zeros((SE, len(facts)), dtype=int); PT = np.zeros((SE, len(facts)))
    CMP = np.full((SE, len(facts)), np.nan)                                    # latent composite: lives fact -> city -> region -> country vs recorded country
    rc = {f[1]: f[2] for f in facts if f[0] == 'citizen'}
    for s in range(SE):
        for f_ in range(FO):
            tr = torch.tensor(np.where(fold != f_)[0]); te = np.where(fold == f_)[0]; torch.manual_seed(seed0 + 100 * s + f_)
            E = nn.Parameter(torch.randn(len(NM), D) * 0.1); OPS = nn.ModuleList(nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, D)) for _ in REL)
            opt = torch.optim.Adam([E] + list(OPS.parameters()), lr=3e-3)
            for st in range(STEPS):
                k = tr[torch.randint(0, len(tr), (256,))]; l = F.cross_entropy(pred(E, OPS, A_[k], R_[k]) @ F.normalize(E, dim=-1).T / 0.05, B_[k]); opt.zero_grad(); l.backward(); opt.step()
            with torch.no_grad():
                EN = F.normalize(E, dim=-1); ix = torch.tensor(te); lg = F.log_softmax(pred(E, OPS, A_[ix], R_[ix]) @ EN.T / 0.05, -1)
                LP[s, te] = lg.gather(-1, B_[ix, None])[:, 0].numpy(); mx = lg.max(-1); TT[s, te] = mx.indices.numpy(); PT[s, te] = mx.values.exp().numpy()
                lv = [i for i in te if facts[i][0] == 'lives' and facts[i][1] in rc]
                if lv:
                    c_ = torch.tensor([NI[facts[i][2]] for i in lv]); v = pred(E, OPS, c_, torch.full_like(c_, RI['city_in'])); r1 = (v @ EN.T).argmax(-1)
                    v = pred(E, OPS, r1, torch.full_like(r1, RI['region_of'])); lg2 = F.log_softmax(v @ EN.T / 0.05, -1)
                    CMP[s, lv] = -lg2.gather(-1, torch.tensor([[NI[rc[facts[i][1]]]] for i in lv]))[:, 0].numpy()
    lp = LP.mean(0); top = np.array([np.bincount(TT[:, i]).argmax() for i in range(len(facts))]); agree = (TT == top[None]).mean(0)
    s4 = np.where(top != B_.numpy(), PT.mean(0) * agree, 0.0) - np.exp(lp)
    # symbolic path contradiction from the RECORDED facts
    rmap = collections.defaultdict(dict)
    for f in facts: rmap[f[0]][f[1]] = f[2]
    res = collections.defaultdict(list)
    for f in facts:
        if f[0] == 'lives': res[f[2]].append(f[1])
    sym = np.full(len(facts), np.nan)
    for i, f in enumerate(facts):
        if f[0] == 'lives' and f[1] in rc:
            k_ = rmap['region_of'].get(rmap['city_in'].get(f[2]))
            sym[i] = float(k_ != rc[f[1]]) if k_ else np.nan
        elif f[0] == 'city_in':
            rs_ = [p for p in res[f[1]] if p in rc]
            if rs_: sym[i] = np.mean([rmap['region_of'].get(f[2]) != rc[p] for p in rs_])
    lat = np.nanmean(CMP, 0) if np.isfinite(CMP).any() else np.full(len(facts), np.nan)
    out = [f"redundant view: recorded country for {frac:.0%} of people ({len(rc)}); facts {len(facts)}, planted errors {int(y.sum())}"]
    for rel in ('lives', 'city_in'):
        m = np.array([f[0] == rel for f in facts])
        for nam, v in (('S1 surprise', -lp), ('S4 confident contradiction', s4), ('S5 path contradiction, latent', lat), ('S5 path contradiction, symbolic', sym),
                       ('S1 + S5 symbolic (rank sum)', np.where(np.isfinite(sym), np.argsort(np.argsort(-lp)) + np.argsort(np.argsort(np.nan_to_num(sym))), np.nan))):
            mm = m & np.isfinite(v)
            if mm.sum() and 0 < y[mm].sum() < mm.sum():
                n_ = max(1, int(y[mm].sum())); pk = y[mm][np.argsort(-v[mm])[:n_]].mean()
                out.append(f"   '{rel}' {nam}: AUC {roc_auc_score(y[mm], v[mm]):.3f} (n {int(mm.sum())}, errors {int(y[mm].sum())}) | precision at the error count {pk:.2f}")
    # BLAME: people whose recorded chain contradicts their recorded country - is the planted fact the most surprising link?
    bl = []
    for p in rc:
        c = rmap['lives'].get(p); r = rmap['city_in'].get(c); k = rmap['region_of'].get(r)
        if k is None or k == rc[p]: continue
        ids = [i for i, f in enumerate(facts) if (f[0], f[1]) in (('lives', p), ('city_in', c), ('region_of', r))]
        bad = [i for i in ids if y[i]]
        if bad: bl.append(int(ids[int(np.argmin(lp[ids]))] in bad))
    if bl: out.append(f"   BLAME: chains contradicting the recorded country with a planted link ({len(bl)}): the most surprising link is the planted one {np.mean(bl):.2f} (chance about 0.33)")
    return out
for frac in (0.0, 0.5, 1.0):
    for line in run(frac): P(line)
    P(f"   ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
