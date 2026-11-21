"""FACT FADING AS THE POOL GROWS (20 Nov 2026; tools/fact_fade.py) - offline, the A6 stand-in trunk, no effect on the package.
kra17 on the GPU: facts taught for 4+ visits fell 0.83 -> 0.60 over visits 8-10 while the question pool grew to ~1,290 facts on a fixed
budget of question rows (4 per step, weight 0.25 x 0.5^(visit-1), floored at 0.125). kra18 doubles the rows (8). Here: real facts from the
distant set's records (head, commonest relational wording, tail), FF_NEW new ones each visit (the pool grows to FF_V x FF_NEW), FF_STEPS
training steps a visit, each with 24 windows of one domain's real text (next-byte loss; the visit's reading) + the question rows, taught in 24
frames and asked in a held-out one ("It is on record that h w "), the answer ranked among itself and 9 other facts' tails (chance 0.10).
Arms (CF_ARMS): FLOOR = kra17 (4 rows, decay floored at 0.125) | DOUBLE = kra18 (8 rows) | TARGET = kra17's budget with half the rows drawn
from the facts the trunk got WRONG at the visit start (rehearse the fading first) | FLOOR25 = floor 0.25 | NODECAY = 4 rows at full weight.
Reports per visit: recall of every cohort (by the visit it entered), all facts so far, the oldest half, control facts (never taught) and
forgetting (bits per byte on held-out real text vs the base).
Usage: TG_CKPT=<pt_A6.pt> python3 tools/fact_fade.py <distant_50k.json.gz>   env FF_V (8), FF_NEW (150), FF_STEPS (250), FF_ARMS, FF_EVAL (60 a cohort)"""
import os, sys, types, copy, time, math, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2); torch.manual_seed(0)
NV = int(os.environ.get('FF_V', 8)); NN = int(os.environ.get('FF_NEW', 150)); ST = int(os.environ.get('FF_STEPS', 250)); NE = int(os.environ.get('FF_EVAL', 60))
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, seed=4)
pm._load_distant(cfg); R = pm._PM['R']
CNT = collections.defaultdict(collections.Counter)
for r in R:
    if 1 <= r[5] <= 8 and any(ch.isalpha() for ch in r[3]) and not pm._rret_listlike(r): CNT[(r[1], r[2])][pm._rnorm(r[3])] += 1
del R, pm._PM['R']
FA, seen = [], set()
for (h, t), c in sorted(CNT.items(), key=lambda kv: -sum(kv[1].values())):
    w = c.most_common(1)[0][0]
    if h in seen or not w or len(f"{h} {w} {t}") > 70 or h == t: continue
    seen.add(h); FA.append((h, w, t))
FA = [FA[j] for j in np.random.RandomState(7).permutation(len(FA))][:NV * NN + NN]
CO = [FA[k * NN:(k + 1) * NN] for k in range(NV)]; CT = FA[NV * NN:]
P(f"{len(CNT)} related pairs; {len(FA)} facts (one per head): {NV} cohorts of {NN} + {len(CT)} control  ({time.time() - T0:.0f}s)")
DW = 192; L_ = 128; PAD = 256; V_ = 257
class Blk(nn.Module):
    def __init__(s, d, hh):
        super().__init__(); s.a = nn.MultiheadAttention(d, hh, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        q = s.n1(x); x = x + s.a(q, q, q, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class LM(nn.Module):
    def __init__(s, d=DW, nl=3):
        super().__init__(); s.emb = nn.Embedding(V_, d); s.pos = nn.Parameter(torch.randn(L_, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl)); s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V_)
    def forward(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for b in s.b: x = b(x, am)
        return s.o(s.n(x))
base = LM(); base.load_state_dict(torch.load(os.environ['TG_CKPT'])); base.eval()
MP = os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile')
DOMS = {d: np.frombuffer(open(f'{MP}/{d}.bin', 'rb').read().replace(b'\n', b' '), np.uint8) for d in ('wiki', 'prose', 'arxiv', 'pubmed', 'law')}
enc = lambda s: s.encode('latin-1', 'replace')[:L_]
def tens(bl):
    bl = [b[:L_] for b in bl]; T = max(len(b) for b in bl); X = torch.full((len(bl), T), PAD, dtype=torch.long)
    for i, b in enumerate(bl): X[i, :len(b)] = torch.tensor(list(b))
    return X, torch.cat([torch.full((len(bl), 1), PAD), X[:, :-1]], 1)
FR = ("{h} {w} {t}.", "In short, {h} {w} {t}.", "Notes: {h} {w} {t}.", "As recorded, {h} {w} {t}.", "From the notes: {h} {w} {t}.", "{h}, as noted, {w} {t}.",
      "It is known that {h} {w} {t}.", "We read that {h} {w} {t}.", "Sources say {h} {w} {t}.", "{h} - {w} {t}.", "Reportedly, {h} {w} {t}.", "Note that {h} {w} {t}.",
      "According to records, {h} {w} {t}.", "Fact: {h} {w} {t}.", "One learns that {h} {w} {t}.", "Indeed, {h} {w} {t}.", "Background: {h} {w} {t}.",
      "As written elsewhere, {h} {w} {t}.", "Remember: {h} {w} {t}.", "Summary - {h} {w} {t}.", "Earlier we saw that {h} {w} {t}.", "Briefly, {h} {w} {t}.")
EVD = {d: [bytes(a[len(a) - 200000 + k * 129:len(a) - 200000 + (k + 1) * 129]) for k in range(64)] for d, a in DOMS.items()}
@torch.no_grad()
def dbits(m):
    out = []
    for e in EVD.values():
        X, Xi = tens(e); out.append(float(F.cross_entropy(m(Xi).reshape(-1, V_), X.reshape(-1), ignore_index=PAD)) / math.log(2))
    return float(np.mean(out))
rr = np.random.RandomState(3); TL = [f[2] for f in FA]
CAND = {f: [f[2]] + [x for x in (TL[j] for j in rr.choice(len(TL), 14, replace=False)) if x != f[2]][:9] for f in FA}
@torch.no_grad()
def rank(m, f):
    pb = enc(f"It is on record that {f[0]} {f[1]} "); bl = [(pb + x.encode('latin-1', 'replace') + b'.')[:L_] for x in CAND[f]]
    X, Xi = tens(bl); lp = F.log_softmax(m(Xi), -1); n0 = len(pb)
    sc = [float(lp[i, n0:len(b)].gather(-1, X[i, n0:len(b), None]).mean()) if len(b) > n0 else -1e9 for i, b in enumerate(bl)]
    return int(np.argmax(sc) == 0)
EV = [c[:NE] for c in CO]; ECT = CT[:NE]
def report(m, v, arm, d0):
    m.eval(); r = [np.mean([rank(m, f) for f in EV[k]]) for k in range(v + 1)]; ct = np.mean([rank(m, f) for f in ECT]); m.train()
    h = max(1, (v + 1) // 2)
    P(f"   {arm:8s} visit {v + 1}: cohorts " + " ".join(f"{x:.2f}" for x in r) + f" | ALL {np.mean(r):.3f} | OLDEST HALF {np.mean(r[:h]):.3f} | NEWEST {r[-1]:.2f} | "
      f"control {ct:.2f} | forgetting {dbits(m) - d0:+.4f}  ({time.time() - T0:.0f}s)")
    return r
d0 = dbits(base)
P(f"BASE: cohort 1 {np.mean([rank(base, f) for f in EV[0]]):.2f} | control {np.mean([rank(base, f) for f in ECT]):.2f} | bits {d0:.4f}")
mn = min(len(a) for a in DOMS.values()) - 200400; TR = {f: [enc(x.format(h=f[0], w=f[1], t=f[2])) for x in FR] for f in FA}
ARMS = {'FLOOR': (4, 0.125, 0), 'DOUBLE': (8, 0.125, 0), 'TARGET': (4, 0.125, 1), 'FLOOR25': (4, 0.25, 0), 'NODECAY': (4, 1.0, 0)}
for arm in os.environ.get('FF_ARMS', 'FLOOR,DOUBLE,TARGET,NODECAY').split(','):
    nr, fl, tg = ARMS[arm]; m = copy.deepcopy(base); m.train(); rs = np.random.RandomState(11); op = torch.optim.Adam(m.parameters(), lr=1e-3); ds_ = list(DOMS)
    for v in range(NV):
        pool = [f for k in range(v + 1) for f in CO[k]]; wq = 0.25 * max(0.5 ** v, fl)          # the package: inf_distill_wake 0.25 x 0.5^(visit - 1), floored
        wrong = None
        if tg and v:                                                               # TARGET: the trunk's own check at the visit start (the whole pool)
            m.eval(); wrong = [f for f in pool if not rank(m, f)]; m.train()
        for st in range(ST):
            wb = [bytes(DOMS[ds_[v % 5]][i:i + 129]) for i in rs.randint(0, mn, 24)]   # this visit's domain (reading competes with the facts)
            if wrong: src = [wrong[i] for i in rs.randint(0, len(wrong), nr // 2)] + [pool[i] for i in rs.randint(0, len(pool), nr - nr // 2)]
            else: src = [pool[i] for i in rs.randint(0, len(pool), nr)]
            fb = [TR[f][rs.randint(len(TR[f]))] for f in src]
            Xf, Xfi = tens(fb); lf = F.cross_entropy(m(Xfi).reshape(-1, V_), Xf.reshape(-1), ignore_index=PAD)
            Xw, Xwi = tens(wb); lw = F.cross_entropy(m(Xwi).reshape(-1, V_), Xw.reshape(-1), ignore_index=PAD)
            loss = lw + wq * lf                                                  # the package: next-byte loss + weight x the MEAN question-row loss (more rows = more facts a step, same push)
            op.zero_grad(); loss.backward(); op.step()
        if wrong is not None: P(f"   {arm:8s} visit {v + 1}: rehearsed first {len(wrong)} of {len(pool)} the trunk got wrong at the visit start")
        report(m, v, arm, d0)
P(f"DONE ({time.time() - T0:.0f}s)")
