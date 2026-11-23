"""22 Nov LATENT SEARCH vs WRITTEN CHAINS (offline, synthetic; LeCun's direction against the trunk's own chain).

A small byte transformer (the stand-in trunk) is pretrained on PREMISES only, each in two wordings:
  person -lives-> city -city_in-> region -region_of-> country -country_in-> continent.
Then two ways to answer multi-hop questions about people:
  TRUNK (autoregressive, criterion 2): chains taught briefly in the arrow form, 3 hops, 'A is a citizen of B > R > K.', for NT people living in
    even-numbered cities; held-out people in ODD cities (bridges never in a taught chain) are asked 'A is a citizen of ' and parsed.
  LATENT (route J, LeCun's direction - nothing generated): name codes read from the trunk's states, one JEPA-style predictor per relation
    (MLP, InfoNCE over the names, trained on the premises), and the composite relation's PROGRAM found by SEARCH over operator sequences
    (length 1-4) scored on the taught endpoints (A, K) only - inference as choosing the lowest-energy path. Then applied to held-out people,
    each hop snapped to the nearest name; a margin gate (top-1 minus top-2) as the energy threshold; also a 4-hop relation (continent)
    whose program is found the same way from a handful of endpoints, which the trunk was never taught.
Usage: LS_STEPS=5000 LS_NT=100 python3 tools/latent_search.py"""
import os, math, time, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
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
PRE = [t.format(a=a, b=b) for rel, fs in FACTS.items() for a, b in fs for t in W[rel]] * 3
L, V, PAD, DW, NL = 96, 256, 0, 192, 3
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        h = s.n1(x); x = x + s.a(h, h, h, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, DW); s.pos = nn.Parameter(torch.randn(L, DW) * 0.02); s.b = nn.ModuleList(Blk(DW, 4) for _ in range(NL)); s.n = nn.LayerNorm(DW); s.o = nn.Linear(DW, V)
    def hid(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for b in s.b: x = b(x, am)
        return s.n(x)
    def forward(s, X): return s.o(s.hid(X))
def tens(seqs):
    X = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
    return X
def step(m, op, seqs):
    X = tens(seqs); Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1); lo = m(Xi).reshape(-1, V); tg = X.reshape(-1)
    l = F.cross_entropy(lo[tg != PAD], tg[tg != PAD]); op.zero_grad(); l.backward(); op.step(); return float(l)
@torch.no_grad()
def gen(m, p, n=70):
    m.eval(); b = bytearray(p.encode('latin-1'))
    for _ in range(n):
        X = tens([bytes(b) + b' ']); Xi = torch.cat([torch.full((1, 1), PAD), X[:, :-1]], 1); c = int(m(Xi)[0, len(b)].argmax())
        if c in (46, 10, PAD) or len(b) >= L - 1: break
        b.append(c)
    m.train(); return bytes(b[len(p):]).decode('latin-1', 'replace').strip()
m = LM(); op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01); ST = int(os.environ.get('LS_STEPS', 5000))
sch = torch.optim.lr_scheduler.OneCycleLR(op, max_lr=1e-3, total_steps=ST, pct_start=0.05)
for st in range(1, ST + 1):
    step(m, op, [PRE[i].encode('latin-1') for i in rs.randint(0, len(PRE), 32)]); sch.step()
    if st % 1000 == 0:
        ok = np.mean([gen(m, f"{a} lives in ").startswith(CIT[ci][0]) for a, ci in HEP[:40]] + [gen(m, f"{c} lies in ").startswith(REG[r][0]) for c, r in CIT[:40]]
                     + [gen(m, f"{r} is part of ").startswith(CTY[k][0]) for r, k in REG[:40]])
        P(f"   pretraining step {st}: premises recalled (greedy) {ok:.2f}  ({time.time() - T0:.0f}s)")
# LATENT ARM (before any chain teaching: it uses the premises the trunk read)
NAMES = sorted({x for fs in FACTS.values() for p_ in fs for x in p_} | set(NOV)); NI = {n: i for i, n in enumerate(NAMES)}
@torch.no_grad()
def codes(mm):
    m.eval(); Z = []
    for i in range(0, len(NAMES), 64):
        bs = [f"About {n}.".encode('latin-1') for n in NAMES[i:i + 64]]; X = tens(bs); Xi = torch.cat([torch.full((len(bs), 1), PAD), X[:, :-1]], 1); H = mm.hid(Xi)
        for k, n in enumerate(NAMES[i:i + 64]): Z.append(H[k, 7:7 + len(n)].mean(0))
    m.train(); Z = torch.stack(Z); Z = Z - Z.mean(0); return F.normalize(Z, dim=-1)
Z = codes(m); D = Z.shape[1]
def fit(pairs, steps=1500):
    torch.manual_seed(11); net = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D)); o = torch.optim.Adam(net.parameters(), lr=1e-3)
    A_ = torch.tensor([NI[a] for a, b in pairs]); B_ = torch.tensor([NI[b] for a, b in pairs])
    for _ in range(steps):
        k = torch.randint(0, len(A_), (min(256, len(A_)),)); l = F.cross_entropy(F.normalize(net(Z[A_[k]]), dim=-1) @ Z.T / 0.05, B_[k]); o.zero_grad(); l.backward(); o.step()
    return net.eval()
OPS = {r: fit(fs) for r, fs in FACTS.items()}
@torch.no_grad()
def run(prog, heads):
    """apply the operator sequence, snapping each hop to the nearest name; returns (answers, min margin over hops)"""
    v = Z[torch.tensor([NI[h] for h in heads])]; mg = torch.full((len(heads),), 9.0)
    for r in prog:
        s = F.normalize(OPS[r](v), dim=-1) @ Z.T; t2 = s.topk(2, -1); mg = torch.minimum(mg, t2.values[:, 0] - t2.values[:, 1]); v = Z[t2.indices[:, 0]]
    return [NAMES[i] for i in t2.indices[:, 0].tolist()], mg
progs = [p for k in (1, 2, 3, 4) for p in itertools.product(list(FACTS), repeat=k)]
def search(ex, tag):
    """ex: (head, answer) endpoints - pick the operator program with the most right answers (ties: larger mean margin); null: answers shuffled"""
    hs = [h for h, a in ex]; ans = [a for h, a in ex]; sh = list(np.random.RandomState(3).permutation(ans)); best, bestn = None, None
    for p_ in progs:
        out, mg = run(p_, hs); acc = np.mean([o == a for o, a in zip(out, ans)]); accn = np.mean([o == a for o, a in zip(out, sh)])
        key = (acc, float(mg.mean()))
        if best is None or key > best[0]: best = (key, p_)
        if bestn is None or accn > bestn: bestn = accn
    P(f"   SEARCH for '{tag}' over {len(progs)} operator programs (length 1-4) on {len(ex)} endpoints: best {' o '.join(best[1])} (right {best[0][0]:.2f}, mean margin {best[0][1]:.3f}) | null (answers shuffled) best {bestn:.2f}")
    return best[1]
NT = int(os.environ.get('LS_NT', 100)); TPOOL = [p_ for p_ in TRP if p_[1] % 2 == 0][:NT]; TC = {ci for a, ci in TPOOL}
HN = [p_ for p_ in HEP if p_[1] not in TC]                                     # held-out people whose city is in no taught chain
P(f"LATENT ARM (route J style; codes from the pretrained trunk, {len(NAMES)} names; predictors on the premises):")
pc = search([(a, CTY[cty(ci)][0]) for a, ci in TPOOL], 'citizen of (3 hops)')
out, mg = run(pc, [a for a, ci in HN]); rt = [o == CTY[cty(ci)][0] for o, (a, ci) in zip(out, HN)]
tr_out, tr_mg = run(pc, [a for a, ci in TPOOL]); tr_r = [o == CTY[cty(ci)][0] for o, (a, ci) in zip(tr_out, TPOOL)]
th = float(np.quantile(tr_mg[torch.tensor(tr_r)].numpy(), 0.25)) if any(tr_r) else 0.0
no_out, no_mg = run(pc, NOV)
P(f"   citizen of, HELD-OUT people, cities in no taught example ({len(HN)}): right {np.mean(rt):.2f} | MARGIN GATE (t keeps 75% of right taught answers = {th:.3f}): "
  f"answered {float((mg >= th).float().mean()):.2f}, right when answered {np.mean([r_ for r_, g_ in zip(rt, (mg >= th).tolist()) if g_]) if bool((mg >= th).any()) else float('nan'):.2f} | "
  f"NEVER-READ names given an answer {float((no_mg >= th).float().mean()):.2f}")
pk = search([(a, CONT[cnt(ci)]) for a, ci in TPOOL[:20]], 'continent (4 hops, 20 endpoints)')
out4, mg4 = run(pk, [a for a, ci in HN]); P(f"   continent (4 hops; the trunk is never taught it), HELD-OUT people ({len(HN)}): right {np.mean([o == CONT[cnt(ci)] for o, (a, ci) in zip(out4, HN)]):.2f}")
# TRUNK ARM: 3-hop chains taught in the arrow form
op2 = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); rows = [f"{a} is a citizen of {CIT[ci][0]} > {REG[reg(ci)][0]} > {CTY[cty(ci)][0]}." for a, ci in TPOOL]
def trunk_eval(ppl):
    res = {'fmt': [], 'b': [], 'r': [], 'k': []}
    for a, ci in ppl:
        pt = [x.strip() for x in gen(m, f"{a} is a citizen of ").split('>')]
        res['fmt'].append(len(pt) == 3); res['b'].append(len(pt) >= 1 and pt[0] == CIT[ci][0]); res['r'].append(len(pt) >= 2 and pt[1] == REG[reg(ci)][0]); res['k'].append(len(pt) == 3 and pt[2] == CTY[cty(ci)][0])
    return ' '.join(f"{k} {np.mean(v):.2f}" for k, v in res.items())
P(f"TRUNK ARM (3-hop chains taught in the arrow form for {len(TPOOL)} people; never-taught people in cities in no taught chain, {len(HN)}):")
TE = int(os.environ.get('LS_TEACH', 1500))
FRM = int(os.environ.get('LS_FRAMES', 0))                                      # 22 Nov STEP FRAMES (kra41's step-1 frame, every step): each premise also
FR = ([f"{a} is a citizen of {CIT[ci][0]} >" for a, ci in TRP + HEP] + [f"{c} > {REG[r][0]} >" for c, r in CIT] + [f"{r} > {CTY[k][0]}." for r, k in REG]) if FRM else []
if FRM >= 2:                                                                   # 23 Nov MIDDLE-STEP FRAME IN CONTEXT: every person's chain PREFIX up to the
    FR += [f"{a} is a citizen of {CIT[ci][0]} > {REG[reg(ci)][0]} >" for a, ci in TRP + HEP]   #   middle step ('A is a citizen of B > R >': premises only,
                                                                               #   the country never shown) - the standalone 'B > R >' did not carry into the chain   #   in the chain's own
if FRM: P(f"   STEP FRAMES on: {len(FR)} premise rows in the chain's own frames ('A is a citizen of B >', 'B > R >', 'R > K.'), 8 a step")   # context at its step; C never shown
for st in range(1, TE + 1):
    fr_ = [FR[i].encode('latin-1') for i in rs.randint(0, len(FR), 8)] if FR else []
    step(m, op2, [rows[i].encode('latin-1') for i in rs.randint(0, len(rows), 8)] + fr_ + [PRE[i].encode('latin-1') for i in rs.randint(0, len(PRE), 24 - len(fr_))])
    if st in (300, 750, 1500) or st == TE: P(f"   step {st}: TAUGHT {trunk_eval(TPOOL[:60])} || NEVER TAUGHT {trunk_eval(HN)}  ({time.time() - T0:.0f}s)")
P(f"   never-read names, the trunk's chain: well-formed 3-hop chains given {np.mean([len(gen(m, f'{a} is a citizen of ').split('>')) == 3 for a in NOV[:30]]):.2f}")
P(f"DONE ({time.time() - T0:.0f}s)")
