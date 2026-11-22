"""22 Nov REASONING WITHOUT THE LANGUAGE MODEL (offline, synthetic; LeCun's direction taken all the way).

The same five-level world as tools/latent_search.py (same seed -> the same names and facts; person -lives-> city -city_in-> region
-region_of-> country -country_in-> continent, every premise in two wordings, read 3 times). NO next-byte objective anywhere:
  JEPA ENCODER: a bidirectional byte transformer trained ONLY by latent prediction (I-JEPA style): a random 1-2 word span of a premise
    is masked in the context encoder's input; a predictor maps the context states at the masked span to the EMA target encoder's states
    of that span in the unmasked sentence (InfoNCE over the batch). No labels: the spans are random words, not names.
  ARM D, PURE JEPA REASONING: a hop is the JEPA predictor itself queried with masked text, '<head> lives in ______.', its output snapped
    to the nearest name code (the target encoder's states of each name in the sentences that contain it). Multi-hop answers by chaining
    hops; the composite relation's PROGRAM found by search over hop sequences scored on endpoints only; margin gate as the energy
    threshold; a 4-hop relation from 20 endpoints. No operator is fitted on labelled pairs: everything comes from reading sentences.
  ARM B: route J (one fitted predictor per relation, on labelled premise pairs) on the JEPA encoder's name codes.
  ARM C: route J on RANDOM name codes (control: how much of route J's success is a learned table rather than the encoder's knowledge).
Compare with tools/latent_search.py (route J on the causal trunk's codes, and the trunk's own written 3-hop chains).
Usage: JR_STEPS=5000 python3 tools/jepa_reason.py"""
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
PRE1 = sorted({t.format(a=a, b=b) for rel, fs in FACTS.items() for a, b in fs for t in W[rel]}); PRE = PRE1 * 3
L, V, PAD, MSK, DW, NL = 64, 256, 0, 1, 192, 3
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, kp):
        h = s.n1(x); x = x + s.a(h, h, h, key_padding_mask=kp, need_weights=False)[0]; return x + s.m(s.n2(x))
class Enc(nn.Module):                                                          # bidirectional: no causal mask
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, DW); s.pos = nn.Parameter(torch.randn(L, DW) * 0.02); s.b = nn.ModuleList(Blk(DW, 4) for _ in range(NL)); s.n = nn.LayerNorm(DW)
    def forward(s, X):
        kp = X == PAD; x = s.emb(X) + s.pos[:X.shape[1]]
        for b in s.b: x = b(x, kp)
        return s.n(x)
def tens(seqs):
    X = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
    return X
def spans(s):
    """byte spans of the words (trailing '.' excluded)"""
    out = []; i = 0
    for w in s.split(' '):
        w_ = w.rstrip('.'); out.append((i, i + len(w_))); i += len(w) + 1
    return out
ctx, tgt = Enc(), Enc(); tgt.load_state_dict(ctx.state_dict())
for p_ in tgt.parameters(): p_.requires_grad = False
pred = nn.Sequential(nn.Linear(DW, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, DW))
ST = int(os.environ.get('JR_STEPS', 5000)); BS = int(os.environ.get('JR_BS', 48))
opt = torch.optim.AdamW(list(ctx.parameters()) + list(pred.parameters()), lr=1e-3, weight_decay=0.01)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=ST, pct_start=0.05)
def span_mean(H, sp):
    return torch.stack([H[k, a:b].mean(0) for k, (a, b) in enumerate(sp)])
def predict(seqs, sp):
    """the predictor's latent for the masked span of each (already masked) sequence"""
    return pred(span_mean(ctx(tens(seqs)), sp))
NAMES = sorted({x for fs in FACTS.values() for p_ in fs for x in p_} | set(NOV)); NI = {n: i for i, n in enumerate(NAMES)}
@torch.no_grad()
def name_codes():
    """each name's code: the target encoder's mean state over the name in every premise containing it (never-read names: 'About <n>.')"""
    tgt.eval(); acc = {n: [] for n in NAMES}; items = []
    for rel, fs in FACTS.items():
        for a, b in fs:
            for t in W[rel]:
                s = t.format(a=a, b=b); ia = s.index(a); ib = s.index(b, ia + len(a)) if b in s[ia + len(a):] else s.index(b)
                items.append((s, a, (ia, ia + len(a)))); items.append((s, b, (ib, ib + len(b))))
    items += [(f"About {n}.", n, (6, 6 + len(n))) for n in NOV]
    for i in range(0, len(items), 256):
        ch = items[i:i + 256]; H = tgt(tens([s.encode('latin-1') for s, n, sp in ch])); z = span_mean(H, [sp for s, n, sp in ch])
        for (s, n, sp), v in zip(ch, z): acc[n].append(v)
    return F.normalize(torch.stack([torch.stack(acc[n]).mean(0) for n in NAMES]), dim=-1)
QW = {r: W[r][0] for r in W}                                                    # the query wording of each relation (the first)
MEMO = {}
@torch.no_grad()
def hopD(rel, heads):
    """PURE JEPA hop: '<head> <relation words> ______.' -> predictor -> nearest name code (6-byte mask: the place names' length)"""
    ctx.eval(); pred.eval(); need = [h for h in dict.fromkeys(heads) if (rel, h) not in MEMO]
    for i in range(0, len(need), 256):
        ch = need[i:i + 256]; qs = []; sp = []
        for h in ch:
            s = QW[rel].format(a=h, b='\x01' * 6); a_ = s.index('\x01'); qs.append(s.encode('latin-1')); sp.append((a_, a_ + 6))
        v = F.normalize(predict(qs, sp), dim=-1); sc = v @ ZD.T; t2 = sc.topk(2, -1)
        for h, i1, m1 in zip(ch, t2.indices[:, 0].tolist(), (t2.values[:, 0] - t2.values[:, 1]).tolist()): MEMO[(rel, h)] = (NAMES[i1], m1)
    ctx.train(); pred.train(); return [MEMO[(rel, h)] for h in heads]
def runD(prog, heads):
    cur = list(heads); mg = np.full(len(heads), 9.0)
    for r in prog:
        out = hopD(r, cur); cur = [o for o, m_ in out]; mg = np.minimum(mg, [m_ for o, m_ in out])
    return cur, torch.tensor(mg)
def onehop():
    ok = [o == CIT[ci][0] for (o, _), (a, ci) in zip(hopD('lives', [a for a, ci in HEP[:60]]), HEP[:60])]
    ok += [o == REG[r][0] for (o, _), (c, r) in zip(hopD('city_in', [c for c, r in CIT[:60]]), CIT[:60])]
    ok += [o == CTY[k][0] for (o, _), (r, k) in zip(hopD('region_of', [r for r, k in REG[:60]]), REG[:60])]
    ok += [o == CONT[c] for (o, _), (k, c) in zip(hopD('country_in', [k for k, c in CTY]), CTY)]
    return np.mean(ok)
for st in range(1, ST + 1):
    ss = [PRE[i] for i in rs.randint(0, len(PRE), BS)]; Xs = []; sp = []
    for s in ss:
        ws = spans(s); k = rs.randint(len(ws)); k2 = min(len(ws) - 1, k + rs.randint(2)); a_, b_ = ws[k][0], ws[k2][1]
        Xs.append((s[:a_] + '\x01' * (b_ - a_) + s[b_:]).encode('latin-1')); sp.append((a_, b_))
    with torch.no_grad(): z_ = F.normalize(span_mean(tgt(tens([s.encode('latin-1') for s in ss])), sp), dim=-1)
    p_ = F.normalize(predict(Xs, sp), dim=-1)
    l = F.cross_entropy(p_ @ z_.T / 0.1, torch.arange(BS)); opt.zero_grad(); l.backward(); opt.step(); sch.step()
    with torch.no_grad():
        mo = min(0.999, 0.99 + 0.009 * st / ST)
        for pt, pc in zip(tgt.parameters(), ctx.parameters()): pt.mul_(mo).add_(pc.detach(), alpha=1 - mo)
    if st % 1000 == 0 or st == ST:
        ZD = name_codes(); MEMO.clear()
        P(f"   JEPA step {st}: loss {float(l):.3f} | PURE JEPA one-hop recall (masked query -> nearest name code, {len(NAMES)} names) {onehop():.2f}  ({time.time() - T0:.0f}s)")
ZD = name_codes(); MEMO.clear()
NT = int(os.environ.get('LS_NT', 100)); TPOOL = [p_ for p_ in TRP if p_[1] % 2 == 0][:NT]; TC = {ci for a, ci in TPOOL}
HN = [p_ for p_ in HEP if p_[1] not in TC]
progs = [p for k in (1, 2, 3, 4) for p in itertools.product(list(FACTS), repeat=k)]
def search(runf, ex, tag):
    hs = [h for h, a in ex]; ans = [a for h, a in ex]; sh = list(np.random.RandomState(3).permutation(ans)); best, bestn = None, None
    for p_ in progs:
        out, mg = runf(p_, hs); acc = np.mean([o == a for o, a in zip(out, ans)]); accn = np.mean([o == a for o, a in zip(out, sh)])
        key = (acc, float(mg.mean()))
        if best is None or key > best[0]: best = (key, p_)
        if bestn is None or accn > bestn: bestn = accn
    P(f"   SEARCH for '{tag}' over {len(progs)} programs (length 1-4) on {len(ex)} endpoints: best {' o '.join(best[1])} (right {best[0][0]:.2f}, mean margin {best[0][1]:.3f}) | null (answers shuffled) best {bestn:.2f}")
    return best[1]
def arm(runf, tag):
    P(tag)
    pc = search(runf, [(a, CTY[cty(ci)][0]) for a, ci in TPOOL], 'citizen of (3 hops)')
    out, mg = runf(pc, [a for a, ci in HN]); rt = [o == CTY[cty(ci)][0] for o, (a, ci) in zip(out, HN)]
    tr_out, tr_mg = runf(pc, [a for a, ci in TPOOL]); tr_r = [o == CTY[cty(ci)][0] for o, (a, ci) in zip(tr_out, TPOOL)]
    th = float(np.quantile(tr_mg[torch.tensor(tr_r)].numpy(), 0.25)) if any(tr_r) else 0.0
    no_out, no_mg = runf(pc, NOV); g = (mg >= th)
    P(f"   citizen of, HELD-OUT people, cities in no taught example ({len(HN)}): right {np.mean(rt):.2f} | MARGIN GATE (t keeps 75% of right endpoint answers = {th:.3f}): "
      f"answered {float(g.float().mean()):.2f}, right when answered {np.mean([r_ for r_, g_ in zip(rt, g.tolist()) if g_]) if bool(g.any()) else float('nan'):.2f} | NEVER-READ names given an answer {float((no_mg >= th).float().mean()):.2f}")
    pk = search(runf, [(a, CONT[cnt(ci)]) for a, ci in TPOOL[:20]], 'continent (4 hops, 20 endpoints)')
    o4, _ = runf(pk, [a for a, ci in HN]); P(f"   continent (4 hops), HELD-OUT people ({len(HN)}): right {np.mean([o == CONT[cnt(ci)] for o, (a, ci) in zip(o4, HN)]):.2f}")
arm(runD, f"ARM D, PURE JEPA REASONING (no language model, no fitted operators, no labelled pairs; hops = the JEPA predictor on masked text):")
def routeJ(Z):
    D = Z.shape[1]
    def fit(pairs, steps=1500):
        torch.manual_seed(11); net = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D)); o = torch.optim.Adam(net.parameters(), lr=1e-3)
        A_ = torch.tensor([NI[a] for a, b in pairs]); B_ = torch.tensor([NI[b] for a, b in pairs])
        for _ in range(steps):
            k = torch.randint(0, len(A_), (min(256, len(A_)),)); l = F.cross_entropy(F.normalize(net(Z[A_[k]]), dim=-1) @ Z.T / 0.05, B_[k]); o.zero_grad(); l.backward(); o.step()
        return net.eval()
    OPS = {r: fit(fs) for r, fs in FACTS.items()}
    @torch.no_grad()
    def run(prog, heads):
        v = Z[torch.tensor([NI[h] for h in heads])]; mg = torch.full((len(heads),), 9.0)
        for r in prog:
            s = F.normalize(OPS[r](v), dim=-1) @ Z.T; t2 = s.topk(2, -1); mg = torch.minimum(mg, t2.values[:, 0] - t2.values[:, 1]); v = Z[t2.indices[:, 0]]
        return [NAMES[i] for i in t2.indices[:, 0].tolist()], mg
    return run
Zc = ZD - ZD.mean(0); arm(routeJ(F.normalize(Zc, dim=-1)), "ARM B, ROUTE J on the JEPA encoder's name codes (fitted operators on labelled premise pairs):")
torch.manual_seed(5); arm(routeJ(F.normalize(torch.randn(len(NAMES), DW), dim=-1)), "ARM C, ROUTE J on RANDOM name codes (control):")
P(f"DONE ({time.time() - T0:.0f}s)")
