"""23 Nov CAN A JEPA STORE KNOWLEDGE? (offline, synthetic; the five-level world of tools/latent_search.py, same seed; no next-byte objective)
tools/jepa_reason.py: a bidirectional encoder trained only by masked-word latent prediction (in-batch InfoNCE, 48 sentences) recalled
0.11 of one-hop facts after 5k steps (the next-byte trunk: 1.00). Three ways a JEPA might store facts, each trained the same number of steps:
  WORD  - the v1 recipe (random 1-2 word masks, in-batch negatives) - the baseline;
  BANK  - whole capitalised-name spans masked (no labels: any run of capitalised words) and a MEMORY BANK of the last 4,096 targets as
          negatives (fine identity needs many negatives, not 47);
  SLOT  - JEPA PREDICTING A KEYED MEMORY: each distinct masked name string owns a slot vector (the memory, keyed by the string as the
          keyed units are), updated as an EMA of the target encoder's states of that name; the predictor is trained against ALL slots
          (InfoNCE over the slot table) - memory as the store, the JEPA as the reader and predictor.
Each: one-hop recall by step (the predictor on '<head> lives in ______.' snapped to the nearest name code / slot), and two- and three-hop
answers by chaining its own hops (no fitted operators, no labelled pairs).
Usage: JS_STEPS=8000 python3 tools/jepa_store.py"""
import os, time
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
L, V, PAD, DW, NL = 64, 256, 0, 192, 3
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, kp):
        h = s.n1(x); x = x + s.a(h, h, h, key_padding_mask=kp, need_weights=False)[0]; return x + s.m(s.n2(x))
class Enc(nn.Module):
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
def words(s):
    out = []; i = 0
    for w in s.split(' '):
        w_ = w.rstrip('.'); out.append((i, i + len(w_), w_)); i += len(w) + 1
    return out
def capspans(s):
    """runs of capitalised words (no labels: names are capitalised) -> (start, end, string); the first word of a sentence only if the run continues"""
    ws = words(s); out = []; k = 0
    while k < len(ws):
        if ws[k][2][:1].isupper():
            j = k
            while j + 1 < len(ws) and ws[j + 1][2][:1].isupper(): j += 1
            if not (k == 0 and j == 0 and ws[0][2] in ('Ask', 'The')): out.append((ws[k][0], ws[j][1], ' '.join(w[2] for w in ws[k:j + 1])))
            k = j + 1
        else: k += 1
    return out
span_mean = lambda H, sp: torch.stack([H[k, a:b].mean(0) for k, (a, b) in enumerate(sp)])
NAMES = sorted({x for fs in FACTS.values() for p_ in fs for x in p_}); NI = {n: i for i, n in enumerate(NAMES)}
QW = {r: W[r][0] for r in W}; ST = int(os.environ.get('JS_STEPS', 8000)); BS = 48
def run(variant, phases=None):
    torch.manual_seed(0); r_ = np.random.RandomState(1)
    ctx, tgt = Enc(), Enc(); tgt.load_state_dict(ctx.state_dict())
    for p_ in tgt.parameters(): p_.requires_grad = False
    pred = nn.Sequential(nn.Linear(DW, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, DW))
    opt = torch.optim.AdamW(list(ctx.parameters()) + list(pred.parameters()), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=ST, pct_start=0.05)
    bank = torch.zeros(0, DW); SL = None
    if variant == 'slot':
        keys = sorted({sp[2] for s in PRE1 for sp in capspans(s)}); SI = {k: i for i, k in enumerate(keys)}; SL = F.normalize(torch.randn(len(keys), DW), dim=-1)
    @torch.no_grad()
    def codes():
        """name codes: the target encoder's mean state over the name in every premise containing it (slots for SLOT)"""
        if variant == 'slot': return torch.stack([SL[SI[n]] if n in SI else torch.zeros(DW) for n in NAMES])
        acc = {n: [] for n in NAMES}; items = []
        for s in PRE1:
            for a, b, st in capspans(s):
                if st in acc: items.append((s, st, (a, b)))
        for i in range(0, len(items), 256):
            ch = items[i:i + 256]; z = span_mean(tgt(tens([s.encode('latin-1') for s, n, sp in ch])), [sp for s, n, sp in ch])
            for (s, n, sp), v in zip(ch, z): acc[n].append(v)
        return F.normalize(torch.stack([torch.stack(acc[n]).mean(0) if acc[n] else torch.zeros(DW) for n in NAMES]), dim=-1)
    @torch.no_grad()
    def hop(rel, heads, Z):
        qs, sp = [], []
        for h in heads:
            s = QW[rel].format(a=h, b='\x01' * 6); a_ = s.index('\x01'); qs.append(s.encode('latin-1')); sp.append((a_, a_ + 6))
        out = []
        for i in range(0, len(qs), 256):
            v = F.normalize(pred(span_mean(ctx(tens(qs[i:i + 256])), sp[i:i + 256])), dim=-1); out += [NAMES[j] for j in (v @ Z.T).argmax(-1).tolist()]
        return out
    def evaluate(st):
        ctx.eval(); pred.eval(); Z = codes()
        o1 = hop('lives', [a for a, ci in HEP], Z); r1 = np.mean([o == CIT[ci][0] for o, (a, ci) in zip(o1, HEP)])
        o2 = hop('city_in', [c for c, r in CIT], Z); r2 = np.mean([o == REG[r][0] for o, (c, r) in zip(o2, CIT)])
        c2 = hop('city_in', o1, Z); t2 = np.mean([o == REG[reg(ci)][0] for o, (a, ci) in zip(c2, HEP)])
        c3 = hop('region_of', c2, Z); t3 = np.mean([o == CTY[cty(ci)][0] for o, (a, ci) in zip(c3, HEP)])
        ctx.train(); pred.train()
        P(f"   {variant:5s} step {st}: ONE-HOP recall lives {r1:.2f}, city_in {r2:.2f} | CHAINED by its own hops: 2 hops {t2:.2f}, 3 hops {t3:.2f}  ({time.time() - T0:.0f}s)")
    def evalc(tag):                                                            # continual mode: 'lives' recall per domain of people
        ctx.eval(); pred.eval(); Z = codes(); out = []
        for g, peo in enumerate(GROUPS):
            o = hop('lives', [a for a, ci in peo], Z); out.append(np.mean([x == CIT[ci][0] for x, (a, ci) in zip(o, peo)]))
        if BAD:                                                                # false-memory readout: what the planted people are recalled as
            hs = sorted(BAD); o = hop('lives', hs, Z); tru = {a: CIT[ci][0] for a, ci in GROUPS[0]}
            P(f"      planted people ({len(hs)}): answer the FALSE city {np.mean([x == BAD[a][0] for x, a in zip(o, hs)]):.2f} | the TRUE city {np.mean([x == tru[a] for x, a in zip(o, hs)]):.2f} | other {np.mean([x not in (BAD[a][0], tru[a]) for x, a in zip(o, hs)]):.2f}")
        ctx.train(); pred.train(); P(f"   {variant:5s} {tag}: 'lives' recall by domain " + ' | '.join(f"D{g} {v:.2f}" for g, v in enumerate(out)) + f"  ({time.time() - T0:.0f}s)")
    sched = [(PRE, ST, None)] if phases is None else phases; st = 0
    for pool, n_, tag in sched:
      if callable(pool):                                                       # 23 Nov DREAM replay: the pool is built at the phase start from the
          ctx.eval(); pred.eval(); pool = pool(lambda rel, heads: hop(rel, heads, codes())); ctx.train(); pred.train()   #   model's own recall
      for _ in range(n_):
        st += 1
        ss = [pool[i] for i in r_.randint(0, len(pool), BS)]; Xs, sp, ky = [], [], []
        for s in ss:
            if variant == 'word':
                ws = words(s); k = r_.randint(len(ws)); k2 = min(len(ws) - 1, k + r_.randint(2)); a_, b_ = ws[k][0], ws[k2][1]; key = None
            else:
                cs = capspans(s); a_, b_, key = cs[r_.randint(len(cs))]
            Xs.append((s[:a_] + '\x01' * (b_ - a_) + s[b_:]).encode('latin-1')); sp.append((a_, b_)); ky.append(key)
        with torch.no_grad(): z_ = F.normalize(span_mean(tgt(tens([s.encode('latin-1') for s in ss])), sp), dim=-1)
        p_ = F.normalize(pred(span_mean(ctx(tens(Xs)), sp)), dim=-1)
        if variant == 'word': l = F.cross_entropy(p_ @ z_.T / 0.1, torch.arange(BS))
        elif variant == 'bank':
            neg = bank[:4096]; l = F.cross_entropy(torch.cat([p_ @ z_.T, p_ @ neg.T], 1) / 0.1, torch.arange(BS)); bank = torch.cat([z_.detach(), bank])[:4096]
        else:
            ix = torch.tensor([SI[k] for k in ky]); l = F.cross_entropy(p_ @ SL.clone().T / 0.1, ix)
        opt.zero_grad(); l.backward(); opt.step(); sch.step()
        if variant == 'slot':
            with torch.no_grad():                                                  # the memory slots: EMA of the target encoder's states of each name
                for k_, v in zip(ix.tolist(), z_): SL[k_] = F.normalize(0.9 * SL[k_] + 0.1 * v, dim=0)
        with torch.no_grad():
            mo = min(0.999, 0.99 + 0.009 * st / ST)
            for pt, pc in zip(tgt.parameters(), ctx.parameters()): pt.mul_(mo).add_(pc.detach(), alpha=1 - mo)
        if phases is None and (st % 2000 == 0 or st == ST): evaluate(st)
      if phases is not None: evalc(tag)
P(f"CAN A JEPA STORE KNOWLEDGE? {len(NAMES)} names, {len(PRE1)} premise sentences (x3), {ST} steps each; the next-byte trunk recalls these premises 1.00 by 2k steps")
GROUPS = []; BAD = {}; BADF = float(os.environ.get('JS_DREAMBAD', 0))
if int(os.environ.get('JS_CONT', 0)):                                          # 23 Nov CONTINUAL: three domains of people learned one after another;
    PEO = TRP + HEP; GROUPS = [PEO[g::3] for g in range(3)]                    # geography in every domain's text; forgetting of earlier domains
    def sents(peo):                                                            #   with and without replay
        ps = {a for a, ci in peo}; return [s for s in PRE1 if any((' ' + a + ' ') in (' ' + s + ' ') for a in ps)]
    LIV = {g: sents(GROUPS[g]) for g in range(3)}; ALLP = set(x for g in range(3) for x in LIV[g]); GEO = [s for s in PRE1 if s not in ALLP]
    PH = int(os.environ.get('JS_PHASE', 4000)); RP = float(os.environ.get('JS_REPLAY', 0.0))
    P(f"CONTINUAL KEYED-MEMORY JEPA: 3 domains of people ({', '.join(str(len(LIV[g])) for g in range(3))} sentences) + {len(GEO)} geography sentences; {PH} steps a domain; replay of earlier domains {RP:.0%}")
    DREAM = [float(x) for x in os.environ.get('JS_DREAM', '').split(',') if x]   # 23 Nov: replay of the model's OWN recall of earlier domains
    def dpool(g, rp):                                                          #   ('<person> lives in <recalled city>'), no stored sentences
        def mk(recall):
            cur = (LIV[g] + GEO) * 3; heads = [a for h in range(g) for a, ci in GROUPS[h]]
            if not heads: return cur
            got = recall('lives', heads)
            if BADF and g == 1:                                                # 23 Nov FALSE MEMORY: a fraction of domain 0's recalls replaced by a
                rb = np.random.RandomState(11); cities = [c for c, r in CIT]     #   wrong city in the FIRST dream only; tracked afterwards
                for i, a in enumerate(heads):
                    if rb.rand() < BADF:
                        w_ = cities[rb.randint(len(cities))]
                        while w_ == got[i]: w_ = cities[rb.randint(len(cities))]
                        BAD[a] = (w_, got[i]); got[i] = w_
                P(f"      FALSE MEMORY: {len(BAD)} of {len(heads)} dreamed recalls replaced by a wrong city (first dream only)")
            dw = W['lives'][int(os.environ.get('JS_DREAMW', 0))]; dr = [dw.format(a=a, b=c) for a, c in zip(heads, got)]
            ok = np.mean([c == CIT[ci][0] for c, (a, ci) in zip(got, [x for h in range(g) for x in GROUPS[h]])])
            P(f"      domain {g}: {len(dr)} dream sentences in the wording '{dw}' from the model's own recall of earlier domains (correct {ok:.2f})")
            return cur + dr * max(1, int(round(rp * len(cur) / len(dr) / (1 - rp))))
        return mk
    arms = [('data', r) for r in ((0.0, RP) if RP else (0.0,))] if not DREAM else [('dream', r) for r in DREAM]
    for kind, rp in arms:
        ph = []
        for g in range(3):
            if kind == 'dream': ph.append((dpool(g, rp), PH, f"after domain {g} (DREAM replay {rp:.0%})")); continue
            cur = (LIV[g] + GEO) * 3; old = [x for h in range(g) for x in LIV[h]]
            pool = cur if not (rp and old) else cur + old * max(1, int(round(rp * len(cur) / max(1, len(old)) / (1 - rp))))
            ph.append((pool, PH, f"after domain {g} (replay {rp:.0%})"))
        ST = PH * 3; run('slot', ph)
else:
    for v in os.environ.get('JS_VARIANTS', 'word,bank,slot').split(','): run(v)
P(f"DONE ({time.time() - T0:.0f}s)")
