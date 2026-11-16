"""IMPROVING TYPED INFERENCE (16 Nov 2026; tools/typed_plus.py) - offline, GPU write path, distant_50k, stand-in trunk, label-free choices.
The writes of round 2 (visit 4; ranks 0-200 = the batch, 200-400 = the NEXT TIER the GPU's second batch resembles); the composition table
learned from round 1's confirmations (tools/typed_routes.py TR_STAGE=3w output, TR_OUT). Later text: relational records of documents read
after the write whose first name is a written name; first half / second half.
  1 COMBINED TYPING: the composition wording where the table types the write, else the class-pair wording (route 1)
  2 USEFULNESS CHOOSES THE WORDING: candidates = the class pair's top 4 wordings + the composition wording; each write keeps the one that
    saves the most bits on the FIRST half (label-free); scored on the SECOND half (bits) and by the distant labels (typed accuracy)
  3 PRECISION ON THE NEXT TIER: share right of ranks 200-400, and of those KEPT by usefulness (the typed write saved bits on the first
    half) or by composition support (the table types it), against the batch (ranks 0-200)
Typed accuracy: exact Wikidata relation and with the 'is in' relations as one (P131, P17, P30, P706, P276, P361), oriented.
Usage: TG_CKPT=<trunk> TR_OUT=<typed_routes rounds prefix> python3 tools/typed_plus.py <distant_50k.json.gz>"""
import sys, os, json, types, time, collections, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.kunits import _row_bits, CL
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2)
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=400, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; TY = pm._PM['TY']; PRD = pm._PM['PRD']
h = lambda v: int(round(ND * min(1.0, (v + 1) / 10))); v = int(os.environ.get('TP_V', 3))   # TP_V 3 / 5: round 2 / 3 (the table from the round before)
isrel = lambda r: 1 <= r[5] <= 8 and not pm._rret_listlike(r)
norm = lambda w: ' '.join(t for t in w.split() if t not in ('the', 'a', 'an'))
CONT = {'P131', 'P17', 'P30', 'P706', 'P276', 'P361'}
def cg(x, grp):
    if not grp: return x
    b, inv = (x[:-3], '^-1') if x.endswith('^-1') else (x, ''); return ('CONTAINED' + inv) if b in CONT else x
def rels(X, Y, grp): return {cg(x, grp) for x in PRD.get((X, Y), ())} | {cg(x + '^-1', grp) for x in PRD.get((Y, X), ())}
RMc = collections.defaultdict(collections.Counter)
for r in R:
    if r[0] < ND and r[6] and isrel(r):
        for x in set(PRD.get((r[1], r[2]), ())) | {x + '^-1' for x in PRD.get((r[2], r[1]), ())}: RMc[norm(r[3])][x] += 1
RM = {w: c.most_common(1)[0][0] for w, c in RMc.items()}
def right(X, w, Y, grp): return w in RM and cg(RM[w], grp) in rels(X, Y, grp)
pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(v)], v); W = dict(pm._PM['rti'])
y = {k: e['truth'] for k, e in W.items()}; tier = {k: ('batch' if e['rank'] < 200 else 'next') for k, e in W.items()}
nb = collections.defaultdict(set)
for r in R:
    if r[0] < h(v): nb[r[1]].add(r[2]); nb[r[2]].add(r[1])
DG = {n: len(s) for n, s in nb.items()}
tab = collections.defaultdict(collections.Counter); pw = collections.defaultdict(list)
need = {frozenset((A, b)) for (A, C), e in W.items() for b in e['prem']} | {frozenset((b, C)) for (A, C), e in W.items() for b in e['prem']}
for r in R:
    if r[0] < h(v) and isrel(r):
        if r[1] in DG and r[2] in DG and DG[r[1]] != DG[r[2]]:
            hd, tl, fw = (r[1], r[2], 1) if DG[r[1]] < DG[r[2]] else (r[2], r[1], 0); w_ = norm(r[3])
            if any(ch.isalpha() for ch in w_): tab[(TY.get(hd, 'X'), TY.get(tl, 'X'))][(w_, fw)] += 1
        f = frozenset((r[1], r[2]))
        if f in need: pw[f].append((r[1], r[2], norm(r[3])))
def orient(X, Y):
    cs = pw.get(frozenset((X, Y)))
    if not cs: return None
    a, b, w = collections.Counter(cs).most_common(1)[0][0]; return w if a == X else w + ' ^R'
CLU = {}
if int(os.environ.get('TP_CLUSTER', 0)):                                       # 16 Nov DENSER TABLE: wordings that state the same name pairs are merged
    byw = collections.defaultdict(set)                                         #   (label-free; union of wordings sharing 3+ pairs, Jaccard >= TP_JAC)
    for r in R:
        if r[0] < h(v) and isrel(r): byw[norm(r[3])].add(frozenset((r[1], r[2])))
    byw = {w: s_ for w, s_ in byw.items() if len(s_) >= 3}; inv = collections.defaultdict(list)
    for w, s_ in byw.items():
        for f in s_: inv[f].append(w)
    par = {w: w for w in byw}
    def fnd(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    co = collections.Counter()
    for f, ws in inv.items():
        if len(ws) <= 30:
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)): co[(ws[i], ws[j]) if ws[i] < ws[j] else (ws[j], ws[i])] += 1
    jac = float(os.environ.get('TP_JAC', 0.1))
    for (a, b), n in co.items():
        if n >= 3 and n / len(byw[a] | byw[b]) >= jac: par[fnd(a)] = fnd(b)
    grp = collections.defaultdict(list)
    for w in byw: grp[fnd(w)].append(w)
    for g, ws in grp.items():
        rep = max(ws, key=lambda w: len(byw[w]))
        for w in ws: CLU[w] = rep
    P(f"WORDING CLUSTERS: {len(byw)} wordings with 3+ pairs -> {len(grp)} clusters; largest: " + " | ".join("/".join(sorted(ws, key=lambda w: -len(byw[w]))[:5]) for g, ws in sorted(grp.items(), key=lambda kv: -len(kv[1]))[:4]))
def cl_(w):
    if w is None: return None
    rv = w.endswith(' ^R'); w0 = w[:-3] if rv else w; c = CLU.get(w0, w0); return c + ' ^R' if rv else c
T3 = collections.defaultdict(collections.Counter)                              # the composition table from round 1's confirmations
for r in json.load(open(os.environ['TR_OUT'] + f"_r{(v - 1) // 2}.json")):
    if r['cf'] is None: continue
    for p in r['prem']:
        if p[0] and p[1]: T3[(cl_(p[0]), cl_(p[1]))][cl_(r['cf'])] += 1
CAND = {}; COMP = {}
for (A, C), e in W.items():
    hd, tl = (A, C) if DG.get(A, 0) <= DG.get(C, 0) else (C, A); cp = (TY.get(hd, 'X'), TY.get(tl, 'X')); cs = []
    for (w, fw), _ in tab.get(cp, collections.Counter()).most_common(4): cs.append((hd, w, tl) if fw else (tl, w, hd))
    votes = collections.Counter()
    for b in e['prem']:
        a1, a2 = cl_(orient(A, b)), cl_(orient(b, C))
        if a1 and a2: votes.update(T3.get((a1, a2), {}))
    if votes:
        cw = votes.most_common(1)[0][0]; rev = cw.endswith(' ^R'); w0 = cw[:-3] if rev else cw; COMP[(A, C)] = (C, w0, A) if rev else (A, w0, C)
    CAND[(A, C)] = cs
P(f"WRITES {len(W)} at visit {v + 1}: batch (ranks 0-200) right {np.mean([y[k] for k in W if tier[k] == 'batch']):.3f}, next tier (200-400) right "
  f"{np.mean([y[k] for k in W if tier[k] == 'next']):.3f} | composition table ({sum(len(c) for c in T3.values())} confirmations from round 1) types {len(COMP)} "
  f"({np.mean([k in COMP for k in W]):.2f})  ({time.time() - T0:.0f}s)")
# later text and the trunk
names = {n for k in W for n in k}; mid = (h(v) + h(v + 2)) // 2
spans = [(r[0] < mid, r[7], r[1]) for r in R if h(v) <= r[0] < h(v + 2) and isrel(r) and r[1] in names]
DW = int(os.environ.get('TG_D', 192)); L_ = 128; PAD = 256; V_ = 257
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
lm = LM(); lm.load_state_dict(torch.load(os.environ['TG_CKPT'])); lm.eval()
TXT = [s[1].encode('latin-1', 'replace')[:L_] for s in spans]; PT, EN = [], []
with torch.no_grad():
    for i0 in range(0, len(TXT), 256):
        ch = TXT[i0:i0 + 256]; T = max(len(t) for t in ch); X = torch.full((len(ch), T), PAD, dtype=torch.long)
        for i, t in enumerate(ch): X[i, :len(t)] = torch.tensor(list(t))
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(lm(Xi), -1)[..., :256]; lp = lp - torch.logsumexp(lp, -1, keepdim=True)
        pt = lp.gather(-1, X.clamp(max=255)[..., None])[..., 0].exp().numpy(); en = (-(lp.exp() * lp).sum(-1) / math.log(2)).numpy()
        for i, t in enumerate(ch): PT.append(pt[i, :len(t)]); EN.append(en[i, :len(t)])
first = collections.defaultdict(list); second = []
for i, s in enumerate(spans):
    if s[0]: first[s[2]].append(i)
    else: second.append(i)
P(f"later relational spans naming a written name first: {len(spans)} (first half {sum(len(x) for x in first.values())}, second {len(second)})  ({time.time() - T0:.0f}s)")
def mem(claims):
    M = {'NAME': {}, 'DATE': {}, 'NUMBER': {}}
    for X, w, Y in claims: M['NAME'].setdefault(X.encode('latin-1', 'replace'), collections.deque(maxlen=32)).append((' ' + w + ' ' + Y + '.').encode('latin-1', 'replace')[:CL])
    return dict(mem=M, gate={'NAME': {}, 'ALL': {}})
def bits(claims, idx):
    st = mem(claims); s = 0.0; n = 0
    for i in idx: s += _row_bits(st, TXT[i], PT[i], EN[i], ('NAME',), 'NAME', 0.05)[0]; n += len(TXT[i])
    return 1000 * s / max(1, n)
def gain(claim):
    idx = first.get(claim[0], []); return bits([claim], idx) if idx else None
B = [k for k in W if tier[k] == 'batch']
FORMS = {'F1 connected to': {k: (k[0], 'is connected to', k[1]) for k in B},
         'F2 class pair (route 1)': {k: CAND[k][0] for k in B if CAND[k]},
         'F3 composition, else class pair': {k: COMP.get(k, CAND[k][0] if CAND[k] else None) for k in B}}
uch = {}
for k in B:                                                                    # 2 usefulness chooses among the candidates
    cs = list(CAND[k]) + ([COMP[k]] if k in COMP and COMP[k] not in CAND[k] else [])
    gs = [(gain(c), c) for c in cs]; gs = [(g, c) for g, c in gs if g is not None]
    uch[k] = max(gs)[1] if gs else (CAND[k][0] if CAND[k] else None)
FORMS['F4 usefulness chooses'] = uch
for nm, fm in FORMS.items():
    cl = [(k, c) for k, c in fm.items() if c is not None]
    acc = [(right(*c, 0), right(*c, 1)) for k, c in cl]
    P(f"   {nm:32s} {len(cl)} typed | typed claim right exact {np.mean([a for a, b in acc]):.3f}, 'is in' as one {np.mean([b for a, b in acc]):.3f} | bits saved per 1,000 bytes "
      f"of the SECOND half {bits([c for k, c in cl], second):+.2f} | wordings: " + ", ".join(f"'{w}' {n}" for w, n in collections.Counter(c[1] for k, c in cl).most_common(5)))
# 3 precision on the next tier
for tr in ('batch', 'next'):
    ks = [k for k in W if tier[k] == tr]
    g2 = {k: gain(CAND[k][0]) for k in ks if CAND[k]}
    ku = [k for k in ks if (g2.get(k) or 0) > 0]; kc = [k for k in ks if k in COMP]; kb = [k for k in ks if k in COMP or (g2.get(k) or 0) > 0]
    f = lambda L: f"{np.mean([y[k] for k in L]):.3f} ({len(L)})" if L else 'nan (0)'
    P(f"   PRECISION, {tr:5s}: all {f(ks)} | kept by usefulness (the typed write saved bits on the first half) {f(ku)} | composition-supported {f(kc)} | either {f(kb)} | "
      f"neither {f([k for k in ks if k not in kb])}")
P(f"DONE ({time.time() - T0:.0f}s)")
