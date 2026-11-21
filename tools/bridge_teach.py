"""TEACH THE BRIDGE ON REAL-TEXT CHAINS (21 Nov 2026; tools/bridge_teach.py) - offline, A6 stand-in trunk.
The bridge test (tools/bridge_probe.py) found the trunk does not form the middle name B before spelling it (probe 0.09-0.15 vs 0.84 spelled), and no
new layer composes untaught chains (0.18-0.215). In the synthetic world (RESULTS 18-19 Nov, compose_synth CS_COT) a model taught to WRITE the bridge
('A is a citizen of B > C') composed for never-taught people at 0.92-1.00, against 0.12-0.14 taught directly. Real-text inferences mix many relation
types: does bridge teaching still teach a decomposition, or only memorise the taught chains?
Chains A -w1-> B -w2-> C from the package's code writes (as tools/bridge_probe.py); TRAIN 400 chains, TEST 200 chains on bridges never in a train chain.
The stand-in first stores every premise (PRE steps, KL anchor). Then one arm per run of STEPS (4 premise rows + 4 conclusion rows + 24 anchor rows):
  DIRECT  'Q: A tw ? So A tw C.'                         (today's teaching: the conclusion only)
  BRIDGE  'Q: A tw ? A w1 B. B w2 C. So A tw C.'          (the conclusion with its bridge written out)
Test, TRAIN (taught) and TEST (never taught): DIRECT scored at 'Q: A tw ? So A tw '; BRIDGE: the model decodes greedily from 'Q: A tw ? ' and
C is scored where it writes 'So A tw ' (its own bridge in context). C ranked first among 10 (chance 0.10); also the model's own B right (exact),
the chain format written, and C exact. Forgetting as bits on held-out text of five domains.
Usage: TG_CKPT=<A6 trunk> python3 tools/bridge_teach.py <distant_50k.json.gz>   env BT_W 2400, BT_PRE 3000, BT_STEPS 3000, BT_ARMS DIRECT,BRIDGE"""
import os, sys, types, copy, time, math, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2); torch.manual_seed(0)
NW = int(os.environ.get('BT_W', 2400)); PRE = int(os.environ.get('BT_PRE', 3000)); STEPS = int(os.environ.get('BT_STEPS', 3000)); LR = 1e-3
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=NW, inf_rret=0, inf_rti_ctl=0.0, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rti_typed=1)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; RD = [r for r in R if r[0] < int(round(ND * 0.4))]
pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], RD, 1)
W = [(k, w) for k, w in pm._PM['rti'].items() if w.get('tw') and w.get('prem')]
need = {frozenset((a, b)) for (a, c), w in W for b in w['prem']} | {frozenset((b, c)) for (a, c), w in W for b in w['prem']}
PW = collections.defaultdict(collections.Counter)
for r in RD:
    if 1 <= r[5] <= 8 and frozenset((r[1], r[2])) in need and any(ch.isalpha() for ch in r[3]): PW[frozenset((r[1], r[2]))][(r[1], pm._rnorm(r[3]), r[2])] += 1
del R, RD, pm._PM['R']
def prem(X, Y):
    c = PW.get(frozenset((X, Y)))
    if not c: return None
    c2 = [(v, k) for k, v in c.items() if k[0] == X and k[2] == Y]; return max(c2)[1] if c2 else None
CL = []
for (A_, C_), w in W:
    th, tt = w['th'], w['tt']
    for b in w['prem']:
        p1, p2 = prem(th, b), prem(b, tt)
        if p1 and p2 and b not in (th, tt) and len(f"Q: {th} {w['tw']} ? {p1[0]} {p1[1]} {p1[2]}. {p2[0]} {p2[1]} {p2[2]}. So {th} {w['tw']} {tt}.") < 124:
            CL.append((th, w['tw'], tt, bool(w['truth']), p1, p2)); break
CL = [CL[j] for j in np.random.RandomState(7).permutation(len(CL))]
BR = sorted({c[4][2] for c in CL}); rb = np.random.RandomState(5); rb.shuffle(BR); BTEST = set(BR[:int(len(BR) * 0.3)])
TR = [c for c in CL if c[4][2] not in BTEST][:400]; TE = [c for c in CL if c[4][2] in BTEST][:200]
P(f"{len(CL)} chains that fit the window (right {np.mean([c[3] for c in CL]):.3f}) -> TRAIN {len(TR)}, TEST {len(TE)} on bridges never in a train chain; "
  f"first-hop wordings: {len({c[4][1] for c in CL})} distinct, conclusion wordings {len({c[1] for c in CL})}  ({time.time() - T0:.0f}s)")
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
        Tn = X.shape[1]; am = torch.triu(torch.ones(Tn, Tn, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:Tn]
        for b in s.b: x = b(x, am)
        return s.o(s.n(x))
base = LM(); base.load_state_dict(torch.load(os.environ['TG_CKPT'])); base.eval()
MP = os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile')
DOMS = {d: np.frombuffer(open(f'{MP}/{d}.bin', 'rb').read().replace(b'\n', b' '), np.uint8) for d in ('wiki', 'prose', 'arxiv', 'pubmed', 'law')}
enc = lambda s: s.encode('latin-1', 'replace')[:L_]
def tens(bl):
    bl = [b[:L_] for b in bl]; Tn = max(len(b) for b in bl); X = torch.full((len(bl), Tn), PAD, dtype=torch.long)
    for i, b in enumerate(bl): X[i, :len(b)] = torch.tensor(list(b))
    return X, torch.cat([torch.full((len(bl), 1), PAD), X[:, :-1]], 1)
PF = ("{a} {w} {b}.", "Notes: {a} {w} {b}.", "We read that {a} {w} {b}.", "It is known that {a} {w} {b}.", "Fact: {a} {w} {b}.", "Reportedly, {a} {w} {b}.")
PREM = [enc(f.format(a=p[0], w=p[1], b=p[2])) for c in TR + TE for p in (c[4], c[5]) for f in PF]
q_ = lambda c: f"Q: {c[0]} {c[1]} ? "
so_ = lambda c: f"So {c[0]} {c[1]} "
ROWS = {'DIRECT': [enc(q_(c) + so_(c) + c[2] + '.') for c in TR],
        'BRIDGE': [enc(q_(c) + f"{c[4][0]} {c[4][1]} {c[4][2]}. {c[5][0]} {c[5][1]} {c[5][2]}. " + so_(c) + c[2] + '.') for c in TR]}
EVD = {d: [bytes(a[len(a) - 200000 + k * 129:len(a) - 200000 + (k + 1) * 129]) for k in range(64)] for d, a in DOMS.items()}
@torch.no_grad()
def dbits(m):
    return np.mean([float(F.cross_entropy(m(tens(e)[1]).reshape(-1, V_), tens(e)[0].reshape(-1), ignore_index=PAD)) / math.log(2) for e in EVD.values()])
mn = min(len(a) for a in DOMS.values()) - 200400
def step(m, op, rs, fb):
    ds_ = list(DOMS); wb = [bytes(DOMS[ds_[j % 5]][i:i + 129]) for j, i in enumerate(rs.randint(0, mn, 24))]
    Xf, Xfi = tens(fb); lf = F.cross_entropy(m(Xfi).reshape(-1, V_), Xf.reshape(-1), ignore_index=PAD)
    Xw, Xwi = tens(wb)
    with torch.no_grad(): lo = F.log_softmax(base(Xwi), -1)
    lw = F.kl_div(F.log_softmax(m(Xwi), -1), lo, log_target=True, reduction='batchmean') / Xw.shape[1]
    loss = (8 * lf + 24 * lw) / 32; op.zero_grad(); loss.backward(); op.step()
rr = np.random.RandomState(3); TAILS = [c[2] for c in CL]
CAND = {c: [c[2]] + [x for x in (TAILS[j] for j in rr.choice(len(TAILS), 14, replace=False)) if x != c[2]][:9] for c in TR + TE}
@torch.no_grad()
def rank_at(m, pref, cands):
    pb = enc(pref); bl = [(pb + x.encode('latin-1', 'replace') + b'.')[:L_] for x in cands]
    if len(pb) + 2 >= L_: return 0
    X, Xi = tens(bl); lp = F.log_softmax(m(Xi), -1); n0 = len(pb)
    return int(np.argmax([float(lp[i, n0:len(b)].gather(-1, X[i, n0:len(b), None]).mean()) if len(b) > n0 else -1e9 for i, b in enumerate(bl)]) == 0)
@torch.no_grad()
def greedy(m, prompts, n=110):                                                # the model reads [PAD] + bytes (as in training) and predicts the next byte
    out = [enc(p) for p in prompts]
    for _ in range(n):
        Tn = max(len(b) for b in out) + 1; X = torch.full((len(out), min(Tn, L_)), PAD, dtype=torch.long)
        for i, b in enumerate(out): X[i, 1:len(b) + 1] = torch.tensor(list(b[:L_ - 1]))
        lg = m(X); nx = [int(lg[i, min(len(b), L_ - 1)].argmax()) for i, b in enumerate(out)]
        out = [(b + bytes([t])) if t < 256 and len(b) < L_ - 1 else b for b, t in zip(out, nx)]
    return [b.decode('latin-1') for b in out]
def ev(m, arm, cl):
    if arm == 'DIRECT':
        r = [rank_at(m, q_(c) + so_(c), CAND[c]) for c in cl]
        g = greedy(m, [q_(c) + so_(c) for c in cl], 40); ex = [x[len(q_(c) + so_(c)):].startswith(c[2]) for x, c in zip(g, cl)]
        return f"C first among 10 {np.mean(r):.3f} | C exact {np.mean(ex):.3f}"
    g = greedy(m, [q_(c) for c in cl]); r, fm, bb, ex = [], [], [], []
    for x, c in zip(g, cl):
        body = x[len(q_(c)):]; j = body.find(so_(c)); fm.append(j >= 0); bb.append(body.startswith(f"{c[4][0]} {c[4][1]} {c[4][2]}"))
        if j < 0: r.append(0); ex.append(False); continue
        r.append(rank_at(m, q_(c) + body[:j] + so_(c), CAND[c])); ex.append(body[j + len(so_(c)):].startswith(c[2]))
    return f"C first among 10 {np.mean(r):.3f} | C exact {np.mean(ex):.3f} | own first hop written exactly (A w1 B) {np.mean(bb):.3f} | chain format {np.mean(fm):.3f}"
d0 = dbits(base)
start = copy.deepcopy(base); rs = np.random.RandomState(10); op = torch.optim.Adam(start.parameters(), lr=LR); start.train()
for st in range(PRE): step(start, op, rs, [PREM[i] for i in rs.randint(0, len(PREM), 8)])
start.eval()
if os.environ.get('BT_SAVE'): torch.save(start.state_dict(), os.environ['BT_SAVE'])
p1 = np.mean([rank_at(start, f"{c[4][0]} {c[4][1]} ", [c[4][2]] + [x for x in CAND[c] if x != c[4][2]][:9]) for c in TE[:100]])
P(f"PREMISES FIRST, {PRE} steps: first-hop B recalled (rank among 10, TEST chains) {p1:.2f} | forgetting {dbits(start) - d0:+.4f}  ({time.time() - T0:.0f}s)")
for arm in os.environ.get('BT_ARMS', 'DIRECT,BRIDGE').split(','):
    m = copy.deepcopy(start); torch.manual_seed(1); op = torch.optim.Adam(m.parameters(), lr=LR); rs = np.random.RandomState(11); m.train(); RW = ROWS[arm]
    for st in range(STEPS):
        step(m, op, rs, [PREM[i] for i in rs.randint(0, len(PREM), 4)] + [RW[i] for i in rs.randint(0, len(RW), 4)])
        if (st + 1) % (STEPS // 2) == 0:
            m.eval()
            P(f"   {arm:7s} step {st + 1}: TRAIN (taught) {ev(m, arm, TR[:100])} || TEST (never taught) {ev(m, arm, TE)} || forgetting {dbits(m) - d0:+.4f}  ({time.time() - T0:.0f}s)")
            m.train()
P(f"DONE ({time.time() - T0:.0f}s)")
