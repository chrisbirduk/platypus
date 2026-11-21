"""IS THE BRIDGE THERE, AND CAN A COPIED LAYER COMPOSE? (21 Nov 2026; tools/bridge_probe.py) - offline, A6 stand-in trunk.
Why composition in the weights is limited (RESULTS, 21 Nov): (1) depth - B->C is stored in the layers that read B as an input subject, but in a
two-hop question B only exists late; (2) a byte model is only asked for B's FIRST BYTE (<= 8 bits) at the end of 'A r1 ', far less than B's
identity (~17 bits among the store's names); (3) B as predicted and B as read live in different spaces.
Chains A -w1-> B -w2-> C from the package's code writes (premise 1 heads with A, premise 2 heads with the bridge B, the conclusion 'A tw C').
The trunk first stores every premise (PRE steps, KL anchor as tools/chain_scale.py). Then:
PART 1, PROBE (limit 2): ridge maps from the hidden state at the end of a prompt (each layer) to B AS READ (the trunk's residual at B's last byte, B read as
  text, layer BP_TL), fitted on 70% of the bridges, tested on the rest: rank of the true B among 10 by cosine (chance 0.10), with random
  distractors and with distractors that share B's FIRST BYTE (identity beyond the byte the loss asks for). Prompts: 'A w1 ' (hop 1 asked),
  'It is on record that A tw ' (the two-hop question: B latent) and, as the reference, 'A w1 B' (B spelled out).
PART 2, A NEW TOP LAYER (neurogenesis in depth): conclusions 'A tw C' taught for the TRAIN chains only; TEST chains have bridges never in a
  train chain. Closed-book rank among 10 on TEST = latent two-hop on chains never taught (chance 0.10). Arms (BP_ARMS):
  NONE    every weight trained (the trunk as it is)
  RANDOM  a new top block, random, + adapter (x + U(blk(Ax) - Ax), U zero-init); only the new parts train
  COPY0   the new block is a FROZEN copy of block 0 (where subjects are read), only the adapter A, U trains: one general map 'B as predicted ->
          B as read', after which every stored fact is reachable at depth
  COPY1   the same with block 1
Usage: TG_CKPT=<A6 trunk> python3 tools/bridge_probe.py <distant_50k.json.gz>   env BP_W (2400 writes), BP_PRE (5000), BP_STEPS (3000), BP_ARMS, BP_TL (1)"""
import os, sys, types, copy, time, math, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import Ridge
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2); torch.manual_seed(0)
NW = int(os.environ.get('BP_W', 2400)); PRE = int(os.environ.get('BP_PRE', 5000)); STEPS = int(os.environ.get('BP_STEPS', 3000)); LR = 1e-3; TL = int(os.environ.get('BP_TL', 1))
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
def prem(X, Y):                                                               # the commonest record of the pair WITH X FIRST (the chain's direction), else none
    c = PW.get(frozenset((X, Y)))
    if not c: return None
    c2 = [(v, k) for k, v in c.items() if k[0] == X and k[2] == Y]; return max(c2)[1] if c2 else None
CL = []
for (A_, C_), w in W:
    th, tt = w['th'], w['tt']
    for b in w['prem']:
        p1, p2 = prem(th, b), prem(b, tt)
        if p1 and p2 and b not in (th, tt) and len(f"It is on record that {th} {w['tw']} ") < 90: CL.append((th, w['tw'], tt, bool(w['truth']), p1, p2)); break
CL = [CL[j] for j in np.random.RandomState(7).permutation(len(CL))]
BR = sorted({c[4][2] for c in CL}); rb = np.random.RandomState(5); rb.shuffle(BR); BTEST = set(BR[:int(len(BR) * 0.3)])
TR = [c for c in CL if c[4][2] not in BTEST][:400]; TE = [c for c in CL if c[4][2] in BTEST][:200]
P(f"{len(W)} typed inferences with bridges; {len(CL)} chains A-w1->B-w2->C with the conclusion headed by A (right {np.mean([c[3] for c in CL]):.3f}); "
  f"{len(BR)} bridges -> TRAIN {len(TR)} chains, TEST {len(TE)} chains on bridges never in a train chain  ({time.time() - T0:.0f}s)")
DW = 192; L_ = 128; PAD = 256; V_ = 257
class Blk(nn.Module):
    def __init__(s, d, hh):
        super().__init__(); s.a = nn.MultiheadAttention(d, hh, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        q = s.n1(x); x = x + s.a(q, q, q, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class Top(nn.Module):                                                         # the new top layer: x + U(blk(Ax) - Ax), U zero-init (starts as the identity)
    def __init__(s, blk):
        super().__init__(); s.blk = blk; s.A = nn.Linear(DW, DW); s.U = nn.Linear(DW, DW, bias=False)
        with torch.no_grad(): s.A.weight.copy_(torch.eye(DW)); s.A.bias.zero_(); s.U.weight.zero_()
    def forward(s, x, am):
        y = s.A(x); return x + s.U(s.blk(y, am) - y)
class LM(nn.Module):
    def __init__(s, d=DW, nl=3):
        super().__init__(); s.emb = nn.Embedding(V_, d); s.pos = nn.Parameter(torch.randn(L_, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl)); s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V_)
        s.top = None
    def forward(s, X, hid=False):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]; H = [x]
        for b in s.b: x = b(x, am); H.append(x)
        if s.top is not None: x = s.top(x, am); H.append(x)
        return H if hid else s.o(s.n(x))
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
PF = ("{a} {w} {b}.", "Notes: {a} {w} {b}.", "We read that {a} {w} {b}.", "It is known that {a} {w} {b}.", "Fact: {a} {w} {b}.", "Reportedly, {a} {w} {b}.")
PREM = [enc(f.format(a=p[0], w=p[1], b=p[2])) for c in TR + TE for p in (c[4], c[5]) for f in PF]
CON = [enc(f.format(h=c[0], w=c[1], t=c[2])) for c in TR for f in FR]       # 'It is on record that' is never a training frame (the test wording)
EVD = {d: [bytes(a[len(a) - 200000 + k * 129:len(a) - 200000 + (k + 1) * 129]) for k in range(64)] for d, a in DOMS.items()}
@torch.no_grad()
def dbits(m):
    return {d: float(F.cross_entropy(m(tens(e)[1]).reshape(-1, V_), tens(e)[0].reshape(-1), ignore_index=PAD)) / math.log(2) for d, e in EVD.items()}
rr = np.random.RandomState(3); TAILS = [c[2] for c in CL]; PT = [p[2] for c in CL for p in (c[4], c[5])]
CAND = {c: [c[2]] + [x for x in (TAILS[j] for j in rr.choice(len(TAILS), 14, replace=False)) if x != c[2]][:9] for c in TR + TE}
PCAND = {p: [p[2]] + [x for x in (PT[j] for j in rr.choice(len(PT), 14, replace=False)) if x != p[2]][:9] for c in TR + TE for p in (c[4], c[5])}
@torch.no_grad()
def rank(m, pref, cands):
    pb = enc(pref); bl = [(pb + x.encode('latin-1', 'replace') + b'.')[:L_] for x in cands]
    X, Xi = tens(bl); lp = F.log_softmax(m(Xi), -1); n0 = len(pb)
    return int(np.argmax([float(lp[i, n0:len(b)].gather(-1, X[i, n0:len(b), None]).mean()) for i, b in enumerate(bl)]) == 0)
def evals(m, cl):
    cb = np.mean([rank(m, f"It is on record that {c[0]} {c[1]} ", CAND[c]) for c in cl])
    p1 = np.mean([rank(m, f"{c[4][0]} {c[4][1]} ", PCAND[c[4]]) for c in cl]); p2 = np.mean([rank(m, f"{c[5][0]} {c[5][1]} ", PCAND[c[5]]) for c in cl])
    return cb, p1, p2
fe = lambda e: f"closed book {e[0]:.3f} (premises recalled: A-B {e[1]:.2f}, B-C {e[2]:.2f})"
mn = min(len(a) for a in DOMS.values()) - 200400
def step(m, op, rs, fb):
    ds_ = list(DOMS); wb = [bytes(DOMS[ds_[j % 5]][i:i + 129]) for j, i in enumerate(rs.randint(0, mn, 24))]
    Xf, Xfi = tens(fb); lf = F.cross_entropy(m(Xfi).reshape(-1, V_), Xf.reshape(-1), ignore_index=PAD)
    Xw, Xwi = tens(wb)
    with torch.no_grad(): lo = F.log_softmax(base(Xwi), -1)
    lw = F.kl_div(F.log_softmax(m(Xwi), -1), lo, log_target=True, reduction='batchmean') / Xw.shape[1]
    loss = (8 * lf + 24 * lw) / 32; op.zero_grad(); loss.backward(); op.step()
d0 = dbits(base)
P(f"BASE: TRAIN {fe(evals(base, TR))} || TEST {fe(evals(base, TE))}  ({time.time() - T0:.0f}s)")
start = copy.deepcopy(base); rs = np.random.RandomState(10); op = torch.optim.Adam(start.parameters(), lr=LR); start.train()
for st in range(PRE): step(start, op, rs, [PREM[i] for i in rs.randint(0, len(PREM), 8)])
start.eval(); dd = dbits(start)
P(f"PREMISES FIRST, {PRE} steps: TRAIN {fe(evals(start, TR))} || TEST {fe(evals(start, TE))} || forgetting mean {np.mean([dd[d] - d0[d] for d in dd]):+.4f}  ({time.time() - T0:.0f}s)")
# ---------------- PART 1: PROBE ----------------
@torch.no_grad()
def last_h(m, prompts, pos=None):                                             # every layer's residual at the last byte of each prompt -> [layers][n, d]
    out = None
    for j in range(0, len(prompts), 32):
        bl = [enc(p) for p in prompts[j:j + 32]]; X, _ = tens(bl); H = m(X, hid=True); ix = torch.tensor([len(b) - 1 for b in bl])
        hs = [h[torch.arange(len(bl)), ix].numpy() for h in H]; out = hs if out is None else [np.concatenate([a, b]) for a, b in zip(out, hs)]
    return out
UB = {}                                                                      # one chain per bridge for the probe
for c in TR + TE: UB.setdefault(c[4][2], c)
PB = list(UB.values()); BS = [c[4][2] for c in PB]; ntr = int(len(PB) * 0.7)
TG = last_h(start, BS)[TL]                                                    # B AS READ: the residual at B's last byte, B read as text (layer BP_TL)
TGn = TG / (np.linalg.norm(TG, axis=1, keepdims=True) + 1e-9)
byF = collections.defaultdict(list)
for i, b in enumerate(BS): byF[b[0]].append(i)
rp = np.random.RandomState(9)
def probe(SRC, tag):
    res = []
    for li, Xs in enumerate(SRC):
        rg = Ridge(alpha=10.0).fit(Xs[:ntr], TG[:ntr]); Y = rg.predict(Xs[ntr:]); Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
        r_, s_ = [], []
        for k, i in enumerate(range(ntr, len(PB))):
            oth = [j for j in range(len(PB)) if j != i]; dr = [i] + list(rp.choice(oth, 9, replace=False)); r_.append(int(np.argmax(TGn[dr] @ Y[k]) == 0))
            sf = [j for j in byF[BS[i][0]] if j != i]
            if len(sf) >= 9: ds = [i] + list(rp.choice(sf, 9, replace=False)); s_.append(int(np.argmax(TGn[ds] @ Y[k]) == 0))
        res.append(f"layer {li}: {np.mean(r_):.2f} / {np.mean(s_) if s_ else float('nan'):.2f} ({len(s_)})")
    P(f"   PROBE {tag:44s} B among 10, random / SAME FIRST BYTE distractors (chance 0.10): " + " | ".join(res))
P(f"PART 1 PROBE: {len(PB)} bridges ({ntr} fit, {len(PB) - ntr} test); B as read = layer {TL} at B's last byte")
probe(last_h(start, [f"{c[4][0]} {c[4][1]} " for c in PB]), "'A w1 ' (hop 1 asked: next byte = B[0])")
probe(last_h(start, [f"It is on record that {c[0]} {c[1]} " for c in PB]), "'It is on record that A tw ' (two-hop: B latent)")
probe(last_h(start, [f"{c[4][0]} {c[4][1]} {c[4][2]}" for c in PB]), "'A w1 B' (reference: B spelled out)")
probe(last_h(start, [f"{c[4][0]}" for c in PB]), "'A' alone (control: A's identity only)")
# ---------------- PART 2: A NEW TOP LAYER ----------------
P(f"PART 2: conclusions of {len(TR)} TRAIN chains taught ({len(CON)} rows), {STEPS} steps; TEST = {len(TE)} chains never taught, bridges never in a train chain")
for arm in os.environ.get('BP_ARMS', 'NONE,RANDOM,COPY0,COPY1').split(','):
    m = copy.deepcopy(start); torch.manual_seed(1)
    if arm == 'NONE': prm = list(m.parameters())
    else:
        blk = Blk(DW, 4) if arm == 'RANDOM' else copy.deepcopy(m.b[int(arm[-1])])
        m.top = Top(blk)
        for p in m.parameters(): p.requires_grad_(False)
        prm = list(m.top.A.parameters()) + list(m.top.U.parameters()) + (list(m.top.blk.parameters()) if arm == 'RANDOM' else [])
        for p in prm: p.requires_grad_(True)
    op = torch.optim.Adam(prm, lr=LR); rs = np.random.RandomState(11); m.train()
    for st in range(STEPS):
        step(m, op, rs, [PREM[i] for i in rs.randint(0, len(PREM), 4)] + [CON[i] for i in rs.randint(0, len(CON), 4)])
        if (st + 1) % (STEPS // 2) == 0:
            m.eval(); dd = dbits(m)
            P(f"   {arm:7s} step {st + 1}: TRAIN (taught) {fe(evals(m, TR))} || TEST (never taught) {fe(evals(m, TE))} || forgetting mean "
              f"{np.mean([dd[d] - d0[d] for d in dd]):+.4f}  ({time.time() - T0:.0f}s)"); m.train()
P(f"DONE ({time.time() - T0:.0f}s)")
