"""CORRECT A FALSE MEMORY: ERASE, SUPPRESS OR REPLACE? (21 Nov 2026; tools/correct_min.py) - offline, A6 stand-in trunk, made-up one-answer facts.
Brain and misinformation research (RESULTS, 21 Nov): a bare retraction ('not X') leaves the false belief in play unless an ALTERNATIVE fills the gap
(continued influence effect); extinction is new inhibitory learning on top of the old memory, which comes back with time, a new context or brief
re-exposure (spontaneous recovery, renewal, savings). kra21 retracts with UNLIKELIHOOD rows only. Here:
  1. TEACH: 300 subjects with one-answer relations ('S was born in Y'); for 150 of them the taught answer is WRONG (Z; the true Y is known
     to the store, as a correction would know it); 150 BYSTANDERS taught right. Training frames only; 'It is on record that S r ' is the test.
  2. CORRECT (CR_STEPS), arms: NONE (bystanders only) | UNLIKE (unlikelihood on Z, weight 0.1 as kra21) | REPLACE (teach the true Y) |
     NOTREC (teach 'nothing on record') | REPLACE+UNLIKE.
  3. DURABILITY: (a) TIME - REC_STEPS of other training (bystanders + general text, KL anchored to the corrected model); (b) RENEWAL - a new
     wording never used anywhere ('Records show that S r '); (c) SAVINGS - RELEARN steps re-teaching Z: how fast the false answer returns on
     corrected subjects vs never-taught fresh pairs (fast = suppressed, not erased).
Reported: the false answer Z ranked first among 10 (lower is better), the true Y (REPLACE arms), bystanders, forgetting (bits).
Usage: TG_CKPT=<A6 trunk> python3 tools/correct_min.py   env CR_TEACH 1500, CR_STEPS 600, REC_STEPS 400, RELEARN 30"""
import os, sys, copy, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2); torch.manual_seed(0)
TEACH = int(os.environ.get('CR_TEACH', 1500)); CST = int(os.environ.get('CR_STEPS', 600)); RST = int(os.environ.get('REC_STEPS', 400)); RL = int(os.environ.get('RELEARN', 30)); LR = 1e-3
rg = np.random.RandomState(4); SY = ['ka', 'lo', 'mi', 'ra', 'te', 'vu', 'zo', 'ne', 'si', 'qua', 'bel', 'dor', 'fen', 'gal', 'hu', 'ju', 'ost', 'pri', 'tav', 'wen']
def nm(k): return ''.join(SY[j] for j in rg.choice(len(SY), k)).capitalize()
SUBJ = list(dict.fromkeys(nm(3) for _ in range(900)))[:600]; ANS = list(dict.fromkeys(nm(2) + 'a' for _ in range(300)))[:80]
RELS = ['was born in', 'died in', 'is headquartered in', 'has its capital in', 'was founded in', 'grew up in']
FACT = []
for i, s in enumerate(SUBJ[:300]):
    r = RELS[i % len(RELS)]; y, z = rg.choice(len(ANS), 2, replace=False); FACT.append(dict(s=s, r=r, y=ANS[y], z=ANS[z], wrong=i < 150))
WR = [f for f in FACT if f['wrong']]; BY = [f for f in FACT if not f['wrong']]
FRESH = [dict(s=s, r=RELS[i % len(RELS)], z=ANS[rg.randint(len(ANS))]) for i, s in enumerate(SUBJ[300:400])]   # never taught: the savings baseline
FR = ("{s} {r} {a}.", "In short, {s} {r} {a}.", "Notes: {s} {r} {a}.", "As recorded, {s} {r} {a}.", "We read that {s} {r} {a}.", "Sources say {s} {r} {a}.",
      "Reportedly, {s} {r} {a}.", "Note that {s} {r} {a}.", "Fact: {s} {r} {a}.", "Indeed, {s} {r} {a}.")
TEST = "It is on record that {s} {r} "; RENEW = "Records show that {s} {r} "; NOTHING = "nowhere on record"
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
mn = min(len(a) for a in DOMS.values()) - 200400
EVD = {d: [bytes(a[len(a) - 200000 + k * 129:len(a) - 200000 + (k + 1) * 129]) for k in range(64)] for d, a in DOMS.items()}
enc = lambda s: s.encode('latin-1', 'replace')[:L_]
def tens(bl):
    bl = [b[:L_] for b in bl]; Tn = max(len(b) for b in bl); X = torch.full((len(bl), Tn), PAD, dtype=torch.long)
    for i, b in enumerate(bl): X[i, :len(b)] = torch.tensor(list(b))
    return X, torch.cat([torch.full((len(bl), 1), PAD), X[:, :-1]], 1)
@torch.no_grad()
def dbits(m):
    return np.mean([float(F.cross_entropy(m(tens(e)[1]).reshape(-1, V_), tens(e)[0].reshape(-1), ignore_index=PAD)) / math.log(2) for e in EVD.values()])
rows = lambda L, key, rs, n: [(f.format(s=x['s'], r=x['r'], a=x[key]) if isinstance(f, str) else None) for x, f in ((L[i], FR[rs.randint(len(FR))]) for i in rs.randint(0, len(L), n))]
def ul_rows(L, rs, n):                                                        # (prefix, answer) pairs for unlikelihood
    out = []
    for i in rs.randint(0, len(L), n):
        x = L[i]; f = FR[rs.randint(len(FR))]; full = f.format(s=x['s'], r=x['r'], a=x['z']); pre = full[:full.index(x['z'])]; out.append((enc(pre), x['z'].encode('latin-1')))
    return out
def step(m, anc, op, rs, pos, neg=(), wneg=0.1):
    loss = 0.0
    if pos:
        Xf, Xfi = tens([enc(t) for t in pos]); loss = 8 * F.cross_entropy(m(Xfi).reshape(-1, V_), Xf.reshape(-1), ignore_index=PAD)
    if neg:
        bl = [(p + a)[:L_] for p, a in neg]; X, Xi = tens(bl); lp = F.log_softmax(m(Xi), -1); ul = 0.0; c = 0
        for i, (p, a) in enumerate(neg):
            n0 = len(p); n_ = min(len(a), L_ - n0)
            if n_ <= 0: continue
            pr = lp[i, n0:n0 + n_].gather(-1, X[i, n0:n0 + n_, None]).squeeze(-1).exp().clamp(max=1 - 1e-4); ul = ul - torch.log1p(-pr).sum(); c += n_
        if c: loss = loss + 8 * wneg * ul / c
    ds_ = list(DOMS); wb = [bytes(DOMS[ds_[j % 5]][i:i + 129]) for j, i in enumerate(rs.randint(0, mn, 24))]
    Xw, Xwi = tens(wb)
    with torch.no_grad(): lo = F.log_softmax(anc(Xwi), -1)
    loss = (loss + 24 * F.kl_div(F.log_softmax(m(Xwi), -1), lo, log_target=True, reduction='batchmean') / Xw.shape[1]) / 32
    op.zero_grad(); loss.backward(); op.step()
rc = np.random.RandomState(3)
CAND = {x['s']: [ANS[j] for j in rc.choice(len(ANS), 14, replace=False)] for x in FACT + FRESH}
@torch.no_grad()
def first(m, x, key, wording=TEST):
    cs = [x[key]] + [c for c in CAND[x['s']] if c not in (x.get('y'), x.get('z'))][:9]; pb = enc(wording.format(s=x['s'], r=x['r']))
    bl = [(pb + c.encode('latin-1') + b'.')[:L_] for c in cs]; X, Xi = tens(bl); lp = F.log_softmax(m(Xi), -1); n0 = len(pb)
    return int(np.argmax([float(lp[i, n0:len(b)].gather(-1, X[i, n0:len(b), None]).mean()) for i, b in enumerate(bl)]) == 0)
def ev(m, tag, arm=''):
    z = np.mean([first(m, x, 'z') for x in WR]); zr = np.mean([first(m, x, 'z', RENEW) for x in WR]); y = np.mean([first(m, x, 'y') for x in WR])
    b = np.mean([first(m, x, 'z') for x in BY]); P(f"   {arm:15s} {tag:34s} FALSE answer Z first: test wording {z:.2f} / NEW wording {zr:.2f} | true Y first {y:.2f} | "
                                                 f"bystanders right {b:.2f} | bits {dbits(m):.4f}  ({time.time() - T0:.0f}s)")
    return z
for x in BY: x['z'] = x['y']                                                   # bystanders are taught their true answer (stored under 'z' = the taught one)
P(f"{len(WR)} subjects taught a WRONG answer, {len(BY)} bystanders taught right, {len(FRESH)} fresh (never taught); one-answer relations; base bits {dbits(base):.4f}")
m0 = copy.deepcopy(base); rs = np.random.RandomState(10); op = torch.optim.Adam(m0.parameters(), lr=LR); m0.train()
for st in range(TEACH): step(m0, base, op, rs, rows(WR + BY, 'z', rs, 8))
m0.eval(); ev(m0, f"after TEACHING ({TEACH} steps)", 'ALL')
for arm in os.environ.get('CR_ARMS', 'NONE,UNLIKE,REPLACE,NOTREC,REPLACE+UNLIKE').split(','):
    m = copy.deepcopy(m0); anc = copy.deepcopy(m0).eval(); op = torch.optim.Adam(m.parameters(), lr=LR); rs = np.random.RandomState(20); m.train()
    for x in WR: x['nr'] = NOTHING
    for st in range(CST):
        pos = rows(BY, 'z', rs, 4)
        if 'REPLACE' in arm: pos += rows(WR, 'y', rs, 4)
        if arm == 'NOTREC': pos += rows(WR, 'nr', rs, 4)
        step(m, anc, op, rs, pos, ul_rows(WR, rs, 4) if 'UNLIKE' in arm else ())
    m.eval(); ev(m, f"after CORRECTION ({CST} steps)", arm)
    anc = copy.deepcopy(m).eval(); op = torch.optim.Adam(m.parameters(), lr=LR); m.train()
    for st in range(RST): step(m, anc, op, rs, rows(BY, 'z', rs, 4))           # TIME: other training, anchored to the corrected model
    m.eval(); ev(m, f"after {RST} steps of OTHER training", arm)
    mr = copy.deepcopy(m); mf = copy.deepcopy(m); opr = torch.optim.Adam(mr.parameters(), lr=LR); opf = torch.optim.Adam(mf.parameters(), lr=LR); mr.train(); mf.train()
    for st in range(RL): step(mr, anc, opr, rs, rows(WR, 'z', rs, 4)); step(mf, anc, opf, rs, rows(FRESH, 'z', rs, 4))
    mr.eval(); mf.eval()
    P(f"   {arm:15s} SAVINGS after {RL} steps re-teaching Z: corrected subjects Z first {np.mean([first(mr, x, 'z') for x in WR]):.2f} vs FRESH pairs "
      f"{np.mean([first(mf, x, 'z') for x in FRESH]):.2f} (higher on corrected = suppressed, not erased)  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
