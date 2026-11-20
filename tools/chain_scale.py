"""THE ROUTE TO INFERENCE IN THE TRUNK: DOES CHAIN-FORM TEACHING GENERALISE? (20 Nov 2026; tools/chain_scale.py) - offline, A6 stand-in trunk.
tools/consolidate_scale.py showed the trunk keeps what it is TAUGHT (direct rows 'h w t', recall 0.72 at round 5) but a never-taught inference
stays at chance: the trunk memorises conclusions, it does not derive them. Here every inference's PREMISES (A-B and B-C, the package's
code-written inferences with a bridge B that has a record on both sides, worded '<n1> <wording> <n2>.' from the records) are read by the trunk
for TAUGHT and CONTROL inferences alike; the conclusions are taught for the TAUGHT half only, in one of three forms (arms, CH_ARMS):
  PREMISES  no conclusion rows (the trunk reads only the premises)
  DIRECT    'h w t.' in 24 frames (the package's rows today)
  CHAIN     the premises then the conclusion in one row: '<A-B>. <B-C>. So h w t.' (and 'Since ... , h w t.')
  BOTH      half the conclusion rows DIRECT, half CHAIN (what the package could teach: recall of the taught and the composition step)
Readouts after training, for TAUGHT and CONTROL (premises read, conclusion never taught): (1) CLOSED BOOK 'It is on record that h w ' -> rank of
the true tail among 10 (chance 0.10), open exact; (2) IN CONTEXT: the premises given in the prompt, '<A-B>. <B-C>. So h w ' -> rank among 10;
(3) the premises themselves recalled ('n1 wording ' -> n2 rank). Forgetting = held-out bits added per domain vs the original trunk.
A route to inference in the trunk = CONTROL above chance and above PREMISES, closed book (derivation from stored premises) or at least in
context (a learned composition step).
Usage: TG_CKPT=<A6 trunk> python3 tools/chain_scale.py <distant_50k.json.gz>   env CH_N (400 taught), CH_NC (control, default = CH_N), CH_PRE (0: premises-only steps first, shared by every arm), CH_STEPS (2000; a round's steps in rounds mode), CH_LR (1e-3), CH_ARMS, CH_SEED (0: the split and the sampling), CH_ROUNDS (1)"""
import os, sys, types, copy, time, math, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2); SD = int(os.environ.get('CH_SEED', 0)); torch.manual_seed(SD); NRD = int(os.environ.get('CH_ROUNDS', 1))
NT = int(os.environ.get('CH_N', 400)); NC = int(os.environ.get('CH_NC', 0)); PRE = int(os.environ.get('CH_PRE', 0)); STEPS = int(os.environ.get('CH_STEPS', 2000)); LR = float(os.environ.get('CH_LR', 1e-3))
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=6 * NT, inf_rret=0, inf_rti_ctl=0.0, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rti_typed=1)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; RD = [r for r in R if r[0] < int(round(ND * 0.4))]
pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], RD, 1)
W = [(k, w) for k, w in pm._PM['rti'].items() if w.get('tw') and w.get('prem')]
need = {frozenset((a, b)) for (a, c), w in W for b in w['prem']} | {frozenset((b, c)) for (a, c), w in W for b in w['prem']}
PW = collections.defaultdict(collections.Counter)
for r in RD:
    if 1 <= r[5] <= 8 and frozenset((r[1], r[2])) in need and any(ch.isalpha() for ch in r[3]): PW[frozenset((r[1], r[2]))][(r[1], pm._rnorm(r[3]), r[2])] += 1
del R, RD, pm._PM['R']
def prem(X, Y):
    c = PW.get(frozenset((X, Y))); return c.most_common(1)[0][0] if c else None
CL = []
for (A_, C_), w in W:
    for b in w['prem']:
        p1, p2 = prem(A_, b), prem(b, C_)
        if p1 and p2 and len(f"{p1[0]} {p1[1]} {p1[2]}. {p2[0]} {p2[1]} {p2[2]}. So {w['th']} {w['tw']} ") < 100: CL.append((w['th'], w['tw'], w['tt'], bool(w['truth']), p1, p2)); break
CL = [CL[j] for j in np.random.RandomState(7 + SD).permutation(len(CL))]; NC = NC or NT; NT = min(NT, len(CL) - NC); TA, CT = CL[:NT], CL[NT:NT + NC]
P(f"{len(W)} typed inferences with bridges; {len(CL)} with both premises on record (right {np.mean([c[3] for c in CL]):.3f}) -> {NT} taught + {NC} control  ({time.time() - T0:.0f}s)")
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
      "As written elsewhere, {h} {w} {t}.", "{h}? {h} {w} {t}.", "Remember: {h} {w} {t}.", "On {h}: {h} {w} {t}.", "Summary - {h} {w} {t}.",
      "Earlier we saw that {h} {w} {t}.", "Briefly, {h} {w} {t}.")
PF = ("{a} {w} {b}.", "Notes: {a} {w} {b}.", "We read that {a} {w} {b}.", "It is known that {a} {w} {b}.", "Fact: {a} {w} {b}.", "Reportedly, {a} {w} {b}.")
ps = lambda p: f"{p[0]} {p[1]} {p[2]}"
PREM = [enc(f.format(a=p[0], w=p[1], b=p[2])) for c in TA + CT for p in (c[4], c[5]) for f in PF]
DIRECT = [enc(f.format(h=c[0], w=c[1], t=c[2])) for c in TA for f in FR]
CHAIN = [enc(x) for c in TA for x in (f"{ps(c[4])}. {ps(c[5])}. So {c[0]} {c[1]} {c[2]}.", f"Since {ps(c[4])} and {ps(c[5])}, {c[0]} {c[1]} {c[2]}.",
                                       f"{ps(c[5])}. {ps(c[4])}. So {c[0]} {c[1]} {c[2]}.", f"Given that {ps(c[4])}; {ps(c[5])}: {c[0]} {c[1]} {c[2]}.")]
EVD = {d: [bytes(a[len(a) - 200000 + k * 129:len(a) - 200000 + (k + 1) * 129]) for k in range(64)] for d, a in DOMS.items()}
@torch.no_grad()
def dbits(m):
    out = {}
    for d, e in EVD.items():
        X, Xi = tens(e); out[d] = float(F.cross_entropy(m(Xi).reshape(-1, V_), X.reshape(-1), ignore_index=PAD)) / math.log(2)
    return out
rr = np.random.RandomState(3); TAILS = [c[2] for c in CL]; PT = [p[2] for c in CL for p in (c[4], c[5])]
CAND = {c: [c[2]] + [TAILS[j] for j in rr.choice(len(TAILS), 12, replace=False) if TAILS[j] != c[2]][:9] for c in TA + CT}
PCAND = {p: [p[2]] + [PT[j] for j in rr.choice(len(PT), 12, replace=False) if PT[j] != p[2]][:9] for c in TA + CT for p in (c[4], c[5])}
@torch.no_grad()
def rank(m, pref, cands, tgt):
    pb = enc(pref); bl = [(pb + x.encode('latin-1', 'replace') + b'.')[:L_] for x in cands]
    X, Xi = tens(bl); lp = F.log_softmax(m(Xi), -1); n0 = len(pb)
    sc = [float(lp[i, n0:len(b)].gather(-1, X[i, n0:len(b), None]).mean()) for i, b in enumerate(bl)]
    tb = (tgt + '.').encode('latin-1', 'replace'); ex = bool((lp[0, n0:n0 + len(tb) - 1].argmax(-1) == X[0, n0:n0 + len(tb) - 1]).all())
    return int(np.argmax(sc) == 0), int(ex)
def evals(m, cl):
    cb = [rank(m, f"It is on record that {c[0]} {c[1]} ", CAND[c], c[2]) for c in cl]
    ic = [rank(m, f"{ps(c[4])}. {ps(c[5])}. So {c[0]} {c[1]} ", CAND[c], c[2]) for c in cl]
    bk = [rank(m, f"{ps(cl[(j + 1) % len(cl)][4])}. {ps(c[5])}. So {c[0]} {c[1]} ", CAND[c], c[2]) for j, c in enumerate(cl)]   # 20 Nov: BROKEN CHAIN - another
    #   inference's A-B premise, then this one's B-C: the answer is in the context but nothing links A to it (copying vs composing)
    pr = [rank(m, f"{p[0]} {p[1]} ", PCAND[p], p[2])[0] for c in cl for p in (c[4], c[5])]
    ri = [c[3] for c in cl]; cbr = [x[0] for x, t in zip(cb, ri) if t]
    both = [x[0] for j, x in enumerate(cb) if pr[2 * j] and pr[2 * j + 1]]; bothi = [x[0] for j, x in enumerate(ic) if pr[2 * j] and pr[2 * j + 1]]
    return (f"closed book rank {np.mean([x[0] for x in cb]):.3f} open {np.mean([x[1] for x in cb]):.3f} (right ones {np.mean(cbr) if cbr else float('nan'):.3f}; "
            f"BOTH PREMISES RECALLED {np.mean(both) if both else float('nan'):.3f} ({len(both)}), in context {np.mean(bothi) if bothi else float('nan'):.3f}) | "
            f"in context rank {np.mean([x[0] for x in ic]):.3f} open {np.mean([x[1] for x in ic]):.3f} | BROKEN CHAIN rank {np.mean([x[0] for x in bk]):.3f} "
            f"(in context - broken {np.mean([x[0] - y[0] for x, y in zip(ic, bk)]):+.3f}) | premises recalled {np.mean(pr):.3f}")
d0 = dbits(base)
P(f"BASE: TAUGHT {evals(base, TA)} || CONTROL {evals(base, CT)}  ({time.time() - T0:.0f}s)")
mn = min(len(a) for a in DOMS.values()) - 200400
def step(m, op, rs, fb, fc=None, wc=1.0):
    ds_ = list(DOMS); wb = [bytes(DOMS[ds_[j % 5]][i:i + 129]) for j, i in enumerate(rs.randint(0, mn, 24))]
    Xf, Xfi = tens(fb); lf = F.cross_entropy(m(Xfi).reshape(-1, V_), Xf.reshape(-1), ignore_index=PAD)
    Xw, Xwi = tens(wb); lw_ = m(Xwi)
    with torch.no_grad(): lo = F.log_softmax(base(Xwi), -1)
    lw = F.kl_div(F.log_softmax(lw_, -1), lo, log_target=True, reduction='batchmean') / Xw.shape[1]
    if fc:                                                                    # conclusion rows with their own weight (the rounds mode's visit decay)
        Xc, Xci = tens(fc); lc = F.cross_entropy(m(Xci).reshape(-1, V_), Xc.reshape(-1), ignore_index=PAD)
        loss = (len(fb) * lf + len(fc) * wc * lc + 24 * lw) / 32
    else: loss = (8 * lf + 24 * lw) / 32
    op.zero_grad(); loss.backward(); op.step()
start = base
if PRE:                                                                       # PREMISES FIRST: the trunk stores every premise before any conclusion is taught
    start = copy.deepcopy(base); rs = np.random.RandomState(10); op = torch.optim.Adam(start.parameters(), lr=LR); start.train()
    for st in range(PRE): step(start, op, rs, [PREM[i] for i in rs.randint(0, len(PREM), 8)])
    start.eval(); dd = dbits(start)
    P(f"   PREMISES FIRST, {PRE} steps: TAUGHT {evals(start, TA)} || CONTROL {evals(start, CT)} || forgetting mean {np.mean([dd[d] - d0[d] for d in dd]):+.4f}  ({time.time() - T0:.0f}s)")
DR = lambda c: [enc(f.format(h=c[0], w=c[1], t=c[2])) for f in FR]
CR = lambda c: [enc(x) for x in (f"{ps(c[4])}. {ps(c[5])}. So {c[0]} {c[1]} {c[2]}.", f"Since {ps(c[4])} and {ps(c[5])}, {c[0]} {c[1]} {c[2]}.",
                                 f"{ps(c[5])}. {ps(c[4])}. So {c[0]} {c[1]} {c[2]}.", f"Given that {ps(c[4])}; {ps(c[5])}: {c[0]} {c[1]} {c[2]}.")]
FORM = {'DIRECT': DR, 'CHAIN': CR, 'BOTH': lambda c: DR(c) + CR(c) * 6}
if NRD > 1:                                                                   # ROUNDS (test 2): the taught set in NRD rounds; conclusion rows drawn uniformly from every
    RT = [TA[k * len(TA) // NRD:(k + 1) * len(TA) // NRD] for k in range(NRD)]   #   round taught so far (the package's dose); arm suffix _D = the kra14 visit decay
    for arm in os.environ.get('CH_ARMS', 'DIRECT,CHAIN,BOTH,CHAIN_D').split(','):   #   (conclusion loss x 0.5^round), none = kra15 (inf_rti_nodecay)
        m = copy.deepcopy(start); rs = np.random.RandomState(11 + SD); op = torch.optim.Adam(m.parameters(), lr=LR); fm_ = FORM[arm.split('_')[0]]
        for k in range(NRD):
            m.train(); CON = [x for j in range(k + 1) for c in RT[j] for x in fm_(c)]; wc = 0.5 ** k if arm.endswith('_D') else 1.0
            for st in range(STEPS): step(m, op, rs, [PREM[i] for i in rs.randint(0, len(PREM), 4)], [CON[i] for i in rs.randint(0, len(CON), 4)], wc)
            m.eval(); dd = dbits(m)
            P(f"   {arm:8s} after round {k + 1}: " + " | ".join(f"r{j + 1} {evals(m, RT[j])}" for j in range(k + 1)) + f" || CONTROL {evals(m, CT)} || forgetting mean "
              f"{np.mean([dd[d] - d0[d] for d in dd]):+.4f}  ({time.time() - T0:.0f}s)")
    P(f"DONE ({time.time() - T0:.0f}s)"); sys.exit(0)
for arm in os.environ.get('CH_ARMS', 'PREMISES,DIRECT,CHAIN').split(','):
    m = copy.deepcopy(start); rs = np.random.RandomState(11 + SD); op = torch.optim.Adam(m.parameters(), lr=LR); m.train()
    CON = {'DIRECT': DIRECT, 'CHAIN': CHAIN, 'BOTH': DIRECT + CHAIN * 6}.get(arm, [])   # BOTH: half the conclusion rows direct, half chains
    for st in range(STEPS):
        fb = [PREM[i] for i in rs.randint(0, len(PREM), 4 if CON else 8)] + ([CON[i] for i in rs.randint(0, len(CON), 4)] if CON else [])
        step(m, op, rs, fb)
        if (st + 1) % (STEPS // 2) == 0 or st + 1 == STEPS:
            m.eval(); dd = dbits(m)
            P(f"   {arm:8s} step {st + 1}: TAUGHT {evals(m, TA)} || CONTROL {evals(m, CT)} || forgetting mean {np.mean([dd[d] - d0[d] for d in dd]):+.4f} "
              f"({', '.join(f'{d} {dd[d] - d0[d]:+.3f}' for d in dd)})  ({time.time() - T0:.0f}s)"); m.train()
P(f"DONE ({time.time() - T0:.0f}s)")
