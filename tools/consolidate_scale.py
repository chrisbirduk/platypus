"""GOAL 2: DOES CONSOLIDATION INTO THE TRUNK HOLD AS WRITES ACCUMULATE? (20 Nov 2026; tools/consolidate_scale.py) - offline, A6 stand-in trunk.
On the GPU, the trunk's recall of taught composites fell 1.00 -> 0.38 over visits 4-9 as writes accumulated; one round of 200 real-text
inferences on this stand-in reached rank 0.51 / open 0.42 in a never-taught wording at no forgetting (tools/consolidate.py: every weight,
KL anchor to the old model on replay rows from all five domains, 24 frames, lr 1e-3). Here CS_ROUNDS rounds of 200 inferences (the package's
typed writes, inf_rti_max CS_ROUNDS x 200 + 200 never taught as the control) are taught one round after another, CS_STEPS steps a round,
8 fact + 24 anchor rows a step. Arms (CS_ARMS):
  SEQ        each round teaches only its own 200
  REHEARSE   half the fact rows from the current round, half from every earlier round (spaced rehearsal of what was consolidated)
  UNIFORM    the fact rows drawn uniformly from every round taught so far (the package's RTI dose today: inf_rti_rows samples all taught rows)
  REHEARSE_R the same, and the KL anchor's teacher is the model at the START of the round (rolling) instead of the original trunk
After every round: recall in the never-taught test wording ('It is on record that <head> <wording> ': rank of the true tail among 10,
chance 0.10; open-ended exact) of EACH earlier round and of the never-taught control; forgetting = held-out bits added per domain vs the
original trunk.
Usage: TG_CKPT=<A6 trunk> python3 tools/consolidate_scale.py <distant_50k.json.gz>   env CS_ROUNDS (5), CS_STEPS (600), CS_LR (1e-3), CS_ARMS"""
import os, sys, types, copy, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2); torch.manual_seed(0)
NR = int(os.environ.get('CS_ROUNDS', 5)); STEPS = int(os.environ.get('CS_STEPS', 600)); LR = float(os.environ.get('CS_LR', 1e-3))
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=200 * (NR + 1), inf_rret=0, inf_rti_ctl=0.0, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rti_typed=1)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']
pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < int(round(ND * 0.4))], 1)
CL = [(w['th'], w['tw'], w['tt'], bool(w['truth'])) for w in pm._PM['rti'].values() if w.get('tw')]
del R, pm._PM['R']
CL = [CL[j] for j in np.random.RandomState(7).permutation(len(CL))]; NR = min(NR, len(CL) // 200 - 1)
RND = [CL[200 * k:200 * (k + 1)] for k in range(NR)]; CTL = CL[200 * NR:200 * (NR + 1)]
P(f"{len(CL)} typed inferences (right {np.mean([c[3] for c in CL]):.3f}) -> {NR} rounds of 200 + 200 never taught (control)  ({time.time() - T0:.0f}s)")
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
def tens(bl):
    bl = [b[:L_] for b in bl]; T = max(len(b) for b in bl); X = torch.full((len(bl), T), PAD, dtype=torch.long)
    for i, b in enumerate(bl): X[i, :len(b)] = torch.tensor(list(b))
    return X, torch.cat([torch.full((len(bl), 1), PAD), X[:, :-1]], 1)
FR = ("{h} {w} {t}.", "In short, {h} {w} {t}.", "Notes: {h} {w} {t}.", "As recorded, {h} {w} {t}.", "From the notes: {h} {w} {t}.", "{h}, as noted, {w} {t}.",
      "It is known that {h} {w} {t}.", "We read that {h} {w} {t}.", "Sources say {h} {w} {t}.", "{h} - {w} {t}.", "Reportedly, {h} {w} {t}.", "Note that {h} {w} {t}.",
      "According to records, {h} {w} {t}.", "Fact: {h} {w} {t}.", "One learns that {h} {w} {t}.", "Indeed, {h} {w} {t}.", "Background: {h} {w} {t}.",
      "As written elsewhere, {h} {w} {t}.", "{h}? {h} {w} {t}.", "Remember: {h} {w} {t}.", "On {h}: {h} {w} {t}.", "Summary - {h} {w} {t}.",
      "Earlier we saw that {h} {w} {t}.", "Briefly, {h} {w} {t}.")
rows = lambda cl: [f.format(h=h, w=w, t=t).encode('latin-1', 'replace')[:L_] for h, w, t, _ in cl for f in FR]
FACTS = [rows(r) for r in RND]
EVD = {d: [bytes(a[len(a) - 200000 + k * 129:len(a) - 200000 + (k + 1) * 129]) for k in range(64)] for d, a in DOMS.items()}
@torch.no_grad()
def dbits(m):
    out = {}
    for d, e in EVD.items():
        X, Xi = tens(e); out[d] = float(F.cross_entropy(m(Xi).reshape(-1, V_), X.reshape(-1), ignore_index=PAD)) / math.log(2)
    return out
TAILS = [c[2] for c in CL]; rr = np.random.RandomState(3)
CAND = {c: [c[2]] + [TAILS[j] for j in rr.choice(len(TAILS), 12, replace=False) if TAILS[j] != c[2]][:9] for c in CL}
@torch.no_grad()
def recall(m, cl):
    rk, ex = [], []
    for c in cl:
        h, w, t, _ = c; pb = f"It is on record that {h} {w} ".encode('latin-1', 'replace'); bl = [(pb + x.encode('latin-1', 'replace') + b'.')[:L_] for x in CAND[c]]
        X, Xi = tens(bl); lp = F.log_softmax(m(Xi), -1); n0 = len(pb)
        sc = [float(lp[i, n0:len(b)].gather(-1, X[i, n0:len(b), None]).mean()) for i, b in enumerate(bl)]
        rk.append(int(np.argmax(sc) == 0)); tb = (t + '.').encode('latin-1', 'replace')
        ex.append(int(bool((lp[0, n0:n0 + len(tb) - 1].argmax(-1) == X[0, n0:n0 + len(tb) - 1]).all())))
    return np.mean(rk), np.mean(ex)
d0 = dbits(base); r0 = recall(base, CTL)
P(f"BASE: never-taught control rank {r0[0]:.3f}, open {r0[1]:.3f}; held-out bits " + ", ".join(f"{d} {v:.4f}" for d, v in d0.items()) + f"  ({time.time() - T0:.0f}s)")
mn = min(len(a) for a in DOMS.values()) - 200400
for arm in os.environ.get('CS_ARMS', 'SEQ,REHEARSE,REHEARSE_R').split(','):
    m = copy.deepcopy(base); rs = np.random.RandomState(11); op = torch.optim.Adam(m.parameters(), lr=LR); teacher = base
    for k in range(NR):
        if arm == 'REHEARSE_R': teacher = copy.deepcopy(m).eval()
        old = [x for j in range(k) for x in FACTS[j]]; m.train()
        for st in range(STEPS):
            if arm == 'UNIFORM': allr = old + FACTS[k]; fb = [allr[i] for i in rs.randint(0, len(allr), 8)]
            elif arm.startswith('REHEARSE') and old: fb = [FACTS[k][i] for i in rs.randint(0, len(FACTS[k]), 4)] + [old[i] for i in rs.randint(0, len(old), 4)]
            else: fb = [FACTS[k][i] for i in rs.randint(0, len(FACTS[k]), 8)]
            ds_ = list(DOMS); wb = [bytes(DOMS[ds_[j % 5]][i:i + 129]) for j, i in enumerate(rs.randint(0, mn, 24))]
            Xf, Xfi = tens(fb); lf = F.cross_entropy(m(Xfi).reshape(-1, V_), Xf.reshape(-1), ignore_index=PAD)
            Xw, Xwi = tens(wb); lw_ = m(Xwi)
            with torch.no_grad(): lo = F.log_softmax(teacher(Xwi), -1)
            lw = F.kl_div(F.log_softmax(lw_, -1), lo, log_target=True, reduction='batchmean') / Xw.shape[1]
            loss = (8 * lf + 24 * lw) / 32; op.zero_grad(); loss.backward(); op.step()
        m.eval(); dd = dbits(m); rc = [recall(m, RND[j]) for j in range(k + 1)]; rcc = recall(m, CTL)
        P(f"   {arm:10s} after round {k + 1}: recall rank / open by round taught: " + " | ".join(f"r{j + 1} {a:.2f} / {b:.2f}" for j, (a, b) in enumerate(rc))
          + f" || all taught {np.mean([a for a, _ in rc]):.3f} / {np.mean([b for _, b in rc]):.3f} | never taught {rcc[0]:.2f} / {rcc[1]:.2f} || forgetting "
          + ", ".join(f"{d} {dd[d] - d0[d]:+.3f}" for d in dd) + f" (mean {np.mean([dd[d] - d0[d] for d in dd]):+.4f})  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
