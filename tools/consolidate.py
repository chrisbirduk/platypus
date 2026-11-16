"""CONSOLIDATION INTO THE TRUNK: WHICH UPDATE KEEPS KEPT INFERENCES WITHOUT FORGETTING? (16 Nov 2026; tools/consolidate.py)
Build B (a sparse memory layer in the trunk) failed in the package's continually trained trunk. Simpler ways to move kept, typed inferences
into a trunk's own weights, on the stand-in trunk (TG_CKPT, d 192, 3 blocks): the 200 written inferences of the GPU path at visit 2
(distant_50k), typed as the package types them ('head <wording> tail', inf_rti_typed), taught in six self-written frames, 8 fact rows +
24 held-in wiki rows a step (the GPU's mix), CS_STEPS steps, for each update rule and learning rate:
  FULL      every weight
  KL        every weight, the wiki rows' loss = KL to the OLD model's predictions (learning without forgetting) instead of cross-entropy
  MLPOUT    every block's MLP output matrix only (the GPU builds' question-row path)
  LORA      a rank-CS_R adapter on every block's MLP (input and output), the base frozen
Readouts: recall in a test wording never taught ('It is on record that <head> <wording> ') - rank of the true tail among 10 tails of written
inferences, open-ended exact; for right and wrong inferences; held-out wiki bits per byte added (forgetting).
Usage: TG_CKPT=<trunk> python3 tools/consolidate.py <distant_50k.json.gz>   env CS_STEPS (300), CS_LRS (1e-4,3e-4,1e-3), CS_RULES, CS_R (8)"""
import os, sys, types, copy, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2); torch.manual_seed(0)
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rti_typed=1)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']
pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < int(round(ND * 0.2))], 1)
CL = [(w['th'], w['tw'], w['tt'], bool(w['truth'])) for w in pm._PM['rti'].values() if w.get('tw')]
del R, pm._PM['R']
P(f"{len(CL)} typed inferences (right {np.mean([c[3] for c in CL]):.3f}), e.g. " + "; ".join(f"{h} {w} {t}" for h, w, t, _ in CL[:4]) + f"  ({time.time() - T0:.0f}s)")
DW = int(os.environ.get('TG_D', 192)); L_ = 128; PAD = 256; V_ = 257
class Blk(nn.Module):
    def __init__(s, d, hh):
        super().__init__(); s.a = nn.MultiheadAttention(d, hh, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
        s.lora = None
    def forward(s, x, am):
        q = s.n1(x); x = x + s.a(q, q, q, attn_mask=am, need_weights=False)[0]; z = s.n2(x)
        if s.lora is None: return x + s.m(z)
        A1, B1, A2, B2 = s.lora; u = s.m[0](z) + (z @ A1) @ B1; u = s.m[1](u); return x + s.m[2](u) + (u @ A2) @ B2
class LM(nn.Module):
    def __init__(s, d=DW, nl=3):
        super().__init__(); s.emb = nn.Embedding(V_, d); s.pos = nn.Parameter(torch.randn(L_, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl)); s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V_)
    def forward(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for b in s.b: x = b(x, am)
        return s.o(s.n(x))
base = LM(); base.load_state_dict(torch.load(os.environ['TG_CKPT'])); base.eval()
wiki = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin', 'rb').read().replace(b'\n', b' '); WK = np.frombuffer(wiki, np.uint8); HO = len(WK) - 200000
def tens(bl):
    bl = [b[:L_] for b in bl]; T = max(len(b) for b in bl); X = torch.full((len(bl), T), PAD, dtype=torch.long)
    for i, b in enumerate(bl): X[i, :len(b)] = torch.tensor(list(b))
    return X, torch.cat([torch.full((len(bl), 1), PAD), X[:, :-1]], 1)
FR = ("{h} {w} {t}.", "In short, {h} {w} {t}.", "Notes: {h} {w} {t}.", "As recorded, {h} {w} {t}.", "From the notes: {h} {w} {t}.", "{h}, as noted, {w} {t}.")
FR24 = FR + ("It is known that {h} {w} {t}.", "We read that {h} {w} {t}.", "Sources say {h} {w} {t}.", "{h} - {w} {t}.", "Reportedly, {h} {w} {t}.", "Note that {h} {w} {t}.",
             "According to records, {h} {w} {t}.", "Fact: {h} {w} {t}.", "One learns that {h} {w} {t}.", "Indeed, {h} {w} {t}.", "Background: {h} {w} {t}.",
             "As written elsewhere, {h} {w} {t}.", "{h}? {h} {w} {t}.", "Remember: {h} {w} {t}.", "On {h}: {h} {w} {t}.", "Summary - {h} {w} {t}.",
             "Earlier we saw that {h} {w} {t}.", "Briefly, {h} {w} {t}.")       # 16 Nov CS_FRAMES 24: knowledge augmentation (varied wordings while learning)
if int(os.environ.get('CS_FRAMES', 6)) == 24: FR = FR24
FACT = [f.format(h=h, w=w, t=t).encode('latin-1', 'replace')[:L_] for h, w, t, _ in CL for f in FR]
EV = [bytes(WK[HO + k * 129:HO + (k + 1) * 129]) for k in range(64)]
@torch.no_grad()
def wbits(m):
    X, Xi = tens(EV); lg = m(Xi); return float(F.cross_entropy(lg.reshape(-1, V_), X.reshape(-1), ignore_index=PAD)) / math.log(2)
tails = [c[2] for c in CL]; rr = np.random.RandomState(3)
CANDS = [[t] + [tails[j] for j in rr.choice(len(tails), 12, replace=False) if tails[j] != t][:9] for h, w, t, _ in CL]
@torch.no_grad()
def recall(m):
    rk, ex = [], []
    for (h, w, t, y), cs in zip(CL, CANDS):
        pb = f"It is on record that {h} {w} ".encode('latin-1', 'replace'); bl = [(pb + c.encode('latin-1', 'replace') + b'.')[:L_] for c in cs]
        X, Xi = tens(bl); lp = F.log_softmax(m(Xi), -1); n0 = len(pb)
        sc = [float(lp[i, n0:len(b)].gather(-1, X[i, n0:len(b), None]).mean()) for i, b in enumerate(bl)]
        rk.append(int(np.argmax(sc) == 0)); tb = (t + '.').encode('latin-1', 'replace')
        ex.append(int(bool((lp[0, n0:n0 + len(tb) - 1].argmax(-1) == X[0, n0:n0 + len(tb) - 1]).all())))
    rk, ex = np.array(rk), np.array(ex); yv = np.array([c[3] for c in CL])
    return rk.mean(), ex.mean(), rk[yv].mean(), rk[~yv].mean()
b0 = wbits(base); r0 = recall(base)
P(f"BASE: held-out wiki {b0:.4f} bits per byte | test-wording rank {r0[0]:.3f} (right {r0[2]:.3f}, wrong {r0[3]:.3f}), open {r0[1]:.3f}  ({time.time() - T0:.0f}s)")
STEPS = int(os.environ.get('CS_STEPS', 300)); RK = int(os.environ.get('CS_R', 8))
for rule in os.environ.get('CS_RULES', 'FULL,KL,MLPOUT,LORA').split(','):
    for lr in [float(x) for x in os.environ.get('CS_LRS', '1e-4,3e-4,1e-3').split(',')]:
        m = copy.deepcopy(base); m.train(); rs = np.random.RandomState(11)
        if rule == 'LORA':
            for p in m.parameters(): p.requires_grad_(False)
            ps = []
            for b in m.b:
                d = DW; A1 = nn.Parameter(torch.randn(d, RK) * 0.01); B1 = nn.Parameter(torch.zeros(RK, 4 * d)); A2 = nn.Parameter(torch.randn(4 * d, RK) * 0.01); B2 = nn.Parameter(torch.zeros(RK, d))
                b.lora = (A1, B1, A2, B2); ps += [A1, B1, A2, B2]
        elif rule == 'MLPOUT': ps = [b.m[2].weight for b in m.b] + [b.m[2].bias for b in m.b]
        else: ps = list(m.parameters())
        op = torch.optim.Adam(ps, lr=lr)
        for st in range(STEPS):
            fb = [FACT[i] for i in rs.randint(0, len(FACT), 8)]; wb = [bytes(WK[i:i + 129]) for i in rs.randint(0, HO - 200, 24)]
            Xf, Xfi = tens(fb); lf = F.cross_entropy(m(Xfi).reshape(-1, V_), Xf.reshape(-1), ignore_index=PAD) * (0.0 if rule == 'NONE' else 1.0)   # NONE: wiki only (the drift control)
            Xw, Xwi = tens(wb); lw_ = m(Xwi)
            if rule == 'KL':
                with torch.no_grad(): lo = F.log_softmax(base(Xwi), -1)
                lw = F.kl_div(F.log_softmax(lw_, -1), lo, log_target=True, reduction='batchmean') / Xw.shape[1]
            else: lw = F.cross_entropy(lw_.reshape(-1, V_), Xw.reshape(-1), ignore_index=PAD)
            loss = (8 * lf + 24 * lw) / 32; op.zero_grad(); loss.backward(); op.step()
        m.eval(); bb = wbits(m); r = recall(m)
        P(f"   {rule:6s} lr {lr:g}: test-wording rank {r[0]:.3f} (right {r[2]:.3f}, wrong {r[3]:.3f}), open {r[1]:.3f} | wiki {bb:.4f} ({bb - b0:+.4f} bits: forgetting)  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
