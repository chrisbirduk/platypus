"""TOY JEPA TRACE SELECTION (no effect on the package): a context kept as many separate per-fact traces ACCOMPANIED by JEPA latents.
Does a question, mapped into the same latent space, pick the trace that holds its answer?
Uses the toy of toy_compose.py and its cached stage-A reader (frozen). Facts are FRESH random entity pairs, so the reader can only answer
from memory. Each fact of a context is read alone into its own delta memory (the content, no bottleneck: the reader answers 'A r' -> B
from it). A trace encoder Ec maps each memory to a latent z; a question encoder Eq maps the question tokens to a predicted z.
  JEPA     Eq is trained to PREDICT Ec(memory) of the fact the question asks about (MSE) with SIGReg on both latents (LeJEPA: no
           negatives, no stop-gradient). The trace whose z is nearest (cosine) to the prediction is read.
  NCE      the same encoders trained contrastively (InfoNCE over the batch, temperature 0.1) - a retrieval baseline for the JEPA loss.
  VJEPA    V-JEPA style masked latent prediction: ONE token encoder; the online encoder reads the question (the fact with its object
           masked), a predictor predicts what a MOVING-AVERAGE target encoder (momentum 0.996, stop-gradient) makes of the whole fact text;
           L1 loss, no SIGReg. Each trace is written with its key = the target encoder's latent of its fact text (the text is at hand when
           the trace is written); the nearest key (L1) is read.
  FROZEN   V-JEPA 2-AC style: the encoder is FROZEN and pretrained - the reader trunk's own hidden states on the fact text (last position
           and mean, standardised) are the key; only a predictor from the question is trained (L1).
  CONF     no index: every trace is read and the one giving the most confident answer is chosen (the selection of toy_multitrace).
  SUM      no index: all traces added and read at once.
  ORACLE   the right trace given.
Contexts of NCTX facts, RANDOM (subjects all different) or HARD (facts come in pairs sharing a subject with different relations, so the
index has to use the relation, not only the name). Questions in WORDING 1 ('A r'), WORDING 2 ('so r A ,') for relations Eq saw in that
wording, and WORDING 2 on HELD-OUT relations (Eq saw those relations only in wording 1): does the latent abstract over wording? The answer
is always read with wording 1 (the reader knows no other); wording only changes which trace is picked.
Reported: answer accuracy (and the share of questions whose own trace was picked).
TWO-HOP ROLLOUT (V-JEPA 2-AC planning analogue): context = a chain A r1 B, B r2 C plus same-subject distractors (A and B each with another
relation) and random facts; question 'A r1 r2'; two more facts with relation r2 and one with r1 about other subjects, so the relation alone cannot pick a trace (at
most 1/3). DECODE: hop 1 picked by the VJEPA index, read -> B'; hop 2 picked by the index for 'B' r2',
read -> C. Control: hop 2 asked about a RANDOM entity instead of B' (must fall to chance). LATENT: a hop-2 predictor P2 maps (the key of the
trace hop 1 RETRIEVED, r2) to the key of the B-C trace without decoding B, trained TEACHER-FORCED (true hop-1 key) or on its own ROLLOUT
(the key its own hop-1 prediction retrieves, as V-JEPA 2-AC trains on its own predictions); scored by whether the B-C trace is picked (reading it then needs B; with the true B
the reader answers from the picked trace). Control: P2 given another fact's key (must fall to chance).
usage: python3 toy_jsel.py [seed] [stageA checkpoint] [steps]"""
import sys, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CK = sys.argv[2] if len(sys.argv) > 2 else f'toy_stageA_v2_conv3_sepq_s{SEED}.pt'
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_compose.py')).read()
head = src[:src.index('# ---------------------------------------------------------------- stage A')]
sys.argv = [sys.argv[0], str(SEED)]; exec(head)                            # same vocabulary and model class (TOY_ARCH must match CK)
torch.set_num_threads(1); assert os.path.exists(CK), CK
m = Toy(); m.load_state_dict(torch.load(CK)); m.eval()
for p in m.parameters(): p.requires_grad_(False)
t0 = time.time(); rsm = np.random.RandomState(9000 + SEED); torch.manual_seed(9000 + SEED)
H, DH = m.mix[0].h, m.mix[0].dh; DM = m.nb * H * DH * DH; DZ = 128
def fresh(n):
    a = rsm.randint(E0, V, size=n); b = rsm.randint(E0, V, size=n); r = rsm.randint(NREL, size=n)
    return [(int(x), R0 + int(q), int(y if y != x else E0 + (y - E0 + 1) % nent)) for x, q, y in zip(a, r, b)]
def hard(n):                                                               # pairs: same subject, two different relations, two objects
    out = []
    for f in fresh(n // 2):
        r2 = R0 + (f[1] - R0 + 1 + int(rsm.randint(NREL - 1))) % NREL; out += [f, (f[0], r2, int(rsm.randint(E0, V)))]
    return out
@torch.no_grad()
def mems(fs): return torch.cat([S.reshape(len(fs), -1) for S in m.mem([ftxt(f) for f in fs])], 1)     # (n, DM), each fact alone
def unflat(x): return [x[:, i * H * DH * DH:(i + 1) * H * DH * DH].reshape(-1, H, DH, DH) for i in range(m.nb)]
@torch.no_grad()
def read(prompts, Mf): return m.run(torch.tensor(prompts), unflat(Mf))[0][:, -1]
def wq(f, w): return [f[0], f[1], PAD, PAD] if w == 1 else [SO, f[1], f[0], CONN]   # the two question wordings (padded to 4)
SEEN2 = set(range(R0, R0 + NREL // 2))                                     # relations Eq meets in wording 2 during training
with torch.no_grad(): M0 = mems(fresh(2048)); mu, sd = M0.mean(0), M0.std(0) + 1e-6
class Enc(nn.Module):
    def __init__(s):
        super().__init__()
        s.Ec = nn.Sequential(nn.Linear(DM, 512), nn.GELU(), nn.Linear(512, DZ))
        s.emb = nn.Embedding(V, 64); s.pos = nn.Parameter(torch.randn(4, 64) * 0.02)
        s.Eq = nn.Sequential(nn.Linear(4 * 64, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, DZ))
    def zc(s, Mf): return s.Ec((Mf - mu) / sd)
    def zq(s, Q): return s.Eq((s.emb(torch.tensor(Q)) + s.pos).reshape(len(Q), -1))
def sigreg(z, ns=128, nk=17, tm=5.0):
    U = F.normalize(torch.randn(z.shape[1], ns), dim=0); x = z @ U; t = torch.linspace(0, tm, (nk + 1) // 2); tx = x[:, :, None] * t
    err = ((torch.cos(tx).mean(0) - torch.exp(-t ** 2 / 2)) ** 2 + torch.sin(tx).mean(0) ** 2) * torch.exp(-t ** 2 / 2)
    return z.shape[0] * 2 * torch.trapezoid(err, t, dim=-1).mean()
def train(kind, steps):
    E = Enc(); op = torch.optim.AdamW(E.parameters(), lr=5e-4, weight_decay=1e-4); B = 128; ls = []
    for st in range(steps):
        fs = fresh(B // 2) + hard(B // 2); zc = E.zc(mems(fs))
        Q = [wq(f, 2 if (f[1] in SEEN2 and rsm.rand() < 0.5) else 1) for f in fs]; zq = E.zq(Q)
        if kind == 'jepa': l = F.mse_loss(zq, zc) + 0.05 * (sigreg(zc) + sigreg(zq)) / B
        else: l = F.cross_entropy(F.normalize(zq, dim=-1) @ F.normalize(zc, dim=-1).T / 0.1, torch.arange(B))
        op.zero_grad(); l.backward(); op.step(); ls.append(float(l))
    print(f"  trained {kind}: loss {np.mean(ls[:20]):.3f} -> {np.mean(ls[-50:]):.3f}  ({time.time()-t0:.0f}s)", flush=True); return E
class TokEnc(nn.Module):                                                   # one encoder for questions and fact texts (4 tokens)
    def __init__(s, dz=DZ):
        super().__init__(); s.emb = nn.Embedding(V, 64); s.pos = nn.Parameter(torch.randn(4, 64) * 0.02)
        s.f = nn.Sequential(nn.Linear(4 * 64, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, dz))
    def forward(s, Q): return s.f((s.emb(torch.tensor(Q)) + s.pos).reshape(len(Q), -1))
class VJ(nn.Module):
    """V-JEPA style: online encoder + predictor on the masked input (the question), EMA target encoder on the full fact text"""
    def __init__(s):
        super().__init__(); s.on = TokEnc(); s.pred = nn.Sequential(nn.Linear(DZ, 512), nn.GELU(), nn.Linear(512, DZ)); s.tg = TokEnc()
        s.tg.load_state_dict(s.on.state_dict())
        for p in s.tg.parameters(): p.requires_grad_(False)
    def predict(s, Q): return s.pred(s.on(Q))
    @torch.no_grad()
    def key(s, fs, Mf=None): return s.tg([ftxt(f) for f in fs])
@torch.no_grad()
def feats(X):                                                              # the frozen reader trunk's hidden states (B, T, d)
    x = m.emb(X) + m.pos(torch.arange(X.shape[1]))
    for i in range(m.nb):
        y, _ = m.mix[i](m.n1[i](x)); x = x + y; x = x + m.mlp[i](m.n2[i](x))
    return torch.cat([x[:, -1], x.mean(1)], 1)
with torch.no_grad(): F0 = feats(torch.tensor([ftxt(f) for f in fresh(2048)])); fmu, fsd = F0.mean(0), F0.std(0) + 1e-6
class FZ(nn.Module):
    """V-JEPA 2-AC style: frozen pretrained encoder (the reader trunk) gives the key; only the question predictor is trained"""
    def __init__(s): super().__init__(); s.q = TokEnc(F0.shape[1])
    def predict(s, Q): return s.q(Q)
    @torch.no_grad()
    def key(s, fs, Mf=None): return (feats(torch.tensor([ftxt(f) for f in fs])) - fmu) / fsd
def train_vj(kind, steps, mom=0.996):
    E = VJ() if kind == 'vjepa' else FZ(); P_ = [p for p in E.parameters() if p.requires_grad]; op = torch.optim.AdamW(P_, lr=5e-4, weight_decay=1e-4); B = 128; ls = []; sp = []
    for st in range(steps):
        fs = fresh(B // 2) + hard(B // 2); Q = [wq(f, 2 if (f[1] in SEEN2 and rsm.rand() < 0.5) else 1) for f in fs]
        l = (E.predict(Q) - E.key(fs)).abs().mean(); op.zero_grad(); l.backward(); op.step(); ls.append(float(l))
        if kind == 'vjepa':
            with torch.no_grad():
                for a_, b_ in zip(E.tg.parameters(), E.on.parameters()): a_.mul_(mom).add_(b_, alpha=1 - mom)
    with torch.no_grad(): K = E.key(fresh(512)); spread = float(K.std(0).mean())                 # collapse check: spread of the keys
    print(f"  trained {kind}: L1 {np.mean(ls[:20]):.3f} -> {np.mean(ls[-50:]):.3f} | key spread across facts {spread:.3f} (0 = collapsed)  ({time.time()-t0:.0f}s)", flush=True); return E
def sel_lat(E):                                                            # LeJEPA / NCE: cosine between question and memory latents
    return lambda q, w, fs, Mf: int((F.normalize(E.zq([wq(q, w)]), dim=-1) @ F.normalize(E.zc(Mf), dim=-1).T).argmax())
def sel_key(E):                                                            # VJEPA / FROZEN: L1 between the prediction and the stored keys
    return lambda q, w, fs, Mf: int((E.predict([wq(q, w)]) - E.key(fs)).abs().sum(-1).argmin())
@torch.no_grad()
def evaluate(SEL, n_ctx, ctx, wording, held, n=150):
    acc = {k: [] for k in ['oracle', 'sum', 'conf'] + list(SEL)}; hit = {k: [] for k in ['conf'] + list(SEL)}
    for _ in range(n):
        fs = ctx(n_ctx)
        cand = [i for i, f in enumerate(fs) if wording == 1 or ((f[1] in SEEN2) != held)]
        if not cand: continue
        j = cand[int(rsm.randint(len(cand)))]; q = fs[j]; Mf = mems(fs); P = [[q[0], q[1]]]
        acc['oracle'].append(int(read(P, Mf[j:j + 1]).argmax()) == q[2]); acc['sum'].append(int(read(P, Mf.sum(0, keepdim=True)).argmax()) == q[2])
        lg = F.log_softmax(read(P * len(fs), Mf), -1); c = int(lg.max(-1).values.argmax()); acc['conf'].append(int(lg[c].argmax()) == q[2]); hit['conf'].append(c == j)
        for k_, f_ in SEL.items():
            c = f_(q, wording, fs, Mf); acc[k_].append(int(read(P, Mf[c:c + 1]).argmax()) == q[2]); hit[k_].append(c == j)
    return {k: float(np.mean(v)) if v else float('nan') for k, v in acc.items()}, {k: float(np.mean(v)) if v else float('nan') for k, v in hit.items()}
print(f"TOY JEPA TRACE SELECTION, seed {SEED}, reader {CK}, per-fact memories of {DM} numbers, latent {DZ}, {STEPS} training steps")
ONLY = os.environ.get('JSEL_ONLY', '')                                     # 'vjepa': only the VJEPA index and the two-hop test
MODELS = {} if ONLY else {k: train(k, STEPS) for k in ('jepa', 'nce')}; MODELS.update({k: train_vj(k, STEPS) for k in (('vjepa',) if ONLY else ('vjepa', 'frozen'))})
SEL = {'VJEPA': sel_key(MODELS['vjepa'])} if ONLY else {'JEPA': sel_lat(MODELS['jepa']), 'NCE': sel_lat(MODELS['nce']), 'VJEPA': sel_key(MODELS['vjepa']), 'FROZEN': sel_key(MODELS['frozen'])}
for n_ctx in (() if ONLY else (24, 96)):
    for cn, ctx in (('RANDOM', fresh), ('HARD', hard)):
        for wn, w, held in (('wording 1', 1, False), ('wording 2, seen relations', 2, False), ('wording 2, HELD-OUT relations', 2, True)):
            a, h = evaluate(SEL, n_ctx, ctx, w, held)
            print(f"  {n_ctx:3d} facts {cn:6s} {wn:30s} | ORACLE {a['oracle']:.2f} | SUM {a['sum']:.2f} | CONF {a['conf']:.2f} (own {h['conf']:.2f}) | "
                  + " | ".join(f"{k_} {a[k_]:.2f} (own {h[k_]:.2f})" for k_ in SEL) + f"  ({time.time()-t0:.0f}s)", flush=True)
# ---------------------------------------------------------------- two-hop rollout
def chain_ctx(n):
    A, r1, B = fresh(1)[0]; _, r2, C = fresh(1)[0]
    oth = lambda r: R0 + (r - R0 + 1 + int(rsm.randint(NREL - 1))) % NREL
    rv = lambda: int(rsm.randint(E0, V))                                    # + two facts with relation r2 and one with r1 about OTHER subjects:
    fs = [(A, r1, B), (B, r2, C), (A, oth(r1), rv()), (B, oth(r2), rv()), (rv(), r2, rv()), (rv(), r2, rv()), (rv(), r1, rv())] + fresh(n - 7)   #   relation alone cannot pick the trace
    o = rsm.permutation(len(fs)); fs = [fs[i] for i in o]; return fs, int(np.argsort(o)[0]), int(np.argsort(o)[1]), (A, r1, B, r2, C)
VJm = MODELS['vjepa']
class P2(nn.Module):
    def __init__(s): super().__init__(); s.r = nn.Embedding(NREL, 32); s.f = nn.Sequential(nn.Linear(DZ + 32, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, DZ))
    def forward(s, z, r): return s.f(torch.cat([z, s.r(torch.tensor(r) - R0)], 1))
def train_p2(mode, steps):
    M_ = P2(); op = torch.optim.AdamW(M_.parameters(), lr=5e-4, weight_decay=1e-4); B = 128; ls = []
    for st in range(steps):
        c1 = fresh(B); c2 = [(f[2], R0 + int(rsm.randint(NREL)), int(rsm.randint(E0, V))) for f in c1]
        with torch.no_grad():
            K1 = VJm.key(c1); k2 = VJm.key(c2)
            if mode == 'teacher': z1 = K1
            else:                                                          # ROLLOUT: the key its own hop-1 prediction retrieves (among the batch)
                z1 = K1[(VJm.predict([wq(f, 1) for f in c1])[:, None] - K1[None]).abs().sum(-1).argmin(1)]
        l = (M_(z1, [f[1] for f in c2]) - k2).abs().mean(); op.zero_grad(); l.backward(); op.step(); ls.append(float(l))
    print(f"  trained hop-2 predictor ({mode}): L1 {np.mean(ls[:20]):.3f} -> {np.mean(ls[-50:]):.3f}  ({time.time()-t0:.0f}s)", flush=True); return M_
P2s = {k: train_p2(k, max(2, STEPS * 2 // 3)) for k in ('teacher', 'rollout')}
@torch.no_grad()
def two_hop(n_ctx, n=200):
    R = {k: [] for k in ('h1', 'dec', 'dec_hit', 'dec_rand', 'dec_rand_hit', 'lat_teacher', 'lat_rollout', 'lat_rollout_ans', 'lat_ctl')}
    for _ in range(n):
        fs, i1, i2, (A, r1, B, r2, C) = chain_ctx(n_ctx); Mf = mems(fs); K = VJm.key(fs)
        z1 = VJm.predict([wq((A, r1, B), 1)]); c1 = int((z1 - K).abs().sum(-1).argmin()); R['h1'].append(c1 == i1)
        Bh = int(read([[A, r1]], Mf[c1:c1 + 1]).argmax())
        c2 = int((VJm.predict([wq((Bh, r2, C), 1)]) - K).abs().sum(-1).argmin()); R['dec'].append(int(read([[Bh, r2]], Mf[c2:c2 + 1]).argmax()) == C); R['dec_hit'].append(c2 == i2)
        Br = int(rsm.randint(E0, V)); c2r = int((VJm.predict([wq((Br, r2, C), 1)]) - K).abs().sum(-1).argmin()); R['dec_rand'].append(int(read([[Br, r2]], Mf[c2r:c2r + 1]).argmax()) == C); R['dec_rand_hit'].append(c2r == i2)
        for k_, M_ in P2s.items():
            c2l = int((M_(K[c1:c1 + 1], [r2]) - K).abs().sum(-1).argmin()); R['lat_' + k_].append(c2l == i2)   # from the RETRIEVED hop-1 key
            if k_ == 'rollout': R['lat_rollout_ans'].append(int(read([[B, r2]], Mf[c2l:c2l + 1]).argmax()) == C)
        o_ = int(rsm.randint(len(fs)))
        while o_ in (i1, i2): o_ = int(rsm.randint(len(fs)))
        R['lat_ctl'].append(int((P2s['rollout'](K[o_:o_ + 1], [r2]) - K).abs().sum(-1).argmin()) == i2)
    return {k: float(np.mean(v)) for k, v in R.items()}
print("TWO-HOP ROLLOUT with the VJEPA index (chain + same-subject distractors + two other facts with relation r2 + random facts; picking by relation alone gives at most 0.33):")
for n_ctx in (24, 96):
    T = two_hop(n_ctx)
    print(f"  {n_ctx:3d} facts | hop 1 trace picked {T['h1']:.2f} | DECODE (read B, then index again) two-hop answer {T['dec']:.2f} (B-C trace picked {T['dec_hit']:.2f}), control with a RANDOM bridge {T['dec_rand']:.2f} (picked {T['dec_rand_hit']:.2f}) | "
          f"LATENT hop-2 trace picked: teacher-forced {T['lat_teacher']:.2f}, rollout-trained {T['lat_rollout']:.2f} (answer when read with the true B {T['lat_rollout_ans']:.2f}), "
          f"control from another fact's key {T['lat_ctl']:.2f} (chance {1/n_ctx:.2f})  ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE {time.time()-t0:.0f}s")
