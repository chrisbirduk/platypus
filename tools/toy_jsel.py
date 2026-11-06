"""TOY JEPA TRACE SELECTION (no effect on the package): a context kept as many separate per-fact traces ACCOMPANIED by JEPA latents.
Does a question, mapped into the same latent space, pick the trace that holds its answer?
Uses the toy of toy_compose.py and its cached stage-A reader (frozen). Facts are FRESH random entity pairs, so the reader can only answer
from memory. Each fact of a context is read alone into its own delta memory (the content, no bottleneck: the reader answers 'A r' -> B
from it). A trace encoder Ec maps each memory to a latent z; a question encoder Eq maps the question tokens to a predicted z.
  JEPA     Eq is trained to PREDICT Ec(memory) of the fact the question asks about (MSE) with SIGReg on both latents (LeJEPA: no
           negatives, no stop-gradient). The trace whose z is nearest (cosine) to the prediction is read.
  NCE      the same encoders trained contrastively (InfoNCE over the batch, temperature 0.1) - a retrieval baseline for the JEPA loss.
  CONF     no index: every trace is read and the one giving the most confident answer is chosen (the selection of toy_multitrace).
  SUM      no index: all traces added and read at once.
  ORACLE   the right trace given.
Contexts of NCTX facts, RANDOM (subjects all different) or HARD (facts come in pairs sharing a subject with different relations, so the
index has to use the relation, not only the name). Questions in WORDING 1 ('A r'), WORDING 2 ('so r A ,') for relations Eq saw in that
wording, and WORDING 2 on HELD-OUT relations (Eq saw those relations only in wording 1): does the latent abstract over wording? The answer
is always read with wording 1 (the reader knows no other); wording only changes which trace is picked.
Reported: answer accuracy (and the share of questions whose own trace was picked).
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
@torch.no_grad()
def evaluate(E, n_ctx, ctx, wording, held, n=150):
    acc = {k: [] for k in ('oracle', 'sum', 'conf', 'idx')}; hit = {'conf': [], 'idx': []}
    for _ in range(n):
        fs = ctx(n_ctx)
        cand = [i for i, f in enumerate(fs) if wording == 1 or ((f[1] in SEEN2) != held)]
        if not cand: continue
        j = cand[int(rsm.randint(len(cand)))]; q = fs[j]; Mf = mems(fs); P = [[q[0], q[1]]]
        acc['oracle'].append(int(read(P, Mf[j:j + 1]).argmax()) == q[2]); acc['sum'].append(int(read(P, Mf.sum(0, keepdim=True)).argmax()) == q[2])
        lg = F.log_softmax(read(P * len(fs), Mf), -1); c = int(lg.max(-1).values.argmax()); acc['conf'].append(int(lg[c].argmax()) == q[2]); hit['conf'].append(c == j)
        if E is not None:
            s = F.normalize(E.zq([wq(q, wording)]), dim=-1) @ F.normalize(E.zc(Mf), dim=-1).T; c = int(s.argmax())
            acc['idx'].append(int(read(P, Mf[c:c + 1]).argmax()) == q[2]); hit['idx'].append(c == j)
    return {k: float(np.mean(v)) if v else float('nan') for k, v in acc.items()}, {k: float(np.mean(v)) if v else float('nan') for k, v in hit.items()}
print(f"TOY JEPA TRACE SELECTION, seed {SEED}, reader {CK}, per-fact memories of {DM} numbers, latent {DZ}, {STEPS} training steps")
MODELS = {k: train(k, STEPS) for k in ('jepa', 'nce')}
for n_ctx in (24, 96):
    for cn, ctx in (('RANDOM', fresh), ('HARD', hard)):
        for wn, w, held in (('wording 1', 1, False), ('wording 2, seen relations', 2, False), ('wording 2, HELD-OUT relations', 2, True)):
            a, h = evaluate(MODELS['jepa'], n_ctx, ctx, w, held); a2, h2 = evaluate(MODELS['nce'], n_ctx, ctx, w, held)
            print(f"  {n_ctx:3d} facts {cn:6s} {wn:30s} | ORACLE {a['oracle']:.2f} | SUM {a['sum']:.2f} | CONF {a['conf']:.2f} (own trace {h['conf']:.2f}) | "
                  f"JEPA {a['idx']:.2f} (own trace {h['idx']:.2f}) | NCE {a2['idx']:.2f} (own trace {h2['idx']:.2f})  ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE {time.time()-t0:.0f}s")
