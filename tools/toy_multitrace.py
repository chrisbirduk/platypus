"""TOY MULTI-TRACE CONTEXT (no effect on the package): compress a context into a JEPA-style trace until the trace is full, then start a new one.
Uses the toy of toy_compose.py (same model class, its cached stage-A reader, frozen). Facts are drawn FRESH (random entity pairs), so the
reader can only answer from memory.
  chunk of k facts, written one after another into ONE delta memory (sequential, as a stream) -> encoder E -> trace z (dz numbers)
  -> decoder D -> a memory the frozen reader reads -> 'A r' -> B for a fact of the chunk. A second decoder D2 rebuilds the raw chunk memory
  from z (no labels): its relative error is the candidate SATURATION signal. E, D, D2 trained on random k in 1..KMAX; SIGReg on z (LeJEPA).
TEST 1 SATURATION CURVE: answer accuracy against k for each trace size, next to the reader with NO bottleneck. For the no-bottleneck figure
  the chunk's facts are read separately and superposed (the format the reader was trained on) - the best the reader can do with k facts.
TEST 2 LABEL-FREE SIGNAL: D2's relative reconstruction error against k, and its correlation with answer accuracy (per chunk).
TEST 3 CONTEXT OF MANY TRACES: a context of NCTX facts split by: ONE trace (everything), FIXED cuts every k* facts (k* from test 1: largest
  k still within 0.1 of the k=1 accuracy), ADAPTIVE cuts (start a new trace when D2's error would pass the error at k*), PER-FACT traces.
  Read out by SUM (every trace decoded, memories added) or SELECT (the trace whose decoded memory gives the answer token the highest
  log p for the question - the question chooses its trace).
usage: python3 toy_multitrace.py [seed] [stageA checkpoint] [step scale]"""
import sys, os, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CK = sys.argv[2] if len(sys.argv) > 2 else f'toy_stageA_v2_s{SEED}.pt'
SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_compose.py')).read()
head = src[:src.index('# ---------------------------------------------------------------- stage A')]
sys.argv = [sys.argv[0], str(SEED)]; exec(head)                            # same vocabulary and model class
torch.set_num_threads(1); assert os.path.exists(CK), CK
m = Toy(); m.load_state_dict(torch.load(CK)); m.eval()
for p in m.parameters(): p.requires_grad_(False)
t0 = time.time(); rsm = np.random.RandomState(7000 + SEED); torch.manual_seed(7000 + SEED)
NB, H, DH = m.nb, m.mix[0].h, m.mix[0].dh; DM = NB * H * DH * DH; KMAX = 24; NCTX = 24
def fresh(n):
    a = rsm.randint(E0, V, size=n); b = rsm.randint(E0, V, size=n); r = rsm.randint(NREL, size=n)
    return [(int(x), R0 + int(q), int(y if y != x else E0 + (y - E0 + 1) % nent)) for x, q, y in zip(a, r, b)]
def flat(Ss): return torch.cat([S.reshape(S.shape[0], -1) for S in Ss], 1)
def unflat(x): return [x[:, i * H * DH * DH:(i + 1) * H * DH * DH].reshape(-1, H, DH, DH) for i in range(NB)]
@torch.no_grad()
def seqmem(chunks):
    """each chunk (list of facts, equal lengths) written one after another into one memory from a zero state: fact i is read with the
    memory left by facts 0..i-1 carried in (positions restart per fact: the toy's position table is 16 long)"""
    Ss = None
    for i in range(len(chunks[0])): _, Ss = m.run(torch.tensor([ftxt(ch[i]) for ch in chunks]), Ss)
    return flat(Ss)
@torch.no_grad()
def supmem(chunks):
    """the reader's own format: each fact read alone, memories added (no bottleneck, best case)"""
    k = len(chunks[0]); out = 0
    for i in range(k): out = out + flat(m.mem([ftxt(ch[i]) for ch in chunks]))
    return out
def logits(prompts, Mf): return m.run(torch.tensor(prompts), unflat(Mf))[0][:, -1]
def sigreg(z, ns=128, nk=17, tm=5.0):
    U = F.normalize(torch.randn(z.shape[1], ns), dim=0); x = z @ U; t = torch.linspace(0, tm, (nk + 1) // 2); tx = x[:, :, None] * t
    err = ((torch.cos(tx).mean(0) - torch.exp(-t ** 2 / 2)) ** 2 + torch.sin(tx).mean(0) ** 2) * torch.exp(-t ** 2 / 2)
    return z.shape[0] * 2 * torch.trapezoid(err, t, dim=-1).mean()
with torch.no_grad():                                                      # normalisation from sequential memories of random length
    Ms = torch.cat([seqmem([fresh(k) for _ in range(32)]) for k in range(1, KMAX + 1, 3)]); mu, sd = Ms.mean(0), Ms.std(0) + 1e-6
class ED(nn.Module):
    def __init__(s, dz):
        super().__init__(); s.E = nn.Sequential(nn.Linear(DM, 1024), nn.GELU(), nn.Linear(1024, dz)); s.D = nn.Sequential(nn.Linear(dz, 1024), nn.GELU(), nn.Linear(1024, DM))
        s.D2 = nn.Sequential(nn.Linear(dz, 1024), nn.GELU(), nn.Linear(1024, DM))
    def enc(s, Mf): return s.E((Mf - mu) / sd)
    def dec(s, z): return s.D(z) * sd + mu
    def recon_err(s, z, Mf): t = (Mf - mu) / sd; return ((s.D2(z) - t) ** 2).sum(1) / (t ** 2).sum(1).clamp_min(1e-6)
def train(dz, steps):
    M_ = ED(dz); op = torch.optim.AdamW(M_.parameters(), lr=5e-4, weight_decay=1e-4)
    for st in range(steps):
        k = int(rsm.randint(1, KMAX + 1)); chs = [fresh(k) for _ in range(48)]; Mf = seqmem(chs); z = M_.enc(Mf)
        j = rsm.randint(k, size=48); fq = [ch[i] for ch, i in zip(chs, j)]
        l = F.cross_entropy(logits([[f[0], f[1]] for f in fq], M_.dec(z)), torch.tensor([f[2] for f in fq])) + M_.recon_err(z, Mf).mean() + 0.05 * sigreg(z) / 48
        op.zero_grad(); l.backward(); op.step()
    return M_
@torch.no_grad()
def acc_k(M_, k, n=200):
    chs = [fresh(k) for _ in range(n)]; j = rsm.randint(k, size=n); fq = [ch[i] for ch, i in zip(chs, j)]; P_ = [[f[0], f[1]] for f in fq]; T_ = torch.tensor([f[2] for f in fq])
    z = M_.enc(seqmem(chs)); a = float((logits(P_, M_.dec(z)).argmax(-1) == T_).float().mean()); e = M_.recon_err(z, seqmem(chs))
    nb = float((logits(P_, supmem(chs)).argmax(-1) == T_).float().mean())
    return a, nb, float(e.mean())
print(f"SANITY reader, no bottleneck, facts read separately and superposed: k=1 {acc_k(ED(8), 1)[1]:.2f}, k=2 {acc_k(ED(8), 2)[1]:.2f}, k=4 {acc_k(ED(8), 4)[1]:.2f}  ({time.time()-t0:.0f}s)", flush=True)
KS = (1, 2, 4, 8, 12, 16, 24); RES = {}; MODELS = {}
print("TEST 1 SATURATION CURVE (accuracy | reader with no bottleneck | TEST 2 label-free reconstruction error), held-out fresh facts:")
for dz in (32, 128, 512):
    M_ = train(dz, max(2, int(4000 * SCALE))); MODELS[dz] = M_; RES[dz] = {k: acc_k(M_, k) for k in KS}
    print(f"  trace {dz:4d}: " + " | ".join(f"k={k} {RES[dz][k][0]:.2f} ({RES[dz][k][1]:.2f}) err {RES[dz][k][2]:.2f}" for k in KS) + f"  ({time.time()-t0:.0f}s)", flush=True)
    A_ = np.array([RES[dz][k][0] for k in KS]); E_ = np.array([RES[dz][k][2] for k in KS])
    print(f"           correlation of reconstruction error with accuracy across k: {np.corrcoef(A_, E_)[0, 1]:+.2f}", flush=True)
dz = 128; M_ = MODELS[dz]; a1 = RES[dz][1][0]
kstar = max([k for k in KS if RES[dz][k][0] >= a1 - 0.1] or [1]); ethr = RES[dz][kstar][2]
@torch.no_grad()
def context_test(policy, n=100):
    """n contexts of NCTX fresh facts; a question about a random fact of each"""
    okS, okP = [], []
    for _ in range(n):
        ctx = fresh(NCTX); j = int(rsm.randint(NCTX)); q = ctx[j]
        if policy == 'one': cuts = [ctx]
        elif policy == 'fixed': cuts = [ctx[i:i + kstar] for i in range(0, NCTX, kstar)]
        elif policy == 'perfact': cuts = [[f] for f in ctx]
        else:                                                              # adaptive: grow the chunk until the label-free error passes ethr
            cuts, cur = [], []
            for f in ctx:
                if cur and float(M_.recon_err(M_.enc(seqmem([cur + [f]])), seqmem([cur + [f]]))[0]) > ethr: cuts.append(cur); cur = []
                cur = cur + [f]
            cuts.append(cur)
        Z = M_.enc(seqmem(cuts) if len(set(len(c) for c in cuts)) == 1 else torch.cat([seqmem([c]) for c in cuts])); D_ = M_.dec(Z)
        lgS = logits([[q[0], q[1]]], D_.sum(0, keepdim=True)); okS.append(int(lgS.argmax(-1)) == q[2])
        lgP = logits([[q[0], q[1]]] * len(cuts), D_); best = int(F.log_softmax(lgP, -1)[:, :].max(-1).values.argmax())   # the trace the question trusts most
        okP.append(int(lgP[best].argmax()) == q[2])
    return float(np.mean(okS)), float(np.mean(okP)), len(cuts)
print(f"TEST 3 CONTEXT OF {NCTX} FACTS, trace {dz}, k* {kstar} (largest k within 0.1 of k=1), adaptive threshold {ethr:.2f}: accuracy SUM of decoded traces / SELECT by the question")
for pol, nm in (('one', 'ONE trace'), ('fixed', f'FIXED cuts every {kstar}'), ('adaptive', 'ADAPTIVE (label-free error)'), ('perfact', 'PER-FACT traces')):
    s_, p_, nc = context_test(pol); print(f"  {nm:28s} sum {s_:.2f} | select {p_:.2f} | traces {nc}  ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE {time.time()-t0:.0f}s")
