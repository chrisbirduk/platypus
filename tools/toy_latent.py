"""TOY LATENT ROUTE (no effect on the package): can facts pass through a JEPA-style trace, and can traces be COMPOSED in latent space?
Uses the toy of toy_compose.py (same data, same seed, its cached stage-A model: a frozen 2-block delta-rule model that reads facts back from
memory). Memory of a text = the per-block delta states after reading it alone (2 blocks x 4 heads x 32 x 32 = 8192 numbers).

TEST 1, BOTTLENECK SWEEP. memory of a fact (+ a distractor fact superposed) -> encoder E -> trace z (dz numbers) -> decoder D -> memory ->
the frozen model answers 'A r' -> B. E and D are trained on the training chains' facts (answer cross-entropy through the frozen model, plus
SIGReg on z, the LeJEPA isotropy term) and scored on the held-out chains' facts: accuracy among all entities (no name path) and among the
names stored beside the trace (NAME PATH: the fact's and the distractor's names, the prompt name excluded; chance 1/3).
TEST 2, LATENT COMPOSITION. Premise traces z_AB = E(M_AB), z_BC = E(M_BC) (E, D from test 1 at dz 256, frozen). A predictor maps them to
the trace of the conclusion's memory, z_AC = E(M['A r0 C .']); the result is decoded and read by the frozen model ('A r0' -> C).
Loss: answer cross-entropy + 0.95 x mean squared error per trace dimension (z_hat against z_AC) + 0.05 x SIGReg / N (the LeJEPA terms,
weighted to sit beside the cross-entropy; SIGReg as LeJEPA defines it is N x the statistic, hence / N). Predictors:
  MLP       P([z1, z2])                                  (no structure)
  BILINEAR  U((V z1) * (W z2)) + MLP                     (multiplicative interaction, low rank 256)
  READER    two-hop attention over a SET of traces (own A-B and B-C plus another chain's two, shuffled): query from A's embedding finds the
            trace with A as subject, its read-out forms the next query, which finds the trace with B as subject; z_hat = g(q1, r_last).
            The hop update is shared, so it can run 3 hops on 3-fact chains it never trained on.
Scored on HELD-OUT chains. Controls: another chain's A-B in place of the own (must FAIL if it composes; the shortcut 'C is in the B-C
trace' passes it), another chain's B-C (must fail either way); for the reader also two chains in the set (always) and THREE-HOP (3 hops).
usage: python3 toy_latent.py [seed] [step scale]   (needs toy_stageA_v2_s<seed>.pt from toy_compose.py in the working directory)"""
import sys, os, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0                    # step-count scale (smoke tests: 0.02)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_compose.py')).read()
head = src[:src.index('# ---------------------------------------------------------------- stage A')]
sys.argv = [sys.argv[0], str(SEED)]; exec(head)                            # identical data split, entities and model class
torch.set_num_threads(1)
CK = f'toy_stageA_v2_s{SEED}.pt'; assert os.path.exists(CK), CK
m = Toy(); m.load_state_dict(torch.load(CK)); m.eval()
for p in m.parameters(): p.requires_grad_(False)
t0 = time.time(); rsl = np.random.RandomState(9000 + SEED); torch.manual_seed(9000 + SEED)
NB, H, DH = m.nb, m.mix[0].h, m.mix[0].dh; DM = NB * H * DH * DH

@torch.no_grad()
def memflat(texts):
    out = []
    for i in range(0, len(texts), 256):
        Ss = m.mem(texts[i:i + 256]); out.append(torch.cat([S.reshape(S.shape[0], -1) for S in Ss], 1))
    return torch.cat(out)
def unflat(x): return [x[:, i * H * DH * DH:(i + 1) * H * DH * DH].reshape(-1, H, DH, DH) for i in range(NB)]
def answer_logits(prompts, Mflat):
    lg, _ = m.run(torch.tensor(prompts), unflat(Mflat)); return lg[:, -1]
def sigreg(z, ns=128, nk=17, tm=5.0):
    """LeJEPA SIGReg: Epps-Pulley statistic against N(0,1) along random unit directions, N x integral over t of |ecf - exp(-t^2/2)|^2 w(t)"""
    U = F.normalize(torch.randn(z.shape[1], ns), dim=0); x = z @ U; t = torch.linspace(0, tm, (nk + 1) // 2); tx = x[:, :, None] * t
    err = ((torch.cos(tx).mean(0) - torch.exp(-t ** 2 / 2)) ** 2 + torch.sin(tx).mean(0) ** 2) * torch.exp(-t ** 2 / 2)
    return z.shape[0] * 2 * torch.trapezoid(err, t, dim=-1).mean()

# all fact memories once (frozen model)
FT = {f: i for i, f in enumerate(sorted(set(facts)))}; FL = sorted(FT, key=FT.get)
MF = memflat([ftxt(f) for f in FL]); mu, sd = MF.mean(0), MF.std(0) + 1e-6
fab = lambda c: (c[0], R0 + c[3], c[1]); fbc = lambda c: (c[1], R0 + c[4], c[2])
TRf = sorted({fab(c) for c in TRN} | {fbc(c) for c in TRN}, key=FT.get); HOf = sorted({fab(c) for c in HOL} | {fbc(c) for c in HOL}, key=FT.get)
with torch.no_grad():                                                     # sanity: the full memory (no bottleneck)
    fa = HOf[:300]; fd = [HOf[(i + 7) % len(HOf)] for i in range(len(fa))]
    lg = answer_logits([[f[0], f[1]] for f in fa], MF[[FT[f] for f in fa]] + MF[[FT[f] for f in fd]])
    print(f"SANITY full memory, held-out facts with a distractor fact: one-hop {float((lg.argmax(-1) == torch.tensor([f[2] for f in fa])).float().mean()):.2f}  ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- TEST 1
class Bottle(nn.Module):
    def __init__(s, dz): super().__init__(); s.E = nn.Linear(DM, dz); s.D = nn.Linear(dz, DM)
    def enc(s, Mf): return s.E((Mf - mu) / sd)
    def dec(s, z): return s.D(z) * sd + mu
def score1(B_, fs, steps=None):
    with torch.no_grad():
        fd = [fs[(i + 11) % len(fs)] for i in range(len(fs))]
        z = B_.enc(MF[[FT[f] for f in fs]] + MF[[FT[f] for f in fd]]); lg = answer_logits([[f[0], f[1]] for f in fs], B_.dec(z))
        tgt = torch.tensor([f[2] for f in fs]); acc = float((lg.argmax(-1) == tgt).float().mean())
        cand = torch.tensor([[f[2], d[0], d[2]] for f, d in zip(fs, fd)])     # names stored beside the trace, prompt name excluded
        np_ = float((lg.gather(1, cand).argmax(-1) == 0).float().mean())
    return acc, np_
BOT = {}
print("TEST 1 BOTTLENECK SWEEP (memory of a fact + a distractor fact -> trace -> decoded memory -> 'A r' -> B), held-out facts:")
for dz in (16, 64, 256, 512):
    B_ = Bottle(dz); op = torch.optim.AdamW(B_.parameters(), lr=1e-3, weight_decay=1e-4)
    for st in range(max(2, int(1500 * SCALE))):
        ix = rsl.randint(len(TRf), size=64); fs = [TRf[i] for i in ix]; fd = [TRf[i] for i in rsl.randint(len(TRf), size=64)]
        z = B_.enc(MF[[FT[f] for f in fs]] + MF[[FT[f] for f in fd]])
        l = F.cross_entropy(answer_logits([[f[0], f[1]] for f in fs], B_.dec(z)), torch.tensor([f[2] for f in fs])) + 0.05 * sigreg(z) / 64
        op.zero_grad(); l.backward(); op.step()
    BOT[dz] = B_; a_tr, n_tr = score1(B_, TRf[:300]); a_ho, n_ho = score1(B_, HOf)
    print(f"  trace {dz:4d}: training facts {a_tr:.2f} (name path {n_tr:.2f}) | HELD-OUT facts {a_ho:.2f} (name path {n_ho:.2f}, chance 0.33)  ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- TEST 2
B_ = BOT[256]
for p in B_.parameters(): p.requires_grad_(False)
RC = R0                                                                    # the conclusion is written as 'A r0 C .'
with torch.no_grad():
    Z = B_.enc(MF)                                                         # trace of every fact
    CON = {c: i for i, c in enumerate(sorted({(c[0], c[2]) for c in ch2}))}
    ZC = B_.enc(memflat([[a, RC, c, DOT] for a, c in sorted(CON, key=CON.get)]))
zab = lambda cs: Z[[FT[fab(c)] for c in cs]]; zbc = lambda cs: Z[[FT[fbc(c)] for c in cs]]
DZ = Z.shape[1]
class MLPP(nn.Module):
    def __init__(s): super().__init__(); s.f = nn.Sequential(nn.Linear(2 * DZ, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, DZ))
    def forward(s, A_, S_): return s.f(torch.cat([S_[:, 0], S_[:, 1]], 1))
class BilP(nn.Module):
    def __init__(s, r=256):
        super().__init__(); s.V = nn.Linear(DZ, r); s.W = nn.Linear(DZ, r); s.U = nn.Linear(r, DZ); s.f = nn.Sequential(nn.Linear(2 * DZ, 256), nn.GELU(), nn.Linear(256, DZ))
    def forward(s, A_, S_): z1, z2 = S_[:, 0], S_[:, 1]; return s.U(s.V(z1) * s.W(z2)) + s.f(torch.cat([z1, z2], 1))
class Reader(nn.Module):
    """hop t: attention over the set with query q_t; read r_t; q_{t+1} = f(r_t) (shared); output g(q_1, r_last). The query starts from A's
    token embedding (frozen trunk embedding), so the reader is told WHO is asked about, never where."""
    def __init__(s, dk=128):
        super().__init__(); s.q0 = nn.Linear(m.emb.weight.shape[1], dk); s.k = nn.Linear(DZ, dk); s.v = nn.Linear(DZ, DZ)
        s.f = nn.Sequential(nn.Linear(DZ, 256), nn.GELU(), nn.Linear(256, dk)); s.g = nn.Sequential(nn.Linear(dk + DZ, 512), nn.GELU(), nn.Linear(512, DZ)); s.dk = dk
    def forward(s, A_, S_, hops=2):
        q = s.q0(m.emb(A_)); q1 = q; K = s.k(S_); Vv = s.v(S_)
        for _ in range(hops):
            a = torch.softmax((K @ q[:, :, None]).squeeze(-1) / math.sqrt(s.dk), -1); r = (a[:, :, None] * Vv).sum(1); q = s.f(r)
        return s.g(torch.cat([q1, r], 1))
def setof(cs, mode='own'):
    """premise set per chain: [A-B, B-C] for the pairwise predictors; for the reader + another chain's two premises, shuffled.
    mode 'wrongAB' / 'wrongBC' swap the own premise for another chain's (the reader's set then still holds two chains)"""
    n = len(cs); oth = [cs[(i + 1) % n] for i in range(n)]; oth2 = [cs[(i + 2) % n] for i in range(n)]
    a = zab(oth2) if mode == 'wrongAB' else zab(cs); b = zbc(oth2) if mode == 'wrongBC' else zbc(cs)
    return torch.stack([a, b], 1), torch.stack([zab(oth), zbc(oth)], 1)
def run_pred(P, cs, mode='own', train=False):
    own, dis = setof(cs, mode); A_ = torch.tensor([c[0] for c in cs])
    if isinstance(P, Reader):
        S_ = torch.cat([own, dis], 1); perm = torch.as_tensor(np.stack([rsl.permutation(4) for _ in cs])); S_ = torch.gather(S_, 1, perm[:, :, None].expand(-1, -1, DZ))
        return P(A_, S_)
    return P(A_, own)
def score2(P, cs, mode='own'):
    with torch.no_grad():
        zh = run_pred(P, cs, mode); lg = answer_logits([[c[0], RC] for c in cs], B_.dec(zh)); tgt = torch.tensor([c[2] for c in cs])
        acc = float((lg.argmax(-1) == tgt).float().mean())
        oth = [cs[(i + 1) % len(cs)] for i in range(len(cs))]
        cand = torch.tensor([[c[2], c[1], o[0], o[1], o[2]] for c, o in zip(cs, oth)])   # names stored beside the premise traces (A excluded)
        np_ = float((lg.gather(1, cand).argmax(-1) == 0).float().mean())
    return acc, np_
def score3(P):
    with torch.no_grad():
        n = len(ch3); A_ = torch.tensor([c[0] for c in ch3]); oth = [ch3[(i + 1) % n] for i in range(n)]
        tr = lambda cc: torch.stack([Z[[FT[(c[0], R0 + c[4], c[1])] for c in cc]], Z[[FT[(c[1], R0 + c[5], c[2])] for c in cc]], Z[[FT[(c[2], R0 + c[6], c[3])] for c in cc]]], 1)
        S_ = torch.cat([tr(ch3), tr(oth)], 1); zh = P(A_, S_, hops=3); lg = answer_logits([[c[0], RC] for c in ch3], B_.dec(zh))
        return float((lg.argmax(-1) == torch.tensor([c[3] for c in ch3])).float().mean())
with torch.no_grad():                                                      # upper bound: the TRUE conclusion trace, decoded
    lg = answer_logits([[c[0], RC] for c in HOL], B_.dec(ZC[[CON[(c[0], c[2])] for c in HOL]]))
    print(f"TEST 2 LATENT COMPOSITION (held-out chains, n {len(HOL)}): upper bound, the true conclusion trace decoded: {float((lg.argmax(-1) == torch.tensor([c[2] for c in HOL])).float().mean()):.2f}", flush=True)
for nm, P in (('MLP', MLPP()), ('BILINEAR', BilP()), ('READER', Reader())):
    op = torch.optim.AdamW(P.parameters(), lr=1e-3, weight_decay=0.05)
    for st in range(max(2, int(3000 * SCALE))):
        cs = [TRN[i] for i in rsl.randint(len(TRN), size=64)]; zh = run_pred(P, cs, train=True)
        zt = ZC[[CON[(c[0], c[2])] for c in cs]]
        l = F.cross_entropy(answer_logits([[c[0], RC] for c in cs], B_.dec(zh)), torch.tensor([c[2] for c in cs])) + 0.95 * ((zh - zt) ** 2).sum(1).mean() / DZ + 0.05 * sigreg(zh) / 64
        op.zero_grad(); l.backward(); op.step()
    a_tr = score2(P, TRN[:150]); a_ho = score2(P, HOL); a_wa = score2(P, HOL, 'wrongAB'); a_wb = score2(P, HOL, 'wrongBC')
    extra = f" | THREE-HOP (3 hops, held-out 3-fact chains) {score3(P):.2f}" if isinstance(P, Reader) else ""
    print(f"  {nm:9s} train {a_tr[0]:.2f} | HELD-OUT {a_ho[0]:.2f} (name path {a_ho[1]:.2f}, chance 0.20) | another chain's A-B {a_wa[0]:.2f} (must fail) | "
          f"another chain's B-C {a_wb[0]:.2f}{extra}{' | set of 4 traces, 2 chains' if isinstance(P, Reader) else ' | pair input'}  ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE {time.time()-t0:.0f}s")
