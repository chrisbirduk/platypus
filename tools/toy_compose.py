"""TOY COMPOSITION TEST: does superposed-memory training teach a delta-rule model to compose two stored facts on chains it never trained on?

A small 2-block delta-rule model (token level, tied or separate query) is first taught to READ its memory (stage A: one-hop questions answered
from a memory that holds the fact, with a distractor fact superposed, plus language-model rows on the fact text) - the ability the local trunk
lacks. Stage B then trains two-hop answers from SUPERPOSED memories (each premise read alone from a zero state, states added), as the
merged-memory dreams do, on the training chains only. Held-out chains' facts are seen in stage A (read like any fact) but never as two-hop.

Arms (same stage-A start): base (superposed two-hop, key/value projections only), +chain (contrastive chaining loss, v_B ~ k_B),
+onehop (stage-A one-hop rows kept during stage B), all (all parameters trained), and a NO-STAGE-B control.
Readouts on trained and held-out chains, accuracy among all entities and log p(C):
  superposed | no memory | B-C premise only | random memory | superposed WITH A SECOND CHAIN's premises (must follow A, not just find an end)
  | composed operator M_BC M_AB read once | one-hop from memory | three-hop on held-out 3-chains (three memories superposed).
Extra arms (argument 4, comma list): distract (every training memory also holds another chain's two premises, so the name-in-the-B-C-memory
shortcut stops paying), composed (trained from the composed operator M_BC M_AB instead of the sum), wd (AdamW weight decay 0.1, for long
runs, as in the grokking literature); combine with '+', e.g. distract+chain+wd. Stage A is cached per seed (toy_stageA_s<seed>.pt).
usage: python3 toy_compose.py [seed] [steps_B] [steps_A] [arms]"""
import sys, math, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS_B = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
STEPS_A = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
rs = np.random.RandomState(SEED); torch.manual_seed(SEED)

# ---------------------------------------------------------------- data
NREL = 8; NCH2 = 600; NCH3 = 60
SO, CONN, DOT, PAD = 0, 1, 2, 3; R0 = 4; E0 = R0 + NREL
nent = 3 * NCH2 + 4 * NCH3; V = E0 + nent
ents = list(range(E0, V)); rs.shuffle(ents)
ch2 = [(ents[3*i], ents[3*i+1], ents[3*i+2], int(rs.randint(NREL)), int(rs.randint(NREL))) for i in range(NCH2)]
o_ = 3 * NCH2
ch3 = [(ents[o_+4*i], ents[o_+4*i+1], ents[o_+4*i+2], ents[o_+4*i+3], *[int(x) for x in rs.randint(NREL, size=3)]) for i in range(NCH3)]
idx = rs.permutation(NCH2); HO = set(idx[:NCH2 // 4].tolist())             # a quarter of the 2-chains held out
TRN = [c for i, c in enumerate(ch2) if i not in HO]; HOL = [c for i, c in enumerate(ch2) if i in HO]
facts = []                                                                 # every fact (held-out chains' facts included: read like any fact)
for A, B, C, r1, r2 in ch2: facts += [(A, R0 + r1, B), (B, R0 + r2, C)]
for A, B, C, D, r1, r2, r3 in ch3: facts += [(A, R0 + r1, B), (B, R0 + r2, C), (C, R0 + r3, D)]
ftxt = lambda f: [f[0], f[1], f[2], DOT]

# ---------------------------------------------------------------- model
class Delta(nn.Module):
    def __init__(s, d, h, sepq):
        super().__init__(); s.h = h; s.dh = d // h; s.Wk = nn.Linear(d, d, bias=False); s.Wv = nn.Linear(d, d, bias=False)
        s.Wq = nn.Linear(d, d, bias=False) if sepq else None; s.Wb = nn.Linear(d, h); s.Wo = nn.Linear(d, d, bias=False)
    def forward(s, x, S0=None, cap=None):
        B, T, D = x.shape; k = F.normalize(s.Wk(x).view(B, T, s.h, s.dh), dim=-1); v = s.Wv(x).view(B, T, s.h, s.dh)
        q = F.normalize(s.Wq(x).view(B, T, s.h, s.dh), dim=-1) if s.Wq is not None else k; b = torch.sigmoid(s.Wb(x))
        S = torch.zeros(B, s.h, s.dh, s.dh) if S0 is None else S0; out = []
        if cap is not None: cap['k'] = k
        for t in range(T):
            kt, vt, bt = k[:, t], v[:, t], b[:, t, :, None, None]
            S = S + bt * torch.einsum('bhv,bhk->bhvk', vt - torch.einsum('bhvk,bhk->bhv', S, kt), kt)
            out.append(torch.einsum('bhvk,bhk->bhv', S, q[:, t]))
        return s.Wo(torch.stack(out, 1).reshape(B, T, D)), S
class Toy(nn.Module):
    def __init__(s, d=128, h=4, nb=2, sepq=False):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Embedding(16, d); s.nb = nb
        s.mix = nn.ModuleList(Delta(d, h, sepq) for _ in range(nb)); s.mlp = nn.ModuleList(nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)) for _ in range(nb))
        s.n1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(nb)); s.n2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(nb)); s.out = nn.Linear(d, V)
    def run(s, X, S0=None, caps=None):
        x = s.emb(X) + s.pos(torch.arange(X.shape[1])); Ss = []
        for i in range(s.nb):
            y, S = s.mix[i](s.n1[i](x), None if S0 is None else S0[i], None if caps is None else caps[i]); x = x + y; x = x + s.mlp[i](s.n2[i](x)); Ss.append(S)
        return s.out(x), Ss
    def mem(s, texts):                                                     # each text read alone from a zero state -> per-block final states
        return s.run(torch.tensor(texts))[1]
def addm(*Ms): return [sum(m[i] for m in Ms) for i in range(len(Ms[0]))]

def lp_answer(model, prompts, ans, M=None):
    """log p of the answer token after each prompt; prompts same length; M = per-block initial states (batch) or None"""
    lg, _ = model.run(torch.tensor(prompts), M); lp = F.log_softmax(lg[:, -1].float(), -1)
    return lp.gather(-1, torch.tensor(ans)[:, None]).squeeze(-1), lp

# ---------------------------------------------------------------- stage A: learn to read memory
def stageA_batch(n=64):
    F_ = [facts[int(i)] for i in rs.randint(len(facts), size=n)]; D_ = [facts[int(i)] for i in rs.randint(len(facts), size=n)]
    return F_, D_
def stageA_loss(model):
    F_, D_ = stageA_batch()
    M = addm(model.mem([ftxt(f) for f in F_]), model.mem([ftxt(f) for f in D_]))   # the fact plus a distractor fact superposed
    l1 = -lp_answer(model, [[f[0], f[1]] for f in F_], [f[2] for f in F_], M)[0].mean()
    lg, _ = model.run(torch.tensor([ftxt(f) for f in F_]))                 # language-model rows on the fact text
    l2 = F.cross_entropy(lg[:, :-1].reshape(-1, V), torch.tensor([ftxt(f) for f in F_])[:, 1:].reshape(-1))
    return l1 + 0.5 * l2

# ---------------------------------------------------------------- stage B pieces
def q2(A, C): return [SO, A, CONN], C
def chain_loss(model, cs, tau=0.1):
    """contrastive chaining: per block/head, M_AB k_q (k_q = key at the last token of the one-hop prompt 'A r1') should match k_B (key of B as
    the subject of 'B r2 C') more than the other chains' k_B"""
    MAB = model.mem([[A, R0 + r1, B, DOT] for A, B, C, r1, r2 in cs])
    caps = [{} for _ in range(model.nb)]; model.run(torch.tensor([[A, R0 + r1] for A, B, C, r1, r2 in cs]), None, caps); kq = [c['k'][:, -1] for c in caps]
    capb = [{} for _ in range(model.nb)]; model.run(torch.tensor([[B, R0 + r2, C, DOT] for A, B, C, r1, r2 in cs]), None, capb); kb = [c['k'][:, 0] for c in capb]
    ls = []
    for i in range(model.nb):
        vh = F.normalize(torch.einsum('bhvk,bhk->bhv', MAB[i], kq[i]), dim=-1).transpose(0, 1); kbi = kb[i].transpose(0, 1)   # (h, n, dh)
        sim = torch.bmm(vh, kbi.transpose(1, 2)) / tau; n = sim.shape[-1]
        ls.append(F.cross_entropy(sim.reshape(-1, n), torch.arange(n).repeat(sim.shape[0])))
    return sum(ls) / len(ls)
def stageB_loss(model, arm):
    cs = [TRN[int(i)] for i in rs.randint(len(TRN), size=64)]
    MAB = model.mem([[A, R0 + r1, B, DOT] for A, B, C, r1, r2 in cs]); MBC = model.mem([[B, R0 + r2, C, DOT] for A, B, C, r1, r2 in cs])
    if 'composed' in arm: M = [torch.einsum('bhvk,bhkj->bhvj', MBC[i], MAB[i]) for i in range(model.nb)]
    else: M = addm(MAB, MBC)
    if 'distract' in arm:                                                    # another chain's two premises superposed too
        pm = torch.as_tensor(rs.permutation(len(cs))); M = addm(M, [m[pm] for m in MAB], [m[pm] for m in MBC]) if 'composed' not in arm else addm(M, [torch.einsum('bhvk,bhkj->bhvj', MBC[i][pm], MAB[i][pm]) for i in range(model.nb)])
    l = -lp_answer(model, [q2(A, C)[0] for A, B, C, _, _ in cs], [C for A, B, C, _, _ in cs], M)[0].mean()
    if 'chain' in arm: l = l + chain_loss(model, cs)
    if 'onehop' in arm: l = l + stageA_loss(model)
    return l

# ---------------------------------------------------------------- readouts
@torch.no_grad()
def readout(model, cs, three=None):
    n = len(cs); A_, B_, C_ = [c[0] for c in cs], [c[1] for c in cs], [c[2] for c in cs]
    MAB = model.mem([[A, R0 + r1, B, DOT] for A, B, C, r1, r2 in cs]); MBC = model.mem([[B, R0 + r2, C, DOT] for A, B, C, r1, r2 in cs])
    perm = [(j + 1) % n for j in range(n)]; P_ = [q2(A, C)[0] for A, B, C, _, _ in cs]
    def ev(M):
        lp, full = lp_answer(model, P_, C_, M); ents_ = torch.tensor(range(E0, V))
        acc = (full[:, E0:].argmax(-1) + E0 == torch.tensor(C_)).float().mean()
        return float(acc), float(lp.mean())
    R = {}
    R['sup'] = ev(addm(MAB, MBC)); R['none'] = ev(None); R['bc'] = ev(MBC)
    R['rand'] = ev(addm([m[perm] for m in MAB], [m[perm] for m in MBC]))
    R['2chains'] = ev(addm(MAB, MBC, [m[perm] for m in MAB], [m[perm] for m in MBC]))   # another chain's two premises superposed too
    cp = lambda X, Y: [torch.einsum('bhvk,bhkj->bhvj', X[i], Y[i]) for i in range(model.nb)]
    R['composed'] = ev(cp(MBC, MAB))
    R['comp_wrongBC'] = ev(cp([m[perm] for m in MBC], MAB))                  # A-B composed with ANOTHER chain's B-C: must fail if it composes
    R['comp_wrongAB'] = ev(cp(MBC, [m[perm] for m in MAB]))                  # another chain's A-B with this B-C: names C, but the chain is broken
    oh = lp_answer(model, [[A, R0 + r1] for A, B, C, r1, r2 in cs], B_, addm(MAB, [m[perm] for m in MBC]))[1]
    R['onehop'] = (float((oh[:, E0:].argmax(-1) + E0 == torch.tensor(B_)).float().mean()), 0.0)
    if three:
        M3 = addm(model.mem([[A, R0 + r1, B, DOT] for A, B, C, D, r1, r2, r3 in three]), model.mem([[B, R0 + r2, C, DOT] for A, B, C, D, r1, r2, r3 in three]),
                  model.mem([[C, R0 + r3, D, DOT] for A, B, C, D, r1, r2, r3 in three]))
        lp, full = lp_answer(model, [[SO, c[0], CONN] for c in three], [c[3] for c in three], M3)
        R['3hop'] = (float((full[:, E0:].argmax(-1) + E0 == torch.tensor([c[3] for c in three])).float().mean()), float(lp.mean()))
        M1 = model.mem([[A, R0 + r1, B, DOT] for A, B, C, D, r1, r2, r3 in three]); M2 = model.mem([[B, R0 + r2, C, DOT] for A, B, C, D, r1, r2, r3 in three])
        M3c = model.mem([[C, R0 + r3, D, DOT] for A, B, C, D, r1, r2, r3 in three])
        lp, full = lp_answer(model, [[SO, c[0], CONN] for c in three], [c[3] for c in three], cp(M3c, cp(M2, M1)))
        R['3hop_composed'] = (float((full[:, E0:].argmax(-1) + E0 == torch.tensor([c[3] for c in three])).float().mean()), float(lp.mean()))
    return R

def fmt(R): return ' '.join(f"{k} {v[0]:.2f}" + (f"/{v[1]:.2f}" if k not in ('onehop',) else '') for k, v in R.items())

# ---------------------------------------------------------------- run
import os
ARMS = sys.argv[4].split(',') if len(sys.argv) > 4 else ['base', 'chain', 'onehop', 'chain+onehop', 'all']
t0 = time.time(); base = Toy(); CK = f'toy_stageA_s{SEED}.pt'
opA = torch.optim.Adam(base.parameters(), lr=1e-3)
if os.path.exists(CK): base.load_state_dict(torch.load(CK)); STEPS_A = 0; print(f"stage A loaded from {CK}", flush=True)
for st in range(STEPS_A):
    l = stageA_loss(base); opA.zero_grad(); l.backward(); opA.step()
    if (st + 1) % max(1, STEPS_A // 3) == 0: print(f"stage A step {st+1}: loss {float(l):.3f} | held-out {fmt(readout(base, HOL[:100]))}  ({time.time()-t0:.0f}s)", flush=True)
stateA = {k: v.clone() for k, v in base.state_dict().items()}
if STEPS_A: torch.save(stateA, CK)
res = {'stageA': {'trained': readout(base, TRN[:150]), 'heldout': readout(base, HOL, ch3)}}
print(f"AFTER STAGE A (no two-hop training): trained {fmt(res['stageA']['trained'])}\n                                     held-out {fmt(res['stageA']['heldout'])}", flush=True)
for arm in ARMS:
    m = Toy(); m.load_state_dict(stateA)
    ps = list(m.parameters()) if arm == 'all' else [p for n, p in m.named_parameters() if n.endswith(('Wk.weight', 'Wv.weight'))]
    op = torch.optim.AdamW(ps, lr=3e-4, weight_decay=0.1) if 'wd' in arm else torch.optim.Adam(ps, lr=3e-4); hist = []
    for st in range(STEPS_B):
        l = stageB_loss(m, arm); op.zero_grad(); l.backward(); op.step()
        if (st + 1) % max(1, STEPS_B // 3) == 0:
            Rt, Rh = readout(m, TRN[:150]), readout(m, HOL, ch3)
            print(f"[{arm}] step {st+1} loss {float(l):.3f}\n   trained  {fmt(Rt)}\n   HELD-OUT {fmt(Rh)}  ({time.time()-t0:.0f}s)", flush=True)
    res[arm] = {'trained': Rt, 'heldout': Rh}
json.dump(res, open(f'toy_compose_s{SEED}_{"_".join(ARMS).replace("+","-")}.json', 'w'), indent=1)
print(f"DONE {time.time()-t0:.0f}s")
