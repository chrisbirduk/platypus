"""TOY: ON-POLICY REHEARSAL (OPD against a pre-visit snapshot) vs STORED DREAMS (no effect on the package).
A small causal byte LM learns the five local domains in the harness's visit order (first visits, then revisits). No real text of an
old domain is ever replayed; old knowledge is protected only by the model's own dreams:
  NONE      no rehearsal (the forgetting baseline)
  STORED    off-policy, as Platypus's dream bank: at the end of each visit the model samples DREAMS (continuations of short prompts cut
            from that visit's text) and stores them with its own next-byte distributions; later steps add a rehearsal batch: KL(stored
            distribution || student) on the stored dream bytes
  ONPOLICY  on-policy distillation: at the start of each visit the model is SNAPSHOT (frozen teacher, no privileged context); during
            the visit the STUDENT samples continuations of the stored dream prompts and is pulled toward the snapshot on its OWN samples:
            reverse KL(student || snapshot) over the full byte distribution at every sampled position
  ONPOL-FWD the same samples with the forward KL(snapshot || student)
Matched budget: every rehearsal policy adds RB rehearsal rows of RL bytes per step (generation refreshed every GEN steps for the
on-policy ones), weight RW. Readouts (two seeds): FINAL mean BPC on held-out text, mean FORGETTING (end minus right after the domain's
last visit), PLASTICITY (BPC right after first visits), wall time.
usage: python3 toy_opd.py [seeds] [steps per visit]"""
import sys, os, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SPV = int(sys.argv[2]) if len(sys.argv) > 2 else 300
torch.set_num_threads(1); t0 = time.time()
DATA = os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile'); DOMS = ['prose', 'wiki', 'arxiv', 'pubmed', 'law']
RAW = {d: open(os.path.join(DATA, d + '.bin'), 'rb').read() for d in DOMS}; TR = {d: b[:2_600_000] for d, b in RAW.items()}; VA = {d: b[-500_000:] for d, b in RAW.items()}
ORDER = [0, 1, 2, 3, 4, 1, 3, 4, 0, 2]; LQ = 128; BS = 16; RB, RL, RW, GEN, PL, ND = 8, 64, 1.0, 4, 16, 64
def chunks(b, n, r, L): return [b[i:i + L] for i in r.randint(0, len(b) - L, size=n)]
class LM(nn.Module):
    def __init__(s, d=128, nl=3):
        super().__init__(); s.emb = nn.Embedding(256, d); s.pos = nn.Parameter(torch.randn(LQ + 1, d) * 0.02)
        s.b = nn.ModuleList(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True) for _ in range(nl)); s.o = nn.Linear(d, 256)
    def forward(s, X):
        T = X.shape[1]; cm = torch.triu(torch.full((T, T), float('-inf')), 1); x = s.emb(X) + s.pos[:T]
        for b in s.b: x = b(x, src_mask=cm)
        return s.o(x)
VAL = {d: torch.tensor([list(c) for c in chunks(VA[d], 48, np.random.RandomState(31 + i), LQ + 1)]) for i, d in enumerate(DOMS)}
@torch.no_grad()
def bpc(m):
    m.eval(); out = []
    for d in DOMS: Y = VAL[d]; out.append(float(F.cross_entropy(m(Y[:, :-1]).reshape(-1, 256), Y[:, 1:].reshape(-1))) / math.log(2))
    m.train(); return out
@torch.no_grad()
def sample(m, prompts, n, temp=1.0):
    """continue each prompt by n bytes; returns the full rows (prompt + continuation)"""
    was = m.training; m.eval(); X = prompts.clone()
    for _ in range(n): lg = m(X[:, -LQ:])[:, -1] / temp; X = torch.cat([X, torch.multinomial(F.softmax(lg, -1), 1)], 1)
    m.train(was); return X
def run(policy, seed):
    torch.manual_seed(seed); r = np.random.RandomState(seed); m = LM(); op = torch.optim.Adam(m.parameters(), lr=1e-3)
    bank = []; prompts = []; after = {}; first_after = {}; snap = None; gen_rows = None; tw = time.time()
    for vi, di in enumerate(ORDER):
        d = DOMS[di]
        if policy.startswith('ONPOL') and prompts:
            snap = LM(); snap.load_state_dict(m.state_dict()); snap.eval()
            for p in snap.parameters(): p.requires_grad_(False)
        for st in range(SPV):
            Y = torch.tensor([list(c) for c in chunks(TR[d], BS, r, LQ + 1)])
            l = F.cross_entropy(m(Y[:, :-1]).reshape(-1, 256), Y[:, 1:].reshape(-1))
            if policy == 'STORED' and bank:
                ix = r.randint(len(bank), size=RB); Xr = torch.stack([bank[i][0] for i in ix]); Tr = torch.stack([bank[i][1] for i in ix]).float()
                lp = F.log_softmax(m(Xr[:, :-1]), -1)[:, PL - 1:]; l = l + RW * (Tr.exp() * (Tr - lp)).sum(-1).mean()
            if policy.startswith('ONPOL') and snap is not None:
                if gen_rows is None or st % GEN == 0:
                    P = torch.stack([prompts[i] for i in r.randint(len(prompts), size=RB)]); gen_rows = sample(m, P, RL)   # the STUDENT's own samples
                Xg = gen_rows; lq = F.log_softmax(m(Xg[:, :-1]), -1)[:, PL - 1:]
                with torch.no_grad(): lt = F.log_softmax(snap(Xg[:, :-1]), -1)[:, PL - 1:]
                kl = (lq.exp() * (lq - lt)).sum(-1) if policy == 'ONPOLICY' else (lt.exp() * (lt - lq)).sum(-1)
                l = l + RW * kl.mean()
            op.zero_grad(); l.backward(); op.step()
        b = bpc(m); after[di] = b[di]
        if di not in first_after: first_after[di] = b[di]
        P = torch.tensor([list(c) for c in chunks(TR[d], ND, r, PL)]); prompts += list(P)   # dream prompts cut from this visit's text
        if policy == 'STORED':
            Xd = sample(m, P, RL)
            with torch.no_grad(): T = F.log_softmax(m(Xd[:, :-1]), -1)[:, PL - 1:].half()
            bank += [(Xd[i], T[i]) for i in range(ND)]
    fin = bpc(m)
    return float(np.mean(fin)), float(np.mean([fin[i] - after[i] for i in range(len(DOMS))])), float(np.mean(list(first_after.values()))), time.time() - tw
print(f"TOY ON-POLICY REHEARSAL, {SEEDS} seeds, {SPV} steps per visit, visit order {ORDER}; rehearsal {RB} rows x {RL} bytes per step, weight {RW}", flush=True)
for policy in ['NONE', 'STORED', 'ONPOLICY', 'ONPOL-FWD']:
    R = np.array([run(policy, s) for s in range(SEEDS)])
    print(f"  {policy:10s} | final mean BPC {R[:,0].mean():.3f} (seeds {' '.join(f'{v:.3f}' for v in R[:,0])}) | mean forgetting {R[:,1].mean():+.3f} "
          f"(seeds {' '.join(f'{v:+.3f}' for v in R[:,1])}) | plasticity (BPC after first visits) {R[:,2].mean():.3f} | {R[:,3].mean():.0f}s per run  ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE {time.time()-t0:.0f}s")
