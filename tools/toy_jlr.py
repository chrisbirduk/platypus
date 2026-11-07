"""TOY: A LEARNING RATE THAT FOLLOWS NOVELTY, WITH JEPA-ROUTED DOMAINS (no effect on the package).
A small causal byte language model learns the five local domains in the harness's visit order (0 1 2 3 4 1 3 4 0 2: first visits,
then revisits). A frozen JEPA encoder (tools/toy_jroute.py's V-JEPA byte encoder, pretrained on chunks of all five domains - it only
supplies the metric space) codes every batch; the learner keeps one PROTOTYPE per slot (the running mean of the codes routed to it) and
routes each batch to the nearest slot, opening a new slot when the batch is far from all of them (label-free; the threshold is
calibrated on the first visit's own spread). NOVELTY of a batch = its distance to the nearest prototype, divided by the running median.
LEARNING-RATE POLICIES (Adam, same base rate; the multiplier is clipped to [0.25, 4]):
  CONST        constant rate
  NOVEL        rate x novelty (novel text learns faster, familiar text slower)
  NOVEL-VISIT  rate set once per visit from its first 10 batches: x2 if they opened a new slot (a new domain), x0.5 for a revisit
  SLOT-AGE     rate x 2 / sqrt(1 + batches already routed to this slot / batches per visit): the routed domain's familiarity
  INVERSE      rate / novelty (control: familiar text learns faster)
  SURPRISE     rate x the learner's own byte loss on the batch / its running mean (the usual loss-based alternative)
  SHUFFLED     the NOVEL multipliers in a random order (control for the mean rate and its spread)
  HALF, DOUBLE constant x0.5 / x2; VISIT-ORACLE the NOVEL-VISIT schedule from the true domain labels (controls for NOVEL-VISIT)
READOUTS per policy, two seeds: BPC on each domain's held-out text after every visit; FINAL mean BPC; mean FORGETTING (BPC at the end
minus BPC right after the domain's last visit); PLASTICITY (BPC right after each first visit); routing accuracy of the slots.
usage: python3 toy_jlr.py [seeds] [steps per visit] [jepa steps]"""
import sys, os, time
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SPV = int(sys.argv[2]) if len(sys.argv) > 2 else 300
JST = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_jroute.py')).read()
sys.argv = [sys.argv[0], '0', str(JST)]
exec(src[:src.index('# windows: first-visit windows')])                   # data, JEPA encoder, its training and coding
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ORDER = [0, 1, 2, 3, 4, 1, 3, 4, 0, 2]; LQ = 128; BS = 16
class LM(nn.Module):
    def __init__(s, d=128, nl=3):
        super().__init__(); s.emb = nn.Embedding(257, d); s.pos = nn.Parameter(torch.randn(LQ, d) * 0.02)
        s.b = nn.ModuleList(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True) for _ in range(nl)); s.o = nn.Linear(d, 256)
        s.register_buffer('cm', torch.triu(torch.full((LQ, LQ), float('-inf')), 1))
    def forward(s, X):
        x = s.emb(X) + s.pos[:X.shape[1]]
        for b in s.b: x = b(x, src_mask=s.cm[:X.shape[1], :X.shape[1]])
        return s.o(x)
def lm_loss(m, Y):                                                          # Y: (B, LQ+1) bytes; predict Y[:,1:] from Y[:,:-1]
    lg = m(Y[:, :-1]); return F.cross_entropy(lg.reshape(-1, 256), Y[:, 1:].reshape(-1), reduction='none').view(Y.shape[0], -1).mean(1)
VAL = {d: torch.tensor([list(c) for c in chunks(VA[d], 48, np.random.RandomState(31 + i), LQ + 1)]) for i, d in enumerate(DOMS)}
@torch.no_grad()
def bpc(m): m.eval(); out = [float(lm_loss(m, VAL[d]).mean()) / math.log(2) for d in DOMS]; m.train(); return out
print(f"TOY NOVELTY LEARNING RATE, {SEEDS} seeds, {SPV} steps per visit, visit order {ORDER}", flush=True)
TG = train(False, JST)                                                      # the frozen JEPA metric space
def run(policy, seed):
    torch.manual_seed(seed); r = np.random.RandomState(seed); m = LM(); op = torch.optim.Adam(m.parameters(), lr=1e-3)
    P, NP_, med, hist, Lrun = [], [], [], [], None; shuf = None; routed = []; after = {}; first_after = {}; mult_log = []
    if policy == 'SHUFFLED':                                                # the NOVEL run's multipliers, permuted
        shuf = list(np.random.RandomState(seed + 5).permutation(NOVMULT[seed]))
    for vi, di in enumerate(ORDER):
        d = DOMS[di]; vm = None; opened = False
        for st in range(SPV):
            cs = chunks(TR[d], BS, r, LQ + 1); Y = torch.tensor([list(c) for c in cs])
            z = jcode(TG, [c[:CH] for c in cs], False).mean(0)
            if P:
                dd = torch.stack([(z - p / n).norm() for p, n in zip(P, NP_)]); j = int(dd.argmin()); nov = float(dd[j])
            else: j, nov = -1, float('inf')
            thr = (np.median(med) * 2.5) if len(med) >= 20 else float('inf')
            if not P or (nov > thr and len(med) >= 20): P.append(z.clone()); NP_.append(1); j = len(P) - 1; new_ = True; opened = opened or st < 10
            else: P[j] += z; NP_[j] += 1; new_ = False
            if nov != float('inf'): med.append(nov); med = med[-400:]
            routed.append((di, j))
            nrel = 1.0 if (nov == float('inf') or len(med) < 20) else nov / max(1e-6, float(np.median(med)))
            if new_ or nov == float('inf'): nrel = 4.0
            l = lm_loss(m, Y); lm_ = float(l.mean()); Lrun = lm_ if Lrun is None else 0.98 * Lrun + 0.02 * lm_
            if policy == 'CONST': k = 1.0
            elif policy == 'NOVEL': k = nrel
            elif policy == 'INVERSE': k = 1.0 / max(1e-6, nrel)
            elif policy == 'SURPRISE': k = lm_ / max(1e-6, Lrun)
            elif policy == 'SHUFFLED': k = shuf.pop() if shuf else 1.0
            elif policy == 'SLOT-AGE': k = 2.0 / math.sqrt(1.0 + NP_[j] / SPV)   # the routed slot's age: new domains learn faster
            elif policy == 'HALF': k = 0.5                                  # controls for NOVEL-VISIT: constant x0.5, constant x2,
            elif policy == 'DOUBLE': k = 2.0
            elif policy == 'VISIT-ORACLE': k = 1.0 if st < 10 else (2.0 if di not in ORDER[:vi] else 0.5)   #   and the same schedule from the true labels
            else:                                                           # NOVEL-VISIT: decided from the visit's first 10 batches
                if st < 10: k = 1.0
                else:
                    if vm is None: vm = 2.0 if (opened or vi == 0) else 0.5          #   (a new slot opened in them: a new domain)
                    k = vm
            k = float(np.clip(k, 0.25, 4.0)); mult_log.append(k)
            for g in op.param_groups: g['lr'] = 1e-3 * k
            op.zero_grad(); l.mean().backward(); op.step()
        b = bpc(m); after[di] = b[di]
        if di not in first_after: first_after[di] = b[di]
    fin = bpc(m); forg = np.mean([fin[i] - after[i] for i in range(len(DOMS))])
    acc = _route_acc(routed)
    return np.mean(fin), forg, np.mean(list(first_after.values())), acc, mult_log
def _route_acc(routed):
    """slot purity: each slot labelled by its majority domain; share of batches whose slot's label is their domain"""
    from collections import Counter
    lab = {s: Counter(d for d, s2 in routed if s2 == s).most_common(1)[0][0] for s in set(s for _, s in routed)}
    return float(np.mean([lab[s] == d for d, s in routed])), len(lab)
NOVMULT = {}
RES = {}
for policy in os.environ.get('JLR_POLICIES', 'CONST,NOVEL,NOVEL-VISIT,SLOT-AGE,INVERSE,SURPRISE,SHUFFLED').split(','):   # SHUFFLED needs NOVEL earlier in the list
    for seed in range(SEEDS):
        f, g, p, (ra, ns), ml = run(policy, seed)
        if policy == 'NOVEL': NOVMULT[seed] = ml
        RES.setdefault(policy, []).append((f, g, p, ra, ns, float(np.mean(ml))))
    R = np.array([x[:4] + (x[5],) for x in RES[policy]])
    print(f"  {policy:12s} | final mean BPC {R[:,0].mean():.3f} (seeds {' '.join(f'{v:.3f}' for v in R[:,0])}) | mean forgetting {R[:,1].mean():+.3f} | "
          f"plasticity (BPC after first visits) {R[:,2].mean():.3f} | routing purity {R[:,3].mean():.2f}, slots {RES[policy][0][4]} | mean rate multiplier {R[:,4].mean():.2f}  ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE {time.time()-t0:.0f}s")
