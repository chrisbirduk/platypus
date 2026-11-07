"""TOY: A GENERIC (NAME-FREE) JEPA TRACE (no effect on the package). Part 2 of toy_jname.py: once the parts a JEPA cannot generalise (the
names) are found, can they be REMOVED to leave a generic trace that keeps the structure (relation, wording) and is the same across encoders
(the Platonic Representation Hypothesis, Huh et al. 2024)?
Same data and models as toy_jname.py (exec'd from it): a byte language model and text JEPAs (EMA target, V-JEPA style), trained twice
with different seeds (A and B) for the convergence test. A fact = one sentence about two test names (never seen in training).
TRACES (mean over the text positions of the JEPA context encoder's output):
  RAW        the sentence as it is
  DROP       RAW, but the name positions left out of the mean (how much identity LEAKS into the other positions through attention)
  MASK-GOLD  the true name spans masked in the input, then encoded (the ceiling: removal with perfect detection)
  MASK-JEPA  the 2 words with the largest GAP z(byte surprise) - z(JEPA surprise) masked (JEPA's own detection; 2 = the number of names)
  MASK-BYTE  the 2 most byte-surprising words masked (detection without JEPA)
  MASK-RAND  2 random words masked (control)
MEASURES (test names; each a number per trace):
  1 NAME INVARIANCE  AUC: pairs with the SAME relation and wording but DIFFERENT names are more alike (cosine) than pairs with the SAME
                     names but a DIFFERENT relation (same wording slot). Generic -> high; name-dominated -> low.
  2 STRUCTURE KEPT   linear probes trained on training-name facts, tested on test-name facts: relation (8 classes, chance 0.125) and
                     wording (4 classes, chance 0.25).
  3 IDENTITY LEFT    AUC: pairs of facts that SHARE a name (different relations) are more alike than pairs that share none. 0.5 = no
                     identity left.
  4 CONVERGENCE      mutual k-nearest-neighbour alignment (k 10) of seed-A and seed-B traces over the same facts (Huh et al.); chance
                     about k / n.
usage: python3 toy_jgeneric.py [seed] [steps]"""
import sys, os, re, time
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_jname.py')).read()
head = src[:src.index('print(f"TOY TEXT JEPA NAME DETECTION')]
sys.argv = [sys.argv[0], str(SEED), str(STEPS)]; exec(head)                # data, models, training and scoring helpers
N = int(os.environ.get("JG_N", 300))
def fact(names, rel, tpl, r): a, b = r.choice(len(names), 2, replace=False); return REL[rel][tpl].format(A=names[a], B=names[b]), (names[a], names[b])
@torch.no_grad()
def gap_words(lm, J, t):
    """per word: (start, end, word, gap) with gap = z(byte surprise) - z(JEPA surprise) inside the sentence"""
    X = torch.tensor([enc(t)]); sp = spans(t)
    Xi = torch.cat([torch.full((1, 1), PAD), X[:, :-1]], 1); nll = F.cross_entropy(lm(Xi)[0], X[0], reduction='none')
    tg = J.target(X)[0]; Xm = X.repeat(len(sp), 1)
    for k, (a, b, w) in enumerate(sp): Xm[k, a:b] = MASK
    pr = J.predict(Xm)
    bs = np.array([float(nll[a:b].mean()) for a, b, w in sp]); js = np.array([float(((pr[k, a:b] - tg[a:b]) ** 2).mean()) for k, (a, b, w) in enumerate(sp)])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    return sp, z(bs) - z(js), bs
@torch.no_grad()
def traces(lm, J, t, names, r):
    sp, gap, bs = gap_words(lm, J, t); X = torch.tensor([enc(t)]); keep = (X[0] != PAD)
    gold = [i for i, (a, b, w) in enumerate(sp) if w in names]
    pick = {'MASK-GOLD': gold, 'MASK-JEPA': list(np.argsort(-gap)[:2]), 'MASK-BYTE': list(np.argsort(-bs)[:2]), 'MASK-RAND': list(r.choice(len(sp), 2, replace=False))}
    out = {}; h = J.enc(X, kpm=(X == PAD))[0]; out['RAW'] = h[keep].mean(0)
    dm = keep.clone()
    for i in gold: dm[sp[i][0]:sp[i][1]] = False
    out['DROP'] = h[dm].mean(0)
    for k, ix in pick.items():
        Xm = X.clone()
        for i in ix: Xm[0, sp[i][0]:sp[i][1]] = MASK
        out[k] = J.enc(Xm, kpm=(Xm == PAD))[0][keep].mean(0)
    det = {k: len(set(ix) & set(gold)) / max(1, len(gold)) for k, ix in pick.items() if k != 'MASK-GOLD'}
    return out, det
KINDS = ['RAW', 'DROP', 'MASK-GOLD', 'MASK-JEPA', 'MASK-BYTE', 'MASK-RAND']
def build(lm, J, names, n, seed):
    """n facts and their traces: for each, a twin with OTHER names (same relation and wording) and a twin with the SAME names (other relation)"""
    r = np.random.RandomState(seed); R = []
    for _ in range(n):
        rel, tpl = r.randint(8), r.randint(4); t, nm = fact(names, rel, tpl, r)
        t2, nm2 = fact(names, rel, tpl, r); rel3 = (rel + 1 + r.randint(7)) % 8; t3 = REL[rel3][tpl].format(A=nm[0], B=nm[1])
        R.append(dict(rel=rel, tpl=tpl, names=set(nm), tr=traces(lm, J, t, set(nm), r), other=traces(lm, J, t2, set(nm2), r)[0], same=traces(lm, J, t3, set(nm), r)[0]))
    return R
def probe(Xtr, ytr, Xte, yte, ncls, steps=400):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; W = nn.Linear(Xtr.shape[1], ncls); op = torch.optim.Adam(W.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps): l = F.cross_entropy(W((Xtr - mu) / sd), ytr); op.zero_grad(); l.backward(); op.step()
    with torch.no_grad(): return float((W((Xte - mu) / sd).argmax(1) == yte).float().mean())
def mknn(A, B, k=10):
    A, B = F.normalize(A, dim=-1), F.normalize(B, dim=-1); n = A.shape[0]
    SA = A @ A.T; SB = B @ B.T; SA.fill_diagonal_(-9); SB.fill_diagonal_(-9)
    ia, ib = SA.topk(k, 1).indices, SB.topk(k, 1).indices
    return float(np.mean([len(set(ia[i].tolist()) & set(ib[i].tolist())) / k for i in range(n)]))
print(f"TOY GENERIC JEPA TRACE, seed {SEED}, {STEPS} steps; {len(TRN_N)} training names, {len(TST_N)} test names", flush=True)
lm = train_lm(STEPS)
JA = train_jepa('ema', STEPS)
torch.manual_seed(SEED + 1000); rs.seed(SEED + 1000); JB = train_jepa('ema', STEPS)
Rtr = build(lm, JA, TRN_N, N, 11); Rte = build(lm, JA, TST_N, N, 12); RteB = build(lm, JB, TST_N, N, 12)   # same facts for A and B (seed 12)
det = {k: np.mean([x['tr'][1][k] for x in Rte]) for k in ('MASK-JEPA', 'MASK-BYTE', 'MASK-RAND')}
print(f"  detection (share of the 2 true names among the 2 words masked), test names: JEPA gap {det['MASK-JEPA']:.2f} | byte surprise {det['MASK-BYTE']:.2f} | random {det['MASK-RAND']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
cos = lambda a, b: float(F.cosine_similarity(a, b, dim=0))
for kd in KINDS:
    s_other = [cos(x['tr'][0][kd], x['other'][kd]) for x in Rte]; s_same = [cos(x['tr'][0][kd], x['same'][kd]) for x in Rte]
    inv = auc(s_other, s_same)
    Xtr = torch.stack([x['tr'][0][kd] for x in Rtr]); Xte = torch.stack([x['tr'][0][kd] for x in Rte])
    pr = probe(Xtr, torch.tensor([x['rel'] for x in Rtr]), Xte, torch.tensor([x['rel'] for x in Rte]), 8)
    pt = probe(Xtr, torch.tensor([x['tpl'] for x in Rtr]), Xte, torch.tensor([x['tpl'] for x in Rte]), 4)
    sh, ns = [], []                                                        # identity left: pairs sharing a name vs not (different relations)
    for i in range(len(Rte)):
        for j in range(i + 1, min(len(Rte), i + 40)):
            if Rte[i]['rel'] == Rte[j]['rel']: continue
            (sh if Rte[i]['names'] & Rte[j]['names'] else ns).append(cos(Xte[i], Xte[j]))
    for x in Rte:                                                          # plus each fact with its same-names twin (different relation)
        sh.append(cos(x['tr'][0][kd], x['same'][kd]))
    idl = auc(sh, ns)
    al = mknn(Xte, torch.stack([x['tr'][0][kd] for x in RteB]))
    print(f"  {kd:10s} | 1 NAME INVARIANCE {inv:.2f} | 2 STRUCTURE relation {pr:.2f} (chance 0.12), wording {pt:.2f} (chance 0.25) | 3 IDENTITY LEFT {idl:.2f} (0.5 = none) | "
          f"4 CONVERGENCE seed A vs B {al:.3f} (chance {10/(N-1):.3f})  ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE {time.time()-t0:.0f}s")
