"""24 Nov: CAN FEEDBACK REPLACE BACKPROPAGATION FOR THE MEMORY'S PREDICTOR? (offline, synthetic; the five-level world of tools/jepa_store.py)

A fact question 'X lives in' (and 'C lies in', 'R is part of', 'K is in') is encoded by a FIXED feature map (hashed character trigrams of the
question, L2-normalised; no learning) and the predictor must output the answer's code (a fixed random unit vector per name); the answer is the
nearest name code. Facts are read in wording 0; PARAPHRASE recall asks wording 1 (never read). Same budget for every arm (AL_EPOCHS passes):
  BP     - a two-layer predictor (tanh hidden) trained by BACKPROPAGATION (MSE to the answer code; Adam) - the reference;
  PC     - the same network trained by PREDICTIVE CODING (Whittington & Bogacz 2017): the output clamped to the answer code, the hidden
           state relaxed to minimise the prediction errors, then every weight updated LOCALLY from its own layer's error (no backward pass);
  ELM    - a FIXED random hidden layer (reservoir), only the readout learned by the local delta rule (LMS);
  HEBB   - a one-shot HEBBIAN associative memory (W += code x features), then SLEEP FEEDBACK: each sleep re-asks every read fact and corrects
           only where the recall is wrong (local error-driven correction, the sleep loop's 'contradicted' signal), no gradients anywhere.
Readouts per arm: one-hop recall on read facts, PARAPHRASE recall (wording 1), TWO HOPS by its own hops (lives -> city_in), THREE HOPS,
NEVER-READ names: margin AUC (does its confidence separate read from never-read heads), and wall time.
Usage: python3 tools/bpfree_sim.py  (env AL_EPOCHS 30, BF_SEEDS 0,1, BF_DIM 2048, BF_HID 1024)"""
import os, sys, time, zlib
import numpy as np, torch, torch.nn.functional as F

torch.set_num_threads(2); T0 = time.time(); P = lambda s: print(s, flush=True)
EP = int(os.environ.get('AL_EPOCHS', 30)); SEEDS = [int(x) for x in os.environ.get('BF_SEEDS', '0,1').split(',')]
DF = int(os.environ.get('BF_DIM', 2048)); H = int(os.environ.get('BF_HID', 1024)); DC = 256
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jepa_store.py')).read(); cut = src.index('PRE1 = sorted(')


def world(seed):
    g = {'__name__': 'bf'}; os.environ['LS_SEED'] = str(seed); exec(compile(src[:cut], 'jepa_store_world', 'exec'), g); return g


def feats(qs):
    X = np.zeros((len(qs), DF), dtype=np.float32)
    for i, q in enumerate(qs):
        t = f"  {q}  "
        for j in range(len(t) - 2): X[i, zlib.crc32(t[j:j + 3].encode()) % DF] += 1.0
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return torch.tensor(X)


def run(seed):
    g = world(seed); W, FACTS, CIT, REG, CTY, NOV = g['W'], g['FACTS'], g['CIT'], g['REG'], g['CTY'], g['NOV']
    names = sorted({x for fs in FACTS.values() for p in fs for x in p} | set(NOV)); NI = {n: i for i, n in enumerate(names)}
    rs = np.random.RandomState(seed); CODE = F.normalize(torch.tensor(rs.randn(len(names), DC).astype(np.float32)), dim=-1)
    ans_pool = {rel: sorted({b for a, b in FACTS[rel]}) for rel in FACTS}
    Q = {rel: (lambda t: (lambda a: t.split('{b}')[0].format(a=a).strip()))(W[rel][0]) for rel in W}
    Qp = {rel: (lambda t: (lambda a: t.split('{b}')[0].format(a=a).strip()))(W[rel][1]) for rel in W}
    tr = [(Q[rel](a), b, rel) for rel, fs in FACTS.items() for a, b in fs]; trp = [(Qp[rel](a), b, rel) for rel, fs in FACTS.items() for a, b in fs]
    Xt = feats([q for q, b, r in tr]); Yt = CODE[torch.tensor([NI[b] for q, b, r in tr])]; Xp = feats([q for q, b, r in trp])

    def decode(Y, rel):                                                         # nearest answer code among the relation's possible answers
        pool = ans_pool[rel]; S = Y @ CODE[torch.tensor([NI[n] for n in pool])].T; t2 = S.topk(2, -1)
        return [pool[j] for j in t2.indices[:, 0].tolist()], (t2.values[:, 0] - t2.values[:, 1]).tolist()

    def evaluate(f):
        R = {}
        right = []
        for rel in FACTS:
            ix = [i for i, x in enumerate(tr) if x[2] == rel]; o, _ = decode(f(Xt[ix]), rel); right += [o_ == tr[i][1] for o_, i in zip(o, ix)]
        R['one-hop'] = np.mean(right); rp = []
        for rel in FACTS:
            ix = [i for i, x in enumerate(trp) if x[2] == rel]; o, _ = decode(f(Xp[ix]), rel); rp += [o_ == trp[i][1] for o_, i in zip(o, ix)]
        R['paraphrase'] = np.mean(rp)
        peo = g['HEP'] + g['TRP'][:200]; c1, m1 = decode(f(feats([Q['lives'](a) for a, ci in peo])), 'lives')
        r2, _ = decode(f(feats([Q['city_in'](c) for c in c1])), 'city_in'); R['two hops'] = np.mean([x == REG[CIT[ci][1]][0] for x, (a, ci) in zip(r2, peo)])
        k3, _ = decode(f(feats([Q['region_of'](r) for r in r2])), 'region_of'); R['three hops'] = np.mean([x == CTY[REG[CIT[ci][1]][1]][0] for x, (a, ci) in zip(k3, peo)])
        _, mn = decode(f(feats([Q['lives'](a) for a in NOV])), 'lives'); mr = [m for x, m, (a, ci) in zip(c1, m1, peo) if x == CIT[ci][0]]
        R['never-read AUC'] = np.mean([[float(a > b) + 0.5 * (a == b) for b in mn] for a in mr]) if mr and mn else float('nan')
        return R

    out = {}
    # BP: two-layer, backpropagation
    torch.manual_seed(seed); W1 = (torch.randn(H, DF) / DF ** 0.5).requires_grad_(); W2 = (torch.randn(DC, H) / H ** 0.5).requires_grad_()
    opt = torch.optim.Adam([W1, W2], lr=1e-3); t = time.time()
    for ep in range(EP):
        for b in torch.randperm(len(tr)).split(64):
            y = torch.tanh(Xt[b] @ W1.T) @ W2.T; l = ((y - Yt[b]) ** 2).sum(-1).mean(); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad(): out['BP'] = evaluate(lambda X: torch.tanh(X @ W1.T) @ W2.T); out['BP']['sec'] = time.time() - t
    # PC: the same network, predictive coding (local updates, no backward pass)
    torch.manual_seed(seed); V1 = torch.randn(H, DF) / DF ** 0.5; V2 = torch.randn(DC, H) / H ** 0.5; lr, eta, TI = 0.05, 0.1, 20; t = time.time()
    for ep in range(EP):
        for b in torch.randperm(len(tr)).split(64):
            x0 = Xt[b]; mu1 = x0 @ V1.T; x1 = mu1.clone(); x2 = Yt[b]
            for _ in range(TI):                                                 # relax the hidden state (output clamped to the answer code)
                e1 = x1 - mu1; e2 = x2 - torch.tanh(x1) @ V2.T
                x1 = x1 + eta * (-e1 + (1 - torch.tanh(x1) ** 2) * (e2 @ V2))
            e1 = x1 - mu1; e2 = x2 - torch.tanh(x1) @ V2.T
            V2 += lr * e2.T @ torch.tanh(x1) / len(b); V1 += lr * e1.T @ x0 / len(b)   # LOCAL: each layer from its own error and its own input
    out['PC'] = evaluate(lambda X: torch.tanh(X @ V1.T) @ V2.T); out['PC']['sec'] = time.time() - t
    # ELM: fixed random hidden layer, readout by the local delta rule
    torch.manual_seed(seed); R1 = torch.randn(H, DF) / DF ** 0.5; U = torch.zeros(DC, H); t = time.time()
    for ep in range(EP):
        for b in torch.randperm(len(tr)).split(64):
            h = torch.tanh(Xt[b] @ R1.T); U += 0.5 * (Yt[b] - h @ U.T).T @ h / len(b)
    out['ELM'] = evaluate(lambda X: torch.tanh(X @ R1.T) @ U.T); out['ELM']['sec'] = time.time() - t
    # HEBB: one-shot Hebbian association, then sleep feedback on wrong recalls only
    Mh = Yt.T @ Xt; t = time.time(); fixed = []
    for ep in range(EP):
        Y = Xt @ Mh.T; wrong = []
        for rel in FACTS:
            ix = [i for i, x in enumerate(tr) if x[2] == rel]; o, _ = decode(Y[ix], rel); wrong += [i for o_, i in zip(o, ix) if o_ != tr[i][1]]
        fixed.append(len(wrong))
        if not wrong: break
        w = torch.tensor(wrong); Mh += 0.5 * (Yt[w] - Y[w]).T @ Xt[w]        #   the sleep's 'contradicted' signal: correct only what it recalls wrongly
    out['HEBB'] = evaluate(lambda X: X @ Mh.T); out['HEBB']['sec'] = time.time() - t; out['HEBB']['wrong per sleep'] = fixed[:6]
    return out


P(f"CAN FEEDBACK REPLACE BACKPROPAGATION? fixed trigram features ({DF}), predictor -> answer code ({DC}); {EP} passes; seeds {SEEDS}")
ALL = {}
for s in SEEDS:
    o = run(s); ALL[s] = o
    for k, v in o.items():
        P(f"   seed {s} {k:4s}: " + ' | '.join(f"{a} {b:.2f}" if isinstance(b, float) else f"{a} {b}" for a, b in v.items()) + f"  ({time.time() - T0:.0f}s)")
P("SUMMARY (mean over seeds): " + ' || '.join(f"{k}: one-hop {np.mean([ALL[s][k]['one-hop'] for s in SEEDS]):.2f}, paraphrase {np.mean([ALL[s][k]['paraphrase'] for s in SEEDS]):.2f}, "
                                              f"two hops {np.mean([ALL[s][k]['two hops'] for s in SEEDS]):.2f}, three {np.mean([ALL[s][k]['three hops'] for s in SEEDS]):.2f}, "
                                              f"AUC {np.mean([ALL[s][k]['never-read AUC'] for s in SEEDS]):.2f}" for k in ('BP', 'PC', 'ELM', 'HEBB')))
P(f"DONE ({time.time() - T0:.0f}s)")
