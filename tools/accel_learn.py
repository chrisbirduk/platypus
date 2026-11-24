"""24 Nov: ACCELERATED LEARNING. Does sleep inference let the model know more facts per fact read?

A family world, learned family by family (continual, with dream protection): each family is a root, 3 mids (parent = root) and 9 leaves
(parent = a mid), so 12 PREMISES 'X has parent Y.' and 9 COMPOSITES 'L has grandparent R.'. While reading a family the model sees its 12
premises plus 2 TAUGHT composites (so the composite relation exists for both arms), READ r times (one gradient step per reading). The other
7 composites are HELD BACK: never read.

SLEEP after each family (same number of steps in both arms) trains only on the model's OWN RECALL (dream protection, never the read text):
every remembered premise and taught composite whose answer passes the margin gate (SELF-CALIBRATED: the 25% quantile of
the margins of its right recalls of the family it just read, at least AL_GATE; closed when it recalls fewer than 4). Arm ON adds STEP-BY-STEP INFERENCE: for every non-root
entity it answers hop 1 (parent) and hop 2 (the parent of its own answer); if both pass the gate it WRITES 'X has grandparent Z.' into the
sleep (re-derived from scratch each sleep, so a write that no longer holds is dropped). Arm OFF has no writes.

Measured, by readings r and arm:
  KNOWN FACTS per family (premises right + composites right, of 21) and FACTS KNOWN PER FACT READ (of 14 read);
  premise recall, HELD-BACK composite accuracy (direct from the weights, 'L has grandparent ' -> rank among all parents), write precision;
  LEARNING TO LEARN: right after reading a family (before its sleep), its premise recall and its held-back composites answered directly,
  early families vs late families (in-weights composition learned from earlier writes would show here).
Answers are ranked among all 4*NF parents (roots and mids) by the summed log-prob of ' Name.'.
Usage: python3 tools/accel_learn.py   env AL_READS (default 2,4,8,16), AL_ARMS (off,on), AL_SEEDS (1,2), AL_FAM (8), AL_SLEEP (60 steps),
       AL_GATE (margin, nats, 1.0), AL_LR (3e-3), AL_DIM (128), AL_LAYERS (3)"""
import os, sys, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(2); T0 = time.time(); P = lambda s: print(s, flush=True)
E = os.environ.get
READS = [int(x) for x in E('AL_READS', '4,8,16,32').split(',')]; ARMS = E('AL_ARMS', 'off,on').split(',')
SEEDS = [int(x) for x in E('AL_SEEDS', '1,2').split(',')]; NF = int(E('AL_FAM', 8)); SLEEP = int(E('AL_SLEEP', 60))
GATE = float(E('AL_GATE', 1.0)); LR = float(E('AL_LR', 3e-3)); D = int(E('AL_DIM', 128)); NL = int(E('AL_LAYERS', 3)); BS = 32; TMAX = 64
SYL = ['ka', 'zo', 'bel', 'qui', 'ra', 'vin', 'tev', 'mon', 'dra', 'sal', 'nor', 'lo', 'pel', 'gal', 'bri', 'is', 'ul', 'tar', 'en', 'os']


def world(seed):
    rng = random.Random(1000 + seed); seen = set(); nm = []
    while len(nm) < 13 * NF:
        s = ''.join(rng.choice(SYL) for _ in range(rng.choice((2, 3)))).capitalize()
        if s not in seen: seen.add(s); nm.append(s)
    par, fam, roots, mids, leaves, taught, held = {}, {}, [], [], [], [], []
    for f in range(NF):
        R, M, Lf = nm[13 * f], nm[13 * f + 1:13 * f + 4], nm[13 * f + 4:13 * f + 13]
        roots.append(R); mids += M; leaves += Lf
        for m in M: par[m] = R
        for i, l in enumerate(Lf): par[l] = M[i // 3]
        for x in [R] + M + Lf: fam[x] = f
        t = rng.sample(Lf, 2); taught += t; held += [l for l in Lf if l not in t]
    return par, fam, roots, mids, leaves, set(taught), held


class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(256, D); s.pos = nn.Embedding(TMAX, D)
        s.blk = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 4 * D, dropout=0.0, batch_first=True, norm_first=True), NL)
        s.ln = nn.LayerNorm(D); s.out = nn.Linear(D, 256)

    def forward(s, x):
        T = x.shape[1]; h = s.emb(x) + s.pos(torch.arange(T))[None]
        h = s.blk(h, mask=nn.Transformer.generate_square_subsequent_mask(T), is_causal=True)
        return s.out(s.ln(h))


def enc(ss):
    b = [list(s.encode()) for s in ss]; T = max(map(len, b)); x = torch.zeros(len(b), T, dtype=torch.long)
    for i, r in enumerate(b): x[i, :len(r)] = torch.tensor(r)
    return x


def step(M, opt, ss):
    x = enc(ss); lo = M(x[:, :-1]); y = x[:, 1:]
    loss = F.cross_entropy(lo.reshape(-1, 256), y.reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step(); return float(loss)


@torch.no_grad()
def hop(M, heads, rel, cands):
    """answers (argmax among cands) and margins (top1 - top2 summed log-prob of ' Name.') for 'X has <rel> '"""
    if not heads: return [], []
    pr = [f"{h} has {rel}" for h in heads]; seqs, cl = [], []
    for p in pr:
        for c in cands: seqs.append(p + f" {c}."); cl.append(len(c) + 2)
    S = np.zeros(len(seqs))
    for i in range(0, len(seqs), 768):
        ch = seqs[i:i + 768]; x = enc(ch); lp = F.log_softmax(M(x[:, :-1]), -1).gather(-1, x[:, 1:, None])[..., 0]
        for j, s in enumerate(ch):
            n = len(s.encode()); S[i + j] = float(lp[j, n - 1 - cl[i + j]:n - 1].sum())
    S = S.reshape(len(heads), len(cands)); o = np.argsort(-S, 1)
    return [cands[k] for k in o[:, 0]], list(S[np.arange(len(heads)), o[:, 0]] - S[np.arange(len(heads)), o[:, 1]])


def run(seed, arm, r):
    par, fam, roots, mids, leaves, taught, held = world(seed); C = roots + mids; gp = {l: par[par[l]] for l in leaves}
    torch.manual_seed(seed); random.seed(seed); M = GPT(); opt = torch.optim.AdamW(M.parameters(), lr=LR, weight_decay=0.0)
    rng = random.Random(seed); bank = []; out = {'wake_pre': [], 'wake_comp': [], 'writes': [], 'wright': []}
    for f in range(NF):
        kids = [x for x in mids + leaves if fam[x] == f]; tl = [l for l in taught if fam[l] == f]; hl = [l for l in held if fam[l] == f]
        new = [f"{x} has parent {par[x]}." for x in kids] + [f"{l} has grandparent {gp[l]}." for l in tl]
        for _ in range(r): step(M, opt, new + (rng.sample(bank, min(len(bank), len(new))) if bank else []))
        a, _ = hop(M, kids, 'parent', C); out['wake_pre'].append(np.mean([y == par[x] for x, y in zip(kids, a)]))
        a, _ = hop(M, hl, 'grandparent', C); out['wake_comp'].append(np.mean([y == gp[x] for x, y in zip(hl, a)]))
        # SLEEP: own recall only (dream protection), plus step-by-step writes in arm ON
        sk = [x for x in mids + leaves if fam[x] <= f]; st = [l for l in taught if fam[l] <= f]
        a, m = hop(M, sk, 'parent', C)                                          # SELF-CALIBRATED GATE: 25% quantile of the margins of its RIGHT recalls of what it
        rm = [g for x, y, g in zip(sk, a, m) if y == par[x] and fam[x] == f]   #   just read (a self-test on stored text, as the JEPA memory's gate); none -> closed
        th = max(GATE, float(np.quantile(rm, 0.25))) if len(rm) >= 4 else float('inf'); out.setdefault('th', []).append(th)
        bank = [f"{x} has parent {y}." for x, y, g in zip(sk, a, m) if g > th]
        a, m = hop(M, st, 'grandparent', C); bank += [f"{x} has grandparent {y}." for x, y, g in zip(st, a, m) if g > th]
        nw = rt = 0
        if arm == 'on':
            hx = [x for x in sk if x not in taught]; a1, m1 = hop(M, hx, 'parent', C); a2, m2 = hop(M, a1, 'parent', C)
            for x, y, g1, g2 in zip(hx, a2, m1, m2):
                if min(g1, g2) > th: bank.append(f"{x} has grandparent {y}."); nw += 1; rt += int(gp.get(x) == y)
        out['writes'].append(nw); out['wright'].append(rt)
        if not bank: bank = list(new)                                           # nothing recalled yet: the sleep rehearses the day (first family only in practice)
        for _ in range(SLEEP): step(M, opt, rng.choices(bank, k=BS))
    a, _ = hop(M, mids + leaves, 'parent', C); pre = np.mean([y == par[x] for x, y in zip(mids + leaves, a)])
    a, _ = hop(M, held, 'grandparent', C); comp = np.mean([y == gp[x] for x, y in zip(held, a)])
    a, _ = hop(M, sorted(taught), 'grandparent', C); tc = np.mean([y == gp[x] for x, y in zip(sorted(taught), a)])
    known = (pre * 12 + (comp * 7 + tc * 2)) * NF; nwr = sum(out['writes']); h = NF // 2
    res = dict(pre=pre, comp=comp, tc=tc, known_fam=known / NF, per_read=known / (14 * NF), writes=out['writes'][-1],
               wprec=(out['wright'][-1] / out['writes'][-1]) if out['writes'][-1] else float('nan'),
               wpre_early=np.mean(out['wake_pre'][:h]), wpre_late=np.mean(out['wake_pre'][h:]),
               wcomp_early=np.mean(out['wake_comp'][:h]), wcomp_late=np.mean(out['wake_comp'][h:]))
    P(f"   seed {seed} | {arm:3s} | reads {r:2d}: premises {pre:.2f} | HELD-BACK composites {comp:.2f} | taught composites {tc:.2f} | "
      f"KNOWN {res['known_fam']:.1f}/21 a family, {res['per_read']:.2f} per fact read | last sleep writes {res['writes']} (right {res['wprec']:.2f}) | "
      f"after reading, before sleep: premises early {res['wpre_early']:.2f} late {res['wpre_late']:.2f}, held-back composites early "
      f"{res['wcomp_early']:.2f} late {res['wcomp_late']:.2f}  ({time.time() - T0:.0f}s)")
    return res


P(f"ACCELERATED LEARNING: {NF} families x (12 premises + 2 taught composites read, 7 composites held back); readings {READS}; arms {ARMS}; "
  f"seeds {SEEDS}; sleep {SLEEP} steps on own recall; gate {GATE} nats; model {NL}x{D}, lr {LR}")
R = {}
for r in READS:
    for s in SEEDS:
        for arm in ARMS: R[(arm, r, s)] = run(s, arm, r)
P("\nSUMMARY (mean over seeds) | readings: arm -> KNOWN facts a family (of 21) / per fact read | premises | HELD-BACK composites (write precision) | "
  "learning to learn: premises after reading early -> late, held-back composites after reading early -> late")
for r in READS:
    for arm in ARMS:
        v = lambda k: np.nanmean([R[(arm, r, s)][k] for s in SEEDS])
        P(f"  reads {r:2d} {arm:3s}: KNOWN {v('known_fam'):5.1f} / {v('per_read'):.2f} | premises {v('pre'):.2f} | held-back {v('comp'):.2f} "
          f"(writes right {v('wprec'):.2f}) | premises {v('wpre_early'):.2f} -> {v('wpre_late'):.2f} | composites {v('wcomp_early'):.2f} -> {v('wcomp_late'):.2f}")
if 'on' in ARMS and 'off' in ARMS:
    off = {r: np.mean([R[('off', r, s)]['known_fam'] for s in SEEDS]) for r in READS}
    on = {r: np.mean([R[('on', r, s)]['known_fam'] for s in SEEDS]) for r in READS}
    for r in READS:
        cand = [q for q in READS if on[q] >= off[r]]
        P(f"  OFF at {r} readings knows {off[r]:.1f}; ON reaches that at " + (f"{min(cand)} readings ({r / min(cand):.1f}x fewer)" if cand else "no level tested"))
P(f"DONE ({time.time() - T0:.0f}s)")
