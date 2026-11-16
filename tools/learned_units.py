"""LEARNED UNIT TYPES (16 Nov 2026; tools/learned_units.py) - offline, no effect on the package. Queue item 27: can the hand-written keyed units
(NAME / DATE / NUMBER detectors of tools/keyed_units.py and platypus/inference/kunits.py) be replaced by GENERIC keys whose types are learned?
Same set-up as keyed_units.py with causal firings (RU_CAUSAL 1): Re-DocRED text the stand-in trunk never read, a keyed CONTINUATION memory
(at a firing the bytes that follow are stored under the key; when the key fires again, stored continuations consistent with the bytes read so
far predict the next byte), scored prequentially on the last LU_EVAL documents, every unit mixed with the trunk by the learned FEEDBACK GATE
(a softmax over the trunk and each firing unit, its logit per (unit, consistent continuations, offset, trunk entropy), one online step on
-log p after each byte is scored). Arms:
  HAND      NAME + DATE + NUMBER, the hand detectors (the package's keyed units)
  GENERIC   a firing at EVERY word end, key = the last 1 or 2 words of the sentence (no hand rule at all), gate per key length
  TYPED     the same firings, key = (TYPE, last 1 or 2 words), TYPE = the nearest of LU_K codes of the trunk's state at the firing (k-means on
            states of early memory documents, then FROZEN - the Discrete Key-Value Bottleneck recipe, so keys never drift); the gate is per
            (type, key length), so learning decides which types are worth keying on
  HAND+TYPED  both
Readouts: bits saved per 1,000 bytes (all scored documents, and after warm-up), and for each learned type its PURITY against the hand units
(share of its firings that end a capitalised word = a NAME firing, a year / month, a number, other) and the gate's learned trust in it.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/learned_units.py   env LU_EVAL (300), LU_K (16), RU_CONT (24), RU_LR (0.05), LU_CAP (64)"""
import os, sys, math, time, collections, re
import numpy as np
os.environ['RU_CAUSAL'] = '1'
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keyed_units.py')).read()
g = {'__file__': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keyed_units.py')}
exec(compile(src[:src.index("if CAUSAL:\n    B3")], 'keyed_units', 'exec'), g)
import torch, torch.nn.functional as F
lm, run, P, docs, build_causal, T0 = g['lm'], g['run'], g['P'], g['docs'], g['build_causal'], g['T0']
NOSP_B, NOSP_A, isyear, isnum, MONTHS = g['NOSP_B'], g['NOSP_A'], g['isyear'], g['isnum'], g['MONTHS']
NE = int(os.environ.get('LU_EVAL', 300)); CL = int(os.environ.get('RU_CONT', 24)); K = int(os.environ.get('LU_K', 16)); CAP = int(os.environ.get('LU_CAP', 64))
LR = float(os.environ.get('RU_LR', 0.05)); torch.set_num_threads(2)
def toks_of(d):
    out = b''; toks = []; prev = None
    for si, sent in enumerate(d['sents']):
        for ti, w in enumerate(sent):
            wb = w.encode('utf-8', 'replace')
            if out and not (w in NOSP_B or prev in NOSP_A): out += b' '
            toks.append((si, ti, len(out), len(out) + len(wb), w)); out += wb; prev = w
    return out, toks
def kind(t):
    w = t[4]
    if w[:1].isupper(): return 'NAME'
    if isyear(w) or w in MONTHS: return 'DATE'
    if isnum(w): return 'NUMBER'
    return 'other'
ev0 = len(docs) - NE
P(f"documents {len(docs)}: memory from the first {ev0}, scored on the last {NE}; {K} learned types; continuations per key capped at {CAP}")
# 1. the frozen type codebook: k-means on trunk states at word ends of the first 300 memory documents
from sklearn.cluster import MiniBatchKMeans
smp = []
for d in docs[:300]:
    b, tk = toks_of(d); _, H = run(lm, b)
    smp += [H[t[3]] for t in tk if t[3] < len(b)]
smp = np.array(smp); km = MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=4096).fit(smp)
P(f"codebook: {K} types from {len(smp)} word-end states  ({time.time() - T0:.0f}s)")
del smp
def firings(d, H, b, tk):
    """generic / typed firings at every word end: (unit, key, byte position); the type read from the state that has seen the word only"""
    out = []; ends = [t for t in tk if t[3] < len(b)]
    cl = km.predict(np.stack([H[t[3]] for t in ends])) if ends else []
    kinds = []
    for t, c in zip(ends, cl): kinds.append((int(c), kind(t)))
    for j, t in enumerate(ends):
        si = t[0]; w1 = t[4]; c = int(cl[j])
        prevw = ends[j - 1][4] if j > 0 and ends[j - 1][0] == si else None
        out.append(('W1', w1, t[3])); out.append((f'T1c{c}', f'{c}|{w1}', t[3]))
        if prevw is not None: out.append(('W2', prevw + ' ' + w1, t[3])); out.append((f'T2c{c}', f'{c}|{prevw} {w1}', t[3]))
    return out, kinds
mem = collections.defaultdict(lambda: collections.deque(maxlen=CAP))
def add(b, fs):
    for u, k, p in fs: mem[(u[:2] if u[0] == 'T' else u, k)].append(b[p:p + CL])
# 2. fill the memory from the memory documents (hand units: build_causal's store; generic / typed: every word end)
for i, d in enumerate(docs[:ev0]):
    b, fh, st = build_causal(d); _, tk = toks_of(d)
    add(b, [f for f in st if f[0] in ('NAME', 'DATE', 'NUMBER')])
    _, H = run(lm, b); fg, _ = firings(d, H, b, tk); add(b, fg)
    if (i + 1) % 500 == 0: P(f"   memory: {i + 1} documents  ({time.time() - T0:.0f}s)")
P(f"memory filled: {len(mem)} keys  ({time.time() - T0:.0f}s)")
def raw(b, fs):
    best = {}
    for u, k, p in fs:
        conts = mem.get((u[:2] if u[0] == 'T' else u, k))
        if not conts: continue
        for off in range(1, min(CL, len(b) - p)):
            seen = b[p:p + off]; cons = [c for c in conts if len(c) > off and c[:off] == seen]
            if not cons: break
            n = len(cons); hit = sum(1 for c in cons if c[off] == b[p + off])
            pm = (hit + 1.0 / 256) / (n + 1.0); lam = n / (n + 2.0)
            q = best.setdefault(p + off, {}).get(u)
            if q is None or lam > q[0]: best[p + off][u] = (lam, pm, n, off)
    return best
ARMS = {'HAND': lambda u: u in ('NAME', 'DATE', 'NUMBER'), 'GENERIC': lambda u: u in ('W1', 'W2'), 'TYPED': lambda u: u[0] == 'T',
        'HAND+TYPED': lambda u: u in ('NAME', 'DATE', 'NUMBER') or u[0] == 'T', 'HAND+GENERIC': lambda u: u in ('NAME', 'DATE', 'NUMBER', 'W1', 'W2')}
nbk = lambda n: 0 if n <= 1 else 1 if n == 2 else 2 if n <= 4 else 3 if n <= 8 else 4
obk = lambda o: 0 if o == 0 else 1 if o <= 3 else 2 if o <= 11 else 3
ebk = lambda e: 0 if e < 0.5 else 1 if e < 1.5 else 2 if e < 3 else 3
gate = {a: {} for a in ARMS}; gres = {a: [0.0, 0.0] for a in ARMS}; W0 = ev0 + NE // 3; nb = nb_late = 0; bb = bb_late = 0.0
pur = collections.defaultdict(collections.Counter); usef = collections.defaultdict(float)
# 3. prequential scoring
for di in range(ev0, len(docs)):
    d = docs[di]; b, fh, st = build_causal(d); _, tk = toks_of(d)
    S, H = run(lm, b); pt = 2.0 ** (-S.astype(np.float64))
    with torch.no_grad(): lpo = F.log_softmax(lm.o(torch.tensor(H)), -1); ent = (-(lpo.exp() * lpo).sum(-1) / math.log(2)).numpy()
    fg, kinds = firings(d, H, b, tk)
    for c, kd in kinds: pur[c][kd] += 1
    late = di >= W0; nb += len(b); bb += float(S.sum())
    if late: nb_late += len(b); bb_late += float(S.sum())
    R = raw(b, [f for f in fh if f[0] in ('NAME', 'DATE', 'NUMBER')] + fg)
    for a, inA in ARMS.items():
        G = gate[a]; gs = 0.0
        for pos, dd in sorted(R.items()):
            dd = {u: v for u, v in dd.items() if inA(u)}
            if not dd: continue
            ks = [(u, nbk(v[2]), obk(v[3]), ebk(ent[pos])) for u, v in dd.items()]
            lg = np.array([0.0] + [G.setdefault(k, math.log(max(dd[k[0]][2], 1) / 2.0)) for k in ks]); w = np.exp(lg - lg.max()); w /= w.sum()
            ps = np.array([pt[pos]] + [dd[k[0]][1] for k in ks]); p = float((w * ps).sum()); sv = S[pos] + math.log2(max(p, 1e-12)); gs += sv
            if a == 'TYPED':
                for j, k in enumerate(ks): usef[k[0]] += w[j + 1] * sv
            for j, k in enumerate(ks): G[k] += LR * w[j + 1] * (ps[j + 1] - p) / max(p, 1e-12)
        gres[a][0] += gs
        if late: gres[a][1] += gs
    add(b, [f for f in st if f[0] in ('NAME', 'DATE', 'NUMBER')] + fg)
    if (di - ev0 + 1) % 50 == 0:
        P(f"   {di - ev0 + 1} documents scored  ({time.time() - T0:.0f}s) | " + " | ".join(f"{a} {1000 * v[0] / nb:.2f}" for a, v in gres.items()))
P(f"SCORED {NE} documents, {nb} bytes; trunk alone {bb / nb:.3f} bits per byte")
P("BITS SAVED per 1,000 bytes with the feedback gate (all | after warm-up):")
for a, v in gres.items(): P(f"   {a:13s} {1000 * v[0] / nb:7.2f} | {1000 * v[1] / nb_late:7.2f}")
P("LEARNED TYPES (frozen codebook): firings at word ends, PURITY vs the hand units, the gate's mean learned trust and the bits the type's units saved (TYPED arm)")
for c in sorted(pur, key=lambda c: -sum(pur[c].values())):
    tot = sum(pur[c].values()); lt = [v for k, v in gate['TYPED'].items() if k[0] in (f'T1c{c}', f'T2c{c}')]
    P(f"   type {c:2d}: {tot:6d} firings | NAME {pur[c]['NAME'] / tot:.2f} DATE {pur[c]['DATE'] / tot:.2f} NUMBER {pur[c]['NUMBER'] / tot:.2f} other {pur[c]['other'] / tot:.2f} | "
      f"trust {np.mean(lt) if lt else float('nan'):+.2f} | saved {usef[f'T1c{c}'] + usef[f'T2c{c}']:+.0f} bits")
P(f"DONE ({time.time() - T0:.0f}s)")
