"""LEARNED SUBJECT KEYS AND KEY DRIFT (16 Nov 2026; tools/learned_keys.py) - offline, no effect on the package. Queue item 28: can the NAME
unit's exact-string key (the package's subject hash) be replaced by a LEARNED key - a code of the trunk's state at the name - and how do the
published answers to key drift compare when the trunk keeps learning? Same keyed continuation memory and feedback gate as tools/keyed_units.py
(causal NAME firings: after every capitalised word, the run so far), Re-DocRED text, memory from the first documents, scored prequentially on
the last LK_EVAL. DRIFT: the memory is WRITTEN with the earlier trunk (TG_CKPT) and READ with the later one (LK_NEW; if absent, the old trunk
continued LK_FT steps on another domain's text, LK_DOM - a continual-learning drift); every arm is scored with the later trunk's predictions, only the keys differ. A learned key = the nearest of LK_K codes
(k-means on name-end states) of the state that has read the name.
  HASH     key = the name string (the package today; drift-free, no aliases)
  REKEY    learned key, the stored entries RE-KEYED with the new trunk (sleep re-keying: codebook refit, every stored key recomputed)
  STALE    learned key, stored with the old trunk, queried with the new one through the old codebook (no fix: drift breaks addressing)
  QDC      as STALE, the query corrected by the measured drift (mean new - old state on the current documents; Query Drift Compensation)
  FROZEN   a frozen key encoder: stored AND queried with the old trunk's states (the Discrete Key-Value Bottleneck recipe)
  HASH+REKEY / HASH+FROZEN   the exact key plus the learned one as a second unit (the gate learns how far to trust each)
Readouts: bits saved per 1,000 bytes (all, and after warm-up); COLLISIONS (codes shared by several distinct names in the memory); ALIAS hits
(positions a learned key predicts where the exact name has nothing stored) and the bits saved there.
Usage: TG_D=192 TG_CKPT=<old trunk> LK_NEW=<new trunk> python3 tools/learned_keys.py   env LK_EVAL (300), LK_K (1024), RU_CONT (24), RU_LR (0.05), LU_CAP (64)"""
import os, sys, math, time, collections
import numpy as np
os.environ['RU_CAUSAL'] = '1'
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keyed_units.py')).read()
g = {'__file__': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keyed_units.py')}
exec(compile(src[:src.index("if CAUSAL:\n    B3")], 'keyed_units', 'exec'), g)
import torch, torch.nn.functional as F
old, run, P, docs, build_causal, T0 = g['lm'], g['run'], g['P'], g['docs'], g['build_causal'], g['T0']
tg = g['g']; new = tg['LM']()
if os.path.exists(os.environ['LK_NEW']): new.load_state_dict(torch.load(os.environ['LK_NEW']))
else:                                                                          # LK_NEW absent: the OLD trunk continued on another domain (a continual-
    import copy                                                                #   learning drift): LK_FT steps (1500) on the mini-Pile's arXiv text, saved
    new = copy.deepcopy(old); new.train(); dom = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/' + os.environ.get('LK_DOM', 'arxiv') + '.bin', 'rb').read()
    arr = np.frombuffer(dom, np.uint8).astype(np.int64); op = torch.optim.Adam(new.parameters(), lr=float(os.environ.get('LK_LR', 3e-4))); rs0 = np.random.RandomState(1)
    for st_ in range(int(os.environ.get('LK_FT', 1500))):
        ix = rs0.randint(0, len(arr) - 129, 16); X = torch.tensor(np.stack([arr[i:i + 128] for i in ix])); Xi = torch.cat([torch.full((16, 1), tg['PAD']), X[:, :-1]], 1)
        loss = F.cross_entropy(new.o(new.t(Xi)).reshape(-1, new.o.out_features), X.reshape(-1)); op.zero_grad(); loss.backward(); op.step()
        if (st_ + 1) % 500 == 0: P(f"   drift fine-tune on {os.environ.get('LK_DOM', 'arxiv')}: step {st_ + 1}, loss {loss.item() / math.log(2):.3f} bits  ({time.time() - T0:.0f}s)")
    torch.save(new.state_dict(), os.environ['LK_NEW'])
new.eval()
NE = int(os.environ.get('LK_EVAL', 300)); CL = int(os.environ.get('RU_CONT', 24)); K = int(os.environ.get('LK_K', 1024)); CAP = int(os.environ.get('LU_CAP', 64))
LR = float(os.environ.get('RU_LR', 0.05)); torch.set_num_threads(2); ev0 = len(docs) - NE
P(f"documents {len(docs)}: memory from the first {ev0}, scored on the last {NE} | old trunk {os.environ.get('TG_CKPT')} new {os.environ['LK_NEW']} | {K} codes")
def names(d):
    b, fh, st = build_causal(d); return b, [f for f in st if f[0] == 'NAME'], [f for f in fh if f[0] == 'NAME']
# 1. states at name ends from both trunks, memory documents
MEMD = []; So, Sn = [], []
for i, d in enumerate(docs[:ev0]):
    b, st, _ = names(d)
    if not st: MEMD.append((b, st, None, None)); continue
    _, Ho = run(old, b); _, Hn = run(new, b); ps = [p for _, _, p in st]
    ho, hn = Ho[ps], Hn[ps]; MEMD.append((b, st, ho, hn)); So.append(ho); Sn.append(hn)
    if (i + 1) % 500 == 0: P(f"   states: {i + 1} documents  ({time.time() - T0:.0f}s)")
So, Sn = np.concatenate(So), np.concatenate(Sn); DRIFT = (Sn - So).mean(0)
P(f"{len(So)} name-end states | drift: mean |new - old| {np.linalg.norm(Sn - So, axis=1).mean():.2f} vs state norm {np.linalg.norm(Sn, axis=1).mean():.2f}, mean shift norm {np.linalg.norm(DRIFT):.2f}  ({time.time() - T0:.0f}s)")
from sklearn.cluster import MiniBatchKMeans
rs = np.random.RandomState(0); sub = rs.choice(len(So), min(len(So), 120000), replace=False)
KO = MiniBatchKMeans(K, random_state=0, n_init=1, batch_size=8192).fit(So[sub]); KN = MiniBatchKMeans(K, random_state=0, n_init=1, batch_size=8192).fit(Sn[sub])
P(f"codebooks fitted  ({time.time() - T0:.0f}s)")
mem = collections.defaultdict(lambda: collections.deque(maxlen=CAP)); who = collections.defaultdict(set)
def store(b, st, codes):
    """codes: arm -> the code of each firing's state as that arm keeps it"""
    for j, (_, name, p) in enumerate(st):
        c = b[p:p + CL]; mem[('HASH', name)].append(c)
        for arm, cc in codes.items():
            mem[(arm, int(cc[j]))].append(c)
            if arm in ('REKEY', 'FROZEN'): who[(arm, int(cc[j]))].add(name)
for b, st, ho, hn in MEMD:                                                     # written in the past with the OLD trunk (REKEY: re-keyed at sleep)
    if st: co = KO.predict(ho); store(b, st, {'REKEY': KN.predict(hn), 'STALE': co, 'QDC': co, 'FROZEN': co})
del MEMD
for arm in ('REKEY', 'FROZEN'):
    ks = [k for k in who if k[0] == arm]; P(f"COLLISIONS {arm}: {len(ks)} codes used, mean distinct names per code {np.mean([len(who[k]) for k in ks]):.1f}, codes holding one name {np.mean([len(who[k]) == 1 for k in ks]):.2f}")
def raw(b, fs):
    best = {}
    for u, k, p in fs:
        conts = mem.get((u, k))
        if not conts: continue
        for off in range(1, min(CL, len(b) - p)):
            seen = b[p:p + off]; cons = [c for c in conts if len(c) > off and c[:off] == seen]
            if not cons: break
            n = len(cons); hit = sum(1 for c in cons if c[off] == b[p + off]); pm = (hit + 1.0 / 256) / (n + 1.0); lam = n / (n + 2.0)
            q = best.setdefault(p + off, {}).get(u)
            if q is None or lam > q[0]: best[p + off][u] = (lam, pm, n, off)
    return best
ARMS = {a: {a} for a in ('HASH', 'REKEY', 'STALE', 'QDC', 'FROZEN')}; ARMS['HASH+REKEY'] = {'HASH', 'REKEY'}; ARMS['HASH+FROZEN'] = {'HASH', 'FROZEN'}
nbk = lambda n: 0 if n <= 1 else 1 if n == 2 else 2 if n <= 4 else 3 if n <= 8 else 4
obk = lambda o: 0 if o == 0 else 1 if o <= 3 else 2 if o <= 11 else 3
ebk = lambda e: 0 if e < 0.5 else 1 if e < 1.5 else 2 if e < 3 else 3
gate = {a: {} for a in ARMS}; gres = {a: [0.0, 0.0] for a in ARMS}; W0 = ev0 + NE // 3; nb = nb_late = 0; bb = bb_late = 0.0
alias = collections.Counter(); alias_bits = collections.Counter()
for di in range(ev0, len(docs)):
    d = docs[di]; b, st, fh = names(d)
    S, Hn = run(new, b); pt = 2.0 ** (-S.astype(np.float64))
    with torch.no_grad(): lpo = F.log_softmax(new.o(torch.tensor(Hn)), -1); ent = (-(lpo.exp() * lpo).sum(-1) / math.log(2)).numpy()
    late = di >= W0; nb += len(b); bb += float(S.sum())
    if late: nb_late += len(b); bb_late += float(S.sum())
    fs = []
    if fh:
        _, Ho = run(old, b); ps = [p for _, _, p in fh]
        cn = KN.predict(Hn[ps]); co = KO.predict(Ho[ps]); cs = KO.predict(Hn[ps]); cq = KO.predict(Hn[ps] - DRIFT)
        for j, (_, name, p) in enumerate(fh):
            fs += [('HASH', name, p), ('REKEY', int(cn[j]), p), ('STALE', int(cs[j]), p), ('QDC', int(cq[j]), p), ('FROZEN', int(co[j]), p)]
    R = raw(b, fs)
    for a, us in ARMS.items():
        G = gate[a]; gs = 0.0
        for pos, dd in sorted(R.items()):
            dd = {u: v for u, v in dd.items() if u in us}
            if not dd: continue
            ks = [(u, nbk(v[2]), obk(v[3]), ebk(ent[pos])) for u, v in dd.items()]
            lg = np.array([0.0] + [G.setdefault(k, math.log(max(dd[k[0]][2], 1) / 2.0)) for k in ks]); w = np.exp(lg - lg.max()); w /= w.sum()
            ps_ = np.array([pt[pos]] + [dd[k[0]][1] for k in ks]); p = float((w * ps_).sum()); sv = S[pos] + math.log2(max(p, 1e-12)); gs += sv
            if len(us) == 1 and 'HASH' not in R[pos] and a != 'HASH': alias[a] += 1; alias_bits[a] += sv
            for j, k in enumerate(ks): G[k] += LR * w[j + 1] * (ps_[j + 1] - p) / max(p, 1e-12)
        gres[a][0] += gs
        if late: gres[a][1] += gs
    if st:                                                                     # the scored document joins the memory (keys as each arm keeps them)
        _, Ho2 = run(old, b); ps = [p for _, _, p in st]; hn = Hn[ps]          #   new writes: each arm's own current encoder
        store(b, st, {'REKEY': KN.predict(hn), 'STALE': KO.predict(hn), 'QDC': KO.predict(hn - DRIFT), 'FROZEN': KO.predict(Ho2[ps])})
    if (di - ev0 + 1) % 50 == 0: P(f"   {di - ev0 + 1} documents scored  ({time.time() - T0:.0f}s) | " + " | ".join(f"{a} {1000 * v[0] / nb:.2f}" for a, v in gres.items()))
P(f"SCORED {NE} documents, {nb} bytes; new trunk alone {bb / nb:.3f} bits per byte")
P("BITS SAVED per 1,000 bytes with the feedback gate (all | after warm-up) | ALIAS positions (a learned key predicts, the exact name has nothing) and bits saved there:")
for a, v in gres.items(): P(f"   {a:12s} {1000 * v[0] / nb:7.2f} | {1000 * v[1] / nb_late:7.2f}" + (f" | alias positions {alias[a]}, saved {alias_bits[a]:+.0f} bits" if a in alias else ''))
P(f"DONE ({time.time() - T0:.0f}s)")
