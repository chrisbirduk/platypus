"""A MEMORY THAT RETURNS KEYS: THREE ROUTES TO TWO-HOP COMPOSITION ON REAL TEXT (16 Nov 2026; tools/key_return.py) - offline, no effect on
the package. Queue item 32. Today the keyed memory returns BYTES (logits added to the trunk's), so a second hop must first spell the middle
name. Here every fact of Re-DocRED's training documents (3,053 Wikipedia documents, gold relations; 'read' = written to the memory) is a slot
(subject as it was written in the fact's evidence sentence, relation); relations use the gold relation id (the ideal wording key - the
package's learned JEPA wording key is near-perfect on the GPU: head loss 0.01 over 29 relations). CHAINS: in one document, A -r1-> B and
B -r2-> C are both facts, A -> C is never stated. Real text writes B under different surface forms in the two facts ('United States' / 'U.S.'),
which is where routes differ. NAME KEY = the stand-in trunk's state over a name's bytes (mean over the bytes, centred, unit length).
  DECODE     today's path, idealised: hop 1 by exact address, its best answer taken as the middle name (spelling assumed perfect),
             hop 2 by exact address with that string
  A POINTER  each slot records the answer names it was written with (counts); a read returns a DISTRIBUTION over them; hop 2 addresses
             with each of the top LK names (exact), C scores summed weighted by hop 1
  B LEARNED  addressed in key space: a read for (key, r) attends over every slot of relation r by cosine of its subject's name key (softmax,
             temperature KR_T); value = the weighted mean of the answers' name keys; soft decode = attention over all known name keys
             (top LK); hop 2 reads with those keys - aliases of B reach B's facts by similarity, and so do look-alike names (collisions)
  C HYBRID   A where the exact address exists, B as the fallback for any hop whose exact address has nothing written
Scored on chains (top-1 / top-5: the answer is any surface form of the gold C entity); also the share of chains whose B is written in
two different surface forms (ALIASED), and accuracy on aliased vs plain chains.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/key_return.py   env KR_T (0.05), KR_LK (5), KR_N (4000 chains max)"""
import os, sys, json, math, time, collections
import numpy as np
import torch
here = os.path.dirname(os.path.abspath(__file__)); sys.argv = [sys.argv[0], '6000', '5']
src = open(os.path.join(here, 'trunk_guides.py')).read(); g = {}
exec(compile(src[:src.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'), g)
lm, run, P = g['lm'], g['run'], g['P']; T0 = time.time(); torch.set_num_threads(2)
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
docs = json.load(open(SCR + '/bench_dl/train_revised.json'))[:int(os.environ.get('KR_DOCS', 100000))]
TAU = float(os.environ.get('KR_T', 0.05)); LK = int(os.environ.get('KR_LK', 5)); NMAX = int(os.environ.get('KR_N', 4000))
def surf(d, e, sents):
    """the entity's surface form in the fact's first evidence sentence (else its first mention)"""
    ms = d['vertexSet'][e]
    for s in sents:
        for m in ms:
            if m['sent_id'] == s: return m['name']
    return ms[0]['name']
facts = []; chains = []
for di, d in enumerate(docs):
    lab = d['labels']; stated = {(l['h'], l['t']) for l in lab} | {(l['t'], l['h']) for l in lab}
    fs = [(l['h'], l['r'], l['t'], surf(d, l['h'], l['evidence']), surf(d, l['t'], l['evidence'])) for l in lab]
    facts += [(hs, r, ts) for _, r, _, hs, ts in fs]
    by_h = collections.defaultdict(list)
    for f in fs: by_h[f[0]].append(f)
    for a, r1, b, as_, bs1 in fs:
        for _, r2, c, bs2, cs in by_h.get(b, ()):
            if c == a or (a, c) in stated: continue
            chains.append((as_, r1, bs1, r2, bs2, {m['name'] for m in d['vertexSet'][c]}))
rs = np.random.RandomState(0); rs.shuffle(chains); chains = chains[:NMAX]
P(f"{len(docs)} documents, {len(facts)} facts written, {len(chains)} chains tested (A -> C never stated; aliased B {np.mean([c[2] != c[4] for c in chains]):.2f})  ({time.time() - T0:.0f}s)")
# pointer memory (A): exact address -> answer counts
MEM = collections.defaultdict(collections.Counter)
for hs, r, ts in facts: MEM[(hs, r)][ts] += 1
# name keys (B)
names = sorted({x for hs, _, ts in facts for x in (hs, ts)} | {n for c in chains for n in c[5]}); NI = {n: i for i, n in enumerate(names)}
K = np.zeros((len(names), lm.o.in_features), np.float32)
for i, n in enumerate(names):
    b = (' ' + n).encode('utf-8', 'replace'); _, H = run(lm, b); K[i] = H[1:].mean(0)
    if (i + 1) % 10000 == 0: P(f"   name keys: {i + 1} of {len(names)}  ({time.time() - T0:.0f}s)")
K -= K.mean(0); K /= np.linalg.norm(K, axis=1, keepdims=True) + 1e-8; KT = torch.tensor(K)
P(f"{len(names)} name keys  ({time.time() - T0:.0f}s)")
RS = collections.defaultdict(list)                                             # relation -> (subject key rows, answer key rows)
for hs, r, ts in facts: RS[r].append((NI[hs], NI[ts]))
RS = {r: (torch.tensor([a for a, _ in v]), torch.tensor([t for _, t in v])) for r, v in RS.items()}
def read_ptr(name, r):
    c = MEM.get((name, r)); tot = sum(c.values()) if c else 0
    return {n: v / tot for n, v in c.most_common(LK)} if c else {}
def read_key(name_ix_w, r):
    """name_ix_w: {name index: weight}; returns {name: weight} by soft decode over all known names"""
    if r not in RS: return {}
    sa, ta = RS[r]; out = torch.zeros(KT.shape[1])
    for ix, w in name_ix_w.items():
        att = torch.softmax((KT[sa] @ KT[ix]) / TAU, 0); out += w * (att[:, None] * KT[ta]).sum(0)
    if out.norm() < 1e-8: return {}
    sc = torch.softmax((KT @ (out / out.norm())) / TAU, 0); v, ix = sc.topk(LK); v = v / v.sum()
    return {names[int(i)]: float(w) for w, i in zip(v, ix)}
def hop(route, cur, r):
    """cur: {name: weight} -> {name: weight} after one hop by the route"""
    out = collections.Counter()
    for n, w in cur.items():
        if route == 'A': res = read_ptr(n, r)
        elif route == 'B': res = read_key({NI[n]: 1.0}, r) if n in NI else {}
        else:
            res = read_ptr(n, r)
            if not res and n in NI: res = read_key({NI[n]: 1.0}, r)
        for m, v in res.items(): out[m] += w * v
    return dict(out.most_common(LK))
res = collections.defaultdict(lambda: collections.Counter())
for ci, (as_, r1, bs1, r2, bs2, gold) in enumerate(chains):
    al = 'aliased' if bs1 != bs2 else 'plain'
    for route in ('DECODE', 'A', 'B', 'C'):
        if route == 'DECODE':
            h1 = read_ptr(as_, r1); top = max(h1, key=h1.get) if h1 else None; out = read_ptr(top, r2) if top else {}
        else: out = hop(route, hop(route, {as_: 1.0}, r1), r2)
        rk = sorted(out, key=out.get, reverse=True)
        t1 = bool(rk) and rk[0] in gold; t5 = any(x in gold for x in rk[:5]); h1ok = None
        for k_, v_ in ((route, 1), (route + '|' + al, 1)): res[k_]['n'] += v_; res[k_]['t1'] += t1; res[k_]['t5'] += t5; res[k_]['any'] += bool(rk)
    if (ci + 1) % 1000 == 0: P(f"   {ci + 1} chains  ({time.time() - T0:.0f}s) | " + " | ".join(f"{r} {res[r]['t1'] / res[r]['n']:.3f}" for r in ('DECODE', 'A', 'B', 'C')))
P("TWO-HOP ON REAL TEXT (A -> C never stated; both hops written): top-1 / top-5 / any answer | ALIASED B (two surface forms) | PLAIN B")
for r in ('DECODE', 'A', 'B', 'C'):
    f = lambda k: f"{res[k]['t1'] / max(1, res[k]['n']):.3f} / {res[k]['t5'] / max(1, res[k]['n']):.3f} / {res[k]['any'] / max(1, res[k]['n']):.2f} (n {res[k]['n']})"
    P(f"   {r:6s} all {f(r)} | aliased {f(r + '|aliased')} | plain {f(r + '|plain')}")
P(f"DONE ({time.time() - T0:.0f}s)")
