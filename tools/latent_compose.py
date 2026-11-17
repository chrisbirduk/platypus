"""COMPOSITION ON REAL TEXT WITH LEARNED RELATION KEYS (17 Nov 2026; tools/latent_compose.py) - offline, no effect on the package.
The user's hypothesis: the system does not compose because it does not use LEARNED keys (fuzzy composition). The rule world already shows
composition in a learned code space (frozen trunk codes + relation operators: held-out composites 1.00, while the trunk's forward pass gets
0/6). Here, on Re-DocRED (Wikipedia, gold relation ids = the ideal relation key; the package's learned JEPA wording key is near-perfect on
the GPU), with labels that DO cover composites (the revised set labels A-C pairs inside a document):
  E1 RULES OVER RELATION KEYS: rules r1 o r2 -> r3 mined on the training documents (support, confidence, Wilson bound); on the held-out
     documents every chain A -r1-> B -r2-> C of LABELLED facts with a kept rule gives 'A r3 C'; precision = labelled (A, r3, C) in the
     document, and recall of the document's composable facts. If this fails, real-text composition fails even with perfect relation keys.
  E2 LATENT OPERATORS ON NAME KEYS (TG_CKPT given): name key = the stand-in trunk's state over the name's bytes; per relation a ridge map
     W_r: key(head) -> key(tail) fitted on the training documents. For each held-out composite (A, r3, C) with a chain A -r1-> B -r2-> C:
     C ranked among the document's entities by  (a) W_r3 key(A) (one learned map), (b) W_r2 W_r1 key(A) (pure latent composition),
     (c) W_r2 key(B) with B read from the memory (memory hop 1, latent hop 2), (d) NULL: W_r2' W_r1 with a random other relation r2';
     chance = 1 / entities. Abstention: accuracy when only the top quartile by margin answers.
Usage: python3 tools/latent_compose.py   env LC_MINSUP (10) LC_MINLB (0.5) LC_LAM (1.0); E2 needs TG_D=192 TG_CKPT=<model file>"""
import os, sys, json, math, time, collections
import numpy as np
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
T0 = time.time(); P = lambda s: print(s, flush=True)
TR = json.load(open(SCR + '/bench_dl/train_revised.json')); DV = json.load(open(SCR + '/bench_dl/dev_revised.json')) + json.load(open(SCR + '/bench_dl/test_revised.json'))
RI = {'P17': 'country', 'P131': 'located in admin. entity', 'P27': 'citizenship', 'P150': 'contains admin. entity', 'P161': 'cast member', 'P175': 'performer', 'P577': 'publication date', 'P569': 'born', 'P570': 'died', 'P19': 'birthplace', 'P20': 'death place', 'P495': 'country of origin', 'P159': 'HQ location', 'P361': 'part of', 'P527': 'has part', 'P710': 'participant', 'P276': 'location', 'P30': 'continent', 'P36': 'capital', 'P1376': 'capital of', 'P54': 'team', 'P102': 'party', 'P264': 'record label', 'P463': 'member of', 'P571': 'inception', 'P50': 'author', 'P170': 'creator', 'P749': 'parent org', 'P355': 'subsidiary', 'P118': 'league', 'P69': 'educated at', 'P108': 'employer', 'P40': 'child', 'P22': 'father', 'P25': 'mother', 'P26': 'spouse', 'P3373': 'sibling', 'P127': 'owned by', 'P178': 'developer', 'P140': 'religion', 'P1441': 'present in work', 'P6': 'head of gov.', 'P35': 'head of state', 'P37': 'official language', 'P1412': 'language', 'P205': 'basin country', 'P206': 'located next to water', 'P403': 'mouth', 'P706': 'located on terrain'}   # names for printing only (the file was not downloaded)
MINSUP = int(os.environ.get('LC_MINSUP', 10)); MINLB = float(os.environ.get('LC_MINLB', 0.5)); LAM = float(os.environ.get('LC_LAM', 1.0))
def wlb(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
def doc_facts(d):
    F = {(l['h'], l['r'], l['t']) for l in d['labels']}; byh = collections.defaultdict(list)
    for h, r, t in F: byh[h].append((r, t))
    return F, byh
def chains(d):
    F, byh = doc_facts(d)
    for a, r1, b in F:
        for r2, c in byh.get(b, ()):
            if c != a and c != b: yield a, r1, b, r2, c, F

# E1: rules over relation keys
cnt = collections.Counter(); hit = collections.Counter()
for d in TR:
    seen = set()
    for a, r1, b, r2, c, F in chains(d):
        if (a, r1, r2, c) in seen: continue
        seen.add((a, r1, r2, c)); cnt[(r1, r2)] += 1
        for r3 in {r for (h, r, t) in F if h == a and t == c}: hit[(r1, r2, r3)] += 1
rules = {}
for (r1, r2, r3), k in hit.items():
    n = cnt[(r1, r2)]
    if k >= MINSUP and wlb(k, n) >= MINLB and (rules.get((r1, r2)) is None or wlb(k, n) > rules[(r1, r2)][1]): rules[(r1, r2)] = (r3, wlb(k, n), k, n)
nm = lambda r: RI.get(r, r)
P(f"E1 RULES OVER RELATION KEYS: {len(TR)} training documents, {len(cnt)} relation pairs in chains; KEPT {len(rules)} (support >= {MINSUP}, Wilson LB >= {MINLB})  ({time.time() - T0:.0f}s)")
for (r1, r2), (r3, lb, k, n) in sorted(rules.items(), key=lambda x: -x[1][2])[:12]:
    P(f"   '{nm(r1)}' o '{nm(r2)}' -> '{nm(r3)}': {k} of {n} (LB {lb:.2f})")
tp = fp = 0; comp_all = 0; comp_hit = 0; byrule = collections.defaultdict(lambda: [0, 0])
for d in DV:
    F, _ = doc_facts(d); out = set()
    for a, r1, b, r2, c, _F in chains(d):
        if (r1, r2) in rules: out.add((a, rules[(r1, r2)][0], c, (r1, r2)))
    for a, r3, c, k_ in out:
        ok = (a, r3, c) in F; tp += ok; fp += not ok; byrule[k_][0] += ok; byrule[k_][1] += 1
    # recall: labelled facts that some chain in the document could compose (A-C linked by a 2-chain)
    comp = {(a, c) for a, r1, b, r2, c, _F in chains(d)}
    for (h, r, t) in F:
        if (h, t) in comp: comp_all += 1; comp_hit += any(o[0] == h and o[1] == r and o[2] == t for o in out)
P(f"E1 HELD-OUT ({len(DV)} documents): rule-composed conclusions {tp + fp}, PRECISION {tp / max(1, tp + fp):.3f} (labelled with that relation in the document) | "
  f"labelled facts reachable by a 2-chain {comp_all}, of them composed by a rule {comp_hit / max(1, comp_all):.3f}")
for k_, (h_, n_) in sorted(byrule.items(), key=lambda x: -x[1][1])[:8]:
    P(f"   '{nm(k_[0])}' o '{nm(k_[1])}' -> '{nm(rules[k_][0])}': {h_} of {n_} ({h_ / max(1, n_):.2f})")

# E2: latent operators on name keys
if not os.environ.get('TG_CKPT'):
    P("E2 skipped (no TG_CKPT)"); sys.exit(0)
import torch
here = os.path.dirname(os.path.abspath(__file__)); sys.argv = [sys.argv[0], '6000', '5']
src = open(os.path.join(here, 'trunk_guides.py')).read(); g = {}
exec(compile(src[:src.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'), g)
lm, run = g['lm'], g['run']; torch.set_num_threads(2)
name = lambda d, e: d['vertexSet'][e][0]['name']
names = sorted({name(d, e) for d in TR + DV for e in range(len(d['vertexSet']))}); NI = {n: i for i, n in enumerate(names)}
K = np.zeros((len(names), lm.o.in_features), np.float32)
for i, n in enumerate(names):
    _, H = run(lm, (' ' + n).encode('utf-8', 'replace')); K[i] = H[1:].mean(0)
    if (i + 1) % 10000 == 0: P(f"   name keys {i + 1} of {len(names)}  ({time.time() - T0:.0f}s)")
K -= K.mean(0); K /= np.linalg.norm(K, axis=1, keepdims=True) + 1e-8; D_ = K.shape[1]
XY = collections.defaultdict(list)
for d in TR:
    for (h, r, t) in doc_facts(d)[0]: XY[r].append((NI[name(d, h)], NI[name(d, t)]))
W = {}
for r, v in XY.items():
    if len(v) < 30: continue
    X = K[[a for a, _ in v]]; Y = K[[b for _, b in v]]
    W[r] = np.linalg.solve(X.T @ X + LAM * np.eye(D_), X.T @ Y)
P(f"E2 LATENT OPERATORS: {len(names)} name keys (d {D_}), {len(W)} relation maps (30+ training facts)  ({time.time() - T0:.0f}s)")
rels = sorted(W); rs = np.random.RandomState(0); res = collections.defaultdict(list)
for d in DV:
    F, _ = doc_facts(d); ents = list(range(len(d['vertexSet']))); CK = K[[NI[name(d, e)] for e in ents]]
    seen = set()
    for a, r1, b, r2, c, _F in chains(d):
        r3s = [r for (h, r, t) in F if h == a and t == c]
        if not r3s or (a, c) in seen or r1 not in W or r2 not in W: continue
        seen.add((a, c)); r3 = r3s[0]; ka = K[NI[name(d, a)]]; kb = K[NI[name(d, b)]]
        def rank(v, excl):
            s = CK @ (v / (np.linalg.norm(v) + 1e-8)); s[excl] = -9; o = np.argsort(-s); return int(o[0] == c), float(s[o[0]] - s[o[1]])
        if r3 in W: res['a ONE MAP W_r3 key(A)'].append(rank(ka @ W[r3], [a]))
        res['b LATENT W_r2 W_r1 key(A)'].append(rank(ka @ W[r1] @ W[r2], [a]))
        res['c MEMORY hop 1 + W_r2 key(B)'].append(rank(kb @ W[r2], [a, b]))
        r2n = rels[rs.randint(len(rels))]
        res['d NULL W_r2(random) W_r1 key(A)'].append(rank(ka @ W[r1] @ W[r2n], [a]))
        res['chance'].append((1.0 / max(1, len(ents) - 1), 0.0))
for k_ in sorted(res):
    v = np.array(res[k_]); acc = v[:, 0].mean()
    if k_ != 'chance' and len(v) > 8:
        q = np.quantile(v[:, 1], 0.75); top = v[v[:, 1] >= q, 0].mean()
        P(f"   {k_:34s} C first among the document's entities {acc:.3f} (n {len(v)}) | top quarter by margin {top:.3f}")
    else: P(f"   {k_:34s} {acc:.3f} (n {len(v)})")
P(f"DONE ({time.time() - T0:.0f}s)")
