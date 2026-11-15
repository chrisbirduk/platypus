"""TYPED INFERENCE FROM REAL TEXT BY RULE MINING (15 Nov 2026; tools/typed_rules.py) - offline, label-free mining, labels only to score.
The GPU path's real-text inferences are UNTYPED ('A is connected to C', 0.74 right) and no judge separates their errors (associated pairs
without a relation; tools/env_judge.py). The GPU's real-text relational module mines no rules (its frames are ',' / '(' / 'in'). Here
AMIE-style rules over the premises store's own records (documents 0 -> 2/10 of the distant set):
  TYPED FACT   a record with 1-6 words between the names whose wording is one of the commonest specific wordings (not ',' 'and' '(' ...),
               typed by the names' entity types: (wording, type A, type B); either direction (inverse relations '^-1')
  RULE         r1(x, y) & r2(y, z) => r3(x, z) (closed path of length 2): support (distinct x-z pairs where body and head hold), PCA
               confidence (support / body pairs whose x has some r3 fact), kept at support >= SUP and PCA >= CONF, head coverage reported
  PROPOSAL     a body pair x-z under a kept rule with NO record between x and z in what was read: the typed inference r3(x, z)
Scored (EVALUATION ONLY): untyped - the pair has some Wikidata relation (the GPU path's 0.74 measure); TYPED - the relation the head wording
names (its commonest Wikidata relation on early labelled records) holds for (x, z) in that direction; LATER - documents 2/10 -> 4/10 state
x-z (label-free confirmation), and with the head's own wording.
Usage: python3 tools/typed_rules.py <train_distant.json.gz>   env TY_NW (150 wordings), TY_SUP (10), TY_CONF (0.3), TY_HUB (40), TY_KEY (full | last), TY_XTYPES (e.g. TIME,NUM), TY_XDOC (1: body facts from different documents)"""
import sys, os, json, gzip, types, time, collections
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time()
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; TY = pm._PM['TY']; LAB = pm._PM['LAB']
n1 = int(round(ND * 0.2)); n2 = int(round(ND * 0.4))
NW, SUP, CONF, HUB = int(os.environ.get('TY_NW', 150)), int(os.environ.get('TY_SUP', 10)), float(os.environ.get('TY_CONF', 0.3)), int(os.environ.get('TY_HUB', 40))
STOP = {',', 'and', 'or', '(', ')', '-', ', and', '/', ':', ';', '.', "'s", '"', 'with', 'the', 'of the', ', the', 'and the', ', a'}
KEY = os.environ.get('TY_KEY', 'full')                                         # 'full' wording, or its 'last' word (coarser relations, more facts)
R = [(r[0], r[1], r[2], (r[4] if KEY == 'last' else r[3])) + tuple(r[4:]) for r in R]
early = [r for r in R if r[0] < n1 and 1 <= r[5] <= 6 and r[1] != r[2]]
wc = collections.Counter(r[3] for r in early)
WORD = [w for w, c in wc.most_common(NW * 3) if w not in STOP and any(ch.isalpha() for ch in w) and len(w) >= 2][:NW]; WS = set(WORD)
rel = lambda w, a, b: (w, TY.get(a, 'X'), TY.get(b, 'X'))
XT = set(os.environ.get('TY_XTYPES', '').split(',')) - {''}; XD = int(os.environ.get('TY_XDOC', 0)); DOC = {}   # types left out; body facts from different documents
F = collections.defaultdict(set); OUT = collections.defaultdict(lambda: collections.defaultdict(set)); PAIR = set()
for r in early:
    PAIR.add(frozenset((r[1], r[2])))
    if r[3] not in WS: continue
    k = rel(r[3], r[1], r[2])
    if XT and (k[1] in XT or k[2] in XT): continue
    F[k].add((r[1], r[2])); OUT[r[1]][k].add(r[2]); OUT[r[2]][k + ('^-1',)].add(r[1]); DOC.setdefault((r[1], r[2]), set()).add(r[0])
RELS = sorted(F, key=lambda k: -len(F[k])); NF = sum(len(v) for v in F.values())
P(f"documents < {n1}: {len(early)} records 1-6 words apart, {len(PAIR)} name pairs; {len(WORD)} specific wordings -> {len(RELS)} typed relations, {NF} typed facts  ({time.time() - T0:.0f}s)")
P("   commonest typed relations: " + ", ".join(f"{k[1]} '{k[0]}' {k[2]} ({len(F[k])})" for k in RELS[:12]))
HEAD = {}
for x, d in OUT.items():
    for k, zs in d.items():
        for z in zs: HEAD.setdefault((x, z), set()).add(k)
HASR = collections.defaultdict(set)                                            # x -> head relations x has (PCA denominator)
for (x, z), ks in HEAD.items():
    for k in ks: HASR[x].add(k)
# bodies: x -r1-> y -r2-> z, y not a hub
BODY = collections.defaultdict(set)
for y, d in OUT.items():
    nin = [(k, x) for k, xs in d.items() for x in xs]                           # y -k-> x, i.e. x -k^-1-> y
    if len(nin) > HUB: continue
    for k1, x in nin:
        r1 = k1[:3] + (('',) if len(k1) == 4 else ('^-1',))                     # x -r1-> y is the inverse of y -k1-> x
        for k2, z in nin:
            if z == x: continue
            if XD:                                                             # the two body facts from DIFFERENT documents (not one sentence's template)
                d1 = DOC.get((x, y), set()) | DOC.get((y, x), set()); d2 = DOC.get((y, z), set()) | DOC.get((z, y), set())
                if not (d1 - d2 and d2 - d1) and d1 == d2: continue
                if len(d1 | d2) < 2: continue
            BODY[(r1, k2)].add((x, z))
P(f"{len(BODY)} body patterns r1(x,y) & r2(y,z), {sum(len(v) for v in BODY.values())} body pairs (hubs over {HUB} skipped)  ({time.time() - T0:.0f}s)")
RULES = []
for (r1, r2), bp in BODY.items():
    if len(bp) < SUP: continue
    cnt = collections.Counter(); den = collections.Counter()
    for x, z in bp:
        for k in HEAD.get((x, z), ()): cnt[k] += 1
    for k, s in cnt.items():
        if s < SUP: continue
        d_ = sum(1 for x, z in bp if k in HASR[x])
        pca = s / max(1, d_)
        if pca >= CONF: RULES.append((r1, r2, k, s, pca, s / len(bp), len(bp)))
RULES.sort(key=lambda t: -t[4])
fmt = lambda k: f"{k[1]} '{k[0]}'{'^-1' if len(k) > 3 and k[3] == '^-1' else ''} {k[2]}"
P(f"RULES kept (support >= {SUP}, PCA confidence >= {CONF}): {len(RULES)}")
for t in RULES[:15]: P(f"   {fmt(t[0])} & {fmt(t[1])} => {fmt(t[2])}  support {t[3]}, PCA {t[4]:.2f}, std {t[5]:.2f}")
# evaluation only: ordered Wikidata relations; a typed relation's commonest Wikidata relation (early labelled records)
PR = collections.defaultdict(set)
for d in json.load(gzip.open(sys.argv[1], 'rt')):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    for l in d['labels']: PR[(cn[l['h']], cn[l['t']])].add(l['r'])
WREL = collections.defaultdict(collections.Counter)
for k in RELS:
    inv = len(k) > 3
    for a, b in F[k]:
        for rr in PR.get((a, b), ()): WREL[k[:3]][rr] += 1
later = collections.defaultdict(set)
for r in R:
    if n1 <= r[0] < n2: later[frozenset((r[1], r[2]))].add(r[3])
props = {}
for r1, r2, k, s, pca, std, nb in RULES:
    for x, z in BODY[(r1, r2)]:
        if frozenset((x, z)) in PAIR: continue
        if (x, z) not in props or props[(x, z)][1] < pca: props[(x, z)] = (k, pca)
P(f"PROPOSALS (typed inferences, the pair never stated in what was read): {len(props)}  ({time.time() - T0:.0f}s)")
def score(L):
    if not L: return "-"
    unt = np.mean([frozenset((x, z)) in LAB for (x, z), (k, p) in L]); typ = []; lat = []; latw = []
    for (x, z), (k, p) in L:
        wr = WREL.get(k[:3]); inv = len(k) > 3 and k[3] == '^-1'
        if wr: top = wr.most_common(1)[0][0]; typ.append(int(top in (PR.get((z, x), ()) if inv else PR.get((x, z), ()))))
        ws = later.get(frozenset((x, z)), set()); lat.append(int(bool(ws))); latw.append(int(k[0] in ws))
    return (f"n {len(L)}: UNTYPED right (some Wikidata relation) {unt:.3f} | TYPED right (the head's relation, that direction) "
            f"{np.mean(typ) if typ else float('nan'):.3f} (n {len(typ)}) | stated later {np.mean(lat):.2f}, in the head's own wording {np.mean(latw):.2f}")
L = sorted(props.items(), key=lambda t: -t[1][1])
P(f"   ALL: {score(L)}")
for lo, hi in ((0.8, 1.01), (0.6, 0.8), (0.45, 0.6), (CONF, 0.45)):
    P(f"   PCA {lo:.2f}-{min(hi, 1):.2f}: {score([t for t in L if lo <= t[1][1] < hi])}")
P(f"   top 200 by confidence: {score(L[:200])}")
P("   examples: " + " | ".join(f"{x} {fmt(k)} {z} ({p:.2f})" for (x, z), (k, p) in L[:8]))
P(f"DONE ({time.time() - T0:.0f}s)")
