"""TYPING WRITTEN INFERENCES ON CONFIRMATION, AND A TRIGGER FROM THE TYPES (15 Nov 2026; tools/type_confirm.py) - offline, the GPU path.
tools/trigger_retest.py: untyped writes ('A is connected to C') are restated by later reading 94% of the time, right and wrong alike, and
no label-free signal separates them (best AUC 0.64). The idea: let the later statement TYPE the write - the words between the two names
in a later sentence ('A was born in C', 'A and C') become the claim's wording - and judge the TYPED claim:
  TYPED      a later record states the pair with 1-6 words between (either order); the claim's wording = the commonest such wording
  FUNCTIONAL how functional that wording is in what was read BEFORE the write (AMIE-style, label-free: per subject seen with it, 1 / its
             number of distinct objects; mean over subjects with 2+ records) - list-like wordings ('and', ',') are not functional
  INFORMATIVE the wording's share of records whose pair is stated by more than one wording... (see below: SPECIFIC)
  SPECIFIC   how specific the wording is: 1 - (the share of all early records that use it) ** 0.5 - common connectors are not specific
  CONTRADICTED under a functional wording (0.7+), a record read AFTER typing gives the subject another object with the same wording
Each signal's AUC against the distant labels (right vs wrong), and what retracting each signal's flagged writes does to the kept precision.
EVALUATION ONLY: the labels also give each typed wording's commonest Wikidata relation (on early labelled records) - how often a typed
claim's wording names the pair's own relation (typing accuracy).
Usage: python3 tools/type_confirm.py <train_distant.json.gz>   env TR_SEL (code+aa), TR_ASYM (1.0)"""
import sys, os, json, gzip, types, math, collections, time
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.compress import _cg_auc
P = lambda *a: print(*a, flush=True); T0 = time.time()
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4,
                            inf_rcode_sel=os.environ.get('TR_SEL', 'code+aa'), inf_rcode_asym=float(os.environ.get('TR_ASYM', 1.0)),
                            inf_rcode_selreport=0, inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; LAB = pm._PM['LAB']
n1 = int(round(ND * 0.2)); n2 = int(round(ND * 0.4)); n3 = int(round(ND * 0.6))
pm._PM['rti'] = {}
pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < n1], 1)
W = pm._PM['rti']; ks = list(W); y = np.array([bool(W[k]['truth']) for k in ks])
P(f"{len(ks)} writes at document {n1} ({cfg.inf_rcode_sel}, asymmetry {cfg.inf_rcode_asym}), labelled right {y.mean():.3f}; typed on {n1}-{n2}, "
  f"contradicted on {n2}-{n3}  ({time.time() - T0:.0f}s)")
# evaluation only: each pair's Wikidata relations (one more pass over the distant set)
PR = collections.defaultdict(set)
D = json.load(gzip.open(sys.argv[1], 'rt'))
for d in D:
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    for l in d['labels']: PR[(cn[l['h']], cn[l['t']])].add(l['r'])
del D
P(f"relation labels for {len(PR)} ordered pairs (evaluation only)  ({time.time() - T0:.0f}s)")
early = [r for r in R if r[0] < n1 and 1 <= r[5] <= 6]
wc = collections.Counter(r[3] for r in early); NE = len(early)
byw = collections.defaultdict(lambda: collections.defaultdict(set))
for r in early: byw[r[3]][r[1]].add(r[2])
def functional(w):
    s = [1.0 / len(o) for a, o in byw.get(w, {}).items() if sum(1 for _ in o) >= 1]
    many = [1.0 / len(o) for a, o in byw.get(w, {}).items()]
    return float(np.mean(many)) if len(many) >= 3 else float('nan')
wrel = collections.defaultdict(collections.Counter)                            # evaluation only: a wording's commonest relation
for r in early:
    for rel in PR.get((r[1], r[2]), ()): wrel[r[3]][rel] += 1
later = collections.defaultdict(list)
for r in R:
    if n1 <= r[0] < n2 and 1 <= r[5] <= 6: later[frozenset((r[1], r[2]))].append(r)
after = collections.defaultdict(list)
for r in R:
    if n2 <= r[0] < n3 and 1 <= r[5] <= 6: after[(r[1], r[3])].append(r[2])
rows = []
for k, t in zip(ks, y):
    A, C = tuple(k) if isinstance(k, tuple) else tuple(k)
    rs_ = later.get(frozenset((A, C)), [])
    if not rs_: rows.append(dict(t=t, typed=0)); continue
    cw = collections.Counter((r[1], r[3]) for r in rs_); (subj, w), n = cw.most_common(1)[0]; obj = C if subj == A else A
    fu = functional(w); sp = 1.0 - math.sqrt(wc.get(w, 0) / max(1, NE))
    con = int(fu == fu and fu >= 0.7 and any(o != obj for o in after.get((subj, w), [])))
    tr = wrel.get(w); acc = None
    if tr: acc = int(tr.most_common(1)[0][0] in PR.get((subj, obj), set()))
    rows.append(dict(t=t, typed=1, w=w, n=n, fu=fu, sp=sp, con=con, acc=acc, nw=len(cw)))
ty = [r for r in rows if r['typed']]
P(f"TYPED by later reading (1-6 words between): {len(ty)} of {len(rows)} ({len(ty) / len(rows):.2f}); right among typed {np.mean([r['t'] for r in ty]):.3f}, "
  f"among untyped {np.mean([r['t'] for r in rows if not r['typed']]) if len(ty) < len(rows) else float('nan'):.3f}")
P("   commonest typed wordings: " + ", ".join(f"'{w}' {c} (right {np.mean([r['t'] for r in ty if r['w'] == w]):.2f})" for w, c in collections.Counter(r['w'] for r in ty).most_common(12)))
ac = [r['acc'] for r in ty if r['acc'] is not None]
P(f"   TYPING ACCURACY (evaluation only): the typed wording's commonest relation is the pair's own: {np.mean(ac):.2f} of {len(ac)} typed claims with a known wording; "
  f"among RIGHT writes {np.mean([r['acc'] for r in ty if r['acc'] is not None and r['t']]):.2f}")
def report(name, sc, L):
    L2 = [(s, r['t']) for s, r in zip(sc, L) if s == s]
    if len(L2) < 20: P(f"   {name}: too few ({len(L2)})"); return
    s = np.array([a for a, b in L2]); t = np.array([b for a, b in L2]); auc = _cg_auc(s[t], s[~t])
    q = np.quantile(s, 0.2); fl = s <= q; kept = t[~fl].mean() if (~fl).any() else float('nan')
    P(f"   {name:46s} AUC {auc:.3f} (n {len(L2)}) | flag the lowest 20%: wrong among flagged {1 - t[fl].mean():.2f} (overall {1 - t.mean():.2f}), kept right {kept:.3f}")
P("SIGNALS vs the distant labels (higher = right):")
report('typed at all (later records, 1-6 words)', [float(r['typed']) for r in rows], rows)
report('typed: number of later records stating it', [float(r.get('n', 0)) for r in rows], rows)
report('typed: wording FUNCTIONAL (label-free)', [r['fu'] for r in ty], ty)
report('typed: wording SPECIFIC (label-free)', [r['sp'] for r in ty], ty)
report('typed: NOT contradicted later (label-free)', [1.0 - r['con'] for r in ty], ty)
report('typed: functional + specific (rank sum)', list(np.argsort(np.argsort([r['fu'] if r['fu'] == r['fu'] else -1 for r in ty])) + np.argsort(np.argsort([r['sp'] for r in ty]))), ty)
P(f"   contradicted (functional wording, another object later): {sum(r['con'] for r in ty)} of {len(ty)}; wrong among them "
  f"{1 - np.mean([r['t'] for r in ty if r['con']]) if any(r['con'] for r in ty) else float('nan'):.2f}")
P(f"DONE ({time.time() - T0:.0f}s)")
