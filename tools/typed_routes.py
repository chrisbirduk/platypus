"""ROUTES TO TYPED INFERENCE FROM WHAT THE TEXT CONFIRMS (16 Nov 2026; tools/typed_routes.py) - offline, the GPU write path, distant set.
Rule mining over types failed (typed proposals right 0.00-0.23 vs 0.74 untyped). Three routes that take the type from the text instead:
  ROUTE 2  TYPE FROM THE CONFIRMING SENTENCE: a written inference A ~ C that a later RELATIONAL record states (1-8 words between the names,
           not a list - the GPU judge's signal 'rel') takes that record's wording and order as its type: 'X <wording> Y'.
  ROUTE 1  COARSE TYPE AT WRITE TIME: head = the more specific name (lower store degree), tail = the other; the claim 'head <wording> tail'
           with the commonest relational wording the store holds between names of those two classes in that degree order (label-free).
  ROUTE 3  TYPED COMPOSITION LEARNED FROM CONFIRMATIONS: for each confirmed inference, its premises' wordings (A-B and B-C, the commonest
           relational wording of each bridge pair at write time, oriented) -> the confirming wording; a table built over earlier rounds
           predicts the next round's confirming wording (label-free check: exact match of the normalised wording), against the
           class-pair table of route 1.
Scoring (evaluation only): a claim 'X w Y' is RIGHT when the wording's commonest Wikidata relation over all labelled records (oriented)
is one of the pair's own relations in that direction. Untyped writes: 0.74 right as pairs; a typed claim must also get the relation and
direction right.
Usage: TR_STAGE=12 python3 tools/typed_routes.py <train_distant.json.gz>     (routes 1 and 2 on the GPU's first batch, visit 2)
       TR_STAGE=3w TR_ROUND=k python3 tools/typed_routes.py <distant_50k.json.gz>   (k = 1, 2, 3: one write round per process, saved)
       TR_STAGE=3 python3 tools/typed_routes.py <distant_50k.json.gz>"""
import sys, os, json, types, time, collections
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time()
ST = os.environ.get('TR_STAGE', '12'); OUT = os.environ.get('TR_OUT', 'tr')
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; TY = pm._PM['TY']; PRD = pm._PM['PRD']
h = lambda v: int(round(ND * min(1.0, (v + 1) / 10)))
STOP = {'the', 'a', 'an'}
norm = lambda w: ' '.join(t for t in w.split() if t not in STOP)
isrel = lambda r: 1 <= r[5] <= 8 and not pm._rret_listlike(r)
GRP = int(os.environ.get('TR_GROUP', 0)); CONT = {'P131', 'P17', 'P30', 'P706', 'P276', 'P361'}   # TR_GROUP 1: 'is in' relations scored as one
def cg(x):
    if not GRP: return x
    b, inv = (x[:-3], '^-1') if x.endswith('^-1') else (x, ''); return ('CONTAINED' + inv) if b in CONT else x
def rels(X, Y): return {cg(x) for x in PRD.get((X, Y), ())} | {cg(x + '^-1') for x in PRD.get((Y, X), ())}
def relmap(upto):
    """each normalised wording's commonest Wikidata relation over the labelled relational records (evaluation only)"""
    c = collections.defaultdict(collections.Counter)
    for r in R:
        if r[0] < upto and r[6] and isrel(r):
            for x in rels(r[1], r[2]): c[norm(r[3])][x] += 1
    return {w: cg(k.most_common(1)[0][0]) for w, k in c.items()}
def writes(v):
    pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(v)], v)
    return dict(pm._PM['rti'])
def degrees(upto):
    nb = collections.defaultdict(set)
    for r in R:
        if r[0] < upto: nb[r[1]].add(r[2]); nb[r[2]].add(r[1])
    return {n: len(s) for n, s in nb.items()}
def confirm(W, lo, hi):
    """the later relational records of each written pair: [(X, Y, wording)] in record order"""
    want = {frozenset(k): k for k in W}; out = collections.defaultdict(list)
    for r in R:
        if lo <= r[0] < hi and isrel(r):
            k = want.get(frozenset((r[1], r[2])))
            if k is not None: out[k].append((r[1], r[2], norm(r[3])))
    return out
def best(cs):
    c = collections.Counter(cs); return c.most_common(1)[0][0]
if ST == '12':
    v = 1; W = writes(v); RM = relmap(h(v + 2)); DG = degrees(h(v)); y = {k: e['truth'] for k, e in W.items()}
    CF = confirm(W, h(v), h(v + 2))
    P(f"WRITES {len(W)} at visit {v + 1} (documents < {h(v)}): right {np.mean(list(y.values())):.3f} | judged on documents {h(v)}-{h(v + 2)}  ({time.time() - T0:.0f}s)")
    # ROUTE 2
    ok2, typ2, rr2, ex = [], [], [], []
    for k in W:
        if not CF.get(k): continue
        X, Y, w = best(CF[k]); good = RM.get(w) in rels(X, Y) if w in RM else False
        ok2.append(good); typ2.append(w); rr2.append(y[k]); ex.append(f"{X} | {w} | {Y} ({'RIGHT' if good else 'wrong'})")
    ok2, rr2 = np.array(ok2), np.array(rr2)
    P(f"ROUTE 2 - TYPE FROM THE CONFIRMING RELATIONAL SENTENCE: {len(ok2)} of {len(W)} writes confirmed ({len(ok2) / len(W):.2f}); of them right as pairs {rr2.mean():.3f}; "
      f"TYPED CLAIM RIGHT (relation and direction) {ok2.mean():.3f}; among pairs that are right {ok2[rr2].mean() if rr2.any() else float('nan'):.3f} | wordings with a known relation "
      f"{np.mean([t in RM for t in typ2]):.2f} | commonest wordings: " + ", ".join(f"'{w}' {n}" for w, n in collections.Counter(typ2).most_common(8)))
    P("   e.g. " + " || ".join(ex[:8]))
    # ROUTE 1: class-pair table, head = lower degree
    tab = collections.defaultdict(collections.Counter)
    for r in R:
        if r[0] < h(v) and isrel(r) and r[1] in DG and r[2] in DG and DG[r[1]] != DG[r[2]]:
            hd, tl, fw = (r[1], r[2], 1) if DG[r[1]] < DG[r[2]] else (r[2], r[1], 0)
            tab[(TY.get(hd, 'X'), TY.get(tl, 'X'))][(norm(r[3]), fw)] += 1
    ok1, rr1, cl1, ok1d = [], [], [], {}
    for (A, C) in W:
        hd, tl = (A, C) if DG.get(A, 0) <= DG.get(C, 0) else (C, A); cp = (TY.get(hd, 'X'), TY.get(tl, 'X'))
        if not tab.get(cp): continue
        (w, fw), _ = tab[cp].most_common(1)[0]; X, Y = (hd, tl) if fw else (tl, hd)
        ok1.append(RM.get(w) in rels(X, Y) if w in RM else False); rr1.append(y[(A, C)]); cl1.append((cp, w, fw)); ok1d[(A, C)] = ok1[-1]
    ok1, rr1 = np.array(ok1), np.array(rr1)
    P(f"ROUTE 1 - COARSE TYPE AT WRITE TIME (class pair, head = the more specific name): {len(ok1)} of {len(W)} typed; TYPED CLAIM RIGHT {ok1.mean():.3f}; among pairs right "
      f"{ok1[rr1].mean() if rr1.any() else float('nan'):.3f} | the claims: " + ", ".join(f"{cp[0]}-{cp[1]} '{w}'{'' if fw else ' (tail first)'} x{n}" for (cp, w, fw), n in collections.Counter(cl1).most_common(6)))
    if len(ok2):                                                               # the two routes on the same writes
        both = [k for k in W if CF.get(k) and k in ok1d]
        P(f"   on the {len(both)} confirmed writes route 1 types: route 2 {ok2.mean():.3f}, route 1 {np.mean([ok1d[k] for k in both]) if both else float('nan'):.3f}")
    P(f"DONE ({time.time() - T0:.0f}s)"); sys.exit(0)
if ST == '3w':
    k_ = int(os.environ['TR_ROUND']); v = {1: 1, 2: 3, 3: 5}[k_]; W = writes(v); CF = confirm(W, h(v), h(v + 2))
    pw = collections.defaultdict(list)                                         # premise wordings at write time
    need = {frozenset((A, b)) for (A, C), e in W.items() for b in e['prem']} | {frozenset((b, C)) for (A, C), e in W.items() for b in e['prem']}
    for r in R:
        if r[0] < h(v) and isrel(r):
            f = frozenset((r[1], r[2]))
            if f in need: pw[f].append((r[1], r[2], norm(r[3])))
    def orient(X, Y):
        cs = pw.get(frozenset((X, Y)))
        if not cs: return None
        a, b, w = best(cs); return w if a == X else w + ' ^R'
    rows = []
    for (A, C), e in W.items():
        prem = [(orient(A, b), orient(b, C)) for b in e['prem']]
        cf = None
        if CF.get((A, C)):
            X, Y, w = best(CF[(A, C)]); cf = w if X == A else w + ' ^R'
        rows.append(dict(A=A, C=C, truth=e['truth'], prem=prem, cf=cf, tA=TY.get(A, 'X'), tC=TY.get(C, 'X')))
    json.dump(rows, open(f"{OUT}_r{k_}.json", 'w'))
    P(f"round {k_} (visit {v + 1}): {len(rows)} writes, confirmed {np.mean([r['cf'] is not None for r in rows]):.2f}, premises with a relational wording "
      f"{np.mean([p is not None for r in rows for q in r['prem'] for p in q]):.2f}  ({time.time() - T0:.0f}s)"); sys.exit(0)
if ST == '3':
    RM = relmap(ND)
    RD = [json.load(open(f"{OUT}_r{k}.json")) for k in (1, 2, 3)]
    def right(r, w):
        if w is None: return False
        rev = w.endswith(' ^R'); w0 = w[:-3] if rev else w; X, Y = (r['C'], r['A']) if rev else (r['A'], r['C'])
        return RM.get(w0) in rels(X, Y) if w0 in RM else False
    T3 = collections.defaultdict(collections.Counter); TC = collections.defaultdict(collections.Counter)
    for k in range(3):
        rows = RD[k]; cf = [r for r in rows if r['cf'] is not None]
        if k > 0:
            m3, m1, g3, g1, n3 = [], [], [], [], 0
            for r in cf:
                votes = collections.Counter()
                for p in r['prem']:
                    if p[0] and p[1]: votes.update(T3.get((p[0], p[1]), {}))
                p3 = votes.most_common(1)[0][0] if votes else None; n3 += p3 is not None
                p1 = TC[(r['tA'], r['tC'])].most_common(1)[0][0] if TC.get((r['tA'], r['tC'])) else None
                if p3 is not None: m3.append(p3 == r['cf']); g3.append(right(r, p3))
                m1.append(p1 == r['cf']); g1.append(right(r, p1))
            a3, a1 = [], []                                                    # at WRITE time, on every write (confirmed or not)
            for r in rows:
                votes = collections.Counter()
                for p in r['prem']:
                    if p[0] and p[1]: votes.update(T3.get((p[0], p[1]), {}))
                if votes: a3.append((right(r, votes.most_common(1)[0][0]), r['truth']))
                p1 = TC[(r['tA'], r['tC'])].most_common(1)[0][0] if TC.get((r['tA'], r['tC'])) else None; a1.append(right(r, p1))
            P(f"ROUND {k + 1} AT WRITE TIME, all {len(rows)} writes (right as pairs {np.mean([r['truth'] for r in rows]):.3f}): composition table types {len(a3)}, typed claim right "
              f"{np.mean([x for x, _ in a3]) if a3 else float('nan'):.3f} (those pairs right {np.mean([t for _, t in a3]) if a3 else float('nan'):.3f}); class-pair table {np.mean(a1):.3f}")
            P(f"ROUND {k + 1}: {len(cf)} confirmed writes | COMPOSITION TABLE (premise wordings -> confirming wording, from earlier rounds' confirmations): predicts {n3} "
              f"({n3 / max(1, len(cf)):.2f}); exact wording {np.mean(m3) if m3 else float('nan'):.3f}, relation right {np.mean(g3) if g3 else float('nan'):.3f} || CLASS-PAIR TABLE "
              f"(same rounds): exact {np.mean(m1):.3f}, relation right {np.mean(g1):.3f} || the confirming wording itself (route 2) right {np.mean([right(r, r['cf']) for r in cf]):.3f}")
        for r in cf:                                                           # learn from this round's confirmations
            for p in r['prem']:
                if p[0] and p[1]: T3[(p[0], p[1])][r['cf']] += 1
            TC[(r['tA'], r['tC'])][r['cf']] += 1
    P(f"   table cells {len(T3)}, commonest: " + "; ".join(f"{a} + {b} -> {c.most_common(1)[0][0]} x{sum(c.values())}" for (a, b), c in sorted(T3.items(), key=lambda kv: -sum(kv[1].values()))[:6]))
    P(f"DONE ({time.time() - T0:.0f}s)")
