"""DOES TYPED COMPOSITION GENERALISE TO NEW PAIRS? (16 Nov 2026; tools/comp_general.py) - GPU write path, distant_50k, the ledger
persisting across rounds (written pairs are never re-written, as on the GPU). After the correction: the exact table (premise wordings A-B,
B-C -> the confirming wording) types only 3-6 of 200 NEW inferences. Coarser keys, backing off from the most specific that has evidence:
  L0 the premises' exact wordings      L1 the premises' last words ('born in' -> 'in')      L2 the three names' classes (A, bridge, C)
  L3 the two ends' classes (A, C)      (L2 / L3 learned from CONFIRMATIONS, unlike route 1's store-wide class-pair wording)
Each round's table is learned from the kept (judge: a later relational record), confirmed writes of the rounds before; applied to the next
round's NEW writes. Coverage and typed accuracy (exact Wikidata relation / 'is in' relations as one), against route 1 (the store's commonest
wording for the class pair, head = the more specific name) on the same writes.
Usage: python3 tools/comp_general.py <distant_50k.json.gz>"""
import sys, types, collections
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rti_typed=1)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; TY = pm._PM['TY']; PRD = pm._PM['PRD']
h = lambda v: int(round(ND * min(1.0, (v + 1) / 10)))
isrel = lambda r: 1 <= r[5] <= 8 and not pm._rret_listlike(r)
norm = pm._rnorm
CONT = {'P131', 'P17', 'P30', 'P706', 'P276', 'P361'}
def cg(x, grp):
    if not grp: return x
    b, inv = (x[:-3], '^-1') if x.endswith('^-1') else (x, ''); return ('CONTAINED' + inv) if b in CONT else x
def rels(X, Y, grp): return {cg(x, grp) for x in PRD.get((X, Y), ())} | {cg(x + '^-1', grp) for x in PRD.get((Y, X), ())}
RMc = collections.defaultdict(collections.Counter)
for r in R:
    if r[6] and isrel(r):
        for x in set(PRD.get((r[1], r[2]), ())) | {x + '^-1' for x in PRD.get((r[2], r[1]), ())}: RMc[norm(r[3])][x] += 1
RM = {w: c.most_common(1)[0][0] for w, c in RMc.items()}
def right(c, grp): return c is not None and c[1] in RM and cg(RM[c[1]], grp) in rels(c[0], c[2], grp)
last = lambda w: None if w is None else ((w[:-3].split() or ['?'])[-1] + ' ^R' if w.endswith(' ^R') else (w.split() or ['?'])[-1])
TAB = [collections.defaultdict(collections.Counter) for _ in range(4)]; pm._PM['rti'] = {}
for rd, v in enumerate((1, 3, 5), 1):
    before = set(pm._PM['rti']); pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(v)], v)
    W = {k: w for k, w in pm._PM['rti'].items() if k not in before}
    need = {frozenset((A, b)) for (A, C), e in W.items() for b in e['prem']} | {frozenset((b, C)) for (A, C), e in W.items() for b in e['prem']}
    pw = collections.defaultdict(list); conf = collections.defaultdict(list); want = {frozenset(k): k for k in W}
    for r in R:
        if not isrel(r): continue
        f = frozenset((r[1], r[2]))
        if r[0] < h(v) and f in need: pw[f].append((r[1], r[2], norm(r[3])))
        if h(v) <= r[0] < h(v + 2) and f in want: conf[want[f]].append((r[1], r[2], norm(r[3])))
    byname = collections.defaultdict(list); DGd = collections.Counter()
    for r in R:
        if r[0] < h(v):
            DGd[r[1]] += 1; DGd[r[2]] += 1
            if isrel(r):
                w_ = norm(r[3])
                if any(ch.isalpha() for ch in w_): byname[r[1]].append((r[1], r[2], w_)); byname[r[2]].append((r[1], r[2], w_))
    def ori(X, Y):
        cs = pw.get(frozenset((X, Y)))
        if not cs: return None
        a, b, w = collections.Counter(cs).most_common(1)[0][0]; return w if a == X else w + ' ^R'
    def keys(A, C, e):
        out = [[], [], [], [(TY.get(A, 'X'), TY.get(C, 'X'))]]
        for b in e['prem']:
            a1, a2 = ori(A, b), ori(b, C)
            if a1 and a2: out[0].append((a1, a2)); out[1].append((last(a1), last(a2)))
            out[2].append((TY.get(A, 'X'), TY.get(b, 'X'), TY.get(C, 'X')))
        return out
    def claim(A, C, cw):
        rv = cw.endswith(' ^R'); w0 = cw[:-3] if rv else cw; return (C, w0, A) if rv else (A, w0, C)
    if rd > 1:                                                                 # apply the earlier rounds' tables to this round's NEW writes
        res = collections.defaultdict(list); lvl = collections.Counter()
        for (A, C), e in W.items():
            ks = keys(A, C, e); got = None
            for L in range(4):
                votes = collections.Counter()
                for k in ks[L]: votes.update(TAB[L].get(k, {}))
                c = claim(A, C, votes.most_common(1)[0][0]) if votes else None
                res[f'L{L} alone'].append((c is not None, right(c, 0), right(c, 1), e['truth']))
                if got is None and c is not None: got = (L, c)
            if got: lvl[got[0]] += 1
            c = got[1] if got else None; res['backoff L0-L3'].append((c is not None, right(c, 0), right(c, 1), e['truth']))
            r1 = (e['th'], e['tw'], e['tt']) if e.get('tw') else None; res['route 1 (store class pair)'].append((r1 is not None, right(r1, 0), right(r1, 1), e['truth']))
            if r1 is not None:                                                 # 16 Nov NAME-SPECIFIC: how the TAIL name is related to names of the head's class
                hd, tl = (r1[0], r1[2]) if DGd.get(r1[0], 0) <= DGd.get(r1[2], 0) else (r1[2], r1[0]); cc = collections.Counter()
                for X, Y, w in byname.get(tl, ()):
                    o = Y if X == tl else X
                    if TY.get(o, 'X') == TY.get(hd, 'X') and w: cc[(w, X == tl)] += 1
                if cc:
                    (w, tfirst), _ = cc.most_common(1)[0]; c2 = (tl, w, hd) if tfirst else (hd, w, tl)
                else: c2 = r1
                res['name-specific (tail), else route 1'].append((True, right(c2, 0), right(c2, 1), e['truth']))
        out = []
        for nm, L in res.items():
            a = np.array(L, bool); cov = a[:, 0]
            out.append(f"{nm}: types {cov.sum()} ({cov.mean():.2f}), right exact {a[cov, 1].mean() if cov.any() else float('nan'):.3f} / 'is in' {a[cov, 2].mean() if cov.any() else float('nan'):.3f}")
        print(f"ROUND {rd} (visit {v + 1}, {len(W)} new writes, right as pairs {np.mean([e['truth'] for e in W.values()]):.3f}; back-off used L0-L3 {dict(sorted(lvl.items()))}) || " + " | ".join(out), flush=True)
    for (A, C), e in W.items():                                                # learn from this round's confirmed writes
        if not conf.get((A, C)): continue
        X, Y, w = collections.Counter(conf[(A, C)]).most_common(1)[0][0]; cw = w if X == A else w + ' ^R'
        ks = keys(A, C, e)
        for L in range(4):
            for k in ks[L]: TAB[L][k][cw] += 1
    print(f"   tables after round {rd}: " + ", ".join(f"L{L} {len(TAB[L])} cells" for L in range(4)), flush=True)
