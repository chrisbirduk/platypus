"""WHY DID THE RETRACTION TRIGGER FAIL ON THE GPU? (14 Nov 2026; tools/trigger_retest.py) - offline, the package's own code path.
GPU cg27a_rret visit 4: the confirmation score (short-gap records + document PMI) AUC 0.517 on the 200 visit-2 writes, flagged wrong 0.38
vs 0.42; offline (tools/confirm_trigger.py, Re-DocRED writes, human labels) AUC 0.72-0.73. Two things differ: the TRUTH (distant Wikidata
labels vs Re-DocRED human labels) and the WRITES (pairs from held-out distant documents vs held-out Re-DocRED documents).
Part 1 reruns the GPU path exactly (platypus.inference.premises: _load_distant, _rcode_step at visit 2 with 2/10 of the distant set,
code + resource allocation, 512 dims, seed 4) and judges the writes on the next 2/10 - each signal's AUC against the distant labels, and
against what the store reads (any later sentence with both names; within 3 words).
Part 2 takes the Re-DocRED writes of tools/fact_compress.py FC_SAVE (human labels) and scores them against the DISTANT labels too: the same
writes, two truths.
Usage: python3 tools/trigger_retest.py <train_distant.json.gz> <ret_s0.json> [<ret_s1.json>]"""
import sys, json, types, math, collections, time
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.compress import _cg_auc
P = lambda *a: print(*a, flush=True); T0 = time.time()
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+ra',
                            inf_rcode_selreport=0, inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; LAB = pm._PM['LAB']
n1 = int(round(ND * 0.2)); n2 = int(round(ND * 0.4))
P(f"distant set: {ND} documents, {len(R)} records, {len(LAB)} labelled pairs; write at {n1} documents, judge on {n1}-{n2}  ({time.time() - T0:.0f}s)")
pm._PM['rti'] = {}
pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < n1], 1)
W = pm._PM['rti']; ks = list(W); y = np.array([bool(W[k]['truth']) for k in ks])
P(f"PART 1 (the GPU path): {len(ks)} written, labelled right {y.mean():.3f}  ({time.time() - T0:.0f}s)")
want = {frozenset(k) for k in ks}; nm = {n for k in ks for n in k}; df = collections.Counter(); dn = set(); byp = collections.defaultdict(lambda: [set(), 0, 0])
for r in R:
    if not (n1 <= r[0] < n2): continue
    for n in (r[1], r[2]):
        if n in nm and (r[0], n) not in dn: dn.add((r[0], n)); df[n] += 1
    fk = frozenset((r[1], r[2]))
    if fk in want: b = byp[fk]; b[0].add(r[0]); b[1] += int(r[5] <= 3); b[2] += 1
nd = n2 - n1
g = lambda k, i: (byp[frozenset(k)][i] if frozenset(k) in byp else (set() if i == 0 else 0))
docs = np.array([len(g(k, 0)) for k in ks], float); short = np.array([g(k, 1) for k in ks], float); recs = np.array([g(k, 2) for k in ks], float)
pmi = np.array([math.log(len(g(k, 0)) + 0.5) + math.log(nd) - math.log(df[k[0]] + 1) - math.log(df[k[1]] + 1) for k in ks])
pop = -np.array([math.log((df[k[0]] + 1) * (df[k[1]] + 1)) for k in ks]); rk = lambda x: np.argsort(np.argsort(x + 1e-9 * np.random.RandomState(7).rand(len(x)))) / (len(x) - 1)
sc = rk(short) + rk(pmi); code = np.array([W[k]['code'] for k in ks]); br = np.array([W[k]['bridges'] for k in ks], float)
P(f"   later reading: pairs stated again {np.mean(docs > 0):.2f} (right {np.mean(docs[y] > 0):.2f} / wrong {np.mean(docs[~y] > 0):.2f}); short-gap {np.mean(short > 0):.2f}")
for name, v in (('documents stating the pair', docs), ('records', recs), ('short-gap records', short), ('document PMI', pmi), ('short + PMI (the trigger)', sc),
                ('popularity of the names', pop), ('the write-time code', code), ('bridges at write', br)):
    f = rk(v) <= 0.2
    P(f"   {name:28s} AUC vs DISTANT labels {_cg_auc(v[y], v[~y]):.3f} | bottom 20% wrong {1 - y[f].mean():.2f} (overall {1 - y.mean():.2f})")
# ---- 14 Nov PREDICTIVE COMPRESSION ('reverse the judgement'): does the inference A~C make what is read LATER cheaper to code? If A and C are
# related, A's new partners should come from C's neighbourhood (and C's from A's) more than chance. For each write: the bits saved coding A's
# new partners (C excluded) with C's old neighbourhood mixed in, against the names' popularity alone - and the same with a CONTROL name of
# C's degree in C's place. Retracting a write whose gain is no better than the control's costs nothing in compression.
def compress_gain(pairs, old_lo, old_hi, new_lo, new_hi):
    nbr = collections.defaultdict(set); deg = collections.Counter()
    for r in R:
        if old_lo <= r[0] < old_hi: nbr[r[1]].add(r[2]); nbr[r[2]].add(r[1])
    for n, v in nbr.items(): deg[n] = len(v)
    names = list(nbr); Z = sum(deg.values()) + len(names)
    p0 = lambda x: (deg.get(x, 0) + 1) / Z
    newp = collections.defaultdict(list); want = {n for k in pairs for n in k}
    for r in R:
        if new_lo <= r[0] < new_hi:
            if r[1] in want: newp[r[1]].append(r[2])
            if r[2] in want: newp[r[2]].append(r[1])
    byd = collections.defaultdict(list)
    for n in names: byd[min(int(np.log2(deg[n] + 1)), 20)].append(n)
    rr = np.random.RandomState(11); lam = 0.3
    def gain(a, c_nb, excl):
        xs = [x for x in newp.get(a, []) if x not in excl]
        if not xs or not c_nb: return 0.0, len(xs)
        q = lambda x: (1 - lam) * p0(x) + lam * (1.0 / len(c_nb) if x in c_nb else 0.0)
        return float(np.sum([np.log2(q(x)) - np.log2(p0(x)) for x in xs])), len(xs)
    out = []
    for a, c in pairs:
        g1, n1_ = gain(a, nbr.get(c, set()) - {a}, {c}); g2, n2_ = gain(c, nbr.get(a, set()) - {c}, {a}); ctl = []
        for _ in range(3):
            pool_c = byd.get(min(int(np.log2(deg.get(c, 0) + 1)), 20)) or names; pool_a = byd.get(min(int(np.log2(deg.get(a, 0) + 1)), 20)) or names
            zc = pool_c[rr.randint(len(pool_c))]; za = pool_a[rr.randint(len(pool_a))]
            ctl.append(gain(a, nbr.get(zc, set()) - {a}, {c})[0] + gain(c, nbr.get(za, set()) - {c}, {a})[0])
        out.append((g1 + g2, float(np.mean(ctl)), n1_ + n2_))
    return np.array(out)
G = compress_gain(ks, 0, n1, n1, n2); net = G[:, 0] - G[:, 1]; has = G[:, 2] > 0; per = net / np.maximum(G[:, 2], 1)
P(f"   PREDICTIVE COMPRESSION ({int(has.sum())} of {len(ks)} writes with new partners): bits saved on later partners with the inference "
  f"{np.mean(G[has, 0]):.1f} vs a same-degree control {np.mean(G[has, 1]):.1f} | AUC vs distant labels: gain {_cg_auc(G[:, 0][y], G[:, 0][~y]):.3f}, over control "
  f"{_cg_auc(net[y], net[~y]):.3f}, per new partner {_cg_auc(per[y], per[~y]):.3f} | bottom 20% of gain over control: wrong {1 - y[rk(net) <= 0.2].mean():.2f} "
  f"(overall {1 - y.mean():.2f}) | retracting would improve compression (gain over control < 0): {np.mean(net < 0):.2f} of writes, wrong among them "
  f"{(1 - y[net < 0].mean()) if (net < 0).any() else float('nan'):.2f}")
P("PART 2: the Re-DocRED writes (human labels) scored against the distant labels as well")
for path in sys.argv[2:]:
    Wr = json.load(open(path)); yh = np.array([w[2] for w in Wr], bool); yd = np.array([frozenset((w[0], w[1])) in LAB for w in Wr], bool)
    P(f"   {path.split('/')[-1]}: human-labelled right {yh.mean():.2f}, distant-labelled right {yd.mean():.2f}, agreement {np.mean(yh == yd):.2f} "
      f"(human right & distant right {np.mean(yh & yd):.2f}, human right only {np.mean(yh & ~yd):.2f}, distant right only {np.mean(~yh & yd):.2f})")
    G2 = compress_gain([(w[0], w[1]) for w in Wr], 0, 5000, 5000, 30000); net2 = G2[:, 0] - G2[:, 1]; h2 = G2[:, 2] > 0
    a2 = _cg_auc(net2[yd], net2[~yd]) if yd.any() and (~yd).any() else float('nan')
    P(f"      PREDICTIVE COMPRESSION on them (old = distant documents 0-5,000, later = 5,000-30,000; {int(h2.sum())} with new partners): AUC gain over "
      f"control vs HUMAN labels {_cg_auc(net2[yh], net2[~yh]):.3f}, vs distant labels {a2:.3f} | bottom 20%: wrong (human) {1 - yh[rk(net2) <= 0.2].mean():.2f} (overall {1 - yh.mean():.2f})")
P(f"DONE ({time.time() - T0:.0f}s)")
