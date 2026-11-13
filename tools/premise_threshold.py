"""HOW GOOD MUST PREMISES BE FOR INFERENCE BY RECALL? (13 Nov 2026; tools/premise_threshold.py) - offline, no effect on the package.
tools/integrate_test.py: A-C proposed from two stored records sharing a name (other documents) is a real relation 0.48 of the time when both
premise records are related, 0.07 when neither (base 0.09). Here the store's precision is SET: it keeps half of the related records at random
plus as many unrelated records as give premise precision p; proposals use kept records only. Reported per p: verifiable proposals and
their gold rate, and the gold rate of proposals supported by 2+ bridges. Tells which premise precision makes A-C proposals usable.
Usage: python3 tools/premise_threshold.py"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_it = open(os.path.join(HERE, 'integrate_test.py')).read()
exec(compile(_it[:_it.index('# the store: for each name')], 'integrate_test', 'exec'))
rel = np.where(y)[0]; unr = np.where(~y)[0]; rq = np.random.RandomState(4)
P(f"records {len(C)}: related {len(rel)}, unrelated {len(unr)} (store precision with everything kept {y.mean():.3f})")
for p in (y.mean(), 0.38, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    kr = rq.choice(rel, len(rel) // 2, replace=False); nu = int(round(len(kr) * (1 - p) / p)) if p < 1 else 0
    ku = rq.choice(unr, min(nu, len(unr)), replace=False); keep = np.zeros(len(C), bool); keep[kr] = True; keep[ku] = True
    nb = collections.defaultdict(list)
    for i in np.where(keep)[0]:
        c = C[i]; nb[c['cA']].append((c['cB'], c['di'])); nb[c['cB']].append((c['cA'], c['di']))
    PR = collections.defaultdict(set)
    for i in np.where(keep)[0]:
        c = C[i]
        for B, Cn in ((c['cA'], c['cB']), (c['cB'], c['cA'])):
            if dfreq[B] > 200: continue
            for A, dA in nb[B]:
                if dA != c['di'] and A != Cn: PR[frozenset((A, Cn))].add(B)
    ks = [k for k in PR if k in ANN]; g = np.array([k in GOLDP for k in ks]); b2 = np.array([len(PR[k]) >= 2 for k in ks])
    P(f"   premise precision {keep[rel].sum() / keep.sum():.2f} (kept {keep.sum()}): verifiable A-C proposals {len(ks)}, gold {g.mean():.3f} | "
      f"with 2+ bridges {int(b2.sum())}, gold {g[b2].mean() if b2.any() else float('nan'):.3f}  ({time.time() - T0:.0f}s)")
P(f"base rate (name pairs in one document) {np.mean([q in GOLDP for q in ANN]):.3f}; DONE ({time.time() - T0:.0f}s)")
