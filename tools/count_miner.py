"""A COUNTING MINER ON REAL FACTS (13 Nov 2026; tools/count_miner.py) - offline, no effect on the package.
tools/dataset_volume.py showed the relational module's miner (_rm_mine: learned embeddings, snap to the nearest name) gets WORSE with more
Re-DocRED gold facts (rules 3 -> 2 -> 1, proposals 0.18 -> 0.00 -> 0.04 gold, 9 wrong writes at 800 documents) and runs out of memory at 3,553.
Here the same facts, the same 20% held out, but rules by EXACT CHAIN COUNTS over the observed facts (no embedding, no snap): for relations
r1, r2 (and their inverses) and a target r3, the subjects a with a chain a -r1-> b -r2-> c (c != a) AND a known r3 object: the rule's accuracy
is the share where a chain ends on a true r3 object. A rule must hold on n >= 20 subjects with a Wilson lower bound >= 0.7 at z = 4 (corrected
for the thousands of candidates; the module's n >= 3 at z = 1.96 let spurious rules through). Proposals: subjects with the chain but no r3
fact, the commonest chain end; checked against ALL gold facts on pairs Re-DocRED annotates, and the held-out facts recovered.
Usage: python3 tools/count_miner.py [volumes]   (env DV_DIR as tools/dataset_volume.py)"""
import os
import sys
import time
import math
import random
import collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'dataset_volume.py')).read()
exec(compile(_src[_src.index('import os'):_src.index('for V in VOL:')].replace('from platypus.inference.distill import _rm_mine', 'pass'), 'dataset_volume', 'exec'))
NMIN = int(os.environ.get('CM_N', 20)); Z = float(os.environ.get('CM_Z', 4.0)); LB = float(os.environ.get('CM_LB', 0.7))


def wilson(p, n, z):
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n) if n else 0.0


for V in VOL:
    t1 = time.time(); F = set()
    for f, _ in per[:V]: F |= f
    F = sorted(F); rng = random.Random(1); held = set(rng.sample(F, len(F) // 5)); obs = [x for x in F if x not in held]
    sub = collections.defaultdict(lambda: collections.defaultdict(set))
    for a, r, b in obs: sub[r][a].add(b); sub['~' + r][b].add(a)
    rels = [r for r in sub if sum(len(v) for v in sub[r].values()) >= 20]; tgt = [r for r in rels if not r.startswith('~')]
    rules = []; ncand = 0
    for r1 in rels:
        for r2 in rels:
            if r2 == '~' + r1 or r1 == '~' + r2: continue                    # r o r-inverse: 'the same as itself'
            ends = {}
            for a, bs in sub[r1].items():
                cs = collections.Counter(c for b in bs for c in sub[r2].get(b, ()) if c != a)
                if cs: ends[a] = cs
            if len(ends) < NMIN: continue
            for r3 in tgt:
                if r3 in (r1, r2) and r1 == r2: continue
                te = [a for a in ends if a in sub[r3]]
                if len(te) < NMIN: continue
                ncand += 1; acc = sum(1 for a in te if ends[a].most_common(1)[0][0] in sub[r3][a]) / len(te)
                lb = wilson(acc, len(te), Z)
                if lb >= LB: rules.append((r1, r2, r3, acc, len(te), lb, ends))
    rules.sort(key=lambda x: -x[5])
    props = {}
    for r1, r2, r3, acc, n, lb, ends in rules:
        if r3 == r1 or r3 == r2: pass
        for a, cs in ends.items():
            if a in sub[r3]: continue
            c = cs.most_common(1)[0][0]
            props.setdefault((a, r3, c), (r1, r2, acc))
    sc = [((a, r, c) in ALLF) for (a, r, c) in props if (a, c) in ALLA]; rec = sum(1 for k in props if k in held)
    trivial = sum(1 for r1, r2, r3, *_ in rules if r3 in (r1, r2))
    P(f"\n== {V} documents: {len(F)} facts ({len(obs)} observed, {len(held)} held out) | relations with 20+ facts (incl. inverses) {len(rels)} | "
      f"candidates tested {ncand} | RULES {len(rules)} ({trivial} with the target among its own hops) | {time.time() - t1:.0f}s")
    for r1, r2, r3, acc, n, lb, _ in rules[:14]: P(f"     {rn(r1.lstrip('~'))}{'^-1' if r1[0] == '~' else ''} o {rn(r2.lstrip('~'))}{'^-1' if r2[0] == '~' else ''} -> {rn(r3)}: {acc:.2f} of {n} (lower bound {lb:.2f})")
    P(f"   PROPOSALS {len(props)}: on annotated pairs {len(sc)}, GOLD {np.mean(sc) if sc else float('nan'):.2f} | HELD-OUT facts recovered {rec} of {len(held)}")
    byr = collections.defaultdict(list)
    for (a, r, c), (r1, r2, acc) in props.items():
        if (a, c) in ALLA: byr[(r1, r2, r)].append((a, r, c) in ALLF)
    for k, v in sorted(byr.items(), key=lambda kv: -len(kv[1]))[:6]: P(f"     by rule {rn(k[0].lstrip('~'))} o {rn(k[1].lstrip('~'))} -> {rn(k[2])}: {len(v)} checkable, gold {np.mean(v):.2f}")
P(f"DONE in {time.time() - T0:.0f}s")
