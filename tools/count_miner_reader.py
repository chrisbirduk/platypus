"""THE COUNTING MINER ON WHAT THE READER RECORDS (13 Nov 2026; tools/count_miner_reader.py) - offline, no effect on the package.
tools/count_miner.py found real rules in Re-DocRED GOLD facts by exact chain counts. Here the same counting miner (n >= CM_N subjects, Wilson
lower bound >= CM_LB at z = CM_Z) on the records the READER makes from text (tools/realtext_mine.py: R1 = cleaned adjacent + subject-anchored
records, mentions linked within a document; frames as relations, inverses included), over the embedded benchmark plus local Wikipedia; and
on the ORACLE records (R1 records on gold pairs labelled with their gold relation type). Proposals checked as tools/realtext_mine.py does:
is the proposed pair a gold pair of the benchmark (any relation).
Usage: python3 tools/count_miner_reader.py"""
import os
import sys
import math
import collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['RTM_ONLY'] = 'none'
_src = open(os.path.join(HERE, 'realtext_mine.py')).read()
exec(compile(_src[:_src.index('# ---- R2:')], 'realtext_mine', 'exec'))
NMIN = int(os.environ.get('CM_N', 20)); Z = float(os.environ.get('CM_Z', 4.0)); LB = float(os.environ.get('CM_LB', 0.7))
wil = lambda p, n: (p + Z * Z / (2 * n) - Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))) / (1 + Z * Z / n) if n else 0.0


def count_mine(recs, frames, tag):
    sub = collections.defaultdict(lambda: collections.defaultdict(set))
    for a, f, b in recs:
        if f in frames and a != b: sub[f][a].add(b); sub['~' + str(f)][b].add(a)
    rels = list(sub); tgt = [r for r in rels if not str(r).startswith('~')]; rules = []; nc = 0
    for r1 in rels:
        for r2 in rels:
            if str(r2) == '~' + str(r1) or str(r1) == '~' + str(r2): continue
            ends = {}
            for a, bs in sub[r1].items():
                cs = collections.Counter(c for b in bs for c in sub[r2].get(b, ()) if c != a)
                if cs: ends[a] = cs
            if len(ends) < NMIN: continue
            for r3 in tgt:
                te = [a for a in ends if a in sub[r3]]
                if len(te) < NMIN: continue
                nc += 1; acc = sum(1 for a in te if ends[a].most_common(1)[0][0] in sub[r3][a]) / len(te)
                if wil(acc, len(te)) >= LB: rules.append((r1, r2, r3, acc, len(te), ends))
    props = {}
    for r1, r2, r3, acc, n, ends in rules:
        for a, cs in ends.items():
            if a not in sub[r3]: props.setdefault((a, r3, cs.most_common(1)[0][0]), (r1, r2))
    g = [x for x in (gp(a, c) for a, r, c in props) if x is not None]; g = [bool(x) for x in g]
    sh = lambda r: (' '.join(r) if isinstance(r, tuple) else str(r))[:40]
    P(f"\n{tag}: {len(recs)} records, {len(sub) // 2} frames | candidates {nc} | RULES {len(rules)}"
      + "".join(f"\n     '{sh(r1)}' o '{sh(r2)}' -> '{sh(r3)}': {acc:.2f} of {n}" for r1, r2, r3, acc, n, _ in sorted(rules, key=lambda x: -x[4])[:12])
      + f"\n   proposals {len(props)}, on benchmark entities {len(g)}, a GOLD PAIR {np.mean(g) if g else float('nan'):.2f}")


count_mine(RO, set(fco), "ORACLE (gold relation types as frames)")
count_mine(R1, set(f for f, _ in fc1.most_common(60)), "R1 the reader's records, 60 commonest frames")
count_mine(R1, set(f for f, _ in fc1.most_common(300)), "R1 the reader's records, 300 commonest frames")
P(f"DONE in {time.time() - T0:.0f}s")
