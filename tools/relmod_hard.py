"""HARD RULE WORLD for the relational module (11 Nov 2026; tools/relmod_hard.py): each failure mode from the composition analysis, planted.

Families of 4 generations (root, 2 children, 4 grandchildren, 8 great-grandchildren) and org charts (root, 3 managers, 9 staff):
  three-hop      great-grandchild = grandchild o child = child o grandchild (and, at a second round, from proposed grandchild facts)
  inverse        'is the parent of' (one-to-many, the inverse of child-of; a singular operator's inverse)
  except self    sibling = child o parent-of, minus self (a matrix cannot express 'not this person')
  paraphrase     'is supervised by' / 'reports to': each supervision fact said one way, the other, or both (40/30/30)
  partial rule   'is mentored by': the skip level for 70% of staff, a random person for 30%
  arbitrary      'works with' (symmetric, 3 each) and 'lives near' (random)
  transfer       30% of families and charts are NEW: only their one-hop facts (child-of, parent-of, supervision) are seen
Half of every composite (grandchild, great-grandchild, sibling, overseen, mentored) is held out in the training groups too. Each rule
found is graded by its TRUE confidence on the full world (TRUE >= 0.9, PARTIAL 0.5-0.9, FALSE < 0.5); proposals are graded against
the full truth; recall of the held-out composites, training groups and new groups apart. Run with the fixes off (as built at first) and
on (paraphrase merge, data-subspace distance, except-self, orthogonal one-to-one operators), and a second round in which round 1's
proposals are added as facts (as step 4 would write them). Usage: python3 tools/relmod_hard.py [seed] [steps] [noise]"""
import os
import sys
import zlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from platypus.inference.distill import _rm_mine  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
NOISE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
torch.set_num_threads(1); rs = np.random.RandomState(seed)
NF = 30
T = {}                                                                         # full truth: relation -> {a: set of b}
seen = set()                                                                   # (rel, a, b) planted
new_grp = set()
add = lambda r, a, b: T.setdefault(r, {}).setdefault(a, set()).add(b)
hold = lambda a, r: zlib.crc32(f"{a}|{r}".encode()) % 2 == 0
people = []
for f in range(NF):
    new = f >= int(NF * 0.7); p = lambda g, i: f"F{f}g{g}n{i}"
    gens = [[p(0, 0)], [p(1, i) for i in range(2)], [p(2, i) for i in range(4)], [p(3, i) for i in range(8)]]
    par = {gens[g][i]: gens[g - 1][i // 2] for g in range(1, 4) for i in range(len(gens[g]))}
    for c, q in par.items():
        add('is a child of', c, q); add('is the parent of', q, c)
        if q in par: add('is a grandchild of', c, par[q])
        if q in par and par[q] in par: add('is a great-grandchild of', c, par[par[q]])
    for c in par:
        for s in par:
            if s != c and par[s] == par[c]: add('is a sibling of', c, s)
    people += [x for g in gens for x in g]
    if new: new_grp.update(x for g in gens for x in g)
for f in range(NF):
    new = f >= int(NF * 0.7); root = f"O{f}r"; mids = [f"O{f}m{i}" for i in range(3)]; staff = [(m, f"O{f}s{i}{j}") for i, m in enumerate(mids) for j in range(3)]
    for m in mids: add('SUP', m, root)
    for m, s in staff:
        add('SUP', s, m); add('is overseen by', s, root)
        add('is mentored by', s, root if rs.rand() < 0.7 else f"O{rs.randint(NF)}m{rs.randint(3)}")
    people += [root] + mids + [s for _, s in staff]
    if new: new_grp.update([root] + mids + [s for _, s in staff])
for a in people:
    for b in rs.choice(len(people), 3, replace=False):
        if people[b] != a: add('works with', a, people[b]); add('works with', people[b], a)
    add('lives near', a, people[rs.randint(len(people))])
ONEHOP = {'is a child of', 'is the parent of', 'SUP', 'works with', 'lives near'}
pairs = {}
for r, m in T.items():
    for a, bs in m.items():
        for b in bs:
            if r == 'is a child of' and NOISE > 0 and rs.rand() < NOISE: b = people[rs.randint(len(people))]
            if r not in ONEHOP and (a in new_grp or hold(a, r)): continue     # composites: never for new groups, half held out elsewhere
            if r == 'SUP':
                u = rs.rand()
                for ph in (['is supervised by'] if u < 0.4 else ['reports to'] if u < 0.7 else ['is supervised by', 'reports to']):
                    pairs.setdefault(ph, set()).add((a, b))
                continue
            pairs.setdefault(r, set()).add((a, b))
T['is supervised by'] = T['reports to'] = T.pop('SUP')
heldout = {(r, a) for r, m in T.items() if r not in ONEHOP | {'is supervised by', 'reports to'} for a in m if a in new_grp or hold(a, r)}


def tconf(r1, r2, r):                                                          # true confidence of r1 o r2 -> r on the full world
    n = k = 0
    for a, bs in T[r1].items():
        cs = {c for b in bs for c in T.get(r2, {}).get(b, ()) if c != a}
        for c in cs: n += 1; k += c in T.get(r, {}).get(a, ())
    return k / n if n else 0.0


DER = set()


def run(tag, pairs, **kw):
    R = _rm_mine(pairs, STEPS, seed * 100 + 7, **kw)
    canon = {a: b for a, b in R['merged']}; real = lambda r: r if r in T else canon.get(r, r)
    grade = lambda t: tconf(real(t[0]), real(t[1]), real(t[2]))
    gs = [(t, grade(t)) for t in R['rules']]
    lab = lambda g: 'TRUE' if g >= 0.9 else 'PARTIAL' if g >= 0.5 else 'FALSE'
    print(f"== {tag} | {R['N']} names, {len(R['rels'])} relations, {R['nfacts']} facts | merged {R['merged']} | one-to-one (orthogonal) {R.get('ortho')} | "
          f"one-hop {R['one']:.2f} | candidates beating the null {R['cand']} (other metric {R['cand_alt']}) | RULES {len(gs)}: "
          f"TRUE {sum(g >= 0.9 for _, g in gs)}, PARTIAL {sum(0.5 <= g < 0.9 for _, g in gs)}, FALSE {sum(g < 0.5 for _, g in gs)}")
    for t, g in sorted(gs, key=lambda x: -x[1]):
        print(f"     {t[0]:24s} o {t[1]:24s} -> {t[2]:26s} dist {t[3]:.2f} null {t[4]:.2f} | predicts {t[6]:.2f} of {t[7]} | true confidence {g:.2f} {lab(g)}")
    props = {}; wr = {}                                                         # a stored-hop chain when both hops are held, else the latent one
    for a, r, c, mg, st, fc, w, dep in R['props']:
        props.setdefault((a, real(r)), set()).update(st if st else [c])
        if w: wr.setdefault((a, r), set()).update(st)                           # WRITTEN (step 4): a rule verified 0.9+, premises stored
    ok = lambda a, r, cs: any(c in T.get(r, {}).get(a, ()) for c in cs)
    np_ = len(props); prec = sum(ok(a, r, cs) for (a, r), cs in props.items()) / max(1, np_)
    by = {}
    for (r, a) in heldout:
        k = (r, 'new' if a in new_grp else 'train'); by.setdefault(k, [0, 0]); by[k][1] += 1
        by[k][0] += ok(a, r, props.get((a, r), set()) | {b for b2, (q, a2) in [((x[2]), (x[0], x[1])) for x in DER] if q == r and a2 == a for b in [b2]})
    print(f"   PROPOSALS {np_} (subject, relation), precision {prec:.2f}; would be WRITTEN (rule verified 0.9+, premises stored) {len(wr)}, precision "
          f"{sum(ok(a, real(r), cs) for (a, r), cs in wr.items()) / max(1, len(wr)):.2f} | HELD-OUT RECALL (proposed or already written): "
          + " | ".join(f"{r} {g} {v[0]}/{v[1]}" for (r, g), v in sorted(by.items())))
    return R, props, real, wr


print(f"== seed {seed}, {STEPS} steps, noise {NOISE:g} | {len(people)} people, new groups {len(new_grp)} | facts seen: " + ", ".join(f"{r} {len(v)}" for r, v in sorted(pairs.items())))
sys.stdout.flush()
run('AS BUILT (no fixes)', pairs, merge=False, data_dist=False, xself=False, ortho=False, guards=False); sys.stdout.flush()
R, props, real, wr = run('FIXED', pairs); sys.stdout.flush()
pairs2 = {r: set(v) for r, v in pairs.items()}                                # ROUND 2: round 1's rules-of-0.9+ proposals WRITTEN (step 4),
for (a, r), cs in wr.items():                                                 #   tagged as derived: hops for later chains, never evidence
    for c in cs: pairs2.setdefault(r, set()).add((a, c)); DER.add((r, a, c))
run('FIXED, ROUND 2 (round 1 written facts as hops, not evidence)', pairs2, derived=set(DER)); sys.stdout.flush()
run('ROUND 2 WITHOUT PROVENANCE (written facts counted as evidence)', pairs2, guards=False)
