"""HOW FAR IS THE LEARNED EXTRACTOR FROM SUPPORTING INFERENCE? (13 Nov 2026; tools/kept_rules.py) - offline, no effect on the package.
The learner of tools/consistency_teacher.py (label-free teacher; held-out AUC 0.63) keeps the top share of ALL candidate records (every pair of
names in one sentence, Re-DocRED train + dev). Do rules come out of what it keeps? Counting miner of tools/count_miner.py (exact chains,
n >= 20 subjects, Wilson z 4 >= 0.7, inverses included), facts = kept records:
  A  kept records that are related, labelled with their GOLD relation type (relation identity solved, the extractor's misses kept missing)
  B  ALL kept records with their wording as the relation (what the store would have: no relation identity, noise included)
  C  ceiling: every gold fact of the same documents (tools/count_miner.py: 36 rules at 3,553 documents)
For A and B: rules, and proposals checked against gold on annotated pairs. Kept shares: 10%, 25%, 50%.
Usage: python3 tools/kept_rules.py"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_ct = open(os.path.join(HERE, 'consistency_teacher.py')).read()
exec(compile(_ct[:_ct.index('# ---- SELF-TRAINING ROUNDS')], 'consistency_teacher', 'exec'))
code = {}; ALLF = set(); ALLA = set()
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    for l in d['labels']:
        code.setdefault((di, cn[l['h']], cn[l['t']]), l['r']); ALLF.add((cn[l['h']], l['r'], cn[l['t']]))
    ALLA |= {(a, b) for a in set(cn) for b in set(cn) if a != b}
lr_ = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, teacher_tr >= np.percentile(teacher_tr, 75))
Xall = feats(np.arange(len(C))); s_all = lr_.decision_function(Xall)
P(f"learner applied to all {len(C)} candidates (related {y.mean():.3f}) | AUC on all {auc(s_all[y], s_all[~y]):.3f}")
def wil(p, n, z): return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n) if n else 0.0
def mine(facts, tag, nmin=20, z=4.0, lb=0.7):
    sub = collections.defaultdict(lambda: collections.defaultdict(set))
    for a, r, b in facts: sub[r][a].add(b); sub['~' + r][b].add(a)
    rels = [r for r in sub if sum(len(v) for v in sub[r].values()) >= 20]; tgt = [r for r in rels if not r.startswith('~')]; rules = []
    for r1 in rels:
        for r2 in rels:
            if r2 == '~' + r1 or r1 == '~' + r2: continue
            ends = {}
            for a, bs in sub[r1].items():
                cs = collections.Counter(x for b in bs for x in sub[r2].get(b, ()) if x != a)
                if cs: ends[a] = cs
            if len(ends) < nmin: continue
            for r3 in tgt:
                te_ = [a for a in ends if a in sub[r3]]
                if len(te_) < nmin: continue
                acc = sum(1 for a in te_ if ends[a].most_common(1)[0][0] in sub[r3][a]) / len(te_)
                if wil(acc, len(te_), z) >= lb: rules.append((r1, r2, r3, acc, len(te_), ends))
    props = {}
    for r1, r2, r3, acc, n, ends in rules:
        for a, cs in ends.items():
            if a not in sub[r3]: props.setdefault((a, r3, cs.most_common(1)[0][0]), 1)
    chk = [(k in ALLF) for k in props if (k[0], k[2]) in ALLA]
    P(f"   {tag}: {len(facts)} facts, {len(rels)} relations with 20+ | RULES {len(rules)}"
      + "".join(f"\n        '{r1[:30]}' o '{r2[:30]}' -> '{r3[:30]}': {acc:.2f} of {n}" for r1, r2, r3, acc, n, _ in sorted(rules, key=lambda r: -r[4])[:5])
      + f"\n      proposals {len(props)}, on annotated pairs {len(chk)}, gold {np.mean(chk) if chk else float('nan'):.2f}")
mine(sorted(ALLF), "C  CEILING: every gold fact")
for share in (0.10, 0.25, 0.50):
    kept = np.where(s_all >= np.percentile(s_all, 100 * (1 - share)))[0]
    rel = [i for i in kept if y[i]]
    P(f"\n== learner keeps the top {share:.0%}: {len(kept)} records, related {y[kept].mean():.3f} | gold facts covered "
      f"{len({(C[i]['cA'], code.get((C[i]['di'], C[i]['cA'], C[i]['cB'])) or code.get((C[i]['di'], C[i]['cB'], C[i]['cA'])), C[i]['cB']) for i in rel})} of {len(ALLF)}")
    FA = set()
    for i in rel:
        c = C[i]; r = code.get((c['di'], c['cA'], c['cB'])); r2 = code.get((c['di'], c['cB'], c['cA']))
        if r: FA.add((c['cA'], r, c['cB']))
        elif r2: FA.add((c['cB'], r2, c['cA']))
    mine(sorted(FA), "A  kept related records, GOLD relation types")
    mine(sorted({(C[i]['cA'], C[i]['f'], C[i]['cB']) for i in kept}), "B  ALL kept records, wording as the relation")
P(f"DONE ({time.time() - T0:.0f}s)")
