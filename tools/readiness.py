"""DISCRIMINATING TESTS AND INFERENCE READINESS (15 Nov 2026; tools/readiness.py) - offline, label-free; labels only score.
After an outside review of the project: (1) judge an inference by the plausible ALTERNATIVE that explains the same evidence - our errors are
mostly ASSOCIATED pairs without a relation, so ask whether the pair (and its bridges) is ever linked by a RELATIONAL wording or only by
lists / coordination ('A, B and C'); (2) measure INFERENCE READINESS of the record graph - how much overlapping relational structure it has
- before and after (3) CROSS-DOCUMENT NAME MERGING (variants of one name: 'US' / 'United States' / 'the United States').
  1 DISCRIMINATING TEST  for the GPU path's 200 writes (tools/select_judge.py, code + AA, 0.74 right), documents 0 -> 4/10 (read by the
                         judging time): records A-C, and A-x / x-C for bridges x, split into LIST-LIKE (the words between are only ',' /
                         'and' / 'or' / '(' / ')' / '-', or the span holds 2+ commas) and RELATIONAL (anything else, 1-8 words); signals:
                         relational records of the pair; share of the pair's records that are relational; bridges linked RELATIONALLY on
                         both sides; and 'list only' (the pair co-occurs, never relationally) - AUC right vs wrong (the current judge 0.64)
  2 READINESS            on the records of documents 0 -> 2/10 (relational ones, 1-8 words): entities with 2+ distinct wordings; facts in at
                         least one two-hop path; distinct (wording, wording) pairs with 5+ chains; paths supported by 2+ documents
  3 MERGING              name variants merged label-free (lower case, leading 'the ' dropped, punctuation stripped; an all-capitals
                         acronym joined to the multi-word name whose initials it is when that name is the acronym's commonest co-mention)
                         - readiness again, and the gold relations' coverage (pairs with a Wikidata relation stated as some record)
Usage: python3 tools/readiness.py <train_distant.json.gz> <writes .json>"""
import sys, re, json, types, time, collections
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.compress import _cg_auc
P = lambda *a: print(*a, flush=True); T0 = time.time()
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10)); R = pm._PM['R']; ND = pm._PM['ND']; LAB = pm._PM['LAB']
n1, n2 = int(round(ND * 0.2)), int(round(ND * 0.4))
LISTW = {',', 'and', 'or', '(', ')', '-', ', and', ', or', ') and', ') ,', ', (', 'and the', ', the', ';', '/'}
islist = lambda r: r[3] in LISTW or r[3] == '' or r[7].count(',') >= 2
# 1 DISCRIMINATING TEST
WR = [(tuple(k), bool(t)) for k, t in json.load(open(sys.argv[2]))]; WN = {n for k, _ in WR for n in k}
rel_n = collections.Counter(); all_n = collections.Counter(); nb_rel = collections.defaultdict(set); nb_any = collections.defaultdict(set)
for r in R:
    if r[0] >= n2 or r[1] == r[2] or not (1 <= r[5] <= 8 or r[3] in LISTW): continue
    fk = frozenset((r[1], r[2])); rl = not islist(r)
    all_n[fk] += 1; rel_n[fk] += int(rl)
    for a, b in ((r[1], r[2]), (r[2], r[1])):
        if a in WN: nb_any[a].add(b); (nb_rel[a].add(b) if rl else None)
TYPx = pm._PM['TY']; ent = lambda n: TYPx.get(n, 'X') not in ('TIME', 'NUM')   # bridges that are entities (not dates or numbers: they attach to neighbours)
y = np.array([t for _, t in WR]); s1, s2, s3, s4, s5, s6, s7 = [], [], [], [], [], [], []
for (a, c), t in WR:
    fk = frozenset((a, c)); s1.append(rel_n[fk]); s2.append(rel_n[fk] / all_n[fk] if all_n[fk] else np.nan)
    s3.append(len(nb_rel[a] & nb_rel[c])); s4.append(len(nb_rel[a] & nb_rel[c]) / max(1, len(nb_any[a] & nb_any[c]))); s5.append(-int(all_n[fk] > 0 and rel_n[fk] == 0))
    s6.append(sum(1 for x in nb_any[a] & nb_any[c] if ent(x))); s7.append(sum(1 for x in nb_rel[a] & nb_rel[c] if ent(x)))
au = lambda s: (lambda s_: _cg_auc(s_[y], s_[~y]))(np.nan_to_num(np.array(s, float), nan=-1.0))
lo = np.array([x == -1 for x in s5])
P(f"1 DISCRIMINATING TEST on {len(WR)} writes (right {y.mean():.3f}), records of documents < {n2}: LIST-ONLY pairs (co-occur, never relationally) {lo.mean():.2f} "
  f"(right among them {y[lo].mean() if lo.any() else float('nan'):.2f}, among the rest {y[~lo].mean():.2f})")
for nm, s in (('relational records of the pair', s1), ('share of the pair\'s records that are relational', s2), ('bridges linked RELATIONALLY on both sides', s3),
              ('share of the bridges linked relationally', s4), ('not list-only', s5), ('bridges that are ENTITIES (no dates / numbers)', s6),
              ('entity bridges linked relationally on both sides', s7), ('all bridges (any link; for comparison)', [len(nb_any[a] & nb_any[c]) for (a, c), _ in WR])):
    P(f"   {nm:52s} AUC {au(s):.3f}")
P(f"   (the current judge: documents stating the pair 0.639)  ({time.time() - T0:.0f}s)")
# 2 READINESS, 3 MERGING
def ready(recs, name):
    out = collections.defaultdict(set); wd = collections.defaultdict(set); fac = set()
    for d, a, b, w in recs: out[a].add((w, b, d)); wd[a].add(w); wd[b].add(w); fac.add((a, w, b))
    multi = np.mean([len(v) >= 2 for v in wd.values()]); inpath = set(); pairs = collections.Counter(); pdocs = collections.defaultdict(set); np2 = 0
    for a, L in out.items():
        for (w1, b, d1) in L:
            for (w2, c, d2) in out.get(b, ()):
                if c == a: continue
                inpath.add((a, w1, b)); inpath.add((b, w2, c)); pairs[(w1, w2)] += 1; pdocs[(a, b, c)].update((d1, d2))
    P(f"   READINESS ({name}): {len(wd)} entities, {len(fac)} facts | entities with 2+ wordings {multi:.2f} | facts in a two-hop path {len(inpath) / max(1, len(fac)):.3f} | "
      f"(wording, wording) pairs with 5+ chains {sum(1 for v in pairs.values() if v >= 5)} | paths {len(pdocs)}, from 2+ documents {np.mean([len(v) >= 2 for v in pdocs.values()]) if pdocs else 0:.2f}")
base = [(r[0], r[1], r[2], r[3]) for r in R if r[0] < n1 and r[1] != r[2] and 1 <= r[5] <= 8 and not islist(r)]
ready(base, 'records as they are')
norm = lambda n: re.sub(r"[^\w ]", "", re.sub(r"^the ", "", n.lower())).strip()
nc = collections.Counter(n for d, a, b, w in base for n in (a, b)); canon = {}
for n in nc:
    k = norm(n); canon.setdefault(k, []).append(n)
M = {n: max(v, key=lambda x: nc[x]) for k, v in canon.items() for n in v}
co = collections.defaultdict(collections.Counter)
for d, a, b, w in base: co[a][b] += 1; co[b][a] += 1
for n in list(nc):
    if n.isupper() and 2 <= len(n) <= 6:
        for m, _ in co[n].most_common(3):
            ws = [x for x in re.split(r"\s+", m) if x and x[0].isupper()]
            if len(ws) >= 2 and ''.join(x[0] for x in ws) == n: M[n] = M.get(m, m); break
merged = [(d, M.get(a, a), M.get(b, b), w) for d, a, b, w in base]; merged = [x for x in merged if x[1] != x[2]]
P(f"3 MERGING: {len(nc)} names -> {len(set(M.values()) | (set(nc) - set(M)))} after merging variants ({sum(1 for n in nc if M.get(n, n) != n)} names merged)")
ready(merged, 'names merged')
gold = {fk for fk in LAB}
cov = lambda recs: len({frozenset((a, b)) for d, a, b, w in recs} & gold)
gm = {frozenset((M.get(x, x) for x in fk)) for fk in gold}
P(f"   gold relations stated as some relational record: {cov(base)} before, {len({frozenset((a, b)) for d, a, b, w in merged} & gm)} after merging")
P(f"DONE ({time.time() - T0:.0f}s)")
