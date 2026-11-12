"""REAL-TEXT VIEWS (12 Nov 2026; tools/realtext_views.py) - is there enough material for LABEL-FREE alignment on ordinary text? A view is
a (question, declaration) pair cut from one sentence by the fast channel: the declaration is the sentence up to an object filler, the
question its subject + the frame before that filler ('Margaret Thatcher was born in' <- 'She was born in Grantham'). Views from the
subject-anchored reader (sentences about the document's title, including pronoun-led ones: the title is supplied) and from the cleaned
adjacent records. Counted on the Re-DocRED benchmark documents against its gold relations:
  - views per document, and the share that need the title supplied (pronoun / frame-led sentences)
  - gold relations with at least one view (the ceiling for label-free recall from one-sentence views), and with views in 2+ frames
  - PARAPHRASE LOAD: distinct frames per gold relation type among the views - how many wordings one relation arrives in
Usage: python3 tools/realtext_views.py"""
import os
import collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'reader_tests2.py')).read()
exec(compile(_src[:_src.index('# ---------------- M1: informative frames')], 'reader_tests2', 'exec'))
P = lambda *a: print(*a, flush=True)
rel = {}
for h, t, fl, ps in G['rel']: rel.setdefault((min(h, t), max(h, t)), set()).update(ps)
V = []                                                                      # (doc, a, frame, b, kind)
for di, d in enumerate(bdocs):
    lr = {si: list_runs(t) for si, t in enumerate(d)}
    title = None
    for a, f, b, si, k in anchored(d):
        title = a; first = segment(d[si], 300, 'B'); named = bool(first) and first[0][0] <= 1
        V.append((di, a, f, b, 'anchored, named' if named else 'anchored, TITLE SUPPLIED'))
    for a, f, b, si, k in recs_full(d):
        if k != 1 or (a in lr[si] and b in lr[si]) or ty(a) == 'tD' or ty(b) == 'tD': continue
        V.append((di, a, f, b, 'adjacent'))
kc = collections.Counter(v[4] for v in V)
P(f"VIEWS: {len(V)} in {len(bdocs)} documents ({len(V) / len(bdocs):.1f} per document) | " + " | ".join(f"{k} {n}" for k, n in kc.most_common()))
cov = collections.defaultdict(set); kinds = collections.defaultdict(set)
for di, a, f, b, k in V:
    g = gp(a, b)
    if g:
        for x in g: cov[x].add(f); kinds[x].add(k)
P(f"GOLD RELATIONS WITH A VIEW: {len(cov)} of {len(gold)} ({len(cov) / len(gold):.3f}) | with views in 2+ distinct frames {sum(1 for v in cov.values() if len(v) >= 2)} | "
  f"reached only through a title-supplied view {sum(1 for x in kinds if kinds[x] == {'anchored, TITLE SUPPLIED'})}")
pt = collections.defaultdict(set); pn = collections.Counter()
for x, fs in cov.items():
    for p in rel.get(x, ()): pt[p] |= fs; pn[p] += 1
top = [p for p, _ in pn.most_common(12)]
P("PARAPHRASE LOAD - distinct frames among the views of each gold relation type (commonest types): "
  + " | ".join(f"{G.get('rel_name', {}).get(p, p)}: {len(pt[p])} frames / {pn[p]} relations" for p in top))
ratio = [len(pt[p]) / pn[p] for p in pn if pn[p] >= 3]
P(f"   over relation types with 3+ covered relations ({len(ratio)}): distinct frames per relation, median {np.median(ratio) if ratio else float('nan'):.2f} "
  f"(1.0 = every relation in its own wording; low = shared wordings)")
for p in top[:4]: P(f"   e.g. {G.get('rel_name', {}).get(p, p)}: {sorted(pt[p], key=len)[:8]}")
P("DONE")
