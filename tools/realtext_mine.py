"""RULES FROM REAL TEXT WITH TODAY'S READING PIECES (13 Nov 2026; tools/realtext_mine.py) - offline, no effect on the package.
The relational module (_rm_mine, every guard) mines rules over records read from real text: the Re-DocRED benchmark documents (embedded, 262)
plus local Wikipedia. Each step adds one reading piece to the last; rules, proposals and their gold rate on benchmark entities reported:
  R0  the reader as it stands: cleaned adjacent records (lists and neighbour dates dropped), subject-anchored records, and records credited to
      the document's title when a sentence opens with a pronoun or frame material (the title rule for pronouns, already in the reader)
  R1  + LINKING MENTIONS within a document: a filler that is the last word(s) of an earlier longer filler ('Thatcher' -> 'Margaret Thatcher')
      or its initials ('AGF' -> 'Asian Games Federation') becomes that filler, so one entity is one name
  R2  + ONE RELATION, MANY WORDINGS BY SHARED PAIRS (counting, no trunk): frames with 15+ records that connect the same name pairs (2+ pairs in
      common, Jaccard >= 0.15 of their pair sets, across documents) are merged (union-find); mining over the merged frames
  ORACLE  the ceiling if relation identity were solved: every R1 record whose pair is a gold relation gets that gold relation type as its
      frame (other records dropped) - shows whether relation identity or the amount of text blocks rules
Usage: python3 tools/realtext_mine.py"""
import os
import re
import sys
import collections
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'reader_tests2.py')).read()
exec(compile(_src[:_src.index('base = [(a, f, b) for a, f, b, pv, k in BR if k in (0, 1)]')], 'reader_tests2', 'exec'))
P(f"\n================ RULES FROM REAL TEXT  ({time.time() - T0:.0f}s) | benchmark documents {len(bdocs)}, local Wikipedia documents {len(wdocs)}")


def doc_records(d):
    lr = {si: list_runs(t) for si, t in enumerate(d)}; out = []
    for a, f, b, si, k in recs_full(d):
        if k == 1 and ((a in lr[si] and b in lr[si]) or ty(a) == 'tD' or ty(b) == 'tD'): continue
        if k in (0, 1): out.append((a, f, b))
    out += [(a, f, b) for a, f, b, si, k in anchored(d)]
    return out


def canon_map(d):
    """within one document: a filler that is the tail of an earlier longer filler, or its initials, maps to that filler"""
    seen = []; mp = {}
    for toks in d:
        for x0, x1 in segment(toks, 300, 'B'):
            x = tuple(toks[x0:x1])
            if x in mp or x in seen: continue
            tgt = None
            for y in reversed(seen):
                if len(y) > len(x) and y[-len(x):] == x and x[-1][:1].isupper(): tgt = y; break
                ini = ''.join(w[0] for w in y if w[:1].isupper())
                if len(x) == 1 and len(ini) >= 2 and x[0] == ini: tgt = y; break
            if tgt is not None: mp[x] = tgt
            else: seen.append(x)
    return mp


R0, R1 = [], []; nlink = 0
for d in bdocs + wdocs:
    rec = doc_records(d); R0 += rec; mp = canon_map(d); nlink += len(mp)
    R1 += [(mp.get(a, a), f, mp.get(b, b)) for a, f, b in rec]
P(f"records: R0 {len(R0)} | R1 {len(R1)} ({nlink} mentions linked to an earlier longer name; e.g. "
  + ", ".join(f"{' '.join(k)} -> {' '.join(v)}" for d in bdocs[:40] for k, v in list(canon_map(d).items())[:1])[:400] + ")")
fc0 = collections.Counter(f for a, f, b in R0); fc1 = collections.Counter(f for a, f, b in R1)
mine_report(R0, set(f for f, _ in fc0.most_common(60)), "R0 the reader as it stands, 60 commonest frames")
mine_report(R1, set(f for f, _ in fc1.most_common(60)), "R1 + linking mentions, 60 commonest frames")
# ---- R2: merge frames that connect the same name pairs
pairs = collections.defaultdict(set)
for a, f, b in R1: pairs[f].add((a, b))
big = [f for f, c in fc1.items() if c >= 15]; par = {f: f for f in big}
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
inv = collections.defaultdict(set)
for f in big:
    for p in pairs[f]: inv[p].add(f)
cand = collections.Counter()
for p, fs in inv.items():
    fs = sorted(fs)
    if len(fs) > 30: continue
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)): cand[(fs[i], fs[j])] += 1
nm = 0
for (f, g), c in cand.items():
    if c >= 2 and c / len(pairs[f] | pairs[g]) >= 0.15 and find(f) != find(g): par[find(f)] = find(g); nm += 1
groups = collections.defaultdict(list)
for f in big: groups[find(f)].append(f)
lab = {}
for r, fs in groups.items():
    rep = max(fs, key=lambda f: fc1[f])
    for f in fs: lab[f] = rep if len(fs) == 1 else f"[{rep}]+{len(fs) - 1}"
multi = sorted([fs for fs in groups.values() if len(fs) > 1], key=lambda fs: -sum(fc1[f] for f in fs))
P(f"R2 merges: {nm} joins, {len(multi)} merged groups of frames | largest: " + " || ".join(" | ".join(sorted(fs, key=lambda f: -fc1[f])[:6]) for fs in multi[:5]))
R2 = [(a, lab.get(f, f), b) for a, f, b in R1]; fc2 = collections.Counter(f for a, f, b in R2)
mine_report(R2, set(f for f, _ in fc2.most_common(60)), "R2 + merged wordings (shared name pairs), 60 commonest frames")
# ---- ORACLE: gold relation types as frames
RO = []
for a, f, b in R1:
    g = gp(a, b)
    if g:
        ps = gold.get(g[0]); code = sorted(ps)[0] if ps else None
        if code: RO.append((a, code, b))
fco = collections.Counter(f for a, f, b in RO)
P(f"ORACLE: {len(RO)} R1 records on gold pairs, {len(fco)} gold relation types; commonest {fco.most_common(8)}")
mine_report(RO, set(fco), "ORACLE gold relation types as frames (relation identity solved)")
P(f"DONE in {time.time() - T0:.0f}s")
