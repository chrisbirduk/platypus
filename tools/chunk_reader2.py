"""FAST CHANNEL, NEXT STEP (11 Nov 2026; tools/chunk_reader2.py): two changes suggested by language structure, scored offline against the
Re-DocRED gold relations like tools/chunk_reader.py (whose counts, segmentation and scoring it reuses):
  CANON   frames simplified: fillers inside a frame become a slot ('was born on [F] in'), so frame variants share their statistics (frames
          follow Zipf's law too); the nested record is produced anyway by the adjacent pair
  REF     reference by salience (centering theory): a sentence that opens with frame material ('He was born in', 'The company was founded by')
          gets the document's CURRENT CENTRE as its subject, the referring word dropped from the frame; the centre is the document's first
          filler ('doc') or the subject filler of the latest sentence that opened with one ('subj')
Then the relational module (_rm_mine, every guard) mines rules on local Wikipedia + the benchmark with the simplified frames, proposals
scored against gold. Usage: python3 tools/chunk_reader2.py"""
import os
import re
import sys
import time
import collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'chunk_reader.py')).read()
exec(_src[:_src.index('\nbsents = ')])                                       # counts, segment(), score() of tools/chunk_reader.py
from platypus.inference.distill import _rm_mine  # noqa: E402


def fstr(toks, s, e, inner, canon):
    if not canon: return ' '.join(toks[s:e])
    parts = []; i = s
    for x0, x1 in inner: parts += toks[i:x0] + ['[F]']; i = x1
    return ' '.join(parts + toks[i:e])


def doc_records(sents, canon=False, ref=None, K=300, skip=3, maxgap=12):
    """records of one document's sentences, with the centre carried from sentence to sentence"""
    out = []; centre = None
    for toks in sents:
        R = segment(toks, K, 'B')
        if centre is None and R: centre = tuple(toks[R[0][0]:R[0][1]])
        for k in range(1, skip + 1):
            for i in range(len(R) - k):
                (a0, a1), (b0, b1) = R[i], R[i + k]
                if 1 <= b0 - a1 <= maxgap: out.append((tuple(toks[a0:a1]), fstr(toks, a1, b0, R[i + 1:i + k], canon), tuple(toks[b0:b1])))
        if ref and centre is not None and (not R or R[0][0] >= 2):          # opens with frame material: 'He was ...', 'The company was ...'
            for j in range(min(skip, len(R))):
                b0, b1 = R[j]
                if b0 - 1 > maxgap: break
                g = fstr(toks, 1, b0, R[:j], canon)
                if g and tuple(toks[b0:b1]) != centre: out.append((centre, g, tuple(toks[b0:b1])))
        if ref == 'subj' and R and R[0][0] <= 1: centre = tuple(toks[R[0][0]:R[0][1]])
    return out


bdocs = [[TOK.findall(s) for s in SPL.split(d)] for d in docs]
print("BASELINE (tools/chunk_reader.py, variant B, K 300, fillers up to 3 apart): 325 gold relations, precision 0.26", flush=True)
res = {}
for canon in (False, True):
    for ref in (None, 'doc', 'subj'):
        recs = [r for d in bdocs for r in doc_records(d, canon, ref)]
        fc = collections.Counter(r[1] for r in recs)
        print(f"  frames {'SIMPLIFIED' if canon else 'as read'}, reference {ref or 'off'}: distinct frames {len(fc)}, records whose frame occurs 5+ times "
              f"{sum(v for v in fc.values() if v >= 5) / max(1, len(recs)):.2f}", flush=True)
        res[(canon, ref)] = score(recs, f"    canon {canon}, ref {ref}")

# relational module on real text: Wikipedia articles + the benchmark, simplified frames, reference by the latest subject
wdocs = [[TOK.findall(s) for s in SPL.split(a)] for a in re.split(r'\n = [^=\n][^\n]*? = \n', raw)]
t0 = time.time(); best_ref = max(('doc', 'subj', None), key=lambda r: res[(True, r)][0])
allrec = [r for d in wdocs + bdocs for r in doc_records(d, True, best_ref)]
fc = collections.Counter(r[1] for r in allrec); top = set(f for f, _ in fc.most_common(60)); deg = collections.Counter()
for a, f, b in allrec:
    if f in top: deg[a] += 1; deg[b] += 1
keep = set(n for n, _ in deg.most_common(4000)); pairs = {}; nf = 0
for a, f, b in allrec:
    if f in top and a in keep and b in keep and a != b and nf < 20000: pairs.setdefault(f, set()).add((' '.join(a), ' '.join(b))); nf += 1
print(f"RELATIONAL MODULE ON REAL TEXT (simplified frames, reference '{best_ref}'): {len(allrec)} records from {len(wdocs)} Wikipedia articles + "
      f"{len(bdocs)} benchmark documents; mining {nf} facts in the 60 commonest frames", flush=True)
R = _rm_mine(pairs, 800, 919)
def isgold(a, c):
    ea, ec = ents_of(a.split()), ents_of(c.split())
    if not ea or not ec: return None
    return any(x != y and (min(x, y), max(x, y)) in gold for x in ea for y in ec)
pr = [(p[0], p[1], c) for p in R['props'] for c in (p[4] or [p[2]])]; g1 = [g for g in (isgold(a, c) for a, r, c in pr) if g is not None]
wr = [(p[0], p[1], c) for p in R['props'] if p[6] for c in p[4]]; g2 = [g for g in (isgold(a, c) for a, r, c in wr) if g is not None]
print(f"   {R['N']} fillers, {len(R['rels'])} frames" + (f", paraphrases merged {R['merged'][:6]}" if R['merged'] else "") + f" | candidates beating the null {R['cand']} | RULES {len(R['rules'])}:", flush=True)
for t in R['rules'][:12]: print(f"     '{t[0]}' o '{t[1]}' -> '{t[2]}' dist {t[3]:.2f} null {t[4]:.2f} | predicts {t[6]:.2f} of {t[7]}", flush=True)
print(f"   PROPOSALS {len(pr)}, on benchmark entities {len(g1)}, gold {np.mean(g1) if g1 else float('nan'):.2f} | would be written {len(wr)}, on benchmark entities "
      f"{len(g2)}, gold {np.mean(g2) if g2 else float('nan'):.2f} | {time.time() - t0:.0f}s", flush=True)
for p in pr[:8]: print('     e.g. proposed', p)
