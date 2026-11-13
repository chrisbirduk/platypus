"""NAME UNITS BY DESCRIPTION LENGTH, FROM COUNTS (12 Nov 2026; tools/mdl_units.py) - offline, no effect on the package.
The trunk's binding gain did not find name boundaries (GPU 0.49, stand-in 0.56). The counting version (de Marcken 1996; the MDL view of
word segmentation): merging two adjacent words into one lexicon unit SAVES bits when they co-occur far more than chance - the corpus
description shrinks by about c12 * log2(c12 * N / (c1 * c2)) - and COSTS the new entry's spelling (8 bits a byte). Counted, case kept, on
local Wikipedia + the Re-DocRED dev text. Scored against the dev set's gold mentions as tools/trunk_guides.py T1:
  (a) joins inside capitalised runs: same mention vs boundary - AUC of the merge gain, against trunk_guides' surprise (0.565) and binding
      gain (0.560 offline / 0.492 GPU);
  (b) exact mention spans (documents 251-500): capitalised runs (F1 0.698) split where the merge gain is below a threshold fitted on
      documents 1-250; and the UNION with the plain runs (guidance that only adds).
Usage: python3 tools/mdl_units.py [documents]"""
import os
import re
import sys
import json
import math
import time
import collections
import numpy as np
P = lambda *a: print(*a, flush=True); T0 = time.time()
ND = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
DEV = os.environ.get('TG_DEV', SCR + '/bench_dl/dev_revised.json')
dev = json.load(open(DEV))[:ND]
raw = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin', 'rb').read().decode('utf-8', 'replace')
TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*|[^\sA-Za-z0-9]")
uni = collections.Counter(); big = collections.Counter()
def count(ws):
    for i, w in enumerate(ws):
        uni[w] += 1
        if i: big[(ws[i - 1], w)] += 1
for line in raw.split('\n'): count(TOK.findall(line))
for d in dev:
    for s in d['sents']: count(s)
N = sum(uni.values()); P(f"counted {N} tokens, {len(uni)} types, {len(big)} bigrams  ({time.time() - T0:.0f}s)")
def gain(a, b):
    c12 = big[(a, b)]
    if c12 == 0: return -8.0 * (len(a) + len(b) + 1)
    return c12 * math.log2(c12 * N / (uni[a] * uni[b])) - 8.0 * (len(a) + len(b) + 1)
cap = lambda w: w[:1].isupper() or w[:1].isdigit()
def runs(toks):
    out = []; cur = None
    for k, w in enumerate(toks):
        j = cur is not None and (cap(w) or (w == 'of' and k + 1 < len(toks) and cap(toks[k + 1])))
        if cap(w) or (j and w == 'of'):
            if j: cur[1] = k
            else:
                if cur: out.append(tuple(cur))
                cur = [k, k]
        else:
            if cur: out.append(tuple(cur))
            cur = None
    if cur: out.append(tuple(cur))
    return [(a, b) for a, b in out if toks[b] != 'of']
def auc(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    if not len(p) or not len(n): return float('nan')
    r = np.concatenate([p, n]).argsort(kind='stable').argsort().astype(float) + 1
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
SAME, BND = [], []; SP, BP = [], []; DOCS = []
pmi = lambda a, b: math.log2((big[(a, b)] + 0.1) * N / (uni[a] * uni[b] + 1e-9))
for d in dev:
    S = []
    for si, s in enumerate(d['sents']):
        gold = set(); mk = {}; starts = set()
        for ei, vs in enumerate(d['vertexSet']):
            for m in vs:
                if m['sent_id'] != si: continue
                gold.add((m['pos'][0], m['pos'][1] - 1)); starts.add(m['pos'][0])
                for k in range(m['pos'][0], m['pos'][1]): mk.setdefault(k, (ei, m['pos'][0]))
        S.append((s, gold))
        for a, b in runs(s):
            for k in range(a + 1, b + 1):
                if not cap(s[k]): continue
                g = gain(s[k - 1], s[k]); same = k in mk and (k - 1) in mk and mk[k] == mk[k - 1] and k not in starts
                (SAME if same else BND).append(g); (SP if same else BP).append(pmi(s[k - 1], s[k]))
    DOCS.append(S)
P(f"(a) joins inside capitalised runs: same mention {len(SAME)} vs boundary {len(BND)} | AUC of the MERGE GAIN {auc(SAME, BND):.3f}, of PMI alone (no lexicon cost) {auc(SP, BP):.3f} "
  f"(median {np.median(SAME):.1f} vs {np.median(BND):.1f} bits) | earlier: trunk surprise 0.565, binding gain 0.560 (stand-in) / 0.492 (GPU)")
SC = {'merge gain': gain, 'PMI': pmi}
def spans(s, th, sc='merge gain'):
    out = []
    for a, b in runs(s):
        s0 = a
        for k in range(a + 1, b + 1):
            if cap(s[k]) and SC[sc](s[k - 1], s[k]) < th: out.append((s0, k - 1)); s0 = k
        out.append((s0, b))
    return set(out)
def prf(docs, th, union=False, sc='merge gain'):
    tp = npred = ngold = 0
    for S in docs:
        for s, gold in S:
            p = spans(s, th, sc) if th is not None else set(runs(s))
            if union: p |= set(runs(s))
            tp += len(gold & p); npred += len(p); ngold += len(gold)
    pr, rc = tp / max(1, npred), tp / max(1, ngold); return pr, rc, 2 * pr * rc / max(1e-9, pr + rc)
h = len(DOCS) // 2
best = max((prf(DOCS[:h], th)[2], th) for th in [-1e9] + list(np.arange(-200, 400.1, 10)))
p0, r0, f0 = prf(DOCS[h:], None); p1, r1, f1 = prf(DOCS[h:], best[1]); p2, r2, f2 = prf(DOCS[h:], best[1], True)
P(f"(b) exact mention spans (documents {h + 1}-{len(DOCS)}): capitalised runs P {p0:.3f} R {r0:.3f} F1 {f0:.3f} | split where the merge gain < {best[1]:.0f} bits "
  f"(fitted on 1-{h}) P {p1:.3f} R {r1:.3f} F1 {f1:.3f} | UNION P {p2:.3f} R {r2:.3f} F1 {f2:.3f}")
bp = max((prf(DOCS[:h], th, sc='PMI')[2], th) for th in [-1e9] + list(np.arange(-10, 20.1, 0.5)))
p3, r3, f3 = prf(DOCS[h:], bp[1], sc='PMI'); p4, r4, f4 = prf(DOCS[h:], bp[1], True, sc='PMI')
P(f"    split where PMI < {bp[1]:.1f} (fitted on 1-{h}) P {p3:.3f} R {r3:.3f} F1 {f3:.3f} | UNION P {p4:.3f} R {r4:.3f} F1 {f4:.3f}")
P(f"DONE ({time.time() - T0:.0f}s)")
