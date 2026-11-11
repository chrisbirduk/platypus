"""FAST SEQUENCE CHANNEL as a reader (11 Nov 2026; tools/chunk_reader.py): frames are the short COMMON word sequences, fillers what varies.

Counted on local Wikipedia (the minipile cache) plus the benchmark text, no labels, no word lists:
  FRAME MATERIAL  a word type is frame material when it is among the K most frequent types (connector words: 'is', 'a', 'town', 'in')
                  or when it rarely stands where the text is open (its Harris slot score is low - see below).
  SLOT SCORE      Harris / infant word segmentation: the entropy of what follows a word's left context; a word that usually stands
                  where many words could stand (after 'in', 'of', 'the') is slot material - a filler.
  ENTITY STAT     how often the type is capitalised when it is not the first word of a sentence (a statistic of the text, not a
                  name pattern); digits count. Used by variant B only.
  COHESION        two filler runs joined by one frame word become one filler when the three-word sequence is a strong collocation
                  ('Gulf of Mexico', 'Church of Scientology': count >= 3, and far above chance).
Each sentence is segmented into fillers and frames; every pair of consecutive fillers gives a record (filler, frame between them,
filler), punctuation included in the frame ('Stockholm , Sweden'). Records scored against the gold relations of the embedded
Re-DocRED benchmark exactly as tools/reader_audit.py scores the current reader (0.8% of gold pairs, 0.23 precision).
Usage: python3 tools/chunk_reader.py"""
import os
import re
import sys
import math
import collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from platypus.inference import probes  # noqa: E402
import platypus.config as C  # noqa: E402

cfg = C.Config(); cfg.inf_bench = 'embedded'; G = probes._bench(cfg)
docs = [d for k in sorted(G['docs']) for d in G['docs'][k]]
raw = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin', 'rb').read().decode('utf-8', 'replace')
TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*|[,;:()\"]")
SPL = re.compile(r'(?<=[.!?])\s+(?=[A-Z"(])')
corpus = [TOK.findall(s) for s in SPL.split(raw.replace('\n', ' '))] + [TOK.findall(s) for d in docs for s in SPL.split(d)]
corpus = [s for s in corpus if 3 <= len(s) <= 80]
uni = collections.Counter(w.lower() for s in corpus for w in s)
big = collections.Counter((a.lower(), b.lower()) for s in corpus for a, b in zip(s, s[1:]))
tri = collections.Counter((a.lower(), b.lower(), c.lower()) for s in corpus for a, b, c in zip(s, s[1:], s[2:]))
N = sum(uni.values())
nxt = collections.defaultdict(collections.Counter)                           # what follows each word (Harris successor variety)
for (a, b), c in big.items(): nxt[a][b] += c
H = {a: -sum(c / sum(cs.values()) * math.log2(c / sum(cs.values())) for c in cs.values()) for a, cs in nxt.items()}
slot = collections.defaultdict(list); cap = collections.Counter(); mid = collections.Counter()
for s in corpus:
    for i, w in enumerate(s):
        lw = w.lower()
        if i > 0:
            slot[lw].append(H.get(s[i - 1].lower(), 0.0)); mid[lw] += 1; cap[lw] += w[:1].isupper()
SLOT = {w: sum(v) / len(v) for w, v in slot.items()}
ENT = {w: (cap[w] / mid[w]) if mid[w] else 0.0 for w in mid}
rank = {w: i for i, (w, _) in enumerate(uni.most_common())}
print(f"counted {len(corpus)} sentences, {N} tokens, {len(uni)} types | slot score: 'in' {SLOT.get('in', 0):.1f} 'Sweden' {SLOT.get('sweden', 0):.1f} "
      f"'town' {SLOT.get('town', 0):.1f} 'husband' {SLOT.get('husband', 0):.1f} 'born' {SLOT.get('born', 0):.1f} | entity stat 'sweden' {ENT.get('sweden', 0):.2f} "
      f"'town' {ENT.get('town', 0):.2f}", flush=True)

t2e = G['_t2e']; gold = {}
for h, t, fl, ps in G['rel']: gold.setdefault((min(h, t), max(h, t)), ps)
def ents_of(span):
    es = set()
    for w in span: es |= t2e.get(w, set())
    return es


def segment(toks, K, var):
    """filler runs: tokens that are NOT frame material; frame material = top-K types, punctuation, (B) non-entity words"""
    def frame_w(w):
        lw = w.lower()
        if not re.match(r'[A-Za-z0-9]', w): return True
        if rank.get(lw, 10 ** 9) < K: return True
        if var == 'B' and not (ENT.get(lw, 0) >= 0.5 or w[:1].isdigit()): return True
        if var == 'C' and SLOT.get(lw, 0) < SLOT_THR: return True
        return False
    fr = [frame_w(w) for w in toks]
    for i in range(1, len(toks) - 1):                                          # COHESION: 'Gulf of Mexico' is one filler
        if fr[i] and not fr[i - 1] and not fr[i + 1] and re.match(r'[a-z]', toks[i]):
            t3 = (toks[i - 1].lower(), toks[i].lower(), toks[i + 1].lower()); c3 = tri.get(t3, 0)
            if c3 >= 3 and c3 * N / max(1, big.get(t3[:2], 1)) / max(1, uni.get(t3[2], 1)) > 50: fr[i] = False
    runs = []; i = 0
    while i < len(toks):
        if fr[i]: i += 1; continue
        j = i
        while j + 1 < len(toks) and not fr[j + 1]: j += 1
        runs.append((i, j + 1)); i = j + 1
    return runs


def records(toks, runs, maxgap, skip=1):
    out = []
    for k in range(1, skip + 1):                                               # skip 2: also fillers one apart ('X , a town in Y , Z')
        for (a0, a1), (b0, b1) in zip(runs, runs[k:]):
            gap = toks[a1:b0]
            if 1 <= len(gap) <= maxgap: out.append((tuple(toks[a0:a1]), ' '.join(gap), tuple(toks[b0:b1])))
    return out


def score(recs, tag):
    inb = hit = 0; pairs = set(); fr = collections.Counter()
    for a, r, b in recs:
        ea, eb = ents_of(a), ents_of(b)
        if not ea or not eb: continue
        inb += 1; ok = False
        for x in ea:
            for y in eb:
                if x != y and (min(x, y), max(x, y)) in gold: pairs.add((min(x, y), max(x, y))); ok = True
        hit += ok
        if ok: fr[r] += 1
    print(f"  {tag}: {len(recs)} records | both arguments benchmark entities {inb} | of those a GOLD pair {hit} ({hit / max(1, inb):.2f}) | "
          f"gold pairs covered {len(pairs)} of {len(gold)} ({len(pairs) / len(gold):.3f}) | commonest frames of gold records {fr.most_common(6)}", flush=True)
    return len(pairs), hit / max(1, inb)


bsents = [TOK.findall(s) for d in docs for s in SPL.split(d)]
print("CURRENT READER (tools/reader_audit.py): gold pairs covered 36 of 4726 (0.008), precision 0.23 | ceiling (both entities in one sentence) 555 (0.12)")
SLOT_THR = 0.0
for var in ('A', 'B'):
    for K in (100, 300, 1000):
        for gap in (8,):
            recs = [r for s in bsents for r in records(s, segment(s, K, var), gap)]
            score(recs, f"variant {var} ({'frequent words are frame material' if var == 'A' else 'frequent words AND words rarely capitalised mid-sentence are frame material'}), K {K}, frame up to {gap} tokens")
for thr in (3.0, 4.0, 5.0):
    SLOT_THR = thr
    recs = [r for s in bsents for r in records(s, segment(s, 100, 'C'), 8)]
    score(recs, f"variant C (Harris: words whose average open-slot entropy < {thr} bits are frame material), K 100")
SLOT_THR = 0.0
for sk, gp in ((2, 8), (2, 12), (3, 12)):
    recs = [r for s in bsents for r in records(s, segment(s, 300, 'B'), gp, sk)]
    score(recs, f"variant B, K 300, fillers up to {sk} apart, frame up to {gp} tokens")
s = [x for x in bsents if 'Stockholm' in x or 'husband' in x][:3]
for x in s: print('   e.g.', ' '.join(x)[:160], '->', records(x, segment(x, 300, 'B'), 8)[:4])
