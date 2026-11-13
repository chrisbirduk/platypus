"""EVENT BOUNDARIES FROM THE TRUNK'S SURPRISE (13 Nov 2026; tools/event_bounds.py) - offline, no effect on the package.
Baldassano et al. 2017: high-level cortex divides a narrative into events; the hippocampus encodes at event boundaries. Can the trunk's surprise
tell the store WHERE a storable unit starts and ends? The stand-in trunk reads unseen Re-DocRED training documents (whole documents, 128-byte
windows); per word, its surprise (bits of the word's bytes). A boundary is placed after word i when the NEXT word's surprise is a local peak
above the document's q-th percentile. Segments are scored by whether they keep each gold-related name pair (both names in one sentence)
TOGETHER and split unrelated pairs APART - a unit that holds whole facts. Against the same number of boundaries placed: at every comma /
semicolon / bracket (punctuation), at random positions, and at fixed spacing; and against sentence boundaries alone.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/event_bounds.py [documents, default 400]"""
import os
import sys
import json
import collections
HERE = os.path.dirname(os.path.abspath(__file__))
ND_ = int(sys.argv[1]) if len(sys.argv) > 1 else 400
sys.argv = [sys.argv[0], '6000', '5']
_tg = open(os.path.join(HERE, 'trunk_guides.py')).read()
exec(compile(_tg[:_tg.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'))
P(f"\n================ EVENT BOUNDARIES  ({time.time() - T0:.0f}s)")
docs = json.load(open(SCR + '/bench_dl/train_revised.json'))
docs = [docs[i] for i in np.random.RandomState(0).permutation(len(docs))[:ND_]]
W, PAIRS = [], []                                                              # per document: words (sentence, index), byte spans
for d in docs:
    b = b''; wl = []
    for si, s in enumerate(d['sents']):
        for ti, w in enumerate(s):
            if b: b += b' '
            e = w.encode('utf-8', 'replace'); wl.append((si, ti, len(b), len(b) + len(e), w)); b += e
    S_, _ = run(lm, b); ws = np.array([float(S_[a:z].sum()) for _, _, a, z, _ in wl])
    pos = {(si, ti): k for k, (si, ti, _, _, _) in enumerate(wl)}
    gold = {(l['h'], l['t']) for l in d['labels']} | {(l['t'], l['h']) for l in d['labels']}
    pr = []
    for si in range(len(d['sents'])):
        ms = sorted((m['pos'][0], m['pos'][1], ei) for ei, vs in enumerate(d['vertexSet']) for m in vs if m['sent_id'] == si)
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y]
                if ea != eb and a1 <= b0 and 1 <= b0 - a1 <= 20: pr.append((pos[(si, a1 - 1)], pos[(si, b0)], (ea, eb) in gold))
    W.append(dict(words=wl, s=ws, sent=[pos[(si, 0)] for si in range(1, len(d['sents']))]))
    PAIRS.append(pr)
P(f"{len(docs)} unseen documents, {sum(len(w['words']) for w in W)} words, {sum(len(p) for p in PAIRS)} name pairs in one sentence "
  f"(related {np.mean([g for p in PAIRS for _, _, g in p]):.3f})  ({time.time() - T0:.0f}s)")
def score(cuts_fn, tag):
    tog_r = tog_u = n_r = n_u = 0; nb = 0; nw = 0; BIN = collections.defaultdict(lambda: [0, 0, 0, 0])   # by distance: related n, together; unrelated n, together
    for w, pr in zip(W, PAIRS):
        cuts = np.zeros(len(w['words']) + 1, bool)
        for c in cuts_fn(w): cuts[c] = True                                     # a cut at k = a boundary before word k
        for k in w['sent']: cuts[k] = True                                      # sentence boundaries always
        cum = np.cumsum(cuts); nb += int(cuts.sum()); nw += len(w['words'])
        for i, j, g in pr:
            together = cum[j] == cum[i]
            if g: n_r += 1; tog_r += together
            else: n_u += 1; tog_u += together
            bn = 0 if j - i <= 3 else (1 if j - i <= 7 else 2); BIN[bn][0 if g else 2] += 1; BIN[bn][1 if g else 3] += together
    tr, tu = tog_r / max(1, n_r), tog_u / max(1, n_u)
    P(f"   {tag:46s} boundaries per 100 words {100 * nb / nw:5.1f} | related pairs kept together {tr:.3f}, unrelated {tu:.3f} | ratio {tr / max(1e-9, tu):.2f} | "
      f"a segment holding a pair: related {n_r * tr / max(1e-9, n_r * tr + n_u * tu):.3f} (base {n_r / (n_r + n_u):.3f}) | SAME DISTANCE (1-3 / 4-7 / 8-20 words apart): ratio "
      + " / ".join(f"{(v[1] / max(1, v[0])) / max(1e-9, v[3] / max(1, v[2])):.2f}" for _, v in sorted(BIN.items())))
rq = np.random.RandomState(3)
score(lambda w: [], 'sentence boundaries only')
for q in (60, 75, 90):
    def surp(w, q=q):
        s = w['s']; th = np.percentile(s, q)
        return [k for k in range(1, len(s) - 1) if s[k] >= th and s[k] >= s[k - 1] and s[k] >= s[k + 1]]
    cnt = np.mean([len(surp(w)) / max(1, len(w['words'])) for w in W])
    score(surp, f"SURPRISE peaks above the {q}th percentile")
    score(lambda w, c=cnt: sorted(rq.choice(np.arange(1, max(2, len(w['words']))), max(0, min(len(w['words']) - 1, int(round(c * len(w['words']))))), replace=False)), f"   random, same number")
    score(lambda w, c=cnt: list(range(max(1, int(1 / max(c, 1e-3))), len(w['words']), max(1, int(1 / max(c, 1e-3))))), f"   fixed spacing, same number")
PUN = {',', ';', '(', ')', ':', '--', '"'}
score(lambda w: [k + 1 for k, (_, _, _, _, t) in enumerate(w['words']) if t in PUN], 'punctuation (, ; ( ) :)')
def surp_pun(w):
    s = w['s']; th = np.percentile(s, 75)
    return [k + 1 for k, (_, _, _, _, t) in enumerate(w['words']) if t in PUN and k + 1 < len(s) and s[k + 1] >= th]
score(surp_pun, 'punctuation followed by a surprising word')
P(f"DONE ({time.time() - T0:.0f}s)")
