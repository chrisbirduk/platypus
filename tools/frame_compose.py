"""COMPOSITION RULES OVER LABEL-FREE RELATION KEYS (17 Nov 2026; tools/frame_compose.py) - offline, Re-DocRED. tools/latent_compose.py E1
showed rules over GOLD relation ids compose real-text facts at 0.80 precision. How much survives when the relation key is what a reader
can see - the words between the two names in a sentence, with the names' types - instead of the gold id?

PREMISES (two readers): IDEAL = the gold-labelled facts readable in one sentence (both names within 12 tokens), keyed by their frame;
REAL = every pair of mentions within 12 tokens in a sentence (what the package's record store reads), keyed the same way.
KEYS: GOLD (the gold relation, IDEAL premises only - the reference), FINE (type A, the 1-3 words between or the last word, type B,
order), COARSE (type A, the last word between, type B, order), TYPES (type A, type B, order).
RULES mined on the training documents: chains A k1 B, B k2 C of premises; the rule (k1, k2) is kept if A and C are linked by some premise
in at least MINSUP chains with a Wilson lower bound >= MINLB. Held-out documents: every chain with a kept rule whose A-C pair is NOT a
premise there gives a composed pair; PRECISION = the pair is related by any gold label in the document (base = random entity pairs of the
same documents); RECALL = gold-related pairs that are not premises but reachable by a chain, composed.
MIXED readers (FC_MIX = list of premise precisions): the IDEAL premises plus non-gold records drawn at random to that precision -
what a reader / extractor of that precision would give.
Usage: python3 tools/frame_compose.py   env FC_MINSUP (10) FC_MINLB (0.5) FC_GAP (12) FC_MIX ('')"""
import os, json, math, collections, time
import numpy as np
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
T0 = time.time(); P = lambda s: print(s, flush=True)
TR = json.load(open(SCR + '/bench_dl/train_revised.json')); DV = json.load(open(SCR + '/bench_dl/dev_revised.json')) + json.load(open(SCR + '/bench_dl/test_revised.json'))
MINSUP = int(os.environ.get('FC_MINSUP', 10)); MINLB = float(os.environ.get('FC_MINLB', 0.5)); GAP = int(os.environ.get('FC_GAP', 12))
def wlb(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
def pair_frames(d):
    """(e1, e2) -> list of (types, mid words, order) for mention pairs of two entities within GAP tokens in one sentence"""
    ty = [vs[0]['type'] for vs in d['vertexSet']]; bys = collections.defaultdict(list); out = collections.defaultdict(list)
    for e, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], e))
    for s, ms in bys.items():
        ms.sort(); toks = d['sents'][s]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[i], ms[j]
                if ea == eb or a1 > b0 or b0 - a1 > GAP: continue
                mid = [w.lower() for w in toks[a1:b0]]
                out[(ea, eb)].append((ty[ea], ty[eb], mid, 'f')); out[(eb, ea)].append((ty[eb], ty[ea], mid, 'r'))
    return out
def key(kind, fr, r=None):
    ta, tb, mid, o = fr
    if kind == 'GOLD': return r
    if kind == 'TYPES': return (ta, tb, o)
    if kind == 'COARSE': return (ta, mid[-1] if mid else '', tb, o)
    return (ta, ' '.join(mid) if 1 <= len(mid) <= 3 else (mid[-1] if mid else ''), tb, o)
def premises(d, kind, reader):
    """{(h, t): set of keys}"""
    pf = pair_frames(d); out = collections.defaultdict(set)
    if reader == 'IDEAL' or kind == 'GOLD' or reader.startswith('MIX'):
        for l in d['labels']:
            fr = pf.get((l['h'], l['t']))
            if fr: out[(l['h'], l['t'])].add(key(kind, fr[0], l['r']))
        if reader.startswith('MIX'):                                          # non-gold records added to reach the premise precision
            pr = float(reader[3:]); gs = {(l['h'], l['t']) for l in d['labels']} | {(l['t'], l['h']) for l in d['labels']}
            ng = [(ab, frs) for ab, frs in sorted(pf.items()) if ab not in gs]; ng = [x for x in ng if x[0][0] < x[0][1]]
            want = int(round(len(out) * (1 - pr) / max(pr, 1e-6))); rr = np.random.RandomState(len(d['sents']) * 7 + len(ng))
            for i in rr.permutation(len(ng))[:want]:
                (a, b), frs = ng[int(i)]; out[(a, b)].add(key(kind, frs[0]))
    else:                                                                      # REAL readers: every co-occurrence, or label-free filters:
        for (a, b), frs in pf.items():                                         #   REPEAT = the pair co-occurs in 2+ sentences of the document;
            if reader == 'REPEAT' and len({tuple(f[2]) for f in frs}) < 2 and len(frs) < 2: continue   #   SHORT = 1-4 words between, the last one a
            if reader == 'SHORT' and not any(1 <= len(f[2]) <= 4 and f[2][-1] in PREP for f in frs): continue   #   preposition or 'is/was';
            if reader == 'BOTH' and not (len(frs) >= 2 and any(1 <= len(f[2]) <= 4 and f[2][-1] in PREP for f in frs)): continue   # BOTH = both
            for fr in frs[:2]: out[(a, b)].add(key(kind, fr))
    return out
PREP = {'in', 'of', 'at', 'from', 'by', 'for', 'on', 'to', 'with', 'is', 'was', 'as', 'near', 'into'}
def chains(prem):
    byh = collections.defaultdict(list)
    for (a, b), ks in prem.items():
        for k in ks: byh[a].append((k, b))
    for (a, b), ks in prem.items():
        for k1 in ks:
            for k2, c in byh.get(b, ()):
                if c != a and c != b: yield a, k1, b, k2, c
def run(kind, reader):
    cnt = collections.Counter(); hit = collections.Counter()
    for d in TR:
        prem = premises(d, kind, reader); seen = set()
        for a, k1, b, k2, c in chains(prem):
            if (a, k1, k2, c) in seen: continue
            seen.add((a, k1, k2, c)); cnt[(k1, k2)] += 1; hit[(k1, k2)] += bool(prem.get((a, c)) or prem.get((c, a)))
    rules = {k for k, n in cnt.items() if hit[k] >= MINSUP and wlb(hit[k], n) >= MINLB}
    comp = rel = 0; reach = got = 0; base = []
    for d in DV:
        prem = premises(d, kind, reader); gold = {(l['h'], l['t']) for l in d['labels']} | {(l['t'], l['h']) for l in d['labels']}
        isp = lambda a, c: (a, c) in prem or (c, a) in prem
        out = set(); allc = set()
        for a, k1, b, k2, c in chains(prem):
            if isp(a, c): continue
            allc.add((a, c))
            if (k1, k2) in rules: out.add((a, c))
        comp += len(out); rel += sum((a, c) in gold for a, c in out)
        rr = [p for p in allc if p in gold]; reach += len(rr); got += sum(p in out for p in rr)
        n = len(d['vertexSet'])
        if n > 2: base.append(len(gold) / (n * (n - 1)))
    P(f"   {kind:6s} {reader:5s} rules {len(rules):5d} | composed {comp:6d}, PRECISION (gold-related) {rel / max(1, comp):.3f} (base {np.mean(base):.3f}) | "
      f"recall of chain-reachable gold pairs {got / max(1, reach):.3f} ({reach})  ({time.time() - T0:.0f}s)")
P(f"RULES OVER RELATION KEYS (train {len(TR)} docs, held out {len(DV)}; support >= {MINSUP}, Wilson LB >= {MINLB}; premises within {GAP} tokens in one sentence)")
MIX = [x for x in os.environ.get('FC_MIX', '').split(',') if x]
RDR = [x for x in os.environ.get('FC_READERS', '').split(',') if x]
RUNS = [(k, r) for r in RDR for k in ('COARSE', 'TYPES')] if RDR else [(k, 'MIX' + m) for m in MIX for k in ('COARSE', 'TYPES')] if MIX else [('GOLD', 'IDEAL'), ('FINE', 'IDEAL'), ('COARSE', 'IDEAL'), ('TYPES', 'IDEAL'), ('FINE', 'REAL'), ('COARSE', 'REAL'), ('TYPES', 'REAL')]
for kind, reader in RUNS:
    run(kind, reader)
P(f"DONE ({time.time() - T0:.0f}s)")
