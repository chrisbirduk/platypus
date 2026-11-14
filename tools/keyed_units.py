"""KEYED UNITS BEYOND NAMES: WHICH ONES PAY FOR THEMSELVES IN BITS? (14 Nov 2026; tools/keyed_units.py) - offline, no effect on the package.
The author's idea: a few generic KEYED UNITS (like 'name'), indexable, extracted from local context, each with its own memory, kept only if
they measurably improve language modelling. Here, counted extractors (no learning yet) on Re-DocRED text (Wikipedia; documents the stand-in
trunk never read), each unit with a keyed CONTINUATION memory: at every firing, the bytes that follow are stored under the key; when the key
fires again later (in the same or a later document), the stored continuations consistent with the bytes read so far predict the next byte,
mixed with the trunk: p = (1 - lam) p_trunk + lam p_mem, lam = n / (n + 2) for n consistent continuations; combined units mix their
predictions weighted by each unit's confidence. Units:
  NAME      key: a run of capitalised words                         (the existing unit)
  FACT      key: a name + the 1-4 words after it ('Paris is the capital of')  - the relation unit: predicts the object
  PLACE     key: a locative word + a name ('in Paris', 'of Bavaria')  - predicts the containing place (', Germany')
  DATE      key: the sentence's last name + 'date' (fires where a year or month starts) - predicts the date
  NUMBER    key: the sentence's last name + the word before a number - predicts the number
Scored prequentially on the last RU_EVAL documents (memories filled from every earlier document and earlier in the same document):
bits saved per 1,000 bytes, alone and on top of NAME, and bits saved per firing.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/keyed_units.py   env RU_EVAL (600), RU_CONT (24)"""
import os, sys, json, math, time, collections, re
import numpy as np
import torch
import torch.nn.functional as F
sys.argv = [sys.argv[0], '6000', '5']
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trunk_guides.py')).read()
g = {}; exec(compile(src[:src.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'), g)
lm, run, P = g['lm'], g['run'], g['P']; T0 = time.time(); torch.set_num_threads(1)
NE = int(os.environ.get('RU_EVAL', 600)); CL = int(os.environ.get('RU_CONT', 24))
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
docs = json.load(open(SCR + '/bench_dl/train_revised.json'))
NOSP_B = {',', '.', ';', ':', ')', "'s", "n't", '%', '!', '?', "'", "''"}; NOSP_A = {'(', '$', '``'}
LOCW = {'in', 'of', 'at', 'from', 'near', 'to'}; MONTHS = {'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'}
isyear = lambda w: bool(re.fullmatch(r'1[0-9]{3}|20[0-9]{2}', w)); isnum = lambda w: bool(re.fullmatch(r'[0-9][0-9,.]*', w))
def build(d):
    """bytes of the document and the firings of each unit: (unit, key, byte position where the predicted continuation starts)"""
    out = b''; toks = []; prev = None
    for si, sent in enumerate(d['sents']):
        for ti, w in enumerate(sent):
            wb = w.encode('utf-8', 'replace')
            if out and not (w in NOSP_B or prev in NOSP_A): out += b' '
            toks.append((si, ti, len(out), len(out) + len(wb), w)); out += wb; prev = w
    fires = []; i = 0; lastname = {}
    while i < len(toks):
        si, ti, a, b_, w = toks[i]
        if w[:1].isupper() and not (ti == 0 and i + 1 < len(toks) and not toks[i + 1][4][:1].isupper() and w.lower() in ('the', 'a', 'in', 'he', 'she', 'it', 'his', 'her', 'this', 'they')):
            j = i
            while j + 1 < len(toks) and toks[j + 1][0] == si and (toks[j + 1][4][:1].isupper() or toks[j + 1][4] in ('of', 'de', 'the') and j + 2 < len(toks) and toks[j + 2][4][:1].isupper()): j += 1
            name = ' '.join(t[4] for t in toks[i:j + 1]); end = toks[j][3]
            fires.append(('NAME', name, end))
            if i > 0 and toks[i - 1][0] == si and toks[i - 1][4] in LOCW: fires.append(('PLACE', toks[i - 1][4] + ' ' + name, end))
            words = []
            for k in range(j + 1, min(j + 5, len(toks))):
                if toks[k][0] != si or toks[k][4][:1].isupper() or isnum(toks[k][4]): break
                words.append(toks[k][4]); fires.append(('FACT', name + ' ' + ' '.join(words), toks[k][3]))
            lastname[si] = name; i = j + 1; continue
        if (isyear(w) or w in MONTHS) and si in lastname: fires.append(('DATE', lastname[si] + ' |date', a))
        elif isnum(w) and i > 0 and si in lastname: fires.append(('NUMBER', lastname[si] + ' |' + toks[i - 1][4], a))
        i += 1
    return out, fires
B = [build(d) for d in docs]; ev0 = len(B) - NE
P(f"documents {len(B)} (memory from all earlier ones; scored: the last {NE}) | firings per unit: " + ", ".join(f"{u} {n}" for u, n in collections.Counter(f[0] for b, fs in B for f in fs).most_common()))
UNITS = ['NAME', 'FACT', 'PLACE', 'DATE', 'NUMBER']
mem = {u: collections.defaultdict(list) for u in UNITS}
PRIO = {'FACT': 0, 'NUMBER': 1, 'DATE': 1, 'PLACE': 2, 'NAME': 3}   # combined units back off from the most specific key to NAME
def add(b, fs):
    for u, k, p in fs: mem[u][k].append(b[p:p + CL])
for b, fs in B[:ev0]: add(b, fs)
def pmem(units, b, fs):
    """per byte position: (lam, p_mem of the actual byte), units combined by pooling their consistent continuations"""
    best = {}
    for u, k, p in fs:
        if u not in units: continue
        conts = mem[u].get(k)
        if not conts: continue
        for off in range(min(CL, len(b) - p)):
            seen = b[p:p + off]; cons = [c for c in conts if len(c) > off and c[:off] == seen]
            if not cons: break
            n = len(cons); hit = sum(1 for c in cons if c[off] == b[p + off])
            pm = (hit + 1.0 / 256) / (n + 1.0); lam = n / (n + 2.0)
            best.setdefault(p + off, {})                                             # per unit, the most confident firing at this position
            q = best[p + off].get(u)
            if q is None or lam > q[0]: best[p + off][u] = (lam, pm)
    out = {}
    for pos, d in best.items():                                                      # units combined as a mixture: p_mem weighted by each
        lam = 1.0 - np.prod([1.0 - l for l, m_ in d.values()])                       #   unit's confidence, the overall weight 1 - prod(1 - lam_u)
        pm = sum(l * m_ for l, m_ in d.values()) / sum(l for l, m_ in d.values()); out[pos] = (lam, pm, '+'.join(sorted(d)))
    return out
res = {c: [0.0, 0, 0] for c in ['NAME', 'FACT', 'PLACE', 'DATE', 'NUMBER', 'NAME+FACT', 'NAME+PLACE', 'NAME+DATE', 'NAME+NUMBER', 'ALL']}
combos = {c: set(c.split('+')) if c != 'ALL' else set(UNITS) for c in res}
nbytes = 0; base_bits = 0.0
for di in range(ev0, len(B)):
    b, fs = B[di]
    S, _ = run(lm, b); pt = 2.0 ** (-S.astype(np.float64)); nbytes += len(b); base_bits += float(S.sum())
    for c, us in combos.items():
        bp = pmem(us, b, fs); saved = 0.0
        for pos, (lam, pm, u) in bp.items(): saved += S[pos] - (-math.log2((1 - lam) * pt[pos] + lam * pm))
        res[c][0] += saved; res[c][1] += len(bp)
    add(b, fs)
    if (di - ev0 + 1) % 150 == 0: P(f"   {di - ev0 + 1} documents scored  ({time.time() - T0:.0f}s)")
P(f"SCORED {NE} documents, {nbytes} bytes; trunk alone {base_bits / nbytes:.3f} bits per byte")
for c, (sv, npos, _) in res.items():
    P(f"   {c:12s} bits saved per 1,000 bytes {1000 * sv / nbytes:7.2f} ({100 * sv / base_bits:5.2f}% of the trunk's bits) | positions predicted {npos} | per predicted byte {sv / max(1, npos):.3f}")
for u in ['FACT', 'PLACE', 'DATE', 'NUMBER']:
    P(f"   {u:7s} on top of NAME: {1000 * (res['NAME+' + u][0] - res['NAME'][0]) / nbytes:+.2f} bits per 1,000 bytes")
P(f"DONE ({time.time() - T0:.0f}s)")
