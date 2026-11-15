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
FEEDBACK GATE (14 Nov, second version): the fixed mixture above lets a weak unit drag down a strong one. The gate LEARNS how far to trust
each unit from the bits it actually saves: a softmax over the trunk (logit 0) and each firing unit, the unit's logit read from a table keyed
by (unit, consistent continuations n, bytes into the continuation, the trunk's own entropy at that byte - known before the byte is read),
started at log(n / 2) (the fixed rule) and moved by one online gradient step on -log p after each byte is scored (prequential: every
byte is scored before the gate learns from it). Reported for all scored documents and for the last two-thirds (after warm-up).
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/keyed_units.py   env RU_EVAL (600), RU_CONT (24), RU_LR (0.05), RU_EVALSET (test or dev:
  memory from all 3,053 training documents, scored on that set instead)"""
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
EVS = os.environ.get('RU_EVALSET', '')                                              # e.g. 'test': memory from all of train, scored on that set
if EVS: ev_docs = json.load(open(SCR + f'/bench_dl/{EVS}_revised.json')); docs = docs + ev_docs; NE = len(ev_docs)
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
def pmem(units, b, fs, raw=False):
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
            if q is None or lam > q[0]: best[p + off][u] = (lam, pm, n, off)
    if raw: return best
    out = {}
    for pos, d in best.items():                                                      # units combined as a mixture: p_mem weighted by each
        lam = 1.0 - np.prod([1.0 - v[0] for v in d.values()])                       #   unit's confidence, the overall weight 1 - prod(1 - lam_u)
        pm = sum(v[0] * v[1] for v in d.values()) / sum(v[0] for v in d.values()); out[pos] = (lam, pm, '+'.join(sorted(d)))
    return out
res = {c: [0.0, 0, 0] for c in ['NAME', 'FACT', 'PLACE', 'DATE', 'NUMBER', 'NAME+FACT', 'NAME+PLACE', 'NAME+DATE', 'NAME+NUMBER', 'NAME+DATE+NUMBER', 'ALL']}
combos = {c: set(c.split('+')) if c != 'ALL' else set(UNITS) for c in res}
LR = float(os.environ.get('RU_LR', 0.05)); W0 = ev0 + NE // 3
nbk = lambda n: 0 if n <= 1 else 1 if n == 2 else 2 if n <= 4 else 3 if n <= 8 else 4
obk = lambda o: 0 if o == 0 else 1 if o <= 3 else 2 if o <= 11 else 3
ebk = lambda e: 0 if e < 0.5 else 1 if e < 1.5 else 2 if e < 3 else 3
gate = {c: {} for c in res}; gres = {c: [0.0, 0.0] for c in res}; nb_late = 0; bb_late = 0.0
nbytes = 0; base_bits = 0.0
for di in range(ev0, len(B)):
    b, fs = B[di]
    S, Hs = run(lm, b); pt = 2.0 ** (-S.astype(np.float64)); nbytes += len(b); base_bits += float(S.sum())
    with torch.no_grad(): lpo = F.log_softmax(lm.o(torch.tensor(Hs)), -1); ent = (-(lpo.exp() * lpo).sum(-1) / math.log(2)).numpy()
    late = di >= W0
    if late: nb_late += len(b); bb_late += float(S.sum())
    for c, us in combos.items():
        bp = pmem(us, b, fs); saved = 0.0
        for pos, (lam, pm, u) in bp.items(): saved += S[pos] - (-math.log2((1 - lam) * pt[pos] + lam * pm))
        res[c][0] += saved; res[c][1] += len(bp)
        G = gate[c]; gs = 0.0                                                         # the feedback gate, prequential
        for pos, d in sorted(pmem(us, b, fs, raw=True).items()):
            ks = [(u, nbk(v[2]), obk(v[3]), ebk(ent[pos])) for u, v in d.items()]
            lg = np.array([0.0] + [G.setdefault(k, math.log(max(d[k[0]][2], 1) / 2.0)) for k in ks])
            w = np.exp(lg - lg.max()); w /= w.sum()
            ps = np.array([pt[pos]] + [d[k[0]][1] for k in ks]); p = float((w * ps).sum())
            gs += S[pos] + math.log2(max(p, 1e-12))
            for j, k in enumerate(ks): G[k] += LR * w[j + 1] * (ps[j + 1] - p) / max(p, 1e-12)
        gres[c][0] += gs
        if late: gres[c][1] += gs
    add(b, fs)
    if (di - ev0 + 1) % 150 == 0: P(f"   {di - ev0 + 1} documents scored  ({time.time() - T0:.0f}s)")
P(f"SCORED {NE} documents, {nbytes} bytes; trunk alone {base_bits / nbytes:.3f} bits per byte")
for c, (sv, npos, _) in res.items():
    P(f"   {c:12s} bits saved per 1,000 bytes {1000 * sv / nbytes:7.2f} ({100 * sv / base_bits:5.2f}% of the trunk's bits) | positions predicted {npos} | per predicted byte {sv / max(1, npos):.3f}")
for u in ['FACT', 'PLACE', 'DATE', 'NUMBER']:
    P(f"   {u:7s} on top of NAME: {1000 * (res['NAME+' + u][0] - res['NAME'][0]) / nbytes:+.2f} bits per 1,000 bytes")
P(f"FEEDBACK GATE (learned trust per unit, lr {LR}): bits saved per 1,000 bytes, all {NE} documents | last {len(B) - W0} (after warm-up)")
for c in res: P(f"   {c:12s} gate {1000 * gres[c][0] / nbytes:7.2f} | late {1000 * gres[c][1] / nb_late:7.2f} ({100 * gres[c][1] / bb_late:5.2f}%)   fixed mixture {1000 * res[c][0] / nbytes:7.2f}")
for u in ['FACT', 'PLACE', 'DATE', 'NUMBER']:
    P(f"   {u:7s} on top of NAME with the gate: all {1000 * (gres['NAME+' + u][0] - gres['NAME'][0]) / nbytes:+.2f} | late {1000 * (gres['NAME+' + u][1] - gres['NAME'][1]) / nb_late:+.2f} bits per 1,000 bytes")
P(f"   DATE+NUMBER on top of NAME with the gate: all {1000 * (gres['NAME+DATE+NUMBER'][0] - gres['NAME'][0]) / nbytes:+.2f} | late {1000 * (gres['NAME+DATE+NUMBER'][1] - gres['NAME'][1]) / nb_late:+.2f}")
P(f"   ALL on top of NAME with the gate: all {1000 * (gres['ALL'][0] - gres['NAME'][0]) / nbytes:+.2f} | late {1000 * (gres['ALL'][1] - gres['NAME'][1]) / nb_late:+.2f}")
lt = collections.defaultdict(list)
for k, v in gate['ALL'].items(): lt[k[0]].append(v)
P("   learned logits (ALL), mean per unit: " + ", ".join(f"{u} {np.mean(v):+.2f} over {len(v)} cells" for u, v in lt.items()))
P(f"DONE ({time.time() - T0:.0f}s)")
