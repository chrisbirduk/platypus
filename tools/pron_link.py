"""SELF-SUPERVISED PRONOUN LINKING (18 Nov 2026; tools/pron_link.py) - offline, distant set, LEARNED and label-free (distant relation labels
for evaluation only). Route 2 to real-text inference: most real facts are stated with a pronoun ('He was born in ...'), so the name-pair
reader misses them. The text supervises itself: a LATER mention of an entity already named in the document is replaced by a pronoun ('he'
for a person-typed mention, 'it' otherwise, capitalised at a sentence start), and a small linker on the stand-in trunk's own states learns to
point from the pronoun back to the entity's earlier mentions (the other earlier mentions of the document are the negatives).
1. Held-out documents, the same construction: linking accuracy against nearest-earlier-mention and first-mention (title) baselines.
2. REAL pronouns in other documents: each is linked to the predicted entity, and the linked entity is paired with the names in its sentence
   within 12 tokens (as the reader pairs names). Precision of these NEW records by the distant labels (pairs never recorded by names in the
   document) against the reader's own name-pair records, and how many right pairs they add.
Usage: TG_D=192 TG_CKPT=<model> python3 tools/pron_link.py <distant.json.gz>   env PL_TRAIN (1500) PL_TEST (300) PL_REAL (1000) PL_EPOCHS (8)"""
import os, sys, gzip, json, math, time, collections
import numpy as np
import torch
import torch.nn.functional as F
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(here))
T0 = time.time(); P = lambda s: print(s, flush=True)
DOCS_RAW = json.load(gzip.open(sys.argv[1]))
NTR, NTE, NRE = int(os.environ.get('PL_TRAIN', 1500)), int(os.environ.get('PL_TEST', 300)), int(os.environ.get('PL_REAL', 1000))
sys.argv = [sys.argv[0], '6000', '1']
src = open(os.path.join(here, 'trunk_guides.py')).read(); g = {}
exec(compile(src[:src.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'), g)
lm, run = g['lm'], g['run']; torch.set_num_threads(2)
NOSP_B = {',', '.', ';', ':', ')', "'s", "n't", '%', '!', '?', "'", "''"}; NOSP_A = {'(', '$', '``'}
PRON = {'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their'}
def build(d, swap=None):
    """document bytes, tokens (sid, tid, a, b, word), mentions sorted; swap = (sid, t0, t1, word): replace those tokens by one word"""
    toks = []; out = b''; prev = None
    for si, sent in enumerate(d['sents']):
        ti = 0
        while ti < len(sent):
            if swap and si == swap[0] and ti == swap[1]: w = swap[3]; span = (ti, swap[2]); ti = swap[2]
            else: w = sent[ti]; span = (ti, ti + 1); ti += 1
            wb = w.encode('utf-8', 'replace')
            if out and not (w in NOSP_B or prev in NOSP_A): out += b' '
            toks.append((si, span, len(out), len(out) + len(wb), w)); out += wb; prev = w
    ix = {}
    for k, (si, (t0, t1), a, b, w) in enumerate(toks):
        for t in range(t0, t1): ix[(si, t)] = k
    ments = []
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs:
            if swap and m['sent_id'] == swap[0] and m['pos'][0] == swap[1]: continue
            k0, k1 = ix.get((m['sent_id'], m['pos'][0])), ix.get((m['sent_id'], m['pos'][1] - 1))
            if k0 is None or k1 is None: continue
            ments.append(dict(e=ei, a=toks[k0][2], b=toks[k1][3], sid=m['sent_id'], t0=m['pos'][0], t1=m['pos'][1], typ=m['type'], name=d['vertexSet'][ei][0]['name']))
    return out, toks, sorted(ments, key=lambda m: m['a'])
def examples(docs, rs):
    ex = []
    for d in docs:
        cands = []
        for ei, vs in enumerate(d['vertexSet']):
            ms = sorted(vs, key=lambda m: (m['sent_id'], m['pos'][0]))
            for m in ms[1:]:
                if m['sent_id'] > ms[0]['sent_id']: cands.append((ei, m))
        if not cands: continue
        ei, m = cands[rs.randint(len(cands))]; w = 'he' if m['type'] == 'PER' else 'it'
        if m['pos'][0] == 0: w = w.capitalize()
        b, toks, ments = build(d, (m['sent_id'], m['pos'][0], m['pos'][1], w))
        pk = next(k for k, t in enumerate(toks) if t[0] == m['sent_id'] and t[1][0] == m['pos'][0])
        pa, pb = toks[pk][2], toks[pk][3]; prev = [x for x in ments if x['b'] <= pa]
        if len(prev) < 2 or not any(x['e'] == ei for x in prev): continue
        ex.append(dict(b=b, pb=pb, pa=pa, prev=prev, e=ei))
    return ex
def feats(E):
    _, H = run(lm, E['b']); hp = H[E['pb'] - 1]; hm = np.stack([H[x['b'] - 1] for x in E['prev']])
    dist = np.array([[math.log1p((E['pa'] - x['b']) / 50.0), float(x is E['prev'][-1]), float(x['a'] == E['prev'][0]['a'])] for x in E['prev']], np.float32)
    return hp, hm, dist
rs = np.random.RandomState(0); idx = rs.permutation(len(DOCS_RAW))
TR = examples([DOCS_RAW[i] for i in idx[:NTR]], rs); TE = examples([DOCS_RAW[i] for i in idx[NTR:NTR + NTE]], rs)
FT = [feats(E) for E in TR]; FE = [feats(E) for E in TE]
P(f"self-supervised examples: {len(TR)} training, {len(TE)} held out (a later mention replaced by a pronoun)  ({time.time() - T0:.0f}s)")
D_ = FT[0][0].shape[0]; Wm = torch.nn.Parameter(torch.zeros(D_, D_)); Wd = torch.nn.Parameter(torch.zeros(3)); op = torch.optim.Adam([Wm, Wd], lr=1e-3)
def score(hp, hm, dist):
    hp = torch.tensor(hp); hm = torch.tensor(hm); hp = hp / (hp.norm() + 1e-6); hm = hm / (hm.norm(dim=-1, keepdim=True) + 1e-6)
    return 10.0 * (hm @ (Wm @ hp) + hm @ hp) + torch.tensor(dist) @ Wd
for ep in range(int(os.environ.get('PL_EPOCHS', 8))):
    tot = 0.0
    for k in rs.permutation(len(TR)):
        hp, hm, dist = FT[k]; s = score(hp, hm, dist); pos = torch.tensor([x['e'] == TR[k]['e'] for x in TR[k]['prev']])
        l = torch.logsumexp(s, 0) - torch.logsumexp(s[pos], 0); op.zero_grad(); (l + 1e-4 * (Wm ** 2).sum()).backward(); op.step(); tot += float(l)
    with torch.no_grad(): acc = np.mean([TE[k]['prev'][int(score(*FE[k]).argmax())]['e'] == TE[k]['e'] for k in range(len(TE))])
    P(f"   epoch {ep + 1}: loss {tot / len(TR):.3f} | held-out linking accuracy {acc:.3f}  ({time.time() - T0:.0f}s)")
near = np.mean([E['prev'][-1]['e'] == E['e'] for E in TE]); first = np.mean([E['prev'][0]['e'] == E['e'] for E in TE])
nearpt = np.mean([next((x for x in reversed(E['prev']) if (x['typ'] == 'PER') == (E['b'][E['pa']:E['pb']].lower() == b'he')), E['prev'][-1])['e'] == E['e'] for E in TE])
P(f"HELD OUT: learned linker {acc:.3f} | nearest earlier mention {near:.3f} | first mention (title) {first:.3f} | nearest of the matching type (uses NER types) {nearpt:.3f}")
# ---- 2. REAL pronouns
LAB = set()
for d in DOCS_RAW:
    for l in d['labels']: LAB.add(frozenset((d['vertexSet'][l['h']][0]['name'], d['vertexSet'][l['t']][0]['name'])))
RE = [DOCS_RAW[i] for i in idx[NTR + NTE:NTR + NTE + NRE]]; newp, namep, nrec = [], [], 0
for d in RE:
    b, toks, ments = build(d); _, H = run(lm, b); rec = set()
    for x in ments:
        for y in ments:
            if x['sid'] == y['sid'] and x['e'] != y['e'] and abs(x['t0'] - y['t0']) <= 12: rec.add(frozenset((x['name'], y['name'])))
    namep += [p_ in LAB for p_ in rec]
    for k, (si, (t0, t1), pa, pb, w) in enumerate(toks):
        if w.lower() not in PRON: continue
        prev = [x for x in ments if x['b'] <= pa]
        if len(prev) < 1: continue
        hm = np.stack([H[x['b'] - 1] for x in prev]); dist = np.array([[math.log1p((pa - x['b']) / 50.0), float(x is prev[-1]), float(x['a'] == prev[0]['a'])] for x in prev], np.float32)
        with torch.no_grad(): ent = prev[int(score(H[pb - 1], hm, dist).argmax())]
        for y in ments:
            if y['sid'] == si and y['e'] != ent['e'] and abs(y['t0'] - t0) <= 12:
                p_ = frozenset((ent['name'], y['name']))
                if len(p_) == 2 and p_ not in rec: newp.append(p_ in LAB); rec.add(p_); nrec += 1
P(f"REAL PRONOUNS in {len(RE)} documents: the reader's name-pair records precision {np.mean(namep):.3f} (n {len(namep)}) | NEW records through linked pronouns "
  f"precision {np.mean(newp) if newp else float('nan'):.3f} (n {len(newp)}) -> {int(np.sum(newp))} right pairs added to {int(np.sum(namep))} (+{np.sum(newp) / max(1, np.sum(namep)):.0%})")
P(f"DONE ({time.time() - T0:.0f}s)")
