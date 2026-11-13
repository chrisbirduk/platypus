"""PRONOUN REFERENCE BY SUBSTITUTION (12 Nov 2026; tools/pronoun_ref.py) - offline, no effect on the package.
Reference ('he', 'she', 'it', 'they') is the largest reading gap and was never measured: Re-DocRED annotates named mentions only. A proxy gold:
a sentence that OPENS with a pronoun and contains a mention of one argument of a gold relation whose OTHER argument is not mentioned in the
sentence - that other argument is what the pronoun stands for (kept only when exactly one such entity exists and it was mentioned earlier).
CANDIDATES: the entities mentioned in the earlier sentences (up to 8, the most recent first). The trunk's choice: replace the pronoun by each
candidate's first-mentioned name ('X's' for His/Her/Its/Their) and keep the candidate whose continuation (the rest of the sentence, up to 60
bytes) costs the fewest bits, read after the previous sentence. Against: the MOST RECENT candidate, the TITLE (the document's first entity),
the MOST MENTIONED so far, a TYPE rule (He/She -> the most recent person; It -> the most recent non-person), and the untrained network.
Usage: TG_CKPT=<model file> python3 tools/pronoun_ref.py [lm steps if no model] [documents]   (the stand-in of tools/trunk_guides.py)"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'trunk_guides.py')).read()
exec(compile(_src[:_src.index('# ---------------------------------------------------------------- T1 boundaries')], 'trunk_guides', 'exec'))
P(f"\n================ PRONOUN REFERENCE  ({time.time() - T0:.0f}s)")
@torch.no_grad()
def bits(m, seqs, spans, bs=256):
    out = np.zeros(len(seqs))
    for i in range(0, len(seqs), bs):
        ch = [s_[-L:] for s_ in seqs[i:i + bs]]; sh = [len(s_) - len(c_) for s_, c_ in zip(seqs[i:i + bs], ch)]
        X = torch.full((len(ch), L), PAD, dtype=torch.long)
        for j, c_ in enumerate(ch): X[j, :len(c_)] = torch.tensor(list(c_))
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(m(Xi), -1)
        nl = -lp.gather(-1, X.clamp(max=V - 1)[..., None])[..., 0] / math.log(2)
        for j, ((a, b), s0) in enumerate(zip(spans[i:i + bs], sh)): out[i + j] = float(nl[j, max(0, a - s0):max(0, b - s0)].sum())
    return out
PRON = {'He': 0, 'She': 0, 'It': 1, 'They': 2, 'His': 3, 'Her': 3, 'Its': 4, 'Their': 5}
dev = json.load(open(DEV))[:ND]; CASES = []
for di, (D, d) in enumerate(zip(DOCS, dev)):
    tk = D['toks']; b_ = D['b']; sb = {}
    for si, ti, a, b, w in tk: sb.setdefault(si, [a, b]); sb[si][1] = b
    msent = collections.defaultdict(set)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: msent[m['sent_id']].add(ei)
    first = {}
    for ei, vs in enumerate(d['vertexSet']):
        m0 = min(vs, key=lambda m: (m['sent_id'], m['pos'][0])); first[ei] = (m0['sent_id'], m0['pos'][0], m0['name'])
    for si, s in enumerate(d['sents']):
        if si == 0 or not s or s[0] not in PRON: continue
        here = msent[si]; tgt = set()
        for lab in d['labels']:
            h, t = lab['h'], lab['t']
            if h in here and t not in here: tgt.add(t)
            if t in here and h not in here: tgt.add(h)
        tgt = {e for e in tgt if first[e][0] < si}
        if len(tgt) != 1: continue
        g = next(iter(tgt))
        order = []                                                           # candidates: entities mentioned before, most recent first
        for sj in range(si - 1, -1, -1):
            for ei in sorted(msent[sj], key=lambda e: -max(m['pos'][0] for m in d['vertexSet'][e] if m['sent_id'] == sj)):
                if ei not in order: order.append(ei)
        cands = order[:8]
        if g not in cands or len(cands) < 2: continue
        cnt = {e: sum(1 for m in d['vertexSet'][e] if m['sent_id'] < si) for e in cands}
        prev = b_[max(sb[si - 1][0], sb[si - 1][1] - 50):sb[si - 1][1]]; rest = b_[tk[[k for k, t_ in enumerate(tk) if t_[0] == si][1]][2] - 1:sb[si][1]][:60]
        CASES.append(dict(di=di, pron=s[0], g=g, cands=cands, cnt=cnt, prev=prev, rest=rest,
                          names={e: first[e][2] for e in cands}, types={e: d['vertexSet'][e][0]['type'] for e in cands}, title=0))
P(f"pronoun-led sentences with a single proxy-gold antecedent among >= 2 earlier entities: {len(CASES)} | pronouns "
  + ", ".join(f"{k} {v}" for k, v in collections.Counter(c['pron'] for c in CASES).most_common()))
def choose(m):
    seqs, sp, own = [], [], []
    for ci, c in enumerate(CASES):
        for e in c['cands']:
            nm = c['names'][e].encode('utf-8', 'replace') + (b"'s" if PRON[c['pron']] >= 3 else b'')
            pre = c['prev'] + b' ' + nm; seqs.append(pre + c['rest']); sp.append((len(pre), len(pre) + len(c['rest']))); own.append((ci, e))
    v = bits(m, seqs, sp); best = {}
    for (ci, e), x in zip(own, v):
        if ci not in best or x < best[ci][0]: best[ci] = (x, e)
    return [best[ci][1] for ci in range(len(CASES))]
def acc(ch): return np.mean([e == c['g'] for e, c in zip(ch, CASES)])
P(f"  chance (1 / candidates) {np.mean([1 / len(c['cands']) for c in CASES]):.3f}")
P(f"  MOST RECENT {acc([c['cands'][0] for c in CASES]):.3f} | TITLE (document's first entity) {acc([0 if 0 in c['cands'] else c['cands'][0] for c in CASES]):.3f} | "
  f"MOST MENTIONED {acc([max(c['cands'], key=lambda e: (c['cnt'][e], -c['cands'].index(e))) for c in CASES]):.3f} | TYPE RULE "
  f"{acc([next((e for e in c['cands'] if (c['types'][e] == 'PER') == (PRON[c['pron']] in (0, 3) and c['pron'] in ('He', 'She', 'His', 'Her'))), c['cands'][0]) for c in CASES]):.3f}")
ct = choose(lm); c0 = choose(lm0)
P(f"  SUBSTITUTION, trained trunk {acc(ct):.3f} | untrained {acc(c0):.3f}")
for p_ in ('He', 'She', 'It', 'They', 'His', 'Her', 'Its', 'Their'):
    ix = [i for i, c in enumerate(CASES) if c['pron'] == p_]
    if len(ix) >= 10: P(f"    {p_:5s} ({len(ix)}): trained {np.mean([ct[i] == CASES[i]['g'] for i in ix]):.2f}, most recent {np.mean([CASES[i]['cands'][0] == CASES[i]['g'] for i in ix]):.2f}, title {np.mean([(0 if 0 in CASES[i]['cands'] else CASES[i]['cands'][0]) == CASES[i]['g'] for i in ix]):.2f}")
P(f"DONE ({time.time() - T0:.0f}s)")
