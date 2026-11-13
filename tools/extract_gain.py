"""CAN THE TRUNK TELL THE READER WHAT TO PULL OUT OF REAL TEXT? (13 Nov 2026; tools/extract_gain.py) - offline, no effect on the package.
The reader keeps ~3% of the facts in Re-DocRED (tools/count_miner_reader.py). Here extraction is driven by the trunk instead of rules: from
every sentence, take EVERY pair of names in it (high recall, low precision) as a candidate record - the stretch of text from the first name
to the second ('Dresden , the capital of Saxony'). Its value to the trunk: does holding that record make OTHER text cheaper to predict -
another document's sentence that mentions the same two names? EXTRACTION GAIN = bits on the second name there, read after the record with
its first name replaced by 'Thing', minus read after the real record (so the gain is what the LINK between the names is worth, not the
names' frequency). Label-free: no gold is used to compute it. Scored against Re-DocRED gold (the pair holds a gold relation in its own
document): AUC gold vs not; precision of the records the gain would keep (top 10% / 25%); against the PAIR-ONLY record ('Dresden Saxony',
names without the words between) and the plain distance between the names (closer pairs are more often related).
If the gain separates related pairs, the trunk can supervise extraction: keep what compresses other text, and it gets better as the trunk does.
Rows added 13 Nov 14:00: the LIST-STYLE record ('A , B') measures what plain co-occurrence is worth; the record's gain beyond it.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/extract_gain.py [max candidates, default 6000]   (the stand-in trunk of tools/trunk_guides.py)"""
import os
import sys
import json
import collections
HERE = os.path.dirname(os.path.abspath(__file__))
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
sys.argv = [sys.argv[0], '6000', '5']
_tg = open(os.path.join(HERE, 'trunk_guides.py')).read()
exec(compile(_tg[:_tg.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'))
P(f"\n================ EXTRACTION GAIN  ({time.time() - T0:.0f}s)")
TH = b'Thing'
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    r = np.concatenate([pos, neg]).argsort(kind='stable').argsort().astype(float) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
DIR = SCR + '/bench_dl'
docs = json.load(open(DIR + '/train_revised.json')) + json.load(open(DIR + '/dev_revised.json'))


@torch.no_grad()
def bits(m, seqs, spans, bs=128):
    out = np.zeros(len(seqs))
    for i in range(0, len(seqs), bs):
        ch = [s_[-L:] for s_ in seqs[i:i + bs]]; sh = [len(s_) - len(c_) for s_, c_ in zip(seqs[i:i + bs], ch)]
        X = torch.full((len(ch), L), PAD, dtype=torch.long)
        for j, c_ in enumerate(ch): X[j, :len(c_)] = torch.tensor(list(c_))
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(lm(Xi), -1)
        nl = -lp.gather(-1, X.clamp(max=V - 1)[..., None])[..., 0] / math.log(2)
        for j, ((a, b), s0) in enumerate(zip(spans[i:i + bs], sh)): out[i + j] = float(nl[j, max(0, a - s0):max(0, b - s0)].sum())
    return out


# ---- every name pair within a sentence, per document; and where each name pair co-occurs in a sentence
cands = []; occ = collections.defaultdict(list)
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    gold = {(l['h'], l['t']) for l in d['labels']} | {(l['t'], l['h']) for l in d['labels']}
    bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms); toks = d['sents'][si]
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb]: continue
                key = tuple(sorted((cn[ea], cn[eb])))
                occ[key].append((di, si, a0, a1, b0, b1, ea, eb))
                if b0 - a1 <= 12 and len(' '.join(toks[a0:b1])) <= 60: cands.append(dict(di=di, si=si, a=(a0, a1), b=(b0, b1), ea=ea, eb=eb, key=key, gold=(ea, eb) in gold, dist=b0 - a1))
# keep candidates whose name pair co-occurs in a sentence of ANOTHER document (the text the record should help predict)
rq = np.random.RandomState(0); usable = []
for c in cands:
    other = [o for o in occ[c['key']] if o[0] != c['di']]
    if other: c['tgt'] = other[rq.randint(len(other))]; usable.append(c)
rq.shuffle(usable); G = [c for c in usable if c['gold']]; NG = [c for c in usable if not c['gold']]
usable = usable[:NMAX]                                                       # the natural mix of related and unrelated pairs
P(f"candidate records (name pairs within 12 words and 60 bytes, one sentence): {len(cands)} in {len(docs)} documents, gold {np.mean([c['gold'] for c in cands]):.3f} | "
  f"with the pair co-mentioned in another document: {len(G) + len(NG)} (gold {len(G)}) | scored {len(usable)} (gold {sum(c['gold'] for c in usable)})")


def txt(toks): return ' '.join(toks).encode('utf-8', 'replace')


W, O, PW, PO, LW, LO = [], [], [], [], [], []
for c in usable:
    d = docs[c['di']]; toks = d['sents'][c['si']]; (a0, a1), (b0, b1) = c['a'], c['b']
    rec = txt(toks[a0:b1]) + b' .'; rec0 = TH + b' ' + txt(toks[a1:b1]) + b' .'
    pr = txt(toks[a0:a1]) + b' ' + txt(toks[b0:b1]) + b' .'; pr0 = TH + b' ' + txt(toks[b0:b1]) + b' .'
    di2, si2, x0, x1, y0, y1, ex, ey = c['tgt']; t2 = docs[di2]['sents'][si2]
    pre = txt(t2[max(0, y0 - 6):y0])[-36:] + b' '; tgt = txt(t2[y0:y1])[:24]   # record + prefix + target fit the trunk's 128-byte window
    lw = txt(toks[a0:a1]) + b' , ' + txt(toks[b0:b1]) + b' .'; lo = TH + b' , ' + txt(toks[b0:b1]) + b' .'   # LIST-STYLE record: co-occurrence alone
    for lst, r in ((W, rec), (O, rec0), (PW, pr), (PO, pr0), (LW, lw), (LO, lo)):
        s_ = r + b' ' + pre; lst.append((s_ + tgt, (len(s_), len(s_) + len(tgt))))
t1 = time.time()
bw = bits(lm, [s for s, _ in W], [p for _, p in W]); bo = bits(lm, [s for s, _ in O], [p for _, p in O])
pw = bits(lm, [s for s, _ in PW], [p for _, p in PW]); po = bits(lm, [s for s, _ in PO], [p for _, p in PO])
lw_ = bits(lm, [s for s, _ in LW], [p for _, p in LW]); lo_ = bits(lm, [s for s, _ in LO], [p for _, p in LO])
P(f"bits computed ({time.time() - t1:.0f}s)")
y = np.array([c['gold'] for c in usable]); gain = bo - bw; pgain = po - pw; dist = -np.array([c['dist'] for c in usable], float)


def rep(tag, sc):
    o = np.argsort(-sc); k1, k2 = max(1, len(sc) // 10), max(1, len(sc) // 4)
    P(f"   {tag:44s} AUC gold vs not {auc(sc[y], sc[~y]):.3f} | kept top 10%: gold {y[o[:k1]].mean():.3f} | top 25%: gold {y[o[:k2]].mean():.3f} "
      f"(base rate {y.mean():.3f}) | mean gain gold {sc[y].mean():+.2f} vs not {sc[~y].mean():+.2f}")


rep('EXTRACTION GAIN (record with the words between)', gain)
rep('pair-only gain (names, no words between)', pgain)
rep('words between minus pair-only', gain - pgain)
rep('closeness of the names (no trunk)', dist)
lg = lo_ - lw_
rep('list-style gain ("A , B": co-occurrence alone)', lg)
rep('WORDS BETWEEN beyond the LIST record', gain - lg)
rk = lambda x: np.argsort(np.argsort(x)).astype(float)
rep('gain + closeness (rank average)', (rk(gain) + rk(dist)) / 2)
P(f"DONE ({time.time() - T0:.0f}s)")
