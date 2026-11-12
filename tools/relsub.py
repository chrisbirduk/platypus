"""ONE RELATION, MANY WORDINGS - BY SUBSTITUTION (12 Nov 2026; tools/relsub.py) - offline, no effect on the package.
Third row of the compression plan: two frames are the same relation if swapping one for the other costs the trunk few bits. For gold
relations of the Re-DocRED dev set with both names in one sentence, a RECORD is (left context, first name, FRAME = the words between,
second name). SUBSTITUTION SCORE of frame j for record i: bits the trunk spends on i's second name after 'left + first name + frame j',
minus after its own frame (symmetric: averaged with j's second name under frame i); low cost = interchangeable. For 600 query records,
each against 30 random other records: is the best-scoring candidate of the same gold relation type (top-1; chance = the share of candidates
of that type), and AUC same type vs not - for FRAME WORDS (tf-idf cosine; earlier T2b: nearest frame 0.54), the trained stand-in trunk
and the same network untrained.
Usage: TG_CKPT=<model file> python3 tools/relsub.py [lm steps if no model] [documents]   (the stand-in of tools/trunk_guides.py)"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'trunk_guides.py')).read()
exec(compile(_src[:_src.index('# ---------------------------------------------------------------- T1 boundaries')], 'trunk_guides', 'exec'))
P(f"\n================ RELATION IDENTITY BY SUBSTITUTION  ({time.time() - T0:.0f}s)")
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
dev = json.load(open(DEV))[:ND]
REC = []; seen = set()
for di, (D, d) in enumerate(zip(DOCS, dev)):
    tk = D['toks']; ix = {(si, ti): k for k, (si, ti, a, b, w) in enumerate(tk)}; b_ = D['b']
    sb = {}
    for si, ti, a, b, w in tk: sb.setdefault(si, [a, b]); sb[si][1] = b
    for lab in d['labels']:
        for mh in d['vertexSet'][lab['h']]:
            for mt in d['vertexSet'][lab['t']]:
                if mh['sent_id'] != mt['sent_id']: continue
                x, y = sorted([mh, mt], key=lambda m: m['pos'][0])
                if x['pos'][1] > y['pos'][0]: continue
                xa, xb = tk[ix[(x['sent_id'], x['pos'][0])]][2], tk[ix[(x['sent_id'], x['pos'][1] - 1)]][3]
                ya, yb = tk[ix[(y['sent_id'], y['pos'][0])]][2], tk[ix[(y['sent_id'], y['pos'][1] - 1)]][3]
                fr = b_[xb:ya]
                if not (2 <= len(fr) <= 40) or (di, xa, ya) in seen: continue
                seen.add((di, xa, ya)); left = b_[max(sb[x['sent_id']][0], xa - 30):xa]
                REC.append(dict(di=di, left=left, x=b_[xa:xb][:30], fr=fr, y=b_[ya:yb][:30], r=lab['r']))
rt = collections.defaultdict(set)
for r in REC: rt[(r['di'], r['x'], r['y'])].add(r['r'])
for r in REC: r['rs'] = rt[(r['di'], r['x'], r['y'])]
P(f"records {len(REC)} (gold relation types {len({r['r'] for r in REC})})")
rq = np.random.RandomState(0); Q = rq.choice(len(REC), min(600, len(REC)), replace=False); K = 30
pairs = []
for qi in Q:
    cs = [int(c) for c in rq.choice(len(REC), K + 5, replace=False) if c != qi and REC[c]['di'] != REC[qi]['di']][:K]
    pairs += [(int(qi), c) for c in cs]
def subscore(m):
    seqs, sp = [], []
    def add(i, frame):
        r = REC[i]; pre = r['left'] + r['x'] + frame; seqs.append(pre + r['y']); sp.append((len(pre), len(pre) + len(r['y'])))
    for i, j in pairs: add(i, REC[j]['fr']); add(j, REC[i]['fr'])
    base = {}
    for i in {x for p in pairs for x in p}: base[i] = None
    bl = list(base); bseq = [REC[i]['left'] + REC[i]['x'] + REC[i]['fr'] + REC[i]['y'] for i in bl]
    bsp = [(len(REC[i]['left'] + REC[i]['x'] + REC[i]['fr']), len(bseq[n])) for n, i in enumerate(bl)]
    bb = dict(zip(bl, bits(m, bseq, bsp))); bs_ = bits(m, seqs, sp)
    return np.array([-((bs_[2 * n] - bb[i]) + (bs_[2 * n + 1] - bb[j])) / 2 for n, (i, j) in enumerate(pairs)])
from sklearn.feature_extraction.text import TfidfVectorizer
TF = TfidfVectorizer(token_pattern=r"[A-Za-z]+|[^\sA-Za-z]").fit([r['fr'].decode('utf-8', 'replace') for r in REC])
Wv = TF.transform([r['fr'].decode('utf-8', 'replace') for r in REC])
wsc = np.array([float((Wv[i] @ Wv[j].T).toarray()[0, 0]) for i, j in pairs])
same = np.array([bool(REC[i]['rs'] & REC[j]['rs']) for i, j in pairs])
def report(nm_, sc):
    top = []; ch = []
    for n0 in range(0, len(pairs), K):
        blk = range(n0, min(n0 + K, len(pairs))); bst = max(blk, key=lambda n: sc[n]); top.append(same[bst]); ch.append(np.mean([same[n] for n in blk]))
    P(f"  {nm_:28s} best candidate of the same gold type {np.mean(top):.3f} (chance {np.mean(ch):.3f}) | AUC same type vs not {auc(sc[same], sc[~same]):.3f}")
report('frame words (tf-idf)', wsc)
st = subscore(lm); report('SUBSTITUTION, trained trunk', st)
s0 = subscore(lm0); report('substitution, untrained', s0)
rk = lambda x: np.argsort(np.argsort(x)).astype(float)
report('words + trained (rank avg)', (rk(wsc) + rk(st)) / 2)
P(f"DONE ({time.time() - T0:.0f}s)")
