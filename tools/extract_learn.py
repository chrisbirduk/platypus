"""LEARNING WHAT TO EXTRACT FROM THE TRUNK'S FEEDBACK (13 Nov 2026; tools/extract_learn.py) - offline, no effect on the package.
The idea under test: machinery that over-generates structural pieces of wording, and feedback from the trunk that teaches which pieces are
needed. One record's feedback is noisy (extraction gain AUC 0.45-0.58); a LEARNER pools it over many records, so a wording that pays off a
little but consistently ('X , the capital of Y', 'X was born in Y') can be learned as worth extracting.
  CANDIDATES  every pair of names in one sentence (<= 12 words, <= 60 bytes apart) with the words between - Re-DocRED train + dev
  FEEDBACK    for candidates whose names meet again in another document: the bits the record saves the trunk on the second name there,
              minus what the same two names as a LIST ('A , B .') save (plain co-occurrence) - label-free. Variant: minus the names alone.
  LEARNER     ridge regression on the feedback, from features any reader has (no gold): the words between (hashed words and word pairs),
              its first and last word, its length, commas / brackets / digits, how often that wording recurs among all candidates
  TEST        documents held out (a fifth, never used for feedback): score EVERY candidate there; against gold - AUC related vs not,
              precision and recall of the top 10% / 25% kept. Baselines: closeness, the wording's frequency, random; CEILING: the same
              learner and features trained on the GOLD labels (what these features allow at best). The current reader: ~3% recall, ~0.23 precision.
EL_TARGET=frame (13 Nov 14:20): feedback on the TARGET's relating words (the words between the names where they meet again, same order)
against another record's relating words - relation wording predicting relation wording, not the second name.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/extract_learn.py [max feedback records, default 8000]"""
import os
import sys
import json
import collections
HERE = os.path.dirname(os.path.abspath(__file__))
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
sys.argv = [sys.argv[0], '6000', '5']
_tg = open(os.path.join(HERE, 'trunk_guides.py')).read()
exec(compile(_tg[:_tg.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'))
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
P(f"\n================ LEARNING WHAT TO EXTRACT  ({time.time() - T0:.0f}s)")
TH = b'Thing'
DIR = SCR + '/bench_dl'
docs = json.load(open(DIR + '/train_revised.json')) + json.load(open(DIR + '/dev_revised.json'))
txt = lambda toks: ' '.join(toks).encode('utf-8', 'replace')


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    _u, _i, _c = np.unique(np.concatenate([pos, neg]), return_inverse=True, return_counts=True); r = (np.cumsum(_c) - (_c - 1) / 2.0)[_i]   # ties share their mean rank (13 Nov fix)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.no_grad()
def bits(items, bs=128):
    out = np.zeros(len(items))
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; cut = [max(0, len(s) - L) for s, _ in ch]; X = torch.full((len(ch), L), PAD, dtype=torch.long)
        for j, (s, _) in enumerate(ch): X[j, :len(s) - cut[j]] = torch.tensor(list(s[cut[j]:]))
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(lm(Xi), -1)
        nl = -lp.gather(-1, X.clamp(max=V - 1)[..., None])[..., 0] / math.log(2)
        for j, ((s, (a, b)), c0) in enumerate(zip(ch, cut)): out[i + j] = float(nl[j, max(0, a - c0):max(0, b - c0)].sum())
    return out


# ---- candidates
rs0 = np.random.RandomState(0); dperm = rs0.permutation(len(docs)); TEST = set(dperm[:len(docs) // 5].tolist())
C = []; occ = collections.defaultdict(list)
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
                (a0, a1), (b0, b1) = ms[x][:2], ms[y][:2]; ea, eb = ms[x][2], ms[y][2]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb]: continue
                key = tuple(sorted((cn[ea], cn[eb]))); occ[key].append((di, si, b0, b1, a0, a1, cn[ea]))
                if b0 - a1 <= 12 and len(' '.join(toks[a0:b1])) <= 60:
                    C.append(dict(di=di, si=si, a=(a0, a1), b=(b0, b1), key=key, gold=(ea, eb) in gold, mid=toks[a1:b0],
                                  A=toks[a0:a1], B=toks[b0:b1], test=di in TEST, cA=cn[ea]))
fcount = collections.Counter(' '.join(c['mid']).lower() for c in C)
y_all = np.array([c['gold'] for c in C]); te = np.array([c['test'] for c in C])
P(f"candidates {len(C)} in {len(docs)} documents (related {y_all.mean():.3f}) | held-out documents {len(TEST)}: {int(te.sum())} candidates "
  f"(related {y_all[te].mean():.3f})")

# ---- trunk feedback on training documents (where the names meet again in ANOTHER training document)
MODE = os.environ.get('EL_TARGET', 'name')                                  # 'frame': feedback on the TARGET's relating words
rq = np.random.RandomState(1); FB = []
for n, c in enumerate(C):
    if c['test']: continue
    oth = [o for o in occ[c['key']] if o[0] != c['di'] and o[0] not in TEST]
    if MODE == 'frame':                                                         # the names meet again in the SAME order, 1-12 words apart
        oth = [o for o in oth if 1 <= o[2] - o[5] <= 12 and o[6] == c['cA']]
    if oth: c['tgt'] = oth[rq.randint(len(oth))]; FB.append(n)
rq.shuffle(FB); FB = FB[:NMAX]
SEQ = {k: [] for k in ('w', 'o', 'lw', 'lo', 'pw', 'po')}
if MODE == 'frame':
    SEQ = {k: [] for k in ('w', 'o')}; pr_ = np.random.RandomState(4).permutation(len(FB))
    for k_, n in enumerate(FB):
        c = C[n]; di2, si2, y0, y1, x0, x1, _ = c['tgt']; t2 = docs[di2]['sents'][si2]; r_ = C[FB[int(pr_[k_])]]
        A, B, M, Mn = txt(c['A']), txt(c['B']), txt(c['mid']), txt(r_['mid'])
        pre = txt(t2[x0:x1]) + b' '; tg = txt(t2[x1:y0])[:40]                    # target: the words between the names where they meet again
        for k, rec in (('w', A + b' ' + M + b' ' + B), ('o', A + b' ' + Mn + b' ' + B)):
            q = rec.replace(b'  ', b' ') + b' . ' + pre; SEQ[k].append((q + tg, (len(q), len(q) + len(tg))))
    t1 = time.time(); Bt = {k: bits(v) for k, v in SEQ.items()}; g = Bt['o'] - Bt['w']; fb_list = fb_pair = g; yf = y_all[FB]
    P(f"FRAME FEEDBACK on {len(FB)} training records ({time.time() - t1:.0f}s), related {yf.mean():.3f}: bits the record's own relating words save on the "
      f"relating words where its names meet again, against another record's words - per-record AUC related vs not {auc(g[yf], g[~yf]):.3f}")
for n in (FB if MODE != 'frame' else []):
    c = C[n]; di2, si2, y0, y1 = c['tgt'][:4]; t2 = docs[di2]['sents'][si2]
    pre = txt(t2[max(0, y0 - 6):y0])[-36:] + b' '; tg = txt(t2[y0:y1])[:24]; A, B, M = txt(c['A']), txt(c['B']), txt(c['mid'])
    for k, rec in (('w', A + b' ' + M + b' ' + B), ('o', TH + b' ' + M + b' ' + B), ('lw', A + b' , ' + B), ('lo', TH + b' , ' + B),
                   ('pw', A + b' ' + B), ('po', TH + b' ' + B)):
        q = rec.replace(b'  ', b' ') + b' . ' + pre; SEQ[k].append((q + tg, (len(q), len(q) + len(tg))))
if MODE != 'frame':
  t1 = time.time(); Bt = {k: bits(v) for k, v in SEQ.items()}
  g = Bt['o'] - Bt['w']; fb_list = g - (Bt['lo'] - Bt['lw']); fb_pair = g - (Bt['po'] - Bt['pw'])
  yf = y_all[FB]
  P(f"trunk feedback on {len(FB)} training records ({time.time() - t1:.0f}s), related {yf.mean():.3f} | per-record AUC related vs not: record gain "
  f"{auc(g[yf], g[~yf]):.3f}, beyond the LIST record {auc(fb_list[yf], fb_list[~yf]):.3f}, beyond the names alone {auc(fb_pair[yf], fb_pair[~yf]):.3f}")

# ---- features (no gold)
HV = HashingVectorizer(n_features=2 ** 16, ngram_range=(1, 2), alternate_sign=False, token_pattern=r"[A-Za-z]+|[^\sA-Za-z]", norm='l2')
def feats(ix):
    mids = [' '.join(C[i]['mid']) for i in ix]
    W = HV.transform(mids); ends = HV.transform([('FIRST_' + (C[i]['mid'][0] if C[i]['mid'] else 'none') + ' LAST_' + (C[i]['mid'][-1] if C[i]['mid'] else 'none')).lower() for i in ix])
    D = np.array([[len(C[i]['mid']), float(',' in C[i]['mid']), float('(' in C[i]['mid'] or ')' in C[i]['mid']), float(any(w[:1].isdigit() for w in C[i]['A'] + C[i]['B'])),
                   float(any(w[:1].isupper() for w in C[i]['mid'])), np.log1p(fcount[' '.join(C[i]['mid']).lower()]), float(len(C[i]['mid']) == 0)] for i in ix], float)
    D = (D - D.mean(0)) / (D.std(0) + 1e-6)
    return hstack([W, ends, csr_matrix(D)]).tocsr()
TE = [i for i, c in enumerate(C) if c['test']]; yt = y_all[TE]; Xte = feats(TE); Xfb = feats(FB)
TRG = [i for i, c in enumerate(C) if not c['test']]; Xtr = feats(TRG)


def rep(tag, sc):
    o = np.argsort(-sc); k1, k2 = len(sc) // 10, len(sc) // 4
    P(f"   {tag:58s} AUC {auc(sc[yt], sc[~yt]):.3f} | top 10%: precision {yt[o[:k1]].mean():.3f}, recall {yt[o[:k1]].sum() / yt.sum():.3f} | "
      f"top 25%: precision {yt[o[:k2]].mean():.3f}, recall {yt[o[:k2]].sum() / yt.sum():.3f}")


P(f"HELD-OUT DOCUMENTS: {len(TE)} candidates, related {yt.mean():.3f} (current reader: ~3% recall of gold facts at ~0.23 precision)")
rep('random', np.random.RandomState(3).rand(len(TE)))
rep('closeness of the names', -np.array([len(C[i]['mid']) for i in TE], float))
rep("the wording's frequency among candidates", np.array([fcount[' '.join(C[i]['mid']).lower()] for i in TE], float))
for nm_, tgt in (('beyond the LIST record', fb_list), ('beyond the names alone', fb_pair), ('raw record gain', g)):
    t_ = np.clip(tgt, np.percentile(tgt, 1), np.percentile(tgt, 99))
    for a_ in (1.0, 10.0):
        r_ = Ridge(alpha=a_).fit(Xfb, t_); rep(f"LEARNED FROM TRUNK FEEDBACK ({nm_}, ridge {a_:g})", r_.predict(Xte))
    fs = collections.defaultdict(list)                                         # pooled by exact wording (no generalisation across wordings)
    for k, n in enumerate(FB): fs[' '.join(C[n]['mid']).lower()].append(t_[k])
    mu = float(np.mean(t_)); rep(f"  pooled per wording ({nm_})", np.array([np.mean(fs[w]) * len(fs[w]) / (len(fs[w]) + 3) + mu * 3 / (len(fs[w]) + 3) if (w := ' '.join(C[i]['mid']).lower()) in fs else mu for i in TE]))
yc = y_all[FB]
rep("CEILING A: same learner on GOLD labels, same feedback records", LogisticRegression(C=1.0, max_iter=2000).fit(Xfb, yc).decision_function(Xte))
rep("CEILING B: same learner on GOLD labels, ALL training candidates", LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y_all[TRG]).decision_function(Xte))
P(f"DONE ({time.time() - T0:.0f}s)")
