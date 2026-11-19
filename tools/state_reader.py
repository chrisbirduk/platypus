"""STATE READER (19 Nov 2026; tools/state_reader.py) - offline, distant set, LABEL-FREE (own-document labels for evaluation only).
Can the reader improve as the TRUNK learns? The self-trained wording reader (usefulness pseudo-labels) is rebuilt on the trunk's own hidden
states instead of word counts: for a record (A before B in a sentence), the state after A, the mean state over the words between, and the state
just before B. Run once per stand-in trunk checkpoint (more training = a trunk that knows more language): if the state reader's score rises
with the trunk's training while the word-count reader stays flat, the reader can learn as the model learns (the GPU readout for kra10).
Reference (evaluation only): the same states trained on TRUE labels (ceiling).
Usage: TG_D=192 TG_CKPT=<trunk.pt> python3 tools/state_reader.py <distant.json.gz>   env SR_DOCS (4000), TG_NL (3), SR_THREADS (2)"""
import os, sys, gzip, json, math, time, collections
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
T0 = time.time(); P = lambda s: print(s, flush=True); torch.set_num_threads(int(os.environ.get('SR_THREADS', 2)))
L = 128; PAD = 256; V = 257; DW = int(os.environ.get('TG_D', 192)); NL = int(os.environ.get('TG_NL', 3))
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        h = s.n1(x); x = x + s.a(h, h, h, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class LM(nn.Module):
    def __init__(s, d=DW, nl=NL):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(L, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl))
        s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V)
    def t(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for b in s.b: x = b(x, am)
        return s.n(x)
lm = LM(); lm.load_state_dict(torch.load(os.environ['TG_CKPT'], map_location='cpu')); lm.eval()
DOCS = json.load(gzip.open(sys.argv[1])); ND = len(DOCS); SD = int(os.environ.get('SR_DOCS', 4000))
NOSP_B = {',', '.', ';', ':', ')', "'s", "n't", '%', '!', '?', "'", "''"}; NOSP_A = {'(', '$', '``'}
def detok(ws):
    out, prev = '', None
    for w in ws:
        if out and not (w in NOSP_B or prev in NOSP_A): out += ' '
        out += w; prev = w
    return out
pd, nd = collections.defaultdict(set), collections.defaultdict(set); recs = []
for di, d in enumerate(DOCS):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    lab = {(l['h'], l['t']) for l in d['labels']}; bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms)
        for x in range(len(ms)):
            for y_ in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y_]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb] or b0 - a1 > 12: continue
                pd[frozenset((cn[ea], cn[eb]))].add(di); nd[cn[ea]].add(di); nd[cn[eb]].add(di)
                if di < SD: recs.append((di, si, a0, a1, b0, b1, cn[ea], cn[eb], (ea, eb) in lab or (eb, ea) in lab))
P(f"{ND} documents (usefulness from all), {len(recs)} records from the first {SD}; trunk {os.environ['TG_CKPT'].split('/')[-1]} ({DW} x {NL})  ({time.time() - T0:.0f}s)")
def use_(r):
    a_ = len(pd[frozenset((r[6], r[7]))]) - 1; nA, nB = len(nd[r[6]]) - 1, len(nd[r[7]]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); sA, sB = np.log(ND / (nA + 2)), np.log(ND / (nB + 2))
    return max(hA * sB - (1 - hA) * 0.5, hB * sA - (1 - hB) * 0.5)
U = np.array([use_(r) for r in recs]); y = np.array([r[8] for r in recs], float); doc = np.array([r[0] for r in recs]); te = doc >= int(SD * 0.8); tr = ~te
def enc(r):
    ws = DOCS[r[0]]['sents'][r[1]]; pa = len(detok(ws[:r[3]]).encode('utf-8', 'replace')); pre = detok(ws[:r[4]]).encode('utf-8', 'replace')
    pb = len(pre) + (1 if r[4] > 0 else 0); cut = max(0, pb + 1 - (L - 1)); b = pre[cut:] + b' '
    return b[:L - 1], max(0, pa - cut), max(1, pb - cut)
F = np.zeros((len(recs), 3 * DW), np.float32)
with torch.no_grad():
    for i0 in range(0, len(recs), 256):
        ch = [enc(r) for r in recs[i0:i0 + 256]]; X = torch.full((len(ch), L), PAD, dtype=torch.long)
        for k, (b, pa, pb) in enumerate(ch): X[k, 1:1 + len(b)] = torch.tensor(list(b))
        H = lm.t(X)
        for k, (b, pa, pb) in enumerate(ch):
            ia, ib = min(pa, L - 1), min(pb, L - 1); g = H[k, ia:ib + 1] if ib >= ia else H[k, ib:ib + 1]
            F[i0 + k] = torch.cat([H[k, ia], g.mean(0), H[k, ib]]).numpy()
P(f"   states computed  ({time.time() - T0:.0f}s)")
sn = np.where(tr & ~np.isnan(U))[0]; hi, lo = np.quantile(U[sn], 0.8), np.quantile(U[sn], 0.5); ps, ng = sn[U[sn] >= hi], sn[U[sn] <= lo]
P(f"held out {te.sum()} (base {y[te].mean():.3f}); pseudo-labels {len(ps)} / {len(ng)} (true related {y[ps].mean():.3f} / {y[ng].mean():.3f})")
Fs = StandardScaler().fit(F[tr]).transform(F)
def fit(Xm, ii, yy, C=0.1): c = LogisticRegression(C=C, max_iter=500, class_weight='balanced'); c.fit(Xm[ii], yy); return c.decision_function(Xm)
hv = HashingVectorizer(n_features=2 ** 18, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
XW = hv.transform([' '.join(DOCS[r[0]]['sents'][r[1]][r[3]:r[4]]).lower() + f" DIST_{min(r[4] - r[3], 12) // 3}" for r in recs])
pl = np.r_[np.ones(len(ps)), np.zeros(len(ng))]; ix = np.r_[ps, ng]
sS = fit(Fs, ix, pl); sW = fit(XW, ix, pl, 1.0); sR = fit(Fs, np.where(tr)[0], y[tr])
def rep(nm, s, m):
    yy, ss = y[m], s[m]; o = np.argsort(-ss + 1e-9 * np.random.RandomState(0).rand(len(ss)))
    P(f"   {nm:56s} AUC {roc_auc_score(yy, ss):.3f} | precision top 1% {yy[o[:max(1, len(o) // 100)]].mean():.3f} 5% {yy[o[:max(1, len(o) // 20)]].mean():.3f} 10% {yy[o[:max(1, len(o) // 10)]].mean():.3f}  (n {m.sum()})")
for lab, m in (('ALL held-out records', te), ('pairs seen ONCE', te & np.isnan(U))):
    P(f"{lab}:")
    rep('WORD-COUNT reader, self-trained', sW, m); rep('STATE reader, self-trained (the trunk\'s own states)', sS, m)
    rep('state + word-count (ranks)', np.argsort(np.argsort(sS)) + np.argsort(np.argsort(sW)), m); rep('reference: STATES on TRUE labels (ceiling)', sR, m)
P(f"DONE ({time.time() - T0:.0f}s)")
