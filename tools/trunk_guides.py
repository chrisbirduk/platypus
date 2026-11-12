"""CAN THE TRUNK GUIDE THE HIPPOCAMPUS? (12 Nov 2026; tools/trunk_guides.py) - offline, readout only, no effect on the package.
After EM-LLM (event boundaries at the model's own surprise) and hippocampal indexing theory (memory keyed by the cortex's own states):
does a byte language model trained on ordinary text (a stand-in for the trunk; the toy architecture) carry what the store's reader lacks?
Scored against the Re-DocRED dev set's own mention annotations (500 documents, every named mention of every entity, with positions).
  T1 BOUNDARIES  where do mentions start and end? Baseline: the reader's cue, runs of capitalised words / numbers ('of' joins two capitalised
                 words). Trunk: the surprise of each word's first byte - a name continues predictably ('New' -> 'York'), a new unit is a
                 surprise. (a) AUC of that surprise on the joins inside capitalised runs: same mention vs a mention boundary; (b) on
                 SENTENCE-INITIAL capitalised words (capitalisation cannot tell): starts a mention vs not; (c) exact-span F1 of the
                 capitalised runs, of the runs split where the trunk is surprised (threshold fitted on the first half of the documents,
                 scored on the second), and the UNION's recall (guidance that only adds candidates).
  T2 LINKING     which mentions in a document are the same entity ('Willi Schneider' ... 'Schneider' ... 'the German')? Pairs of mentions
                 in one document: string features (exact, word overlap, last word, initials) against the trunk's states (mean over the
                 mention's bytes, and the last byte; centred cosine), and an UNTRAINED copy of the same network (byte overlap alone).
                 Logistic combinations fitted on the first half of the documents, scored on the second: pair AUC on all pairs and on HARD
                 pairs (no word in common), and antecedent accuracy (each mention's best-scoring earlier mention is the same entity).
Usage: python3 tools/trunk_guides.py [lm steps] [documents]     env TG_D (width, default 128), TG_NL (layers, default 3), TG_CKPT (model file)"""
import os
import re
import sys
import json
import collections
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
P = lambda *a: print(*a, flush=True); T0 = time.time()
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000; ND = int(sys.argv[2]) if len(sys.argv) > 2 else 500
DW = int(os.environ.get('TG_D', 128)); NL = int(os.environ.get('TG_NL', 3))
torch.set_num_threads(int(os.environ.get('TG_THREADS', 1))); torch.manual_seed(0); rs = np.random.RandomState(0)
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
DEV = os.environ.get('TG_DEV', SCR + '/bench_dl/dev_revised.json')
L = 128; PAD = 256; V = 257
# ---------------------------------------------------------------- documents with token byte offsets
NOSP_B = {',', '.', ';', ':', ')', "'s", "n't", '%', '!', '?', "'", "''"}; NOSP_A = {'(', '$', '``'}
DOCS = []
for d in json.load(open(DEV))[:ND]:
    toks = []; out = b''; prev = None
    for si, sent in enumerate(d['sents']):
        for ti, w in enumerate(sent):
            wb = w.encode('utf-8')
            if out and not (w in NOSP_B or prev in NOSP_A): out += b' '
            toks.append((si, ti, len(out), len(out) + len(wb), w)); out += wb; prev = w
    ix = {(si, ti): k for k, (si, ti, a, b, w) in enumerate(toks)}
    ments = []
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs:
            k0, k1 = ix[(m['sent_id'], m['pos'][0])], ix[(m['sent_id'], m['pos'][1] - 1)]
            ments.append(dict(e=ei, k0=k0, k1=k1, a=toks[k0][2], b=toks[k1][3], name=m['name'], sid=m['sent_id'], typ=m['type']))
    DOCS.append(dict(b=out, toks=toks, ments=sorted(ments, key=lambda m: (m['a'], m['b']))))
nm = sum(len(D['ments']) for D in DOCS)
P(f"documents {len(DOCS)}, mentions {nm}, bytes {sum(len(D['b']) for D in DOCS)}  ({time.time() - T0:.0f}s)")
# ---------------------------------------------------------------- the byte LM (the toy's causal transformer)
raw = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin', 'rb').read()
corpus = np.frombuffer(raw.replace(b'\n', b' ') + b' ' + b' '.join(D['b'] for D in DOCS), dtype=np.uint8)
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
    def forward(s, X): return s.o(s.t(X))
def train(steps):
    m = LM(); op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01); ls = []
    sch = torch.optim.lr_scheduler.OneCycleLR(op, 1e-3, total_steps=max(1, steps), pct_start=0.05)
    for st in range(steps):
        i0 = rs.randint(0, len(corpus) - L - 1, 32); X = torch.tensor(np.stack([corpus[i:i + L] for i in i0]).astype(np.int64))
        Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); l = F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1))
        op.zero_grad(); l.backward(); op.step(); sch.step(); ls.append(l.item())
        if (st + 1) % 1000 == 0: P(f"   step {st + 1}: {np.mean(ls[-200:]) / math.log(2):.2f} bits per byte  ({time.time() - T0:.0f}s)")
    m.eval(); return m
torch.manual_seed(1); lm0 = LM().eval()                                       # the UNTRAINED control (same network, random weights)
CK = os.environ.get('TG_CKPT', '')                                            # TG_CKPT: load the trained model if the file exists, else train and save it
if CK and os.path.exists(CK): lm = LM(); lm.load_state_dict(torch.load(CK)); lm.eval(); P(f"   loaded {CK}")
else:
    torch.manual_seed(0); lm = train(STEPS)
    if CK: torch.save(lm.state_dict(), CK)
@torch.no_grad()
def run(m, b):
    """per byte: surprise (bits) and last-layer state, each byte read with up to 127 bytes before it (windows of 128, stride 64)"""
    n = len(b); S = np.zeros(n, np.float32); H = np.zeros((n, m.o.in_features), np.float32); arr = np.frombuffer(b, np.uint8).astype(np.int64)
    st = 0
    while True:
        a = max(0, st - 64) if st else 0; w = arr[a:a + L]; X = torch.tensor(w)[None]; Xi = torch.cat([torch.full((1, 1), PAD), X[:, :-1]], 1)
        h = m.t(Xi); lp = F.log_softmax(m.o(h), -1)[0]; s_ = (-lp[torch.arange(len(w)), X[0]] / math.log(2)).numpy()
        k = st - a; S[st:a + len(w)] = s_[k:]; H[st:a + len(w)] = h[0, k:].numpy()
        if a + len(w) >= n: break
        st = a + len(w)
    return S, H
for D in DOCS: D['S'], D['H'] = run(lm, D['b']); D['H0'] = run(lm0, D['b'])[1]
P(f"trunk states computed  ({time.time() - T0:.0f}s)")
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    r = np.concatenate([pos, neg]).argsort(kind='stable').argsort().astype(float) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
half = len(DOCS) // 2
def rankavg(p1, n1, p2, n2):
    a = np.concatenate([p1, n1]).argsort().argsort(); b = np.concatenate([p2, n2]).argsort().argsort(); c = (a + b) / 2.0
    return c[:len(p1)], c[len(p1):]
# ---------------------------------------------------------------- T1 boundaries
cap = lambda w: w[:1].isupper() or w[:1].isdigit()
J_same, J_bnd = [], []; I_pos, I_neg = [], []
def runs_of(D, split_at=None):
    tk = D['toks']; out = []; cur = None
    for k, (si, ti, a, b, w) in enumerate(tk):
        joinable = cur is not None and tk[cur[1]][0] == si and (cap(w) or (w == 'of' and k + 1 < len(tk) and cap(tk[k + 1][4]) and tk[k + 1][0] == si))
        if joinable and split_at is not None and cap(w) and D['S'][a] > split_at: joinable = False
        if cap(w) or (joinable and w == 'of'):
            if joinable: cur[1] = k
            else:
                if cur: out.append(tuple(cur))
                cur = [k, k]
        else:
            if cur: out.append(tuple(cur))
            cur = None
    if cur: out.append(tuple(cur))
    return [(a, b) for a, b in out if tk[b][4] != 'of']
for D in DOCS:
    tk = D['toks']; mk = {}; starts = set(); D['starts'] = starts
    for i, m in enumerate(D['ments']):
        starts.add(m['k0'])
        for k in range(m['k0'], m['k1'] + 1): mk.setdefault(k, i)
    for a, b in runs_of(D):
        for k in range(a + 1, b + 1):
            if not cap(tk[k][4]): continue
            same = k in mk and (k - 1) in mk and mk[k] == mk[k - 1] and k not in starts
            (J_same if same else J_bnd).append(D['S'][tk[k][2]])
    for k, (si, ti, a, b, w) in enumerate(tk):
        if ti == 0 and w[:1].isupper(): (I_pos if k in starts else I_neg).append(float(D['S'][a]))
WF = collections.Counter(); WC = collections.Counter()                        # the reader's own statistic: how often a word is capitalised
for mt in re.finditer(rb"(?<=[a-z,;] )([A-Za-z][a-z'\-]*)", corpus.tobytes()):   #   when it is NOT the first word of a sentence
    w = mt.group(1).decode('latin-1'); WF[w.lower()] += 1; WC[w.lower()] += w[:1].isupper()
ENT = lambda w: (WC[w.lower()] + 0.5) / (WF[w.lower()] + 1.0)
I_ent_pos = [ENT(D['toks'][k][4]) for D in DOCS for k, t_ in enumerate(D['toks']) if t_[1] == 0 and t_[4][:1].isupper() and k in D['starts']]
I_ent_neg = [ENT(D['toks'][k][4]) for D in DOCS for k, t_ in enumerate(D['toks']) if t_[1] == 0 and t_[4][:1].isupper() and k not in D['starts']]
P(f"\nT1 BOUNDARIES (word-level; 'surprise' = the trunk's surprise at the word's first byte)")
P(f"  (a) joins inside capitalised runs: same mention {len(J_same)} vs boundary {len(J_bnd)} | AUC of LOW surprise = same mention "
  f"{auc([-x for x in J_same], [-x for x in J_bnd]):.3f} (0.5 = no signal) | mean surprise {np.mean(J_same):.2f} vs {np.mean(J_bnd):.2f} bits")
P(f"  (b) sentence-initial capitalised words: start a mention {len(I_pos)} vs not {len(I_neg)} | AUC of HIGH surprise = mention "
  f"{auc(I_pos, I_neg):.3f} | BASELINE, how often the word is capitalised mid-sentence in the corpus (the reader's entity statistic) {auc(I_ent_pos, I_ent_neg):.3f} | "
  f"rank average of the two {auc(*rankavg(I_pos, I_neg, I_ent_pos, I_ent_neg)):.3f}")
def prf(Ds, split_at=None, union=False):
    tp = npred = ngold = 0
    for D in Ds:
        g = {(m['k0'], m['k1']) for m in D['ments']}; p = set(runs_of(D, split_at))
        if union: p |= set(runs_of(D))
        tp += len(g & p); npred += len(p); ngold += len(g)
    pr, rc = tp / max(1, npred), tp / max(1, ngold); return pr, rc, 2 * pr * rc / max(1e-9, pr + rc)
best = max(((prf(DOCS[:half], th)[2], th) for th in list(np.arange(1.0, 20.01, 0.5)) + [99.0]))
p0, r0, f0 = prf(DOCS[half:]); p1, r1, f1 = prf(DOCS[half:], best[1]); p2, r2, f2 = prf(DOCS[half:], best[1], True)
P(f"  (c) exact mention spans, second half of the documents: CAPITALISED RUNS P {p0:.3f} R {r0:.3f} F1 {f0:.3f} | SPLIT WHERE THE TRUNK IS SURPRISED "
  f"(> {best[1]:.1f} bits, fitted on the first half) P {p1:.3f} R {r1:.3f} F1 {f1:.3f} | UNION (guidance only adds) P {p2:.3f} R {r2:.3f}")
# ---------------------------------------------------------------- T2 linking
def toks_of(s): return [w.lower() for w in re.findall(r"[A-Za-z0-9]+", s)]
def c3(s):
    t = ' ' + re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip() + ' '; return {t[i:i + 3] for i in range(len(t) - 2)}
def initials(s): return ''.join(w[0] for w in re.findall(r"[A-Za-z]+", s) if w[0].isupper()).lower()
mu = np.mean(np.concatenate([D['H'][::7] for D in DOCS]), 0); mu0 = np.mean(np.concatenate([D['H0'][::7] for D in DOCS]), 0)
def vecs(D, H, mu_):
    out = []
    for m in D['ments']:
        v1 = H[m['a']:m['b']].mean(0) - mu_; v2 = H[m['b'] - 1] - mu_
        out.append((v1 / (np.linalg.norm(v1) + 1e-9), v2 / (np.linalg.norm(v2) + 1e-9)))
    return out
ROWS = []                                                                    # (doc half, i, j, same, features...)
for di, D in enumerate(DOCS):
    M = D['ments']; vt = vecs(D, D['H'], mu); v0 = vecs(D, D['H0'], mu0)
    for j in range(len(M)):
        for i in range(j):
            a, b = M[i], M[j]
            if a['a'] == b['a']: continue
            ta, tb = set(toks_of(a['name'])), set(toks_of(b['name']))
            jac = len(ta & tb) / max(1, len(ta | tb)); ex = float(a['name'].lower() == b['name'].lower())
            hd = float(bool(ta) and bool(tb) and toks_of(a['name'])[-1:] == toks_of(b['name'])[-1:])
            ia, ib = initials(a['name']), initials(b['name'])
            ini = float(len(ia) > 1 and (ia == re.sub(r'[^a-z]', '', b['name'].lower()) or ib == re.sub(r'[^a-z]', '', a['name'].lower())))
            ca, cb = c3(a['name']), c3(b['name']); cj = len(ca & cb) / max(1, len(ca | cb))
            ROWS.append((di >= half, di, i, j, a['e'] == b['e'], jac, ex, hd, ini, float(vt[i][0] @ vt[j][0]), float(vt[i][1] @ vt[j][1]),
                         float(v0[i][0] @ v0[j][0]), float(v0[i][1] @ v0[j][1]), float(a['typ'] == b['typ']), float(len(ta & tb) == 0), cj))
R = np.array(ROWS, dtype=np.float64); te = R[:, 0] == 1; y = R[:, 4] == 1; hard = R[:, 14] == 1; vhard = hard & (R[:, 15] < 0.25)
P(f"\nT2 LINKING: {len(R)} mention pairs in one document, {int(y.sum())} the same entity ({int((y & hard).sum())} of those share NO word; {int((y & vhard).sum())} also share few letter-triples, < 0.25)")
FEAT = {'words only': [5, 6, 7, 8], 'string (+ letter triples)': [5, 6, 7, 8, 15], 'trunk': [9, 10], 'untrained': [11, 12],
        'string + trunk': [5, 6, 7, 8, 15, 9, 10], 'string + untrained': [5, 6, 7, 8, 15, 11, 12]}
def fit(cols):
    X = torch.tensor(R[~te][:, cols], dtype=torch.float32); Y = torch.tensor(y[~te], dtype=torch.float32); w = torch.zeros(len(cols), requires_grad=True)
    b = torch.zeros(1, requires_grad=True); op = torch.optim.LBFGS([w, b], max_iter=200)
    pw = float((1 - Y.mean()) / max(1e-6, Y.mean()))
    def cl():
        op.zero_grad(); l = F.binary_cross_entropy_with_logits(X @ w + b, Y, pos_weight=torch.tensor(pw)); l.backward(); return l
    op.step(cl); return (torch.tensor(R[:, cols], dtype=torch.float32) @ w + b).detach().numpy()
def antecedent(score):
    best = {}
    for k in np.where(te)[0]:
        key = (int(R[k, 1]), int(R[k, 3]))
        if key not in best or score[k] > best[key][0]: best[key] = (score[k], bool(y[k]))
    has = {(int(R[k, 1]), int(R[k, 3])) for k in np.where(te & y)[0]}
    return np.mean([best[k][1] for k in has]) if has else float('nan'), len(has)
for nm_, cols in FEAT.items():
    s = fit(cols); acc, n_ = antecedent(s)
    P(f"  {nm_:26s} pair AUC {auc(s[te & y], s[te & ~y]):.3f} | HARD (no shared word) {auc(s[te & y & hard], s[te & ~y & hard]):.3f} | "
    f"VERY HARD (+ few shared letter-triples) {auc(s[te & y & vhard], s[te & ~y & vhard]):.3f} | "
      f"antecedent right {acc:.3f} ({n_} mentions with an earlier mention of their entity)")
P(f"  raw centred cosine (no fitting), HARD pairs: trunk mean {auc(R[te & y & hard, 9], R[te & ~y & hard, 9]):.3f}, last byte "
  f"{auc(R[te & y & hard, 10], R[te & ~y & hard, 10]):.3f} | untrained mean {auc(R[te & y & hard, 11], R[te & ~y & hard, 11]):.3f}, last byte "
  f"{auc(R[te & y & hard, 12], R[te & ~y & hard, 12]):.3f}")
ex_ = [(D['ments'][int(R[k, 2])]['name'], D['ments'][int(R[k, 3])]['name']) for k in rs.permutation(np.where(te & y & vhard)[0])[:10] for D in [DOCS[int(R[k, 1])]]]
P(f"  examples of VERY HARD same-entity pairs: {ex_}")
P(f"DONE ({time.time() - T0:.0f}s)")
