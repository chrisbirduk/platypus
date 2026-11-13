"""RELATION IDENTITY FROM NAME-FREE CODES (13 Nov 2026; tools/relation_free.py) - offline, no effect on the package.
Brain research (the Tolman-Eichenbaum Machine; cognitive maps): relational structure is learned SEPARATELY from the items that fill it - what
stays the same when the names change. Here: a record = the words around and between two names with the names masked ('[X] , the capital of
[Y]'); can name-free codes tell which records express the same relation? Re-DocRED train + dev, gold mentions as names.
  LABEL-FREE ENCODER  hashed word and character n-grams of the masked record -> a linear code (256), trained by InfoNCE: two records whose
                      names are the SAME PAIR in DIFFERENT documents are a positive (the same two things related twice, in different words);
                      the other records in the batch are negatives. No relation label is used.
  EVALUATION          held-out documents (a fifth), gold relation records with both names in one sentence (1-12 words apart): for 600 queries,
                      30 candidates from other documents - is the nearest candidate of the same gold relation type (top-1; chance = the share of
                      the same type), AUC same type vs not. Against: frame words (tf-idf), the same n-grams untrained (cosine), the stand-in
                      trunk's state over the masked record (TG_CKPT; mean of the last layer), and trunk substitution (tools/relsub.py: 0.58-0.60).
RF_JEPA=<steps> (13 Nov 16:10): a JEPA on the name-masked records (masked span predicted in an EMA target encoder's space), its codes scored too.
Rows added 13 Nov 15:40 (Tran, Le & Ananiadou 2020: entity types alone are a strong cue): the gold entity-type pair, the name-shape pair, and
frame words or the encoder plus the type pair.
Usage: [TG_D=192 TG_CKPT=<model file>] python3 tools/relation_free.py [epochs, default 30]"""
import os
import sys
import json
import math
import time
import collections
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get('RF_THREADS', 1)))
P = lambda *a: print(*a, flush=True); T0 = time.time()
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 30
DIR = os.environ.get('DV_DIR', '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad/bench_dl')
docs = json.load(open(DIR + '/train_revised.json')) + json.load(open(DIR + '/dev_revised.json'))
rs0 = np.random.RandomState(0); dperm = rs0.permutation(len(docs)); TEST = set(dperm[:len(docs) // 5].tolist())


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    _u, _i, _c = np.unique(np.concatenate([pos, neg]), return_inverse=True, return_counts=True); r = (np.cumsum(_c) - (_c - 1) / 2.0)[_i]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


REC = []                                                                       # every name pair in a sentence, names masked
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    rel = collections.defaultdict(set)
    for l in d['labels']: rel[(l['h'], l['t'])].add(l['r']); rel[(l['t'], l['h'])].add(l['r'] + '^-1')
    bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms); toks = d['sents'][si]
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y]
                if ea == eb or a1 > b0 or not 1 <= b0 - a1 <= 12 or cn[ea] == cn[eb]: continue
                left = toks[max(0, a0 - 3):a0]; right = toks[b1:b1 + 3]
                txt = ' '.join(left + ['[X]'] + toks[a1:b0] + ['[Y]'] + right).lower()
                tp = lambda e: d['vertexSet'][e][0].get('type', 'X'); sh = lambda t: 'num' if any(ch.isdigit() for ch in ' '.join(t)) else ('cap' if all(w[:1].isupper() for w in t) else 'mix')
                REC.append(dict(di=di, txt=txt, cA=cn[ea], cB=cn[eb], key=tuple(sorted((cn[ea], cn[eb]))), rs=rel.get((ea, eb), set()), test=di in TEST, gt=(tp(ea), tp(eb)), st=(sh(toks[a0:a1]), sh(toks[b0:b1])),
                                raw=' '.join(toks[max(0, a0 - 3):b1 + 3])))
tr = [r for r in REC if not r['test']]; te = [r for r in REC if r['test'] and r['rs']]
bykey = collections.defaultdict(list)
for i, r in enumerate(tr): bykey[r['key']].append(i)
POS = [(i, j) for ids in bykey.values() if len(ids) >= 2 for a, i in enumerate(ids) for j in ids[a + 1:] if tr[i]['di'] != tr[j]['di']]
rq = np.random.RandomState(1); rq.shuffle(POS); POS = POS[:60000]
P(f"records {len(REC)} (training {len(tr)}, held-out gold-relation records {len(te)}) | label-free positive pairs (same names, different documents) "
  f"{len(POS)}, of which the same gold relation {np.mean([bool((tr[i]['rs'] & tr[j]['rs']) or (tr[i]['rs'] & {x.replace('^-1', '') if x.endswith('^-1') else x + '^-1' for x in tr[j]['rs']})) for i, j in POS[:5000]]):.2f} (sample)")
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
HW = HashingVectorizer(n_features=2 ** 18, ngram_range=(1, 3), alternate_sign=False, norm=None, token_pattern=r"\[x\]|\[y\]|[a-z]+|[^\sa-z]")
HC = HashingVectorizer(n_features=2 ** 17, analyzer='char_wb', ngram_range=(3, 4), alternate_sign=False, norm=None)
def feats(rr):
    from scipy.sparse import hstack
    X = hstack([HW.transform([r['txt'] for r in rr]), 0.5 * HC.transform([r['txt'] for r in rr])]).tocsr(); X.data = np.log1p(X.data)
    n = np.sqrt(X.multiply(X).sum(1)).A.ravel() + 1e-9; return X.multiply(1 / n[:, None]).tocsr()
Xtr = feats(tr); Xte = feats(te)
def tt(X):
    X = X.tocoo(); return torch.sparse_coo_tensor(np.vstack([X.row, X.col]), X.data.astype(np.float32), X.shape).coalesce()
torch.manual_seed(0); W = torch.nn.Parameter(torch.randn(Xtr.shape[1], 256) * 0.01); op = torch.optim.Adam([W], lr=2e-3)
Pi = np.array([p[0] for p in POS]); Pj = np.array([p[1] for p in POS]); B = 512
for ep in range(EP):
    perm = rq.permutation(len(POS)); ls = []
    for k in range(0, len(perm) - B + 1, B):
        ix = perm[k:k + B]; a = F.normalize(torch.sparse.mm(tt(Xtr[Pi[ix]]), W), dim=-1); b = F.normalize(torch.sparse.mm(tt(Xtr[Pj[ix]]), W), dim=-1)
        lg = a @ b.t() / 0.1; l = (F.cross_entropy(lg, torch.arange(len(ix))) + F.cross_entropy(lg.t(), torch.arange(len(ix)))) / 2
        op.zero_grad(); l.backward(); op.step(); ls.append(l.item())
    if (ep + 1) % 10 == 0 or ep == 0: P(f"   epoch {ep + 1}: InfoNCE {np.mean(ls):.3f}  ({time.time() - T0:.0f}s)")
with torch.no_grad(): Zte = F.normalize(torch.sparse.mm(tt(Xte), W), dim=-1).numpy()
TF = TfidfVectorizer(token_pattern=r"\[x\]|\[y\]|[a-z]+|[^\sa-z]").fit([r['txt'] for r in tr]); Tte = TF.transform([r['txt'] for r in te])
# queries
inv = lambda s: {x[:-3] if x.endswith('^-1') else x + '^-1' for x in s}
qs = rq.choice(len(te), min(600, len(te)), replace=False); PAIRS = []
for q in qs:
    cs = [int(c) for c in rq.choice(len(te), 40, replace=False) if te[int(c)]['di'] != te[q]['di']][:30]; PAIRS += [(int(q), c) for c in cs]
same = np.array([bool(te[i]['rs'] & te[j]['rs']) for i, j in PAIRS])
def report(nm_, sim):
    top, ch = [], []
    for n0 in range(0, len(PAIRS), 30):
        blk = [n for n in range(n0, min(n0 + 30, len(PAIRS)))]
        if not blk: continue
        b = max(blk, key=lambda n: sim[n]); top.append(same[b]); ch.append(np.mean(same[blk]))
    P(f"   {nm_:46s} nearest candidate of the same relation {np.mean(top):.3f} (chance {np.mean(ch):.3f}) | AUC same vs not {auc(sim[same], sim[~same]):.3f}")
P(f"HELD-OUT: {len(qs)} queries x 30 candidates from other documents (gold relation types {len({x for r in te for x in r['rs']})})")
report('frame words (tf-idf)', np.array([float((Tte[i] @ Tte[j].T).toarray()[0, 0]) for i, j in PAIRS]))
report('n-grams, untrained (cosine)', np.array([float((Xte[i] @ Xte[j].T).toarray()[0, 0]) for i, j in PAIRS]))
report('LABEL-FREE ENCODER (same names, other documents)', np.array([float(Zte[i] @ Zte[j]) for i, j in PAIRS]))
gtm = np.array([float(te[i]['gt'] == te[j]['gt']) for i, j in PAIRS]); stm = np.array([float(te[i]['st'] == te[j]['st']) for i, j in PAIRS])
fw = np.array([float((Tte[i] @ Tte[j].T).toarray()[0, 0]) for i, j in PAIRS])
report('GOLD entity-type pair alone (PER-LOC ...; upper bound)', gtm); report('name SHAPE pair alone (digits / capitalised)', stm)
report('frame words + gold type pair', fw + gtm); report('frame words + shape pair', fw + 0.5 * stm)
JS = int(os.environ.get('RF_JEPA', 0))
if JS:                                                                          # 13 Nov 16:10: a JEPA on the name-masked records (as arm G of the
    import torch.nn as nn                                                       #   JEPA index): a random span of the words is masked for the online
    VJ = 2 ** 15; LJ = 26; MSK = 1; PADJ = 0; XT, YT = 2, 3                     #   encoder; a predictor must give the EMA target encoder's states there
    def tok(t):                                                                 #   (full record, names still masked); L1, no negatives. The record's
        out = []                                                                #   code = mean target state; scored like the rows above.
        for w in t.split()[:LJ]:
            out.append(XT if w == '[x]' else YT if w == '[y]' else 4 + (hash(w) % (VJ - 4)))
        return out + [PADJ] * (LJ - len(out))
    import hashlib
    hash = lambda w: int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
    TT = torch.tensor([tok(r['txt']) for r in tr]); TE_ = torch.tensor([tok(r['txt']) for r in te])
    class Enc(nn.Module):
        def __init__(s):
            super().__init__(); s.e = nn.Embedding(VJ, 128); s.p = nn.Parameter(torch.randn(LJ, 128) * 0.02)
            s.t = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, 0.0, batch_first=True, norm_first=True), 2)
        def forward(s, X): return s.t(s.e(X) + s.p, src_key_padding_mask=(X == PADJ))
    torch.manual_seed(0); on = Enc(); tg = Enc(); tg.load_state_dict(on.state_dict()); [q.requires_grad_(False) for q in tg.parameters()]
    pred = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128)); opj = torch.optim.AdamW(list(on.parameters()) + list(pred.parameters()), lr=5e-4, weight_decay=0.05)
    rj = np.random.RandomState(5); lj = []
    for st in range(JS):
        ix = torch.tensor(rj.randint(0, len(TT), 256)); X = TT[ix]; Xm = X.clone(); M = torch.zeros_like(X, dtype=torch.bool)
        for b in range(len(X)):
            ok = [k for k in range(LJ) if X[b, k] > YT]
            if len(ok) < 2: continue
            L_ = max(1, int(0.4 * len(ok))); s0 = rj.randint(0, len(ok) - L_ + 1)
            for k in ok[s0:s0 + L_]: M[b, k] = True
        Xm[M] = MSK
        with torch.no_grad(): T_ = tg(X)
        H = pred(on(Xm)); l = (H[M] - T_[M]).abs().mean() if M.any() else H.sum() * 0
        opj.zero_grad(); l.backward(); opj.step(); lj.append(l.item())
        with torch.no_grad():
            for a, b in zip(on.parameters(), tg.parameters()): b.mul_(0.996).add_(0.004 * a)
        if (st + 1) % 500 == 0: P(f"   JEPA step {st + 1}: L1 {np.mean(lj[-200:]):.4f}  ({time.time() - T0:.0f}s)")
    with torch.no_grad():
        Z = []
        for i in range(0, len(TE_), 512):
            X = TE_[i:i + 512]; H = tg(X); m = (X != PADJ).float()[..., None]; Z.append(F.normalize((H * m).sum(1) / m.sum(1), dim=-1))
        Z = torch.cat(Z).numpy(); Zc = Z - Z.mean(0); Zc /= np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-9
    report(f'JEPA name-masked codes ({JS} steps, centred)', np.array([float(Zc[i] @ Zc[j]) for i, j in PAIRS]))
    report('JEPA codes + shape pair', np.array([float(Zc[i] @ Zc[j]) for i, j in PAIRS]) + 0.5 * stm)
RT = os.environ.get('RF_TYPES', '')
if RT:                                                                          # 13 Nov 16:00: LEARNED name types (tools/name_types.py)
    NT = json.load(open(RT)); ks = NT['k']; TY = NT['types']
    for ki, k in enumerate(ks):
        lt = lambda r: (TY.get(r['cA'], [-1] * len(ks))[ki], TY.get(r['cB'], [-1] * len(ks))[ki])
        ltm = np.array([float(lt(te[i]) == lt(te[j])) for i, j in PAIRS])
        report(f'LEARNED type pair, k {k}', ltm); report(f'frame words + LEARNED type pair, k {k}', fw + ltm)
report('LABEL-FREE ENCODER + gold type pair', np.array([float(Zte[i] @ Zte[j]) for i, j in PAIRS]) + gtm)
CK = os.environ.get('TG_CKPT', '')
if CK:
    sys.argv = [sys.argv[0], '6000', '5']; _tg = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trunk_guides.py')).read()
    g = {}; exec(compile(_tg[:_tg.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'), g); lm, L, PAD = g['lm'], g['L'], g['PAD']
    @torch.no_grad()
    def enc(texts, bs=128):
        out = []
        for i in range(0, len(texts), bs):
            ch = [t.replace('[x]', 'Thing').replace('[y]', 'Stuff').encode('utf-8', 'replace')[-L:] for t in texts[i:i + bs]]
            X = torch.full((len(ch), L), PAD, dtype=torch.long); m = torch.zeros(len(ch), L)
            for j, c in enumerate(ch): X[j, :len(c)] = torch.tensor(list(c)); m[j, :len(c)] = 1
            Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); T_ = Xi.shape[1]; am = torch.triu(torch.ones(T_, T_, dtype=torch.bool), 1)
            h = lm.emb(Xi) + lm.pos[:T_]
            for b in lm.b: h = b(h, am)
            h = lm.n(h); out.append(F.normalize((h * m[..., None]).sum(1) / m.sum(1, keepdim=True), dim=-1).numpy())
        return np.concatenate(out)
    Hte = enc([r['txt'] for r in te]); Hc = Hte - Hte.mean(0); Hc /= np.linalg.norm(Hc, axis=1, keepdims=True) + 1e-9
    report('trunk state over the masked record (centred)', np.array([float(Hc[i] @ Hc[j]) for i, j in PAIRS]))
P(f"DONE ({time.time() - T0:.0f}s)")
