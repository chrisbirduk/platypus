"""FRAME PROBE (11 Nov 2026; tools/frame_probe.py) - step 1 of the frame-JEPA reader: does a frame encoder trained ONLY by a joint-embedding
objective on unlabelled Wikipedia carry the RELATION, with the entities masked?

TRAINING (no labels): Wikipedia sentences (the local minipile cache); in each, two filler spans are masked ([A], [B]; any other filler
[E]) and the frame encoder (word embeddings -> BiGRU -> the states at [A] and [B] -> a relation code -> a D x D operator) must map the
code of A's span to the code of B's span (InfoNCE against every span in the batch, plus the reverse operator, plus isotropy). A span's
code comes from its own WORDS (hashed word embeddings, mean, linear): v1 gave each span a free code, and most spans occur once, so the
targets were random and the loss stayed at chance (epoch 1: 0.37, below the untrained encoder's 0.39). Fillers
for this probe are capitalised word runs and 4-digit numbers (a scaffold: the learned segmentation is step 2).
TEST: the embedded Re-DocRED benchmark (gold Wikidata relations). Every sentence naming both entities of a gold pair, both masked; is a
frame's nearest other frame of the SAME gold relation type (leave-one-out, cosine)? Against the majority type, an UNTRAINED encoder of the
same shape, and a BAG OF CONNECTOR WORDS (tf-idf of the frame's words). Also a linear probe (5-fold).
Usage: python3 tools/frame_probe.py [epochs] [seed]"""
import os
import re
import sys
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(2)
WIKI = os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin'

raw = open(WIKI, 'rb').read().decode('utf-8', 'replace')
sents = [s.split() for s in re.split(r'(?<=[.!?]) (?=[A-Z"])', raw.replace('\n', ' ')) if 6 <= len(s.split()) <= 50]
wc = collections.Counter(w.lower() for s in sents for w in s)
common = {w for w, _ in wc.most_common(3000)}
def spans(toks):
    """filler spans: capitalised runs (up to 4 words) and 4-digit numbers; a sentence-initial common word is not a filler"""
    out = []; i = 0
    while i < len(toks):
        t = toks[i]
        if re.fullmatch(r'[A-Z][\w\'-]*', t) and not (i == 0 and t.lower() in common):
            j = i
            while j + 1 < len(toks) and j - i < 3 and re.fullmatch(r'[A-Z][\w\'-]*', toks[j + 1]): j += 1
            out.append((i, j + 1)); i = j + 1
        elif re.fullmatch(r'1[0-9]{3}|20[0-9]{2}', t): out.append((i, i + 1)); i += 1
        else: i += 1
    return out
vocab = {w: k + 4 for k, (w, _) in enumerate(wc.most_common(20000))}       # 0 pad, 1 unk, 2 [A], 3 [B]; [E] = 1 (other filler: unknown word)
def frame(toks, sp, a, b):
    out = []; i = 0; pa = pb = None
    marks = {s[0]: s for s in sp}
    while i < len(toks):
        if i == a[0]: pa = len(out); out.append(2); i = a[1]; continue
        if i == b[0]: pb = len(out); out.append(3); i = b[1]; continue
        if i in marks: out.append(1); i = marks[i][1]; continue
        out.append(vocab.get(toks[i].lower(), 1)); i += 1
    return out, pa, pb
train = []; spid = {}
for s in sents:
    sp = spans(s)
    for x, y in zip(sp, sp[1:]):                                              # adjacent filler pairs (a scaffold for the probe)
        f, pa, pb = frame(s, sp, x, y); ka = ' '.join(s[x[0]:x[1]]); kb = ' '.join(s[y[0]:y[1]])
        if ka == kb: continue
        train.append((f, pa, pb, spid.setdefault(ka, len(spid)), spid.setdefault(kb, len(spid))))
print(f"training frames {len(train)} from {len(sents)} Wikipedia sentences, {len(spid)} filler spans", flush=True)

from platypus.inference import probes
import platypus.config as C
cfg = C.Config(); cfg.inf_bench = 'embedded'; G = probes._bench(cfg)
lab = {}
for h, t, fl, ps in G['rel']: lab.setdefault((h, t), set()).update(ps)
names = {int(k): v for k, v in G['ent_name'].items()}
test = []
for d in [d for k in sorted(G['docs']) for d in G['docs'][k]]:
    for s in re.split(r'(?<=[.!?]) (?=[A-Z"])', d):
        toks = s.split(); sp = spans(toks); js = ' '.join(toks)
        for (h, t), ps in lab.items():
            nh, nt = names.get(h), names.get(t)
            if not nh or not nt or nh not in js or nt not in js or nh in nt or nt in nh: continue
            fh = [(i, i + len(nh.split())) for i in range(len(toks)) if ' '.join(toks[i:i + len(nh.split())]) == nh]
            ft = [(i, i + len(nt.split())) for i in range(len(toks)) if ' '.join(toks[i:i + len(nt.split())]) == nt]
            if not fh or not ft: continue
            a, b = sorted([fh[0], ft[0]])
            if a[1] > b[0]: continue
            sp2 = [x for x in sp if x[1] <= a[0] or x[0] >= a[1]]; sp2 = [x for x in sp2 if x[1] <= b[0] or x[0] >= b[1]] + [a, b]
            f, pa, pb = frame(toks, sorted(sp2), a, b); test.append((f, pa, pb, ps, toks[a[1]:b[0]]))
P_ = collections.Counter(p for *_, ps, _ in test for p in ps)
print(f"test frames {len(test)} (gold pairs named in one sentence, entities masked) | commonest types {P_.most_common(6)}", flush=True)


class Enc(nn.Module):
    def __init__(s, V, d=128, r=64, D=32):
        super().__init__(); s.emb = nn.Embedding(V, d, padding_idx=0); s.rnn = nn.GRU(d, d, batch_first=True, bidirectional=True)
        s.rel = nn.Linear(4 * d, r); s.op = nn.Linear(r, D * D); s.opr = nn.Linear(r, D * D); s.D = D
    def code(s, F_, pa, pb):
        h, _ = s.rnn(s.emb(F_)); ix = torch.arange(len(F_)); return s.rel(torch.cat([h[ix, pa], h[ix, pb]], -1))
    def ops(s, rho):
        I = torch.eye(s.D); return I + s.op(rho).view(-1, s.D, s.D), I + s.opr(rho).view(-1, s.D, s.D)
def batch(rows):
    L = max(len(r[0]) for r in rows); F_ = torch.zeros(len(rows), L, dtype=torch.long)
    for i, r in enumerate(rows): F_[i, :len(r[0])] = torch.tensor(r[0])
    return F_, torch.tensor([r[1] for r in rows]), torch.tensor([r[2] for r in rows])
def codes(enc, rows):
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 512):
            F_, pa, pb = batch(rows[i:i + 512]); out.append(F.normalize(enc.code(F_, pa, pb), dim=-1))
    return torch.cat(out).numpy()
def knn(X):
    S = X @ X.T; np.fill_diagonal(S, -9); nn_ = S.argmax(1)
    return float(np.mean([bool(test[i][3] & test[j][3]) for i, j in enumerate(nn_)]))
def linprobe(X):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    y = np.array([sorted(t[3], key=lambda p: -P_[p])[0] for t in test]); keep = np.isin(y, [p for p, c in P_.most_common(8)])
    return float(cross_val_score(LogisticRegression(max_iter=2000), X[keep], y[keep], cv=5).mean()), float(np.mean(y[keep] == collections.Counter(y[keep]).most_common(1)[0][0]))
maj = max(np.mean([p in t[3] for t in test]) for p in P_)
from sklearn.feature_extraction.text import TfidfVectorizer
bow = TfidfVectorizer().fit_transform([' '.join(t[4]) or '_' for t in test]).toarray(); bow = bow / (np.linalg.norm(bow, axis=1, keepdims=True) + 1e-9)
enc0 = Enc(len(vocab) + 4)
r_rand, r_bow = knn(codes(enc0, test)), knn(bow)
print(f"NEAREST OTHER FRAME SHARES A GOLD RELATION TYPE: majority type {maj:.2f} | untrained encoder {r_rand:.2f} | bag of connector words {r_bow:.2f}", flush=True)
print(f"   linear probe (8 commonest types, 5-fold): untrained {linprobe(codes(enc0, test))[0]:.2f} | connector words {linprobe(bow)[0]:.2f} | majority {linprobe(bow)[1]:.2f}", flush=True)

import zlib as _z
spw = {}
for k, i in spid.items(): spw[i] = [_z.crc32(w.lower().encode()) % 65536 for w in k.split()][:4]
SW = torch.zeros(len(spid), 4, dtype=torch.long); SM = torch.zeros(len(spid), 4)
for i, ws in spw.items(): SW[i, :len(ws)] = torch.tensor(ws); SM[i, :len(ws)] = 1
class SpanEnc(nn.Module):
    def __init__(s): super().__init__(); s.e = nn.Embedding(65536, 64); s.l = nn.Linear(64, 32)
    def forward(s, ix): return s.l((s.e(SW[ix]) * SM[ix].unsqueeze(-1)).sum(1) / SM[ix].sum(1, keepdim=True))
enc = Enc(len(vocab) + 4); senc = SpanEnc(); opt = torch.optim.Adam(list(enc.parameters()) + list(senc.parameters()), lr=2e-3)
for ep in range(EPOCHS):
    idx = np.random.permutation(len(train)); ls = []
    for k in range(0, len(idx) - 255, 256):
        rows = [train[i] for i in idx[k:k + 256]]; F_, pa, pb = batch(rows)
        a = torch.tensor([r[3] for r in rows]); b = torch.tensor([r[4] for r in rows])
        P1, P2 = enc.ops(enc.code(F_, pa, pb))
        cand = torch.unique(torch.cat([a, b, torch.randint(len(spid), (1024,))])); zc_ = F.normalize(senc(cand), dim=-1)
        pos = lambda t: (cand.unsqueeze(0) == t.unsqueeze(1)).float().argmax(1); za = zc_[pos(a)]; zb = zc_[pos(b)]
        l1 = F.cross_entropy(F.normalize(torch.einsum('bd,bde->be', za, P1), dim=-1) @ zc_.t() / 0.1, pos(b))
        l2 = F.cross_entropy(F.normalize(torch.einsum('bd,bde->be', zb, P2), dim=-1) @ zc_.t() / 0.1, pos(a))
        zc = zc_ - zc_.mean(0); cov = zc.t() @ zc / len(cand)
        loss = l1 + l2 + 0.1 * ((cov * 32 - torch.eye(32)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); ls.append(float(l1 + l2))
    X = codes(enc, test)
    print(f"epoch {ep + 1}: loss {np.mean(ls[:50]):.3f} -> {np.mean(ls[-50:]):.3f} | TRAINED FRAME CODE: nearest frame same type {knn(X):.2f} | linear probe {linprobe(X)[0]:.2f}", flush=True)
X = codes(enc, test)
both = np.concatenate([X, bow], 1); both /= np.linalg.norm(both, axis=1, keepdims=True)
print(f"FINAL: majority {maj:.2f} | untrained {r_rand:.2f} | connector words {r_bow:.2f} | TRAINED FRAME CODE {knn(X):.2f} | trained + connector words {knn(both):.2f}", flush=True)
ex = collections.defaultdict(list)
S = X @ X.T; np.fill_diagonal(S, -9)
for i in np.random.RandomState(1).permutation(len(test))[:6]:
    j = int(S[i].argmax()); print(f"   e.g. [A] {' '.join(test[i][4][:8])} [B] {sorted(test[i][3])} -> nearest: [A] {' '.join(test[j][4][:8])} [B] {sorted(test[j][3])}")
