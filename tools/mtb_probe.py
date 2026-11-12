"""MATCHING THE BLANKS ON THE FAST CHANNEL (12 Nov 2026; tools/mtb_probe.py) - after Baldini Soares et al. 2019: frames that link the SAME
filler pair in different sentences state (mostly) the same relation, so a frame encoder trained to bring them together, with the
fillers blanked, should group frames by relation type with no labels. The fast channel now supplies the filler spans the original method
took from annotation. Encoder: word embeddings -> BiGRU over '[A] frame words [B]' -> mean -> 128-d; InfoNCE over pairs of frames
sharing a filler pair (local Wikipedia + the benchmark, adjacent and document-centre records), in-batch negatives.
Test as T2b in tools/reader_tests.py: benchmark gold-pair records, is the nearest other record of the same gold relation type? Against
the frame words (tf-idf, 0.44 in T2b) and the untrained encoder. Usage: python3 tools/mtb_probe.py [epochs]"""
import os
import sys
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'reader_tests.py')).read()
exec(compile(_src[:_src.index('# ---------------- T1 REDUNDANCY')], 'reader_tests', 'exec'))
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
torch.manual_seed(0); torch.set_num_threads(1)
P = lambda *a: print(*a, flush=True)
REC = [(a, f, b, pv) for a, f, b, pv, k in BR + WR if k in (0, 1)]
grp = collections.defaultdict(list)
for a, f, b, pv in REC: grp[(min(a, b), max(a, b))].append((f, pv))
pos = []
for key, L in grp.items():
    seen = {}
    for f, pv in L: seen.setdefault(pv, f)
    fs = list(seen.values())
    for i in range(len(fs)):
        for j in range(i + 1, min(len(fs), i + 4)): pos.append((fs[i], fs[j]))
wc = collections.Counter(w for f, _ in [(f, 0) for a, f, b, pv in REC] for w in f.split())
vocab = {w: i + 2 for i, (w, _) in enumerate(wc.most_common(20000))}
enc_ids = lambda f: [vocab.get(w, 1) for w in ('[A] ' + f + ' [B]').split()][:24]
vocab['[A]'] = len(vocab) + 2; vocab['[B]'] = len(vocab) + 2
rel = {}
for h, t, fl, ps in G['rel']: rel.setdefault((min(h, t), max(h, t)), set()).update(ps)
ex = []
for a, f, b, pv, k in BR:
    if k not in (0, 1): continue
    g = gp(a, b)
    if g: ex.append((f, set().union(*[rel[x] for x in g])))
P(f"positive pairs (frames sharing a filler pair, different sentences): {len(pos)} | test records {len(ex)}")


class Enc(nn.Module):
    def __init__(s, V):
        super().__init__(); s.e = nn.Embedding(V + 4, 96, padding_idx=0); s.g = nn.GRU(96, 96, batch_first=True, bidirectional=True); s.o = nn.Linear(192, 128)
    def forward(s, X):
        h, _ = s.g(s.e(X)); m = (X > 0).float().unsqueeze(-1); return F.normalize(s.o((h * m).sum(1) / m.sum(1).clamp(min=1)), dim=-1)
def pad(L):
    X = torch.zeros(len(L), max(len(x) for x in L), dtype=torch.long)
    for i, x in enumerate(L): X[i, :len(x)] = torch.tensor(x)
    return X
def codes(enc):
    with torch.no_grad(): return torch.cat([enc(pad([enc_ids(e[0]) for e in ex[i:i + 512]])) for i in range(0, len(ex), 512)]).numpy()
def knn(X):
    S = X @ X.T; np.fill_diagonal(S, -9); nn_ = S.argmax(1); return float(np.mean([bool(ex[i][1] & ex[j][1]) for i, j in enumerate(nn_)]))
from sklearn.feature_extraction.text import TfidfVectorizer
bow = TfidfVectorizer(token_pattern=r'[^ ]+').fit_transform([e[0] for e in ex]).toarray(); bow /= np.linalg.norm(bow, axis=1, keepdims=True) + 1e-9
enc = Enc(len(vocab) + 2)
P(f"BEFORE: frame words {knn(bow):.2f} | untrained encoder {knn(codes(enc)):.2f}")
opt = torch.optim.Adam(enc.parameters(), lr=2e-3); rs = np.random.RandomState(0)
for ep in range(EPOCHS):
    idx = rs.permutation(len(pos)); ls = []
    for k in range(0, len(idx) - 255, 256):
        bt = [pos[i] for i in idx[k:k + 256]]; z1 = enc(pad([enc_ids(x) for x, _ in bt])); z2 = enc(pad([enc_ids(y) for _, y in bt]))
        lg = z1 @ z2.t() / 0.1; y = torch.arange(len(bt)); loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.t(), y)) / 2
        opt.zero_grad(); loss.backward(); opt.step(); ls.append(float(loss))
    X = codes(enc); both = np.concatenate([X, bow], 1); both /= np.linalg.norm(both, axis=1, keepdims=True)
    P(f"epoch {ep + 1}: loss {np.mean(ls[:30]):.2f} -> {np.mean(ls[-30:]):.2f} | MTB frame code {knn(X):.2f} | MTB + frame words {knn(both):.2f}")
P("DONE")
