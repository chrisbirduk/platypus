"""TOY TRANSFORMER TESTS (12 Nov 2026; tools/toy_compose_rows.py) - three ideas from recent work, on a small causal transformer trained
from scratch on synthetic facts (CPU):
  (2) TWO-HOP CURSE (Balesni et al. 2024): facts learned in SEPARATE documents are not composed latently; facts in the SAME document can
      be. 400 chains A -r1-> B -r2-> C; 300 train chains also teach the two-hop question 'Q A r1 r2 ?' -> C; the 100 held-out chains are
      never asked during training. Their premises are shown SEPARATELY ('A r1 B .' and 'B r2 C .' as two sequences) or CO-LOCATED (also
      'A r1 B . B r2 C .' in one sequence). Test: 'Q A r1 r2 ?' -> C on the held-out chains.
  (6) WORDING DIVERSITY (EntiGraph, 2025): one-hop facts 'A r B .'; 70% of them also taught as questions '<prefix> A r ?' -> B with K
      training prefixes (2, 7 or 20 random word strings); test the taught facts in UNSEEN prefixes (new strings of the same words).
  (3) VIEW ALIGNMENT (LLM-JEPA, 2025): + a cosine term aligning the question's last state with the declaration's state after 'A r' (the
      position that predicts B); same test.
Usage: python3 tools/toy_compose_rows.py [seed] [steps]"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
torch.set_num_threads(1)
P = lambda *a: print(*a, flush=True)
NE, NR, NW = 1200, 6, 30
V = NE + NR + NW + 6; PAD, DOT, Q, QM = 0, 1, 2, 3                         # tokens: 0 pad, 1 '.', 2 'Q', 3 '?', then entities, relations, words
ENT = lambda i: 6 + i; REL = lambda i: 6 + NE + i; WRD = lambda i: 6 + NE + NR + i


class LM(nn.Module):
    def __init__(s, d=128, L=2, T=16):
        super().__init__(); s.e = nn.Embedding(V, d); s.p = nn.Embedding(T, d)
        s.b = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True), L); s.o = nn.Linear(d, V)
    def forward(s, X):
        T = X.shape[1]; m = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        h = s.b(s.e(X) + s.p(torch.arange(T)), mask=m); return h, s.o(h)


def pad(L):
    X = torch.zeros(len(L), max(len(x) for x in L), dtype=torch.long)
    for i, x in enumerate(L): X[i, :len(x)] = torch.tensor(x)
    return X


def train(seqs, steps, sd, align=None, aw=0.5):
    torch.manual_seed(sd); rr = np.random.RandomState(sd); m = LM(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for st in range(steps):
        b = [seqs[i] for i in rr.randint(len(seqs), size=96)]; X = pad(b); h, lg = m(X)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), ignore_index=PAD)
        if align:                                                              # VIEW ALIGNMENT: question's last state ~ declaration's state after 'A r'
            ab = [align[i] for i in rr.randint(len(align), size=48)]; Xq = pad([q for q, d in ab]); Xd = pad([d for q, d in ab])
            hq, _ = m(Xq); hd, _ = m(Xd); lq = torch.tensor([len(q) - 1 for q, d in ab])
            loss = loss + aw * (1 - F.cosine_similarity(hq[torch.arange(len(ab)), lq], hd[:, 1], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def answer_acc(m, prompts, golds):
    with torch.no_grad():
        X = pad(prompts); _, lg = m(X); last = torch.tensor([len(p) - 1 for p in prompts])
        s = lg[torch.arange(len(prompts)), last][:, 6:6 + NE]; return float((s.argmax(-1) + 6 == torch.tensor(golds)).float().mean())


rs = np.random.RandomState(seed)
# ---------------- (2) TWO-HOP CURSE ----------------
perm = rs.permutation(NE); ch = [(ENT(perm[3 * i]), ENT(perm[3 * i + 1]), ENT(perm[3 * i + 2]), REL(rs.randint(NR)), REL(rs.randint(NR))) for i in range(400)]
trc, tec = ch[:300], ch[300:]
for cond in ('SEPARATE', 'CO-LOCATED'):
    seqs = []
    for A, B, C, r1, r2 in ch: seqs += [[A, r1, B, DOT], [B, r2, C, DOT]]
    for A, B, C, r1, r2 in trc: seqs += [[Q, A, r1, r2, QM, C]] * 2 + [[A, r1, B, DOT, B, r2, C, DOT]]
    if cond == 'CO-LOCATED':
        for A, B, C, r1, r2 in tec: seqs += [[A, r1, B, DOT, B, r2, C, DOT]]
    m = train(seqs, STEPS, seed)
    one = answer_acc(m, [[A, r1] for A, B, C, r1, r2 in tec], [B for A, B, C, r1, r2 in tec])
    tr = answer_acc(m, [[Q, A, r1, r2, QM] for A, B, C, r1, r2 in trc], [C for A, B, C, r1, r2 in trc])
    te = answer_acc(m, [[Q, A, r1, r2, QM] for A, B, C, r1, r2 in tec], [C for A, B, C, r1, r2 in tec])
    P(f"(2) {cond:10s}: held-out chains' first hop {one:.2f} | two-hop question, trained chains {tr:.2f} | HELD-OUT CHAINS (never asked) {te:.2f}  (chance {1 / NE:.4f})")

# ---------------- (6) WORDING DIVERSITY and (3) VIEW ALIGNMENT ----------------
facts = [(ENT(rs.randint(NE)), REL(rs.randint(NR)), ENT(rs.randint(NE))) for _ in range(600)]
taught = facts[:420]; untaught = facts[420:]
mk = lambda: [WRD(w) for w in rs.choice(NW, rs.randint(1, 4), replace=False)]
test_pref = [mk() for _ in range(10)]
for K in (2, 7, 20):
    pref = [mk() for _ in range(K)]
    for al in (False, True):
        seqs = [[A, r, B, DOT] for A, r, B in facts]; align = []
        for A, r, B in taught:
            for p in pref: seqs.append(p + [A, r, QM, B]); align.append((p + [A, r, QM], [A, r, B, DOT]))
        m = train(seqs, STEPS, seed + 1, align if al else None)
        def acc(F_, prefs):
            pr, gd = [], []
            for A, r, B in F_:
                for p in prefs: pr.append(p + [A, r, QM]); gd.append(B)
            return answer_acc(m, pr, gd)
        P(f"(6/3) K {K:2d} training prefixes, view alignment {'ON ' if al else 'off'}: taught facts, training prefixes {acc(taught, pref):.2f} | "
          f"UNSEEN PREFIXES {acc(taught, test_pref):.2f} | never-taught facts (declared only), unseen prefixes {acc(untaught, test_pref):.2f}")
P("DONE")
