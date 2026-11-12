"""TOY ALIGNMENT TESTS 2 (12 Nov 2026; tools/toy_align2.py) - how to improve VIEW ALIGNMENT (question state at the answer pulled toward the
state where the fact's declaration 'A r B .' predicts B; the declaration a stop-gradient target, as in cg27a_align). Small causal
transformer from scratch (as tools/toy_compose_rows.py: 2 layers, 128 wide), facts 'A r B .', questions '<prefix> A r ?' -> B with 7
training prefixes, tested in 10 UNSEEN prefixes:
  A  READINGS       never-taught facts' declarations read 2 / 8 / 32 / 128 times (4 groups, one run): is the fact in the weights (the model
                    completes 'A r' -> B) and does alignment reach it - is the real bottleneck storage or routing?
  B  LABEL-FREE     facts ALIGNED but never asked (no answer label anywhere: the alignment needs only the question and the declaration's
                    state) against a never-aligned control - can alignment cover every fact read, not just taught ones?
  C  OBJECTIVE      1,500 facts: cosine vs InfoNCE (other facts' declarations as negatives) vs a predictor head (LLM-JEPA) + cosine
  D  WHERE          align at the answer position (as built) vs the SUBJECT's position (subject enrichment, ROME / Geva) vs both vs the
                    middle layer's state
Usage: python3 tools/toy_align2.py [tests, e.g. AB] [seed] [steps]"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
TESTS = sys.argv[1] if len(sys.argv) > 1 else 'ABCD'
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
torch.set_num_threads(1)
P = lambda *a: print(*a, flush=True)
NE, NR, NW = 1200, 6, 30
V = NE + NR + NW + 6; PAD, DOT, Q, QM = 0, 1, 2, 3
ENT = lambda i: 6 + i; REL = lambda i: 6 + NE + i; WRD = lambda i: 6 + NE + NR + i


class LM(nn.Module):
    def __init__(s, d=128, L=2, T=16):
        super().__init__(); s.e = nn.Embedding(V, d); s.p = nn.Embedding(T, d)
        s.ls = nn.ModuleList([nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True) for _ in range(L)])
        s.o = nn.Linear(d, V); s.pred = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
    def forward(s, X):
        T = X.shape[1]; m = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); h = s.e(X) + s.p(torch.arange(T)); hs = []
        for l in s.ls: h = l(h, src_mask=m); hs.append(h)
        return hs, s.o(h)


def pad(L):
    X = torch.zeros(len(L), max(len(x) for x in L), dtype=torch.long)
    for i, x in enumerate(L): X[i, :len(x)] = torch.tensor(x)
    return X


def train(seqs, wts, steps, sd, align=None, mode='cos', where='ans', aw=0.5):
    """seqs sampled with probabilities wts; align = [(question, declaration)]: question's last state ~ declaration's state after 'A r'"""
    torch.manual_seed(sd); rr = np.random.RandomState(sd); m = LM(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    p = np.asarray(wts, float); p /= p.sum()
    for st in range(steps):
        b = [seqs[i] for i in rr.choice(len(seqs), size=96, p=p)]; X = pad(b); hs, lg = m(X)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), ignore_index=PAD)
        if align:
            ab = [align[i] for i in rr.randint(len(align), size=48)]; Xq = pad([q for q, d in ab]); Xd = pad([d for q, d in ab])
            hq, _ = m(Xq)
            with torch.no_grad(): hd, _ = m(Xd)
            lq = torch.tensor([len(q) - 1 for q, d in ab]); sq = torch.tensor([len(q) - 3 for q, d in ab]); ar = torch.arange(len(ab))
            lay = 0 if where == 'mid' else -1
            pairs = []
            if where in ('ans', 'both', 'mid'): pairs.append((hq[lay][ar, lq], hd[lay][:, 1]))
            if where in ('subj', 'both'): pairs.append((hq[lay][ar, sq], hd[lay][:, 0]))
            for a, t in pairs:
                if mode == 'pred': a = m.pred(a)
                if mode == 'nce':
                    lg2 = F.normalize(a, dim=-1) @ F.normalize(t, dim=-1).t() / 0.1; loss = loss + aw * F.cross_entropy(lg2, ar)
                else: loss = loss + aw * (1 - F.cosine_similarity(a, t, dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def acc(m, prompts, golds):
    if not prompts: return float('nan')
    with torch.no_grad():
        X = pad(prompts); _, lg = m(X); last = torch.tensor([len(p) - 1 for p in prompts])
        s = lg[torch.arange(len(prompts)), last][:, 6:6 + NE]; return float((s.argmax(-1) + 6 == torch.tensor(golds)).float().mean())


rs = np.random.RandomState(seed)
mk = lambda: [WRD(w) for w in rs.choice(NW, rs.randint(1, 4), replace=False)]
pref = [mk() for _ in range(7)]; test_pref = [mk() for _ in range(10)]
qa = lambda F_, prefs: ([p + [A, r, QM] for A, r, B in F_ for p in prefs], [B for A, r, B in F_ for p in prefs])
decl_acc = lambda m, F_: acc(m, [[A, r] for A, r, B in F_], [B for A, r, B in F_])
NS = STEPS * 96


def world(n):
    return [(ENT(rs.randint(NE)), REL(rs.randint(NR)), ENT(rs.randint(NE))) for _ in range(n)]


if 'A' in TESTS:
    facts = world(600); taught = facts[:420]; un = facts[420:]; grp = [un[i::4] for i in range(4)]; reads = (2, 8, 32, 128)
    seqs, wts = [], []
    for A, r, B in taught: seqs.append([A, r, B, DOT]); wts.append(1.0)
    for A, r, B in taught:
        for p in pref: seqs.append(p + [A, r, QM, B]); wts.append(1.0)
    base = sum(wts)                                                        # ordinary sequences ~ NS / base reads each
    for g, n in zip(grp, reads):
        for A, r, B in g: seqs.append([A, r, B, DOT]); wts.append(n * base / NS)
    al = [(p + [A, r, QM], [A, r, B, DOT]) for A, r, B in taught for p in pref]
    P(f"A READINGS: taught facts read ~{NS / sum(wts):.0f} times each; never-taught groups read {reads} times")
    for on in (False, True):
        m = train(seqs, wts, STEPS, seed + 1, al if on else None)
        P(f"   alignment {'ON ' if on else 'off'}: taught, unseen prefixes {acc(m, *qa(taught, test_pref)):.2f} | NEVER TAUGHT by readings "
          + " | ".join(f"{n}: completes 'A r' {decl_acc(m, g):.2f}, question {acc(m, *qa(g, test_pref)):.2f}" for g, n in zip(grp, reads)))

if 'B' in TESTS:
    facts = world(600); taught = facts[:300]; alonly = facts[300:450]; ctl = facts[450:]
    seqs = [[A, r, B, DOT] for A, r, B in facts] + [p + [A, r, QM, B] for A, r, B in taught for p in pref]
    al_t = [(p + [A, r, QM], [A, r, B, DOT]) for A, r, B in taught for p in pref]; al_l = [(p + [A, r, QM], [A, r, B, DOT]) for A, r, B in alonly for p in pref]
    P("B LABEL-FREE: 300 facts taught (questions with answers), 150 ALIGNED ONLY (never asked), 150 control (read only)")
    for tag, al in (('no alignment', None), ('taught facts aligned', al_t), ('taught + ALIGNED-ONLY facts aligned', al_t + al_l)):
        m = train(seqs, [1.0] * len(seqs), STEPS, seed + 2, al)
        P(f"   {tag}: taught {acc(m, *qa(taught, test_pref)):.2f} | ALIGNED ONLY, never asked {acc(m, *qa(alonly, test_pref)):.2f} | control {acc(m, *qa(ctl, test_pref)):.2f}")

if 'C' in TESTS:
    facts = world(1500); taught = facts[:1050]; un = facts[1050:]
    seqs = [[A, r, B, DOT] for A, r, B in facts] + [p + [A, r, QM, B] for A, r, B in taught for p in pref]
    al = [(p + [A, r, QM], [A, r, B, DOT]) for A, r, B in taught for p in pref]
    P("C OBJECTIVE: 1,500 facts (1,050 taught)")
    for tag, a_, md in (('no alignment', None, 'cos'), ('cosine', al, 'cos'), ('InfoNCE (other declarations negatives)', al, 'nce'), ('predictor + cosine', al, 'pred')):
        m = train(seqs, [1.0] * len(seqs), STEPS, seed + 3, a_, mode=md)
        P(f"   {tag}: taught {acc(m, *qa(taught, test_pref)):.2f} | never taught {acc(m, *qa(un, test_pref)):.2f} | completes 'A r' (never taught) {decl_acc(m, un):.2f}")

if 'D' in TESTS:
    facts = world(600); taught = facts[:420]; un = facts[420:]
    seqs = [[A, r, B, DOT] for A, r, B in facts] + [p + [A, r, QM, B] for A, r, B in taught for p in pref]
    al = [(p + [A, r, QM], [A, r, B, DOT]) for A, r, B in taught for p in pref]
    P("D WHERE: align at the answer position (as built), the subject's position, both, or the middle layer")
    for wh in ('ans', 'subj', 'both', 'mid'):
        m = train(seqs, [1.0] * len(seqs), STEPS, seed + 4, al, where=wh)
        P(f"   {wh}: taught {acc(m, *qa(taught, test_pref)):.2f} | never taught {acc(m, *qa(un, test_pref)):.2f}")
P("DONE")
