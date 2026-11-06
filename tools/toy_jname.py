"""TOY: CAN A TEXT JEPA FIND THE NAMES ITSELF? (no effect on the package)
Idea: a JEPA predicts REPRESENTATIONS, so it can learn to drop what context cannot predict. A name is the extreme case: context predicts
THAT a name goes there, not WHICH. So at a name, byte surprise (a byte language model) should be HIGH while JEPA latent surprise is LOW;
at relation and function words both are low or both high. Names = where the two disagree. Second test: SWAP - replace a word by a random
pseudo-word; if the JEPA latent of the REST of the text barely moves, the word is a variable (a name), not structure.
Data: byte-level documents of 2-3 sentences about invented people (names drawn from syllables; training and test names disjoint), 8
asymmetric relations x 4 wordings each (wording 4 held out from training), names reused across sentences (first and repeat mentions),
plus filler sentences of ordinary words.
Models: a causal byte language model (byte surprise); two text JEPAs, both a bidirectional byte transformer encoder + predictor trained to
predict the latents of masked words (whole-word masking, 15% of words + random byte spans):
  EMA     target = a moving-average copy of the encoder (V-JEPA / data2vec style, momentum 0.996), smooth-L1 on layer-normed targets
  LEJEPA  one encoder, targets with stop-gradient, SIGReg on the latents instead of an EMA (the project's usual recipe)
Readouts on TEST names, per word (gold labels used only for scoring):
  1 DETECTION AUC, name vs other words: byte surprise (high = name), JEPA surprise (reported as AUC of LOW surprise = name), and the GAP
    z(byte surprise) - z(JEPA surprise); for first mentions, repeat mentions, and the held-out wording.
  2 SWAP AUC: 1 - cosine of the rest-of-text latent after the word is replaced by a random pseudo-word (low = name).
  3 WHAT THE LATENT KEEPS: the same test name in two different documents - is its latent nearest to its own other mention among 50 test
    names (identity kept) - for the JEPA target latents and the byte language model's hidden states.
usage: python3 toy_jname.py [seed] [steps]
env: JN_KINDS (ema,lejepa), JN_CKPTS (e.g. 1000,3000,6000: compact readouts during training), JN_NOREP=1 (training documents whose names
never repeat: tests whether keeping a name's identity is rewarded by its repeat mentions)"""
import sys, time, math, re
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
torch.set_num_threads(1); rs = np.random.RandomState(SEED); torch.manual_seed(SEED); t0 = time.time()
# ---------------------------------------------------------------- data
SYL = ['ba', 'ko', 'mi', 'ru', 'te', 'lo', 'fa', 'zen', 'dor', 'vi', 'sa', 'nu', 'kel', 'ta', 'pex', 'li', 'gor', 'bri', 'mo', 'an',
       'ri', 'vo', 'du', 'sel', 'ka', 'ne', 'tor', 'mi', 'lu', 'qua']
def mkname(r): return ''.join(r.choice(SYL, r.randint(2, 4))).capitalize()
ALLN = sorted({mkname(rs) for _ in range(6000)}); rs.shuffle(ALLN)
TRN_N, TST_N = ALLN[:2500], ALLN[2500:3100]
REL = [["{A} reports to {B}.", "{B} is the manager of {A}.", "{A} answers to {B}.", "The manager of {A} is {B}."],
       ["{A} trained under {B}.", "{B} was the teacher of {A}.", "{A} was taught by {B}.", "{B} taught {A}."],
       ["{A} lives north of {B}.", "{B} lives south of {A}.", "The home of {A} is north of {B}.", "North of {B} lives {A}."],
       ["{A} sold a horse to {B}.", "{B} bought a horse from {A}.", "{A} sold {B} a horse.", "A horse was sold by {A} to {B}."],
       ["{A} wrote to {B}.", "{B} got a letter from {A}.", "{A} sent {B} a letter.", "A letter from {A} reached {B}."],
       ["{A} is the parent of {B}.", "{B} is the child of {A}.", "{A} raised {B}.", "{B} was raised by {A}."],
       ["{A} beat {B} at chess.", "{B} lost to {A} at chess.", "{A} won against {B} at chess.", "At chess, {B} was beaten by {A}."],
       ["{A} owes money to {B}.", "{B} lent money to {A}.", "{A} is in debt to {B}.", "{B} is owed money by {A}."]]
FILL = ["It was a cold day.", "The market opened late.", "Rain fell on the town.", "The road was long and dry.", "Nobody saw the boat.",
        "The bell rang twice.", "Bread was cheap that year.", "The river was high."]
TRAIN_T, HELD_T = [0, 1, 2], [3]
L = 128; PAD, MASK = 256, 257; V = 258
import os
NOREP = os.environ.get('JN_NOREP', '0') == '1'                             # training documents whose names never repeat (control)
def doc(names, tpl, r=rs, norep=False):
    """2-3 sentences: (A,B), (B,C) or (A,C), maybe a third about two of them, maybe a filler; returns text and gold word labels.
    norep: every sentence about two fresh names (no name repeats inside the document)"""
    A, B, C = [names[i] for i in r.choice(len(names), 3, replace=False)]
    pairs = [(A, B), [(B, C), (A, C)][r.randint(2)]]
    if r.rand() < 0.5: pairs.append([(A, B), (C, A), (B, A)][r.randint(3)])
    if norep:
        pairs = [tuple(names[i] for i in r.choice(len(names), 2, replace=False)) for _ in pairs]
        return ' '.join(REL[r.randint(len(REL))][tpl[r.randint(len(tpl))]].format(A=a, B=b) for a, b in pairs), {x for pr in pairs for x in pr}
    ss = []
    for a, b in pairs: ss.append(REL[r.randint(len(REL))][tpl[r.randint(len(tpl))]].format(A=a, B=b))
    if r.rand() < 0.5: ss.insert(r.randint(len(ss) + 1), FILL[r.randint(len(FILL))])
    return ' '.join(ss), {A, B, C}
def enc(t): b = list(t.encode())[:L]; return b + [PAD] * (L - len(b))
def batch(n, names=TRN_N, tpl=TRAIN_T): return torch.tensor([enc(doc(names, tpl, norep=NOREP)[0]) for _ in range(n)])
WORDS = sorted({w for R in REL for t in R for w in re.findall(r'[A-Za-z]+', t.replace('{A}', '').replace('{B}', ''))} |
               {w for f in FILL for w in re.findall(r'[A-Za-z]+', f)})
def spans(t):
    """word spans (start, end, word) of a text"""
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r'[A-Za-z]+', t) if m.end() <= L]
# ---------------------------------------------------------------- models
class Blk(nn.Module):
    def __init__(s, d, h, causal):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d); s.causal = causal
    def forward(s, x, kpm):
        T = x.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1) if s.causal else None
        h = s.n1(x); x = x + s.a(h, h, h, attn_mask=am, key_padding_mask=None if s.causal else kpm, need_weights=False)[0]
        return x + s.m(s.n2(x))
class Tr(nn.Module):
    def __init__(s, d=128, h=4, nl=4, causal=False, vin=V):
        super().__init__(); s.emb = nn.Embedding(vin, d); s.pos = nn.Parameter(torch.randn(L, d) * 0.02); s.b = nn.ModuleList(Blk(d, h, causal) for _ in range(nl)); s.n = nn.LayerNorm(d)
    def forward(s, X, kpm=None, x_in=None):
        x = (s.emb(X) if x_in is None else x_in) + s.pos[:X.shape[1]]
        for b in s.b: x = b(x, kpm)
        return s.n(x)
class LM(nn.Module):
    def __init__(s): super().__init__(); s.t = Tr(causal=True, nl=3); s.o = nn.Linear(128, V)
    def forward(s, X): return s.o(s.t(X))
def train_lm(steps):
    m = LM(); op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01); ls = []
    for st in range(steps):
        X = batch(32); Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1)   # PAD as the start symbol
        lg = m(Xi); msk = (X != PAD).float(); l = (F.cross_entropy(lg.reshape(-1, V), X.reshape(-1), reduction='none') * msk.reshape(-1)).sum() / msk.sum()
        op.zero_grad(); l.backward(); op.step(); ls.append(float(l))
    print(f"  byte LM: {np.mean(ls[:20]) / math.log(2):.2f} -> {np.mean(ls[-100:]) / math.log(2):.2f} bits per byte  ({time.time()-t0:.0f}s)", flush=True); m.eval(); return m
def sigreg(z, ns=128, nk=17, tm=5.0):
    U = F.normalize(torch.randn(z.shape[1], ns), dim=0); x = z @ U; t = torch.linspace(0, tm, (nk + 1) // 2); tx = x[:, :, None] * t
    err = ((torch.cos(tx).mean(0) - torch.exp(-t ** 2 / 2)) ** 2 + torch.sin(tx).mean(0) ** 2) * torch.exp(-t ** 2 / 2)
    return z.shape[0] * 2 * torch.trapezoid(err, t, dim=-1).mean()
def mask_words(X):
    """whole-word masking of ~15% of words + ~5% random byte spans; returns the masked input and the mask"""
    M = torch.zeros_like(X, dtype=torch.bool)
    for i in range(X.shape[0]):
        t = bytes([b for b in X[i].tolist() if b < 256]).decode('latin-1')
        for a, b, w in spans(t):
            if rs.rand() < 0.15: M[i, a:b] = True
        for _ in range(rs.randint(0, 3)):
            a = rs.randint(0, max(1, len(t) - 4)); M[i, a:a + rs.randint(1, 5)] = True
    M &= (X != PAD); return X.masked_fill(M, MASK), M
class JEPA(nn.Module):
    def __init__(s, kind):
        super().__init__(); s.kind = kind; s.enc = Tr(nl=4); s.pred = Tr(nl=2, vin=1); s.tgt = None
        if kind == 'ema':
            s.tgt = Tr(nl=4); s.tgt.load_state_dict(s.enc.state_dict())
            for p in s.tgt.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def target(s, X):
        """per-position target latents of the full text (EMA copy, or the encoder itself with stop-gradient), layer-normed"""
        z = (s.tgt if s.kind == 'ema' else s.enc)(X, kpm=(X == PAD)); return F.layer_norm(z, z.shape[-1:])
    def predict(s, Xm):
        h = s.enc(Xm, kpm=(Xm == PAD)); return s.pred(Xm, kpm=(Xm == PAD), x_in=h)
def train_jepa(kind, steps, ck=None, at=()):
    m = JEPA(kind); P_ = [p for p in m.parameters() if p.requires_grad]; op = torch.optim.AdamW(P_, lr=5e-4, weight_decay=0.01); ls = []
    for st in range(steps):
        if ck is not None and st in at: m.eval(); ck(m, st, ls); m.train()
        X = batch(32); Xm, M = mask_words(X); p = m.predict(Xm)
        if kind == 'ema': tg = m.target(X)
        else:
            z = m.enc(X, kpm=(X == PAD)); tg = F.layer_norm(z, z.shape[-1:]).detach()
        l = F.smooth_l1_loss(p[M], tg[M])
        if kind == 'lejepa':
            zz = z[X != PAD]; zz = zz[torch.randperm(zz.shape[0])[:512]]; l = l + 0.05 * sigreg(zz) / zz.shape[0]
        op.zero_grad(); l.backward(); op.step(); ls.append(float(l))
        if kind == 'ema':
            with torch.no_grad():
                for a, b in zip(m.tgt.parameters(), m.enc.parameters()): a.mul_(0.996).add_(b, alpha=0.004)
    with torch.no_grad(): sp = float(m.target(batch(64)).reshape(-1, 128).std(0).mean())
    print(f"  JEPA {kind}: loss {np.mean(ls[:20]):.3f} -> {np.mean(ls[-100:]):.3f} | target spread {sp:.3f} (0 = collapsed)  ({time.time()-t0:.0f}s)", flush=True); m.eval(); return m
# ---------------------------------------------------------------- readouts
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
@torch.no_grad()
def word_scores(lm, J, t):
    """per word of text t: (label, byte surprise, JEPA surprise, swap sensitivity)"""
    X = torch.tensor([enc(t)]); sp = spans(t); names_seen = set(); out = []
    Xi = torch.cat([torch.full((1, 1), PAD), X[:, :-1]], 1); nll = F.cross_entropy(lm(Xi)[0], X[0], reduction='none')
    tg = J.target(X)[0]; Xm = X.repeat(len(sp), 1)
    for k, (a, b, w) in enumerate(sp): Xm[k, a:b] = MASK
    pr = J.predict(Xm); keep = (X[0] != PAD)
    zr0 = tg
    Xs = []; rng = np.random.RandomState(hash(t) % 2 ** 31)
    for a, b, w in sp:                                                     # swap: the word replaced by a random pseudo-word
        pw = mkname(rng); pw = pw if w[0].isupper() else pw.lower(); Xs.append(enc(t[:a] + pw + t[b:]))
    zs = J.target(torch.tensor(Xs))
    for k, (a, b, w) in enumerate(sp):
        lab = ('name_rep' if w in names_seen else 'name_first') if w in NAMESET else ('func' if w.lower() in FUNC else 'content')
        if w in NAMESET: names_seen.add(w)
        js = float(((pr[k, a:b] - tg[a:b]) ** 2).mean())
        rest = keep.clone(); rest[a:] = False                              # the text BEFORE the word (positions unchanged by the swap)
        r2 = keep.clone(); r2[:b] = False; n_after = int(r2.sum())          # and after it, aligned from the end of the word
        za = zr0[rest].mean(0) if rest.any() else zr0[keep].mean(0)
        zb = zs[k][:a].mean(0) if a > 0 else zs[k][keep].mean(0)
        sw = 1 - float(F.cosine_similarity(za, zb, dim=0))
        if n_after > 0:
            L2 = len(Xs[k]) - Xs[k].count(PAD); bb = L2 - n_after
            sw += 1 - float(F.cosine_similarity(zr0[b:b + n_after].mean(0), zs[k][bb:bb + n_after].mean(0), dim=0))
        out.append((lab, float(nll[a:b].mean()), js, sw))
    return out
FUNC = {'is', 'the', 'of', 'to', 'a', 'at', 'by', 'from', 'was', 'and', 'on', 'that', 'it', 'in'}
NAMESET = set(TRN_N) | set(TST_N)
def detection(lm, J, tpl, n=250):
    R = []
    for _ in range(n): R += word_scores(lm, J, doc(TST_N, tpl)[0])
    lab = np.array([r[0] for r in R]); bs = np.array([r[1] for r in R]); js = np.array([r[2] for r in R]); sw = np.array([r[3] for r in R])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9); gap = z(bs) - z(js)
    oth = ~np.isin(lab, ['name_first', 'name_rep']); res = {}
    for nm, sel in (('first mentions', lab == 'name_first'), ('repeat mentions', lab == 'name_rep')):
        res[nm] = (auc(bs[sel], bs[oth]), auc(-js[sel], -js[oth]), auc(gap[sel], gap[oth]), auc(-sw[sel], -sw[oth]))
    means = {c: (bs[lab == c].mean(), js[lab == c].mean(), sw[lab == c].mean()) for c in ('name_first', 'name_rep', 'content', 'func')}
    return res, means
@torch.no_grad()
def identity(lm, J, n=50, reps=6):
    """the same test name in two documents: is its latent nearest to its own other mention (among n names)?"""
    nm = list(rs.choice(TST_N, n, replace=False)); H = {'jepa': [[], []], 'lm': [[], []]}
    for side in (0, 1):
        for x in nm:
            others = [y for y in TST_N if y != x]; t, _ = doc([x] + list(rs.choice(others, 2, replace=False)), TRAIN_T)
            while not re.search(r'\b' + x + r'\b', t) or len(t) > L: t, _ = doc([x] + list(rs.choice(others, 2, replace=False)), TRAIN_T)
            mm = re.search(r'\b' + x + r'\b', t); a, b = mm.start(), mm.end(); X = torch.tensor([enc(t)])
            H['jepa'][side].append(J.target(X)[0, a:b].mean(0))
            Xi = torch.cat([torch.full((1, 1), PAD), X[:, :-1]], 1); H['lm'][side].append(lm.t(Xi)[0, a + 1:b + 1].mean(0))
    out = {}
    W = []                                                                 # WITHIN a document: a repeat mention's latent nearest its own first mention?
    for _ in range(150):
        t, nms = doc(TST_N, TRAIN_T); X = torch.tensor([enc(t)]); z = F.normalize(J.target(X)[0], dim=-1); first = {}; reps = []
        for a, b, w in spans(t):
            if w in nms:
                if w in first: reps.append((w, z[a:b].mean(0)))
                else: first[w] = z[a:b].mean(0)
        for w, v in reps:
            if len(first) > 1: W.append(max(first, key=lambda k: float(F.cosine_similarity(v, first[k], dim=0))) == w)
    out['within'] = float(np.mean(W)) if W else float('nan')
    for k, (A, B) in H.items():
        A, B = F.normalize(torch.stack(A), dim=-1), F.normalize(torch.stack(B), dim=-1); out[k] = float(((A @ B.T).argmax(1) == torch.arange(n)).float().mean())
    return out
print(f"TOY TEXT JEPA NAME DETECTION, seed {SEED}, {STEPS} steps; {len(TRN_N)} training names, {len(TST_N)} test names (never seen)", flush=True)
lm = train_lm(STEPS)
CK = [int(x) for x in os.environ.get('JN_CKPTS', '').split(',') if x]      # checkpoints: a compact readout during training
def ckpt(J, st, ls):
    rT, _ = detection(lm, J, TRAIN_T, n=100); rH, _ = detection(lm, J, HELD_T, n=100); idn = identity(lm, J)
    f = lambda r, m: f"gap {r[m][2]:.2f} lowJ {r[m][1]:.2f} swap {r[m][3]:.2f}"
    print(f"  [CKPT {st:5d}{' NO-REPEAT training' if NOREP else ''}] loss {np.mean(ls[-100:]) if ls else float('nan'):.3f} | training wordings first {f(rT, 'first mentions')}, repeat {f(rT, 'repeat mentions')} | "
          f"HELD-OUT wording first {f(rH, 'first mentions')}, repeat {f(rH, 'repeat mentions')} | identity across documents {idn['jepa']:.2f} (byte LM {idn['lm']:.2f}), "
          f"within a document {idn['within']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
for kind in tuple(os.environ.get('JN_KINDS', 'ema,lejepa').split(',')):
    J = train_jepa(kind, STEPS, ckpt if CK else None, set(CK))
    for tn, tpl in (('training wordings', TRAIN_T), ('HELD-OUT wording', HELD_T)):
        res, means = detection(lm, J, tpl)
        for mn, (a1, a2, a3, a4) in res.items():
            print(f"  [{kind}] {tn:18s} {mn:16s} | AUC name vs other words: byte surprise {a1:.2f} | LOW JEPA surprise {a2:.2f} | GAP byte - JEPA {a3:.2f} | SWAP (rest of text barely moves) {a4:.2f}", flush=True)
        print(f"  [{kind}] {tn:18s} means (byte surprise nats / JEPA surprise / swap change): " + " | ".join(f"{c} {v[0]:.2f} / {v[1]:.3f} / {v[2]:.3f}" for c, v in means.items()), flush=True)
    idn = identity(lm, J)
    print(f"  [{kind}] IDENTITY KEPT? same test name in two documents found among 50 by its latent: JEPA target latent {idn['jepa']:.2f} | byte LM hidden state {idn['lm']:.2f} (chance 0.02) | WITHIN a document, a repeat mention nearest its own first mention {idn['within']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE {time.time()-t0:.0f}s")
