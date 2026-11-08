"""TOY: SEVERAL DOMAINS ACTIVE AT ONCE, GATED BY A CONTINUAL JEPA (no effect on the package).
Question: can the JEPA's activation decide which domain parts (slots) are active - possibly more than one - instead of one hard slot
per visit? Five local domains (mini-Pile: prose, wiki, arxiv, pubmed, law), visited in order, then revisited in a shuffled order.

MODEL     a small causal byte transformer (shared trunk) + one EXPERT per slot (a private residual MLP and its own head), like the package's
          private units and per-domain heads. A slot's prediction p_k = softmax(head_k(h + mlp_k(h))); a gate w mixes them: p = sum_k w_k p_k.
JEPA      a V-JEPA-style byte encoder (online + predictor, moving-average target, L1 on masked spans and whole words), trained ONLY on the
          text of the visits so far, a few steps beside every LM step - NO PRETRAINING. Each slot keeps up to 400 stored chunks of its
          visits (like the sentence store); they are RE-ENCODED with the current target every 100 steps, so prototypes do not go stale.
GATE      for chunk t, from the JEPA code of chunk t-1 (causal): distance to each slot centroid -> softmax(-d / tau); tau fitted on the
          held-out half of the stored chunks (the system knows which slot it stored each chunk in - no true labels).
SLOTS     one slot is opened at each domain's first visit (as the package's router does locally: 1.00 on revisits). Whether the JEPA can
          tell a new domain itself is measured separately (NOVELTY AUC at each first visit).
TRAINING  two arms on the same data and the same JEPA:
          HARD  the visit's slot only (as the package: one slot per visit)
          SOFT  the JEPA gate during training too (mixture likelihood; every active slot gets gradient)
INFERENCE gating modes, each arm: VISIT (one slot for the whole sequence: the first chunk's domain, as the package routes a visit),
          ORACLE (true domain per chunk), JEPA-HARD (argmax), JEPA-SOFT (full mixture), JEPA-TOP2, BYTES-SOFT (the same gate from a byte
          profile), BPC-POST (causal Bayesian posterior from each slot's likelihood of the previous chunk, as heads.inferred_logp_multi),
          UNIFORM (equal mixture: does mixing by itself help?).
TEXT      SINGLE: 8 consecutive chunks of one domain's held-out text; MIXED: 8 chunks alternating between two domains in segments of 1-3
          chunks (held-out text).
READOUTS  per visit: gate accuracy per chunk (JEPA, bytes) on held-out text of the domains seen; novelty AUC at first visits.
          End: BPC per mode and arm on SINGLE and MIXED; mean active slots (weights > 0.1); FORGETTING per arm (BPC with the arm's own gate on each domain at the
          end of its first visit, at the end of pass 1 and at the end).
usage: python3 toy_jgate.py [seed] [steps per visit]"""
import sys, os, re, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SPV = int(sys.argv[2]) if len(sys.argv) > 2 else 400
torch.set_num_threads(int(os.environ.get('JG_THREADS', '2'))); rs = np.random.RandomState(SEED); torch.manual_seed(SEED); t0 = time.time()
DATA = os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile')
DOMS = ['prose', 'wiki', 'arxiv', 'pubmed', 'law']; ND = len(DOMS); TRAIN_BYTES, VAL_BYTES = 2_600_000, 500_000
RAW = {d: open(os.path.join(DATA, d + '.bin'), 'rb').read() for d in DOMS}
TR = {i: RAW[d][:TRAIN_BYTES] for i, d in enumerate(DOMS)}; VA = {i: RAW[d][-VAL_BYTES:] for i, d in enumerate(DOMS)}
CH = 128; MASK = 257; V = 258; NSEQ = 8
def T(bs): return torch.tensor([list(b) for b in bs])
def rand_chunks(b, n, r, L=CH): return [b[i:i + L] for i in r.randint(0, len(b) - L, size=n)]
def rand_pairs(b, n, r):
    """n pairs of consecutive chunks (previous, current)"""
    st = r.randint(0, len(b) - 2 * CH, size=n); return [b[i:i + CH] for i in st], [b[i + CH:i + 2 * CH] for i in st]
# ---------------- JEPA encoder (as tools/toy_jroute.py) ----------------
class Blk(nn.Module):
    def __init__(s, d, h, causal=False):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d); s.causal = causal
    def forward(s, x):
        h = s.n1(x); m = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), 1) if s.causal else None
        x = x + s.a(h, h, h, attn_mask=m, need_weights=False)[0]; return x + s.m(s.n2(x))
class Enc(nn.Module):
    def __init__(s, d=96, nl=3):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(CH, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl)); s.n = nn.LayerNorm(d)
    def forward(s, X, x_in=None):
        x = (s.emb(X) if x_in is None else x_in) + s.pos[:X.shape[1]]
        for b in s.b: x = b(x)
        return s.n(x)
def mask_spans(X, r):
    M = torch.zeros_like(X, dtype=torch.bool)
    for i in range(X.shape[0]):
        row = bytes(X[i].tolist()); sp = [m.span() for m in re.finditer(rb'[A-Za-z]+', row)]
        for a, b in sp:
            if r.rand() < 0.15: M[i, a:b] = True
        for _ in range(r.randint(1, 4)): a = r.randint(0, CH - 4); M[i, a:a + r.randint(2, 8)] = True
    return X.masked_fill(M, MASK), M
class JEPA:
    def __init__(s):
        s.on, s.tg, s.pr = Enc(), Enc(), Enc(nl=1); s.tg.load_state_dict(s.on.state_dict())
        for p in s.tg.parameters(): p.requires_grad_(False)
        s.op = torch.optim.AdamW(list(s.on.parameters()) + list(s.pr.parameters()), lr=5e-4, weight_decay=0.01); s.r = np.random.RandomState(SEED + 11); s.ls = []
    def step(s, X):
        Xm, M = mask_spans(X, s.r)
        with torch.no_grad(): z = s.tg(X); z = F.layer_norm(z, z.shape[-1:])
        p = s.pr(Xm, x_in=s.on(Xm)); l = F.smooth_l1_loss(p[M], z[M]); s.op.zero_grad(); l.backward(); s.op.step(); s.ls.append(l.item())
        with torch.no_grad():
            for a, b in zip(s.tg.parameters(), s.on.parameters()): a.mul_(0.996).add_(b, alpha=0.004)
    @torch.no_grad()
    def code(s, bs):
        out = []
        for a in range(0, len(bs), 256):
            z = s.tg(T(bs[a:a + 256])); out.append(F.layer_norm(z, z.shape[-1:]).mean(1))
        return torch.cat(out) if out else torch.zeros(0, 96)
def bcode(bs):
    out = np.zeros((len(bs), 256 + 1024), dtype=np.float32)
    for i, b in enumerate(bs):
        a = np.frombuffer(b, dtype=np.uint8).astype(np.int64); out[i, :256] = np.bincount(a, minlength=256); out[i, 256:] = np.bincount((a[:-1] * 257 + a[1:]) % 1024, minlength=1024)
    out[:, :256] /= out[:, :256].sum(1, keepdims=True) + 1e-9; out[:, 256:] /= out[:, 256:].sum(1, keepdims=True) + 1e-9
    return torch.tensor(np.sqrt(out))
# ---------------- slots: stored chunks, centroids, gate ----------------
class Gate:
    """per slot: stored chunks; centroid from even-indexed chunks, tau fitted on odd-indexed ones (the slot each chunk was stored in)"""
    def __init__(s, fn): s.fn = fn; s.C = None; s.tau = 1.0
    def refresh(s, store):
        ks = sorted(store); Z = {k: s.fn(store[k]) for k in ks}
        s.ks = ks; s.C = torch.stack([Z[k][0::2].mean(0) for k in ks])
        D = torch.cat([torch.cdist(Z[k][1::2], s.C) for k in ks]); y = torch.cat([torch.full((Z[k][1::2].shape[0],), i) for i, k in enumerate(ks)])
        best = None
        for tau in np.geomspace(0.003, 3, 25) * float(D.median()):
            ll = float(F.log_softmax(-D / tau, 1).gather(1, y[:, None]).mean())
            if best is None or ll > best[0]: best = (ll, tau)
        s.tau = best[1]
    def weights(s, prev_bs, nslot):
        """(n, nslot) gate weights from the previous chunks; slots not yet opened get 0"""
        D = torch.cdist(s.fn(prev_bs), s.C); w = torch.zeros(len(prev_bs), nslot); w[:, s.ks] = F.softmax(-D / s.tau, 1); return w
    def dist(s, bs): return torch.cdist(s.fn(bs), s.C)
def top2(w):
    v, i = w.topk(min(2, w.shape[1]), 1); o = torch.zeros_like(w).scatter_(1, i, v); return o / o.sum(1, keepdim=True)
# ---------------- the language model with per-slot experts ----------------
class LM(nn.Module):
    def __init__(s, d=128, nl=2, hid=256):
        super().__init__(); s.emb = nn.Embedding(256, d); s.pos = nn.Parameter(torch.randn(CH, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4, causal=True) for _ in range(nl)); s.n = nn.LayerNorm(d)
        s.mlp = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, d)) for _ in range(ND))
        s.head = nn.ModuleList(nn.Linear(d, 256) for _ in range(ND))
    def trunk(s, X):
        x = s.emb(X) + s.pos[:X.shape[1]]
        for b in s.b: x = b(x)
        return s.n(x)
    def slot_logp(s, X, k, h=None):
        h = s.trunk(X) if h is None else h; return F.log_softmax(s.head[k](h + s.mlp[k](h)), -1)
    def mix_logp(s, X, w):
        """log of sum_k w_k p_k (w: (B, ND)); slots with weight 0 everywhere are skipped"""
        h = s.trunk(X); out = None
        for k in range(ND):
            if float(w[:, k].max()) <= 0: continue
            lp = s.slot_logp(X, k, h) + torch.log(w[:, k].clamp_min(1e-12))[:, None, None]
            out = lp if out is None else torch.logaddexp(out, lp)
        return out
def bits(lp, X):
    """bits per byte of each row: predict bytes 1.. from the bytes before (the first byte of a chunk is not scored)"""
    return -lp[:, :-1].gather(-1, X[:, 1:, None]).squeeze(-1).mean(1) / math.log(2)
# ---------------- data for evaluation ----------------
r = np.random.RandomState(SEED + 31); NS, NM = 24, 120
def seq_single(d, r): i = r.randint(0, VAL_BYTES - CH * (NSEQ + 1)); b = VA[d][i:i + CH * (NSEQ + 1)]; return [b[j * CH:(j + 1) * CH] for j in range(NSEQ + 1)], [d] * (NSEQ + 1)
def seq_mixed(r):
    a, b = r.choice(ND, 2, replace=False); ch, lab, cur = [], [], a; ia, ib = r.randint(0, VAL_BYTES - 20 * CH, size=2); pos = {a: ia, b: ib}
    while len(ch) < NSEQ + 1:
        for _ in range(r.randint(1, 4)):
            if len(ch) >= NSEQ + 1: break
            ch.append(VA[cur][pos[cur]:pos[cur] + CH]); pos[cur] += CH; lab.append(cur)
        cur = b if cur == a else a
    return ch, lab
SINGLE = [seq_single(d, r) for d in range(ND) for _ in range(NS)]; MIXED = [seq_mixed(r) for _ in range(NM)]
GATE_TEST = {d: rand_chunks(VA[d], 120, r) for d in range(ND)}
FORG = {d: rand_pairs(VA[d], 96, r) for d in range(ND)}                           # (previous, current) held-out chunk pairs
@torch.no_grad()
def arm_bits(a, m, d, nslot):
    """held-out BPC of domain d with the arm's own gate: HARD the domain's slot, SOFT the JEPA mixture from the previous chunk"""
    pv, cu = FORG[d]; X = T(cu)
    if a == 'HARD': return float(bits(m.slot_logp(X, d), X).mean())
    w = GJ.weights(pv, ND); w[:, nslot:] = 0; w = w / w.sum(1, keepdim=True); return float(bits(m.mix_logp(X, w), X).mean())
# ---------------- run ----------------
ORDER = list(range(ND)) + list(np.random.RandomState(SEED + 5).permutation(ND))
print(f"TOY JEPA GATING, seed {SEED}, {SPV} steps per visit, visit order {ORDER}; continual JEPA (no pretraining), hard and soft training arms", flush=True)
jp = JEPA(); GJ = Gate(jp.code); GB = Gate(bcode); store = {}
arms = {'HARD': LM(), 'SOFT': LM()}; arms['SOFT'].load_state_dict(arms['HARD'].state_dict())
opt = {a: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01) for a, m in arms.items()}
rt = np.random.RandomState(SEED + 41); END = {a: {} for a in arms}; FIRSTEND = {a: {} for a in arms}; NOV = []
@torch.no_grad()
def eval_modes(m, seqs, nslot, modes):
    """bits per byte per mode over chunks 1..NSEQ of each sequence (chunk 0 is context for the gate only); mean active slots per mode"""
    out = {k: [] for k in modes}; act = {k: [] for k in modes}
    prev = [s[0][t - 1] for s in seqs for t in range(1, NSEQ + 1)]; cur = [s[0][t] for s in seqs for t in range(1, NSEQ + 1)]
    lab = torch.tensor([s[1][t] for s in seqs for t in range(1, NSEQ + 1)]); vis = torch.tensor([s[1][0] for s in seqs for t in range(1, NSEQ + 1)])
    X, Xp = T(cur), T(prev); oh = lambda y: F.one_hot(y, ND).float()
    wj = GJ.weights(prev, ND); wb = GB.weights(prev, ND)
    lpp = torch.stack([-bits(m.slot_logp(Xp, k), Xp) * math.log(2) * (CH - 1) if k < nslot else torch.full((len(prev),), -1e9) for k in range(ND)], 1)
    W = {'VISIT': oh(vis), 'ORACLE': oh(lab), 'JEPA-HARD': oh(wj.argmax(1)), 'JEPA-SOFT': wj, 'JEPA-TOP2': top2(wj), 'BYTES-SOFT': wb,
         'BPC-POST': F.softmax(lpp, 1), 'UNIFORM': torch.cat([torch.full((len(prev), nslot), 1.0 / nslot), torch.zeros(len(prev), ND - nslot)], 1)}
    for k in modes:
        w = W[k]; out[k] = float(bits(m.mix_logp(X, w), X).mean()); act[k] = float((w > 0.1).float().sum(1).mean())
    return out, act, float((wj.argmax(1) == lab).float().mean())
MODES = ['VISIT', 'ORACLE', 'JEPA-HARD', 'JEPA-SOFT', 'JEPA-TOP2', 'BYTES-SOFT', 'BPC-POST', 'UNIFORM']
for vi, d in enumerate(ORDER):
    first = d not in store
    if first:
        if store:                                                                              # NOVELTY before the new slot exists
            GJ.refresh(store); GB.refresh(store)
            new = rand_chunks(TR[d][:400_000], 120, rt); old = [c for k in store for c in GATE_TEST[k][:60]]
            for nm, G in (('JEPA', GJ), ('BYTES', GB)):
                a_ = G.dist(new).min(1).values.numpy(); b_ = G.dist(old).min(1).values.numpy()
                NOV.append((vi, nm, float((a_[:, None] > b_[None, :]).mean() + 0.5 * (a_[:, None] == b_[None, :]).mean())))
        store[d] = rand_chunks(TR[d][:200_000], 64, rt)                                        # the slot opens with the start of the visit
    else: store[d] = (store[d] + rand_chunks(TR[d], 64, rt))[-400:]
    GJ.refresh(store); GB.refresh(store); nslot = max(store) + 1
    for st in range(SPV):
        jp.step(T(rand_chunks(TR[d], 24, rt)))
        pv, cu = rand_pairs(TR[d], 16, rt); X = T(cu)
        for a, m in arms.items():
            if a == 'HARD': lp = m.slot_logp(X, d)
            else:
                w = GJ.weights(pv, ND); lp = m.mix_logp(X, w)
            l = -lp[:, :-1].gather(-1, X[:, 1:, None]).mean(); opt[a].zero_grad(); l.backward(); opt[a].step()
        if (st + 1) % 100 == 0:
            store[d] = (store[d] + rand_chunks(TR[d], 32, rt))[-400:]; GJ.refresh(store); GB.refresh(store)
    seen = sorted(store); gt = [c for k in seen for c in GATE_TEST[k]]; gy = torch.tensor([k for k in seen for _ in GATE_TEST[k]])
    acc_j = float((GJ.weights(gt, ND).argmax(1) == gy).float().mean()); acc_b = float((GB.weights(gt, ND).argmax(1) == gy).float().mean())
    nv = ' | '.join(f"{n} {a:.2f}" for v_, n, a in NOV if v_ == vi) if first and vi > 0 else '-'
    with torch.no_grad():
        for a, m in arms.items():
            for k in seen: END[a][k] = arm_bits(a, m, k, nslot)
            if first: FIRSTEND[a][d] = END[a][d]
    print(f"visit {vi + 1:2d} {DOMS[d]:6s} ({'FIRST' if first else 'revisit'}) | JEPA loss {np.mean(jp.ls[-50:]):.3f}, tau {GJ.tau:.3g} | gate accuracy per chunk "
          f"(chance {1 / len(seen):.2f}): JEPA {acc_j:.2f}, BYTES {acc_b:.2f} | NOVELTY AUC (new domain farther than the seen ones): {nv} | "
          f"BPC on this domain (own gate) HARD {END['HARD'][d]:.3f} SOFT {END['SOFT'][d]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    if vi == ND - 1: AFTER1 = {a: dict(END[a]) for a in arms}
LAST = {}
for vi, d in enumerate(ORDER): LAST[d] = vi
print("RESULTS (bits per byte on held-out text, lower is better; active = slots with weight > 0.1)", flush=True)
for nm, seqs in (('SINGLE', SINGLE), ('MIXED', MIXED)):
    for a, m in arms.items():
        o, act, ga = eval_modes(m, seqs, ND, MODES)
        print(f"  {nm:6s} {a:4s} | " + ' | '.join(f"{k} {o[k]:.3f} ({act[k]:.1f})" for k in MODES) + f" | JEPA gate right {ga:.2f}", flush=True)
for a, m in arms.items():
    with torch.no_grad():
        fin = {k: arm_bits(a, m, k, ND) for k in range(ND)}
    print(f"  FORGETTING {a} (own gate: HARD the slot, SOFT the JEPA mixture): end of first visit -> end of pass 1 -> end " + ', '.join(f"{DOMS[k]} {FIRSTEND[a][k]:.3f} -> {AFTER1[a][k]:.3f} -> {fin[k]:.3f}" for k in range(ND))
          + f" | pass-1 forgetting (domains 1-4) {np.mean([AFTER1[a][k] - FIRSTEND[a][k] for k in range(ND - 1)]):+.3f} | mean end {np.mean(list(fin.values())):.3f}", flush=True)
print(f"done ({time.time() - t0:.0f}s)", flush=True)
