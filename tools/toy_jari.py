"""TOY: AN AUDIT-SUPERVISED JEPA INDEX - ARI-style amortised reasoning (Tchuindjo, Shah and Khattab 2026) with Platypus's own audit as
the judge (no effect on the package).
ARI trains a retriever at inference time on a reasoning model's preferences. Platypus has a label-free judge already: every conclusion the
system VERIFIED comes with the premise sentences its audit re-read. So the index can be trained on (question form -> the premise the
system actually used) against (question form -> the other stored sentences naming the same name).
STORE (invented names, never in any training text): chains A -r0-> B -r1-> C written as fact sentences ("{A} works with {B}."); every
name also has 3 filler sentences ("{A} said the weather would turn.") and every A one dead-end fact with a fresh name; plus real prose
sentences (MiniPile) as background. A share of the chains is CONCLUDED (the audit verified 'So A is connected to C' through B): their two
premises are the audit pairs. The rest are HELD OUT: never concluded, never paired - the test of whether the index generalises (ARI: 29%
of the relevant documents it surfaced were never judged).
ARMS (the index encoder of the package: byte transformer, EMA target, predictor, L1; InfoNCE as arm C):
  V        cut-sentence -> whole sentence, L1 (cg26s)
  C        V + InfoNCE over the batch (cg26u arm C)
  C+AUD    C + the audit pairs: question form 'So A is connected to ' -> premise 1, 'So B is connected to ' -> premise 2, contrastive
           against the batch and against the other sentences naming the same name (hard negatives)
  C+SHUF   control: the same pairs with the premise replaced by a RANDOM sentence naming the same name (labels without the audit)
  G, G+AUD the generic trace (names masked as one id, kept as slots), without and with the audit pairs
READOUTS (held-out chains; HYBRID = among the stored sentences naming the question's name, ranked by the key; chance about 0.2):
  conclusion form   'So A is connected to ' -> premise 1 (the question the system actually asks)
  written           'A r0 ' -> premise 1
  held-out wording  'A <paraphrase of r0> ' -> premise 1
  TWO-HOP           conclusion form -> s1, read its other name B', then 'So B' is connected to ' among the sentences naming B' (s1 excluded)
                    -> premise 2; right only if both hops are
  plus the same on the concluded (trained) chains, and full-store top-1 for the conclusion form.
usage: python3 toy_jari.py [seeds] [steps]"""
import sys, os, re, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
ARMS = os.environ.get('JARI_ARMS', 'V,C,C+AUD,C+SHUF,G,G+AUD').split(',')
torch.set_num_threads(1); t0 = time.time()
DATA = os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile')
L, D, NL, B, LR, MOM, NCE, AW = 128, 128, 3, 64, 5e-4, 0.996, 0.25, 0.5
REL = {'works with': ['is a colleague of', 'shares an office with'], 'lives near': ['resides close to', 'has a house next to'],
       'reports to': ['answers to', 'is managed by'], 'is a partner of': ['is in business with', 'co-owns a firm with'],
       'trained under': ['was taught by', 'apprenticed with'], 'is linked to': ['is associated with', 'is tied to']}
RELS = list(REL)
FILL = ['{A} said the weather would turn.', 'Nobody had seen {A} since the spring.', '{A} kept the old ledger in a drawer.',
        'The letter from {A} arrived late.', 'In the end {A} chose the quieter road.', '{A} was reading by the window.',
        'Some people still remember {A} at the fair.', 'The garden behind {A} was overgrown.']
SYL = ['ka', 'to', 'mi', 'ra', 've', 'lo', 'du', 'sen', 'tar', 'bel', 'nor', 'qui', 'fa', 'zo', 'pel', 'dra', 'mon', 'ris', 'tev', 'gal',
       'hun', 'cor', 'vin', 'sal', 'bri', 'ed', 'an', 'ul', 'or', 'is']
def names(n, r):
    out = set()
    while len(out) < n:
        s = ''.join(r.choice(SYL, r.randint(2, 4))); s = s[0].upper() + s[1:]
        if 5 <= len(s) <= 10: out.add(s)
    return sorted(out)
def sentences(txt, n, r):
    ss = [s.strip() for s in re.split(r'(?<=[.!?])\s+', txt) if 30 <= len(s.strip()) <= 120 and s.isascii()]
    return [ss[i] for i in r.choice(len(ss), min(n, len(ss)), replace=False)]
def world(seed):
    r = np.random.RandomState(seed); NM = names(1200, r); r.shuffle(NM); k = 0; chains = []; store = []
    for _ in range(300):
        A, Bn, C = NM[k:k + 3]; k += 3; r0, r1 = r.choice(RELS, 2)
        p1, p2 = f"{A} {r0} {Bn}.", f"{Bn} {r1} {C}."; X = NM[k]; k += 1
        dead = f"{A} {r.choice([x for x in RELS if x != r0])} {X}."
        chains.append(dict(A=A, B=Bn, C=C, r0=r0, r1=r1, p1=p1, p2=p2)); store += [p1, p2, dead]
        for nm in (A, Bn, C, X):
            for t in r.choice(FILL, 3, replace=False): store.append(t.format(A=nm))
    bg = sentences(open(os.path.join(DATA, 'prose.bin'), 'rb').read()[:3_000_000].decode('latin-1'), 4000, r)
    store = store + bg; r.shuffle(store)
    nconc = 180; conc, held = chains[:nconc], chains[nconc:]
    used = set(c[n] for c in chains for n in ('A', 'B', 'C')) | set(NM[:k])
    rx = re.compile(r'\b(' + '|'.join(sorted(used, key=len, reverse=True)) + r')\b')
    by = {}
    for i, s in enumerate(store):
        for m in set(rx.findall(s)): by.setdefault(m, []).append(i)
    return store, conc, held, rx, by
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(258, D); s.pos = nn.Parameter(torch.randn(L, D) * 0.02)
        s.b = nn.ModuleList(nn.TransformerEncoderLayer(D, 4, 4 * D, 0.0, batch_first=True, norm_first=True) for _ in range(NL)); s.ln = nn.LayerNorm(D)
    def forward(s, X):
        m = (X != 256); x = s.emb(X) + s.pos[:X.shape[1]]
        for b in s.b: x = b(x, src_key_padding_mask=~m)
        x = s.ln(x); return (x * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
def ids(s, gen, rx):
    b = list(s.encode('latin-1', 'replace'))
    if gen:
        for m in rx.finditer(s):
            for j in range(m.start(), m.end()): b[j] = 257
    return b
def batch(ts, gen, rx):
    bs = [ids(t, gen, rx)[:L] for t in ts]; X = torch.full((len(bs), max(4, max(len(b) for b in bs))), 256, dtype=torch.long)
    for j, b in enumerate(bs): X[j, :len(b)] = torch.tensor(b)
    return X
def cut(s, r):
    w = [i + 1 for i, c in enumerate(s) if c == ' ']
    if len(w) < 2: return s[:max(1, len(s) // 2)]
    return s[:w[-1]] if r.rand() < 0.5 else s[:w[int(r.randint(1, len(w)))]]
def nce(p, z, tau_scale):
    return F.cross_entropy(-torch.cdist(p, z, p=1) / max(1e-3, float(z.abs().mean()) * D * 0.1 * tau_scale), torch.arange(len(p)))
def train(arm, store, conc, rx, by, seed):
    torch.manual_seed(seed); r = np.random.RandomState(seed + 7); gen = arm.startswith('G')
    on, tg = Enc(), Enc(); tg.load_state_dict(on.state_dict()); pr = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
    for p in tg.parameters(): p.requires_grad_(False)
    op = torch.optim.AdamW(list(on.parameters()) + list(pr.parameters()), lr=LR, weight_decay=0.01)
    pairs = []
    for c in conc:                                                          # the audit pairs: question form -> the premise it used
        pairs.append((f"So {c['A']} is connected to ", c['p1'], c['A'])); pairs.append((f"So {c['B']} is connected to ", c['p2'], c['B']))
    if arm.endswith('SHUF'):                                                # control: a random sentence naming the same name instead
        pairs = [(q, store[by[n][int(r.randint(len(by[n])))]], n) for q, _, n in pairs]
    nc = NCE if arm != 'V' and arm != 'G' else 0.0; aud = 'AUD' in arm or 'SHUF' in arm
    for st in range(STEPS):
        src = [store[i] for i in r.choice(len(store), B, replace=False)]
        with torch.no_grad(): z = tg(batch(src, gen, rx))
        p = pr(on(batch([cut(s, r) for s in src], gen, rx))); l = (p - z).abs().mean()
        if nc > 0: l = l + nc * nce(p, z, 1.0)
        if aud:
            sel = [pairs[i] for i in r.choice(len(pairs), 16, replace=False)]; tgt = [s for _, s, _ in sel]
            for q, s, n in sel:                                             # hard negatives: other sentences naming the same name
                oth = [store[i] for i in by[n] if store[i] != s]
                tgt += [oth[int(j)] for j in r.choice(len(oth), min(2, len(oth)), replace=False)]
            with torch.no_grad(): za = tg(batch(tgt, gen, rx))
            pa = pr(on(batch([q for q, _, _ in sel], gen, rx)))
            la = F.cross_entropy(-torch.cdist(pa, za, p=1) / max(1e-3, float(za.abs().mean()) * D * 0.1), torch.arange(len(sel)))
            l = l + AW * la
        op.zero_grad(); l.backward(); op.step()
        with torch.no_grad():
            for a, b in zip(tg.parameters(), on.parameters()): a.mul_(MOM).add_(b, alpha=1 - MOM)
    on.eval(); pr.eval(); return on, tg, pr, gen
@torch.no_grad()
def evaluate(model, store, chains, rx, by):
    on, tg, pr, gen = model
    K = torch.cat([tg(batch(store[a:a + 512], gen, rx)) for a in range(0, len(store), 512)])
    pos = {s: i for i, s in enumerate(store)}
    def q(t): return pr(on(batch([t], gen, rx)))[0]
    def hyb(t, n, excl=()):
        c = [i for i in by.get(n, []) if i not in excl]
        return c[int((K[c] - q(t)).abs().sum(-1).argmin())] if c else None
    R = {k: [] for k in ('conc', 'writ', 'held', 'two', 'full')}
    for c in chains:
        i1, i2 = pos[c['p1']], pos[c['p2']]
        s1 = hyb(f"So {c['A']} is connected to ", c['A']); R['conc'].append(s1 == i1)
        R['writ'].append(hyb(f"{c['A']} {c['r0']} ", c['A']) == i1)
        R['held'].append(hyb(f"{c['A']} {REL[c['r0']][0]} ", c['A']) == i1)
        R['full'].append(int((K - q(f"So {c['A']} is connected to ")).abs().sum(-1).argmin()) == i1)
        b2 = [m for m in rx.findall(store[s1]) if m != c['A']] if s1 is not None else []
        s2 = hyb(f"So {b2[0]} is connected to ", b2[0], excl=(s1,)) if b2 else None
        R['two'].append(s1 == i1 and s2 == i2)
    return {k: float(np.mean(v)) for k, v in R.items()}
print(f"TOY AUDIT-SUPERVISED JEPA INDEX, {SEEDS} seeds, {STEPS} steps, arms {ARMS}", flush=True)
RES = {a: [] for a in ARMS}
for seed in range(int(os.environ.get('JARI_SEED0', 0)), int(os.environ.get('JARI_SEED0', 0)) + SEEDS):   # JARI_SEED0: resume at a later seed
    store, conc, held, rx, by = world(100 + seed)
    if seed == int(os.environ.get('JARI_SEED0', 0)): print(f"  store {len(store)} sentences ({len(conc)} concluded chains, {len(held)} held out); sentences naming an A: median {int(np.median([len(by[c['A']]) for c in held]))}", flush=True)
    for arm in ARMS:
        ta = time.time(); m = train(arm, store, conc, rx, by, seed); RES[arm].append((evaluate(m, store, held, rx, by), evaluate(m, store, conc, rx, by)))
        h, t = RES[arm][-1]
        print(f"  seed {seed} {arm:7s} | HELD-OUT chains: conclusion form {h['conc']:.2f}, written {h['writ']:.2f}, held-out wording {h['held']:.2f}, "
              f"TWO-HOP {h['two']:.2f}, full-store top-1 {h['full']:.2f} | trained chains: conclusion form {t['conc']:.2f}, two-hop {t['two']:.2f}  ({time.time()-ta:.0f}s)", flush=True)
print("MEAN over seeds (held-out chains; hybrid chance about 0.2):")
for arm in ARMS:
    H = {k: np.mean([x[0][k] for x in RES[arm]]) for k in RES[arm][0][0]}; T = {k: np.mean([x[1][k] for x in RES[arm]]) for k in RES[arm][0][1]}
    print(f"  {arm:7s} | conclusion form {H['conc']:.2f} | written {H['writ']:.2f} | held-out wording {H['held']:.2f} | TWO-HOP {H['two']:.2f} | "
          f"full-store {H['full']:.2f} || trained chains: conclusion {T['conc']:.2f}, two-hop {T['two']:.2f}", flush=True)
print(f"DONE {time.time()-t0:.0f}s")
