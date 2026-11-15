"""DOES A WRITTEN INFERENCE IMPROVE THE MODEL? (16 Nov 2026; tools/useful_writes.py) - offline, the GPU write path, distant set, stand-in trunk.
The author's criterion: an inference should be added if it lowers bits (or improves inference). GPU cg27a_kmf / kmr: the keyed memory's
'A is connected to C' answer COSTS bits on later real text naming A, for right inferences as much as wrong ones. Here the same 200 writes
are stored in a keyed continuation memory (key: the head name; continuation: the rest of the claim) in different FORMS, and the model
(trunk + memory through a learned gate, as inference.kunits; prequential) is scored on LATER real text that names a written name
(relational records of documents read after the write, 'X <words> Y', with no context): bits saved against the trunk alone.
  F1  'A is connected to C.'  (A = the alphabetically first name, as the GPU writes)
  F2  TYPED (route 1): 'head <wording> tail' - the class pair's commonest relational wording, head = the more specific name
Then USEFULNESS AS THE SELECTOR: each write's own bits saved on the text of the first half of the later documents (label-free) decides
whether it is kept; the kept set is scored on the second half (held out): total bits saved, and the share right (distant labels, scoring).
Usage: TG_CKPT=<stand-in trunk> python3 tools/useful_writes.py <distant.json.gz>"""
import sys, os, types, time, collections, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.kunits import _row_bits, CL
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2)
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; TY = pm._PM['TY']
h = lambda v: int(round(ND * min(1.0, (v + 1) / 10))); v = 1
isrel = lambda r: 1 <= r[5] <= 8 and not pm._rret_listlike(r)
norm = lambda w: ' '.join(t for t in w.split() if t not in ('the', 'a', 'an'))
pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(v)], v); W = dict(pm._PM['rti'])
y = {k: e['truth'] for k, e in W.items()}
nb = collections.defaultdict(set)
for r in R:
    if r[0] < h(v): nb[r[1]].add(r[2]); nb[r[2]].add(r[1])
DG = {n: len(s) for n, s in nb.items()}
tab = collections.defaultdict(collections.Counter)
for r in R:
    if r[0] < h(v) and isrel(r) and DG.get(r[1], 0) != DG.get(r[2], 0) and r[1] in DG and r[2] in DG:
        hd, tl, fw = (r[1], r[2], 1) if DG[r[1]] < DG[r[2]] else (r[2], r[1], 0); tab[(TY.get(hd, 'X'), TY.get(tl, 'X'))][(norm(r[3]), fw)] += 1
FORMS = {}
for (A, C) in W:
    hd, tl = (A, C) if DG.get(A, 0) <= DG.get(C, 0) else (C, A); cp = (TY.get(hd, 'X'), TY.get(tl, 'X'))
    FORMS.setdefault('F1 connected to', {})[(A, C)] = (A, ' is connected to ' + C + '.')
    if tab.get(cp):
        (w, fw), _ = tab[cp].most_common(1)[0]; X, Y = (hd, tl) if fw else (tl, hd)
        FORMS.setdefault('F2 typed', {})[(A, C)] = (X, ' ' + w + ' ' + Y + '.')
# later text: relational records of documents read after the write naming a written name; first half / second half of that window
names = {n for k in W for n in k}; mid = (h(v) + h(v + 2)) // 2
spans = [(r[0] < mid, r[7], frozenset((r[1], r[2])) in {frozenset(k) for k in W}, r[1], r[2]) for r in R if h(v) <= r[0] < h(v + 2) and isrel(r) and (r[1] in names)]
P(f"WRITES {len(W)} (right {np.mean(list(y.values())):.3f}) | later relational spans whose FIRST name is a written name: {len(spans)} ({sum(s[2] for s in spans)} state a written pair)  ({time.time() - T0:.0f}s)")
DW = int(os.environ.get('TG_D', 192)); L_ = 128; PAD = 256; V_ = 257
class Blk(nn.Module):
    def __init__(s, d, hh):
        super().__init__(); s.a = nn.MultiheadAttention(d, hh, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        q = s.n1(x); x = x + s.a(q, q, q, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class LM(nn.Module):
    def __init__(s, d=DW, nl=3):
        super().__init__(); s.emb = nn.Embedding(V_, d); s.pos = nn.Parameter(torch.randn(L_, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl)); s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V_)
    def forward(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for b in s.b: x = b(x, am)
        return s.o(s.n(x))
lm = LM(); lm.load_state_dict(torch.load(os.environ['TG_CKPT'])); lm.eval()
TXT = [s[1].encode('latin-1', 'replace')[:L_] for s in spans]; PT, EN = [], []
with torch.no_grad():
    for i0 in range(0, len(TXT), 256):
        ch = TXT[i0:i0 + 256]; T = max(len(t) for t in ch); X = torch.full((len(ch), T), PAD, dtype=torch.long)
        for i, t in enumerate(ch): X[i, :len(t)] = torch.tensor(list(t))
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(lm(Xi), -1)[..., :256]; lp = lp - torch.logsumexp(lp, -1, keepdim=True)
        pt = lp.gather(-1, X.clamp(max=255)[..., None])[..., 0].exp().numpy(); en = (-(lp.exp() * lp).sum(-1) / math.log(2)).numpy()
        for i, t in enumerate(ch): PT.append(pt[i, :len(t)]); EN.append(en[i, :len(t)])
P(f"trunk scored {len(TXT)} spans: {np.mean([-np.log2(np.maximum(p, 1e-12)).mean() for p in PT]):.3f} bits per byte  ({time.time() - T0:.0f}s)")
def mem_of(form, keep=None):
    M = {'NAME': {}, 'DATE': {}, 'NUMBER': {}}
    for k, (X, cont) in FORMS[form].items():
        if keep is not None and k not in keep: continue
        M['NAME'].setdefault(X.encode('latin-1', 'replace'), collections.deque(maxlen=32)).append(cont.encode('latin-1', 'replace')[:CL])
    return dict(mem=M, gate={'NAME': {}, 'ALL': {}})
def score(st, idx):
    tot = 0.0; nbytes = 0
    for i in idx: tot += _row_bits(st, TXT[i], PT[i], EN[i], ('NAME',), 'NAME', 0.05)[0]; nbytes += len(TXT[i])
    return tot, nbytes
allidx = list(range(len(TXT))); pair = [i for i in allidx if spans[i][2]]; other = [i for i in allidx if not spans[i][2]]
for form in FORMS:
    st = mem_of(form); s_all, n_all = score(st, allidx)
    st = mem_of(form); s_p, n_p = score(st, pair); st = mem_of(form); s_o, n_o = score(st, other)
    P(f"   {form:16s} ({len(FORMS[form])} writes): bits saved per 1,000 bytes of later text {1000 * s_all / n_all:+.2f} | on spans stating a written pair {1000 * s_p / max(1, n_p):+.2f} "
      f"({len(pair)}) | on spans of a written name with another partner {1000 * s_o / max(1, n_o):+.2f} ({len(other)})")
# usefulness as the selector: each write's own bits saved on the FIRST half (label-free) decides; scored on the SECOND half
first = [i for i in allidx if spans[i][0]]; second = [i for i in allidx if not spans[i][0]]
byname = collections.defaultdict(list)
for i in first: byname[spans[i][3]].append(i)
for form in FORMS:
    gain = {}
    for k, (X, cont) in FORMS[form].items():
        idx = byname.get(X, [])
        if not idx: continue
        st = mem_of(form, keep={k}); gain[k] = score(st, idx)[0]
    keep = {k for k, g in gain.items() if g > 0}
    s_all = score(mem_of(form), second); s_keep = score(mem_of(form, keep), second)
    rk = np.mean([y[k] for k in keep]) if keep else float('nan'); rd = np.mean([y[k] for k in gain if k not in keep]) if len(gain) > len(keep) else float('nan')
    P(f"   USEFULNESS SELECTOR, {form}: {len(gain)} writes with later text in the first half, KEPT (saved bits there) {len(keep)} (right {rk:.3f}; dropped right {rd:.3f}) | "
      f"bits saved per 1,000 bytes on the SECOND half: all writes {1000 * s_all[0] / max(1, s_all[1]):+.2f}, kept only {1000 * s_keep[0] / max(1, s_keep[1]):+.2f}")
P(f"DONE ({time.time() - T0:.0f}s)")
