"""THE FULL INFERENCE LOOP, OFFLINE OVER THREE ROUNDS (16 Nov 2026; tools/loop_sim.py) - GPU write path, distant_50k, stand-in trunk.
Each round (writes at visits 2 / 4 / 6; the store's written pairs persist, so later rounds write new pairs, as on the GPU) the rule
proposes 400 never-stated pairs (the batch 0-200, the next tier 200-400); each is TYPED (the composition table's wording where it types
it, else the class pair's; the table learned from earlier rounds' KEPT writes that a later relational record confirmed - no labels); then
a POLICY keeps some:
  JUDGE       the batch only, kept if a relational record states the pair in the window after the write (the GPU now)
  USEFUL      all 400, kept if the typed write saved bits on the FIRST half of that window (prediction gain, label-free)
  JUDGE+USE   the batch by the judge, the next tier by usefulness
  ALL BATCH   the batch, no filter (control)
Kept writes accumulate in a keyed continuation memory over rounds (as the model's memory would); each round, the model (trunk + that
memory, learned gate, prequential) is scored on the SECOND half of the round's window (relational spans whose first name is a written
name): bits saved per 1,000 bytes. Distant labels score only: share right of what is kept, and of the cumulative memory.
Usage: TG_CKPT=<trunk> python3 tools/loop_sim.py <distant_50k.json.gz>"""
import sys, os, types, time, collections, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.kunits import _row_bits, CL
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2)
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=int(os.environ.get('LS_SEED', 4)), inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=400, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; TY = pm._PM['TY']
h = lambda v: int(round(ND * min(1.0, (v + 1) / 10)))
import re as _re
def _list2(r):                                                                 # LS_LIST2 1: the wider list filter ('X , Y and Z' too; tools/judge_list.py)
    w = ' ' + r[3] + ' '
    return bool(_re.search(r' , .*\b(and|or)\b', w)) or bool(_re.search(r'\b(and|or)\b.* , ', w)) or r[3].strip() in ('and', 'or', ', and', ', or', 'as well as', 'and the')
isrel = (lambda r: 1 <= r[5] <= 8 and not pm._rret_listlike(r) and not _list2(r)) if int(os.environ.get('LS_LIST2', 0)) else (lambda r: 1 <= r[5] <= 8 and not pm._rret_listlike(r))
norm = lambda w: ' '.join(t for t in w.split() if t not in ('the', 'a', 'an'))
LISTW = pm._RRET_LISTW
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
def trunk(txts):
    PT, EN = [], []
    with torch.no_grad():
        for i0 in range(0, len(txts), 256):
            ch = txts[i0:i0 + 256]; T = max(len(t) for t in ch); X = torch.full((len(ch), T), PAD, dtype=torch.long)
            for i, t in enumerate(ch): X[i, :len(t)] = torch.tensor(list(t))
            Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(lm(Xi), -1)[..., :256]; lp = lp - torch.logsumexp(lp, -1, keepdim=True)
            pt = lp.gather(-1, X.clamp(max=255)[..., None])[..., 0].exp().numpy(); en = (-(lp.exp() * lp).sum(-1) / math.log(2)).numpy()
            for i, t in enumerate(ch): PT.append(pt[i, :len(t)]); EN.append(en[i, :len(t)])
    return PT, EN
def mem(claims):
    M = {'NAME': {}, 'DATE': {}, 'NUMBER': {}}
    for X, w, Y in claims: M['NAME'].setdefault(X.encode('latin-1', 'replace'), collections.deque(maxlen=32)).append((' ' + w + ' ' + Y + '.').encode('latin-1', 'replace')[:CL])
    return dict(mem=M, gate={'NAME': {}, 'ALL': {}})
import re as _re2
_RUN2 = _re2.compile(rb"[A-Z][\w'\-]*(?: (?:of |de |the )?[A-Z][\w'\-]*)*")
def bits_name(claims, idx, TXT, PT, EN):
    """16 Nov: the GPU feedback readout's signal - bits saved only on the SECOND name of each span (its last capitalised run)"""
    st = mem(claims); s = 0.0; n = 0
    for i in idx:
        b = TXT[i]; runs = [(m.start(), m.end()) for m in _RUN2.finditer(b)]
        if len(runs) < 2: continue
        s0, e0 = runs[-1]; mk = np.zeros(len(b), bool); mk[s0:e0] = True
        s += _row_bits(st, b, PT[i], EN[i], ('NAME',), 'NAME', 0.05, mask=mk)[0]; n += 1
    return s / max(1, n)
def bits(claims, idx, TXT, PT, EN):
    st = mem(claims); s = 0.0; n = 0
    for i in idx: s += _row_bits(st, TXT[i], PT[i], EN[i], ('NAME',), 'NAME', 0.05)[0]; n += len(TXT[i])
    return 1000 * s / max(1, n)
POL = tuple(os.environ.get('LS_POL', 'JUDGE,USEFUL,JUDGE+USE,ALL BATCH').split(','))   # also EITHER (judge or useful, all 400), USEFUL>t (gain above t bits per 1,000 bytes)
KEPT = {p: {} for p in POL}; T3 = {p: collections.defaultdict(collections.Counter) for p in POL}
pm._PM['rti'] = {}
for rd, v in enumerate(tuple(int(x) for x in os.environ.get('LS_ROUNDS', '1,3,5').split(',')), 1):
    before = set(pm._PM['rti']); pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(v)], v)
    W = {k: w for k, w in pm._PM['rti'].items() if k not in before}; y = {k: w['truth'] for k, w in W.items()}; tier = {k: w['rank'] < 200 for k, w in W.items()}
    nb = collections.defaultdict(set); tab = collections.defaultdict(collections.Counter); pw = collections.defaultdict(list); mid = (h(v) + h(v + 2)) // 2
    need = {frozenset((A, b)) for (A, C), e in W.items() for b in e['prem']} | {frozenset((b, C)) for (A, C), e in W.items() for b in e['prem']}
    for r in R:
        if r[0] < h(v): nb[r[1]].add(r[2]); nb[r[2]].add(r[1])
    DG = {n: len(s) for n, s in nb.items()}; want = {frozenset(k): k for k in W}; conf = collections.defaultdict(list)
    for r in R:
        if r[0] < h(v) and isrel(r):
            if r[1] in DG and r[2] in DG and DG[r[1]] != DG[r[2]]:
                hd, tl, fw = (r[1], r[2], 1) if DG[r[1]] < DG[r[2]] else (r[2], r[1], 0); w_ = norm(r[3]); w9 = ' '.join(w_.replace('"', ' ').split()).strip(' ,;()')
                if any(ch.isalpha() for ch in w_) and w9 not in LISTW and w9 not in ('and', 'or'): tab[(TY.get(hd, 'X'), TY.get(tl, 'X'))][(w_, fw)] += 1
            f = frozenset((r[1], r[2]))
            if f in need: pw[f].append((r[1], r[2], norm(r[3])))
        if h(v) <= r[0] < h(v + 2) and isrel(r):
            k = want.get(frozenset((r[1], r[2])))
            if k: conf[k].append((r[1], r[2], norm(r[3])))
    def orient(X, Y):
        cs = pw.get(frozenset((X, Y)))
        if not cs: return None
        a, b, w = collections.Counter(cs).most_common(1)[0][0]; return w if a == X else w + ' ^R'
    spans = [(r[0] < mid, r[7], r[1]) for r in R if h(v) <= r[0] < h(v + 2) and isrel(r)]
    names = {n for p in POL for k in list(KEPT[p]) + list(W) for n in k}
    spans = [s for s in spans if s[2] in names]; TXT = [s[1].encode('latin-1', 'replace')[:L_] for s in spans]; PT, EN = trunk(TXT)
    first = collections.defaultdict(list); second = [i for i, s in enumerate(spans) if not s[0]]
    for i, s in enumerate(spans):
        if s[0]: first[s[2]].append(i)
    out = []
    for p in POL:
        typed = {}
        for (A, C), e in W.items():
            hd, tl = (A, C) if DG.get(A, 0) <= DG.get(C, 0) else (C, A); cp = (TY.get(hd, 'X'), TY.get(tl, 'X')); votes = collections.Counter()
            for b in e['prem']:
                a1, a2 = orient(A, b), orient(b, C)
                if a1 and a2: votes.update(T3[p].get((a1, a2), {}))
            if votes:
                cw = votes.most_common(1)[0][0]; rv = cw.endswith(' ^R'); w0 = cw[:-3] if rv else cw; typed[(A, C)] = (C, w0, A) if rv else (A, w0, C)
            elif tab.get(cp):
                (w, fw), _ = tab[cp].most_common(1)[0]; typed[(A, C)] = (hd, w, tl) if fw else (tl, w, hd)
        def useful_name(k):
            c = typed.get(k)
            if c is None or not first.get(c[0]): return False
            return bits_name([c], first[c[0]], TXT, PT, EN) > 0
        def useful(k, t=0.0):
            c = typed.get(k)
            if c is None or not first.get(c[0]): return False
            return bits([c], first[c[0]], TXT, PT, EN) > t
        judged = lambda k: bool(conf.get(k))
        if p == 'JUDGE': keep = [k for k in W if tier[k] and judged(k)]
        elif p == 'USEFUL': keep = [k for k in W if useful(k)]
        elif p == 'JUDGE+USE': keep = [k for k in W if (tier[k] and judged(k)) or (not tier[k] and useful(k))]
        elif p == 'BOTH-NAME': keep = [k for k in W if judged(k) and useful_name(k)]
        elif p == 'USEFUL-NAME': keep = [k for k in W if useful_name(k)]
        elif p == 'BOTH': keep = [k for k in W if judged(k) and useful(k)]
        elif p == 'EITHER': keep = [k for k in W if judged(k) or useful(k)]
        elif p == 'EITHER-BATCH': keep = [k for k in W if tier[k] and (judged(k) or useful(k))]
        elif p.startswith('USEFUL>'): keep = [k for k in W if useful(k, float(p[7:]))]
        else: keep = [k for k in W if tier[k]]
        for k in keep:
            if k in typed: KEPT[p][k] = typed[k]
            if conf.get(k):                                                    # the composition table learns from kept, confirmed writes (no labels)
                X, Y, w = collections.Counter(conf[k]).most_common(1)[0][0]; cf = w if X == k[0] else w + ' ^R'
                for b in W[k]['prem']:
                    a1, a2 = orient(k[0], b), orient(b, k[1])
                    if a1 and a2: T3[p][(a1, a2)][cf] += 1
        allk = list(KEPT[p]); b2 = bits([KEPT[p][k] for k in allk], second, TXT, PT, EN)
        yk = [y[k] for k in keep]; yall = [pm._PM['rti'][k]['truth'] for k in allk]
        out.append(f"{p}: kept {len(keep)} (right {np.mean(yk) if yk else float('nan'):.3f}; composition-typed {np.mean([k in typed and typed[k][1] not in ('in',) for k in keep]) if keep else 0:.2f}) | memory {len(allk)} "
                   f"(right {np.mean(yall) if yall else float('nan'):.3f}) saves {b2:+.2f}")
    P(f"ROUND {rd} (visit {v + 1}): written {len(W)} (batch right {np.mean([y[k] for k in W if tier[k]]):.3f}, next tier {np.mean([y[k] for k in W if not tier[k]]):.3f}) | "
      f"{len(spans)} later spans || " + " || ".join(out) + f"  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
