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
SELX, SELY = [], []
PRD = pm._PM['PRD']; CONT = {'P131', 'P17', 'P30', 'P706', 'P276', 'P361'}
def _cg(x, grp):
    if not grp: return x
    b_, inv = (x[:-3], '^-1') if x.endswith('^-1') else (x, ''); return ('CONTAINED' + inv) if b_ in CONT else x
def _rels(X, Y, grp): return {_cg(x, grp) for x in PRD.get((X, Y), ())} | {_cg(x + '^-1', grp) for x in PRD.get((Y, X), ())}
_RMc = collections.defaultdict(collections.Counter)
for r in R:
    if r[6] and 1 <= r[5] <= 8 and not pm._rret_listlike(r):
        for x in set(PRD.get((r[1], r[2]), ())) | {x + '^-1' for x in PRD.get((r[2], r[1]), ())}: _RMc[norm(r[3])][x] += 1
RMAP = {w: c.most_common(1)[0][0] for w, c in _RMc.items()}
def tright(c, grp): return c[1] in RMAP and _cg(RMAP[c[1]], grp) in _rels(c[0], c[2], grp)                                                            # 16 Nov LEARNED SELECTION: write-time features -> usefulness, from earlier rounds
pm._PM['rti'] = {}
ALLW = {}                                                                      # 16 Nov (queue 34, LS_STAB 1): every written inference across rounds
for rd, v in enumerate(tuple(int(x) for x in os.environ.get('LS_ROUNDS', '1,3,5').split(',')), 1):
    before = set(pm._PM['rti']); pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(v)], v)
    W = {k: w for k, w in pm._PM['rti'].items() if k not in before}; y = {k: w['truth'] for k, w in W.items()}; tier = {k: w['rank'] < 200 for k, w in W.items()}
    nb = collections.defaultdict(set); tab = collections.defaultdict(collections.Counter); pw = collections.defaultdict(list); mid = h(v) + int(float(os.environ.get('LS_UWIN', 0.5)) * (h(v + 2) - h(v)))   # LS_UWIN: share of the window the judge and usefulness see
    need = {frozenset((A, b)) for (A, C), e in W.items() for b in e['prem']} | {frozenset((b, C)) for (A, C), e in W.items() for b in e['prem']}
    for r in R:
        if r[0] < h(v): nb[r[1]].add(r[2]); nb[r[2]].add(r[1])
    DG = {n: len(s) for n, s in nb.items()}; want = {frozenset(k): k for k in W}; conf = collections.defaultdict(list); HF = collections.defaultdict(list)
    for r in R:
        if r[0] < h(v) and isrel(r):
            if r[1] in DG and r[2] in DG and DG[r[1]] != DG[r[2]]:
                hd, tl, fw = (r[1], r[2], 1) if DG[r[1]] < DG[r[2]] else (r[2], r[1], 0); w_ = norm(r[3]); w9 = ' '.join(w_.replace('"', ' ').split()).strip(' ,;()')
                if any(ch.isalpha() for ch in w_) and w9 not in LISTW and w9 not in ('and', 'or') and (not int(os.environ.get('LS_CLEAN', 0)) or (w_[0].isalnum() and w_[-1].isalnum())): tab[(TY.get(hd, 'X'), TY.get(tl, 'X'))][(w_, fw)] += 1
            f = frozenset((r[1], r[2]))
            if f in need: pw[f].append((r[1], r[2], norm(r[3])))
        if h(v) <= r[0] < (mid if int(os.environ.get('LS_HALF', 1)) else h(v + 2)) and isrel(r):   # 16 Nov: the judge (and retyping) see only the FIRST half
            k = want.get(frozenset((r[1], r[2])))                              #   (the second half is the held-out scoring text)
            if k: conf[k].append((r[1], r[2], norm(r[3])))
            HF[r[1]].append(r[2])                                              # 16 Nov (queue 35): head-first records, the typed confirmation signal
    def orient(X, Y):
        cs = pw.get(frozenset((X, Y)))
        if not cs: return None
        a, b, w = collections.Counter(cs).most_common(1)[0][0]; return w if a == X else w + ' ^R'
    spans = [(r[0] < mid, r[7], r[1]) for r in R if h(v) <= r[0] < h(v + 2) and isrel(r)]
    names = {n for p in POL for k in list(KEPT[p]) + list(W) for n in k} | {n for k in ALLW for n in k}
    spans = [s for s in spans if s[2] in names]; TXT = [s[1].encode('latin-1', 'replace')[:L_] for s in spans]; PT, EN = trunk(TXT)
    first = collections.defaultdict(list); second = [i for i, s in enumerate(spans) if not s[0]]
    for i, s in enumerate(spans):
        if s[0]: first[s[2]].append(i)
    out = []
    ET9 = ['PER', 'LOC', 'ORG', 'TIME', 'NUM', 'MISC']
    def fx(k):                                                                 # write-time features of a candidate
        e = W[k]; A, C = k; dA, dC = DG.get(A, 0), DG.get(C, 0)
        oh = lambda t: [float(TY.get(t, 'X') == c) for c in ET9]
        return [float(e.get('score', 0)), float(e['rank']), float(len(e.get('prem', ()))), np.log1p(min(dA, dC)), np.log1p(max(dA, dC)), abs(np.log1p(dA) - np.log1p(dC))] + oh(A) + oh(C)
    for p in POL:
        typed = {}
        if p == 'BOTH-RETYPE-T':                                              # 16 Nov: the retyped kept inferences' wordings (label-free) type NEW writes by class pair
            RTT = collections.defaultdict(collections.Counter)
            for c in KEPT[p].values(): RTT[(TY.get(c[0], 'X'), TY.get(c[2], 'X'))][c[1]] += 1
        for (A, C), e in W.items():
            hd, tl = (A, C) if DG.get(A, 0) <= DG.get(C, 0) else (C, A); cp = (TY.get(hd, 'X'), TY.get(tl, 'X')); votes = collections.Counter()
            if p == 'BOTH-RETYPE-T':
                f_, b_ = RTT.get((TY.get(hd, 'X'), TY.get(tl, 'X'))), RTT.get((TY.get(tl, 'X'), TY.get(hd, 'X')))
                nf, nb_ = (sum(f_.values()) if f_ else 0), (sum(b_.values()) if b_ else 0)
                if nf + nb_ >= 3:
                    typed[(A, C)] = (hd, f_.most_common(1)[0][0], tl) if nf >= nb_ else (tl, b_.most_common(1)[0][0], hd); continue
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
        def typed_ok(k):                                                       # 16 Nov: confirmations > contradictions among head-first records of the
            c = typed.get(k)                                                   #   judge's window; a second name the store links to the tail is no contradiction
            if c is None: return False
            zs = HF.get(c[0], ()); cf = sum(1 for z in zs if z == c[2]); ct = sum(1 for z in zs if z != c[2] and z not in nb.get(c[2], ()))
            return cf > ct
        if p == 'JUDGE': keep = [k for k in W if tier[k] and judged(k)]
        elif p == 'USEFUL': keep = [k for k in W if useful(k)]
        elif p == 'JUDGE-ALL': keep = [k for k in W if judged(k)]                # 16 Nov: the judge over all 400 written (BOTH's candidate set)
        elif p == 'JUDGE+USE': keep = [k for k in W if (tier[k] and judged(k)) or (not tier[k] and useful(k))]
        elif p in ('SEL-BOTH', 'RULE-BOTH'):                                  # 16 Nov: 200 chosen by the rule (top ranks) or by a selector trained on earlier
            if p == 'RULE-BOTH' or not SELX: cand = [k for k in W if tier[k]] #   rounds' USEFULNESS (label-free); then judge AND useful as usual
            else:
                from sklearn.linear_model import LogisticRegression
                Xa, ya = np.array(SELX), np.array(SELY); mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
                clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit((Xa - mu) / sd, ya)
                ks = list(W); sc = clf.decision_function((np.array([fx(k) for k in ks]) - mu) / sd); cand = [ks[i] for i in np.argsort(-sc)[:200]]
            keep = [k for k in cand if judged(k) and useful(k)]
            if p == 'SEL-BOTH': print(f"      {p}: the 200 written right {np.mean([y[k] for k in cand]):.3f} (rule's top 200 {np.mean([y[k] for k in W if tier[k]]):.3f}); overlap with the rule's {sum(tier[k] for k in cand)}", flush=True)
        elif p == 'BOTH-ADAPT':                                                # 16 Nov ADAPTIVE BUDGET: write in chunks of 100 by rank; stop when under
            keep = []                                                          #   LS_STOP of a chunk passes judge AND usefulness
            ADAPT_END = 400
            for c0 in range(0, 400, 100):
                ch = [k for k in W if c0 <= W[k]['rank'] < c0 + 100]; kk = [k for k in ch if judged(k) and useful(k)]; keep += kk
                if len(kk) < float(os.environ.get('LS_STOP', 0.1)) * max(1, len(ch)): ADAPT_END = c0 + 100; break
        elif p == 'BOTH-NAME': keep = [k for k in W if judged(k) and useful_name(k)]
        elif p == 'USEFUL-NAME': keep = [k for k in W if useful_name(k)]
        elif p == 'BOTH-RETRACT': keep = [k for k in W if judged(k) and useful(k)]
        elif p in ('BOTH-RETYPE', 'BOTH-RETYPE-T'): keep = [k for k in W if judged(k) and useful(k)]
        elif p == 'BOTH': keep = [k for k in W if judged(k) and useful(k)]
        elif p == 'BOTH-OR-CC': keep = [k for k in W if judged(k) and (useful(k) or typed_ok(k))]   # 16 Nov (queue 35): coverage for the keep rule
        elif p == 'JUDGE-CC': keep = [k for k in W if judged(k) and typed_ok(k)]
        elif p == 'JUDGE-NOHARM':                                              # 16 Nov JUDGE AND NOT HARMFUL: drop only judged pairs MEASURED as not useful;
            def harmful(k):                                                    #   pairs with no later span of the head pass on the judge alone
                c = typed.get(k)
                return c is not None and bool(first.get(c[0])) and bits([c], first[c[0]], TXT, PT, EN) <= 0
            keep = [k for k in W if judged(k) and not harmful(k)]
            print(f"      {p}: judged {sum(judged(k) for k in W)}, of them no feedback {sum(judged(k) and not (typed.get(k) and first.get(typed[k][0])) for k in W)}", flush=True)
        elif p == 'EITHER': keep = [k for k in W if judged(k) or useful(k)]
        elif p == 'EITHER-BATCH': keep = [k for k in W if tier[k] and (judged(k) or useful(k))]
        elif p.startswith('USEFUL>'): keep = [k for k in W if useful(k, float(p[7:]))]
        else: keep = [k for k in W if tier[k]]
        for k in keep:
            if k in typed: KEPT[p][k] = typed[k]
            if p in ('BOTH-RETYPE', 'BOTH-RETYPE-T') and conf.get(k):                             # 16 Nov RETYPE ON CONFIRMATION: the kept inference rewritten in the wording of
                X, Y, w = collections.Counter(conf[k]).most_common(1)[0][0]    #   the later relational sentence that states it
                short_ = len(w.split()) <= 4 and not any(ch in w for ch in ':()[];"')
                if w and any(ch.isalpha() for ch in w) and (short_ or not int(os.environ.get('LS_RT_SHORT', 0))): KEPT[p][k] = (X, w, Y)
            if conf.get(k):                                                    # the composition table learns from kept, confirmed writes (no labels)
                X, Y, w = collections.Counter(conf[k]).most_common(1)[0][0]; cf = w if X == k[0] else w + ' ^R'
                for b in W[k]['prem']:
                    a1, a2 = orient(k[0], b), orient(b, k[1])
                    if a1 and a2: T3[p][(a1, a2)][cf] += 1
        nret = 0
        if p == 'BOTH-RETRACT':                                                # 16 Nov RETRACTION BY USEFULNESS: every older memory entry re-measured on this
            for k in [k for k in KEPT[p] if k not in W]:                       #   round's first-half text; dropped if it now costs bits (label-free)
                c = KEPT[p][k]
                if first.get(c[0]) and bits([c], first[c[0]], TXT, PT, EN) < 0: del KEPT[p][k]; nret += 1
        allk = list(KEPT[p]); b2 = bits([KEPT[p][k] for k in allk], second, TXT, PT, EN)
        yk = [y[k] for k in keep]; yall = [pm._PM['rti'][k]['truth'] for k in allk]
        if p in ('BOTH-RETYPE', 'BOTH-RETYPE-T'):                             # typed accuracy of ALL this round's new writes as typed at write time
            tw_ = [typed[k] for k in W if k in typed]; out.append(f"[{p} new writes typed at write time: exact {np.mean([tright(c, 0) for c in tw_]):.3f} / 'is in' {np.mean([tright(c, 1) for c in tw_]):.3f}]")
        mm = list(KEPT[p].values()); te = np.mean([tright(c, 0) for c in mm]) if mm else float('nan'); tg = np.mean([tright(c, 1) for c in mm]) if mm else float('nan')
        out.append(f"{p}: kept {len(keep)} (right {np.mean(yk) if yk else float('nan'):.3f}){f'; retracted {nret}' if p == 'BOTH-RETRACT' else ''} | memory {len(allk)} (right {np.mean(yall) if yall else float('nan'):.3f}; typed claims right "
                   f"exact {te:.3f} / 'is in' {tg:.3f}) saves {b2:+.2f}")
    if int(os.environ.get('LS_DEFER', 0)) and 'BOTH-ADAPT' in POL:            # 16 Nov TRUE ADAPTIVE WRITES: candidates past the stop were never written -
        dfr = [k for k in W if W[k]['rank'] >= ADAPT_END]                      #   they leave the ledger and may be proposed again, with more evidence
        for k in dfr: del pm._PM['rti'][k]
        out.append(f"[deferred {len(dfr)} unwritten candidates (stop at rank {ADAPT_END})]")
    if any(p == 'SEL-BOTH' for p in POL):                                     # this round's usefulness labels for the next rounds' selector (typed by the
        typed = {}                                                             #   class pair, as the rounds start)
        for (A, C), e in W.items():
            hd, tl = (A, C) if DG.get(A, 0) <= DG.get(C, 0) else (C, A); cp = (TY.get(hd, 'X'), TY.get(tl, 'X'))
            if tab.get(cp): (w, fw), _ = tab[cp].most_common(1)[0]; typed[(A, C)] = (hd, w, tl) if fw else (tl, w, hd)
        for k in W:
            c = typed.get(k); u = bool(c is not None and first.get(c[0]) and bits([c], first[c[0]], TXT, PT, EN) > 0)
            SELX.append(fx(k)); SELY.append(u)
    if int(os.environ.get('LS_STAB', 0)):                                      # 16 Nov USEFULNESS STABILITY: each round every written inference so far is
        for (A, C), e in W.items():                                            #   re-measured on this round's first-half text (as the GPU re-measures at every
            hd, tl = (A, C) if DG.get(A, 0) <= DG.get(C, 0) else (C, A); cp = (TY.get(hd, 'X'), TY.get(tl, 'X'))   #   visit); keep rule on the LATEST measure
            if tab.get(cp):                                                    #   vs the CUMULATIVE one (bits summed over every round since the write)
                (w, fw), _ = tab[cp].most_common(1)[0]; ALLW[(A, C)] = dict(c=(hd, w, tl) if fw else (tl, w, hd), y=bool(e['truth']), j=bool(conf.get((A, C))), hist=[], prev=None)
        for k, a in ALLW.items():
            c = a['c']; a['hist'].append(bits([c], first[c[0]], TXT, PT, EN) if first.get(c[0]) else None)
        J = [a for a in ALLW.values() if a['j']]; flips = 0; nlast = ncum = 0; ylast = []; ycum = []
        for a in J:
            ms = [b for b in a['hist'] if b is not None]; last = (ms[-1] > 0) if ms else False; cum = (sum(ms) > 0) if ms else False
            if a['prev'] is not None and a['prev'] != last: flips += 1
            a['prev'] = last
            if last: ylast.append(a['y'])
            if cum: ycum.append(a['y'])
        out.append(f"[STABILITY over {len(J)} judged inferences so far (right {np.mean([a['y'] for a in J]) if J else float('nan'):.3f}): LATEST measure passes {len(ylast)} right {np.mean(ylast) if ylast else float('nan'):.3f}, "
                   f"decision flips since last round {flips} | CUMULATIVE passes {len(ycum)} right {np.mean(ycum) if ycum else float('nan'):.3f} | measured this round {sum(1 for a in J if a['hist'][-1] is not None)}]")
    P(f"ROUND {rd} (visit {v + 1}): written {len(W)} (batch right {np.mean([y[k] for k in W if tier[k]]):.3f}, next tier {np.mean([y[k] for k in W if not tier[k]]):.3f}) | "
      f"{len(spans)} later spans || " + " || ".join(out) + f"  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
