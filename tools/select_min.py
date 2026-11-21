"""SELECT BEFORE TEACHING (21 Nov 2026; tools/select_min.py) - offline, distant set, the package's own writes; A6 stand-in trunk for part B.
kra21 GPU visit 6: RE-CONFIRMATION retracted 72 confirmed inferences at right 0.82 vs 0.84 kept - not selective, and with unlikelihood rows it would
unlearn about 4 right inferences per wrong one (offline, tools/retract_min.py, it was already about 2 right per wrong; we judged it by the precision
of what was left, not by the cost of what it removed). This tool compares rules that act BEFORE teaching, scored by what matters: WRONG REMOVED per
RIGHT LOST on the confirmed set (kra21's top-quarter rule), not only the precision kept.
PART A (documents only): writes from the first SM_FRAC of the documents, confirmation on the next slice (to SM_CONF, top-quarter record), then
  teach-gates: a SECOND LOOK in a delay window (to SM_DELAY: a relational record of the pair again, any / rated), more evidence in the confirming
  window (two distinct documents, two rated records, the best record's rating), and their combinations.
PART B (the model's own signal, A6 stand-in): the stand-in stores the confirmed inferences' premises (PRE steps, KL anchor), then learns the
  conclusions of HALF the confirmed set (TEACH steps). For the OTHER half (never taught): OWN vs SWAPPED subject - the answer's mean log-probability
  after 'It is on record that A tw ' minus after the same with another inference's subject (and rank among 10, as kra21's PRIOR CONTROL), before
  and after teaching the first half. kra21 GPU: held-back right +0.09 to +0.15, wrong +0.00 to +0.03. If the margin separates right from wrong
  (AUC), it is a per-inference teach-gate: hold back the lowest-margin share, alone or with part A's best rule.
Usage: TG_CKPT=<A6 trunk> python3 tools/select_min.py <distant_50k.json.gz>   env SM_FRAC 0.4, SM_CONF 0.6, SM_DELAY 0.8, SM_W 2400, SM_PRE 3000,
  SM_TEACH 2000, SM_PARTB 1"""
import os, sys, types, copy, time, math, collections, zlib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time()
FR_ = float(os.environ.get('SM_FRAC', 0.4)); FC_ = float(os.environ.get('SM_CONF', 0.6)); FD_ = float(os.environ.get('SM_DELAY', 0.8)); NW = int(os.environ.get('SM_W', 2400))
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa', inf_rcode_asym=1.0,
                            inf_rcode_selreport=0, inf_rti_max=NW, inf_rret=0, inf_rti_ctl=0.0, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rcode_reader=0.5, inf_premises_seed_ctx=1, inf_rti_typed=1)
pm._load_distant(cfg); ND = pm._PM['ND']; CX = pm._PM.get('CTX') or {}; R = pm._PM['R']; n0, n1, n2 = (int(round(ND * f)) for f in (FR_, FC_, FD_))
y = np.array([bool(r[6]) for r in R]); hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
wtx = lambda r: f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(int(r[5]), 12) // 3} " + ' '.join('C_' + w for w in CX.get(id(r), '').split())
iRD = np.array([i for i, r in enumerate(R) if r[0] < n0]); sd = iRD[np.random.RandomState(4 + 707).permutation(len(iRD))[:20000]]
c = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); c.fit(hv.transform([wtx(R[i]) for i in sd]), y[sd])
score = np.concatenate([c.decision_function(hv.transform([wtx(R[i]) for i in range(j, min(len(R), j + 200000))])) for j in range(0, len(R), 200000)])
RD = [R[i] for i in iRD]; pm._PM['rdscore'] = score[iRD]; pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], RD, 3)
W = pm._PM['rti']; K = list(W); T = {k: bool(W[k]['truth']) for k in K}
q50, q75, q90 = (float(np.quantile(score[iRD], q)) for q in (0.5, 0.75, 0.9))
P(f"{len(RD)} records read ({n0} of {ND} documents); {len(K)} written (right {np.mean(list(T.values())):.3f}); confirm on {n0}-{n1}, delay window {n1}-{n2}  ({time.time() - T0:.0f}s)")
FK = {frozenset(k): k for k in K}; EC = collections.defaultdict(list); ED = collections.defaultdict(list)
need = {frozenset((a, b)) for (a, c_), w in W.items() for b in (w.get('prem') or [])[:3]} | {frozenset((b, c_)) for (a, c_), w in W.items() for b in (w.get('prem') or [])[:3]}
PW = collections.defaultdict(collections.Counter)
for i, r in enumerate(R):
    rel = 1 <= r[5] <= 8 and not pm._rret_listlike(r); fk = frozenset((r[1], r[2]))
    if r[0] < n0:
        if rel and fk in need and any(ch.isalpha() for ch in r[3]): PW[fk][(r[1], pm._rnorm(r[3]), r[2])] += 1
        continue
    if not rel or fk not in FK or r[0] >= n2: continue
    (EC if r[0] < n1 else ED)[FK[fk]].append((int(r[0]), float(score[i])))
del R
CS = [k for k in K if any(s >= q75 for d, s in EC.get(k, []))]                 # kra21's confirmation: one record in the reader's top quarter
nr = sum(T[k] for k in CS); nw = len(CS) - nr
P(f"CONFIRMED (top quarter, kra21): {len(CS)} (right {nr / max(len(CS), 1):.3f}; {nw} wrong)")
def rep(name, keep):
    kp = [k for k in CS if keep(k)]; rm = [k for k in CS if not keep(k)]; rw = sum(not T[k] for k in rm); rg = sum(T[k] for k in rm)
    P(f"   TEACH ONLY IF {name:58s}: taught {len(kp):4d} right {np.mean([T[k] for k in kp]) if kp else float('nan'):.3f} ({sum(T[k] for k in kp)} right) | "
      f"wrong removed {rw / max(nw, 1):.2f} | right lost {rg / max(nr, 1):.2f} | RIGHT LOST PER WRONG REMOVED {rg / rw if rw else float('nan'):.2f}")
    return rm
EV = lambda k: EC.get(k, []); DL = lambda k: ED.get(k, [])
P("PART A - documents only (base rate: removing at random loses %.2f right per wrong)" % (nr / max(nw, 1)))
rep('(all confirmed: kra21)', lambda k: True)
rep('2+ distinct documents confirm (any rating)', lambda k: len({d for d, s in EV(k)}) >= 2)
rep('2+ distinct documents, one top quarter + one top half', lambda k: len({d for d, s in EV(k) if s >= q50}) >= 2)
rep('2+ records rated top quarter', lambda k: sum(s >= q75 for d, s in EV(k)) >= 2)
rep('best confirming record in the top tenth', lambda k: max(s for d, s in EV(k)) >= q90)
rep('SECOND LOOK: a relational record again in the delay window', lambda k: len(DL(k)) >= 1)
rep('SECOND LOOK: a top-half record again in the delay window', lambda k: any(s >= q50 for d, s in DL(k)))
rep('SECOND LOOK: a top-quarter record again in the delay window', lambda k: any(s >= q75 for d, s in DL(k)))
rep('2+ documents confirm OR a second look (top half)', lambda k: len({d for d, s in EV(k)}) >= 2 or any(s >= q50 for d, s in DL(k)))
rep('mean rating of all records (confirm + delay) above the top-quarter line', lambda k: np.mean([s for d, s in EV(k) + DL(k)]) >= q75)
P(f"PART A done ({time.time() - T0:.0f}s)")
if os.environ.get('SM_DUMP'):                                                  # a sample for a hand audit of 'wrong' (distant labels are incomplete: the true rate)
    import json
    def pr0(X, Y):
        cc = PW.get(frozenset((X, Y)))
        return ' '.join(max(cc.items(), key=lambda kv: kv[1])[0]) if cc else None
    rd = np.random.RandomState(12); wr = [k for k in CS if not T[k]]; rt = [k for k in CS if T[k]]
    smp = [wr[j] for j in rd.permutation(len(wr))[:60]] + [rt[j] for j in rd.permutation(len(rt))[:30]]
    json.dump([dict(claim=f"{W[k].get('th')} {W[k].get('tw')} {W[k].get('tt')}", label_right=T[k], pair=list(k),
                    premises=[x for b in (W[k].get('prem') or [])[:2] for x in (pr0(W[k].get('th') or k[0], b), pr0(b, W[k].get('tt') or k[1])) if x])
               for k in smp], open(os.environ['SM_DUMP'], 'w'), indent=1)
    P(f"DUMPED {len(smp)} confirmed inferences ({min(60, len(wr))} labelled wrong, {min(30, len(rt))} right) to {os.environ['SM_DUMP']}")
if not int(os.environ.get('SM_PARTB', 1)): P(f"DONE ({time.time() - T0:.0f}s)"); sys.exit()
# ---------------- PART B: the model's own signal (A6 stand-in) ----------------
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2); torch.manual_seed(0)
PRE = int(os.environ.get('SM_PRE', 3000)); TEACH = int(os.environ.get('SM_TEACH', 2000)); LR = 1e-3
def prem(X, Y):
    cc = PW.get(frozenset((X, Y)))
    if not cc: return None
    c2 = [(v, kk) for kk, v in cc.items() if kk[0] == X and kk[2] == Y] or [(v, kk) for kk, v in cc.items()]; return max(c2)[1]
IT = []
for k in CS:
    w = W[k]
    if not w.get('tw') or len(f"It is on record that {w['th']} {w['tw']} ") >= 90: continue
    ps = [p for b in (w.get('prem') or [])[:3] for p in (prem(w['th'], b), prem(b, w['tt'])) if p]
    IT.append(dict(k=k, th=w['th'], tw=w['tw'], tt=w['tt'], right=T[k], ps=ps, half=zlib.crc32(repr(k).encode()) % 2))
H1 = [e for e in IT if e['half'] == 0]; H2 = [e for e in IT if e['half'] == 1]
P(f"PART B: {len(IT)} confirmed typed inferences ({sum(bool(e['ps']) for e in IT)} with a premise on record); TEACH half {len(H1)} (right {np.mean([e['right'] for e in H1]):.3f}), "
  f"SCORE half {len(H2)} (right {np.mean([e['right'] for e in H2]):.3f}, {sum(not e['right'] for e in H2)} wrong)")
DW = 192; L_ = 128; PAD = 256; V_ = 257
class Blk(nn.Module):
    def __init__(s, d, hh):
        super().__init__(); s.a = nn.MultiheadAttention(d, hh, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        q = s.n1(x); x = x + s.a(q, q, q, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class LM(nn.Module):
    def __init__(s, d=DW, nl=3):
        super().__init__(); s.emb = nn.Embedding(V_, d); s.pos = nn.Parameter(torch.randn(L_, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl)); s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V_)
    def forward(s, X):
        Tn = X.shape[1]; am = torch.triu(torch.ones(Tn, Tn, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:Tn]
        for b in s.b: x = b(x, am)
        return s.o(s.n(x))
base = LM(); base.load_state_dict(torch.load(os.environ['TG_CKPT'])); base.eval()
MP = os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile')
DOMS = {d: np.frombuffer(open(f'{MP}/{d}.bin', 'rb').read().replace(b'\n', b' '), np.uint8) for d in ('wiki', 'prose', 'arxiv', 'pubmed', 'law')}
enc = lambda s: s.encode('latin-1', 'replace')[:L_]
def tens(bl):
    bl = [b[:L_] for b in bl]; Tn = max(len(b) for b in bl); X = torch.full((len(bl), Tn), PAD, dtype=torch.long)
    for i, b in enumerate(bl): X[i, :len(b)] = torch.tensor(list(b))
    return X, torch.cat([torch.full((len(bl), 1), PAD), X[:, :-1]], 1)
FR = ("{h} {w} {t}.", "In short, {h} {w} {t}.", "Notes: {h} {w} {t}.", "As recorded, {h} {w} {t}.", "From the notes: {h} {w} {t}.", "{h}, as noted, {w} {t}.",
      "It is known that {h} {w} {t}.", "We read that {h} {w} {t}.", "Sources say {h} {w} {t}.", "{h} - {w} {t}.", "Reportedly, {h} {w} {t}.", "Note that {h} {w} {t}.")
PF = ("{a} {w} {b}.", "Notes: {a} {w} {b}.", "We read that {a} {w} {b}.", "It is known that {a} {w} {b}.", "Fact: {a} {w} {b}.", "Reportedly, {a} {w} {b}.")
PREM = [enc(f.format(a=p[0], w=p[1], b=p[2])) for e in IT for p in e['ps'] for f in PF] or [enc("Notes: nothing.")]
CON = [enc(f.format(h=e['th'], w=e['tw'], t=e['tt'])) for e in H1 for f in FR]  # 'It is on record that' is never a training frame
mn = min(len(a) for a in DOMS.values()) - 200400
EVD = {d: [bytes(a[len(a) - 200000 + k * 129:len(a) - 200000 + (k + 1) * 129]) for k in range(64)] for d, a in DOMS.items()}
@torch.no_grad()
def dbits(m):
    return {d: float(F.cross_entropy(m(tens(e)[1]).reshape(-1, V_), tens(e)[0].reshape(-1), ignore_index=PAD)) / math.log(2) for d, e in EVD.items()}
def step(m, op, rs, fb):
    ds_ = list(DOMS); wb = [bytes(DOMS[ds_[j % 5]][i:i + 129]) for j, i in enumerate(rs.randint(0, mn, 24))]
    Xf, Xfi = tens(fb); lf = F.cross_entropy(m(Xfi).reshape(-1, V_), Xf.reshape(-1), ignore_index=PAD)
    Xw, Xwi = tens(wb)
    with torch.no_grad(): lo = F.log_softmax(base(Xwi), -1)
    lw = F.kl_div(F.log_softmax(m(Xwi), -1), lo, log_target=True, reduction='batchmean') / Xw.shape[1]
    loss = (8 * lf + 24 * lw) / 32; op.zero_grad(); loss.backward(); op.step()
rr = np.random.RandomState(3); TAILS = sorted({e['tt'] for e in IT})
CAND = {id(e): [e['tt']] + [x for x in (TAILS[j] for j in rr.choice(len(TAILS), 14, replace=False)) if x != e['tt']][:9] for e in IT}
SWP = {id(e): H2[(i + 1) % len(H2)]['th'] for i, e in enumerate(H2)}
@torch.no_grad()
def lps(m, pref, cands):
    pb = enc(pref); bl = [(pb + x.encode('latin-1', 'replace') + b'.')[:L_] for x in cands]
    X, Xi = tens(bl); lp = F.log_softmax(m(Xi), -1); n0_ = len(pb)
    return np.array([float(lp[i, n0_:len(b)].gather(-1, X[i, n0_:len(b), None]).mean()) for i, b in enumerate(bl)])
def signal(m, tag):
    mg, ro, rsw, ow = [], [], [], []
    for e in H2:
        a = lps(m, f"It is on record that {e['th']} {e['tw']} ", CAND[id(e)]); b = lps(m, f"It is on record that {SWP[id(e)]} {e['tw']} ", CAND[id(e)])
        mg.append(a[0] - b[0]); ow.append(a[0]); ro.append(int(np.argmax(a) == 0)); rsw.append(int(np.argmax(b) == 0))
    mg, ro, rsw = np.array(mg), np.array(ro), np.array(rsw); yy = np.array([e['right'] for e in H2])
    d = ro - rsw
    sg = lambda v: f"{v.mean():+.3f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.3f})" if len(v) > 1 else "nan"
    P(f"   {tag}: OWN - SWAPPED rank among 10: right {sg(d[yy])} (own {ro[yy].mean():.2f}) / wrong {sg(d[~yy])} (own {ro[~yy].mean():.2f}) || log-prob margin: right {sg(mg[yy])} / "
      f"wrong {sg(mg[~yy])} | AUC right vs wrong: margin {roc_auc_score(yy, mg):.3f}, own log-prob alone {roc_auc_score(yy, np.array(ow)):.3f}  ({time.time() - T0:.0f}s)")
    return mg
d0 = dbits(base)
m = copy.deepcopy(base); rs = np.random.RandomState(10); op = torch.optim.Adam(m.parameters(), lr=LR); m.train()
for st in range(PRE): step(m, op, rs, [PREM[i] for i in rs.randint(0, len(PREM), 8)])
m.eval(); dd = dbits(m); P(f"PREMISES FIRST, {PRE} steps, forgetting mean {np.mean([dd[d] - d0[d] for d in dd]):+.4f}")
mg0 = signal(m, "BEFORE teaching half 1 (score half 2)")
m.train()
for st in range(TEACH): step(m, op, rs, [PREM[i] for i in rs.randint(0, len(PREM), 4)] + [CON[i] for i in rs.randint(0, len(CON), 4)])
m.eval(); dd = dbits(m); P(f"TAUGHT half 1, {TEACH} steps, forgetting mean {np.mean([dd[d] - d0[d] for d in dd]):+.4f}")
mg1 = signal(m, "AFTER teaching half 1 (score half 2, never taught)")
yy = np.array([e['right'] for e in H2]); n2r = int(yy.sum()); n2w = len(yy) - n2r
for nm_, v in (('margin after teaching', mg1), ('margin gain from teaching', mg1 - mg0)):
    for q in (0.1, 0.25):
        cut = np.quantile(v, q); rm = v < cut; rw = int((rm & ~yy).sum()); rg = int((rm & yy).sum())
        P(f"   HOLD BACK the lowest {q:.0%} by {nm_:26s}: wrong removed {rw / max(n2w, 1):.2f} | right lost {rg / max(n2r, 1):.2f} | RIGHT LOST PER WRONG REMOVED "
          f"{rg / rw if rw else float('nan'):.2f} | taught right {yy[~rm].mean():.3f} (base {yy.mean():.3f}, random removal {n2r / max(n2w, 1):.2f} right per wrong)")
P(f"DONE ({time.time() - T0:.0f}s)")
