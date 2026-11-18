"""COMPOSE WITH LEARNED LABEL-FREE MAPS, THEN CONFIRM (17 Nov 2026; tools/latent_confirm.py) - offline, distant set, label-free (distant
labels for evaluation only). tools/latent_compose.py E3: label-free relation maps compose real-text chains at 2.7-3.8x chance (Re-DocRED);
tools/compose_confirm.py: confirmation by later reading lifts composed pairs about 4x. Together: for every chain A -k1-> B -k2-> C of
co-occurrence records in a held-out document (k = the coarse frame key), the map's prediction C_hat = the document entity nearest to
W_k2 key(B) (memory hop 1, latent hop 2), with its margin; conclusion (A, C_hat) if A-C_hat is not a record in the document.
Precision by the distant labels for: base (random within-document pairs never recorded together), every chain's C (no map), the MAP'S
choice C_hat, the map's choice in its confident quarter, and each CONFIRMED by a later document (any wording / relational last word).
LF_JOINT 1 (18 Nov): name keys learned jointly with the maps (label-free; latent_compose E4). Also: confirmed by 2+ later documents.
Usage: TG_D=192 TG_CKPT=<model> python3 tools/latent_confirm.py <distant.json.gz>   env LF_TRAIN (0.5) LF_LAM (1.0) LF_DOCS (10000) LF_JOINT (0) LF_JSTEPS (3000)"""
import os, sys, math, time, types, collections
import numpy as np
import torch
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(here))
DIST = sys.argv[1]
import platypus.inference.premises as pm
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=DIST, inf_rret_list2=0))
R, TY, LAB, ND = pm._PM['R'], pm._PM['TY'], pm._PM['LAB'], pm._PM['ND']
ND = min(ND, int(os.environ.get('LF_DOCS', 10000))); R = [r for r in R if r[0] < ND]
cut = int(ND * float(os.environ.get('LF_TRAIN', 0.5))); LAM = float(os.environ.get('LF_LAM', 1.0)); GLUE = {'and', 'or', 'the', 'a', 'an', 'with', 'to'}
sys.argv = [sys.argv[0], '6000', '5']
src = open(os.path.join(here, 'trunk_guides.py')).read(); g = {}
exec(compile(src[:src.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'), g)
lm, run = g['lm'], g['run']; torch.set_num_threads(2)
names = sorted({x for r in R for x in (r[1], r[2])}); NI = {n: i for i, n in enumerate(names)}
K = np.zeros((len(names), lm.o.in_features), np.float32)
for i, n in enumerate(names):
    _, H = run(lm, (' ' + n).encode('utf-8', 'replace')); K[i] = H[1:].mean(0)
    if (i + 1) % 20000 == 0: P(f"   name keys {i + 1} of {len(names)}  ({time.time() - T0:.0f}s)")
K -= K.mean(0); K /= np.linalg.norm(K, axis=1, keepdims=True) + 1e-8; D_ = K.shape[1]
P(f"{ND} docs ({cut} training), {len(R)} records, {len(names)} name keys  ({time.time() - T0:.0f}s)")
key = lambda a, b, lw, o: (TY.get(a, 'X'), lw, TY.get(b, 'X'), o)
by_doc = collections.defaultdict(lambda: collections.defaultdict(set)); seen_docs = collections.defaultdict(set); rel_docs = collections.defaultdict(set)
for r in R:
    if r[5] > 12: continue
    by_doc[r[0]][(r[1], r[2])].add(key(r[1], r[2], r[4], 'f')); by_doc[r[0]][(r[2], r[1])].add(key(r[2], r[1], r[4], 'r'))
    p_ = frozenset((r[1], r[2])); seen_docs[p_].add(r[0])
    if r[4].isalpha() and r[4] not in GLUE: rel_docs[p_].add(r[0])
XY = collections.defaultdict(list)
for d in range(cut):
    for (a, b), ks in by_doc[d].items():
        for k in ks: XY[k].append((NI[a], NI[b]))
W = {}
for k, v in XY.items():
    if len(v) < 50: continue
    X = K[[a for a, _ in v]]; Y = K[[b for _, b in v]]; W[k] = np.linalg.solve(X.T @ X + LAM * np.eye(D_), X.T @ Y)
P(f"{len(W)} label-free maps (keys with 50+ training records)  ({time.time() - T0:.0f}s)")
if int(os.environ.get('LF_JOINT', 0)):                                         # 18 Nov: LEARNED KEYS jointly with the maps (as latent_compose E4)
    import torch.nn.functional as F_
    ks = sorted(W); kix = {k: i for i, k in enumerate(ks)}
    TRP = np.array([(NI[a], kix[k], NI[b]) for d in range(cut) for (a, b), kk in by_doc[d].items() for k in kk if k in kix])
    Et = torch.nn.Parameter(torch.tensor(K)); Wt = torch.nn.Parameter(torch.tensor(np.stack([W[k] for k in ks])).float()); K0 = torch.tensor(K)
    opt = torch.optim.Adam([Et, Wt], lr=1e-3); rsj = np.random.RandomState(5)
    for st in range(int(os.environ.get('LF_JSTEPS', 3000))):
        b_ = TRP[rsj.randint(0, len(TRP), 512)]; A_ = F_.normalize(Et[b_[:, 0]], dim=-1); Bv = F_.normalize(Et[b_[:, 2]], dim=-1)
        pred = F_.normalize(torch.einsum('nd,nde->ne', A_, Wt[b_[:, 1]]), dim=-1)
        loss = F_.cross_entropy(pred @ Bv.t() / 0.1, torch.arange(len(b_))) + ((Et[b_[:, 0]] - K0[b_[:, 0]]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 1000 == 0: P(f"   joint step {st}: loss {loss.item():.3f}  ({time.time() - T0:.0f}s)")
    with torch.no_grad(): K = F_.normalize(Et, dim=-1).numpy(); W = {k: Wt[kix[k]].detach().numpy() for k in ks}
    P(f"LEARNED KEYS jointly with the maps ({len(TRP)} training records)  ({time.time() - T0:.0f}s)")
res = collections.defaultdict(list); margins = []; mapped = []; meta = []
for d in range(cut, ND):
    pr = by_doc[d]; ents = sorted({x for p in pr for x in p})
    if len(ents) < 3: continue
    EK = K[[NI[e] for e in ents]]; EI = {e: i for i, e in enumerate(ents)}; byh = collections.defaultdict(list)
    for (a, b), ks in pr.items():
        for k in ks: byh[a].append((k, b))
    done = set()
    for (a, b), ks in pr.items():
        k1_ = sorted(ks)[0]
        for k2, c in byh.get(b, ()):
            if c == a or (a, c) in pr: continue
            res['every chain C (no map)'].append(frozenset((a, c)) in LAB)
            if k2 not in W or (a, b, k2) in done: continue
            done.add((a, b, k2)); v = K[NI[b]] @ W[k2]; s = EK @ (v / (np.linalg.norm(v) + 1e-8)); s[EI[a]] = -9; s[EI[b]] = -9
            o = np.argsort(-s); ch = ents[o[0]]
            if (a, ch) in pr: continue
            p_ = frozenset((a, ch)); y = p_ in LAB; m = float(s[o[0]] - s[o[1]])
            later = any(x > d for x in seen_docs.get(p_, ())); later_rel = any(x > d for x in rel_docs.get(p_, ()))
            two = sum(1 for x in rel_docs.get(p_, ()) if x > d) >= 2
            mapped.append((y, m, later, later_rel, two)); meta.append((d, (k1_, k2), p_))
    rs = np.random.RandomState(d)
    for _ in range(min(20, len(ents))):
        i, j = rs.randint(len(ents)), rs.randint(len(ents))
        if i != j and (ents[i], ents[j]) not in pr: res['base: random pair never recorded together'].append(frozenset((ents[i], ents[j])) in LAB)
M = np.array(mapped, dtype=float); q = np.quantile(M[:, 1], 0.75)
def show(nm, v): P(f"   {nm:58s} precision {np.mean(v) if len(v) else float('nan'):.3f} (n {len(v)})")
show('base: random pair never recorded together', res['base: random pair never recorded together'])
show('every chain C (no map)', res['every chain C (no map)'])
show("MAP'S CHOICE C_hat (memory hop 1 + W_k2 key(B))", M[:, 0])
show("MAP'S CHOICE, confident quarter", M[M[:, 1] >= q, 0])
show("MAP'S CHOICE + CONFIRMED later (any wording)", M[M[:, 2] > 0, 0])
show("MAP'S CHOICE + CONFIRMED later (relational wording)", M[M[:, 3] > 0, 0])
show("MAP'S CHOICE, confident quarter + CONFIRMED (relational)", M[(M[:, 1] >= q) & (M[:, 3] > 0), 0])
show("MAP'S CHOICE + CONFIRMED TWICE (relational, 2+ later docs)", M[M[:, 4] > 0, 0])
show("MAP'S CHOICE, confident quarter + CONFIRMED TWICE", M[(M[:, 1] >= q) & (M[:, 4] > 0), 0])
# 18 Nov RELIABILITY LEARNED FROM CONFIRMATION (writable BEFORE confirmation): confirmation cannot filter a write - a twice-confirmed pair has
#   been READ twice. Instead, the confirmation history of each (k1, k2) key pair is learned on a CALIBRATION period (predictions in the first
#   half of the held-out documents, confirmed by documents before the TEST period starts), and new predictions in the TEST period are written
#   if their key pair's past confirmation rate is high - judged by the distant labels, with no confirmation of their own.
cut2 = cut + (ND - cut) // 2; cal = collections.defaultdict(lambda: [0, 0])
for (y, m, la, lr, tw), (d, u, p_) in zip(mapped, meta):
    if d < cut2: cal[u][0] += any(d < x < cut2 for x in rel_docs.get(p_, ())); cal[u][1] += 1
test = [(i, cal[u][0] / cal[u][1] if cal[u][1] >= 5 else None) for i, (d, u, p_) in enumerate(meta) if d >= cut2]
P(f"RELIABILITY FROM CONFIRMATION: calibration predictions {sum(v[1] for v in cal.values())} in docs {cut}-{cut2}, {sum(1 for v in cal.values() if v[1] >= 5)} key pairs with 5+; "
  f"test predictions {len(test)} in docs {cut2}-{ND}")
Mt = M[[i for i, _ in test]]; rr = np.array([r if r is not None else -1 for _, r in test]); qt = np.quantile(Mt[:, 1], 0.75)
show("TEST: map's choice (all)", Mt[:, 0]); show("TEST: confident quarter", Mt[Mt[:, 1] >= qt, 0])
for thr in (0.05, 0.10, 0.20, 0.30):
    show(f"TEST: key pair confirmed >= {thr:.2f} in calibration", Mt[rr >= thr, 0]); show(f"TEST: ... and confident quarter", Mt[(rr >= thr) & (Mt[:, 1] >= qt), 0])
P(f"DONE ({time.time() - T0:.0f}s)")
