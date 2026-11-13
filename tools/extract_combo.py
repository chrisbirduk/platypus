"""BOTH TEACHERS TOGETHER (13 Nov 2026; tools/extract_combo.py) - offline, no effect on the package.
tools/consistency_teacher.py (functional / selective wording, corroboration, rule chains; learner top 25% 0.38 related) + the trunk's prediction
feedback of tools/extract_learn.py (bits a record saves where its names meet again, beyond the names alone; learner top 25% 0.31) as one more
vote in the label-free teacher, on the same candidates and held-out documents. Does prediction add to consistency?
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/extract_combo.py"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_ct = open(os.path.join(HERE, 'consistency_teacher.py')).read()
exec(compile(_ct[:_ct.index('# ---- SELF-TRAINING ROUNDS')], 'consistency_teacher', 'exec'))
sys.argv = [sys.argv[0], '6000', '5']
_tg = open(os.path.join(HERE, 'trunk_guides.py')).read()
_g = {}; exec(compile(_tg[:_tg.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'), _g)
import torch, torch.nn.functional as F
lm, L, PAD, V = _g['lm'], _g['L'], _g['PAD'], _g['V']
txt = lambda toks: ' '.join(toks).encode('utf-8', 'replace'); TH = b'Thing'
@torch.no_grad()
def bits(items, bs=128):
    out = np.zeros(len(items))
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; cut = [max(0, len(s) - L) for s, _ in ch]; X = torch.full((len(ch), L), PAD, dtype=torch.long)
        for j, (s, _) in enumerate(ch): X[j, :len(s) - cut[j]] = torch.tensor(list(s[cut[j]:]))
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(lm(Xi), -1)
        nl = -lp.gather(-1, X.clamp(max=V - 1)[..., None])[..., 0] / math.log(2)
        for j, ((s, (a, b)), c0) in enumerate(zip(ch, cut)): out[i + j] = float(nl[j, max(0, a - c0):max(0, b - c0)].sum())
    return out
# where each (canonical) name pair meets again: the second name's sentence position, from the documents' gold mentions (as extract_learn)
occ = collections.defaultdict(list)
for di, d in enumerate(docs):
    if di in TEST: continue
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms)
        for x in range(len(ms)):
            for y_ in range(x + 1, len(ms)):
                if ms[x][2] != ms[y_][2] and ms[x][1] <= ms[y_][0] and cn[ms[x][2]] != cn[ms[y_][2]]:
                    occ[tuple(sorted((cn[ms[x][2]], cn[ms[y_][2]])))].append((di, si, ms[y_][0], ms[y_][1]))
rq = np.random.RandomState(1); FBi = []; SEQ = {k: [] for k in ('w', 'o', 'pw', 'po')}
for k_, i in enumerate(TR):
    c = C[i]; oth = [o for o in occ[tuple(sorted((c['cA'], c['cB'])))] if o[0] != c['di']]
    if not oth: continue
    di2, si2, y0, y1 = oth[rq.randint(len(oth))]; t2 = docs[di2]['sents'][si2]; pre = txt(t2[max(0, y0 - 6):y0])[-36:] + b' '; tg = txt(t2[y0:y1])[:24]
    A, B, M = txt(C[i]['A']), txt(C[i]['B']), txt(C[i]['mid']); FBi.append(k_)
    for k, rec in (('w', A + b' ' + M + b' ' + B), ('o', TH + b' ' + M + b' ' + B), ('pw', A + b' ' + B), ('po', TH + b' ' + B)):
        q = rec.replace(b'  ', b' ') + b' . ' + pre; SEQ[k].append((q + tg, (len(q), len(q) + len(tg))))
t1 = time.time(); Bt = {k: bits(v) for k, v in SEQ.items()}; fb = (Bt['o'] - Bt['w']) - (Bt['po'] - Bt['pw'])
TRK = np.full(len(TR), np.nan); TRK[FBi] = fb
P(f"trunk feedback on {len(FBi)} training records ({time.time() - t1:.0f}s): per-record AUC {auc(fb[ytr[FBi]], fb[~ytr[FBi]]):.3f}")
base_t = rk(a2) + rk(a3) + 2 * a1 + 2 * a4
for w_ in (0.0, 1.0, 2.0, 4.0):
    lab = base_t + w_ * rk(TRK)
    hi = lab >= np.percentile(lab, 75)
    rep(f"LEARNER, consistency + {w_:g} x trunk feedback (top quarter as positives)", LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, hi).decision_function(Xte))
sel = np.array(FBi)
rep("LEARNER from trunk feedback alone, same records (ridge)", Ridge(alpha=10.0).fit(Xtr[sel], np.clip(fb, np.percentile(fb, 1), np.percentile(fb, 99))).predict(Xte))
P(f"DONE ({time.time() - T0:.0f}s)")
