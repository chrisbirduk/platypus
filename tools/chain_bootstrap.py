"""PREMISES THAT CHAIN: BOOTSTRAPPED, RULE-FREE INFERENCE (13 Nov 2026; tools/chain_bootstrap.py) - offline, no effect on the package.
GPU cg27a_prem: no label-free way of choosing single premises beat random for A-C inference (random 0.19, trunk surprise 0.12, type
association 0.08); type association picks relations that do not chain. The literature (bootstrapped relation extraction - DIPRE, Snowball,
Batista et al. 2015; rule confidence under incompleteness - AMIE; corroboration - multi-instance learning; self-training on verified
outputs - STaR) suggests learning WHICH KINDS of premise pair compose, from inferences that later reading confirms, and growing from
trusted seeds with drift control. Here, on the Re-DocRED store (train + dev, gold for checking only):
  KINDS        each record's wording class (record-level compression, tools/record_mdl.py, K 16 M 128, with RM_EXTRA distant documents) and
               the bridge's side; an A-C proposal through bridge B has the kind (class and side of A-B, class and side of B-C)
  CONFIRMED    label-free: A and C are later read together in one sentence somewhere (a record of the store)
  COMPOSABILITY  per kind, the confirmed share of its proposals on TRAINING pairs (half the A-C pairs by hash; Wilson lower bound); applied
               to TEST pairs, which are never used to learn it (their own confirmation is not used)
  POLICIES on test pairs, gold rate where Re-DocRED annotates the pair: all / 2+ bridges / 3+ bridges (seeds) / composability top 10, 25% /
               composability with 2+ bridges / the stand-in trunk's congruence (the gain from A, as cg27a_rti's check) / composability AND
               trunk congruence (two keys) / confirmed by later reading (an inference that waits for the text)
  BUILT ON INFERENCE (hop 2): accepted inferences (composability top 25%, 2+ bridges) used as premises with stored records -> A-D; gold
               rate of hop 2 against hop 1 (does building on inference drift?)
Usage: TG_D=192 TG_CKPT=<model file> RM_EXTRA=30000 python3 tools/chain_bootstrap.py"""
import os
import sys
import math
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('RM_K', '16'); os.environ.setdefault('RM_M', '128'); os.environ.setdefault('RM_R', '1'); os.environ.setdefault('RM_INIT', 'split')
_rm = open(os.path.join(HERE, 'record_mdl.py')).read()
exec(compile(_rm[:_rm.index('KBEST = min(FITS')], 'record_mdl', 'exec'))
gn, gw = FITS[(int(os.environ['RM_K']), int(os.environ['RM_M']))]
import zlib
# ---- gold (checking only) and the store's sentence pairs (label-free confirmation)
GOLDP = set(); ANN = set(); dfreq = collections.Counter()
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    for l in d['labels']: GOLDP.add(frozenset((cn[l['h']], cn[l['t']])))
    ns = set(cn); ANN |= {frozenset((x, z)) for x in ns for z in ns if x != z}
    for n in ns: dfreq[n] += 1
SENT = {}
for i, c in enumerate(C): SENT.setdefault(frozenset((c['cA'], c['cB'])), set()).add(c['di'])
for i in range(N, len(Af)): SENT.setdefault(frozenset((NAMES[Af[i]], NAMES[Bf[i]])), set()).add(('d', int(XD[i - N])))   # + the distant set's sentences
# ---- proposals A-C through a shared name from another document, with their kinds
kind1 = lambda i, B: (int(gw[w[i]]), 0 if C[i]['cB'] == B else 1)               # the record's wording class, and which side the bridge is on
nb = collections.defaultdict(list)
for i, c in enumerate(C): nb[c['cA']].append((c['cB'], c['di'], i)); nb[c['cB']].append((c['cA'], c['di'], i))
PR = collections.defaultdict(lambda: dict(br=set(), kinds=[], docs=set()))
for i, c in enumerate(C):
    for B, Cn in ((c['cA'], c['cB']), (c['cB'], c['cA'])):
        if dfreq[B] > 200: continue
        for A, dA, j in nb[B]:
            if dA == c['di'] or A == Cn: continue
            k1, k2 = kind1(j, B), kind1(i, B); kd = (k1, k2) if k1 <= k2 else (k2, k1)
            ta, tb, tc = int(gn[NI[A]]), int(gn[NI[B]]), int(gn[NI[Cn]]); kt = ('T', min(ta, tc), tb, max(ta, tc))   # coarser kind: the name-type triple
            p = PR[frozenset((A, Cn))]; p['br'].add(B); p['kinds'].append(kd); p['kt'] = p.get('kt', set()) | {kt}; p['docs'] |= {dA, c['di']}
keys = list(PR); P(f"proposals {len(keys)} | verifiable {sum(k in ANN for k in keys)} | gold among verifiable {np.mean([k in GOLDP for k in keys if k in ANN]):.3f}  ({time.time() - T0:.0f}s)")
conf = np.array([bool(SENT.get(k, set()) - PR[k]['docs']) for k in keys])        # read together in a sentence of a document the premises are not from
QR = float(os.environ.get('CB_RELQ', 0.75))                                       # 13 Nov late: CONFIRMED IN A RELATIONAL WORDING - the later sentence's record
SREL = {}                                                                          #   is among the consistency learner's top (1 - CB_RELQ) (label-free)
thr_rel = np.quantile(s_all, QR)
for i, c in enumerate(C):
    if s_all[i] >= thr_rel: SREL.setdefault(frozenset((c['cA'], c['cB'])), set()).add(c['di'])
confr = np.array([bool(SREL.get(k, set()) - PR[k]['docs']) for k in keys])
if os.environ.get('CB_CONF', 'any') == 'rel': conf_any = conf; conf = confr        # composability learned from relational confirmations
tr = np.array([zlib.crc32((repr(sorted(k)) + os.environ.get('SPLIT_SEED', '')).encode()) % 2 == 0 for k in keys])
P(f"confirmed by later reading: {conf.mean():.3f} of proposals (train pairs {conf[tr].mean():.3f})")
# ---- composability per kind, learned on TRAIN pairs only
kc = collections.Counter(); kn = collections.Counter()
for k, t, cf in zip(keys, tr, conf):
    if not t: continue
    for kd in set(PR[k]['kinds']): kn[kd] += 1; kc[kd] += int(cf)
base = conf[tr].mean()
def wilson(s, n, z=1.64):
    if n == 0: return 0.0
    p = s / n; return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
KC = {kd: wilson(kc[kd], kn[kd]) for kd in kn}
comp = np.array([max((KC.get(kd, 0.0) for kd in set(PR[k]['kinds'])), default=0.0) for k in keys])
tc_ = collections.Counter(); tn_ = collections.Counter()                         # composability by the name-type triple (coarser, more examples)
for k, t, cf in zip(keys, tr, conf):
    if t:
        for kt in PR[k]['kt']: tn_[kt] += 1; tc_[kt] += int(cf)
KT = {kt: wilson(tc_[kt], tn_[kt]) for kt in tn_}
compt = np.array([max((KT.get(kt, 0.0) for kt in PR[k]['kt']), default=0.0) for k in keys])
P(f"type-triple kinds {len(tn_)}; top: " + ", ".join(f"{kt[1:]} {tc_[kt]}/{tn_[kt]}" for kt in sorted(KT, key=lambda q: -KT[q])[:6]))
nbr = np.array([len(PR[k]['br']) for k in keys])
P(f"kinds {len(kn)} learned on train pairs (base confirmed {base:.3f}); top kinds: " + ", ".join(f"{kd} {kc[kd]}/{kn[kd]}" for kd in sorted(KC, key=lambda q: -KC[q])[:6]))
TE = np.where(~tr & np.array([k in ANN for k in keys]))[0]; yte = np.array([keys[i] in GOLDP for i in TE])
P(f"TEST pairs verifiable {len(TE)}, gold {yte.mean():.3f}")
rq = np.random.RandomState(7)
def pol(tag, mask=None, score=None, frac=None):
    ix = np.arange(len(TE)) if mask is None else np.where(mask[TE])[0]
    if score is not None and frac is not None:
        s = score[TE][ix]; ix = ix[np.argsort(-s + 1e-9 * rq.rand(len(s)))[:max(1, int(len(ix) * frac))]]
    P(f"   {tag:62s} {len(ix):6d} inferences, gold {yte[ix].mean() if len(ix) else float('nan'):.3f}")
    return set(TE[ix])
P("POLICIES on test pairs (gold rate where Re-DocRED annotates the pair):")
pol('all proposals'); pol('2+ bridges', nbr >= 2); pol('3+ bridges (seeds)', nbr >= 3)
pol('composability, top 25%', None, comp, 0.25); pol('composability, top 10%', None, comp, 0.10)
pol('composability top 25% AND 2+ bridges', (comp >= np.quantile(comp[TE], 0.75)) & (nbr >= 2))
pol('composability top 25% among 2+ bridges', nbr >= 2, comp, 0.25)
pol('type-triple composability, top 25%', None, compt, 0.25); pol('type-triple composability, top 10%', None, compt, 0.10)
pol('type-triple composability top 25% AND 2+ bridges', (compt >= np.quantile(compt[TE], 0.75)) & (nbr >= 2))
pol('confirmed by later reading (waits for the text)', conf)
pol(f'confirmed in a RELATIONAL wording (learner top {1 - QR:.0%})', confr)
pol('confirmed in a relational wording AND 2+ bridges', confr & (nbr >= 2))
pol('composability top 25% OR confirmed in a relational wording', (comp >= np.quantile(comp[TE], 0.75)) | confr)
# ---- the stand-in trunk's congruence: the gain from A (as cg27a_rti), on a sample of test proposals
CKP = os.environ.get('TG_CKPT', '')
if CKP:
    sys.argv = [sys.argv[0], '6000', '5']; _tg = open(os.path.join(HERE, 'trunk_guides.py')).read(); g = {}
    exec(compile(_tg[:_tg.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'), g); lm, Lw, PADw = g['lm'], g['L'], g['PAD']
    import torch, torch.nn.functional as F
    @torch.no_grad()
    def bits(seqs, bs=256):
        out = np.zeros(len(seqs))
        for s0 in range(0, len(seqs), bs):
            ch = seqs[s0:s0 + bs]; X = torch.full((len(ch), Lw), PADw, dtype=torch.long)
            for j, (sb, a_, b_) in enumerate(ch): X[j, :len(sb[:Lw])] = torch.tensor(list(sb[:Lw]))
            Xi = torch.cat([torch.full((len(ch), 1), PADw), X[:, :-1]], 1); lp = F.log_softmax(lm(Xi), -1); nl = -lp.gather(-1, X[..., None])[..., 0] / math.log(2)
            for j, (sb, a_, b_) in enumerate(ch): out[s0 + j] = float(nl[j, a_:min(b_, Lw)].sum()) / max(1, min(b_, Lw) - a_)
        return out
    cls = {n: int(gn[NI[n]]) for n in NAMES[:NR]}; byc = collections.defaultdict(list)
    for n, k in cls.items(): byc[k].append(n)
    SAMP = TE[rq.permutation(len(TE))[:3000]]; seqs = []; meta = []
    for i in SAMP:
        A, Cn = sorted(keys[i]); pool = [n for n in byc[cls.get(Cn, -1)] if n not in (A, Cn)]
        if len(pool) < 9: continue
        cands = [Cn] + [pool[j] for j in rq.choice(len(pool), 9, replace=False)]; q = f"{A} is connected to ".encode('utf-8', 'replace'); q0 = b"Someone is connected to "
        for c in cands: cb = c.encode('utf-8', 'replace'); seqs += [(q + cb + b'.', len(q), len(q) + len(cb)), (q0 + cb + b'.', len(q0), len(q0) + len(cb))]
        meta.append(i)
    bt = bits(seqs); cong = np.full(len(keys), np.nan)
    for m, i in enumerate(meta):
        v = bt[20 * m + 1:20 * m + 20:2] - bt[20 * m:20 * m + 20:2]; cong[i] = -float((v > v[0]).sum())   # 0 = A helps C most; -9 = least
    ok = np.isfinite(cong[TE]); P(f"TRUNK CONGRUENCE (stand-in trunk, gain from A among 10 same-class names) on {int(ok.sum())} test proposals ({time.time() - T0:.0f}s):")
    yy = yte[ok]; cg = cong[TE][ok]; cp = comp[TE][ok]; nb2 = nbr[TE][ok] >= 2
    P(f"   AUC gold vs not {auc(cg[yy], cg[~yy]):.3f} | passes (C in the top 3) {np.mean(cg >= -2):.2f}: gold {yy[cg >= -2].mean():.3f} vs failed {yy[cg < -2].mean():.3f} (sample gold {yy.mean():.3f})")
    for tag, m in (('trunk congruence (top 3)', cg >= -2), ('composability top 25% AND trunk congruence (two keys)', (cp >= np.quantile(cp, 0.75)) & (cg >= -2)),
                   ('2+ bridges AND trunk congruence', nb2 & (cg >= -2)), ('2+ bridges AND composability top 25% AND congruence', nb2 & (cp >= np.quantile(cp, 0.75)) & (cg >= -2))):
        P(f"   {tag:62s} {int(m.sum()):6d} inferences, gold {yy[m].mean() if m.any() else float('nan'):.3f}")
# ---- BOOTSTRAPPING ROUNDS (CB_ROUNDS; 13 Nov late): grow the confirmed set from the most composable kinds, re-learn, re-score
RND = int(os.environ.get('CB_ROUNDS', 0)); GROW = float(os.environ.get('CB_GROW', 0.02))
if RND:
    S_ = conf & tr; vtr = np.array([k in ANN for k in keys]); gtr = np.array([k in GOLDP for k in keys])
    P(f"BOOTSTRAPPING: seeds = train proposals confirmed in a relational wording {int(S_.sum())} (gold where checkable {gtr[S_ & vtr].mean():.3f}); each round the "
      f"{GROW:.0%} of train proposals with the most composable type triple join the confirmed set")
    for r in range(RND + 1):
        tc2 = collections.Counter(); tn2 = collections.Counter()
        for i in np.where(tr)[0]:
            for kt in PR[keys[i]]['kt']: tn2[kt] += 1; tc2[kt] += int(S_[i])
        K2 = {kt: wilson(tc2[kt], tn2[kt]) for kt in tn2}; sc2 = np.array([max((K2.get(kt, 0.0) for kt in PR[k]['kt']), default=0.0) for k in keys])
        s_t = sc2[TE]; o = np.argsort(-s_t + 1e-9 * rq.rand(len(s_t))); g10 = yte[o[:len(o) // 10]].mean(); g25 = yte[o[:len(o) // 4]].mean()
        P(f"   round {r}: confirmed set {int(S_.sum())} (gold where checkable {gtr[S_ & vtr].mean():.3f}) | TEST composability top 10% gold {g10:.3f}, top 25% {g25:.3f}")
        if r == RND: break
        cand = np.where(tr & ~S_)[0]; add = cand[np.argsort(-sc2[cand])[:int(tr.sum() * GROW)]]
        P(f"      added {len(add)}: gold where checkable {gtr[add][vtr[add]].mean() if vtr[add].any() else float('nan'):.3f} ({int(vtr[add].sum())} checkable)")
        S_ = S_.copy(); S_[add] = True
# ---- HOP 2, memory-light (CB_HOP2=lite): a sample of accepted inferences as premises; only verifiable pairs kept
if os.environ.get('CB_HOP2', '1') == 'lite':
    for tag, acc in (('confirmed in a relational wording', np.where(confr)[0]), ('type-triple composability top 10%', np.where(compt >= np.quantile(compt, 0.90))[0]),
                     ('all proposals (sample)', np.arange(len(keys)))):
        smp = acc[rq.permutation(len(acc))[:3000]]; g1 = np.array([keys[i] in GOLDP for i in smp if keys[i] in ANN]); H2 = collections.Counter(); seen2 = set(keys[i] for i in smp)
        for i in smp:
            X_, Y_ = tuple(keys[i])
            for a_, b_ in ((X_, Y_), (Y_, X_)):
                if dfreq[b_] > 200: continue
                for D_, dD, j in nb[b_]:
                    if D_ == a_: continue
                    k2 = frozenset((a_, D_))
                    if k2 in ANN and k2 not in seen2: H2[k2] += 1
        k2s = list(H2); g2 = np.array([k in GOLDP for k in k2s]); m2 = np.array([H2[k] >= 2 for k in k2s])
        P(f"HOP 2 from {len(smp)} accepted inferences ({tag}; hop-1 gold {g1.mean() if len(g1) else float('nan'):.3f} of {len(g1)} checkable): {len(k2s)} verifiable hop-2 "
          f"inferences, gold {g2.mean() if len(g2) else float('nan'):.3f}" + (f" | 2+ routes {int(m2.sum())}, gold {g2[m2].mean():.3f}" if m2.any() else '') + f"  ({time.time() - T0:.0f}s)")
if os.environ.get('CB_HOP2', '1') in ('0', 'lite'): P(f"DONE ({time.time() - T0:.0f}s)"); sys.exit(0)
# ---- BUILT ON INFERENCE: accepted inferences as premises (hop 2)
ACC = [keys[i] for i in np.where((nbr >= 2) & (comp >= np.quantile(comp[nbr >= 2], 0.75)))[0]]
P(f"BUILT ON INFERENCE: {len(ACC)} accepted inferences (2+ bridges, composability top 25%; gold where checkable "
  f"{np.mean([k in GOLDP for k in ACC if k in ANN]):.3f} of {sum(k in ANN for k in ACC)}) used as premises:")
H2 = collections.defaultdict(set)
for k in ACC:
    A, Cn = tuple(k)
    for X, Y in ((A, Cn), (Cn, A)):
        if dfreq[Y] > 200: continue
        for D_, dD, j in nb[Y]:
            if D_ != X and frozenset((X, D_)) not in PR: H2[frozenset((X, D_))].add(Y)
k2 = [k for k in H2 if k in ANN]; g2 = np.array([k in GOLDP for k in k2]); m2 = np.array([len(H2[k]) >= 2 for k in k2])
P(f"   hop 2 (accepted inference + stored record, pairs not proposed at hop 1): {len(H2)} proposals, verifiable {len(k2)}, gold {g2.mean() if len(g2) else float('nan'):.3f}"
  + (f" | 2+ bridges {int(m2.sum())}, gold {g2[m2].mean():.3f}" if m2.any() else '') + f" | hop 1 all proposals gold {np.mean([k in GOLDP for k in keys if k in ANN]):.3f}")
P(f"DONE ({time.time() - T0:.0f}s)")
