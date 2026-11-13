"""13 Nov PREMISES AT SCALE (inf_premises; READOUT ONLY - nothing it builds reaches training or the fast store).
Tests in the model what the offline tests of 13 November found (tools/record_mdl.py, premise_threshold.py, surprise_store.py):
  a SEPARATE record store reads DocRED's distant Wikipedia set (inf_premises_path; a 1/inf_premises_visits share more each visit) and the
  benchmark documents of the domains read: a record = two names in one sentence, at most 12 words apart (span <= 60 characters), the words between.
  TYPES      a three-way block model (name class, wording class, name class) fitted by co-clustering, WARM-STARTED from the last readout's
             classes (the store keeps learning); name classes against the benchmark's gold entity types (and the distant set's automatic types)
  RELATION   held-out-style cue: benchmark gold-related records, does an equal name-class pair + wording class pick the same gold relation
  SURPRISE   the TRUNK's bits per byte over each new record, scored once when the record enters (SLIMM: what the trunk already predicts)
  PREMISES   related share of the records each label-free policy keeps (top 5 / 10 / 25 %): random, closeness, trunk surprise, block association,
             surprise + association; distant records against the distant set's labels (Wikidata, noisy), benchmark records against gold
  INFERENCE  A-C proposed through a shared name B from another document (B in at most 200 documents), store kept by each policy; checked
             where A and C are named in one document (distant labels); the offline premise threshold: 0.13 at premise precision 0.38, 0.31 at 0.7
"""
import re
import gzip
import json
import time
import collections
import numpy as np
from scipy.sparse import csr_matrix
from platypus.common import _fail
from platypus.inference.compress import _cg_auc, _cg_bits

_PM = {}
_SENT = re.compile(r'[^.!?]+[.!?]*')


def _load_distant(cfg):
    """records of the distant set, once: (doc, A, B, words between, last word, n words, labelled, record text); names' automatic types"""
    p = str(getattr(cfg, 'inf_premises_path', '') or '')
    if _PM.get('path') == p: return
    _PM.update(path=p, R=[], TY={}, LAB=set(), ND=0)
    if not p: return
    import os
    if not os.path.exists(p):
        print(f"  [PREMISES AT SCALE] distant set not found at {p}: benchmark documents only (the notebook downloads it)", flush=True); return
    t0 = time.time(); D = json.load(gzip.open(p, 'rt') if p.endswith('.gz') else open(p)); R = []; TY = collections.defaultdict(collections.Counter); LAB = set()
    for di in range(len(D)):
        d = D[di]; D[di] = None
        cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
        lab = {(l['h'], l['t']) for l in d['labels']} | {(l['t'], l['h']) for l in d['labels']}
        for l in d['labels']: LAB.add(frozenset((cn[l['h']], cn[l['t']])))
        bys = collections.defaultdict(list)
        for ei, vs in enumerate(d['vertexSet']):
            for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei)); TY[cn[ei]][m.get('type', 'X')] += 1
        for si, ms in bys.items():
            ms = sorted(ms); toks = d['sents'][si]
            for x in range(len(ms)):
                for y in range(x + 1, len(ms)):
                    (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y]
                    if ea == eb or a1 > b0 or cn[ea] == cn[eb] or b0 - a1 > 12: continue
                    sp = ' '.join(toks[a0:b1])
                    if len(sp) > 60: continue
                    mid = toks[a1:b0]; R.append((di, cn[ea], cn[eb], ' '.join(mid).lower(), mid[-1].lower() if mid else '', len(mid), (ea, eb) in lab, sp))
    _PM.update(R=R, TY={n: c.most_common(1)[0][0] for n, c in TY.items()}, LAB=LAB, ND=di + 1)
    print(f"  [PREMISES AT SCALE] distant set read: {_PM['ND']} documents, {len(R)} records ({time.time() - t0:.0f} s)", flush=True)


def _bench_records(cfg, seen):
    """benchmark records of the domains read (gold mentions): (doc key, A, B, words between, last word, n words, gold-related, text, domain,
    relation types, type pair). Mention entity ids are per document: each entity is mapped to the benchmark's global entity by its mention
    texts (ambiguous names: the candidate with the most gold relations to the document's other entities); unmapped entities have no gold
    relation in the benchmark (numbers, dates) and count as unrelated"""
    from platypus.inference.probes import _bench
    from platypus.inference.compress import _BENCH_MENT
    import zlib, base64
    G = _bench(cfg); MT = json.loads(zlib.decompress(base64.b85decode(_BENCH_MENT)).decode('utf-8')); tp = G.get('ent_type', {})
    RT = collections.defaultdict(set); inv = collections.defaultdict(set); adj = collections.defaultdict(set)
    for h, t, fl, ps in G['rel']:
        RT[(int(h), int(t))] |= set(ps); RT[(int(t), int(h))] |= {p + '^-1' for p in ps}; adj[int(h)].add(int(t)); adj[int(t)].add(int(h))
    for e, n in G['ent_name'].items(): inv[n].add(int(e))
    out = []; nmap = [0, 0]
    for d in seen:
        for di, t in enumerate(G['docs'].get(str(d), [])):
            ms = sorted(MT.get(str(d), [])[di] if di < len(MT.get(str(d), [])) else []); tx = collections.defaultdict(list)
            for ma, mb, mv in ms: tx[mv].append(t[ma:mb])
            cn = {v: collections.Counter(xs).most_common(1)[0][0] for v, xs in tx.items()}
            cand = {v: set().union(*[inv.get(x, set()) for x in set(xs)]) for v, xs in tx.items()}
            allc = set().union(*cand.values()) if cand else set(); gid = {}
            for v, cs in cand.items():
                if cs: gid[v] = max(sorted(cs), key=lambda g: len(adj[g] & (allc - {g})))
            nmap[0] += len(gid); nmap[1] += len(cand)
            for sa, sb in [(m.start(), m.end()) for m in _SENT.finditer(t)]:
                mm = [m for m in ms if sa <= m[0] and m[1] <= sb]
                for x in range(len(mm)):
                    for y in range(x + 1, len(mm)):
                        (ia, ib, iv), (ja, jb, jv) = mm[x], mm[y]
                        A, B = cn[iv], cn[jv]
                        if iv == jv or ib > ja or A == B or ja - ia > 60: continue
                        mid = t[ib:ja].split()
                        if len(mid) > 12: continue
                        rs = RT.get((gid[iv], gid[jv]), set()) if (iv in gid and jv in gid) else set()
                        out.append((f'b{d}.{di}', A, B, ' '.join(mid).lower(), mid[-1].lower() if mid else '', len(mid), bool(rs), t[ia:jb], int(d), rs,
                                    (tp.get(str(gid.get(iv, -1)), 'X'), tp.get(str(gid.get(jv, -1)), 'X')), gid.get(iv, -1), gid.get(jv, -1)))
    _PM['bmap'] = nmap
    return out


def _fit(a, b, w, NN, NW, K, M, gn, gw, iters=15, al=0.1):
    """three-way block model by co-clustering (as tools/record_mdl.py), from the given classes"""
    one = np.ones(len(a)); r = np.random.RandomState(0)
    for _ in range(iters):
        T = np.zeros((K, M, K)); np.add.at(T, (gn[a], gw[w], gn[b]), 1); T += al
        PL = np.log(T / T.sum((1, 2), keepdims=True)).reshape(K, M * K); PR = np.log(T / T.sum((0, 1), keepdims=True)).transpose(2, 0, 1).reshape(K, K * M)
        g2 = np.asarray(csr_matrix((one, (a, gw[w] * K + gn[b])), shape=(NN, M * K)) @ PL.T + csr_matrix((one, (b, gn[a] * M + gw[w])), shape=(NN, K * M)) @ PR.T).argmax(1)
        for e in set(range(K)) - set(g2.tolist()): g2[r.randint(NN)] = e
        ch = (g2 != gn).mean(); gn = g2
        T = np.zeros((K, M, K)); np.add.at(T, (gn[a], gw[w], gn[b]), 1); T += al
        PW = np.log(T / T.sum((0, 2), keepdims=True)).transpose(1, 0, 2).reshape(M, K * K)
        h2 = np.asarray(csr_matrix((one, (w, gn[a] * K + gn[b])), shape=(NW, K * K)) @ PW.T).argmax(1)
        for e in set(range(M)) - set(h2.tolist()): h2[r.randint(NW)] = e
        ch2 = (h2 != gw).mean(); gw = h2
        if ch < 0.002 and ch2 < 0.002: break
    T = np.zeros((K, M, K)); np.add.at(T, (gn[a], gw[w], gn[b]), 1)
    return gn, gw, T + al


def _purity(g, gt, wt):
    m = wt > 0
    if not m.any(): return float('nan'), float('nan')
    from sklearn.metrics import normalized_mutual_info_score
    g, gt, wt = g[m], gt[m], wt[m]; tot = sum(max(wt[(g == c) & (gt == t)].sum() for t in set(gt[g == c])) for c in set(g))
    return tot / wt.sum(), normalized_mutual_info_score(np.repeat(gt, wt), np.repeat(g, wt))


def premises_readout(model, cfg, S_slot, vis):
    t0 = time.time(); _load_distant(cfg); rs = np.random.RandomState(int(cfg.seed) + 777 + vis)
    seen = sorted(set(int(k) for k in (model.__dict__.get('_t2s') or {}).keys()))
    if not seen: seen = sorted(set(int(d) for d in range(int(vis) + 1)))
    nv = max(1, int(getattr(cfg, 'inf_premises_visits', 10))); ndv = int(round(_PM.get('ND', 0) * min(1.0, (vis + 1) / nv)))
    RD = [r for r in _PM.get('R', []) if r[0] < ndv]; RB = _bench_records(cfg, seen); nD = len(RD)
    # ---- the store's records: names, wordings (backed off below 3 occurrences: last word, then length), documents
    ALL = [(r[1], r[3], r[4], r[5], r[2]) for r in RD] + [(r[1], r[3], r[4], r[5], r[2]) for r in RB]
    fc = collections.Counter(x[1] for x in ALL); lc = collections.Counter('~' + x[2] for x in ALL)
    wk = lambda x: x[1] if fc[x[1]] >= 3 else ('~' + x[2] if lc['~' + x[2]] >= 3 else '~?%d' % min(x[3], 6))
    NI = {}; WI = {}
    a = np.array([NI.setdefault(x[0], len(NI)) for x in ALL]); b = np.array([NI.setdefault(x[4], len(NI)) for x in ALL]); w = np.array([WI.setdefault(wk(x), len(WI)) for x in ALL])
    NN, NW = len(NI), len(WI); NAMES = list(NI); WK = list(WI); y = np.array([bool(r[6]) for r in RD] + [bool(r[6]) for r in RB])
    from platypus.inference.probes import _bench
    G = _bench(cfg); bty = {}
    for r in RB:
        if r[11] >= 0: bty[r[1]] = r[10][0]
        if r[12] >= 0: bty[r[2]] = r[10][1]
    GTB = np.array([bty.get(n, '') for n in NAMES]); GTD = np.array([_PM['TY'].get(n, '') for n in NAMES])
    occB = np.bincount(a[nD:], minlength=NN) + np.bincount(b[nD:], minlength=NN); occD = np.bincount(a[:nD], minlength=NN) + np.bincount(b[:nD], minlength=NN)
    nall = occB + occD
    msg = [f"store: {ndv} distant documents ({nD} records) + {len(RB)} benchmark records of {len(seen)} domains (entities mapped to the gold {_PM['bmap'][0]} of {_PM['bmap'][1]}; "
           f"records gold-related {np.mean([r[6] for r in RB]) if RB else float('nan'):.3f}); {NN} names, {NW} wordings"]
    # ---- TYPES (warm-started from the last readout: the store keeps learning)
    FIT = {}
    for K in [int(k) for k in str(getattr(cfg, 'inf_premises_k', '4,16')).split(',') if k.strip()]:
        M = int(getattr(cfg, 'inf_premises_m', 128)); old = _PM.get(('cls', K, M), ({}, {}))
        gn = np.array([old[0].get(n, rs.randint(K)) for n in NAMES]); gw = np.array([old[1].get(k, rs.randint(M)) for k in WK])
        gn, gw, T = _fit(a, b, w, NN, NW, K, M, gn, gw); FIT[K] = (gn, gw, T, M)
        _PM[('cls', K, M)] = (dict(zip(NAMES, gn.tolist())), dict(zip(WK, gw.tolist())))
        pb, nb_ = _purity(gn, GTB, occB * (GTB != '')); pd, nd_ = _purity(gn, GTD, occD * (GTD != ''))
        bk = []
        for lo, hi in ((1, 2), (3, 9), (10, 49), (50, 10 ** 9)):
            m = (occB > 0) & (nall >= lo) & (nall <= hi)
            if m.sum() >= 20: bk.append(f"{lo}-{hi if hi < 1e9 else '+'}: {_purity(gn[m], GTB[m], np.ones(int(m.sum()), int))[0]:.2f} of {int(m.sum())}")
        msg.append(f"TYPES K {K} M {M}: benchmark names purity {pb:.3f}, NMI {nb_:.3f} (by store records per name: {', '.join(bk)}) | distant names (automatic types) purity {pd:.3f}, NMI {nd_:.3f}")
    # ---- RELATION cue on the benchmark's gold-related records (other documents)
    try:
        HR = [i for i, r in enumerate(RB) if r[9]]
        if len(HR) >= 60:
            K0 = min(FIT); gn, gw = FIT[K0][0], FIT[K0][1]; PAIRS = []
            for q in rs.choice(len(HR), min(300, len(HR)), replace=False):
                PAIRS += [(int(q), int(c)) for c in rs.choice(len(HR), 40, replace=False) if RB[HR[int(c)]][0] != RB[HR[int(q)]][0]][:30]
            same = np.array([bool(RB[HR[i]][9] & RB[HR[j]][9]) for i, j in PAIRS])
            cue = lambda f: np.array([float(f(i) == f(j)) for i, j in PAIRS])
            ncp = cue(lambda i: (gn[a[nD + HR[i]]], gn[b[nD + HR[i]]])); wcl = cue(lambda i: gw[w[nD + HR[i]]]); gtp = cue(lambda i: RB[HR[i]][10])
            msg.append(f"RELATION (K {K0}; {len(PAIRS)} pairs, same gold relation {same.mean():.3f}): AUC name-class pair + wording class {_cg_auc((ncp + wcl)[same], (ncp + wcl)[~same]):.3f} "
                       f"| gold type pair (reference) {_cg_auc(gtp[same], gtp[~same]):.3f}")
    except Exception as e: _fail('inference.premises:relation'); msg.append(f"RELATION unavailable ({type(e).__name__})")
    # ---- SURPRISE: the trunk scores each record once, when it enters
    SB = _PM.setdefault('sb', {}); SD = _PM.setdefault('sd', {})
    wsl = int(S_slot(1)) if 1 in seen else int(S_slot(seen[-1])); smax = int(getattr(cfg, 'inf_premises_score_max', 400000))
    newd = [i for i in range(nD) if i not in SD]; newd = [newd[i] for i in rs.permutation(len(newd))[:smax]] if len(newd) > smax else newd
    newb = [i for i, r in enumerate(RB) if (r[0], r[1], r[2], r[7]) not in SB]
    was = model.training; model.eval(); t1 = time.time()
    try:
        if newd:
            for i, v in zip(newd, _cg_bits(model, wsl, [(RD[i][7].encode('utf-8', 'replace'), 1, len(RD[i][7].encode('utf-8', 'replace'))) for i in newd], bs=256)):
                SD[i] = float(v) / max(1, len(RD[i][7].encode('utf-8', 'replace')) - 1)
        bysl = collections.defaultdict(list)
        for i in newb: bysl[int(S_slot(RB[i][8]))].append(i)
        for sl, ix in bysl.items():
            for i, v in zip(ix, _cg_bits(model, sl, [(RB[i][7].encode('latin-1', 'replace'), 1, len(RB[i][7].encode('latin-1', 'replace'))) for i in ix], bs=256)):
                SB[(RB[i][0], RB[i][1], RB[i][2], RB[i][7])] = float(v) / max(1, len(RB[i][7]) - 1)
    finally: model.train(was)
    tsc = time.time() - t1
    # ---- PREMISES by policy (distant: scored records, distant labels; benchmark: all records, gold)
    rk = lambda x: np.argsort(np.argsort(x, kind='stable'), kind='stable') / max(1, len(x) - 1)
    K1 = max(FIT); gn, gw, T, M = FIT[K1]; pg = T.sum((0, 2)) / T.sum()
    assoc = np.log(T[gn[a], gw[w], gn[b]] / T[gn[a], :, gn[b]].sum(-1)) - np.log(pg[gw[w]])
    clos = -np.array([r[5] for r in RD] + [r[5] for r in RB], float)
    def pol(ix, yy, surp):
        sc = {'random': rs.rand(len(ix)), 'closeness': clos[ix], 'trunk surprise': surp, 'block association': assoc[ix], 'surprise + association': rk(surp) + rk(assoc[ix])}
        out = []
        for k, s in sc.items():
            o = np.argsort(-s + 1e-9 * rs.rand(len(s))); out.append(f"{k} {yy[o[:max(1, len(s) // 20)]].mean():.2f}/{yy[o[:max(1, len(s) // 10)]].mean():.2f}/{yy[o[:max(1, len(s) // 4)]].mean():.2f}"
                                                                   + (f" AUC {_cg_auc(s[yy], s[~yy]):.3f}" if k == 'trunk surprise' else ''))
        return sc, ", ".join(out)
    ixd = np.array(sorted(i for i in SD if i < nD), int); sd = np.array([SD[i] for i in ixd]); scd = None
    if len(ixd) >= 100:
        scd, txt = pol(ixd, y[ixd], sd)
        msg.append(f"PREMISES, distant records scored {len(ixd)} (labelled {y[ixd].mean():.3f}) - related share kept at top 5%/10%/25%: {txt}")
    ixb = np.arange(nD, nD + len(RB)); sbv = np.array([SB.get((r[0], r[1], r[2], r[7]), np.nan) for r in RB])
    if len(RB) >= 100 and np.isfinite(sbv).all():
        _, txt = pol(ixb, y[ixb], sbv); msg.append(f"PREMISES, benchmark records {len(RB)} (gold-related {y[ixb].mean():.3f}): {txt}")
    # ---- INFERENCE: A-C through a shared name, store kept by each policy (distant scored records; distant labels)
    if scd is not None:
        try:
            dn = collections.defaultdict(set)
            for i in range(nD): dn[RD[i][1]].add(RD[i][0]); dn[RD[i][2]].add(RD[i][0])
            LAB = _PM['LAB']; res = []; amax = int(getattr(cfg, 'inf_premises_ac_max', 600000))
            sub = np.sort(np.random.RandomState(int(cfg.seed) + 4242).permutation(len(ixd))[:amax])   # a fixed sample of the scored store (time)
            for k in ('random', 'trunk surprise', 'block association'):
                for frac in (0.25, 0.10):
                    keep = ixd[sub][np.argsort(-scd[k][sub])[:int(len(sub) * frac)]]; nb = collections.defaultdict(list); PR = collections.defaultdict(set); np_ = 0
                    for i in keep: nb[RD[i][1]].append((RD[i][2], RD[i][0])); nb[RD[i][2]].append((RD[i][1], RD[i][0]))
                    for i in keep:
                        if np_ > 3000000: break
                        for B_, C_ in ((RD[i][1], RD[i][2]), (RD[i][2], RD[i][1])):
                            if len(dn[B_]) > 200: continue
                            for A_, dA in nb[B_]:
                                if dA != RD[i][0] and A_ != C_: PR[frozenset((A_, C_))].add(B_); np_ += 1
                    ks = [p for p in PR if len(p) == 2 and (lambda u, v: bool(dn[u] & dn[v]))(*tuple(p))]
                    g = np.array([p in LAB for p in ks]); b2 = np.array([len(PR[p]) >= 2 for p in ks])
                    res.append(f"{k} {frac:.0%} (premises {y[keep].mean():.2f}): {len(ks)} checkable, right {g.mean() if len(g) else float('nan'):.3f}"
                               + (f", 2+ bridges {int(b2.sum())} right {g[b2].mean():.3f}" if b2.any() else ''))
            msg.append("INFERENCE A-C (distant labels; offline threshold: 0.13 at premises 0.38, 0.31 at 0.70): " + " | ".join(res))
        except Exception as e: _fail('inference.premises:infer'); msg.append(f"INFERENCE unavailable ({type(e).__name__}: {e})"[:160])
    print(f"  [PREMISES AT SCALE | visit {vis + 1}] " + " || ".join(msg) + f"  (trunk scoring {len(newd) + len(newb)} records {tsc:.0f} s; total {time.time() - t0:.0f} s)", flush=True)
    if int(getattr(cfg, 'inf_rti_write', 0)) and scd is not None:
        try: _rti_propose(model, cfg, S_slot, seen, RD, ixd, scd, gn, NI, NAMES, nall, vis)
        except Exception as e: _fail('inference.premises:rti'); print(f"  [REAL-TEXT INFERENCE] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


_RTI_R = 'is connected to'


def _rti_propose(model, cfg, S_slot, seen, RD, ixd, scd, gn, NI, NAMES, nall, vis):
    """13 Nov END-TO-END REAL-TEXT INFERENCE (inf_rti_write; with inf_premises). READ (the premises store) -> CHOOSE premises (the top
    inf_rti_keep of scored distant records by trunk surprise + block association, label-free) -> INFER A-C through a shared name B from another
    document, kept only with 2+ distinct bridges -> CHECK by the model: after 'A is connected to ', does the trunk rank C among the top
    inf_rti_rank of C and 9 names of C's class? (the trunk's own agreement, no labels) -> WRITE the passing ones (at most inf_rti_max a readout)
    as facts 'A is connected to C' into the question pool (_rti_write). Truth where checkable: A and C named in one distant document, related
    by its (Wikidata) labels."""
    t0 = time.time(); rs = np.random.RandomState(int(cfg.seed) + 9191 + vis); nD = len(RD)
    dn = collections.defaultdict(set)
    for i in range(nD): dn[RD[i][1]].add(RD[i][0]); dn[RD[i][2]].add(RD[i][0])
    LAB = _PM['LAB']; rk = lambda x: np.argsort(np.argsort(x, kind='stable'), kind='stable') / max(1, len(x) - 1)
    sc = rk(scd['trunk surprise']) + rk(scd['block association']); keep = ixd[np.argsort(-sc)[:int(len(ixd) * float(getattr(cfg, 'inf_rti_keep', 0.1)))]]
    nb = collections.defaultdict(list); PR = collections.defaultdict(set)
    for i in keep: nb[RD[i][1]].append((RD[i][2], RD[i][0])); nb[RD[i][2]].append((RD[i][1], RD[i][0]))
    for i in keep:
        for B_, C_ in ((RD[i][1], RD[i][2]), (RD[i][2], RD[i][1])):
            if len(dn[B_]) > 200: continue
            for A_, dA in nb[B_]:
                if dA != RD[i][0] and A_ != C_: PR[(A_, C_)].add(B_)
    W = _PM.setdefault('rti', {}); done = {frozenset(k) for k in W}
    nbm = int(getattr(cfg, 'inf_rti_bridges', 2))
    P2 = [(k, len(v)) for k, v in PR.items() if len(v) >= nbm and frozenset(k) not in done and k[0] in NI and k[1] in NI]
    P2.sort(key=lambda x: -x[1]); P2 = P2[:int(getattr(cfg, 'inf_rti_n', 600))]
    if not P2:
        print(f"  [REAL-TEXT INFERENCE | visit {vis + 1}] no new proposals with {nbm}+ bridges ({len(PR)} proposals from {len(keep)} premises)", flush=True); return
    byc = collections.defaultdict(list)                                       # distractors: names of C's class seen 3+ times in the store
    for n, i in NI.items():
        if nall[i] >= 3: byc[int(gn[i])].append(n)
    seqs, meta = [], []
    for (A_, C_), nbr in P2:
        pool = [n for n in byc[int(gn[NI[C_]])] if n not in (A_, C_)]
        if len(pool) < 9: continue
        cands = [C_] + [pool[j] for j in rs.choice(len(pool), 9, replace=False)]; q = f"{A_} {_RTI_R} ".encode('utf-8', 'replace')
        for c in cands:
            cb = c.encode('utf-8', 'replace'); seqs.append((q + cb + b'.', len(q), len(q) + len(cb)))
        meta.append((A_, C_, nbr, [len(c.encode('utf-8', 'replace')) for c in cands]))
    dsl = 1 if 1 in seen else seen[-1]; was = model.training; model.eval()
    try: bt = _cg_bits(model, int(S_slot(dsl)), seqs, bs=256) if seqs else np.zeros(0)
    finally: model.train(was)
    rows = []; nmax = int(getattr(cfg, 'inf_rti_max', 200)); rmax = int(getattr(cfg, 'inf_rti_rank', 3))
    for k, (A_, C_, nbr, ls) in enumerate(meta):
        v = bt[10 * k:10 * k + 10] / np.array(ls); rank = int((v < v[0]).sum())   # bits per byte; 0 = C is the trunk's first choice
        chk = bool(dn[A_] & dn[C_]); tr = (frozenset((A_, C_)) in LAB) if chk else None
        rows.append((A_, C_, nbr, rank, tr))
    ok = [r for r in rows if r[3] < rmax]; ok.sort(key=lambda r: (r[3], -r[2])); wr = ok[:nmax]
    for A_, C_, nbr, rank, tr in wr: W[(A_, C_)] = dict(vis=vis, d=int(dsl), truth=tr, bridges=nbr, rank=rank)
    f = lambda L: (f"{np.mean([r[4] for r in L if r[4] is not None]):.3f} of {sum(r[4] is not None for r in L)} checkable" if any(r[4] is not None for r in L) else "none checkable")
    ra = np.array([r[3] for r in rows if r[4] is not None], float); ta = np.array([r[4] for r in rows if r[4] is not None], bool)
    au = _cg_auc(-ra[ta], -ra[~ta]) if ta.any() and (~ta).any() else float('nan')
    print(f"  [REAL-TEXT INFERENCE | visit {vis + 1}] READ {nD} distant records -> CHOSE {len(keep)} premises (top {float(getattr(cfg, 'inf_rti_keep', 0.1)):.0%} by surprise + association) "
          f"-> INFERRED {len(PR)} A-C, {sum(1 for v in PR.values() if len(v) >= nbm)} with {nbm}+ bridges, {len(rows)} new ones checked -> CHECK (the trunk ranks C in its top {rmax} of 10 "
          f"after 'A {_RTI_R}'): passed {len(ok)}, right {f(ok)}; failed {len(rows) - len(ok)}, right {f([r for r in rows if r[3] >= rmax])} | the check tells right from wrong: AUC {au:.3f} "
          f"-> WROTE {len(wr)} (right {f(wr)}; {len(W)} in all) ({time.time() - t0:.0f} s)", flush=True)


def rti_write(model, cfg, S_slot, ents, held, ctl, vis):
    """THE ONE MODEL CHANGE of cg27a_rti (inf_rti_write 1): real-text inferences that passed the check join the question pool as facts
    'A is connected to C', taught by the interleaved rows like any fact; a fifth is held back by the pool's own hash (control). READOUT: recall
    from the trunk (test wording 'It is on record that A is connected to ', store and memory off) of inferences written at an earlier readout,
    taught vs held back, split by whether the inference is RIGHT, WRONG (a false memory if recalled) or unknown."""
    from platypus.inference.distill import _dt_recall
    W = _PM.get('rti') or {}
    if not W:
        print(f"  [REAL-TEXT INFERENCE WRITES | visit {vis + 1}] none yet", flush=True); return ents, held
    have = {repr(e['key']) for e in ents + held}; T_, H_ = [], []
    for (A_, C_), w in W.items():
        e = dict(key=('fact', A_, _RTI_R, C_), A=A_, r=_RTI_R, ans=C_, d=w['d'], pfs=[f"{A_} {_RTI_R} "], seen=2, src=(), rm=w['vis'], truth=w['truth'])
        if repr(e['key']) in have: continue
        (H_ if ctl(e) else T_).append(e)
    old = lambda L, t: [e for e in L if e['rm'] < vis and e['truth'] is t]
    g = lambda d: f"{d['facts'][0]:.2f} ({d['facts'][1]})"
    rec = {t: (_dt_recall(model, S_slot, old(T_, t)), _dt_recall(model, S_slot, old(H_, t))) for t in (True, False, None)}
    print(f"  [REAL-TEXT INFERENCE WRITES | visit {vis + 1}] {len(W)} inferences written ({sum(1 for w in W.values() if w['vis'] == vis)} this readout) -> question pool: {len(T_)} taught, "
          f"{len(H_)} held back || FROM THE TRUNK, written at an earlier readout, TAUGHT vs HELD BACK: RIGHT inferences {g(rec[True][0])} vs {g(rec[True][1])} | WRONG ones "
          f"(false memories if recalled) {g(rec[False][0])} vs {g(rec[False][1])} | not checkable {g(rec[None][0])} vs {g(rec[None][1])}", flush=True)
    return ents + T_, held + H_
