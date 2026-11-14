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
                        mid = re.findall(r"\w+|[^\w\s]", t[ib:ja])               # punctuation as its own token, as DocRED's records
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
    bd_ = collections.defaultdict(set)
    for r in RB: bd_[r[0]].update((r[1], r[2]))
    _PM['bench_co'] = {frozenset((x, z)) for ns in bd_.values() for x in ns for z in ns if x != z}
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
    if int(getattr(cfg, 'inf_premises_typed', 0)):
        try: _typed_readout(cfg, RD, RB, a, b, w, NN, NW, NAMES, WK, gn, y, rs, vis)
        except Exception as e: _fail('inference.premises:typed'); print(f"  [TYPED INFERENCE] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
    if int(getattr(cfg, 'inf_rti_write', 0)) and scd is not None:
        try: _rti_propose(model, cfg, S_slot, seen, RD, ixd, scd, gn, NI, NAMES, nall, vis)
        except Exception as e: _fail('inference.premises:rti'); print(f"  [REAL-TEXT INFERENCE] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


_RTI_R = 'is connected to'


def _shape(n):
    if any(ch.isdigit() for ch in n): return 'num'
    c = sum(t[:1].isupper() for t in n.split()); return f"cap{min(c, 3)}" if c else 'low'


def _typed_readout(cfg, RD, RB, a, b, w, NN, NW, NAMES, WK, gn, y, rs, vis):
    """13 Nov TYPED INFERENCE readout (inf_premises_typed; nothing written): tools/typed_infer.py + tools/chain_bootstrap.py inside the model.
    UNIVERSAL SCHEMA (PPMI + SVD over the store's wordings x ordered name pairs, k-means -> latent relations) -> RELATION IDENTITY on the
    benchmark's gold-related records; a RELATIONAL-WORDING SCORER (label-free teachers: corroboration, functional and selective wordings ->
    a learner on hashed features of the words between) -> premise precision; TYPED, COUPLED INFERENCE on a sample of the store (premise
    relations -> the relation A and C are later read in, in a relational wording; learned on half the pairs, scored on the other half;
    type coupling on the compressed name types) and TYPED HOP 2; checked against the distant labels (offline, Re-DocRED gold: typed
    composability + coupling 0.160 vs 0.108 for all, typed hop 2 0.196 vs untyped 0.14)."""
    import zlib, math as _m
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    t0 = time.time(); nD = len(RD); NR = nD + len(RB); msg = []
    dk = np.array([r[0] for r in RD] + [-1 - i for i in range(len(RB))], np.int64)   # document per record (benchmark documents: one key each record's doc)
    bd = {}; dk[nD:] = [-1 - bd.setdefault(r[0], len(bd)) for r in RB]
    # ---- universal schema
    PI = {}; pid = np.array([PI.setdefault((int(x), int(z)), len(PI)) for x, z in zip(a, b)])
    M_ = csr_matrix((np.ones(NR), (w, pid)), shape=(NW, len(PI))); M_.sum_duplicates(); M_.data[:] = 1.0
    rsum = np.asarray(M_.sum(1)).ravel(); csum = np.asarray(M_.sum(0)).ravel(); Mc = M_.tocoo(); pm = np.log(M_.sum() / (rsum[Mc.row] * csum[Mc.col]) + 1e-12); kp = pm > 0
    E = normalize(TruncatedSVD(64, random_state=0).fit_transform(csr_matrix((pm[kp], (Mc.row[kp], Mc.col[kp])), shape=M_.shape)))
    rel = MiniBatchKMeans(64, random_state=0, batch_size=4096, n_init=3).fit_predict(E)
    HR = [nD + i for i, r in enumerate(RB) if r[9]]; ra = np.random.RandomState(int(cfg.seed) + 31 + vis)
    if len(HR) >= 60:
        PA = []
        for q in ra.choice(len(HR), min(300, len(HR)), replace=False):
            PA += [(int(q), int(c)) for c in ra.choice(len(HR), 40, replace=False) if RB[HR[int(c)] - nD][0] != RB[HR[int(q)] - nD][0]][:30]
        same = np.array([bool(RB[HR[i] - nD][9] & RB[HR[j] - nD][9]) for i, j in PA])
        cosw = np.array([float(E[w[HR[i]]] @ E[w[HR[j]]]) for i, j in PA]); tpr = np.array([float((gn[a[HR[i]]], gn[b[HR[i]]]) == (gn[a[HR[j]]], gn[b[HR[j]]])) for i, j in PA])
        msg.append(f"UNIVERSAL SCHEMA ({NW} wordings x {len(PI)} pairs, recurring {int((csum >= 2).sum())}): RELATION IDENTITY (benchmark gold, {len(PA)} pairs) wording vector "
                   f"{_cg_auc(cosw[same], cosw[~same]):.3f}, + name-type pair {_cg_auc((cosw + tpr)[same], (cosw + tpr)[~same]):.3f} (offline 0.658 / 0.669 at 30k)")
    # ---- relational-wording scorer (label-free teachers -> learner)
    mid = [r[3] for r in RD] + [r[3] for r in RB]; An = [NAMES[i] for i in a]; Bn = [NAMES[i] for i in b]
    trip = collections.defaultdict(set); fsub = collections.defaultdict(lambda: collections.defaultdict(set)); fA = collections.defaultdict(collections.Counter); fB = collections.defaultdict(collections.Counter)
    for i in range(NR): trip[(a[i], mid[i], b[i])].add(int(dk[i])); fsub[mid[i]][a[i]].add(b[i]); fA[mid[i]][_shape(An[i])] += 1; fB[mid[i]][_shape(Bn[i])] += 1
    FUN = {}
    for f_, d_ in fsub.items():
        ss = [v for v in d_.values() if len(v) >= 1]
        if sum(1 for v in ss if len(v) >= 1) >= 3: FUN[f_] = float(np.mean([len(v) == 1 for v in ss]))
    ent = lambda c_: -sum(v / sum(c_.values()) * _m.log2(v / sum(c_.values())) for v in c_.values())
    SEL = {f_: -(ent(fA[f_]) + ent(fB[f_])) for f_ in fA if sum(fA[f_].values()) >= 5}
    t1 = np.array([float(len(trip[(a[i], mid[i], b[i])] - {int(dk[i])}) > 0) for i in range(NR)])
    t2 = np.array([FUN.get(mid[i], np.nan) for i in range(NR)]); t3 = np.array([SEL.get(mid[i], np.nan) for i in range(NR)])
    rkn = lambda x: (lambda z: np.argsort(np.argsort(z, kind='stable'), kind='stable') / max(1, len(z) - 1))(np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(x).any() else 0.0, x))
    teach = rkn(t2) + rkn(t3) + 2 * t1
    HV = HashingVectorizer(n_features=2 ** 16, ngram_range=(1, 2), alternate_sign=False, token_pattern=r"[A-Za-z]+|[^\sA-Za-z]", norm='l2')
    txt = lambda i: mid[i] + ' FIRST_' + (mid[i].split() or ['none'])[0] + ' LAST_' + (mid[i].split() or ['none'])[-1] + ' SA_' + _shape(An[i]) + ' SB_' + _shape(Bn[i])
    trs = ra.permutation(NR)[:int(getattr(cfg, 'inf_premises_typed_train', 200000))]
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(HV.transform([txt(i) for i in trs]), teach[trs] >= np.percentile(teach[trs], 75))
    srel = np.concatenate([lr.decision_function(HV.transform([txt(i) for i in range(s0, min(NR, s0 + 200000))])) for s0 in range(0, NR, 200000)])
    def topk(sc, yy):
        o = np.argsort(-sc + 1e-9 * ra.rand(len(sc))); return '/'.join(f"{yy[o[:max(1, len(o) * k // 100)]].mean():.2f}" for k in (5, 10, 25))
    yb = y[nD:]; yd = y[:nD]
    msg.append(f"RELATIONAL WORDING (label-free learner): benchmark gold AUC {_cg_auc(srel[nD:][yb], srel[nD:][~yb]):.3f}, top 5/10/25% {topk(srel[nD:], yb)} (base {yb.mean():.2f}) | "
               f"distant labels AUC {_cg_auc(srel[:nD][yd], srel[:nD][~yd]):.3f}, {topk(srel[:nD], yd)} (base {yd.mean():.2f}) (offline 0.55-0.60 at top 5%)")
    # ---- typed, coupled inference on a sample of the store
    nk = np.unique(np.concatenate([a, b]).astype(np.int64) * (1 << 30) + np.concatenate([dk, dk]) + (1 << 29)) >> 30; dfr = np.bincount(nk, minlength=NN)
    smp = np.concatenate([ra.permutation(nD)[:int(getattr(cfg, 'inf_premises_typed_n', 80000))], np.arange(nD, NR)]); smp.sort()
    thr = np.quantile(srel, 0.75); SREL = collections.defaultdict(list)
    for i in np.where(srel >= thr)[0]: SREL[(min(a[i], b[i]), max(a[i], b[i]))].append((int(rel[w[i]]), int(dk[i])))
    LAB = _PM.get('LAB', set()); RBG = {}
    for r in RB:
        if r[6]: RBG[frozenset((r[1], r[2]))] = True
    nb = collections.defaultdict(list)
    for i in smp: nb[a[i]].append((b[i], int(dk[i]), i)); nb[b[i]].append((a[i], int(dk[i]), i))
    side = lambda i, B: 0 if b[i] == B else 1
    PR = {}
    for i in smp:
        for B, Cn in ((a[i], b[i]), (b[i], a[i])):
            if dfr[B] > 200: continue
            for A, dA, j in nb[B]:
                if dA == int(dk[i]) or A == Cn: continue
                k1, k2 = (int(rel[w[j]]), side(j, B)), (int(rel[w[i]]), side(i, B)); kd = (k1, k2) if k1 <= k2 else (k2, k1)
                key = (min(A, Cn), max(A, Cn)); p = PR.get(key)
                if p is None: p = PR[key] = [set(), set(), set()]
                p[0].add(kd); p[1].update((dA, int(dk[i]))); p[2].add(B)
    keys = list(PR); trm = np.array([zlib.crc32(repr(k).encode()) % 2 == 0 for k in keys])
    cf = [[r for r, d_ in SREL.get(k, ()) if d_ not in PR[k][1]] for k in keys]
    CT = collections.defaultdict(collections.Counter); KN = collections.Counter()
    for k, t_, c_ in zip(keys, trm, cf):
        if t_:
            for kd in PR[k][0]:
                KN[kd] += 1
                for r_ in set(c_): CT[kd][r_] += 1
    def wil(s_, n_, z=1.64):
        if n_ == 0: return 0.0
        p_ = s_ / n_; return (p_ + z * z / (2 * n_) - z * _m.sqrt(p_ * (1 - p_) / n_ + z * z / (4 * n_ * n_))) / (1 + z * z / n_)
    def typ(k):
        best = (0.0, -1)
        for kd in PR[k][0]:
            if CT[kd]: r_, n_ = CT[kd].most_common(1)[0]; best = max(best, (wil(n_, KN[kd]), r_))
        return best
    TY = [typ(k) for k in keys]; comp = np.array([t[0] for t in TY]); prel = np.array([t[1] for t in TY])
    ARG = collections.defaultdict(collections.Counter)
    for i in np.where(srel >= np.quantile(srel, 0.5))[0]: ARG[int(rel[w[i]])][tuple(sorted((int(gn[a[i]]), int(gn[b[i]]))))] += 1
    cpl = np.array([r_ >= 0 and sum(ARG[r_].values()) > 0 and ARG[r_][tuple(sorted((int(gn[k[0]]), int(gn[k[1]]))))] / sum(ARG[r_].values()) >= 0.01 for k, r_ in zip(keys, prel)])
    dn = collections.defaultdict(set)
    for i in range(NR): dn[a[i]].add(int(dk[i])); dn[b[i]].add(int(dk[i]))
    chk = np.array([bool(dn[k[0]] & dn[k[1]]) for k in keys]); tru = np.array([frozenset((NAMES[k[0]], NAMES[k[1]])) in LAB or frozenset((NAMES[k[0]], NAMES[k[1]])) in RBG for k in keys])
    TEv = np.where(~trm & chk)[0]; nbr = np.array([len(PR[k][2]) for k in keys]); conf = np.array([bool(c_) for c_ in cf])
    pol = lambda tag, m: f"{tag} {int(m[TEv].sum())} right {tru[TEv][m[TEv]].mean() if m[TEv].any() else float('nan'):.3f}"
    q90 = np.quantile(comp[TEv], 0.90) if len(TEv) else 1.0
    res = [pol('all', np.ones(len(keys), bool)), pol('2+ bridges', nbr >= 2), pol('typed composability top 10%', comp >= q90), pol('+ type coupling', (comp >= q90) & cpl),
           pol('confirmed in a relational wording', conf), pol('relational confirmation + 2+ bridges', conf & (nbr >= 2))]
    msg.append(f"TYPED INFERENCE ({len(smp)} premises, {len(keys)} proposals, {sum(1 for t_, c_ in zip(trm, cf) if t_ and c_)} train pairs confirmed; test checkable {len(TEv)}, distant labels / "
               f"benchmark gold): " + " | ".join(res))
    acc = np.where((comp >= q90) & cpl)[0]; acc = acc[ra.permutation(len(acc))[:3000]]; H2 = {}
    for i in acc:
        X_, Y_ = keys[i]; r3 = int(prel[i])
        for a_, b_ in ((X_, Y_), (Y_, X_)):
            if dfr[b_] > 200: continue
            for D_, dD, j in nb[b_]:
                if D_ == a_: continue
                k2 = (min(a_, D_), max(a_, D_))
                if k2 in PR or not (dn[a_] & dn[D_]): continue
                k1, kk = (r3, 0), (int(rel[w[j]]), side(j, b_)); kd = (k1, kk) if k1 <= kk else (kk, k1)
                H2[k2] = max(H2.get(k2, 0.0), wil(CT[kd].most_common(1)[0][1], KN[kd]) if CT[kd] else 0.0)
    if H2:
        k2s = list(H2); g2 = np.array([frozenset((NAMES[k[0]], NAMES[k[1]])) in LAB for k in k2s]); c2 = np.array([H2[k] for k in k2s])
        m2 = c2 >= (np.quantile(c2[c2 > 0], 0.75) if (c2 > 0).any() else 1.0)
        msg.append(f"TYPED HOP 2 (from {len(acc)} accepted): {len(k2s)} checkable, right {g2.mean():.3f} | composability-gated {int(m2.sum())}, right {g2[m2].mean() if m2.any() else float('nan'):.3f}")
    print(f"  [TYPED INFERENCE | visit {vis + 1}] " + " || ".join(msg) + f"  ({time.time() - t0:.0f} s)", flush=True)


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
    pol_ = str(getattr(cfg, 'inf_rti_policy', 'surprise'))                     # 13 Nov GPU cg27a_prem visit 2: A-C from association-kept premises 0.04-0.07
    sc = {'surprise': rk(scd['trunk surprise']), 'both': rk(scd['trunk surprise']) + rk(scd['block association']),   #   right vs 0.13 (surprise), 0.18 (random)
          'random': scd['random']}.get(pol_, rk(scd['trunk surprise']))
    keep = ixd[np.argsort(-sc)[:int(len(ixd) * float(getattr(cfg, 'inf_rti_keep', 0.1)))]]
    nb = collections.defaultdict(list); PR = collections.defaultdict(set); PRR = collections.defaultdict(set)
    for i in keep: nb[RD[i][1]].append((RD[i][2], RD[i][0], i)); nb[RD[i][2]].append((RD[i][1], RD[i][0], i))
    for i in keep:
        for B_, C_ in ((RD[i][1], RD[i][2]), (RD[i][2], RD[i][1])):
            if len(dn[B_]) > 200: continue
            for A_, dA, j in nb[B_]:
                if dA != RD[i][0] and A_ != C_: PR[(A_, C_)].add(B_); PRR[(A_, C_)].update((i, j))
    W = _PM.setdefault('rti', {}); done = {frozenset(k) for k in W}
    nbm = int(getattr(cfg, 'inf_rti_bridges', 2))
    nov = int(getattr(cfg, 'inf_rti_novel', 0)); bco = _PM.get('bench_co', set())
    def novel(k):                                                             # 14 Nov NOVELTY: A and C never together in anything the TRUNK reads - the benchmark
        if nov != 1: return True                                              #   documents and the evidence spans it replays (each span holds A and B or B and C,
        return frozenset(k) not in bco and not any(k[0] in RD[i][7] and k[1] in RD[i][7] for i in PRR[k])   # never both). The distant documents are read
                                                                              #   only by the record store, so A and C may share one there: that is what keeps
                                                                              #   the inference checkable (excluding premise documents left 4 of 300 checkable)
    isnov = lambda k: frozenset(k) not in bco and not any(k[0] in RD[i][7] and k[1] in RD[i][7] for i in PRR[k])   # inf_rti_novel 2: TAG only (write as cg27a_rti,
                                                                              #   report the novel ones apart - the filter left few checkable inferences locally)
    if int(getattr(cfg, 'inf_rti_code', 0)):                                  # 14 Nov cg27a_rcode: choose what to write by the compressed fact table
        return _rcode_step(model, cfg, S_slot, seen, RD, vis)
    if int(getattr(cfg, 'inf_rti_confirm', 0)):                               # 14 Nov cg27a_rtc: predict now, write only when later text confirms
        return _rtc_step(model, cfg, S_slot, seen, RD, PR, PRR, dn, NI, nbm, len(keep), vis)
    P2 = [(k, len(v)) for k, v in PR.items() if len(v) >= nbm and frozenset(k) not in done and k[0] in NI and k[1] in NI and novel(k)]
    P2.sort(key=lambda x: -x[1]); P2 = P2[:int(getattr(cfg, 'inf_rti_n', 600))]
    if not P2:
        print(f"  [REAL-TEXT INFERENCE | visit {vis + 1}] no new proposals with {nbm}+ bridges ({len(PR)} proposals from {len(keep)} premises)", flush=True); return
    byc = collections.defaultdict(list)                                       # distractors: names of C's class, sorted by how often the store
    for n, i in NI.items(): byc[int(gn[i])].append((int(nall[i]), n))          #   has seen them (matched to C's familiarity below)
    for c_ in byc: byc[c_].sort(); byc[c_] = (np.array([x[0] for x in byc[c_]]), [x[1] for x in byc[c_]])
    seqs, meta = [], []
    for (A_, C_), nbr in P2:
        cnt_, nms_ = byc[int(gn[NI[C_]])]; j0 = int(np.searchsorted(cnt_, nall[NI[C_]]))   # the 30 names nearest C in familiarity: the trunk
        near = [n for n in nms_[max(0, j0 - 16):j0 + 16] if n not in (A_, C_)]            #   favours familiar names, which is not the check
        if len(near) < 9: continue
        cands = [C_] + [near[j] for j in rs.choice(len(near), 9, replace=False)]; q = f"{A_} {_RTI_R} ".encode('utf-8', 'replace')
        q0 = f"Someone {_RTI_R} ".encode('utf-8', 'replace')                      # each name's cost without A: the check is the GAIN from A
        for c in cands:
            cb = c.encode('utf-8', 'replace'); seqs.append((q + cb + b'.', len(q), len(q) + len(cb))); seqs.append((q0 + cb + b'.', len(q0), len(q0) + len(cb)))
        meta.append((A_, C_, nbr, [len(c.encode('utf-8', 'replace')) for c in cands]))
    dsl = 1 if 1 in seen else seen[-1]; was = model.training; model.eval()
    try: bt = _cg_bits(model, int(S_slot(dsl)), seqs, bs=256) if seqs else np.zeros(0)
    finally: model.train(was)
    rows = []; nmax = int(getattr(cfg, 'inf_rti_max', 200)); rmax = int(getattr(cfg, 'inf_rti_rank', 3))
    for k, (A_, C_, nbr, ls) in enumerate(meta):
        b2 = bt[20 * k:20 * k + 20]; v = (b2[1::2] - b2[0::2]) / np.array(ls)     # bits per byte saved by A (neutral minus with A)
        rank = int((v > v[0]).sum())                                            # 0 = A helps C more than any of the 9 other names
        chk = bool(dn[A_] & dn[C_]); tr = (frozenset((A_, C_)) in LAB) if chk else None
        rows.append((A_, C_, nbr, rank, tr))
    ok = [r for r in rows if r[3] < rmax]; ok.sort(key=lambda r: (r[3], -r[2])); wr = ok[:nmax]
    for A_, C_, nbr, rank, tr in wr: W[(A_, C_)] = dict(vis=vis, d=int(dsl), truth=tr, bridges=nbr, rank=rank, novel=bool(isnov((A_, C_))),
                                                         ev=[RD[i][7] for i in sorted(PRR[(A_, C_)])[:6]])   # the premise statements (evidence replay)
    f = lambda L: (f"{np.mean([r[4] for r in L if r[4] is not None]):.3f} of {sum(r[4] is not None for r in L)} checkable" if any(r[4] is not None for r in L) else "none checkable")
    ra = np.array([r[3] for r in rows if r[4] is not None], float); ta = np.array([r[4] for r in rows if r[4] is not None], bool)
    au = _cg_auc(-ra[ta], -ra[~ta]) if ta.any() and (~ta).any() else float('nan')
    print(f"  [REAL-TEXT INFERENCE | visit {vis + 1}] READ {nD} distant records -> CHOSE {len(keep)} premises (top {float(getattr(cfg, 'inf_rti_keep', 0.1)):.0%} by {pol_}) "
          f"-> INFERRED {len(PR)} A-C, {sum(1 for v in PR.values() if len(v) >= nbm)} with {nbm}+ bridges, {len(rows)} new ones checked -> CHECK (A helps the trunk predict C more than all but {rmax - 1} of 9 same-class names; "
          f"'A {_RTI_R}' vs 'Someone {_RTI_R}'): passed {len(ok)}, right {f(ok)}; failed {len(rows) - len(ok)}, right {f([r for r in rows if r[3] >= rmax])} | the check tells right from wrong: AUC {au:.3f} "
          f"-> WROTE {len(wr)} (right {f(wr)}; novel - A and C never together in anything the trunk reads - {sum(1 for r in wr if isnov((r[0], r[1])))}, right "
          f"{f([r for r in wr if isnov((r[0], r[1]))])}; {len(W)} in all) ({time.time() - t0:.0f} s)", flush=True)


def _relscorer(cfg, RD, rs):
    """the label-free RELATIONAL-WORDING scorer of _typed_readout on the distant records alone (teachers: corroboration in another document,
    functional and selective wordings -> a logistic learner on hashed features of the words between): returns score(indices), threshold
    (the learner's top quarter on its training sample), and the training sample's AUC against the distant labels"""
    import math as _m
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    nD = len(RD); trip = collections.defaultdict(set); fsub = collections.defaultdict(lambda: collections.defaultdict(set))
    fA = collections.defaultdict(collections.Counter); fB = collections.defaultdict(collections.Counter)
    for r in RD: trip[(r[1], r[3], r[2])].add(r[0]); fsub[r[3]][r[1]].add(r[2]); fA[r[3]][_shape(r[1])] += 1; fB[r[3]][_shape(r[2])] += 1
    FUN = {f_: float(np.mean([len(v) == 1 for v in d_.values()])) for f_, d_ in fsub.items() if len(d_) >= 3}
    ent = lambda c_: -sum(v / sum(c_.values()) * _m.log2(v / sum(c_.values())) for v in c_.values())
    SEL = {f_: -(ent(fA[f_]) + ent(fB[f_])) for f_ in fA if sum(fA[f_].values()) >= 5}
    trs = rs.permutation(nD)[:int(getattr(cfg, 'inf_premises_typed_train', 200000))]
    t1 = np.array([float(len(trip[(RD[i][1], RD[i][3], RD[i][2])] - {RD[i][0]}) > 0) for i in trs])
    t2 = np.array([FUN.get(RD[i][3], np.nan) for i in trs]); t3 = np.array([SEL.get(RD[i][3], np.nan) for i in trs])
    rkn = lambda x: (lambda z: np.argsort(np.argsort(z, kind='stable'), kind='stable') / max(1, len(z) - 1))(np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(x).any() else 0.0, x))
    teach = rkn(t2) + rkn(t3) + 2 * t1
    HV = HashingVectorizer(n_features=2 ** 16, ngram_range=(1, 2), alternate_sign=False, token_pattern=r"[A-Za-z]+|[^\sA-Za-z]", norm='l2')
    txt = lambda i: RD[i][3] + ' FIRST_' + (RD[i][3].split() or ['none'])[0] + ' LAST_' + (RD[i][3].split() or ['none'])[-1] + ' SA_' + _shape(RD[i][1]) + ' SB_' + _shape(RD[i][2])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(HV.transform([txt(i) for i in trs]), teach >= np.percentile(teach, 75))
    s0 = lr.decision_function(HV.transform([txt(i) for i in trs])); yy = np.array([bool(RD[i][6]) for i in trs])
    score = lambda ix: lr.decision_function(HV.transform([txt(i) for i in ix])) if len(ix) else np.zeros(0)
    return score, float(np.quantile(s0, 0.75)), _cg_auc(s0[yy], s0[~yy]) if yy.any() and (~yy).any() else float('nan')


def _rtc_step(model, cfg, S_slot, seen, RD, PR, PRR, dn, NI, nbm, nkeep, vis):
    """14 Nov PREDICT, THEN CONFIRM (inf_rti_confirm; THE ONE MODEL CHANGE of cg27a_rtc against cg27a_rti). Every A-C the store infers
    (through a shared name, inf_rti_bridges+ bridges) is held as a PREDICTION, nothing written. When the store later reads a NEW document in
    which A and C are stated together in a RELATIONAL wording (the label-free scorer's top quarter; not one of the premise documents), the
    prediction is CONFIRMED: the confirming sentence joins the trunk's real-text replay and the inference is written as 'A is connected to C'
    (a third held back by its own hash: read, not taught). Offline (tools/chain_bootstrap.py, three seeds) confirmed inferences are right
    0.36 against 0.105 for all. READOUT controls: UNPREDICTED facts - new relational A-C records no prediction anticipated - are read the
    same way and never taught."""
    t0 = time.time(); rs = np.random.RandomState(int(cfg.seed) + 5151 + vis); LAB = _PM['LAB']; nD = len(RD)
    PEND = _PM.setdefault('rtc_pend', {}); W = _PM.setdefault('rti', {}); pmax = int(getattr(cfg, 'inf_rtc_pend_max', 400000))
    nd_now = (max(r[0] for r in RD) + 1) if RD else 0; nd_old = _PM.get('rtc_nd', nd_now)
    newi = [i for i in range(nD) if RD[i][0] >= nd_old] if nd_old < nd_now else []
    # ---- CONFIRM: predictions from earlier visits stated in a new document in a relational wording
    cand = [i for i in newi if frozenset((RD[i][1], RD[i][2])) in PEND and PEND[frozenset((RD[i][1], RD[i][2]))]['vis'] < vis
            and RD[i][0] not in PEND[frozenset((RD[i][1], RD[i][2]))]['docs']]
    score, thr, auc0 = _relscorer(cfg, RD, rs) if newi else (None, 0.0, float('nan'))
    sc = score(cand) if cand else np.zeros(0)
    conf = {}
    for i, v in zip(cand, sc):
        k = frozenset((RD[i][1], RD[i][2]))
        if v >= thr: conf.setdefault(k, []).append(i)
    dsl = 1 if 1 in seen else seen[-1]; nmax = int(getattr(cfg, 'inf_rti_max', 200)); wr = []
    done = {frozenset(k) for k in W}
    for k, ix in sorted(conf.items(), key=lambda kv: -len(kv[1])):
        if k in done or len(wr) >= nmax: continue
        A_, C_ = PEND[k]['ac']; tr = k in LAB
        W[(A_, C_)] = dict(vis=vis, d=int(dsl), truth=tr, bridges=PEND[k]['nbr'], rank=-1, novel=False, pv=PEND[k]['vis'], ev=[RD[i][7] for i in ix[:3]])
        wr.append((k, tr, vis - PEND[k]['vis']))
    # ---- UNPREDICTED control: new relational A-C records that no prediction anticipated (read, never taught)
    nun = int(getattr(cfg, 'inf_rtc_unpred', 100)); un = []
    if newi and nun > 0:
        pool = [i for i in rs.permutation(newi)[:20 * nun] if frozenset((RD[i][1], RD[i][2])) not in PEND and (RD[i][1], RD[i][2]) not in W and (RD[i][2], RD[i][1]) not in W]
        sp = score(pool) if pool else np.zeros(0)
        for i, v in zip(pool, sp):
            if v >= thr and len(un) < nun:
                k = frozenset((RD[i][1], RD[i][2])); W[(RD[i][1], RD[i][2])] = dict(vis=vis, d=int(dsl), truth=k in LAB, bridges=0, rank=-1, novel=False, pv=-1, unpred=True, ev=[RD[i][7]])
                un.append(k in LAB)
    # ---- PREDICT: this visit's inferences join the pending predictions
    nnew = 0
    for k_, v_ in PR.items():
        if len(v_) < nbm or k_[0] not in NI or k_[1] not in NI: continue
        k = frozenset(k_)
        if k in PEND or k in done or len(PEND) >= pmax: continue
        PEND[k] = dict(vis=vis, ac=k_, nbr=len(v_), docs={RD[i][0] for i in PRR[k_]}); nnew += 1
    _PM['rtc_nd'] = nd_now
    # ---- readout
    lab_c = [RD[i][6] for i in cand]; mrel = sc >= thr if len(sc) else np.zeros(0, bool)
    rr = lambda L: f"{np.mean(L):.3f} of {len(L)}" if len(L) else "none"
    pv = [x[2] for x in wr]; ages = collections.Counter(pv)
    print(f"  [REAL-TEXT CONFIRMATION | visit {vis + 1}] PREDICTIONS pending {len(PEND)} (+{nnew} this visit from {nkeep} premises, {nbm}+ bridges) | NEW documents "
          f"{max(0, nd_now - nd_old)} ({len(newi)} records) -> {len(cand)} records state a pending A-C (labelled {rr([bool(x) for x in lab_c])}); relational wording "
          f"(label-free scorer, top quarter; its AUC on distant labels {auc0:.3f}) {int(mrel.sum())}, labelled {rr([bool(x) for x, m in zip(lab_c, mrel) if m])} vs other "
          f"{rr([bool(x) for x, m in zip(lab_c, mrel) if not m])} -> CONFIRMED and WRITTEN {len(wr)} (right {rr([x[1] for x in wr])}; visits from prediction to "
          f"confirmation {dict(sorted(ages.items()))}) | UNPREDICTED control (new relational records, read only) {len(un)}, right {rr(un)} | {len(W)} in all ({time.time() - t0:.0f} s)", flush=True)


def _rcode_step(model, cfg, S_slot, seen, RD, vis):
    """14 Nov INFERENCE BY COMPRESSING THE FACT TABLE (inf_rti_code; THE ONE MODEL CHANGE of cg27a_rcode against cg27a_rti). The record
    compressor codes which names go together independently given their classes, so the facts - where an inference lives (A-B, B-C make
    A-C cheap) - are never compressed. Here every name gets a low-rank CODE of the store's name x name record graph (PPMI, truncated SVD,
    inf_rcode_k dimensions; no labels). HELD-OUT documents (inf_rcode_hold of the distant documents read, by hash): their records leave the
    graph, and so does every record anywhere pairing two names that co-occur in one of them, so nothing in the store states a candidate pair.
    Candidates: pairs co-occurring in a held-out document, both names in the graph; truth: the distant labels (evaluation only). The
    inf_rti_max candidates with the most similar codes are WRITTEN as 'A is connected to C' (a third held back, inf_rti_ctl). Offline
    (tools/fact_compress.py, Re-DocRED, two splits): codes right 0.32-0.37 in their top 5% where a chain exists (3.8x base) against 0.24 for
    the best chaining score; 2x base where no chain exists (chaining at chance)."""
    import zlib
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    t0 = time.time(); LAB = _PM['LAB']; W = _PM.setdefault('rti', {}); nD = len(RD); hf = float(getattr(cfg, 'inf_rcode_hold', 0.1))
    hold = lambda d: zlib.crc32(b'rcode%d' % int(d)) % 1000 < hf * 1000
    HD = np.array([hold(r[0]) for r in RD], bool)
    co = collections.defaultdict(set)                                         # held-out documents: the names they hold together
    for i in np.where(HD)[0]: co[RD[i][0]].update((RD[i][1], RD[i][2]))
    CAND = set()
    for ns in co.values():
        ns = sorted(ns)
        if len(ns) > 60: ns = ns[:60]
        CAND.update(frozenset((ns[x], ns[y])) for x in range(len(ns)) for y in range(x + 1, len(ns)))
    NI = {}; ia, ib = [], []
    for i in range(nD):
        if HD[i] or RD[i][1] == RD[i][2] or frozenset((RD[i][1], RD[i][2])) in CAND: continue
        ia.append(NI.setdefault(RD[i][1], len(NI))); ib.append(NI.setdefault(RD[i][2], len(NI)))
    NN = len(NI); G = csr_matrix((np.ones(len(ia), np.float32), (np.array(ia), np.array(ib))), shape=(NN, NN)); G = G + G.T; G.sum_duplicates(); G.data[:] = 1.0
    deg = np.asarray(G.sum(1)).ravel(); Gc = G.tocoo(); tot = float(G.sum())
    v = np.log(tot * Gc.data / (deg[Gc.row] * deg[Gc.col]) + 1e-12); mk = v > 0
    Pm = csr_matrix((v[mk].astype(np.float32), (Gc.row[mk], Gc.col[mk])), shape=(NN, NN))
    k = int(getattr(cfg, 'inf_rcode_k', 128)); U = TruncatedSVD(k, random_state=int(cfg.seed)).fit_transform(Pm).astype(np.float32)
    U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9); t1 = time.time() - t0
    ks = [c for c in CAND if all(n in NI and deg[NI[n]] > 0 for n in c)]
    if not ks:
        print(f"  [COMPRESSED FACT TABLE | visit {vis + 1}] no candidates ({len(CAND)} held-out pairs)", flush=True); return
    XA = np.array([NI[sorted(c)[0]] for c in ks]); XB = np.array([NI[sorted(c)[1]] for c in ks]); yy = np.array([c in LAB for c in ks])
    code = np.einsum('ij,ij->i', U[XA], U[XB])
    rs = np.random.RandomState(int(cfg.seed) + 31 + vis); sub = rs.permutation(len(ks))[:200000]
    cn_ = np.asarray(G[XA[sub]].multiply(G[XB[sub]]).sum(1)).ravel()
    iw = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0).astype(np.float32)
    aa = np.asarray(G[XA[sub]].multiply(G[XB[sub]]).multiply(iw[None, :]).sum(1)).ravel()
    def line(mask, title):
        yv = yy[sub][mask]
        if yv.sum() == 0 or (~yv).sum() == 0: return f"{title}: {int(mask.sum())} pairs, labelled {yv.mean() if len(yv) else float('nan'):.3f}"
        out = []
        for nm, sv in (('random', rs.rand(int(mask.sum()))), ('degree product', (deg[XA[sub]] * deg[XB[sub]])[mask]), ('Adamic-Adar', aa[mask]), (f'CODE {k}', code[sub][mask])):
            o = np.argsort(-sv + 1e-9 * rs.rand(len(sv))); out.append(f"{nm} AUC {_cg_auc(sv[yv], sv[~yv]):.3f}, top 5/10/25% "
                                                                      + "/".join(f"{yv[o[:max(1, len(o) * q // 100)]].mean():.3f}" for q in (5, 10, 25)))
        return f"{title}: {int(mask.sum())} pairs, labelled {yv.mean():.3f} - " + "; ".join(out)
    sel_ = str(getattr(cfg, 'inf_rcode_sel', 'code')); selmsg = ''
    if sel_ in ('code+aa', 'code+ra') or int(getattr(cfg, 'inf_rcode_selreport', 0)):   # 14 Nov SELECTION (cg27a_rcode3): the code and Adamic-Adar agree (rank sum);
        aa_all = np.zeros(len(ks), np.float32)                                #   offline (Re-DocRED gold, two splits) the top 100 right 0.42-0.45 vs 0.38-0.41
        for s0 in range(0, len(ks), 100000):                                  #   for the code alone (top 50: 0.50-0.62 vs 0.32-0.46)
            sl0 = slice(s0, s0 + 100000); aa_all[sl0] = np.asarray(G[XA[sl0]].multiply(G[XB[sl0]]).multiply(iw[None, :]).sum(1)).ravel()
        ra_all = np.zeros(len(ks), np.float32); iw1 = (1.0 / np.maximum(deg, 1)).astype(np.float32)   # 14 Nov RESOURCE ALLOCATION (shared neighbours weighted 1/degree):
        for s0 in range(0, len(ks), 100000):                                  #   offline (two splits) code + RA holds precision as more is written - level with
            sl0 = slice(s0, s0 + 100000); ra_all[sl0] = np.asarray(G[XA[sl0]].multiply(G[XB[sl0]]).multiply(iw1[None, :]).sum(1)).ravel()   #   code + AA at 100, +5-7 at 200
        rkf = lambda x: np.argsort(np.argsort(x, kind='stable'), kind='stable') / max(1, len(x) - 1); both = rkf(code) + rkf(aa_all); bra = rkf(code) + rkf(ra_all)
        top = lambda sv, n: f"{yy[np.argsort(-sv)[:n]].mean():.3f}"
        selmsg = (" || SELECTION RULES, labelled among the top 100 / 200 / 400 of all candidates: code " + " / ".join(top(code, n) for n in (100, 200, 400)) + "; Adamic-Adar "
                  + " / ".join(top(aa_all, n) for n in (100, 200, 400)) + "; code + Adamic-Adar (rank sum) " + " / ".join(top(both, n) for n in (100, 200, 400))
                  + "; code + resource allocation (rank sum) " + " / ".join(top(bra, n) for n in (100, 200, 400)) + f" (writing by: {sel_})")
    done = {frozenset(x) for x in W}; o = np.argsort(-(both if sel_ == 'code+aa' else bra if sel_ == 'code+ra' else code)); nmax = int(getattr(cfg, 'inf_rti_max', 200)); dsl = 1 if 1 in seen else seen[-1]; wr = []
    for j in o:
        if len(wr) >= nmax: break
        c = ks[j]
        if c in done: continue
        A_, C_ = sorted(c); W[(A_, C_)] = dict(vis=vis, d=int(dsl), truth=bool(yy[j]), bridges=int(G[NI[A_]].multiply(G[NI[C_]]).sum()), rank=-1, novel=True, code=float(code[j]))
        wr.append(bool(yy[j]))
    print(f"  [COMPRESSED FACT TABLE | visit {vis + 1}] store graph {NN} names, {int(G.nnz // 2)} name pairs from {len(ia)} records (held out: {int(HD.sum())} records of "
          f"{len(co)} documents, and every record pairing two names of one of them) -> CODES {k} dims ({t1:.0f} s) | candidates never stated together {len(ks)} "
          f"(labelled {yy.mean():.3f}) || " + " || ".join([line(np.ones(len(sub), bool), 'ALL'), line(cn_ > 0, 'WITH A BRIDGE'), line(cn_ == 0, 'NO BRIDGE')])
          + selmsg + f" || WROTE the top {len(wr)} by {sel_}: right {np.mean(wr) if wr else float('nan'):.3f} ({len(W)} in all) ({time.time() - t0:.0f} s)", flush=True)


def rti_evidence(cfg, S_slot, seen):
    """13 Nov EVIDENCE REPLAY (inf_rti_evidence; THE ONE MODEL CHANGE of cg27a_rti2): the premise statements of every written inference
    (taught and held back alike) join the real-text replay pool of the inference's slot, so the trunk reads the evidence behind what it is
    taught (tools/false_fade.py: right inferences keep their taught gain only when later reading supports them; wrong ones fade either way)"""
    from platypus.inference.distill import _rtx_slot                          # the slot real text is replayed through (cg27a_rti: Wikipedia's, inf_distill_rtx_slot 1)
    out = {}
    for k, w_ in (_PM.get('rti') or {}).items():
        sl_ = _rtx_slot(cfg, S_slot, w_['d'], seen)
        if sl_ is None: continue
        for sp in w_.get('ev', ()): out.setdefault(int(sl_), []).append(sp.encode('utf-8', 'replace'))
    return out


def rti_write(model, cfg, S_slot, ents, held, ctl, vis):
    """THE ONE MODEL CHANGE of cg27a_rti (inf_rti_write 1): real-text inferences that passed the check join the question pool as facts
    'A is connected to C', taught by the interleaved rows like any fact; a fifth is held back by the pool's own hash (control). READOUT: recall
    from the trunk (test wording 'It is on record that A is connected to ', store and memory off) of inferences written at an earlier readout,
    taught vs held back, split by whether the inference is RIGHT, WRONG (a false memory if recalled) or unknown."""
    import zlib
    from platypus.inference.distill import _dt_recall
    W = _PM.get('rti') or {}
    if not W:
        print(f"  [REAL-TEXT INFERENCE WRITES | visit {vis + 1}] none yet", flush=True); return ents, held
    have = {repr(e['key']) for e in ents + held}; T_, H_, U_ = [], [], []
    for (A_, C_), w in W.items():
        e = dict(key=('fact', A_, _RTI_R, C_), A=A_, r=_RTI_R, ans=C_, d=w['d'], pfs=[f"{A_} {_RTI_R} "], seen=2, src=(), rm=w['vis'], truth=w['truth'], novel=w.get('novel'))
        if repr(e['key']) in have: continue
        hc = float(getattr(cfg, 'inf_rti_ctl', 0.0))                          # 13 Nov: its own held-back share (a third in cg27a_rti2), else the pool's fifth
        held_ = (zlib.crc32((repr(e['key']) + '|rti').encode()) % 1000 < hc * 1000) if hc > 0 else ctl(e)
        if w.get('unpred'): U_.append(e); continue                            # 14 Nov cg27a_rtc control: unpredicted facts are read, never taught
        (H_ if held_ else T_).append(e)
    old = lambda L, t: [e for e in L if e['rm'] < vis and e['truth'] is t]
    g = lambda d: f"{d['facts'][0]:.2f} ({d['facts'][1]})"
    rec = {t: (_dt_recall(model, S_slot, old(T_, t)), _dt_recall(model, S_slot, old(H_, t))) for t in (True, False, None)}
    print(f"  [REAL-TEXT INFERENCE WRITES | visit {vis + 1}] {len(W)} inferences written ({sum(1 for w in W.values() if w['vis'] == vis)} this readout) -> question pool: {len(T_)} taught, "
          f"{len(H_)} held back || FROM THE TRUNK, written at an earlier readout, TAUGHT vs HELD BACK: RIGHT inferences {g(rec[True][0])} vs {g(rec[True][1])} | WRONG ones "
          f"(false memories if recalled) {g(rec[False][0])} vs {g(rec[False][1])} | not checkable {g(rec[None][0])} vs {g(rec[None][1])}", flush=True)
    if int(getattr(cfg, 'inf_rti_novel', 0)):                                 # 14 Nov: the NOVEL ones only (A and C never together in anything the trunk reads)
        nv = lambda L: [e for e in L if e.get('novel')]
        rn = {t: (_dt_recall(model, S_slot, old(nv(T_), t)), _dt_recall(model, S_slot, old(nv(H_), t))) for t in (True, False, None)}
        print(f"  [REAL-TEXT INFERENCE WRITES, NOVEL ONLY | visit {vis + 1}] {len(nv(T_))} taught, {len(nv(H_))} held back | TAUGHT vs HELD BACK: RIGHT {g(rn[True][0])} vs "
              f"{g(rn[True][1])} | WRONG {g(rn[False][0])} vs {g(rn[False][1])} | not checkable {g(rn[None][0])} vs {g(rn[None][1])}", flush=True)
    if int(getattr(cfg, 'inf_rti_rank_recall', 0)):                           # 14 Nov RANK RECALL (readout): open-ended exact recall cannot see real multi-word
        from platypus.inference.distill import _dw_cand_mc                     #   names (offline: rank 0.74 while exact 0.01); here the answer among itself and
        pool_ = sorted({e['ans'] for e in H_}) or sorted({e['ans'] for e in T_ + H_ + U_})   # distractors: HELD-BACK answers only (never taught), so taught
        rr_ = np.random.RandomState(int(cfg.seed) + 77 + vis)                     #   and held-back facts face the same untaught distractors (chance 0.10)
        def rk_(L):
            L = [e for e in L if e['rm'] < vis][:150]; hit = []
            for e in L:
                alts = [x for x in pool_ if x not in (e['ans'], e['A'])]
                if len(alts) < 9: continue
                cs = [e['ans']] + [alts[j] for j in rr_.choice(len(alts), 9, replace=False)]
                hit.append(int(_dw_cand_mc(model, int(S_slot(e['d'])), f"It is on record that {e['A']} {e['r']} ", cs) == 0))
            return f"{np.mean(hit):.2f} ({len(hit)})" if hit else "nan (0)"
        tr_ = lambda L, t: [e for e in L if e['truth'] is t]
        dl_ = (model.__dict__.get('_dtw') or {}).get('rtil') or []
        print(f"  [REAL-TEXT INFERENCE RANK RECALL | visit {vis + 1}] the answer first among itself and 9 held-back answers (chance 0.10), from the trunk, test wording: TAUGHT right "
              f"{rk_(tr_(T_, True))} / wrong {rk_(tr_(T_, False))} | HELD BACK right {rk_(tr_(H_, True))} / wrong {rk_(tr_(H_, False))}"
              + (f" | UNPREDICTED right {rk_(tr_(U_, True))} / wrong {rk_(tr_(U_, False))}" if U_ else '')
              + (f" || dose rows last visit: loss {np.mean(dl_[:20]):.3f} (first 20) -> {np.mean(dl_[-20:]):.3f} (last 20), {len(dl_)} steps" if dl_ else ''), flush=True)
    if U_:
        ru = {t: _dt_recall(model, S_slot, old(U_, t)) for t in (True, False)}
        print(f"  [REAL-TEXT CONFIRMATION RECALL | visit {vis + 1}] PREDICTED and confirmed: taught RIGHT {g(rec[True][0])} / WRONG {g(rec[False][0])} | held back (read the "
              f"confirming sentence only) RIGHT {g(rec[True][1])} / WRONG {g(rec[False][1])} || UNPREDICTED (read only) RIGHT {g(ru[True])} / WRONG {g(ru[False])}", flush=True)
    coh = sorted({e['rm'] for e in T_ + H_ if e['rm'] < vis})                 # 13 Nov: by WHEN the inference was written (fresh ones may be recalled, then fade)
    if coh:
        by = lambda L, t, v: [e for e in L if e['truth'] is t and e['rm'] == v]
        print(f"  [REAL-TEXT INFERENCE WRITES BY AGE | visit {vis + 1}] TAUGHT vs HELD BACK, right / wrong: " + " | ".join(
            f"written at visit {v + 1}: right {g(_dt_recall(model, S_slot, by(T_, True, v)))} vs {g(_dt_recall(model, S_slot, by(H_, True, v)))}, wrong "
            f"{g(_dt_recall(model, S_slot, by(T_, False, v)))} vs {g(_dt_recall(model, S_slot, by(H_, False, v)))}" for v in coh), flush=True)
    return ents + T_, held + H_ + U_
