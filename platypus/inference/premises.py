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
    _PM['list2'] = bool(int(getattr(cfg, 'inf_rret_list2', 0)))
    if _PM.get('path') == p: return
    _PM.update(path=p, R=[], TY={}, LAB=set(), ND=0)
    if not p: return
    import os
    if not os.path.exists(p):
        print(f"  [PREMISES AT SCALE] distant set not found at {p}: benchmark documents only (the notebook downloads it)", flush=True); return
    t0 = time.time(); D = json.load(gzip.open(p, 'rt') if p.endswith('.gz') else open(p)); R = []; TY = collections.defaultdict(collections.Counter); LAB = set(); PRD = {}
    cx9 = {} if int(getattr(cfg, 'inf_premises_seed_ctx', 0)) else None        # 20 Nov kra14: 3 words of context each side, keyed by the record (seeded reader)
    for di in range(len(D)):
        d = D[di]; D[di] = None
        cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
        lab = {(l['h'], l['t']) for l in d['labels']} | {(l['t'], l['h']) for l in d['labels']}
        for l in d['labels']: LAB.add(frozenset((cn[l['h']], cn[l['t']]))); PRD.setdefault((cn[l['h']], cn[l['t']]), set()).add(l['r'])   # PRD: ordered relations (evaluation only)
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
                    if cx9 is not None: cx9[id(R[-1])] = ' '.join(w.lower() for w in toks[max(0, a0 - 3):a0]) + ' ' + ' '.join(w.lower() for w in toks[b1:b1 + 3])
        if int(getattr(cfg, 'inf_premises_wide', 0)):                         # 19 Nov kra13 WIDER RECORDS (offline: labelled pairs covered 0.32 -> 0.52): the same sentence
            S9 = d['sents']; sub9 = None                                       #   13-30 words apart, the next sentence, and pronoun-led sentences (the article's subject);
            for si, ms in sorted(bys.items()):                                 #   their length field is 13+ so the relational checks (1-8 words) never count them
                ms = sorted(ms)
                if si == min(bys) and ms: sub9 = ms[0][2]
                for x in range(len(ms)):
                    for y in range(x + 1, len(ms)):
                        (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y]
                        if ea == eb or a1 > b0 or cn[ea] == cn[eb] or not (12 < b0 - a1 <= 30): continue
                        mw = [w.lower() for w in S9[si][a1:a1 + 6]] + ['...'] + [w.lower() for w in S9[si][b0 - 6:b0]]
                        R.append((di, cn[ea], cn[eb], ' '.join(mw), mw[-1], b0 - a1, (ea, eb) in lab, f"{cn[ea]} ... {cn[eb]}"))
                if si + 1 in bys:
                    for (a0, a1, ea) in ms[-3:]:
                        for (b0, b1, eb) in sorted(bys[si + 1])[:3]:
                            if ea == eb or cn[ea] == cn[eb] or b0 > 15: continue
                            mw = [w.lower() for w in S9[si][a1:a1 + 8]] + ['<s>'] + [w.lower() for w in S9[si + 1][:b0]]
                            R.append((di, cn[ea], cn[eb], ' '.join(mw), mw[-1], 13 + len(mw), (ea, eb) in lab, f"{cn[ea]} <s> {cn[eb]}"))
                if sub9 is not None and si != min(bys) and S9[si] and S9[si][0].lower() in ('he', 'she', 'it', 'they', 'his', 'her', 'its', 'their'):
                    for (b0, b1, eb) in ms[:4]:
                        if eb == sub9 or cn[eb] == cn[sub9] or b0 > 15: continue
                        mw = ['<pron>'] + [w.lower() for w in S9[si][1:b0]]
                        R.append((di, cn[sub9], cn[eb], ' '.join(mw), mw[-1], 13 + len(mw), (sub9, eb) in lab, f"{cn[sub9]} <pron> {cn[eb]}"))
    _PM.update(R=R, TY={n: c.most_common(1)[0][0] for n, c in TY.items()}, LAB=LAB, ND=di + 1, PRD=PRD, CTX=cx9 or {})
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
    if int(getattr(cfg, 'inf_premises_subst', 0)) and len(RB) >= 100:          # 19 Nov kra8 SUBSTITUTION READER (label-free; gold for evaluation): how much the TRUNK's
        try:                                                                     #   prediction of B depends on A - bits on B's bytes with A as written vs A swapped for a
            SS = _PM.setdefault('ss', {}); ns = int(getattr(cfg, 'inf_premises_subst', 0)); rr = np.random.RandomState(int(cfg.seed) + 99)   #   same-type name of another
            pool = collections.defaultdict(list)                                 #   record (offline with a 192 x 3 stand-in: chance, AUC 0.51 - the stand-in knows no facts)
            for r in RB: pool[r[10][0]].append(r[1])
            todo = [i for i, r in enumerate(RB) if (r[0], r[1], r[2], r[7]) not in SS and r[7].startswith(r[1]) and r[7].endswith(r[2]) and len(r[7]) > len(r[1]) + len(r[2])]
            bysl = collections.defaultdict(list)
            for i in todo: bysl[int(S_slot(RB[i][8]))].append(i)
            was = model.training; model.eval()
            try:
                for sl, ix in bysl.items():
                    seqs = []
                    for i in ix:
                        r = RB[i]; tb = r[7].encode('latin-1', 'replace'); seqs.append((tb, len(tb) - len(r[2]), len(tb)))
                        for _ in range(ns):
                            pl = pool.get(r[10][0]) or pool[next(iter(pool))]; z = pl[rr.randint(len(pl))]
                            if z == r[1]: z = pl[rr.randint(len(pl))]
                            sb2 = (z + r[7][len(r[1]):]).encode('latin-1', 'replace'); seqs.append((sb2, len(sb2) - len(r[2]), len(sb2)))
                    bits = _cg_bits(model, sl, seqs, bs=256)
                    for k, i in enumerate(ix):
                        v = bits[k * (ns + 1):(k + 1) * (ns + 1)]; SS[(RB[i][0], RB[i][1], RB[i][2], RB[i][7])] = (float(np.mean(v[1:]) - v[0]) / max(1, len(RB[i][2])), float(v[0]) / max(1, len(RB[i][2])))
            finally: model.train(was)
            ok = np.array([(r[0], r[1], r[2], r[7]) in SS for r in RB]); yb = np.array([bool(r[6]) for r in RB])[ok]
            if ok.sum() >= 100 and 0 < yb.sum() < len(yb):
                gv = np.array([SS[(r[0], r[1], r[2], r[7])][0] for r, k in zip(RB, ok) if k]); ov = np.array([SS[(r[0], r[1], r[2], r[7])][1] for r, k in zip(RB, ok) if k])
                dv = -np.array([r[5] for r, k in zip(RB, ok) if k], float)
                pdocs = collections.defaultdict(set)
                for r in RD: pdocs[frozenset((r[1], r[2]))].add(('d', r[0]))
                for r in RB: pdocs[frozenset((r[1], r[2]))].add(('b', r[0]))
                ag = np.array([len(pdocs[frozenset((r[1], r[2]))] - {('b', r[0])}) for r, k in zip(RB, ok) if k], float)
                rk2 = lambda x: np.argsort(np.argsort(x, kind='stable'), kind='stable').astype(float)
                def a9(v): o = np.argsort(-v); return f"AUC {_cg_auc(v[yb], v[~yb]):.3f} (top 10% {yb[o[:max(1, len(v) // 10)]].mean():.2f})"
                msg.append(f"SUBSTITUTION READER, benchmark records {int(ok.sum())} (gold-related {yb.mean():.3f}; {ns} swaps): GAIN bits/byte of B when A is swapped {a9(gv)} | "
                           f"B's own surprise {a9(-ov)} | distance {a9(dv)} | agreement (other documents) {a9(np.log1p(ag))} | gain + distance + agreement (ranks) {a9(rk2(gv) + rk2(dv) + rk2(np.log1p(ag)))}")
        except Exception as e: _fail('inference.premises:subst'); msg.append(f"SUBSTITUTION READER unavailable ({type(e).__name__}: {e})"[:160])
    scE = {}
    if int(getattr(cfg, 'inf_premises_wording', 0)) and len(RB) >= 100 and nD >= 1000:   # 19 Nov kra9 SELF-TRAINED WORDING READER (label-free; labels for evaluation):
        try:                                                                   #   usefulness over time (would the stored pair predict the other name when one turns up
            from sklearn.feature_extraction.text import HashingVectorizer       #   again? hits minus misses, weighted by surprise) gives pseudo-labels on the store's
            from sklearn.linear_model import LogisticRegression                  #   records seen in other documents; a logistic regression on the words between the
            pdw = collections.defaultdict(set); ndw = collections.defaultdict(set)   # names learns which wordings state a relation, and scores every record, including
            for r in RD: pdw[frozenset((r[1], r[2]))].add(('d', r[0])); ndw[r[1]].add(('d', r[0])); ndw[r[2]].add(('d', r[0]))   # pairs seen once (offline 50k:
            for r in RB: pdw[frozenset((r[1], r[2]))].add(('b', r[0])); ndw[r[1]].add(('b', r[0])); ndw[r[2]].add(('b', r[0]))   # AUC 0.646, with usefulness 0.672)
            NDw = max(2, len({x for v in ndw.values() for x in v}))
            def usew(r, own):
                a_ = len(pdw[frozenset((r[1], r[2]))] - {own}); nA, nB = len(ndw[r[1]] - {own}), len(ndw[r[2]] - {own})
                if not a_: return np.nan
                hA, hB = a_ / (nA + 2), a_ / (nB + 2); sA, sB = np.log(NDw / (nA + 2)), np.log(NDw / (nB + 2))
                return max(hA * sB - (1 - hA) * 0.5, hB * sA - (1 - hB) * 0.5)
            txw = lambda r: f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(int(r[5]), 12) // 3}"
            hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
            cut = int(ndv * 0.8); trI = [i for i in range(nD) if RD[i][0] < cut]; teI = [i for i in range(nD) if RD[i][0] >= cut]
            uT = np.array([usew(RD[i], ('d', RD[i][0])) for i in trI]); okT = ~np.isnan(uT)
            if okT.sum() >= 200:
                hi_, lo_ = np.quantile(uT[okT], 0.8), np.quantile(uT[okT], 0.5); ps9 = [i for i, u in zip(trI, uT) if u >= hi_]; ng9 = [i for i, u in zip(trI, uT) if u <= lo_]
                kn9 = rv9 = 0
                if int(getattr(cfg, 'inf_premises_knowledge', 0)):            # 19 Nov kra13: every training record of a CONFIRMED real-text inference is a positive
                    cfk = {frozenset(k) for k, w in (_PM.get('rti') or {}).items() if w.get('state') == 'confirmed'}   #   (offline: +0.004-0.006 AUC, steady)
                    kp9 = [i for i in trI if frozenset((RD[i][1], RD[i][2])) in cfk]; kn9 = len(kp9); ngs = set(ng9)
                    ps9 = sorted(set(ps9) | set(kp9)); ng9 = [i for i in ng9 if i not in set(kp9)]
                if int(getattr(cfg, 'inf_premises_reverb', 0)):               # 19 Nov kra13: ReVerb relation phrases (verb / verb + preposition / verb + words +
                    try:                                                       #   preposition, used with 10+ distinct name pairs) as extra positives (offline: phrase
                        import spacy                                           #   recall 0.111 -> 0.181 at no AUC cost)
                        nlp8 = _PM.get('nlp_pos') or spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer']); _PM['nlp_pos'] = nlp8
                        phr = [i for i in trI if RD[i][5] >= 2 and not _rret_listlike(RD[i]) and '(' not in RD[i][3]]
                        pbm = collections.defaultdict(set)
                        for i in phr: pbm[RD[i][3]].add(frozenset((RD[i][1], RD[i][2])))
                        cm8 = [m for m, v in pbm.items() if len(v) >= int(getattr(cfg, 'inf_premises_reverb', 10))]; ok8 = set()
                        for m, dd in zip(cm8, nlp8.pipe(cm8, batch_size=1024)):
                            tg = [t.pos_ for t in dd]
                            if tg and tg[0] in ('VERB', 'AUX') and (len(tg) == 1 or (len(tg) == 2 and tg[1] in ('ADP', 'PART')) or (tg[-1] == 'ADP' and all(t in ('NOUN', 'ADJ', 'ADV', 'PRON', 'DET', 'VERB', 'AUX', 'PROPN', 'PART') for t in tg[1:-1]))): ok8.add(m)
                        ngs = set(ng9); rvp = [i for i in phr if RD[i][3] in ok8 and i not in ngs]; rv9 = len(rvp); ps9 = sorted(set(ps9) | set(rvp))
                    except Exception as e8: _fail('inference.premises:reverb'); msg.append(f"REVERB unavailable ({type(e8).__name__}: {e8})"[:120])
                sd9 = int(getattr(cfg, 'inf_premises_seed', 0))
                if sd9 > 0:                                                    # 20 Nov kra14 SEEDED READER: a small LABELLED seed (records of documents already read, with
                    CX9 = _PM.get('CTX') or {}                                  #   their own-document labels) replaces the label-free teacher, which only hurt once a
                    if CX9: txw = lambda r: f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(int(r[5]), 12) // 3} " + ' '.join('C_' + w for w in CX9.get(id(r), '').split())
                    sdI = [trI[j] for j in np.random.RandomState(int(cfg.seed) + 707 + vis).permutation(len(trI))[:sd9]]   # seed was given (offline, 30k documents: AUC
                    ps9 = [i for i in sdI if RD[i][6]]; ng9 = [i for i in sdI if not RD[i][6]]   # 0.667 / 0.694 / 0.732 with context at 2k / 5k / 20k labels; label-free 0.63-0.65)
                    msg.append(f"SEEDED READER: {len(sdI)} labelled records ({len(ps9)} related) from documents already read, context words {'on' if CX9 else 'off'}; label-free teaching off")
                clf = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); clf.fit(hv.transform([txw(RD[i]) for i in ps9 + ng9]), np.r_[np.ones(len(ps9)), np.zeros(len(ng9))])
                for _ in range(0 if sd9 > 0 else int(getattr(cfg, 'inf_premises_mi', 0))):      # 19 Nov kra13: MULTI-INSTANCE ('at least one'): a positive pair keeps only its mention the reader
                    sc8 = clf.decision_function(hv.transform([txw(RD[i]) for i in ps9])); bag = {}   #   scores highest (offline: pair AUC 0.654 -> 0.664, phrase recall +1/3)
                    for i, v8 in zip(ps9, sc8):
                        k8 = frozenset((RD[i][1], RD[i][2]))
                        if k8 not in bag or v8 > bag[k8][1]: bag[k8] = (i, v8)
                    ps9 = sorted(v[0] for v in bag.values()); clf = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced')
                    clf.fit(hv.transform([txw(RD[i]) for i in ps9 + ng9]), np.r_[np.ones(len(ps9)), np.zeros(len(ng9))])
                if float(getattr(cfg, 'inf_rcode_reader', 0.0)) > 0:          # the reader's score for every store record (the compressed fact table's graph, kra13)
                    _PM['rdscore'] = np.concatenate([clf.decision_function(hv.transform([txw(RD[i]) for i in range(j, min(nD, j + 200000))])) for j in range(0, nD, 200000)])
                if kn9 or rv9: msg.append(f"READER TEACHING: + {kn9} records of confirmed inferences, + {rv9} ReVerb phrase records, multi-instance rounds {int(getattr(cfg, 'inf_premises_mi', 0))}")
                def ev(L, own_of, yy):
                    sw = clf.decision_function(hv.transform([txw(r) for r in L])); dv_ = -np.array([r[5] for r in L], float)
                    uu = np.array([usew(r, own_of(r)) for r in L]); sn = ~np.isnan(uu); rk3 = lambda x: np.argsort(np.argsort(x, kind='stable'), kind='stable').astype(float)
                    cb = np.where(sn, rk3(sw) + rk3(np.nan_to_num(uu, nan=-9.0)), rk3(sw) * 0.5)
                    def a9(v, m=None):
                        v_, y_ = (v, yy) if m is None else (v[m], yy[m])
                        if y_.sum() == 0 or y_.sum() == len(y_): return 'n/a'
                        o = np.argsort(-v_); return f"AUC {_cg_auc(v_[y_], v_[~y_]):.3f} (top 5% {y_[o[:max(1, len(v_) // 20)]].mean():.2f})"
                    return (f"distance {a9(dv_)} | usefulness {a9(np.nan_to_num(uu, nan=-9.0))} | WORDING READER {a9(sw)} | wording + usefulness {a9(cb)} | "
                            f"pairs seen once: wording {a9(sw, ~sn)} (n {int((~sn).sum())})")
                yB = np.array([bool(r[6]) for r in RB]); yD = np.array([bool(RD[i][6]) for i in teI])
                if int(getattr(cfg, 'inf_premises_e2e', 0)) and scd is not None:   # 19 Nov kra11: the reader's scores for every scored distant record (end-to-end readout below)
                    swE = clf.decision_function(hv.transform([txw(RD[i]) for i in ixd])); uE = np.array([usew(RD[i], ('d', RD[i][0])) for i in ixd])
                    rkE = lambda x: np.argsort(np.argsort(x, kind='stable'), kind='stable').astype(float)
                    scE = {'WORDING READER': swE, 'WORDING + USEFULNESS': np.where(np.isnan(uE), rkE(swE) * 0.5, rkE(swE) + rkE(np.nan_to_num(uE, nan=-9.0)))}
                msg.append(f"WORDING READER, self-trained from usefulness ({len(ps9)} / {len(ng9)} pseudo-labels; true related among them {np.mean([RD[i][6] for i in ps9]):.2f} / "
                           f"{np.mean([RD[i][6] for i in ng9]):.2f}, evaluation only) || benchmark records {len(RB)} (gold-related {yB.mean():.3f}): {ev(RB, lambda r: ('b', r[0]), yB)} || "
                           f"held-out distant records {len(teI)} (own-document labels {yD.mean():.3f}): {ev([RD[i] for i in teI], lambda r: ('d', r[0]), yD)}")
                if int(getattr(cfg, 'inf_premises_states', 0)):                # 19 Nov kra9 STATE READER: the same pseudo-labels, but the reader reads the TRUNK'S OWN STATES
                    import torch; from platypus.inference.closed_form import _ws_batch, _ws_run   #   (after A, mean over the words between, just before B; last block) instead of word counts.
                    rr9 = np.random.RandomState(int(cfg.seed) + 31 + vis); nS = int(getattr(cfg, 'inf_premises_states', 0))   # Repeated at every premises readout: does the
                    def txt9(r):                                                #   reader improve as the trunk learns? (offline 192 x 3 stand-ins at 6000 / 18000 steps)
                        if len(r) > 7 and isinstance(r[7], str) and r[7].startswith(r[1]) and r[7].endswith(r[2]) and len(r[7]) > len(r[1]) + len(r[2]): return r[7]
                        return f"{r[1]} {r[3]} {r[2]}".replace(' ,', ',').replace(' .', '.').replace(" 's", "'s")
                    def feats(L, slot_of):
                        out = np.zeros((len(L), 0), np.float32); cap = {}; h9 = model.blocks[-1].register_forward_hook(lambda m_, i_, o_: cap.__setitem__('h', o_.detach()))
                        was = model.training; model.eval(); rows = {}
                        try:
                            by9 = collections.defaultdict(list)
                            for i, r in enumerate(L): by9[slot_of(r)].append(i)
                            for sl, ix in by9.items():
                                for i0 in range(0, len(ix), 128):
                                    Lm = max(8, min(120, int(getattr(model.cfg, 'seq_len', 64)) - 1))   # _ws_batch keeps the FIRST seq_len bytes: cut from the left here
                                    ch = ix[i0:i0 + 128]; fb = [txt9(L[i]).encode('latin-1', 'replace') for i in ch]; tb = [x[-Lm:] for x in fb]; X, _ = _ws_batch(model, tb)
                                    with torch.no_grad(): _ws_run(model, X, sl)
                                    for j, i in enumerate(ch):
                                        r = L[i]; b = tb[j]; ct = len(fb[j]) - len(b); H = cap['h'][j].float(); nH = H.shape[0]
                                        lb = min(nH - 1, max(0, len(b) - len(r[2].encode('latin-1', 'replace')) - 1))
                                        la = min(lb, max(0, len(r[1].encode('latin-1', 'replace')) - 1 - ct))
                                        rows[i] = torch.cat([H[la], H[la:lb + 1].mean(0), H[lb]]).cpu().numpy()
                        finally: h9.remove(); model.train(was)
                        return np.stack([rows[i] for i in range(len(L))])
                    k9 = min(nS, len(ps9), len(ng9)); pS = [ps9[i] for i in rr9.choice(len(ps9), k9, replace=False)]; nN = [ng9[i] for i in rr9.choice(len(ng9), k9, replace=False)]
                    tS = [teI[i] for i in rr9.choice(len(teI), min(len(teI), 2 * nS), replace=False)]
                    sld = int(S_slot(seen[-1])) if seen else 0
                    Ftr = feats([RD[i] for i in pS + nN], lambda r: sld); mu9, sd9 = Ftr.mean(0), Ftr.std(0) + 1e-6
                    cS = LogisticRegression(C=0.1, max_iter=500, class_weight='balanced'); cS.fit((Ftr - mu9) / sd9, np.r_[np.ones(k9), np.zeros(k9)])
                    FB = (feats(RB, lambda r: int(S_slot(r[8])) if len(r) > 8 else sld) - mu9) / sd9; FD = (feats([RD[i] for i in tS], lambda r: sld) - mu9) / sd9
                    sB9, sD9 = cS.decision_function(FB), cS.decision_function(FD); yD9 = np.array([bool(RD[i][6]) for i in tS])
                    swB = clf.decision_function(hv.transform([txw(r) for r in RB])); swD = clf.decision_function(hv.transform([txw(RD[i]) for i in tS]))
                    def a8(v, yy):
                        if yy.sum() == 0 or yy.sum() == len(yy): return 'n/a'
                        o = np.argsort(-v); return f"AUC {_cg_auc(v[yy], v[~yy]):.3f} (top 5% {yy[o[:max(1, len(v) // 20)]].mean():.2f})"
                    rk8 = lambda x: np.argsort(np.argsort(x, kind='stable'), kind='stable').astype(float)
                    _PM.setdefault('states_curve', []).append((vis + 1, _cg_auc(sB9[yB], sB9[~yB]) if 0 < yB.sum() < len(yB) else float('nan')))
                    msg.append(f"STATE READER (the trunk's own states, last block; {k9} + {k9} pseudo-labelled training records) || benchmark records: STATES {a8(sB9, yB)} | word counts {a8(swB, yB)} | "
                               f"states + words (ranks) {a8(rk8(sB9) + rk8(swB), yB)} || held-out distant {len(tS)}: STATES {a8(sD9, yD9)} | word counts {a8(swD, yD9)} | "
                               f"states + words {a8(rk8(sD9) + rk8(swD), yD9)} || CURVE over readouts (visit: benchmark AUC): " + ", ".join(f"{v_}: {a_:.3f}" for v_, a_ in _PM['states_curve']))
        except Exception as e: _fail('inference.premises:wording'); msg.append(f"WORDING READER unavailable ({type(e).__name__}: {e})"[:160])
        if int(getattr(cfg, 'inf_premises_parse', 0)) and locals().get('clf') is not None:     # 19 Nov kra10 PARSER READER (readout; spaCy en_core_web_sm, an outside parser, by the user's
            try:                                                               #   decision): the dependency path between the two names in the record's text (labels, the
                import spacy                                                   #   word where the branches meet, length, pure coordination), self-trained from the same
                nlp9 = _PM.get('nlp') or spacy.load('en_core_web_sm', disable=['ner']); _PM['nlp'] = nlp9   # usefulness pseudo-labels, alone and with the word counts
                CO9 = {'conj', 'cc', 'punct', 'appos', 'compound', 'npadvmod'}; nP = int(getattr(cfg, 'inf_premises_parse', 0)); rp9 = np.random.RandomState(int(cfg.seed) + 53 + vis)
                def pth(r, dd):
                    t9 = r[7] if len(r) > 7 and isinstance(r[7], str) else ''
                    sa, sb = dd.char_span(0, len(r[1]), alignment_mode='expand'), dd.char_span(len(t9) - len(r[2]), len(t9), alignment_mode='expand')
                    if sa is None or sb is None or not t9.startswith(r[1]) or not t9.endswith(r[2]): return 'PATH_none'
                    i, j = sa.root, sb.root; anc = {}; t, n = i, 0
                    while True:
                        anc[t.i] = n
                        if t.head.i == t.i or n > 30: break
                        t = t.head; n += 1
                    dn = []; t = j
                    while t.i not in anc and t.head.i != t.i and len(dn) < 30: dn.append(t.dep_); t = t.head
                    if t.i not in anc: return 'PATH_none'
                    lca = t; up = []; t = i
                    while t.i != lca.i: up.append(t.dep_); t = t.head
                    dp = up + dn[::-1]
                    return (f"PATH_{'^'.join(up)}|{lca.lemma_.lower()}|{'v'.join(dn[::-1])} LCA_{lca.lemma_.lower()} PLEN_{min(len(dp), 8)} "
                            f"COORD_{int(bool(dp) and all(x in CO9 for x in dp))}")
                def pfe(L): return [pth(r, dd) for r, dd in zip(L, nlp9.pipe([r[7] if len(r) > 7 and isinstance(r[7], str) else '' for r in L], batch_size=256))]
                k8 = min(nP, len(ps9), len(ng9)); pP = [ps9[i] for i in rp9.choice(len(ps9), k8, replace=False)]; nP_ = [ng9[i] for i in rp9.choice(len(ng9), k8, replace=False)]
                tP = [teI[i] for i in rp9.choice(len(teI), min(len(teI), 2 * nP), replace=False)]
                trL = [RD[i] for i in pP + nP_]; ytr = np.r_[np.ones(k8), np.zeros(k8)]; teL = [RD[i] for i in tP]; yP = np.array([bool(r[6]) for r in teL]); yB = np.array([bool(r[6]) for r in RB])
                ptr, pB_, pD_ = pfe(trL), pfe(RB), pfe(teL)
                cP = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); cP.fit(hv.transform(ptr), ytr)
                cWP = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); cWP.fit(hv.transform([txw(r) + ' ' + p for r, p in zip(trL, ptr)]), ytr)
                def a7(v, yy):
                    if yy.sum() == 0 or yy.sum() == len(yy): return 'n/a'
                    o = np.argsort(-v); return f"AUC {_cg_auc(v[yy], v[~yy]):.3f} (top 5% {yy[o[:max(1, len(v) // 20)]].mean():.2f})"
                co8 = lambda P_: np.array([' COORD_1' in x for x in P_])
                msg.append(f"PARSER READER (spaCy dependency path, self-trained from usefulness; {k8} + {k8} records; coordination paths {co8(pB_).mean():.2f} of benchmark records, "
                           f"related among them {yB[co8(pB_)].mean() if co8(pB_).any() else float('nan'):.2f} vs {yB[~co8(pB_)].mean() if (~co8(pB_)).any() else float('nan'):.2f}) || "
                           f"benchmark: PATH {a7(cP.decision_function(hv.transform(pB_)), yB)} | WORDS {a7(clf.decision_function(hv.transform([txw(r) for r in RB])), yB)} | "
                           f"WORDS + PATH {a7(cWP.decision_function(hv.transform([txw(r) + ' ' + p for r, p in zip(RB, pB_)])), yB)} || held-out distant {len(teL)}: "
                           f"PATH {a7(cP.decision_function(hv.transform(pD_)), yP)} | WORDS {a7(clf.decision_function(hv.transform([txw(r) for r in teL])), yP)} | "
                           f"WORDS + PATH {a7(cWP.decision_function(hv.transform([txw(r) + ' ' + p for r, p in zip(teL, pD_)])), yP)}")
            except Exception as e: _fail('inference.premises:parse'); msg.append(f"PARSER READER unavailable ({type(e).__name__}: {e})"[:160])
    # ---- INFERENCE: A-C through a shared name, store kept by each policy (distant scored records; distant labels)
    if scd is not None:
        try:
            dn = collections.defaultdict(set)
            for i in range(nD): dn[RD[i][1]].add(RD[i][0]); dn[RD[i][2]].add(RD[i][0])
            LAB = _PM['LAB']; res = []; amax = int(getattr(cfg, 'inf_premises_ac_max', 600000))
            sub = np.sort(np.random.RandomState(int(cfg.seed) + 4242).permutation(len(ixd))[:amax])   # a fixed sample of the scored store (time)
            sc3 = dict(scd); sc3.update(scE); E2 = bool(scE)
            for k in (('random', 'trunk surprise', 'block association') + (tuple(scE) if E2 else ())):
                for frac in ((0.25, 0.10, 0.02) if E2 else (0.25, 0.10)):
                    keep = ixd[sub][np.argsort(-sc3[k][sub])[:int(len(sub) * frac)]]; nb = collections.defaultdict(list); PR = collections.defaultdict(set); np_ = 0
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


def rti_ent(k, w, **kw):
    """16 Nov: the pool entry of a written inference. Identity (key) always ('fact', A, _RTI_R, C) with A, C the sorted pair; with TYPED
    WRITES (inf_rti_typed) the question is 'head <wording> ' and the answer the tail (w['th'], w['tw'], w['tt']), else 'A is connected to C'"""
    A_, C_ = k
    if w.get('tw'): h_, r_, t_ = w['th'], w['tw'], w['tt']
    else: h_, r_, t_ = A_, _RTI_R, C_
    e = dict(key=('fact', A_, _RTI_R, C_), A=h_, r=r_, ans=t_, d=w['d'], rti=True)
    e.update(kw); return e


def _rnorm(w):
    return ' '.join(t for t in w.split() if t not in ('the', 'a', 'an'))


def is_rti(e):
    return bool(e.get('rti')) or e.get('r') == _RTI_R


def rti_wkey(e):
    """the ledger key (A, C) of an inference entry"""
    k = e.get('key')
    return (k[1], k[3]) if k and k[0] == 'fact' and len(k) > 3 and k[2] == _RTI_R else (e['A'], e['ans'])


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
    ndv_ = (max(r[0] for r in RD) + 1) if RD else 0; _PM['ndv_now'] = int(ndv_)   # 17 Nov: composed writes are judged from here
    if int(getattr(cfg, 'inf_rret', 0)):
        try: _rret_judge(cfg, RD, ndv_, vis)
        except Exception as e9:
            _fail('inference.premises:rret'); print(f"  [RETRACTION TRIGGER | visit {vis + 1}] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
    hold = lambda d: zlib.crc32(b'rcode%d' % int(d)) % 1000 < hf * 1000
    HD = np.array([hold(r[0]) for r in RD], bool)
    co = collections.defaultdict(set)                                         # held-out documents: the names they hold together
    for i in np.where(HD)[0]: co[RD[i][0]].update((RD[i][1], RD[i][2]))
    CAND = set()
    for ns in co.values():
        ns = sorted(ns)
        if len(ns) > 60: ns = ns[:60]
        CAND.update(frozenset((ns[x], ns[y])) for x in range(len(ns)) for y in range(x + 1, len(ns)))
    NI = {}; ia, ib = [], []; rq9 = float(getattr(cfg, 'inf_rcode_reader', 0.0)); rds9 = _PM.get('rdscore')   # 19 Nov kra13: the graph from the READER's top share
    rthr9 = float(np.quantile(rds9, 1 - rq9)) if rq9 > 0 and rds9 is not None and len(rds9) == nD else None       #   (offline: top half sharpens the top 1% 0.17 -> 0.24)
    for i in range(nD):
        if HD[i] or RD[i][1] == RD[i][2] or frozenset((RD[i][1], RD[i][2])) in CAND: continue
        if rthr9 is not None and rds9[i] < rthr9: continue
        ia.append(NI.setdefault(RD[i][1], len(NI))); ib.append(NI.setdefault(RD[i][2], len(NI)))
    NN = len(NI); G = csr_matrix((np.ones(len(ia), np.float32), (np.array(ia), np.array(ib))), shape=(NN, NN)); G = G + G.T; G.sum_duplicates(); G.data[:] = 1.0
    deg = np.asarray(G.sum(1)).ravel(); Gc = G.tocoo(); tot = float(G.sum())
    v = np.log(tot * Gc.data / (deg[Gc.row] * deg[Gc.col]) + 1e-12); mk = v > 0
    Pm = csr_matrix((v[mk].astype(np.float32), (Gc.row[mk], Gc.col[mk])), shape=(NN, NN))
    k = int(getattr(cfg, 'inf_rcode_k', 128)); U = TruncatedSVD(k, random_state=int(cfg.seed)).fit_transform(Pm).astype(np.float32)
    U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9); t1 = time.time() - t0
    ks = [c for c in sorted(CAND, key=lambda c: tuple(sorted(c))) if all(n in NI and deg[NI[n]] > 0 for n in c)]   # 26 Nov: sorted, so argsort ties do not depend on the hash seed
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
        asy_ = np.abs(np.log((deg[XA] + 1.0) / (deg[XB] + 1.0)))              # 14 Nov ASYMMETRY (inf_rcode_asym): one name clearly more specific than the other
        top = lambda sv, n: f"{yy[np.argsort(-sv)[:n]].mean():.3f}"
        selmsg = (" || SELECTION RULES, labelled among the top 100 / 200 / 400 of all candidates: code " + " / ".join(top(code, n) for n in (100, 200, 400)) + "; Adamic-Adar "
                  + " / ".join(top(aa_all, n) for n in (100, 200, 400)) + "; code + Adamic-Adar (rank sum) " + " / ".join(top(both, n) for n in (100, 200, 400))
                  + "; code + resource allocation (rank sum) " + " / ".join(top(bra, n) for n in (100, 200, 400))
                  + "; the same among pairs with |log degree ratio| > 1: code + AA " + " / ".join(top(np.where(asy_ > 1, both, -9), n) for n in (100, 200, 400))
                  + ", code + RA " + " / ".join(top(np.where(asy_ > 1, bra, -9), n) for n in (100, 200, 400)) + f" (writing by: {sel_})")
    sc_ = (both if sel_ == 'code+aa' else bra if sel_ == 'code+ra' else code); ath = float(getattr(cfg, 'inf_rcode_asym', 0.0))
    if ath > 0:                                                               # 14 Nov ASYMMETRY FILTER (cg27a_rasym): write only pairs whose names differ in store
        asy0 = np.abs(np.log((deg[XA] + 1.0) / (deg[XB] + 1.0)))              #   degree by more than e^ath (offline, Re-DocRED gold, two splits: top 200
        sc_ = np.where(asy0 > ath, sc_, -np.inf)                               #   0.585 / 0.525 vs 0.395; GPU path: wrong 0.28 vs 0.42)
    if _PM.get('save_cand'):                                                   # 15 Nov (offline tools/learned_selector.py): every candidate's write-time features
        _PM['cand'] = dict(pairs=[tuple(sorted(c)) for c in ks], code=np.asarray(code, np.float32), aa=np.asarray(aa_all, np.float32), ra=np.asarray(ra_all, np.float32),
                           dA=deg[XA].astype(np.float32), dB=deg[XB].astype(np.float32), y=yy.copy(), rule=np.asarray(sc_, np.float32), ndv=int(ndv_))
    TT_ = None
    if int(getattr(cfg, 'inf_rti_typed', 0)):                                  # 16 Nov TYPED WRITES: the class pair's commonest relational wording in the store,
        TT_ = collections.defaultdict(collections.Counter); TYP_ = _PM.get('TY', {})   #   head = the more specific name (lower store degree)
        for r in RD:
            if not (1 <= r[5] <= 8) or _rret_listlike(r) or r[1] not in NI or r[2] not in NI: continue
            d1, d2 = deg[NI[r[1]]], deg[NI[r[2]]]
            if d1 == d2: continue
            hd_, tl_, fw_ = (r[1], r[2], 1) if d1 < d2 else (r[2], r[1], 0)
            w_ = ' '.join(t for t in r[3].split() if t not in ('the', 'a', 'an'))
            w9_ = ' '.join(t for t in w_.replace('"', ' ').split()).strip(' ,;()')
            if w_ and any(ch.isalpha() for ch in w_) and w9_ not in _RRET_LISTW and w9_ not in ('and', 'or') and (not int(getattr(cfg, 'inf_rti_clean', 0)) or (w_[0].isalnum() and w_[-1].isalnum())): TT_[(TYP_.get(hd_, 'X'), TYP_.get(tl_, 'X'))][(w_, fw_)] += 1
    done = {frozenset(x) for x in W}; o = np.argsort(-sc_); o = o[np.isfinite(sc_[o])]; nmax = int(getattr(cfg, 'inf_rti_max', 200)); dsl = 1 if 1 in seen else seen[-1]; wr = []
    if int(getattr(cfg, 'inf_rti_adapt', 0)) and _PM.get('rti_budget'): nmax = int(_PM['rti_budget'])   # 16 Nov ADAPTIVE BUDGET (set by the last judgement)
    for j in o:
        if len(wr) >= nmax: break
        c = ks[j]
        if c in done: continue
        A_, C_ = sorted(c); cm_ = G[NI[A_]].multiply(G[NI[C_]]).indices
        if int(getattr(cfg, 'inf_rti_nameshape', 0)) and any(any(ch.isdigit() or ch in '()"' for ch in n_) for n_ in (A_, C_)): continue   # 21 Nov kra22 NAME SHAPE: no numbers, years or fragments as names (hand audit: half the truly false)
        if 'INV' not in _PM or len(_PM['INV']) != len(NI): _PM['INV'] = {i: n for n, i in NI.items()}
        W[(A_, C_)] = dict(vis=vis, d=int(dsl), truth=bool(yy[j]), bridges=int(len(cm_)), rank=len(wr), novel=True, code=float(code[j]), ndv=int(ndv_),
                           prem=[_PM['INV'][int(x)] for x in cm_[:5]], rule=str(sel_), score=float(sc_[j]), state='inferred')   # 15 Nov PROVENANCE: premises (bridges), rule, score
        if int(getattr(cfg, 'inf_rti_tentative', 0)): W[(A_, C_)].update(tent=True)   # 19 Nov kra12: TENTATIVE until later reading confirms it (not taught, not in the memory)
        if TT_ is not None:
            hd_, tl_ = (A_, C_) if deg[NI[A_]] <= deg[NI[C_]] else (C_, A_); cp_ = (_PM['TY'].get(hd_, 'X'), _PM['TY'].get(tl_, 'X'))
            if TT_.get(cp_):
                (w_, fw_), _ = TT_[cp_].most_common(1)[0]; th_, tt_ = (hd_, tl_) if fw_ else (tl_, hd_)
                W[(A_, C_)].update(th=th_, tw=w_, tt=tt_)
        wr.append(bool(yy[j]))
    if int(getattr(cfg, 'inf_rti_chain', 0)):                                  # 20 Nov kra16 CHAIN FORM (inf_rti_chain): each new inference keeps its premises as
        new8 = [k for k, w in W.items() if w.get('vis') == vis and w.get('prem')]   #   text ('<A> <wording> <B>. <B> <wording> <C>.', the commonest recorded wording
        need8 = {frozenset((a, b)) for a, c in new8 for b in W[(a, c)]['prem']} | {frozenset((b, c)) for a, c in new8 for b in W[(a, c)]['prem']}   #   of each pair,
        pw8 = collections.defaultdict(collections.Counter)                     #   through the first bridge with a record on both sides); the dose rows of a taught
        for r in RD:                                                           #   inference are then given half the time after its premises (distill.py), so the
            if 1 <= r[5] <= 8 and not _rret_listlike(r) and any(ch.isalpha() for ch in r[3]):   #   trunk learns the composition step, not only the conclusion
                f8 = frozenset((r[1], r[2]))                                   #   (the stand-in, tools/chain_scale.py: chain rows lift never-taught chains given
                if f8 in need8: pw8[f8][(r[1], _rnorm(r[3]), r[2])] += 1       #   in context 0.110 -> 0.135)
        nch8 = 0
        for (A_, C_) in new8:
            for b in W[(A_, C_)]['prem']:
                c1, c2 = pw8.get(frozenset((A_, b))), pw8.get(frozenset((b, C_)))
                if c1 and c2:
                    p1, p2 = c1.most_common(1)[0][0], c2.most_common(1)[0][0]; W[(A_, C_)]['chain'] = f"{p1[0]} {p1[1]} {p1[2]}. {p2[0]} {p2[1]} {p2[2]}."
                    W[(A_, C_)]['chain_p'] = (f"{p1[0]} {p1[1]} {p1[2]}.", f"{p2[0]} {p2[1]} {p2[2]}."); nch8 += 1; break   # 20 Nov kra20: the two premises apart (the broken-chain control)
        print(f"  [CHAIN FORM | visit {vis + 1}] {nch8} of {len(new8)} new inferences keep their premises as text (both sides on record)", flush=True)
    if int(getattr(cfg, 'inf_rti_comp', 0)):                                   # 16 Nov TYPED COMPOSITION (inf_rti_comp): the premises' relational wordings (A-B, B-C,
        new9 = [k for k, w in W.items() if w.get('vis') == vis]                #   oriented) at write time; the table learned from kept, confirmed inferences types
        need9 = {frozenset((a, b)) for a, c in new9 for b in W[(a, c)]['prem']} | {frozenset((b, c)) for a, c in new9 for b in W[(a, c)]['prem']}   #   them where it can
        pw9 = collections.defaultdict(list)
        for r in RD:
            if 1 <= r[5] <= 8 and not _rret_listlike(r):
                f9 = frozenset((r[1], r[2]))
                if f9 in need9: pw9[f9].append((r[1], r[2], _rnorm(r[3])))
        def ori9(X, Y):
            cs = pw9.get(frozenset((X, Y)))
            if not cs: return None
            a, b, w = collections.Counter(cs).most_common(1)[0][0]; return w if a == X else w + ' ^R'
        T3 = _PM.get('T3') or {}; ncomp = 0
        for (A_, C_) in new9:
            pw = [(ori9(A_, b), ori9(b, C_)) for b in W[(A_, C_)]['prem']]; W[(A_, C_)]['pw'] = pw; votes = collections.Counter()
            for a1, a2 in pw:
                if a1 and a2: votes.update(T3.get((a1, a2), {}))
            if votes:
                cwd = votes.most_common(1)[0][0]; rv = cwd.endswith(' ^R'); w0 = cwd[:-3] if rv else cwd
                W[(A_, C_)].update(th=C_ if rv else A_, tw=w0, tt=A_ if rv else C_, comp=True); ncomp += 1
        print(f"  [TYPED COMPOSITION | visit {vis + 1}] {ncomp} of {len(new9)} new inferences typed by the composition table ({len(T3)} cells; learned from kept, confirmed "
              f"inferences) | premises with a relational wording {np.mean([p is not None for k in new9 for q in W[k]['pw'] for p in q]) if new9 else 0:.2f}", flush=True)
    if TT_ is not None:
        ty_ = [(k, w) for k, w in W.items() if w.get('vis') == vis and w.get('tw')]
        print(f"  [TYPED WRITES | visit {vis + 1}] {len(ty_)} of {len(wr)} written inferences typed (class pair -> the store's commonest relational wording, head = the more "
              f"specific name): " + " | ".join(f"{w['th']} {w['tw']} {w['tt']} ({'right' if w['truth'] else 'wrong'})" for k, w in ty_[:6])
              + " | wordings: " + ", ".join(f"'{a}' {n}" for a, n in collections.Counter(w['tw'] for k, w in ty_).most_common(6)), flush=True)
    print(f"  [COMPRESSED FACT TABLE | visit {vis + 1}] store graph {NN} names, {int(G.nnz // 2)} name pairs from {len(ia)} records (held out: {int(HD.sum())} records of "
          f"{len(co)} documents, and every record pairing two names of one of them) -> CODES {k} dims ({t1:.0f} s) | candidates never stated together {len(ks)} "
          f"(labelled {yy.mean():.3f}) || " + " || ".join([line(np.ones(len(sub), bool), 'ALL'), line(cn_ > 0, 'WITH A BRIDGE'), line(cn_ == 0, 'NO BRIDGE')])
          + selmsg + f" || WROTE the top {len(wr)} by {sel_}{(' among |log degree ratio| > %g' % float(getattr(cfg, 'inf_rcode_asym', 0.0))) if float(getattr(cfg, 'inf_rcode_asym', 0.0)) > 0 else ''}: right {np.mean(wr) if wr else float('nan'):.3f} ({len(W)} in all) ({time.time() - t0:.0f} s)", flush=True)


def _rret_key_held(cfg, A_, C_, tag):
    import zlib
    k = repr(('fact', A_, _RTI_R, C_)); hc = float(getattr(cfg, 'inf_rti_ctl', 0.0)) if tag == '|rti' else float(getattr(cfg, 'inf_rret_ctl', 0.5))
    return zlib.crc32((k + tag).encode()) % 1000 < hc * 1000


_RRET_LISTW = {',', 'and', 'or', '(', ')', '-', ', and', ', or', ') and', ') ,', ', (', 'and the', ', the', ';', '/'}


def _rret_listlike(r):
    """15 Nov: a record whose names are linked only by a list or coordination ('A, B and C'), not by relational wording"""
    if r[3] in _RRET_LISTW or r[3] == '' or str(r[7]).count(',') >= 2: return True
    if _PM.get('list2'):                                                       # 16 Nov WIDER LIST FILTER (inf_rret_list2): 'X , Y and Z' (one comma + and / or)
        w = ' ' + r[3] + ' '
        return bool(re.search(r' , .*\b(and|or)\b', w)) or bool(re.search(r'\b(and|or)\b.* , ', w)) or r[3].strip() in ('and', 'or', ', and', ', or', 'as well as', 'and the')
    return False


def _rret_judge(cfg, RD, ndv_, vis):
    """14 Nov RETRACTION TRIGGER (inf_rret; THE ONE MODEL CHANGE of cg27a_rret). Each written inference not yet judged, written at an earlier
    readout, is judged by what the store has read SINCE it was written (documents ndv at the write .. now), label-free: SHORT = records stating
    the pair with 3 or fewer words between the names; PMI = log((documents stating the pair + 0.5) x documents read / (documents naming A + 1)
    (documents naming C + 1)). Rank sum within the batch; the least confirmed inf_rret_q are FLAGGED. Flagged TAUGHT inferences are
    RETRACTED - no longer taught, unlikelihood rows on their answer - except a control share (inf_rret_ctl, own hash) kept as before.
    Offline (tools/confirm_trigger.py, Re-DocRED gold, two splits, 200 writes each, 25k documents read after): AUC 0.72-0.73, the bottom 20%
    wrong 0.88 / 0.93 against 0.63 / 0.59 overall (likelihood ratio 4-8; re-ranking by the store's own score: about 2)."""
    W = _PM.get('rti') or {}
    if int(getattr(cfg, 'inf_rti_tentative', 0)):                              # 19 Nov kra12 CONFIRM, CONSOLIDATE, FADE (the user's design; offline tools/confirm_reader.py:
        TN = [k for k, w in W.items() if w.get('tent') and w['vis'] < vis and w.get('ndv', ndv_) < ndv_]   #   relationally confirmed pairs right 0.60-0.72 vs 0.33 tentative;
        rcv = int(getattr(cfg, 'inf_rti_reconf', 0))                           # 21 Nov kra21 RE-CONFIRMATION (inf_rti_reconf = R visits; MODEL CHANGE when > 0): a confirmed
        RC = [k for k, w in W.items() if rcv > 0 and w.get('state') == 'confirmed' and not w.get('ret') and vis - int(w.get('conf', vis)) >= rcv]   #   inference whose pair has
        if not TN and not RC:                                                  #   NO relational record in the documents read since its confirmation, R visits on, is RETRACTED:
            print(f"  [CONFIRM AND CONSOLIDATE | visit {vis + 1}] nothing tentative to judge ({len(W)} written)", flush=True); return   # no longer taught + unlikelihood rows
        dc9 = collections.defaultdict(set); cdn9 = int(getattr(cfg, 'inf_rti_conf_docs', 0)); rw9 = collections.defaultdict(list)   # 21 Nov kra22 INDEPENDENT CONFIRMATION (inf_rti_conf_docs = D): also D+ distinct documents
        want = {frozenset(k): k for k in TN}; rel9 = collections.Counter(); any9 = collections.Counter(); cm9 = int(getattr(cfg, 'inf_rti_conf_min', 1))   #   (inf_rret). Offline
        wantR = {frozenset(k): k for k in RC}; rec9 = collections.Counter()      #   tools/retract_min.py: removes 0.35-0.38 of the wrong confirmed, 0.19 of the right
        rds8 = _PM.get('rdscore'); rt8 = None                                  # 20 Nov kra17 RATED CONFIRMATION (inf_rti_conf_rated): only a record the seeded reader
        if int(getattr(cfg, 'inf_rti_conf_rated', 0)):                         #   puts in its top half confirms (offline tools/confirm_min.py, 2,400 writes, 30k
            if rds8 is not None and len(rds8) == len(RD): rt8 = float(np.quantile(rds8, float(getattr(cfg, 'inf_rti_conf_q', 0.5))))   # documents read after: right 0.604 -> 0.659,
            else: print(f"  [CONFIRM AND CONSOLIDATE | visit {vis + 1}] rated confirmation unavailable (reader scores {None if rds8 is None else len(rds8)} vs {len(RD)} records): any relational record confirms", flush=True)
        for i8, r in enumerate(RD):                                            #   right ones kept 0.62 -> 0.56; kra21: inf_rti_conf_q 0.75 = the top quarter, 0.708 -> 0.765)
            fk = frozenset((r[1], r[2]))
            if fk in want and r[0] >= W[want[fk]].get('ndv', 0):
                any9[fk] += 1
                if 1 <= r[5] <= 8 and not _rret_listlike(r) and (rt8 is None or rds8[i8] >= rt8): rel9[fk] += 1
                if 1 <= r[5] <= 8 and not _rret_listlike(r): dc9[fk].add(int(r[0]))   # 21 Nov kra22: distinct documents with a relational record (any rating)
                if 1 <= r[5] <= 8 and not _rret_listlike(r) and (rt8 is None or rds8[i8] >= rt8): rw9[fk].append((r[1], _rnorm(r[3]), r[2]))   # 21 Nov kra23: confirming wordings   #   a RELATIONAL record of the pair in a document read since the write
            if fk in wantR and r[0] >= W[wantR[fk]].get('ndc', ndv_) and 1 <= r[5] <= 8 and not _rret_listlike(r): rec9[fk] += 1   # since its confirmation (any rating)
        rt9 = []
        for k in RC:
            if rec9[frozenset(k)] == 0: W[k].update(state='retracted', flag=True, ret=True, retv=vis); rt9.append(W[k]['truth'])
        if rcv > 0:
            print(f"  [RE-CONFIRMATION | visit {vis + 1}] MODEL CHANGE: {len(RC)} inferences confirmed {rcv}+ visits ago checked for a relational record of the pair since: "
                  f"RETRACTED (none) {len(rt9)} (right {np.mean([bool(x) for x in rt9]) if rt9 else float('nan'):.2f}) | re-confirmed {len(RC) - len(rt9)} "
                  f"(right {np.mean([bool(W[k]['truth']) for k in RC if not W[k].get('ret')]) if len(RC) > len(rt9) else float('nan'):.2f}) || all retracted so far "
                  f"{sum(1 for w in W.values() if w.get('state') == 'retracted')}", flush=True)
        if not TN:
            print(f"  [CONFIRM AND CONSOLIDATE | visit {vis + 1}] nothing tentative to judge ({len(W)} written)", flush=True); return
        fv = int(getattr(cfg, 'inf_rti_fade', 4)); cf9, fd9, st9 = [], [], []
        for k in TN:
            w = W[k]; fk = frozenset(k)
            if rel9[fk] >= cm9 and (cdn9 <= 1 or len(dc9[fk]) >= cdn9): w.update(tent=False, conf=vis, ndc=ndv_, state='confirmed', judged=vis, flag=False, ret=False); cf9.append(w['truth'])
            if rel9[fk] >= cm9 and (cdn9 <= 1 or len(dc9[fk]) >= cdn9) and int(getattr(cfg, 'inf_rti_conf_retype', 0)) and rw9.get(fk):   # 21 Nov kra23 RE-WORD ON
                cl9 = [x for x in rw9[fk] if 1 <= len(x[1].split()) <= 4 and all(t.isalpha() for t in x[1].split()) and x[1] not in _RRET_LISTW and x[1] not in ('and', 'or', 'with', 'to')]
                if cl9:                                                        #   CONFIRMATION: the confirming record's own clean wording and order replace the generic typed
                    X9, w9_, Y9 = collections.Counter(cl9).most_common(1)[0][0]; w.update(th=X9, tw=w9_, tt=Y9, retyped=vis)   #   one ('in'); kra22: two taught answers for one
                                                                               #   subject and wording were recalled 0.44-0.46 vs 0.78 without a conflict
            elif vis - w['vis'] >= fv: w.update(tent=False, faded=vis, state='faded', judged=vis, flag=True, ret=False); fd9.append(w['truth'])
            else: st9.append(w['truth'])
        rr = lambda L: f"{len(L)} (right {np.mean([bool(x) for x in L]):.2f})" if L else "0"
        allc = [w['truth'] for w in W.values() if w.get('state') == 'confirmed']; allf = [w['truth'] for w in W.values() if w.get('state') == 'faded']
        print(f"  [CONFIRM AND CONSOLIDATE | visit {vis + 1}] MODEL CHANGE: tentative inferences judged on documents read since their write (a relational record of the pair "
              f"confirms; unconfirmed after {fv} visits fade - removed, never taught against): CONFIRMED now {rr(cf9)} | FADED now {rr(fd9)} | still tentative {rr(st9)} || "
              f"ALL SO FAR: confirmed {rr(allc)}, faded {rr(allf)}, tentative {rr([w['truth'] for w in W.values() if w.get('tent')])} | any later record of the pair "
              f"(plain recurrence, evaluation) {sum(1 for k in TN if any9[frozenset(k)])} of {len(TN)}", flush=True)
        return
    groups = collections.defaultdict(list)
    for k, w in W.items():
        if w.get('judged') is None and w['vis'] < vis and 'ndv' in w and w['ndv'] < ndv_: groups[w['ndv']].append(k)
    if not groups:
        print(f"  [RETRACTION TRIGGER | visit {vis + 1}] nothing to judge ({len(W)} written)", flush=True); return
    q = float(getattr(cfg, 'inf_rret_q', 0.2)); out = []
    for lo, ks in sorted(groups.items()):
        nd = max(1, ndv_ - lo); df = collections.Counter(); byp = collections.defaultdict(lambda: [set(), 0]); seenpair = set()
        want = {frozenset(k) for k in ks}; nm = {n for k in ks for n in k}; dn = set(); relA = collections.Counter(); relS = collections.Counter(); cw9 = collections.defaultdict(list)
        hd9 = {(W[k].get('th') or k[0]): k for k in ks}; tl9 = {(W[k].get('tt') or k[1]) for k in ks}; HF9 = collections.defaultdict(list); LK9 = set()   # 16 Nov TYPED CONFIRMATION
        for r in RD:
            if 1 <= r[5] <= 8 and not _rret_listlike(r):                       #   (inf_rret_sig 'rel+cc'): head-first records since the write; pairs the store
                if r[0] >= lo and r[1] in hd9: HF9[r[1]].append(r[2])          #   linked before it (a container of the tail is no contradiction)
                elif r[0] < lo and (r[1] in tl9 or r[2] in tl9): LK9.add(frozenset((r[1], r[2])))
            if frozenset((r[1], r[2])) in want and 1 <= r[5] <= 8 and not _rret_listlike(r):   # 15 Nov: RELATIONAL records of the pair (not lists)
                relA[frozenset((r[1], r[2]))] += 1; relS[frozenset((r[1], r[2]))] += int(r[0] >= lo)
                if r[0] >= lo: cw9[frozenset((r[1], r[2]))].append((r[1], r[2], _rnorm(r[3])))
            if r[0] < lo: continue
            for n in (r[1], r[2]):
                if n in nm and (r[0], n) not in dn: dn.add((r[0], n)); df[n] += 1
            fk = frozenset((r[1], r[2]))
            if fk in want:
                b = byp[fk]; b[0].add(r[0]); b[1] += int(r[5] <= 3)
        short = np.array([byp[frozenset(k)][1] if frozenset(k) in byp else 0 for k in ks], float)
        pmi = np.array([np.log((len(byp[frozenset(k)][0]) if frozenset(k) in byp else 0) + 0.5) + np.log(nd) - np.log(df[k[0]] + 1) - np.log(df[k[1]] + 1) for k in ks])
        rk = lambda x: np.argsort(np.argsort(x + 1e-9 * np.random.RandomState(7).rand(len(x)), kind='stable'), kind='stable') / max(1, len(x) - 1)
        docs = np.array([len(byp[frozenset(k)][0]) if frozenset(k) in byp else 0 for k in ks], float)
        if int(getattr(cfg, 'inf_rti_adapt', 0)):                              # 16 Nov ADAPTIVE BUDGET (offline: a precision dial, up to 0.93): blocks of 100 by
            stp = float(getattr(cfg, 'inf_rti_adapt_stop', 0.1)); nb9 = 0      #   write rank; the next write goes one block past the last leading block in which
            for b0 in range(0, 10000, 100):                                    #   at least inf_rti_adapt_stop of the inferences have a confirming relational record
                bl = [k for k in ks if b0 <= W[k]['rank'] < b0 + 100]
                if not bl or np.mean([relS[frozenset(k)] > 0 for k in bl]) < stp: break
                nb9 += 1
            _PM['rti_budget'] = int(min(int(getattr(cfg, 'inf_rti_adapt_max', 400)), max(100, 100 * (nb9 + 1))))
            out.append(f"ADAPTIVE BUDGET: {nb9} leading blocks of 100 with confirmations >= {stp:.2f} -> next write {_PM['rti_budget']}")
        sig_ = str(getattr(cfg, 'inf_rret_sig', 'short+pmi'))                  # 15 Nov: 'docs' = documents stating the pair since the write (offline on the
        ra_ = np.array([relA[frozenset(k)] for k in ks], float); rs_ = np.array([relS[frozenset(k)] for k in ks], float)
        def _cc9(k):
            hh, tt = (W[k].get('th') or k[0]), (W[k].get('tt') or k[1]); zs = HF9.get(hh, ())
            cf = sum(1 for z in zs if z == tt); ct = sum(1 for z in zs if z != tt and frozenset((z, tt)) not in LK9); return (cf - ct) / max(1, cf + ct)
        cc_ = np.array([_cc9(k) for k in ks], float)
        for j9, k9 in enumerate(ks): W[k9]['cc'] = float(cc_[j9])                 # 17 Nov: the typed confirmation score, kept for the keep rule (inf_kmc_useful 3)
        sc = (rk(docs) if sig_ == 'docs' else rk(ra_) if sig_ == 'rel' else rk(ra_) + rk(cc_) if sig_ == 'rel+cc' else rk(short) + rk(pmi)); thr = np.quantile(sc, q); fl = sc <= thr   # 'rel' (cg27a_kmr): relational
        yv_ = np.array([bool(W[k]['truth']) for k in ks]); au_ = lambda x: _cg_auc(x[yv_], x[~yv_]) if yv_.any() and (~yv_).any() else float('nan')   #   records (0.697 offline)
        y = np.array([bool(W[k]['truth']) for k in ks]); taught = np.array([not _rret_key_held(cfg, k[0], k[1], '|rti') for k in ks])
        ctl_ = np.array([_rret_key_held(cfg, k[0], k[1], '|rret') for k in ks]); ret = fl & taught & ~ctl_
        for j, k in enumerate(ks): W[k].update(judged=vis, flag=bool(fl[j]), ret=bool(ret[j]), rscore=float(sc[j]))
        if int(getattr(cfg, 'inf_rti_retype', 0)):                             # 16 Nov RETYPE ON CONFIRMATION: a kept inference that a later relational record
            nrt = 0                                                            #   states is rewritten in that record's wording and order
            for j, k in enumerate(ks):
                cs = cw9.get(frozenset(k))
                if fl[j] or not cs: continue
                X, Y, w = collections.Counter(cs).most_common(1)[0][0]
                if w and any(ch.isalpha() for ch in w): W[k].update(th=X, tw=w, tt=Y, retyped=vis); nrt += 1
            out.append(f"RETYPED ON CONFIRMATION {nrt} kept inferences (e.g. " + "; ".join(f"{W[k]['th']} {W[k]['tw']} {W[k]['tt']} ({'right' if W[k]['truth'] else 'wrong'})" for k in ks if W[k].get('retyped') == vis)[:300] + ")")
        if int(getattr(cfg, 'inf_rti_comp', 0)):                               # 16 Nov TYPED COMPOSITION: kept inferences that a later relational record
            T3 = _PM.setdefault('T3', collections.defaultdict(collections.Counter)); nl9 = 0   #   states teach the table premise wordings -> that wording
            for j, k in enumerate(ks):
                cs = cw9.get(frozenset(k))
                if fl[j] or not cs: continue
                X, Y, w = collections.Counter(cs).most_common(1)[0][0]; cf = w if X == k[0] else w + ' ^R'
                for a1, a2 in W[k].get('pw', ()):
                    if a1 and a2: T3[(a1, a2)][cf] += 1; nl9 += 1
            out.append(f"COMPOSITION TABLE learned {nl9} premise-pair entries from this batch's kept, confirmed inferences ({len(T3)} cells)")
        a_ = _cg_auc(sc[y], sc[~y]) if y.any() and (~y).any() else float('nan')
        out.append(f"[TYPED CONFIRMATION (evaluation)] AUC relational count {au_(rk(ra_)):.3f}, typed confirm - contradict {au_(cc_):.3f} (covers {np.mean([bool(HF9.get(W[k].get('th') or k[0])) for k in ks]):.2f}), "
                   f"both {au_(rk(ra_) + rk(cc_)):.3f}")
        out.append(f"[signal {sig_}] written with {lo} documents read, judged on {nd} read since: {len(ks)} inferences (taught {int(taught.sum())}), wrong {1 - y.mean():.2f} | AUC right vs wrong of "
                   f"the confirmation score {a_:.3f} | FLAGGED {int(fl.sum())}: wrong {1 - y[fl].mean() if fl.any() else float('nan'):.2f} (unflagged {1 - y[~fl].mean() if (~fl).any() else float('nan'):.2f}) "
                   f"| no confirming record {int((short == 0).sum())} | RETRACTED {int(ret.sum())} (wrong {1 - y[ret].mean() if ret.any() else float('nan'):.2f}), flagged taught kept as control "
                   f"{int((fl & taught & ctl_).sum())} | SIGNALS (evaluation only): documents since {au_(docs):.3f}, relational records {au_(ra_):.3f} (since the write {au_(rs_):.3f}), "
                   f"list-only pairs {int(((ra_ == 0) & (docs > 0)).sum())} right {yv_[(ra_ == 0) & (docs > 0)].mean() if ((ra_ == 0) & (docs > 0)).any() else float('nan'):.2f}")
    print(f"  [RETRACTION TRIGGER | visit {vis + 1}] " + " || ".join(out), flush=True)


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
    nup = 0
    for e in ents + held:                                                      # 16 Nov RETYPE ON CONFIRMATION: pool entries follow their ledger entry's new type
        if is_rti(e):
            w9 = W.get(rti_wkey(e)) or {}
            if w9.get('retyped') is not None and (e['A'], e['r'], e['ans']) != (w9['th'], w9['tw'], w9['tt']):
                e.update(A=w9['th'], r=w9['tw'], ans=w9['tt']); e['pfs'] = [f"{e['A']} {e['r']} "]; nup += 1
    if nup: print(f"  [RETYPE ON CONFIRMATION | visit {vis + 1}] {nup} taught inferences now taught in their confirming wording", flush=True)
    have = {repr(e['key']) for e in ents + held}; T_, H_, U_, X_ = [], [], [], []
    for (A_, C_), w in W.items():
        e = rti_ent((A_, C_), w, seen=2, src=(), rm=w['vis'], truth=w['truth'], novel=w.get('novel')); e['pfs'] = [f"{e['A']} {e['r']} "]
        if repr(e['key']) in have: continue
        hc = float(getattr(cfg, 'inf_rti_ctl', 0.0))                          # 13 Nov: its own held-back share (a third in cg27a_rti2), else the pool's fifth
        held_ = (zlib.crc32((repr(e['key']) + '|rti').encode()) % 1000 < hc * 1000) if hc > 0 else ctl(e)
        if w.get('unpred'): U_.append(e); continue                            # 14 Nov cg27a_rtc control: unpredicted facts are read, never taught
        if w.get('tent') or w.get('faded') is not None: continue               # 19 Nov kra12: tentative (not yet confirmed) and faded inferences are neither taught nor in the memory
        if w.get('ret') and not held_: e['flag'] = True; X_.append(e); continue   # 14 Nov cg27a_rret: RETRACTED - no longer taught (unlikelihood rows)
        if w.get('flag'): e['flag'] = True
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
        from platypus.inference.jepa import cfg_seq_
        sq_ = int(cfg_seq_(model))                                             # 20 Nov: the scorer cuts text at the window; prompts that do not fit are skipped
        def rk_(L, chain=False):
            L = [e for e in L if e['rm'] < vis and (not chain or (W.get(rti_wkey(e)) or {}).get('chain'))][:150]; hit = []
            for e in L:
                alts = [x for x in pool_ if x not in (e['ans'], e['A'])]
                if len(alts) < 9: continue
                cs = [e['ans']] + [alts[j] for j in rr_.choice(len(alts), 9, replace=False)]
                pre_ = (W[rti_wkey(e)]['chain'] + ' ') if chain else ''          # 20 Nov kra16: IN CONTEXT - the inference's premises before the question
                if len((pre_ + f"It is on record that {e['A']} {e['r']} ").encode('latin-1', 'replace')) + max(len(c) for c in cs) + 2 > sq_: continue   # 20 Nov: must fit the window (else every candidate ties and the first, the true one, wins)
                hit.append(int(_dw_cand_mc(model, int(S_slot(e['d'])), f"{pre_}It is on record that {e['A']} {e['r']} ", cs) == 0))
            return f"{np.mean(hit):.2f} ({len(hit)})" if hit else "nan (0)"
        tr_ = lambda L, t: [e for e in L if e['truth'] is t]
        dl_ = (model.__dict__.get('_dtw') or {}).get('rtil') or []
        print(f"  [REAL-TEXT INFERENCE RANK RECALL | visit {vis + 1}] the answer first among itself and 9 held-back answers (chance 0.10), from the trunk, test wording: TAUGHT right "
              f"{rk_(tr_(T_, True))} / wrong {rk_(tr_(T_, False))} | HELD BACK right {rk_(tr_(H_, True))} / wrong {rk_(tr_(H_, False))}"
              + (f" | UNPREDICTED right {rk_(tr_(U_, True))} / wrong {rk_(tr_(U_, False))}" if U_ else '')
              + (f" || dose rows last visit: loss {np.mean(dl_[:20]):.3f} (first 20) -> {np.mean(dl_[-20:]):.3f} (last 20), {len(dl_)} steps" if dl_ else ''), flush=True)
        from platypus.inference.distill import _kmc_pending                    # 20 Nov kra20 (readout): how many 'taught' inferences ACTUALLY train the trunk -
        trn_ = lambda e: not _kmc_pending(cfg, [(b'', b'', 0, repr(e['key']))])   #   rows of the others are skipped by the keep rule (inf_kmc / inf_kmc_useful);
        Ty_ = [e for e in T_ if trn_(e)]; Tn_ = [e for e in T_ if not trn_(e)]  #   kra17: about 2% of the confirmed ones under inf_kmc_useful 1
        print(f"  [REAL-TEXT INFERENCE TRAINED | visit {vis + 1}] of {len(T_)} taught (question pool), {len(Ty_)} pass the keep rule and train the trunk "
              f"(right {np.mean([bool(e['truth']) for e in Ty_]) if Ty_ else float('nan'):.2f}), {len(Tn_)} do not | rank among 10, TRAINED right {rk_(tr_(Ty_, True))} / wrong "
              f"{rk_(tr_(Ty_, False))} | NOT TRAINED right {rk_(tr_(Tn_, True))} / wrong {rk_(tr_(Tn_, False))} | HELD BACK right {rk_(tr_(H_, True))} / wrong {rk_(tr_(H_, False))}", flush=True)
        rp_ = np.random.RandomState(int(cfg.seed) + 93 + vis)                  # 21 Nov kra21 (readout): PRIOR CONTROL - the same question and candidates with the
        def sw_(L):                                                            #   subject SWAPPED for another inference's subject: a rise that survives the swap is a
            L = [e for e in L if e['rm'] < vis][:150]                          #   preference for the answer NAME (taught names made familiar), not knowledge of A
            if len(L) < 2: return "nan (0)"
            h0, h1 = [], []
            for i, e in enumerate(L):
                alts = [x for x in pool_ if x not in (e['ans'], e['A'])]
                if len(alts) < 9: continue
                cs = [e['ans']] + [alts[j] for j in rp_.choice(len(alts), 9, replace=False)]; oA = L[(i + 1) % len(L)]['A']
                q0, q1 = f"It is on record that {e['A']} {e['r']} ", f"It is on record that {oA} {e['r']} "
                if max(len(q0.encode('latin-1', 'replace')), len(q1.encode('latin-1', 'replace'))) + max(len(c) for c in cs) + 2 > sq_: continue
                h0.append(int(_dw_cand_mc(model, int(S_slot(e['d'])), q0, cs) == 0)); h1.append(int(_dw_cand_mc(model, int(S_slot(e['d'])), q1, cs) == 0))
            if not h0: return "nan (0)"
            d = np.array(h0) - np.array(h1)
            return f"own subject {np.mean(h0):.2f} | SWAPPED subject {np.mean(h1):.2f} | own - swapped {d.mean():+.2f} (SE {d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan'):.2f}) (n {len(d)})"
        try:
            print(f"  [REAL-TEXT INFERENCE PRIOR CONTROL | visit {vis + 1}] rank among 10, the question's subject vs another inference's subject (same candidates; knowledge of A "
                  f"if own > swapped): TRAINED right {sw_(tr_(Ty_, True))} || TRAINED wrong {sw_(tr_(Ty_, False))} || HELD BACK right {sw_(tr_(H_, True))} || HELD BACK wrong {sw_(tr_(H_, False))}", flush=True)
        except Exception as e9: print(f"  [REAL-TEXT INFERENCE PRIOR CONTROL | visit {vis + 1}] unavailable ({type(e9).__name__}: {e9})"[:200], flush=True)
        if int(getattr(cfg, 'inf_rti_src', 0)):                               # 21 Nov kra22 (readout): TRANSFER SOURCE - kra21 visit 5: held-back RIGHT own - swapped
            try:                                                               #   +0.15 (SE 0.05), so teaching lifts other right answers for their own subject. Where
                rs_ = np.random.RandomState(int(cfg.seed) + 97 + vis)          #   from? Split the held-back inferences by what they share with the TRAINED ones: a
                nm_ = set(); an_ = set(); br_ = set()                          #   name (either side of a taught pair), the answer, a bridge (premise middle name),
                for e in Ty_:                                                  #   or NOTHING. A lift with nothing shared is the strongest evidence of a learned step
                    k_ = rti_wkey(e); nm_.update(k_); an_.add(e['ans']); br_.update((W.get(k_) or {}).get('prem') or [])
                def pr_(L):
                    out = []
                    for i, e in enumerate([e for e in L if e['rm'] < vis][:150]):
                        alts = [x for x in pool_ if x not in (e['ans'], e['A'])]
                        if len(alts) < 9 or len(L) < 2: continue
                        cs = [e['ans']] + [alts[j] for j in rs_.choice(len(alts), 9, replace=False)]; oA = L[(i + 1) % len(L)]['A']
                        q0, q1 = f"It is on record that {e['A']} {e['r']} ", f"It is on record that {oA} {e['r']} "
                        if max(len(q0.encode('latin-1', 'replace')), len(q1.encode('latin-1', 'replace'))) + max(len(c) for c in cs) + 2 > sq_: continue
                        out.append((e, int(_dw_cand_mc(model, int(S_slot(e['d'])), q0, cs) == 0) - int(_dw_cand_mc(model, int(S_slot(e['d'])), q1, cs) == 0)))
                    return out
                def gs_(P, f):
                    d = np.array([x for e, x in P if f(e)], dtype=float)
                    return f"{d.mean():+.2f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.2f}, n {len(d)})" if len(d) > 1 else f"nan (n {len(d)})"
                sA = lambda e: e['A'] in nm_; sC = lambda e: e['ans'] in an_
                sB = lambda e: bool(set((W.get(rti_wkey(e)) or {}).get('prem') or []) & br_); sN = lambda e: not (e['A'] in nm_ or e['ans'] in nm_ or sB(e))
                parts9 = []
                for lab_, L_ in (('HELD BACK right', tr_(H_, True)), ('HELD BACK wrong', tr_(H_, False))):
                    P_ = pr_(L_)
                    parts9.append(f"{lab_} (n {len(P_)}): all {gs_(P_, lambda e: True)} | subject is a taught name {gs_(P_, sA)} / not {gs_(P_, lambda e: not sA(e))} | "
                                  f"answer is a taught answer {gs_(P_, sC)} / not {gs_(P_, lambda e: not sC(e))} | shares a bridge {gs_(P_, sB)} / not {gs_(P_, lambda e: not sB(e))} | "
                                  f"NOTHING SHARED {gs_(P_, sN)}")
                print(f"  [REAL-TEXT INFERENCE TRANSFER SOURCE | visit {vis + 1}] own - swapped subject (rank among 10, as PRIOR CONTROL), held-back inferences split by what "
                      f"they share with the {len(Ty_)} TRAINED ones ({len(nm_)} names, {len(an_)} answers, {len(br_)} bridges): " + " || ".join(parts9), flush=True)
            except Exception as e9: print(f"  [REAL-TEXT INFERENCE TRANSFER SOURCE | visit {vis + 1}] unavailable ({type(e9).__name__}: {e9})"[:200], flush=True)
            try:                                                               # 21 Nov kra22 (readout): TAUGHT RECALL BREAKDOWN - kra21 visit 6: taught inferences written
                rb_ = np.random.RandomState(int(cfg.seed) + 99 + vis)          #   at visit 2 fell 0.97 -> 0.66 open in one visit with the SAME dose rows (6,426 both
                AR_ = collections.defaultdict(set)                             #   visits; dose loss 0.010 -> 0.003), so not dilution. Split by: the write visit, the keep
                for e in T_: AR_[(e['A'], e['r'])].add(e['ans'])               #   rule's verdict now, a CONFLICT (another taught answer for the same subject and wording),
                def bk_(L):                                                    #   and the prompt: the bare training form 'A r ' vs the test wording
                    h0, h1 = [], []
                    for e in L[:120]:
                        alts = [x for x in pool_ if x not in (e['ans'], e['A'])]
                        if len(alts) < 9: continue
                        cs = [e['ans']] + [alts[j] for j in rb_.choice(len(alts), 9, replace=False)]; q1 = f"It is on record that {e['A']} {e['r']} "
                        if len(q1.encode('latin-1', 'replace')) + max(len(c) for c in cs) + 2 > sq_: continue
                        h0.append(int(_dw_cand_mc(model, int(S_slot(e['d'])), f"{e['A']} {e['r']} ", cs) == 0)); h1.append(int(_dw_cand_mc(model, int(S_slot(e['d'])), q1, cs) == 0))
                    return f"bare {np.mean(h0):.2f} / test {np.mean(h1):.2f} ({len(h0)})" if h0 else "- (0)"
                old9 = [e for e in T_ if e['truth'] is True and e['rm'] < vis]; parts8 = []
                for wv in sorted({int((W.get(rti_wkey(e)) or {}).get('vis', -1)) for e in old9}):
                    parts8.append(f"written at visit {wv + 1}: {bk_([e for e in old9 if int((W.get(rti_wkey(e)) or {}).get('vis', -1)) == wv])}")
                parts8.append(f"keep rule TRAINS {bk_([e for e in old9 if trn_(e)])} | does NOT {bk_([e for e in old9 if not trn_(e)])}")
                parts8.append(f"CONFLICT (another taught answer, same subject and wording) {bk_([e for e in old9 if len(AR_[(e['A'], e['r'])]) > 1])} | none {bk_([e for e in old9 if len(AR_[(e['A'], e['r'])]) == 1])}")
                print(f"  [REAL-TEXT INFERENCE TAUGHT RECALL BREAKDOWN | visit {vis + 1}] taught RIGHT inferences, rank among 10, bare training form 'A r ' / test wording: " + " || ".join(parts8), flush=True)
            except Exception as e9: print(f"  [REAL-TEXT INFERENCE TAUGHT RECALL BREAKDOWN | visit {vis + 1}] unavailable ({type(e9).__name__}: {e9})"[:200], flush=True)
        if int(getattr(cfg, 'inf_rti_judge', 0)):                             # 21 Nov kra23 (readout): SELF-SUPPORT AS A JUDGE - kra21 visit 8: trained WRONG inferences
            try:                                                               #   had half the subject-specific support of trained right ones (+0.21 vs +0.45). Per
                from platypus.inference.distill import _dw_cand_scores         #   inference: the answer's mean log-prob after its own subject minus after another
                from sklearn.metrics import roc_auc_score                      #   inference's subject; AUC right vs wrong, by group. JUST CONFIRMED (not yet trained)
                def sm_(L):                                                    #   is the one usable as a teach-gate; TRAINED as a retraction judge
                    out = []
                    for i, e in enumerate(L[:150]):
                        oA = L[(i + 1) % len(L)]['A']
                        if oA == e['A'] or len(L) < 2: continue
                        a0 = _dw_cand_scores(model, int(S_slot(e['d'])), f"It is on record that {e['A']} {e['r']} ", [e['ans']])[0]
                        a1 = _dw_cand_scores(model, int(S_slot(e['d'])), f"It is on record that {oA} {e['r']} ", [e['ans']])[0]
                        if np.isfinite(a0) and np.isfinite(a1): out.append((bool(e['truth']), a0 - a1, a0))
                    return out
                def au_(P_):
                    y_ = [t for t, m_, a_ in P_]
                    if len(set(y_)) < 2: return f"n {len(P_)}"
                    return (f"margin AUC {roc_auc_score(y_, [m_ for t, m_, a_ in P_]):.3f}, own log-prob AUC {roc_auc_score(y_, [a_ for t, m_, a_ in P_]):.3f} "
                            f"(n {len(P_)}, {sum(not t for t in y_)} wrong; margin right {np.mean([m_ for t, m_, a_ in P_ if t]):+.2f} / wrong {np.mean([m_ for t, m_, a_ in P_ if not t]):+.2f})")
                jc_ = [e for e in T_ + H_ if (W.get(rti_wkey(e)) or {}).get('conf') == vis and e['truth'] is not None]
                print(f"  [REAL-TEXT INFERENCE SELF-SUPPORT JUDGE | visit {vis + 1}] own - swapped subject log-prob of the answer, right vs wrong: JUST CONFIRMED (never trained; "
                      f"a teach-gate) {au_(sm_(jc_))} || TRAINED {au_(sm_([e for e in Ty_ if e['truth'] is not None]))} || HELD BACK {au_(sm_([e for e in H_ if e['truth'] is not None]))}", flush=True)
            except Exception as e9: print(f"  [REAL-TEXT INFERENCE SELF-SUPPORT JUDGE | visit {vis + 1}] unavailable ({type(e9).__name__}: {e9})"[:200], flush=True)
        if int(getattr(cfg, 'inf_rti_audit', 0)):                             # 21 Nov kra23 (readout): AUDIT SAMPLE - 40 inferences confirmed at this sleep, printed
            try:                                                               #   with their distant label, for a hand audit of the true false-memory rate
                ra_ = np.random.RandomState(int(cfg.seed) + 101 + vis); ks_ = [k for k, w in W.items() if w.get('conf') == vis and w.get('state') == 'confirmed']
                sm9 = [ks_[j] for j in ra_.permutation(len(ks_))[:40]]
                print(f"  [AUDIT SAMPLE | visit {vis + 1}] {len(sm9)} of {len(ks_)} inferences confirmed at this sleep (label from the distant set): " +
                      " | ".join(f"{W[k].get('th', k[0])} {W[k].get('tw', '~')} {W[k].get('tt', k[1])} [{'R' if W[k]['truth'] else 'W'}]" for k in sm9), flush=True)
            except Exception as e9: print(f"  [AUDIT SAMPLE | visit {vis + 1}] unavailable ({type(e9).__name__}: {e9})"[:200], flush=True)
        if int(getattr(cfg, 'inf_rti_chain', 0)):                             # 20 Nov kra16: the same with the premises IN CONTEXT; HELD BACK = never taught, so above
            print(f"  [REAL-TEXT INFERENCE IN CONTEXT | visit {vis + 1}] premises before the question (inferences with both premises on record), rank among 10: TAUGHT right "
                  f"{rk_(tr_(T_, True), True)} / wrong {rk_(tr_(T_, False), True)} | HELD BACK (never taught: a learned composition step if above the line above) right "
                  f"{rk_(tr_(H_, True), True)} / wrong {rk_(tr_(H_, False), True)}", flush=True)
            rm_ = np.random.RandomState(int(cfg.seed) + 91 + vis)                # 20 Nov kra18 (readout only): MATCHED - the same inferences and the same 9
            def mt_(L):                                                          #   distractors three ways: closed book, their OWN premises before the question,
                L = [e for e in L if e['rm'] < vis and (W.get(rti_wkey(e)) or {}).get('chain_p')][:150]   # and ANOTHER inference's premises (any context vs these);
                if len(L) < 2: return "nan (0)"                                  #   20 Nov kra20: + BROKEN CHAIN - another inference's A-B premise then this one's
                h = [[], [], [], []]                                             #   own B-C premise: C is in the context but nothing links A to it (copying vs reasoning)
                for i, e in enumerate(L):
                    alts = [x for x in pool_ if x not in (e['ans'], e['A'])]
                    if len(alts) < 9: continue
                    cs = [e['ans']] + [alts[j] for j in rm_.choice(len(alts), 9, replace=False)]
                    q = f"It is on record that {e['A']} {e['r']} "
                    oth_ = W[rti_wkey(L[(i + 1) % len(L)])]; own_ = W[rti_wkey(e)]
                    pre4 = ('', own_['chain'] + ' ', oth_['chain'] + ' ', oth_['chain_p'][0] + ' ' + own_['chain_p'][1] + ' ')
                    if max(len((p + q).encode('latin-1', 'replace')) for p in pre4) + max(len(c) for c in cs) + 2 > sq_: continue
                    for k, p in enumerate(pre4):
                        h[k].append(int(_dw_cand_mc(model, int(S_slot(e['d'])), p + q, cs) == 0))
                if not h[0]: return "nan (0)"
                d = np.array(h[1]) - np.array(h[0])
                d2 = np.array(h[1]) - np.array(h[3])
                return (f"closed {np.mean(h[0]):.2f} | own premises {np.mean(h[1]):.2f} | other premises {np.mean(h[2]):.2f} | BROKEN CHAIN (answer in context, A not linked) {np.mean(h[3]):.2f} | "
                        f"own - broken {d2.mean():+.2f} (SE {d2.std(ddof=1) / np.sqrt(len(d2)) if len(d2) > 1 else float('nan'):.2f}) | own - closed {d.mean():+.2f} "
                        f"(SE {d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan'):.2f}) (n {len(d)})")
            print(f"  [REAL-TEXT INFERENCE IN CONTEXT, MATCHED | visit {vis + 1}] the same inferences and distractors, rank among 10: TAUGHT right {mt_(tr_(T_, True))} || "
                  f"HELD BACK right {mt_(tr_(H_, True))} || HELD BACK wrong {mt_(tr_(H_, False))}", flush=True)
    model.__dict__['_rret_x'] = X_                                           # retracted: the unlikelihood rows are built from these at the pool rebuild
    if int(getattr(cfg, 'inf_rret', 0)):                                      # 14 Nov RETRACTION RECALL: retracted vs flagged-but-kept (control) vs unflagged taught
        Kf = [e for e in T_ if e.get('flag')]; Nf = [e for e in T_ if not e.get('flag') and (W.get((e['A'], e['ans'])) or {}).get('judged') is not None]
        rr2 = np.random.RandomState(int(cfg.seed) + 78 + vis); pool2 = sorted({e['ans'] for e in H_})
        def rk2(L):
            if not int(getattr(cfg, 'inf_rti_rank_recall', 0)) or len(pool2) < 10: return ''
            from platypus.inference.distill import _dw_cand_mc
            hit = []
            for e in L[:150]:
                alts = [x for x in pool2 if x not in (e['ans'], e['A'])]
                cs = [e['ans']] + [alts[j] for j in rr2.choice(len(alts), 9, replace=False)]
                hit.append(int(_dw_cand_mc(model, int(S_slot(e['d'])), f"It is on record that {e['A']} {e['r']} ", cs) == 0))
            return f", rank {np.mean(hit):.2f}" if hit else ''
        def both(L, t):
            L = [e for e in L if e['truth'] is t]; return f"{g(_dt_recall(model, S_slot, L))}{rk2(L)}"
        ul_ = (model.__dict__.get('_dtw') or {}).get('rretl') or []
        print(f"  [RETRACTION RECALL | visit {vis + 1}] from the trunk, test wording, open-ended (and rank among 10 held-back answers): RETRACTED right {both(X_, True)} / wrong "
              f"{both(X_, False)} | FLAGGED, KEPT (control) right {both(Kf, True)} / wrong {both(Kf, False)} | JUDGED, NOT FLAGGED right {both(Nf, True)} / wrong {both(Nf, False)}"
              + (f" || unlikelihood rows last visit: loss {np.mean(ul_[:20]):.4f} (first 20) -> {np.mean(ul_[-20:]):.4f} (last 20), {len(ul_)} steps" if ul_ else ''), flush=True)
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
