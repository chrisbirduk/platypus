"""RELATION TYPES FROM THE TRUNK'S REPRESENTATION, AND TYPED RULES OVER THEM (15 Nov 2026; inf_rtype; READOUT, nothing written).

The untyped real-text inferences ('A is connected to C', 0.74 right) cannot be judged label-free: their errors are associated pairs without
a relation (tools/env_judge.py, tools/stab_trigger.py). Typed claims are needed. Offline:
  - tools/rel_direction.py (stand-in trunk): the pooled representation of 'A r C' is ADDITIVE (subject + relation + object (+ year), R^2
    0.98-1.00), the relation's effect subspace largely apart from the names' (overlap 0.11-0.18 vs 0.32-0.44 between subject and object)
  - tools/rel_types.py (stand-in trunk, 30,000 real records): k-means types of the pooled states beat SURFACE wordings against Wikidata
    (purity 0.40 vs 0.34, NMI 0.32-0.34 vs 0.29); a synthetic name subspace projected out helps a little; rules over the types too sparse
    on 30,000 records (typed right 0.20-0.44)
Here at GPU scale, the trunk's own states over the premises store's records read so far (inf_rtype_n of them, both names in 2+ records):
  TYPES    k-means (inf_rtype_k) over the pooled middle-block states of 'A <words> C.' - raw, and with the NAME subspace of a synthetic
           factorial (16 subjects x 12 relation phrases x 16 objects, on the same trunk) projected out; also the factorial's own numbers:
           variance shares, additive R^2, subspace overlaps on THIS trunk
  RULES    AMIE closed paths r1(x,y) & r2(y,z) => r3(x,z) over the types (either direction), support >= 5, PCA confidence >= 0.3; proposals =
           body pairs never stated in the records read
Scored only (evaluation): type purity / NMI against the Wikidata relations of the records with one (surface wordings as the baseline), the
typed proposals (the head type's commonest relation holds for x-z, that direction) and untyped (some relation). One line per readout:
[RELATION TYPES FROM THE TRUNK]."""
import time, collections
import numpy as np
import torch

_RT = {}


def _rt_pooled(model, texts, sl, blk):
    from platypus.inference.closed_form import _ws_batch, _ws_run
    out = []; cap = {}; h9 = blk.register_forward_hook(lambda m_, i_, o_: cap.__setitem__('h', (o_[0] if isinstance(o_, tuple) else o_).detach()))
    try:
        for i0 in range(0, len(texts), 256):
            ch = [t.encode('latin-1', 'replace') for t in texts[i0:i0 + 256]]; X, M = _ws_batch(model, [b + b' ' for b in ch])
            with torch.no_grad(): _ws_run(model, X, sl)
            H = cap['h'].float(); m = M[:, :H.shape[1]].float()[:, :, None]
            out.append(((H * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    finally: h9.remove()
    return np.concatenate(out)


def reltype_readout(model, cfg, S_slot, vis):
    from platypus.inference.premises import _PM, _load_distant
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import normalized_mutual_info_score
    t0 = time.time()
    if not _PM.get('R'): _load_distant(cfg)
    R = _PM.get('R') or []
    if not R: print(f"  [RELATION TYPES FROM THE TRUNK | visit {vis + 1}] no distant records", flush=True); return
    if 'PRD' not in _PM:                                                       # evaluation only: ordered Wikidata relations (kept by the loader)
        _PM['path'] = None; _load_distant(cfg); R = _PM.get('R') or []
    PR = _PM.get('PRD') or {}; LAB = _PM.get('LAB', set())
    nv = max(1, int(getattr(cfg, 'inf_premises_visits', 10))); ndv = int(round(_PM.get('ND', 0) * min(1.0, (vis + 1) / nv)))
    early = [r for r in R if r[0] < ndv and 1 <= r[5] <= 8 and r[1] != r[2] and len(r[1]) < 40 and len(r[2]) < 40]
    nc = collections.Counter(n for r in early for n in (r[1], r[2])); PAIR = {frozenset((r[1], r[2])) for r in early}
    pool = [r for r in early if nc[r[1]] >= 2 and nc[r[2]] >= 2]; g = np.random.RandomState(int(cfg.seed) + 6060 + vis)
    S = [pool[i] for i in g.choice(len(pool), min(int(getattr(cfg, 'inf_rtype_n', 60000)), len(pool)), replace=False)] if pool else []
    if len(S) < 1000: print(f"  [RELATION TYPES FROM THE TRUNK | visit {vis + 1}] too few records ({len(S)})", flush=True); return
    try: sl = int(S_slot(1))
    except Exception: sl = 0
    blk = model.blocks[len(model.blocks) // 2]
    X = _rt_pooled(model, [f"{r[1]} {r[3]} {r[2]}." for r in S], sl, blk); X = X - X.mean(0)
    # the factorial on this trunk: additivity, variance shares, subspace overlaps; its name subspace
    REL9 = ['was born in', 'died in', 'is a citizen of', 'worked for', 'played for', 'is a member of', 'is located in', 'was written by', 'was directed by',
            'is part of', 'comes from', 'is based in']
    syl = ['ka', 'lo', 'mi', 'ner', 'tas', 'vol', 'ru', 'den', 'sa', 'por', 'el', 'ith', 'qua', 'bren', 'dor', 'fen']; g2 = np.random.RandomState(1)
    nmf = lambda: (''.join(g2.choice(syl, 2)) + ' ' + ''.join(g2.choice(syl, 3))).title(); SB = [nmf() for _ in range(16)]; OB = [nmf() for _ in range(16)]
    rows = [(a, r, o) for a in range(16) for r in range(12) for o in range(16)]; ix = np.array(rows)
    Z = _rt_pooled(model, [f"{SB[a]} {REL9[r]} {OB[o]}." for a, r, o in rows], sl, blk); Z -= Z.mean(0); tot = (Z ** 2).sum()
    eff = [np.array([Z[ix[:, c] == v].mean(0) for v in range(ix[:, c].max() + 1)]) for c in range(3)]; fit = sum(e[ix[:, c]] for c, e in enumerate(eff))
    share = [float((e[ix[:, c]] ** 2).sum() / tot) for c, e in enumerate(eff)]; r2 = float(1 - ((Z - fit) ** 2).sum() / tot)
    bas = [np.linalg.svd(e - e.mean(0), full_matrices=False)[2][:8].T for e in eff]; ov = lambda a, b: float(np.mean(np.linalg.svd(a.T @ b, compute_uv=False) ** 2))
    Bn = np.linalg.qr(np.concatenate([bas[0][:, :4], bas[2][:, :4]], 1))[0]; Xo = X - (X @ Bn) @ Bn.T
    lab = np.array([next(iter(PR[(r[1], r[2])])) if len(PR.get((r[1], r[2]), ())) == 1 else '' for r in S]); has = lab != ''
    def purity(c):
        ct = collections.defaultdict(collections.Counter)
        for a, b in zip(c[has], lab[has]): ct[a][b] += 1
        return sum(v.most_common(1)[0][1] for v in ct.values()) / max(1, has.sum()), normalized_mutual_info_score(lab[has], c[has])
    KT = int(getattr(cfg, 'inf_rtype_k', 64)); TY = {}
    for nm, Zr in (('raw', X), ('names out', Xo)): TY[nm] = MiniBatchKMeans(KT, random_state=0, n_init=3, batch_size=4096).fit_predict(Zr)
    # 15 Nov: names MASKED with their entity types as placeholders, and a MATCHING-THE-BLANKS key (offline, tools/rel_mtb.py: purity 0.34 ->
    # 0.49 type-masked, 0.57 with the key; NMI 0.29 -> 0.57): two records of the same name pair in different documents are positives
    ETY = _PM.get('TY') or {}; TN = {'PER': 'a person', 'LOC': 'a place', 'ORG': 'an organisation', 'TIME': 'a date', 'NUM': 'a number', 'MISC': 'a thing'}
    tm = lambda r: f"{TN.get(ETY.get(r[1]), 'a thing')} {r[3]} {TN.get(ETY.get(r[2]), 'a thing')}."
    Xt = _rt_pooled(model, [tm(r) for r in S], sl, blk); TY['type-masked'] = MiniBatchKMeans(KT, random_state=0, n_init=3, batch_size=4096).fit_predict(Xt - Xt.mean(0))
    byp = collections.defaultdict(list)
    for r in early: byp[frozenset((r[1], r[2]))].append(r)
    mp = []
    for L_ in byp.values():
        o = [r for r in L_[1:] if r[0] != L_[0][0]]
        if o: mp.append((L_[0], o[0]))
    mp = [mp[i] for i in g.choice(len(mp), min(20000, len(mp)), replace=False)] if mp else []
    if len(mp) >= 512:
        import torch.nn.functional as F_
        dev = model.embed.weight.device; A_ = _rt_pooled(model, [tm(a) for a, b in mp], sl, blk); B_ = _rt_pooled(model, [tm(b) for a, b in mp], sl, blk)
        mu = np.concatenate([A_, B_]).mean(0); sd = np.concatenate([A_, B_]).std(0) + 1e-6
        A_ = torch.tensor((A_ - mu) / sd, dtype=torch.float32, device=dev); B_ = torch.tensor((B_ - mu) / sd, dtype=torch.float32, device=dev)
        head = torch.nn.Sequential(torch.nn.Linear(A_.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 64)).to(dev); opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        for st in range(3000):
            ix = torch.as_tensor(g.choice(len(A_), 256, replace=False), device=dev); z1 = F_.normalize(head(A_[ix]), dim=-1); z2 = F_.normalize(head(B_[ix]), dim=-1); ar = torch.arange(256, device=dev)
            lo = 0.5 * (F_.cross_entropy(z1 @ z2.t() / 0.1, ar) + F_.cross_entropy(z2 @ z1.t() / 0.1, ar)); opt.zero_grad(); lo.backward(); opt.step()
        with torch.no_grad(): Zk = F_.normalize(head(torch.tensor((Xt - mu) / sd, dtype=torch.float32, device=dev)), dim=-1).cpu().numpy()
        TY[f'MTB key, type-masked ({len(mp)} pairs, loss {float(lo.detach()):.2f})'] = MiniBatchKMeans(KT, random_state=0, n_init=3, batch_size=4096).fit_predict(Zk)
    wc = collections.Counter(r[3] for r in S); tw = {w: i for i, (w, _) in enumerate(wc.most_common(KT - 1))}; TY['surface wordings'] = np.array([tw.get(r[3], KT - 1) for r in S])
    def mine(c):
        OUT = collections.defaultdict(lambda: collections.defaultdict(set)); HEAD = {}
        for r, t in zip(S, c):
            k = int(t); OUT[r[1]][(k, 0)].add(r[2]); OUT[r[2]][(k, 1)].add(r[1]); HEAD.setdefault((r[1], r[2]), set()).add((k, 0)); HEAD.setdefault((r[2], r[1]), set()).add((k, 1))
        HASR = collections.defaultdict(set)
        for (x, z), ks in HEAD.items(): HASR[x] |= ks
        BODY = collections.defaultdict(set)
        for y, d in OUT.items():
            nin = [(k, x) for k, xs in d.items() for x in xs]
            if len(nin) > 60: continue
            for k1, x in nin:
                for k2, z in nin:
                    if z != x: BODY[((k1[0], 1 - k1[1]), k2)].add((x, z))
        rules = []
        for (r1, r2), bp in BODY.items():
            if len(bp) < 5: continue
            cnt = collections.Counter(k for x, z in bp for k in HEAD.get((x, z), ()))
            for k, s_ in cnt.items():
                if s_ >= 5:
                    pca = s_ / max(1, sum(1 for x, z in bp if k in HASR[x]))
                    if pca >= 0.3: rules.append((r1, r2, k, pca))
        TREL = collections.defaultdict(collections.Counter)
        for r, t in zip(S, c):
            for rr in PR.get((r[1], r[2]), ()): TREL[(int(t), 0)][rr] += 1
            for rr in PR.get((r[2], r[1]), ()): TREL[(int(t), 1)][rr] += 1
        props = {}
        for r1, r2, k, pca in rules:
            for x, z in BODY[(r1, r2)]:
                if frozenset((x, z)) not in PAIR and ((x, z) not in props or props[(x, z)][1] < pca): props[(x, z)] = (k, pca)
        typ = [int(TREL[k].most_common(1)[0][0] in PR.get((x, z), ())) for (x, z), (k, p) in props.items() if TREL[k]]
        unt = np.mean([frozenset(p) in LAB for p in props]) if props else float('nan')
        return f"{len(rules)} rules, {len(props)} typed proposals, TYPED right {np.mean(typ) if typ else float('nan'):.3f}, untyped {unt:.3f}"
    parts = []
    for nm, c in TY.items():
        pu, nmi = purity(c); parts.append(f"{nm}: purity {pu:.3f}, NMI {nmi:.3f}; {mine(c)}")
    print(f"  [RELATION TYPES FROM THE TRUNK | visit {vis + 1}] FACTORIAL on this trunk (16 x 12 x 16, middle block, pooled): variance subject / RELATION / object "
          f"{share[0]:.2f} / {share[1]:.2f} / {share[2]:.2f}, additive R^2 {r2:.2f}, overlap RELATION-subject {ov(bas[1], bas[0]):.2f}, RELATION-object {ov(bas[1], bas[2]):.2f}, "
          f"subject-object {ov(bas[0], bas[2]):.2f} (random ~{8 / X.shape[1]:.2f}) || {len(S)} real records read (documents < {ndv}), {int(has.sum())} with one Wikidata "
          f"relation (evaluation only), {KT} types || " + " || ".join(parts) + f" (the untyped written inferences: 0.74 right) | {time.time() - t0:.0f}s", flush=True)
