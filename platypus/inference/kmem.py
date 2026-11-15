"""KEYED MEMORY (readout, 15 Nov 2026; inf_kmem): a sparse key-to-key memory addressed by the SUBJECT NAME and a JEPA WORDING KEY.

Offline (tools/jepa_mem.py, frozen stand-in trunk, Re-DocRED facts): taught facts recalled in an unseen wording at rank 0.93-1.00; two
relations of one subject kept apart only with the wording half; two-hop answers for chains never taught 0.73-0.97 (first hop decoded to
a known name); retraction EXACT by erasing the slots a fact's own questions reach (recall to 0.00, bystanders within 0.04, also with the
memory nearly full); refresh (a quarter of each write batch re-writing earlier facts) holds old facts. This readout asks the same at GPU
scale, on the system's own question pool (the taught facts and the written real-text inferences), with the trunk TRAINING underneath.

  NAME KEY     the trunk's last block over the name in 'About NAME.' (distill._name_codes: mean, centred, unit length), recomputed at
               every readout (the trunk moves)
  WORDING KEY  the last block over the question prefix with the subject replaced by 'someone', mean and last byte; a small head trained
               JEPA/InfoNCE-style so the system's TRAINING frames of one relation meet. The test wording ('It is on record that someone
               r ') and the paraphrases (_DT_PARA) are never seen by the head
  MEMORY       product keys n x n (inf_kmem_n), top 8 per read; half 1 = a fixed projection of the subject's name key, half 2 = the
               wording key; every slot's VALUE lives in name-key space. Writing = gradient on the values of the slots a fact's queries
               reach (nothing else changes); the loss puts the read nearest the object's name key among all known names
  READ         test-wording query -> value -> nearest name: OPEN (exact among every known name) and RANK (the answer first among itself and
               9 other answers; chance 0.10)

Two memories: PERSISTENT (kept across readouts: new facts written with inf_kmem_rows rows each, plus a share inf_kmem_replay of each write
batch re-writing earlier facts; read with today's keys, so key drift as the trunk trains shows as recall by the readout a fact was
written at) and REBUILT (written afresh from the whole pool at this readout: what the memory can hold at this scale). Groups: the pool's
facts (taught vs held back - held-back facts are never written: the control), the written real-text inferences RIGHT / WRONG, the
PARAPHRASED relation phrase, TWO-HOP chains of taught facts (decoded through the nearest name), and ERASE: on a copy of the rebuilt
memory, the slots reached by the WRONG inferences' (ORACLE: the labels pick them - the mechanism, not a trigger) and, if any, the
retraction trigger's FLAGGED inferences' training-frame questions are set to zero: their recall before -> after, and the bystanders'.
Training is not changed (own parameters, forked RNG)."""
import math
import numpy as np
import torch
import torch.nn.functional as F

_KM_DK = 32
_KM_KK = 8


def _km_prefix_states(model, texts):
    """last block over each text: (mean over its bytes, the last byte's state)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    cap = {}; h9 = model.blocks[-1].register_forward_hook(lambda m_, i_, o_: cap.__setitem__('h', o_.detach())); M_, L_ = [], []
    try:
        for i0 in range(0, len(texts), 64):
            ch = [t.encode('latin-1', 'replace') for t in texts[i0:i0 + 64]]; X, _ = _ws_batch(model, ch)
            with torch.no_grad(): _ws_run(model, X, 0)
            h = cap['h'].float()
            for j, b in enumerate(ch):
                n = max(1, min(len(b) - 1, h.shape[1])); M_.append(h[j, :n].mean(0)); L_.append(h[j, n - 1].clone())
    finally: h9.remove()
    return torch.stack(M_), torch.stack(L_)


def _first(sc):
    """the answer (index 0) strictly above every other candidate: an erased read (all zeros) ties and does NOT count"""
    return int(bool(sc[0] > sc[1:].max()))


class _KMem:
    def __init__(s, n, d, dev, seed, lr):
        g = torch.Generator(device='cpu').manual_seed(seed); s.n = n
        s.k1 = F.normalize(torch.randn(n, _KM_DK, generator=g), dim=-1).to(dev); s.k2 = F.normalize(torch.randn(n, _KM_DK, generator=g), dim=-1).to(dev)
        s.R1 = (torch.randn(d, _KM_DK, generator=g) / math.sqrt(d)).to(dev)
        s.v = torch.zeros(n * n, d, device=dev, requires_grad=True); s.opt = torch.optim.Adam([s.v], lr=lr); s.touched = set()
    def read(s, nk, wk):
        q1 = F.normalize(nk @ s.R1, dim=-1); a1, i1 = (q1 @ s.k1.t()).topk(_KM_KK, -1); a2, i2 = (wk @ s.k2.t()).topk(_KM_KK, -1)
        sc = (a1[:, :, None] + a2[:, None, :]).flatten(1); ix = (i1[:, :, None] * s.n + i2[:, None, :]).flatten(1)
        top, j = sc.topk(_KM_KK, -1); idx = ix.gather(-1, j); w = F.softmax(top / 0.05, -1)
        return (w[..., None] * s.v[idx]).sum(1), idx


def kmem_readout(model, cfg, S_, S_slot, ents, held, vis):
    from platypus.inference.distill import _name_codes, _dt_forms, _DT_PARA
    fs = [e for e in ents if e['key'][0] == 'fact']; hf = [e for e in held if e['key'][0] == 'fact']
    if len(fs) < 16:
        print(f"  [KEYED MEMORY | visit {vis + 1}] too few taught facts ({len(fs)})", flush=True); return
    dev = model.embed.weight.device; seed = int(cfg.seed) + 9191; FR = _dt_forms(cfg); TI = len(FR)
    n_ = int(getattr(cfg, 'inf_kmem_n', 128)); rows_ = int(getattr(cfg, 'inf_kmem_rows', 20)); rep_ = float(getattr(cfg, 'inf_kmem_replay', 0.25))
    lr_ = float(getattr(cfg, 'inf_kmem_lr', 0.05))
    names = sorted({e['A'] for e in fs + hf} | {e['ans'] for e in fs + hf}); ix = {n: i for i, n in enumerate(names)}
    NK = _name_codes(model, names); d = NK.shape[1]
    rels = sorted({e['r'] for e in fs + hf})
    tx = [f.format(A='someone', r=r) for r in rels for f in FR] + [f"It is on record that someone {r} " for r in rels] \
        + [f"someone {_DT_PARA.get(r, r)} " for r in rels]
    pm, pl = _km_prefix_states(model, tx); PC = torch.cat([pm, pl], -1); PC = (PC - PC.mean(0)) / (PC.std(0) + 1e-6)
    NR = len(rels); fi_ = lambda r, k: rels.index(r) * TI + k; ti_ = lambda r: NR * TI + rels.index(r); pi_ = lambda r: NR * TI + NR + rels.index(r)
    torch.manual_seed(seed + vis); g = np.random.RandomState(seed + vis)
    head = torch.nn.Sequential(torch.nn.Linear(PC.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, _KM_DK)).to(dev)
    oh = torch.optim.Adam(head.parameters(), lr=1e-3)
    if NR >= 2 and TI >= 2:
        for st in range(1500):
            f1 = g.randint(0, TI, NR); f2 = (f1 + g.randint(1, TI, NR)) % TI
            z1 = F.normalize(head(PC[[r * TI + int(a) for r, a in enumerate(f1)]]), dim=-1); z2 = F.normalize(head(PC[[r * TI + int(b) for r, b in enumerate(f2)]]), dim=-1)
            lo = F.cross_entropy(z1 @ z2.t() / 0.1, torch.arange(NR, device=dev)); oh.zero_grad(); lo.backward(); oh.step()
    with torch.no_grad(): WK = F.normalize(head(PC), dim=-1)
    def wacc(rows_ix):
        tr = torch.stack([WK[[fi_(r, k) for k in range(TI)]].mean(0) for r in rels]); return float(((WK[rows_ix] @ tr.t()).argmax(-1) == torch.arange(NR, device=dev)).float().mean())
    w_te, w_pa = wacc([ti_(r) for r in rels]), wacc([pi_(r) for r in rels])
    ans_pool = sorted({e['ans'] for e in fs + hf})
    def write(M, facts, steps, old=()):
        nb = 256; nrp = int(nb * rep_) if old else 0
        for st in range(steps):
            fb = [facts[g.randint(len(facts))] for _ in range(nb - nrp)] + [old[g.randint(len(old))] for _ in range(nrp)]; ks = g.randint(0, TI, nb)
            val, idx = M.read(NK[[ix[e['A']] for e in fb]], WK[[fi_(e['r'], int(k)) for e, k in zip(fb, ks)]])
            lo = F.cross_entropy(F.normalize(val, dim=-1) @ NK.t() / 0.05, torch.tensor([ix[e['ans']] for e in fb], device=dev))
            M.opt.zero_grad(); lo.backward(); M.opt.step(); M.touched.update(idx.flatten().tolist())
    @torch.no_grad()
    def recall(M, facts, wk_of, nmax=400):
        facts = facts[:nmax]
        if not facts: return float('nan'), float('nan'), 0
        val, _ = M.read(NK[[ix[e['A']] for e in facts]], WK[[wk_of(e) for e in facts]]); vn = F.normalize(val, dim=-1)
        op = (vn @ NK.t()).argmax(-1).tolist(); oe = np.mean([names[o] == e['ans'] for o, e in zip(op, facts)]); rr = np.random.RandomState(seed + 7 + vis); hit = []
        for k, e in enumerate(facts):
            alts = [x for x in ans_pool if x not in (e['ans'], e['A'])]
            if len(alts) < 9: continue
            cs = [e['ans']] + [alts[j] for j in rr.choice(len(alts), 9, replace=False)]; hit.append(_first(vn[k] @ NK[[ix[c] for c in cs]].t()))
        return oe, (np.mean(hit) if hit else float('nan')), len(facts)
    te = lambda e: ti_(e['r']); pa = lambda e: pi_(e['r'])
    fmt = lambda x: f"{x[0]:.2f} / {x[1]:.2f} ({x[2]})" if x[2] else "- (0)"
    RTI_R = 'is connected to'
    kind = lambda L, t=None: [e for e in L if (e['r'] != RTI_R if t == 'fact' else (e['r'] == RTI_R and e.get('truth') is t))]
    # REBUILT: the whole pool written afresh
    M = _KMem(n_, d, dev, seed, lr_); steps = max(300, int(math.ceil(rows_ * len(fs) / 256))); write(M, fs, steps)
    rf, rh = recall(M, kind(fs, 'fact'), te), recall(M, kind(hf, 'fact'), te)
    rr_, rw_ = recall(M, kind(fs, True), te), recall(M, kind(fs, False), te)
    hr_, hw_ = recall(M, kind(hf, True), te), recall(M, kind(hf, False), te)
    pf = recall(M, [e for e in kind(fs, 'fact') if e['r'] in _DT_PARA], pa); pr = recall(M, [e for e in kind(fs, True) + kind(fs, False) if e['r'] in _DT_PARA], pa)
    byA = {}
    for e in fs: byA.setdefault(e['A'], []).append(e)
    direct = {(e['A'], e['ans']) for e in fs + hf}
    ch = [(a, b) for a in fs for b in byA.get(a['ans'], []) if b['ans'] != a['A'] and (a['A'], b['ans']) not in direct][:400]
    two = "no chains"
    if ch:
        with torch.no_grad():
            v1, _ = M.read(NK[[ix[a['A']] for a, b in ch]], WK[[te(a) for a, b in ch]]); bn = (F.normalize(v1, dim=-1) @ NK.t()).argmax(-1)
            v2, _ = M.read(NK[bn], WK[[te(b) for a, b in ch]]); vn = F.normalize(v2, dim=-1); rr = np.random.RandomState(seed + 8 + vis); hit = []
            op = (vn @ NK.t()).argmax(-1).tolist(); oe = np.mean([names[o] == b['ans'] for o, (a, b) in zip(op, ch)])
            for k, (a, b) in enumerate(ch):
                alts = [x for x in ans_pool if x not in (b['ans'], a['A'])]
                if len(alts) >= 9:
                    cs = [b['ans']] + [alts[j] for j in rr.choice(len(alts), 9, replace=False)]; hit.append(_first(vn[k] @ NK[[ix[c] for c in cs]].t()))
            hop1 = np.mean([names[int(bn[k])] == a['ans'] for k, (a, b) in enumerate(ch)])
        two = f"open {oe:.2f} / rank {np.mean(hit) if hit else float('nan'):.2f} (n {len(ch)}; first hop decoded right {hop1:.2f})"
    print(f"  [KEYED MEMORY | visit {vis + 1}] name x JEPA-wording key memory ({n_}x{n_} slots, top {_KM_KK}; {len(names)} names, {len(rels)} relations; wording head: "
          f"test wording's relation {w_te:.2f}, paraphrase {w_pa:.2f}) || REBUILT from {len(fs)} taught facts ({steps} write steps, {len(M.touched)} slots written), "
          f"open / rank (chance 0.10) in the TEST wording: facts TAUGHT {fmt(rf)} vs HELD BACK {fmt(rh)} | inferences RIGHT {fmt(rr_)} / WRONG {fmt(rw_)} vs held back "
          f"{fmt(hr_)} / {fmt(hw_)} | PARAPHRASED relation: facts {fmt(pf)}, inferences {fmt(pr)} | TWO-HOP, chain never taught: {two}", flush=True)
    # ERASE on a copy: oracle (the WRONG inferences) and the trigger's flags
    snap = M.v.detach().clone(); out = []
    for tag, X_ in (('ORACLE: the WRONG inferences', kind(fs, False)), ('FLAGGED by the trigger', [e for e in fs if e.get('flag')])):
        if not X_: continue
        Xk = {repr(e['key']) for e in X_}; keep_f = kind(fs, 'fact'); keep_r = [e for e in kind(fs, True) if repr(e['key']) not in Xk]
        b4 = [recall(M, X_, te), recall(M, keep_r, te), recall(M, keep_f, te)]
        with torch.no_grad():
            _, idx = M.read(NK[[ix[e['A']] for e in X_ for k in range(TI)]], WK[[fi_(e['r'], k) for e in X_ for k in range(TI)]])
            ids = torch.unique(idx.flatten()); M.v[ids] = 0.0
        af = [recall(M, X_, te), recall(M, keep_r, te), recall(M, keep_f, te)]
        out.append(f"{tag} ({len(X_)}; {len(ids)} slots): {fmt(b4[0])} -> {fmt(af[0])} | bystanders: RIGHT inferences {fmt(b4[1])} -> {fmt(af[1])}, facts {fmt(b4[2])} -> {fmt(af[2])}")
        with torch.no_grad(): M.v.copy_(snap)
    if out: print(f"  [KEYED MEMORY, ERASE | visit {vis + 1}] slots reached by the retracted facts' training-frame questions set to zero (a copy): " + " || ".join(out), flush=True)
    # PERSISTENT: new facts written, earlier ones refreshed; recall by the readout they were written at
    K = model.__dict__.get('_kmem')
    if K is None: K = model.__dict__['_kmem'] = dict(M=_KMem(n_, d, dev, seed + 1, lr_), when={})
    if K['M'].v.shape[1] != d: return
    new = [e for e in fs if repr(e['key']) not in K['when']]; old = [e for e in fs if repr(e['key']) in K['when']]
    if new: write(K['M'], new, max(50, int(math.ceil(rows_ * len(new) / (256 * (1 - rep_ if old else 1))))), old)
    for e in new: K['when'][repr(e['key'])] = vis
    ages = sorted({K['when'][repr(e['key'])] for e in fs}); parts = []
    for a in ages:
        L = [e for e in fs if K['when'].get(repr(e['key'])) == a]; parts.append(f"written at visit {a + 1}: {fmt(recall(K['M'], L, te))}")
    print(f"  [KEYED MEMORY, PERSISTENT | visit {vis + 1}] kept across readouts, read with today's keys ({len(new)} new facts written, {len(old)} refreshed at "
          f"{rep_:.0%} of each batch; {len(K['M'].touched)} of {n_ * n_} slots written), open / rank in the test wording by when written: " + " | ".join(parts), flush=True)


# ======================================================================================================================================
# 15 Nov: THE KEYED MEMORY INTEGRATED ON THE QUESTION ROWS (inf_kmi; MODEL CHANGE of cg27a_kmi). The system's own question rows (the pool's
# wake rows and the written inferences' dose rows) READ the memory where their answers are predicted: the read (a logit vector over bytes)
# is ADDED to the trunk's logits there, so the model's answer to a question is trunk + memory. The address comes from the row's own bytes:
# SUBJECT = the latest completed capitalised run of 3+ bytes in the question; WORDING = the question with that subject replaced by
# 'someone', through the trunk (last block, mean + last byte) and a JEPA head trained on the system's own frames (warm-started at every
# readout so the keys stay put); OFFSET = which answer byte. The memory learns from the same rows: cross-entropy of (trunk logits,
# DETACHED, + memory read) on the answer bytes, its own optimiser - so the trunk's training is exactly as without it (its losses, its
# gradients, its random stream) and the memory learns what the trunk does not yet know. Ordinary text never reads it.
# Offline (tools/int_mem.py, tools/jepa_mem.py; inference.kmem readout on the GPU, cg27a_kmem): taught facts 0.68-0.92 open in an unseen
# wording against the trunk's 0.06-0.25; erase exact.
# ======================================================================================================================================
import re as _re
_KMI_RUN = _re.compile(rb"[A-Z][A-Za-z0-9'\-\.]*(?: [A-Z][A-Za-z0-9'\-\.]*)*")
_KMI_MAXO = 31


def _kmi_subject(pb):
    """the latest capitalised run of 3+ bytes that is complete before the prompt's end (its next byte is read), or None"""
    best = None
    for m in _KMI_RUN.finditer(pb):
        if m.end() - m.start() >= 3 and m.end() < len(pb): best = m
    return pb[best.start():best.end()].decode('latin-1') if best else None


def _kmi_state(model, cfg, V):
    K = model.__dict__.get('_kmi')
    if K is None:
        dev = model.embed.weight.device; d = int(model.embed.weight.shape[1]); n = int(getattr(cfg, 'inf_kmi_n', 128)); g = torch.Generator(device='cpu').manual_seed(int(cfg.seed) + 5151)
        K = dict(n=n, d=d, V=V, k1=F.normalize(torch.randn(n, _KM_DK, generator=g), dim=-1).to(dev), k2=F.normalize(torch.randn(n, _KM_DK, generator=g), dim=-1).to(dev),
                 R1=(torch.randn(d, _KM_DK, generator=g) / math.sqrt(d)).to(dev), EOFF=F.normalize(torch.randn(_KMI_MAXO + 1, _KM_DK, generator=g), dim=-1).to(dev),
                 v=torch.zeros(n * n, V, device=dev, requires_grad=True), head=None, mu=None, nk={}, wk={}, nmu=None, steps=0, rows=0, hit=0, loss=[], touched=set())
        K['opt'] = torch.optim.Adam([K['v']], lr=float(getattr(cfg, 'inf_kmi_lr', 0.02))); model.__dict__['_kmi'] = K
        K['namekey'] = str(getattr(cfg, 'inf_kmi_namekey', 'trunk')); K['HN'] = 1 << 16; K['gate'] = int(getattr(cfg, 'inf_kmi_gate', 0))
        K['HR'] = F.normalize(torch.randn(K['HN'], d, generator=g), dim=-1).to(dev) if K['namekey'] == 'hash' else None
    return K


def _kmi_hash_key(K, n):
    """15 Nov: a NAME-IDENTITY key independent of the trunk: the sum of fixed random vectors of the name's byte 3-grams (lower case, with
    boundary marks), unit length - one name always the same key, different names nearly orthogonal, spelling variants close. GPU cg27a_kmem:
    the trunk's name codes DRIFT as it trains (facts written at visit 2 read 0.58 -> 0.49 -> 0.20 open by visit 4) and CROWD (written
    inferences 0.57 -> 0.05 open as the trunk learned them, while the trunk itself reached 0.75-0.88 rank)"""
    b = (' ' + n.lower() + ' ').encode('latin-1', 'replace'); ids = [((b[i] * 65599 + b[i + 1]) * 65599 + b[i + 2]) % K['HN'] for i in range(len(b) - 2)] or [0]
    return F.normalize(K['HR'][ids].sum(0), dim=-1)


def _kmi_name(model, K, names):
    from platypus.inference.distill import _name_codes
    if K.get('namekey') == 'hash':
        for n in dict.fromkeys(names):
            if n not in K['nk']: K['nk'][n] = _kmi_hash_key(K, n)
        return torch.stack([K['nk'][n] for n in names])
    new = [n for n in dict.fromkeys(names) if n not in K['nk']]
    if new:
        from platypus.inference.closed_form import _ws_batch, _ws_run
        cap = {}; h9 = model.blocks[-1].register_forward_hook(lambda m_, i_, o_: cap.__setitem__('h', o_.detach()))
        try:
            for i0 in range(0, len(new), 64):
                ch = new[i0:i0 + 64]; X, _ = _ws_batch(model, [f"About {n}.".encode('latin-1', 'replace') for n in ch])
                with torch.no_grad(): _ws_run(model, X, 0)
                for j, n in enumerate(ch): K['nk'][n] = cap['h'][j, 5:5 + len(n.encode('latin-1', 'replace'))].float().mean(0)
        finally: h9.remove()
    E = torch.stack([K['nk'][n] for n in names])
    return F.normalize(E - (K['nmu'] if K['nmu'] is not None else E.mean(0)), dim=-1)


def _kmi_wording(model, K, masked):
    new = [m for m in dict.fromkeys(masked) if m not in K['wk']]
    if new and K['head'] is not None:
        pm, pl = _km_prefix_states(model, new); Z = torch.cat([pm, pl], -1)
        with torch.no_grad(): W = F.normalize(K['head']((Z - K['mu'][0]) / K['mu'][1]), dim=-1)
        for m, w in zip(new, W): K['wk'][m] = w
    return torch.stack([K['wk'][m] for m in masked]) if all(m in K['wk'] for m in masked) else None


def _kmi_read(K, nk, wk, offs):
    q1 = F.normalize(nk @ K['R1'], dim=-1); w2 = F.normalize(wk + 0.5 * K['EOFF'][offs], dim=-1)
    a1, i1 = (q1 @ K['k1'].t()).topk(_KM_KK, -1); a2, i2 = (w2 @ K['k2'].t()).topk(_KM_KK, -1); n = K['n']
    sc = (a1[:, :, None] + a2[:, None, :]).flatten(1); ix = (i1[:, :, None] * n + i2[:, None, :]).flatten(1)
    top, j = sc.topk(_KM_KK, -1); idx = ix.gather(-1, j); w = F.softmax(top / 0.05, -1)
    K['_w'] = w                                                                 # the read weights of the last read (the gate uses them)
    return (w[..., None] * K['v'][idx]).sum(1), idx


def _kmi_addr(model, K, prompts, ns):
    """per row (prompt bytes, answer length): (name keys, wording keys, offsets, row index, position) of every addressed answer byte"""
    rows = []
    for r, (pb, n_) in enumerate(zip(prompts, ns)):
        A = _kmi_subject(pb)
        if A is None or n_ <= 0: continue
        rows.append((r, A, pb.decode('latin-1').replace(A, 'someone'), n_))
    if not rows: return None
    W = _kmi_wording(model, K, [m for _, _, m, _ in rows])
    if W is None: return None
    N = _kmi_name(model, K, [A for _, A, _, _ in rows]); nk, wk, off, rr, pp = [], [], [], [], []; K['_subj'] = []
    for k, (r, A, m, n_) in enumerate(rows):
        for j in range(n_): nk.append(N[k]); wk.append(W[k]); off.append(min(j, _KMI_MAXO)); rr.append(r); pp.append(j); K['_subj'].append(A)
    return torch.stack(nk), torch.stack(wk), torch.tensor(off, device=N.device), rr, pp


def kmi_rows_loss(model, cfg, items, lg, X):
    """the memory learns from question rows (items: (prompt, answer, ...)); lg: the trunk's logits for X = prompt + answer; trunk untouched"""
    if not int(getattr(cfg, 'inf_kmi', 0)): return
    K = _kmi_state(model, cfg, int(lg.shape[-1]))
    if K['head'] is None: return
    try:
        p0s = [len(it[0]) - 1 for it in items]; ns = [min(len(it[1]), lg.shape[1] - p0) for it, p0 in zip(items, p0s)]
        a = _kmi_addr(model, K, [bytes(it[0]) if not isinstance(it[0], bytes) else it[0] for it in items], ns)
        K['rows'] += len(items)
        if a is None: return
        nk, wk, off, rr, pp = a; K['hit'] += len(set(rr))
        z = torch.stack([lg[r, p0s[r] + j].detach().float() for r, j in zip(rr, pp)]); tg = torch.stack([X[r, p0s[r] + 1 + j] for r, j in zip(rr, pp)])
        mv, idx = _kmi_read(K, nk, wk, off); lo = F.cross_entropy(z + mv, tg)
        K['opt'].zero_grad(); lo.backward(); K['opt'].step(); K['steps'] += 1; K['loss'].append(float(lo.detach()))
        K.setdefault('subj', set()).update(K['_subj'])                          # 15 Nov GATE: the subjects written, and the slots written
        if 'wmask' not in K: K['wmask'] = torch.zeros(K['n'] ** 2, dtype=torch.bool, device=idx.device)
        K['wmask'][idx.flatten()] = True
        if len(K['touched']) < 10 ** 6: K['touched'].update(idx.flatten().tolist())
    except Exception:
        from platypus.common import _fail
        _fail('inference.kmem:kmi_rows')


def kmi_logits(model, K, prompts_b, X, lg):
    """the model's logits WITH the memory for rows whose question is prompts_b[r] (answer bytes after it): lg + memory read at answer positions"""
    p0s = [len(pb) - 1 for pb in prompts_b]; ns = [lg.shape[1] - p0 for p0 in p0s]
    a = _kmi_addr(model, K, prompts_b, [min(n, _KMI_MAXO + 1) for n in ns])
    if a is None: return lg
    nk, wk, off, rr, pp = a; mv, idx = _kmi_read(K, nk, wk, off); out = lg.clone().float(); subj = K['_subj']
    if K.get('gate'):                                                           # 15 Nov GATE (inf_kmi_gate): read only where the address is one the memory has
        wm = K.get('wmask'); wf = (K['_w'] * wm[idx].float()).sum(-1) if wm is not None else torch.zeros(len(rr), device=out.device)   # written: the
        ok = [(s_ in K.get('subj', ())) and float(wf[k]) >= 0.5 for k, s_ in enumerate(subj)]   #   subject was written AND half the read weight is on written slots
        first = {}
        for k, r in enumerate(rr): first.setdefault(r, ok[k])                   # the decision is the answer's FIRST byte's, for the whole answer
        K['_open'] = first
        for k, (r, j) in enumerate(zip(rr, pp)):
            if first[r]: out[r, p0s[r] + j] += mv[k]
        return out
    for k, (r, j) in enumerate(zip(rr, pp)): out[r, p0s[r] + j] += mv[k]
    return out


def kmi_refresh(model, cfg, ents, held, vis):
    """at every readout: the name keys (the trunk moved) and the JEPA wording head (warm start) over the pool's relations and frames"""
    from platypus.inference.distill import _dt_forms, _DT_PARA
    fs = [e for e in ents + held if e['key'][0] == 'fact']
    if not fs: return None
    K = _kmi_state(model, cfg, int(cfg.vocab)); dev = model.embed.weight.device; FR = _dt_forms(cfg); TI = len(FR)
    rels = sorted({e['r'] for e in fs}); NR = len(rels)
    if K.get('namekey') != 'hash':                                             # trunk name codes move with the trunk: recomputed; hashed keys never move
        K['nk'] = {}; names = sorted({e['A'] for e in fs} | {e['ans'] for e in fs}); _kmi_name(model, K, names); K['nmu'] = torch.stack([K['nk'][n] for n in names]).mean(0)
    tx = [f.format(A='someone', r=r) for r in rels for f in FR]; pm, pl = _km_prefix_states(model, tx); PC = torch.cat([pm, pl], -1)
    if K['head'] is None:
        K['mu'] = (PC.mean(0), PC.std(0) + 1e-6)
        K['head'] = torch.nn.Sequential(torch.nn.Linear(PC.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, _KM_DK)).to(dev); K['hopt'] = torch.optim.Adam(K['head'].parameters(), lr=1e-3)
    PCn = (PC - K['mu'][0]) / K['mu'][1]; g = np.random.RandomState(int(cfg.seed) + 5252 + vis)
    if NR >= 2 and TI >= 2:
        for st in range(1500 if vis == 0 or 'trained' not in K else 300):
            f1 = g.randint(0, TI, NR); f2 = (f1 + g.randint(1, TI, NR)) % TI
            z1 = F.normalize(K['head'](PCn[[r * TI + int(a) for r, a in enumerate(f1)]]), dim=-1); z2 = F.normalize(K['head'](PCn[[r * TI + int(b) for r, b in enumerate(f2)]]), dim=-1)
            lo = F.cross_entropy(z1 @ z2.t() / 0.1, torch.arange(NR, device=dev)); K['hopt'].zero_grad(); lo.backward(); K['hopt'].step()
        K['trained'] = True
    K['wk'] = {}
    _kmi_wording(model, K, tx + [f"It is on record that someone {r} " for r in rels] + [f"someone {_DT_PARA[r]} " for r in rels if r in _DT_PARA])
    return K


def kmi_readout(model, cfg, S_slot, ents, held, vis):
    """READOUT: the model's answers WITH the memory vs WITHOUT (trunk only), test wording, on the pool's facts and the written inferences"""
    from platypus.inference.distill import _DT_PARA
    from platypus.inference.closed_form import _ws_batch, _ws_run
    K = model.__dict__.get('_kmi')
    fs = [e for e in ents if e['key'][0] == 'fact']; hf = [e for e in held if e['key'][0] == 'fact']
    if K is None or K['head'] is None or not fs:
        print(f"  [INTEGRATED KEYED MEMORY | visit {vis + 1}] not yet trained", flush=True); return
    RTI_R = 'is connected to'; rs = np.random.RandomState(int(cfg.seed) + 5353 + vis); pool = sorted({e['ans'] for e in fs + hf})
    def score(L, form, cap=60, greedy=True):
        L = L[:cap]; on, off, oon, ooff, gop = [], [], [], [], []
        for e in L:
            pr = form(e); pb = pr.encode('latin-1', 'replace'); alts = [x for x in pool if x not in (e['ans'], e['A'])]
            if len(alts) < 9: continue
            cs = [e['ans']] + [alts[j] for j in rs.choice(len(alts), 9, replace=False)]; cb = [(c + '.').encode('latin-1', 'replace') for c in cs]
            X, _ = _ws_batch(model, [pb + c for c in cb]); sl = int(S_slot(e['d']))
            with torch.no_grad():
                lg = _ws_run(model, X, sl).float(); K['_open'] = {}; lgm = kmi_logits(model, K, [pb] * len(cb), X, lg); gop.append(int(bool(K['_open'].get(0, False))))
                for z, acc in ((lg, off), (lgm, on)):
                    lp = F.log_softmax(z, -1); p0 = len(pb) - 1; sc = []
                    for j, c in enumerate(cb):
                        n_ = min(len(c), lp.shape[1] - p0); sc.append(float(lp[j, torch.arange(p0, p0 + n_, device=lp.device), X[j, p0 + 1:p0 + 1 + n_]].mean()) if n_ > 0 else -1e9)
                    acc.append(int(sc[0] > max(sc[1:])))
                tb = e['ans'].encode('latin-1', 'replace'); outs = []
                for use in ((False, True) if greedy else ()):                                      # greedy, byte by byte: the exact name
                    o = b''
                    for _ in range(len(tb)):
                        Xg, _ = _ws_batch(model, [pb + o + b' ']); zg = _ws_run(model, Xg, sl).float()
                        if use: zg = kmi_logits(model, K, [pb], Xg, zg)
                        q = len(pb) + len(o) - 1
                        if q >= zg.shape[1]: break
                        o += bytes([int(zg[0, q].argmax())])
                    outs.append(int(o == tb))
                if outs: ooff.append(outs[0]); oon.append(outs[1])
        if not on: return "- (0)"
        g_ = f", gate open {np.mean(gop):.2f}" if K.get('gate') else ''
        if not oon: return f"WITH - / {np.mean(on):.2f} vs trunk - / {np.mean(off):.2f} ({len(on)}{g_})"
        return f"WITH {np.mean(oon):.2f} / {np.mean(on):.2f} vs trunk {np.mean(ooff):.2f} / {np.mean(off):.2f} ({len(on)}{g_})"
    te = lambda e: f"It is on record that {e['A']} {e['r']} "; pa = lambda e: f"It is on record that {e['A']} {_DT_PARA[e['r']]} "
    kind = lambda L, t=None: [e for e in L if (e['r'] != RTI_R if t == 'fact' else (e['r'] == RTI_R and e.get('truth') is t))]
    rs.shuffle(fs); rs.shuffle(hf)
    lm = K['loss']; ml = f"{np.mean(lm[:50]):.3f} -> {np.mean(lm[-50:]):.3f}" if lm else "-"
    print(f"  [INTEGRATED KEYED MEMORY | visit {vis + 1}] the model's answer = trunk + memory read (question rows only; subject key: {K.get('namekey')}; {K['steps']} memory steps, "
          f"{K['hit']} of {K['rows']} rows addressed, {len(K['touched'])} of {K['n'] ** 2} slots used, memory loss {ml}), test wording, open (greedy exact) / "
          f"rank among 10 (chance 0.10) || taught FACTS {score(kind(fs, 'fact'), te)} | HELD BACK facts {score(kind(hf, 'fact'), te, 30)} | taught inferences "
          f"RIGHT {score(kind(fs, True), te)} / WRONG {score(kind(fs, False), te, 40)} | held back inferences {score(kind(hf, True) + kind(hf, False), te, 30)} | "
          f"PARAPHRASED facts {score([e for e in kind(fs, 'fact') if e['r'] in _DT_PARA], pa, 40)}", flush=True)
    if K.get('gate'):                                                          # ABSTENTION: questions about names the system never read
        from platypus.inference.distill import _DT_REFUSE_SYL
        rels = sorted({e['r'] for e in fs}); g9 = np.random.RandomState(int(cfg.seed) + 5454 + vis); op9 = []
        for k in range(60):
            nm = ''.join(g9.choice(_DT_REFUSE_SYL, 2)).title() + ' ' + ''.join(g9.choice(_DT_REFUSE_SYL, 3)).title()
            pb = f"It is on record that {nm} {rels[g9.randint(len(rels))]} ".encode('latin-1'); X, _ = _ws_batch(model, [pb + b'Xx.'])
            with torch.no_grad(): lg = _ws_run(model, X, 0).float(); K['_open'] = {}; kmi_logits(model, K, [pb], X, lg)
            op9.append(int(bool(K['_open'].get(0, False))))
        print(f"  [INTEGRATED KEYED MEMORY, GATE | visit {vis + 1}] the memory answers only where its address was written ({len(K.get('subj', ()))} subjects written, "
              f"{int(K['wmask'].sum()) if 'wmask' in K else 0} slots): gate OPEN on questions about NEVER-READ names {np.mean(op9):.2f} (60; the memory abstains on the rest - "
              f"the trunk answers these with a stored name, false-memory probe a)", flush=True)
    if int(getattr(cfg, 'inf_kmi_counter', 0)):                                # COUNTER-WRITE (oracle, on a copy): unlearn the WRONG inferences, re-write the rest
        try: _kmi_counter_readout(model, cfg, K, S_slot, kind(fs, True), kind(fs, False), kind(fs, 'fact'), vis)
        except Exception as e9:
            from platypus.common import _fail
            _fail('inference.kmem:counter'); print(f"  [INTEGRATED KEYED MEMORY, COUNTER-WRITE] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
    K['loss'] = lm[-200:]


def _kmi_counter_readout(model, cfg, K, S_slot, right, wrong, facts, vis):
    """ORACLE (the labels pick the wrong inferences - the mechanism, not a trigger), on a COPY of the memory: 60 steps driving the memory's
    support for the WRONG inferences' answer bytes (log-probability with it minus without) to zero while re-writing the RIGHT ones (same subjects share slots) - offline (tools/jepa_mem.py JM_M2) this
    removed the retracted (open 0.00) and kept the sibling (0.85-0.95); zeroing the slots killed both"""
    from platypus.inference.distill import _dt_forms
    from platypus.inference.closed_form import _ws_batch, _ws_run
    if not wrong or not right: return
    FR = _dt_forms(cfg); g = np.random.RandomState(int(cfg.seed) + 5555 + vis)
    def rows(L, n):
        it = [(FR[g.randint(len(FR))].format(A=e['A'], r=e['r']).encode('latin-1', 'replace'), (e['ans'] + '.').encode('latin-1', 'replace'), int(S_slot(e['d']))) for e in (L[g.randint(len(L))] for _ in range(n))]
        X, _ = _ws_batch(model, [a + b for a, b, c in it])
        with torch.no_grad(): lg = _ws_run(model, X, it[0][2]).float()
        return it, X, lg
    def evalg(L):                                                              # rank among 10 by mean answer-byte log-probability, WITH the memory (no gate)
        hit, sup = [], []; pool = sorted({e['ans'] for e in right + wrong + facts}); rr = np.random.RandomState(int(cfg.seed) + 5656)
        for e in L[:40]:
            alts = [x for x in pool if x not in (e['ans'], e['A'])]
            if len(alts) < 9: continue
            cs = [e['ans']] + [alts[j] for j in rr.choice(len(alts), 9, replace=False)]; pb = f"It is on record that {e['A']} {e['r']} ".encode('latin-1', 'replace')
            cb = [(c + '.').encode('latin-1', 'replace') for c in cs]; X, _ = _ws_batch(model, [pb + c for c in cb])
            with torch.no_grad():
                lg = _ws_run(model, X, int(S_slot(e['d']))).float(); gt = K.get('gate'); K['gate'] = 0; z = kmi_logits(model, K, [pb] * len(cb), X, lg); K['gate'] = gt
                lp = F.log_softmax(z, -1); l0 = F.log_softmax(lg, -1); p0 = len(pb) - 1; n0 = min(len(cb[0]), lp.shape[1] - p0); ar = torch.arange(p0, p0 + n0, device=lp.device)
                sc = [float(lp[j, torch.arange(p0, p0 + min(len(c), lp.shape[1] - p0), device=lp.device), X[j, p0 + 1:p0 + 1 + min(len(c), lp.shape[1] - p0)]].mean()) for j, c in enumerate(cb)]
                sup.append(float((lp[0, ar, X[0, p0 + 1:p0 + 1 + n0]] - l0[0, ar, X[0, p0 + 1:p0 + 1 + n0]]).mean()) / math.log(2))
            hit.append(int(sc[0] > max(sc[1:])))
        return (np.mean(hit) if hit else float('nan'), np.mean(sup) if sup else float('nan'))
    b4 = [evalg(x) for x in (wrong, right, facts)]; snap = K['v'].detach().clone(); st0 = K['opt'].state_dict()
    for st in range(60):
        for L, ul in ((wrong, True), (right, False)):
            it, X, lg = rows(L, 32); p0s = [len(a) - 1 for a, b, c in it]; ns = [min(len(b), lg.shape[1] - p0) for (a, b, c), p0 in zip(it, p0s)]
            a = _kmi_addr(model, K, [a for a, b, c in it], ns)
            if a is None: continue
            nk, wk, off, rr, pp = a; mv, _ = _kmi_read(K, nk, wk, off)
            z0 = torch.stack([lg[r, p0s[r] + j] for r, j in zip(rr, pp)]); z = z0 + mv; tg = torch.stack([X[r, p0s[r] + 1 + j] for r, j in zip(rr, pp)])
            if ul:                                                              # unlearn: the memory's SUPPORT for the wrong answer's bytes driven to zero or below
                lo = (F.log_softmax(z, -1).gather(-1, tg[:, None]) - F.log_softmax(z0, -1).gather(-1, tg[:, None])).clamp(min=0).mean()
            else: lo = F.cross_entropy(z, tg)
            K['opt'].zero_grad(); lo.backward(); K['opt'].step()
    af = [evalg(x) for x in (wrong, right, facts)]
    with torch.no_grad(): K['v'].copy_(snap)
    K['opt'].load_state_dict(st0)
    print(f"  [INTEGRATED KEYED MEMORY, COUNTER-WRITE | visit {vis + 1}] ORACLE on a copy (the labels pick them): 60 steps unlearning the WRONG inferences while "
          f"re-writing the RIGHT ones; rank WITH the memory (chance 0.10) and the MEMORY'S SUPPORT for the answer (bits per answer byte the memory adds; 0 = none): "
          f"WRONG {b4[0][0]:.2f}, {b4[0][1]:+.2f} -> {af[0][0]:.2f}, {af[0][1]:+.2f} | RIGHT {b4[1][0]:.2f}, {b4[1][1]:+.2f} -> {af[1][0]:.2f}, {af[1][1]:+.2f} | "
          f"facts {b4[2][0]:.2f}, {b4[2][1]:+.2f} -> {af[2][0]:.2f}, {af[2][1]:+.2f}", flush=True)
