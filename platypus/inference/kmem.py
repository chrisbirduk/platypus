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
import math, collections
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
    from platypus.inference.premises import is_rti as _is_rti            # 16 Nov: typed writes ('head wording tail') are inferences too
    kind = lambda L, t=None: [e for e in L if (not _is_rti(e) if t == 'fact' else (_is_rti(e) and e.get('truth') is t))]
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
        K['allt'] = int(getattr(cfg, 'inf_kma', 0)); K['gatew'] = float(getattr(cfg, 'inf_kmi_gatew', 0.5))   # 15 Nov ALL-TEXT address; gate weight
        if K['allt']: K['namekey'] = 'hash'                                     #   (all text: the identity hash, never the trunk's moving name codes)
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


# ======================================================================================================================================
# 15 Nov ALL-TEXT ADDRESSING (inf_kma; MODEL CHANGE when on, cg27a_kma): the memory's address is computed at EVERY byte of ANY text from
# the trunk's own forward pass - no separate masked pass, no question template - so the model (trunk + gated memory) predicts all text
# through it; the memory still WRITES only on the answer bytes of question rows.
#   SUBJECT   the latest COMPLETED capitalised run of 3+ bytes (causal: a run followed by a space stays open one byte longer - the next
#             byte may continue it - so the address never depends on the byte being predicted); its key is the byte 3-gram identity hash
#   WORDING   the bytes from the subject's end to the current position - or, inside a capitalised run that began after the subject (a
#             name being spelled), to that run's start; key = a small head over the MIDDLE block's states on those bytes (mean + last),
#             trained JEPA/InfoNCE-style so the training frames of one relation meet across subjects (in context: the subject is NOT masked)
#   OFFSET    bytes of the name being spelled so far (0 at its first byte), added to the wording key
#   GATE      (inf_kmi_gate) the subject was written AND inf_kmi_gatew of the read weight lies on written slots; a name being spelled takes
#             its first byte's decision
# Offline (tools/int_mem.py IM_GATE=1, stand-in trunk): gated reading on all text cost wiki +0.000-0.001 / dev +0.015-0.036 bits with recall
# unchanged (rank 1.00, exact 0.69-0.77); writes on answer bytes only (here) remove the frame words from the gate's subject index.
# ======================================================================================================================================
_KMA_MAXW = 96


def _kma_parse(b):
    """per state index i (it has read b[:i+1]): (i, subjects, wording start, wording end, offset) for every addressed position, or None"""
    runs = [(m.start(), m.end()) for m in _KMI_RUN.finditer(b)]
    if not runs: return None
    S0 = np.array([s for s, e in runs]); E0 = np.array([e for s, e in runs]); nb = len(b)
    ext = E0 + np.array([int(e < nb and b[e] == 32) for e in E0])              # a run followed by a space may go on: open one byte longer
    el = np.where(E0 - S0 >= 3)[0]
    if not len(el): return None
    t = np.arange(1, nb + 1)
    k = np.searchsorted(ext[el] + 1, t, 'right') - 1                          # the latest eligible run completed by t
    j = np.searchsorted(S0, t, 'left') - 1                                     # the latest run begun before t
    sj = el[np.maximum(k, 0)]; ws = E0[sj]; jj = np.maximum(j, 0)
    cur = (j >= 0) & (t <= ext[jj]) & (S0[jj] > ws)                            # a name being spelled after the subject
    we = np.where(cur, S0[jj], t); off = np.where(cur, t - S0[jj], 0)
    ok = (k >= 0) & (we - ws >= 1) & (we - ws <= _KMA_MAXW) & (off <= _KMI_MAXO)
    i = np.where(ok)[0]
    if not len(i): return None
    return i, [b[S0[s]:E0[s]].decode('latin-1') for s in sj[i]], ws[i], we[i], off[i]


def _kma_run(model, X, sl, grad=False):
    """the trunk's logits of X[:, :-1] and the middle block's states (detached, float), one forward"""
    from platypus.inference.closed_form import _ws_run
    cap = {}; blk = model.blocks[len(model.blocks) // 2]
    h9 = blk.register_forward_hook(lambda m_, i_, o_: cap.__setitem__('h', (o_[0] if isinstance(o_, tuple) else o_).detach()))
    try: lg = _ws_run(model, X, int(sl or 0), grad)
    finally: h9.remove()
    return lg, cap['h'].float()


def _kma_feats(X, H, spans=None):
    """addressed positions of every row of X (bytes read: X[:, :T], T = H's length) and their wording features (middle states, mean + last)"""
    T = H.shape[1]; Xc = X[:, :T].detach().cpu().numpy().astype(np.uint8); R, I, S, WS, WE, OF = [], [], [], [], [], []
    for r in range(Xc.shape[0]):
        if spans is not None and r not in spans: continue
        a = _kma_parse(bytes(Xc[r]))
        if a is None: continue
        i, sj, ws, we, off = a
        if spans is not None:
            m = (i >= spans[r][0]) & (i < spans[r][1])
            if not m.any(): continue
            i, ws, we, off = i[m], ws[m], we[m], off[m]; sj = [s for s, q in zip(sj, m) if q]
        R.append(np.full(len(i), r)); I.append(i); S += sj; WS.append(ws); WE.append(we); OF.append(off)
    if not S: return None
    R, I, WS, WE, OF = map(np.concatenate, (R, I, WS, WE, OF)); dev = H.device; C = H.cumsum(1)
    rt = torch.as_tensor(R, device=dev); a_ = torch.as_tensor(WS, device=dev); b_ = torch.as_tensor(WE, device=dev)
    Z = torch.cat([(C[rt, b_ - 1] - C[rt, a_ - 1]) / (b_ - a_).float()[:, None], H[rt, b_ - 1]], -1)
    return dict(r=R, i=I, s=S, ws=WS, we=WE, off=OF, Z=Z, offt=torch.as_tensor(OF, device=dev))


def _kma_keys(model, K, a):
    if len(K['nk']) > 200000: K['nk'] = {}                                      # the hash keys are a cache (never move): bounded
    with torch.no_grad(): a['wk'] = F.normalize(K['head']((a['Z'] - K['mu'][0]) / K['mu'][1]), dim=-1)
    a['nk'] = _kmi_name(model, K, a['s']); return a


def _kma_gate(K, a, idx, no_rti=False):
    """open where the subject was written and inf_kmi_gatew of the read weight is on written slots; a name being spelled follows its first byte"""
    wm = K.get('wmask'); dev = idx.device
    if wm is None: return torch.zeros(len(a['s']), dtype=torch.bool, device=dev)
    wf = (K['_w'] * wm[idx].float()).sum(-1); subj = K.get('subj', set())
    ok = (wf >= float(K.get('gatew', 0.5))).cpu().numpy() & np.array([s in subj for s in a['s']])
    if no_rti and K.get('wmask_rti') is not None:                              # 15 Nov GATE RULE (readout): reads that fall mostly on the untyped
        ok &= ((K['_w'] * K['wmask_rti'][idx].float()).sum(-1) < 0.5).cpu().numpy()   #   inferences' slots stay shut on text
    sp = np.where(a['off'] > 0)[0]
    if len(sp):
        key = {(int(r), int(i)): q for q, (r, i) in enumerate(zip(a['r'], a['i']))}
        for q in sp:
            q0 = key.get((int(a['r'][q]), int(a['we'][q]) - 1))
            if q0 is not None: ok[q] = ok[q0]
    return torch.as_tensor(ok, device=dev)


def kma_logits(model, K, X, lg, H, gate=None, no_rti=False):
    """the model's logits: trunk + the (gated) memory read at EVERY addressed byte; also (row, state index) -> open, addressed and open counts"""
    out = lg.clone().float(); a = _kma_feats(X, H)
    if a is None: return out, {}, 0, 0
    _kma_keys(model, K, a); mv, idx = _kmi_read(K, a['nk'], a['wk'], a['offt'])
    g_ = K.get('gate') if gate is None else gate
    ok = _kma_gate(K, a, idx, no_rti) if g_ else torch.ones(len(a['s']), dtype=torch.bool, device=out.device)
    rt = torch.as_tensor(a['r'], device=out.device); it = torch.as_tensor(a['i'], device=out.device)
    if bool(ok.any()): out = out.index_put((rt[ok], it[ok]), mv[ok].to(out.dtype), accumulate=True)
    okl = ok.tolist(); return out, {(int(r), int(i)): o for r, i, o in zip(a['r'], a['i'], okl)}, len(okl), int(sum(okl))


def _kma_addr_rows(model, K, X, sl, p0s, ns):
    """as _kmi_addr, from the all-text address: the answer bytes of each row (state p0 + j predicts answer byte j)"""
    _, H = _kma_run(model, X, sl); spans = {r: (p0, p0 + n) for r, (p0, n) in enumerate(zip(p0s, ns)) if n > 0}
    a = _kma_feats(X, H, spans)
    if a is None: return None
    _kma_keys(model, K, a); K['_subj'] = list(a['s'])
    return a['nk'], a['wk'], a['offt'], [int(r) for r in a['r']], [int(i) - p0s[int(r)] for r, i in zip(a['r'], a['i'])]


def _kmi_track(K, items, rr, z, mv, tg):
    """15 Nov READOUT (Go-CLS, Sun et al. 2023: consolidate what is PREDICTABLE; learning speed as its estimate): per written inference,
    the memory's loss on its answer bytes each time it is learned, and the trunk's own loss on them (how predictable it was without it)"""
    try:
        from platypus.inference.premises import _RTI_R
        with torch.no_grad():
            cm = F.cross_entropy(z.detach() + mv.detach(), tg, reduction='none'); ct = F.cross_entropy(z.detach(), tg, reduction='none')
        acc = {}
        for q, r in enumerate(rr): acc.setdefault(r, []).append(q)
        L = K.setdefault('ls', {})
        for r, qs in acc.items():
            k = str(items[r][3]) if len(items[r]) > 3 else ''
            if _RTI_R not in k: continue
            key = __import__('ast').literal_eval(k); kk = (key[1], key[3])
            if len(L.get(kk, ())) < 64: L.setdefault(kk, []).append((K['steps'], float(cm[qs].mean()), float(ct[qs].mean())))
    except Exception:
        pass


def kmi_rows_loss(model, cfg, items, lg, X, sl=None):
    """the memory learns from question rows (items: (prompt, answer, ...)); lg: the trunk's logits for X = prompt + answer; trunk untouched"""
    if not int(getattr(cfg, 'inf_kmi', 0)): return
    K = _kmi_state(model, cfg, int(lg.shape[-1]))
    if K['head'] is None: return
    if K.get('allt'):                                                          # 15 Nov ALL-TEXT ADDRESS (inf_kma): from the rows' own forward
        try:
            p0s = [len(it[0]) - 1 for it in items]; ns = [max(0, min(len(it[1]), lg.shape[1] - p0)) for it, p0 in zip(items, p0s)]
            a = _kma_addr_rows(model, K, X, sl, p0s, ns); K['rows'] += len(items)
            if a is None: return
            nk, wk, off, rr, pp = a; K['hit'] += len(set(rr)); dev = lg.device
            rt = torch.as_tensor(rr, device=dev); it_ = torch.as_tensor([p0s[r] + j for r, j in zip(rr, pp)], device=dev)
            z = lg[rt, it_].detach().float(); tg = X[rt, it_ + 1]
            mv, idx = _kmi_read(K, nk, wk, off); lo = F.cross_entropy(z + mv, tg); _kmi_track(K, items, rr, z, mv, tg)
            K['opt'].zero_grad(); lo.backward()
            if int(getattr(cfg, 'inf_kmi_train', 1)): K['opt'].step()          # 15 Nov CONTROL (inf_kmi_train 0): everything computed, the values never move
            K['steps'] += 1; K['loss'].append(float(lo.detach())); K.setdefault('subj', set()).update(K['_subj'])
            if 'wmask' not in K: K['wmask'] = torch.zeros(K['n'] ** 2, dtype=torch.bool, device=idx.device)
            K['wmask'][idx.flatten()] = True
            try:                                                               # 15 Nov: the slots written by the untyped inferences' rows (gate-rule readout)
                from platypus.inference.premises import _RTI_R
                ri = [q for q, r in enumerate(rr) if len(items[r]) > 3 and _RTI_R in str(items[r][3])]
                if ri:
                    if 'wmask_rti' not in K: K['wmask_rti'] = torch.zeros(K['n'] ** 2, dtype=torch.bool, device=idx.device)
                    K['wmask_rti'][idx[ri].flatten()] = True
            except Exception: pass
            if len(K['touched']) < 10 ** 6: K['touched'].update(idx.flatten().tolist())
        except Exception:
            from platypus.common import _fail
            _fail('inference.kmem:kma_rows')
        return
    try:
        p0s = [len(it[0]) - 1 for it in items]; ns = [min(len(it[1]), lg.shape[1] - p0) for it, p0 in zip(items, p0s)]
        a = _kmi_addr(model, K, [bytes(it[0]) if not isinstance(it[0], bytes) else it[0] for it in items], ns)
        K['rows'] += len(items)
        if a is None: return
        nk, wk, off, rr, pp = a; K['hit'] += len(set(rr))
        z = torch.stack([lg[r, p0s[r] + j].detach().float() for r, j in zip(rr, pp)]); tg = torch.stack([X[r, p0s[r] + 1 + j] for r, j in zip(rr, pp)])
        mv, idx = _kmi_read(K, nk, wk, off); lo = F.cross_entropy(z + mv, tg); _kmi_track(K, items, rr, z, mv, tg)
        K['opt'].zero_grad(); lo.backward()
        if int(getattr(cfg, 'inf_kmi_train', 1)): K['opt'].step()              # 15 Nov CONTROL (inf_kmi_train 0): everything computed, the values never move
        K['steps'] += 1; K['loss'].append(float(lo.detach()))
        K.setdefault('subj', set()).update(K['_subj'])                          # 15 Nov GATE: the subjects written, and the slots written
        if 'wmask' not in K: K['wmask'] = torch.zeros(K['n'] ** 2, dtype=torch.bool, device=idx.device)
        K['wmask'][idx.flatten()] = True
        if len(K['touched']) < 10 ** 6: K['touched'].update(idx.flatten().tolist())
    except Exception:
        from platypus.common import _fail
        _fail('inference.kmem:kmi_rows')


def kmi_logits(model, K, prompts_b, X, lg, sl=None):
    """the model's logits WITH the memory for rows whose question is prompts_b[r] (answer bytes after it): lg + memory read at answer positions"""
    p0s = [len(pb) - 1 for pb in prompts_b]; ns = [lg.shape[1] - p0 for p0 in p0s]
    if K.get('allt'):                                                          # 15 Nov ALL-TEXT: the read at every addressed byte of the row
        _, H = _kma_run(model, X, sl); out, opn, _, _ = kma_logits(model, K, X, lg, H)
        K['_open'] = {r: bool(opn.get((r, p0), False)) for r, p0 in enumerate(p0s)}
        return out
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


def _kma_refresh(model, cfg, K, fs, FR, vis, S_slot):
    """ALL-TEXT wording head: InfoNCE over the middle block's states at the end of each training frame WITH its real subject - one relation's
    frames meet across subjects and frames (anchor and positive differ in both); warm start after the first readout"""
    from platypus.inference.closed_form import _ws_batch
    dev = model.embed.weight.device; g = np.random.RandomState(int(cfg.seed) + 5252 + vis); by = {}
    for e in fs: by.setdefault(e['r'], {}).setdefault(e['A'], e['d'])
    rels = [r for r in sorted(by) if len(by[r]) >= 2]
    if len(rels) < 2: return
    pre = []
    for ri, r in enumerate(rels):
        As = sorted(by[r]); As = [As[q] for q in g.choice(len(As), min(12, len(As)), replace=False)]
        for A in As:
            for k, f in enumerate(FR): pre.append((f.format(A=A, r=r).encode('latin-1', 'replace'), ri, A, k, int(S_slot(by[r][A])) if S_slot else 0))
    Z, lab = [], []
    for sl in sorted({p[4] for p in pre}):
        P_ = [p for p in pre if p[4] == sl]
        for i0 in range(0, len(P_), 64):
            ch = P_[i0:i0 + 64]; X, _ = _ws_batch(model, [p[0] + b'x' for p in ch])
            with torch.no_grad(): _, H = _kma_run(model, X, sl)
            a = _kma_feats(X, H, {q: (len(p[0]) - 1, len(p[0])) for q, p in enumerate(ch)})
            if a is None: continue
            for q, r in enumerate(a['r']): Z.append(a['Z'][q]); lab.append(ch[int(r)][1:4])
    if len(Z) < 8: return
    FT = torch.stack(Z)
    if K['head'] is None:
        K['mu'] = (FT.mean(0), FT.std(0) + 1e-6)
        K['head'] = torch.nn.Sequential(torch.nn.Linear(FT.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, _KM_DK)).to(dev); K['hopt'] = torch.optim.Adam(K['head'].parameters(), lr=1e-3)
    FTn = (FT - K['mu'][0]) / K['mu'][1]; idx = {}
    for q, (ri, A, k) in enumerate(lab): idx.setdefault(ri, []).append(q)
    RL = [ri for ri in idx if len({lab[q][1] for q in idx[ri]}) >= 2]
    if len(RL) < 2: return
    for st in range(1500 if 'trained' not in K else 300):
        a_, b_ = [], []
        for ri in RL:
            q = idx[ri][g.randint(len(idx[ri]))]; c = [p for p in idx[ri] if lab[p][1] != lab[q][1] and lab[p][2] != lab[q][2]] or [p for p in idx[ri] if lab[p][1] != lab[q][1]]
            a_.append(q); b_.append(c[g.randint(len(c))])
        z1 = F.normalize(K['head'](FTn[a_]), dim=-1); z2 = F.normalize(K['head'](FTn[b_]), dim=-1)
        lo = F.cross_entropy(z1 @ z2.t() / 0.1, torch.arange(len(RL), device=dev)); K['hopt'].zero_grad(); lo.backward(); K['hopt'].step()
    K['trained'] = True; K['hloss'] = float(lo.detach()); K['hrels'] = len(RL)


def kmi_refresh(model, cfg, ents, held, vis, S_slot=None):
    """at every readout: the name keys (the trunk moved) and the JEPA wording head (warm start) over the pool's relations and frames"""
    from platypus.inference.distill import _dt_forms, _DT_PARA
    fs = [e for e in ents + held if e['key'][0] == 'fact']
    if not fs: return None
    K = _kmi_state(model, cfg, int(cfg.vocab)); dev = model.embed.weight.device; FR = _dt_forms(cfg); TI = len(FR)
    if K.get('allt'):
        _kma_refresh(model, cfg, K, fs, FR, vis, S_slot); return K
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


def _kmi_status_readout(model, cfg, K, fs, score, te, vis):
    """15 Nov READOUTS. BY STATUS: the written inferences split by consolidation status (judged and KEPT - taught to the trunk; FLAGGED - never
    taught; AWAITING judgement - memory only) and by truth, WITH the memory vs the trunk: kept ones should reach the trunk, flagged and
    awaiting ones should not; also by the visit written (a fresh batch is not yet in the memory). LEARNING SPEED AS A JUDGE (Go-CLS):
    per inference, the memory's loss at its first and latest learning and the trunk's own loss at its first (predictability), AUC right vs
    wrong (label-free signals; the labels only score them)"""
    from platypus.inference.premises import _PM, _RTI_R, rti_ent, is_rti, rti_wkey
    from platypus.inference.compress import _cg_auc
    W = _PM.get('rti') or {}
    if not W: return
    inf_ = [e for e in fs if is_rti(e) and rti_wkey(e) in W]
    st = lambda e: W[rti_wkey(e)]
    grp = [('KEPT right', lambda w: w.get('judged') is not None and not w.get('flag') and w['truth']), ('KEPT wrong', lambda w: w.get('judged') is not None and not w.get('flag') and not w['truth']),
           ('FLAGGED right', lambda w: w.get('flag') and not w.get('ret') and w['truth']), ('FLAGGED wrong', lambda w: w.get('flag') and not w.get('ret') and not w['truth']),
           ('RETRACTED (removed from the memory)', lambda w: bool(w.get('ret'))),
           ('AWAITING (memory only)', lambda w: w.get('judged') is None)]
    seen = {rti_wkey(e) for e in inf_}                                         # retracted inferences leave the taught pool: take them from the ledger
    inf_ = inf_ + [rti_ent((A, C), w) for (A, C), w in W.items() if w.get('ret') and (A, C) not in seen]
    parts = [f"{nm} {score([e for e in inf_ if f(st(e))], te, 30, False)}" for nm, f in grp]
    vs = sorted({st(e).get('vis') for e in inf_ if st(e).get('vis') is not None})
    parts += [f"written at visit {v + 1}: {score([e for e in inf_ if st(e).get('vis') == v], te, 30, False)}" for v in vs[-2:]]
    print(f"  [INTEGRATED KEYED MEMORY, BY STATUS | visit {vis + 1}] written inferences, rank among 10 WITH the memory vs the trunk alone || " + " | ".join(parts), flush=True)
    L = K.get('ls') or {}; rows = []
    for kk, h in L.items():
        if kk in W and len(h) >= 4: rows.append((bool(W[kk]['truth']), h))
    if len(rows) >= 20 and any(t for t, h in rows) and any(not t for t, h in rows):
        y = np.array([t for t, h in rows])
        sig = {'memory loss at first learning (low = predictable)': np.array([-np.mean([x[1] for x in h[:2]]) for t, h in rows]),
               'memory loss now (low = learned)': np.array([-np.mean([x[1] for x in h[-2:]]) for t, h in rows]),
               'LEARNING SPEED (loss first - now)': np.array([np.mean([x[1] for x in h[:2]]) - np.mean([x[1] for x in h[-2:]]) for t, h in rows]),
               "TRUNK's own loss at first (predictability)": np.array([-np.mean([x[2] for x in h[:2]]) for t, h in rows])}
        print(f"  [LEARNING SPEED AS A JUDGE (Go-CLS) | visit {vis + 1}] {len(rows)} written inferences learned 4+ times (right {y.mean():.2f}); AUC right vs wrong "
              f"(label-free signals; 0.5 = none; the current judge 0.64): " + " | ".join(f"{k} {_cg_auc(v[y], v[~y]):.3f}" for k, v in sig.items()), flush=True)


def _kma_twohop(model, cfg, K, S_slot, fs, pool, te, vis, cap=30, memonly=False):
    """15 Nov READOUT, TWO-HOP THROUGH THE MEMORY (after REMERGE, Kumaran & McClelland 2012: the hippocampus's output fed back as the next
    query): chains of TAUGHT facts A r1 B, B r2 C never taught as a chain. The model decodes B greedily after 'It is on record that A r1 ', then
    is asked 'So B' r2 ' with its OWN decoded B' - with the all-text address the decoded name becomes the next subject. WITH the memory (gated)
    vs the trunk alone (each decodes its own B'): first hop exact, and C's rank among 10 (chance 0.10); also with the TRUE B given (ceiling)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    from platypus.inference.premises import _RTI_R, is_rti, rti_wkey
    if memonly:                                                                # 15 Nov: chains with a hop ONLY the memory holds (an inference awaiting
        from platypus.inference.premises import _PM                            #   judgement: never taught to the trunk) - composition through the memory
        W = _PM.get('rti') or {}; MO = [e for e in fs if is_rti(e) and (W.get(rti_wkey(e)) or {}).get('judged', 0) is None]
        F0 = [e for e in fs if not is_rti(e)] + MO; mo = {id(e) for e in MO}; byA = {}
        for e in F0: byA.setdefault(e['A'], []).append(e)
        ch = [(e1, e2) for e1 in F0 for e2 in byA.get(e1['ans'], ()) if e2['ans'] not in (e1['A'], e1['ans']) and (id(e1) in mo or id(e2) in mo)]
    else:
        F_ = [e for e in fs if not is_rti(e)]; byA = {}
        for e in F_: byA.setdefault(e['A'], []).append(e)
        ch = [(e1, e2) for e1 in F_ for e2 in byA.get(e1['ans'], ()) if e2['ans'] not in (e1['A'], e1['ans'])]
    tag = 'TWO-HOP, MEMORY-ONLY HOP' if memonly else 'TWO-HOP'
    if len(ch) < 5:
        print(f"  [INTEGRATED KEYED MEMORY, {tag} | visit {vis + 1}] too few chains ({len(ch)})", flush=True); return
    rs = np.random.RandomState(int(cfg.seed) + 5858 + vis); ch = [ch[i] for i in rs.choice(len(ch), min(cap, len(ch)), replace=False)]
    def dec(pb, sl, use, n):
        o = b''
        for _ in range(n):
            Xg, _ = _ws_batch(model, [pb + o + b' ']); zg = _ws_run(model, Xg, sl).float()
            if use: zg = kmi_logits(model, K, [pb + o], Xg, zg, sl)
            q = len(pb) + len(o) - 1
            if q >= zg.shape[1]: break
            o += bytes([int(zg[0, q].argmax())])
        return o
    def rank(pb, C, sl, use):
        alts = [x for x in pool if x != C]
        if len(alts) < 9: return None
        cs = [C] + [alts[j] for j in rs.choice(len(alts), 9, replace=False)]; cb = [(c + '.').encode('latin-1', 'replace') for c in cs]
        X, _ = _ws_batch(model, [pb + c for c in cb]); z = _ws_run(model, X, sl).float()
        if use: z = kmi_logits(model, K, [pb] * len(cb), X, z, sl)
        lp = F.log_softmax(z, -1); p0 = len(pb) - 1; sc = []
        for j, c in enumerate(cb):
            n_ = min(len(c), lp.shape[1] - p0); sc.append(float(lp[j, torch.arange(p0, p0 + n_, device=lp.device), X[j, p0 + 1:p0 + 1 + n_]].mean()) if n_ > 0 else -1e9)
        return int(sc[0] > max(sc[1:]))
    res = {True: [[], [], []], False: [[], [], []]}
    with torch.no_grad():
        for e1, e2 in ch:
            sl = int(S_slot(e1['d'])); pb = te(e1).encode('latin-1', 'replace'); B = e1['ans'].encode('latin-1', 'replace')
            for use in (True, False):
                o = dec(pb, sl, use, len(B)); res[use][0].append(int(o == B))
                b2 = o.decode('latin-1').strip() or 'someone'
                r2 = rank(f"It is on record that {b2} {e2['r']} ".encode('latin-1', 'replace'), e2['ans'], int(S_slot(e2['d'])), use)
                r3 = rank(te(e2).encode('latin-1', 'replace'), e2['ans'], int(S_slot(e2['d'])), use)
                if r2 is not None: res[use][1].append(r2)
                if r3 is not None: res[use][2].append(r3)
    m = lambda L: f"{np.mean(L):.2f}" if L else '-'
    print(f"  [INTEGRATED KEYED MEMORY, {tag} | visit {vis + 1}] {len(ch)} chains {'with a hop ONLY the memory holds (an inference awaiting judgement)' if memonly else 'of taught facts'} (A r1 B, B r2 C; never taught as a chain), the model's OWN decoded B "
          f"asked next: WITH the memory first hop exact {m(res[True][0])}, C rank among 10 {m(res[True][1])} (true B given {m(res[True][2])}) | trunk alone "
          f"{m(res[False][0])}, {m(res[False][1])} (true B {m(res[False][2])}) (chance 0.10)", flush=True)


def _kma_ptr_reads(model, K, prompts, sl):
    """the memory's read at the FIRST answer byte of each prompt: (slot indices, read weights) per prompt (None where unaddressed)"""
    from platypus.inference.closed_form import _ws_batch
    out = [None] * len(prompts)
    for i0 in range(0, len(prompts), 64):
        ch = prompts[i0:i0 + 64]; X, _ = _ws_batch(model, [pb + b'x' for pb in ch]); p0s = [len(pb) - 1 for pb in ch]
        a = _kma_addr_rows(model, K, X, sl, p0s, [1] * len(ch))
        if a is None: continue
        nk, wk, off, rr, pp = a; _, idx = _kmi_read(K, nk, wk, off); w = K['_w']
        for k, (r, j) in enumerate(zip(rr, pp)):
            if j == 0: out[i0 + r] = (idx[k].tolist(), w[k].tolist())
    return out


def _kma_tcode(model, K, names):
    """the trunk's own code of each name (last block, mean over the name's bytes in 'About <name>.'), unit length - the LEARNED key of route C"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    C = K.setdefault('tcode', {}); new = [n for n in dict.fromkeys(names) if n not in C]
    if new:
        cap = {}; h9 = model.blocks[-1].register_forward_hook(lambda m_, i_, o_: cap.__setitem__('h', o_.detach()))
        try:
            for i0 in range(0, len(new), 64):
                ch = new[i0:i0 + 64]; X, _ = _ws_batch(model, [f"About {n}.".encode('latin-1', 'replace') for n in ch])
                with torch.no_grad(): _ws_run(model, X, 0)
                for j, n in enumerate(ch): C[n] = cap['h'][j, 5:5 + len(n.encode('latin-1', 'replace'))].float().mean(0)
        finally: h9.remove()
    return torch.stack([C[n] for n in names])


def _kma_twohop_key(model, cfg, K, S_slot, fs, pool, vis, cap=60):
    """16 Nov READOUT, A MEMORY THAT RETURNS KEYS (route C, tools/key_return.py; inf_kmc_keyread): every written entry's slot address at its
    answer's first byte records the ANSWER NAME (a pointer table beside the slots' byte values); a read returns a DISTRIBUTION over answer names
    (route A); hop 2 asks about each of the top KR names with its weight - and, for a name that is not a written subject, about the nearest
    written subject by the trunk's own name code (route C's learned fallback: aliases). Chains A r1 B, B r2 C of entries in the memory (facts
    and inferences), never taught as a chain: first hop (the pointer's best name = B), C's rank among 10 by the composed score, and C top-1 among
    every answer name - vs the model decoding B and asking next (the TWO-HOP readout above)."""
    from platypus.inference.premises import is_rti
    te = lambda A, r: f"It is on record that {A} {r} ".encode('latin-1', 'replace')
    by_d = {}
    for e in fs: by_d.setdefault(int(S_slot(e['d'])), []).append(e)
    PTR = {}
    for sl, L in by_d.items():                                                 # the pointer table: slot -> {answer name: read weight}
        rd = _kma_ptr_reads(model, K, [te(e['A'], e['r']) for e in L], sl)
        for e, x in zip(L, rd):
            if x is None: continue
            for s, w in zip(*x): d_ = PTR.setdefault(s, {}); d_[e['ans']] = d_.get(e['ans'], 0.0) + w
    subj = sorted({e['A'] for e in fs})
    if not PTR or len(subj) < 5:
        print(f"  [KEY-RETURNING MEMORY, TWO-HOP | visit {vis + 1}] too few entries", flush=True); return
    SC = F.normalize(_kma_tcode(model, K, subj), dim=-1); SI = {n: i for i, n in enumerate(subj)}
    def dist(prompts, sl):
        res = []
        for x in _kma_ptr_reads(model, K, prompts, sl):
            d_ = {}
            if x is not None:
                for s, w in zip(*x):
                    for n, v in PTR.get(s, {}).items(): d_[n] = d_.get(n, 0.0) + w * v
            z = sum(d_.values()); res.append({n: v / z for n, v in d_.items()} if z > 0 else {})
        return res
    byA = {}
    for e in fs: byA.setdefault(e['A'], []).append(e)
    ch = [(e1, e2) for e1 in fs for e2 in byA.get(e1['ans'], ()) if e2['ans'] not in (e1['A'], e1['ans'])]
    if len(ch) < 5:
        print(f"  [KEY-RETURNING MEMORY, TWO-HOP | visit {vis + 1}] too few chains ({len(ch)})", flush=True); return
    rs = np.random.RandomState(int(cfg.seed) + 6161 + vis); ch = [ch[i] for i in rs.choice(len(ch), min(cap, len(ch)), replace=False)]
    KR = int(getattr(cfg, 'inf_kmc_keyread_k', 5)); res = {m: [[], [], []] for m in ('A', 'C')}; nfb = 0; ninf = 0
    with torch.no_grad():
        for e1, e2 in ch:
            ninf += int(is_rti(e1) or is_rti(e2)); d1 = dist([te(e1['A'], e1['r'])], int(S_slot(e1['d'])))[0]
            Bs = sorted(d1, key=d1.get, reverse=True)[:KR]; alts = [x for x in pool if x != e2['ans']]
            if len(alts) < 9: continue
            cs = [e2['ans']] + [alts[j] for j in rs.choice(len(alts), 9, replace=False)]
            for m in ('A', 'C'):
                qs = []
                for b in Bs:
                    q = b
                    if m == 'C' and b not in SI:                               # route C: not a written subject -> the nearest written subject by the trunk's code
                        v = F.normalize(_kma_tcode(model, K, [b]), dim=-1)[0]; q = subj[int((SC @ v).argmax())]; nfb += 1
                    qs.append(q)
                d2s = dist([te(q, e2['r']) for q in qs], int(S_slot(e2['d']))) if qs else []
                sc = {}
                for b, d2 in zip(Bs, d2s):
                    for n, v in d2.items(): sc[n] = sc.get(n, 0.0) + d1[b] * v
                res[m][0].append(int(bool(Bs) and Bs[0] == e1['ans']))
                s10 = [sc.get(c, 0.0) for c in cs]; res[m][1].append(int(s10[0] > max(s10[1:])))
                res[m][2].append(int(bool(sc) and max(sc, key=sc.get) == e2['ans']))
    mm = lambda L: f"{np.mean(L):.2f}" if L else '-'
    print(f"  [KEY-RETURNING MEMORY, TWO-HOP | visit {vis + 1}] {len(ch)} chains of entries in the memory ({ninf} with an inference hop; never taught as a chain), "
          f"pointer table over {len(PTR)} slots, top {KR} names carried to hop 2 || ROUTE A (pointers): first hop {mm(res['A'][0])}, C rank among 10 {mm(res['A'][1])}, "
          f"C top-1 among all answers {mm(res['A'][2])} | ROUTE C (pointers + the trunk's name code for names that are not written subjects; {nfb} fallbacks): "
          f"{mm(res['C'][0])}, {mm(res['C'][1])}, {mm(res['C'][2])} (compare TWO-HOP: the model decoding B and asking next; chance 0.10)", flush=True)


def kma_compose_write(model, cfg, K, S_slot, fs, vis):
    """17 Nov THE COMPOSING BUILD (inf_kmc_compose; ONE model change): the key-returning memory WRITES two-hop conclusions. For entries
    A r1 B in the memory (facts and kept inferences), the read for (A, r1) returns a distribution over answer names (pointer table, route A);
    each top name B - or, if B is not a written subject, the nearest written subject by the trunk's own name code (route C) - is asked with
    each wording r2 the memory holds about it; the best answer C gives the conclusion 'A r2 C' (typed by the second hop's relation; score =
    p(B) x p(C)). Pairs already written, or stated in the memory either way, are skipped; the top inf_kmc_compose_n are written to the
    inference ledger like the fact-table code's writes - judged by later reading, kept by the keep rule, taught (consolidated) only if kept."""
    import platypus.inference.premises as pm
    from platypus.inference.premises import is_rti
    W = pm._PM.setdefault('rti', {}); LAB = pm._PM.get('LAB') or set()
    te = lambda A, r: f"It is on record that {A} {r} ".encode('latin-1', 'replace')
    rmode = str(getattr(cfg, 'inf_kmc_compose_rule', 'any'))                 # 17 Nov (kcp on the GPU: copying r2 wrote 'A r2 C' for any r1, r2 - wrong in
    par9 = {}                                                                  #   general): 'rules' = only relation pairs the relational module verified
    for a9, b9 in (model.__dict__.get('_rm_merged') or ()): par9[str(a9).strip()] = str(b9).strip()   #   (r1 o r2 -> r3), 'same' = r o r -> r,
    def canon(r):                                                              #   'same+rules' = both; paraphrases merged by the module count as one
        r = str(r).strip(); seen9 = set()
        while r in par9 and r not in seen9: seen9.add(r); r = par9[r]
        return r
    RM = {}
    if 'rules' in rmode:
        for r1_, r2_, r3_ in list(model.__dict__.get('_rm_rules') or []) + list(model.__dict__.get('_rm_rules_real') or []): RM[(canon(r1_), canon(r2_))] = r3_
    def rule3(r1, r2):
        if rmode == 'any': return r2
        c1, c2 = canon(r1), canon(r2)
        if (c1, c2) in RM: return RM[(c1, c2)]
        return str(r1).strip() if ('same' in rmode and c1 == c2) else None
    try:                                                                       # RULE-WORLD TRUTH for composed writes about its names (the distant labels
        from platypus.inference.probes import inf_rule_world, _RW_REL           #   cannot judge them): grand = parent o parent, skip = mgr o mgr
        RW9 = inf_rule_world(cfg); ph9 = {k: v.split('{a} ')[1].split(' {b}')[0] for k, v in _RW_REL.items()}
        tmap9 = {ph9['grand']: RW9.get('parent', {}), ph9['skip']: RW9.get('mgr', {})}; rwn9 = set(RW9.get('parent', {})) | set(RW9.get('mgr', {})) | set(RW9.get('parent', {}).values()) | set(RW9.get('mgr', {}).values())
    except Exception:
        tmap9 = {}; rwn9 = set()
    def rwtruth(A, r3, C):                                                     # True / False for rule-world names, None otherwise
        if A not in rwn9: return None
        m = tmap9.get(canon(r3)) or tmap9.get(str(r3).strip())
        if m is None: return False
        return m.get(m.get(A)) == C
    by_d = {}
    for e in fs: by_d.setdefault(int(S_slot(e['d'])), []).append(e)
    PTR = {}
    for sl, L in by_d.items():
        for e, x in zip(L, _kma_ptr_reads(model, K, [te(e['A'], e['r']) for e in L], sl)):
            if x is None: continue
            for s_, w_ in zip(*x): d_ = PTR.setdefault(s_, {}); d_[e['ans']] = d_.get(e['ans'], 0.0) + w_
    subj = {}
    for e in fs: subj.setdefault(e['A'], []).append(e)
    sl_of = {e['A']: int(S_slot(e['d'])) for e in fs}
    if not PTR or len(subj) < 5:
        print(f"  [COMPOSED WRITES | visit {vis + 1}] too few entries", flush=True); return
    SN = sorted(subj); SC = F.normalize(_kma_tcode(model, K, SN), dim=-1)
    def dist(prompts, sl):
        res = []
        for x in _kma_ptr_reads(model, K, prompts, sl):
            d_ = {}
            if x is not None:
                for s_, w_ in zip(*x):
                    for n, v in PTR.get(s_, {}).items(): d_[n] = d_.get(n, 0.0) + w_ * v
            z = sum(d_.values()); res.append({n: v / z for n, v in d_.items()} if z > 0 else {})
        return res
    stated = {frozenset((e['A'], e['ans'])) for e in fs}; done = {frozenset(k) for k in W}
    rs = np.random.RandomState(int(cfg.seed) + 7171 + vis); E1 = list(fs)
    if len(E1) > int(getattr(cfg, 'inf_kmc_compose_src', 3000)): E1 = [E1[i] for i in rs.choice(len(E1), int(getattr(cfg, 'inf_kmc_compose_src', 3000)), replace=False)]
    h1 = {}                                                                   # hop 1, by slot
    for sl in sorted({int(S_slot(e['d'])) for e in E1}):
        L = [e for e in E1 if int(S_slot(e['d'])) == sl]
        for e, d1 in zip(L, dist([te(e['A'], e['r']) for e in L], sl)): h1[id(e)] = d1
    q2 = {}; nfb = 0                                                          # hop 2 prompts: (B as asked, r2) -> its slot
    for e in E1:
        for b in sorted(h1[id(e)], key=h1[id(e)].get, reverse=True)[:3]:
            q = b
            if q not in subj:
                if rmode != 'any': continue                                    # rule-based composing: exact chains only (no stand-in subject)
                v = F.normalize(_kma_tcode(model, K, [b]), dim=-1)[0]; q = SN[int((SC @ v).argmax())]; nfb += 1
            for r2 in list(dict.fromkeys(x['r'] for x in subj[q]))[:4]: q2[(q, r2)] = sl_of[q]
    a2 = {}
    for sl in sorted(set(q2.values())):
        ks = [k for k, v in q2.items() if v == sl]
        for k, d2 in zip(ks, dist([te(*k) for k in ks], sl)): a2[k] = d2
    cand = {}; nrej = 0
    for e in E1:
        d1 = h1[id(e)]
        for b in sorted(d1, key=d1.get, reverse=True)[:3]:
            if b not in subj and rmode != 'any': continue
            q = b if b in subj else SN[int((SC @ F.normalize(_kma_tcode(model, K, [b]), dim=-1)[0]).argmax())]
            for r2 in list(dict.fromkeys(x['r'] for x in subj[q]))[:4]:
                d2 = a2.get((q, r2)) or {}
                if not d2: continue
                r3 = rule3(e['r'], r2)
                if r3 is None: nrej += 1; continue
                c = max(d2, key=d2.get); sc = d1[b] * d2[c]; A = e['A']
                if c in (A, b, q) or frozenset((A, c)) in stated or frozenset((A, c)) in done: continue
                if sc > cand.get((A, c), (0,))[0]: cand[(A, c)] = (sc, r3, b, e)
    top = sorted(cand.items(), key=lambda x: -x[1][0])[:int(getattr(cfg, 'inf_kmc_compose_n', 100))]; ndv = int(pm._PM.get('ndv_now', 0)); ys = []
    yr = []
    for i, ((A, c), (sc, r2, b, e)) in enumerate(top):
        k = tuple(sorted((A, c))); t9 = rwtruth(A, r2, c); y = bool(t9) if t9 is not None else frozenset((A, c)) in LAB; ys.append(y)
        if t9 is not None: yr.append(bool(t9))
        W[k] = dict(vis=vis, d=int(e['d']), truth=bool(y), bridges=1, rank=i, novel=True, code=float(sc), ndv=ndv, prem=[b], rule='compose', score=float(sc),
                    state='inferred', th=A, tw=r2, tt=c, comp=True)
    print(f"  [COMPOSED WRITES | visit {vis + 1}] the memory's key reads over {len(E1)} entries ({sum(1 for x in E1 if is_rti(x))} inferences) -> {len(cand)} two-hop "
          f"conclusions never stated ({nfb} hop-2 names by the trunk's code; relation pairs: {rmode}, {len(RM)} verified rules, {nrej} chains with no rule skipped); "
          f"WROTE the top {len(top)}: right {np.mean(ys) if ys else float('nan'):.3f} (rule-world names by the rule world's truth: {np.mean(yr) if yr else float('nan'):.3f} of "
          f"{len(yr)}; others by distant labels, which undercount) - judged, kept and taught like any inference || "
          + " | ".join(f"{A} {r2} {c} via {b} ({'right' if (rwtruth(A, r2, c) if rwtruth(A, r2, c) is not None else frozenset((A, c)) in LAB) else 'wrong'})"
          for (A, c), (sc, r2, b, e) in top[:5]), flush=True)


def kma_probe_twohop(model, cfg, K, S_slot, fs, vis):
    """17 Nov READOUT (inf_kmc_probe2; a PREDICTION-TIME option, nothing trained): the standard TWO-HOP PROBE QUESTIONS ('So A is connected
    to' C, the same chains as the inference eval, C among 20 probe names, chance 0.05) answered by the key-returning memory: hop 1 reads
    (A, r1) -> a distribution over answer names (pointer table); hop 2 reads (B, r2) for each of the top 5 names B; the candidates are ranked
    by sum over B of p(B) p(C | B). Chains with both facts in visited domains; coverage = both facts are entries in the memory."""
    from platypus.inference.probes import inf_probes
    te = lambda A, r: f"It is on record that {A} {r} ".encode('latin-1', 'replace')
    by_d = {}
    for e in fs: by_d.setdefault(int(S_slot(e['d'])), []).append(e)
    PTR = {}
    for sl, L in by_d.items():
        for e, x in zip(L, _kma_ptr_reads(model, K, [te(e['A'], e['r']) for e in L], sl)):
            if x is None: continue
            for s_, w_ in zip(*x): d_ = PTR.setdefault(s_, {}); d_[e['ans']] = d_.get(e['ans'], 0.0) + w_
    def dist(prompts, sl):
        res = []
        for x in _kma_ptr_reads(model, K, prompts, sl):
            d_ = {}
            if x is not None:
                for s_, w_ in zip(*x):
                    for n, v in PTR.get(s_, {}).items(): d_[n] = d_.get(n, 0.0) + w_ * v
            z = sum(d_.values()); res.append({n: v / z for n, v in d_.items()} if z > 0 else {})
        return res
    seen = set(int(k) for k in (model.__dict__.get('_t2s') or {}).keys())
    ch = [c for c in inf_probes(cfg) if len(c['names']) >= 3 and all(int(d) in seen for d in c['doms'][:2])]
    if not ch or not PTR:
        print(f"  [TWO-HOP PROBES THROUGH THE MEMORY | visit {vis + 1}] nothing to ask ({len(ch)} chains, {len(PTR)} pointer slots)", flush=True); return
    alln = [n for c in inf_probes(cfg) for n in c['names']]; have = {(e['A'], str(e['r']).strip(), e['ans']) for e in fs}
    rs = np.random.RandomState(77); KR = 5; h1 = {}
    for sl in sorted({int(S_slot(c['doms'][0])) for c in ch}):
        L = [c for c in ch if int(S_slot(c['doms'][0])) == sl]
        for c, d1 in zip(L, dist([te(c['names'][0], c['rels'][0]) for c in L], sl)): h1[id(c)] = d1
    q2 = {}
    for c in ch:
        for b in sorted(h1[id(c)], key=h1[id(c)].get, reverse=True)[:KR]: q2[(b, c['rels'][1], int(S_slot(c['doms'][1])))] = None
    for sl in sorted({k[2] for k in q2}):
        ks = [k for k in q2 if k[2] == sl]
        for k, d2 in zip(ks, dist([te(k[0], k[1]) for k in ks], sl)): q2[k] = d2
    ok, ok1, cov, okc = [], [], [], []
    for c in ch:
        A, B, C = c['names'][:3]; d1 = h1[id(c)]
        oth = [x for x in rs.choice(alln, min(len(alln), 60), replace=False) if x not in (A, B, C)][:19]; cs = [C] + oth
        sc = np.zeros(len(cs))
        for b in sorted(d1, key=d1.get, reverse=True)[:KR]:
            d2 = q2.get((b, c['rels'][1], int(S_slot(c['doms'][1])))) or {}
            sc += d1[b] * np.array([d2.get(x, 0.0) for x in cs])
        r_ = bool(sc.max() > 0 and int(np.argmax(sc)) == 0); ok.append(r_)
        ok1.append(bool(d1) and max(d1, key=d1.get) == B)
        cv = (A, str(c['rels'][0]).strip(), B) in have and (B, str(c['rels'][1]).strip(), C) in have; cov.append(cv)
        if cv: okc.append(r_)
    print(f"  [TWO-HOP PROBES THROUGH THE MEMORY | visit {vis + 1}] PREDICTION-TIME (nothing trained): {len(ch)} two-hop probe chains in visited domains, "
          f"C among 20 probe names (chance 0.05) || C ranked first {np.mean(ok):.2f} | first hop right (B the top name) {np.mean(ok1):.2f} | both facts in the memory "
          f"{np.mean(cov):.2f}; C first on those {np.mean(okc) if okc else float('nan'):.2f} (n {len(okc)}) | compare 'inference eval' two-hop FROM THE WEIGHTS", flush=True)


def kma_rule_compose(model, cfg, K, S_slot, fs, vis):
    """18 Nov READOUT, ROUTE A (inf_kmc_rulecomp; PREDICTION-TIME, nothing trained): COMPOSITION INSIDE THE MODEL through the key-returning
    memory - a question with a composite relation ('A is a grandchild of') is answered by TWO READS of the memory: the relational module's
    verified rule r3 = r1 o r2 gives the two relations; read 1 (A, r1) returns a distribution over names B; read 2 (B, r2) for the top 5 B
    returns C; C is ranked among the rule world's names. On the rule world's composites NEVER planted - and separately on the PURE families
    (inf_rule_world_pure: no composite planted in the family) where the trunk's own forward pass got 0 of 81 on the GPU (cg27a_kcr3). With no
    learned rule yet, the true rule (ORACLE) is used and labelled as such."""
    from platypus.inference.probes import inf_rule_world, _RW_REL
    W = inf_rule_world(cfg)
    if not W.get('held'): return
    ph = {k: v.split('{a} ')[1].split(' {b}')[0] for k, v in _RW_REL.items()}
    rules = {str(t[2]).strip(): (str(t[0]).strip(), str(t[1]).strip()) for t in (model.__dict__.get('_rm_rules') or [])}
    src = 'LEARNED rules' if rules else 'ORACLE rules (none learned yet)'
    if not rules: rules = {ph['grand']: (ph['parent'], ph['parent']), ph['skip']: (ph['mgr'], ph['mgr'])}
    te = lambda A, r: f"It is on record that {A} {r} ".encode('latin-1', 'replace')
    by_d = {}; seen = set(int(k) for k in (model.__dict__.get('_t2s') or {}).keys()); fs = [e for e in fs if 'chain' not in e]; fs0 = list(fs)   # 19 Nov: chain-form written composites are not recorded facts
    if int(getattr(cfg, 'inf_kmc_rulecomp', 0)) >= 2:                         # mode 2: the pointer table also covers every rule-world PREMISE the model has READ
        fs += [{'d': d_, 'A': a_, 'r': ph[k_], 'ans': b_} for (d_, s_, k_, a_, b_) in W['planted'] if int(d_) in seen and k_ in ('parent', 'mgr')]   # (the memory is written on every byte of read text; only the decoding table is extended)
    for e in fs: by_d.setdefault(int(S_slot(e['d'])), []).append(e)
    PTR = {}
    for sl, L in by_d.items():
        for e, x in zip(L, _kma_ptr_reads(model, K, [te(e['A'], e['r']) for e in L], sl)):
            if x is None: continue
            for s_, w_ in zip(*x): d_ = PTR.setdefault(s_, {}); d_[e['ans']] = d_.get(e['ans'], 0.0) + w_
    def dist(prompts, sl, P_=None):
        P_ = PTR if P_ is None else P_; res = []
        for x in _kma_ptr_reads(model, K, prompts, sl):
            d_ = {}
            if x is not None:
                for s_, w_ in zip(*x):
                    for n, v in P_.get(s_, {}).items(): d_[n] = d_.get(n, 0.0) + w_ * v
            z = sum(d_.values()); res.append({n: v / z for n, v in d_.items()} if z > 0 else {})
        return res
    pure = W.get('pure') or set()
    have = {(e['A'], str(e['r']).strip(), e['ans']) for e in fs}; par = {**W.get('parent', {}), **W.get('mgr', {})}
    tg = [(d, l, r3, C) for (d, sent, k, l, C) in W['held'] if int(d) in seen and ph.get(k) in rules for r3 in [ph.get(k)]]
    if not tg or not PTR:
        print(f"  [ROUTE A: COMPOSITION THROUGH THE MEMORY | visit {vis + 1}] nothing to ask ({len(tg)} composites, {len(PTR)} pointer slots)", flush=True); return
    names = sorted(set(par) | set(par.values())); KR = int(getattr(cfg, 'inf_kmc_rulecomp_k', 5)); out = {'all': [], 'pure': [], 'cov': [], 'covpure': []}; first = []; dg = {'hop2': [], 'top1': [], 'conf': []}
    for sl in sorted({int(S_slot(d)) for d, _, _, _ in tg}):
        L = [t for t in tg if int(S_slot(t[0])) == sl]
        d1s = dist([te(l, rules[r3][0]) for d, l, r3, C in L], sl); q2 = {}
        for (d, l, r3, C), d1 in zip(L, d1s):
            for b in sorted(d1, key=d1.get, reverse=True)[:KR]: q2[(b, rules[r3][1])] = None
        ks = list(q2)
        for k, d2 in zip(ks, dist([te(b, r) for b, r in ks], sl)): q2[k] = d2
        tb = list({(par.get(l), rules[r3][1]) for d, l, r3, C in L if par.get(l)})       # DIAGNOSTIC: the second hop alone, from the TRUE B
        tbd = dict(zip(tb, dist([te(b_, r_) for b_, r_ in tb], sl)))
        for (d, l, r3, C), d1 in zip(L, d1s):
            d2t = tbd.get((par.get(l), rules[r3][1])) or {}; h2 = bool(d2t) and max(d2t, key=d2t.get) == C; dg['hop2'].append(h2)
            b1 = max(d1, key=d1.get) if d1 else None; d21 = q2.get((b1, rules[r3][1])) or {}
            dg['top1'].append(bool(d21) and max(d21, key=d21.get) == C); dg.setdefault(('dom', int(d)), []).append(h2)
            sc = {}
            for b in sorted(d1, key=d1.get, reverse=True)[:KR]:
                for n, v in (q2.get((b, rules[r3][1])) or {}).items(): sc[n] = sc.get(n, 0.0) + d1[b] * v
            ok = bool(sc) and max(sc, key=sc.get) == C; dg['conf'].append((max(sc.values()) if sc else 0.0, ok)); first.append(bool(d1) and max(d1, key=d1.get) == par.get(l))
            cov = (l, rules[r3][0], par.get(l)) in have and (par.get(l), rules[r3][1], C) in have
            out['all'].append(ok); dg.setdefault(('f1', first[-1]), []).append(ok); dg.setdefault(('d', int(d)), []).append(ok)
            if l in pure: out['pure'].append(ok)
            if cov: out['cov'].append(ok)
            if cov and l in pure: out['covpure'].append(ok)
    m = lambda v: f"{np.mean(v):.2f} (n {len(v)})" if v else "n/a"
    print(f"  [ROUTE A: COMPOSITION THROUGH THE MEMORY | visit {vis + 1}] PREDICTION-TIME, {src}, pointer table over {'every premise read' if int(getattr(cfg, 'inf_kmc_rulecomp', 0)) >= 2 else 'pool facts'}: composites NEVER planted, asked with the composite "
          f"relation, answered by two memory reads (C first among {len(names)} rule-world names, chance {1 / max(1, len(names)):.3f}) || ALL {m(out['all'])} | "
          f"PURE FAMILIES {m(out['pure'])} | first hop right {np.mean(first) if first else float('nan'):.2f} || both premises in the memory: ALL {m(out['cov'])}, "
          f"PURE {m(out['covpure'])} | compare the trunk alone on pure-family held-back composites (RELATIONAL MODULE WRITES, PURE FAMILIES)", flush=True)
    if int(getattr(cfg, 'inf_kmc_rulecomp_write', 0)) and not int(getattr(cfg, 'inf_kmc_rulecomp_dwrite', 0)) and not int(getattr(cfg, 'inf_kmc_rulej_write', 0)) and (src.startswith('LEARNED') or int(getattr(cfg, 'inf_kmc_rulecomp_oracle_write', 0))):
        # 18 Nov ROUTE A WRITES (MODEL CHANGE, inf_kmc_rulecomp_write): the model's own composition through two memory reads becomes what the relational
        # module writes into the question pool (instead of its symbolic chaining over stored facts). LABEL-FREE: the subjects are those with a recorded
        # r1 fact and no recorded r3 fact; the pointer table maps slots to names through the POOL'S facts only (what the reader recorded).
        PW = {}; bw = {}
        for e in fs0: bw.setdefault(int(S_slot(e['d'])), []).append(e)
        for sl, L in bw.items():
            for e, x in zip(L, _kma_ptr_reads(model, K, [te(e['A'], e['r']) for e in L], sl)):
                if x is None: continue
                for s_, w_ in zip(*x): d_ = PW.setdefault(s_, {}); d_[e['ans']] = d_.get(e['ans'], 0.0) + w_
        if int(getattr(cfg, 'inf_kmc_rulecomp_norm', 0)):                      # 18 Nov: the SLOT-NORMALISED table (crowded slots count less; local kraw visit 2:
            nW = max(2, len({n for dd in PW.values() for n in dd}))           #   first hop 0.97 vs 0.86 raw, composite 0.95 vs 0.80)
            PW = {s_: {n: v / sum(dd.values()) * np.log(1.0 + nW / len(dd)) for n, v in dd.items()} for s_, dd in PW.items()}
        hasr = {(e['A'], str(e['r']).strip()) for e in fs0}; subj = {}
        for e in fs0:
            for r3, (r1, r2) in rules.items():
                if str(e['r']).strip() == r1 and (e['A'], r3) not in hasr: subj[(e['A'], r3)] = int(S_slot(e['d']))
        thr = float(getattr(cfg, 'inf_kmc_rulecomp_wthr', 0.5)); RW = model.__dict__.setdefault('_ra_written', {}); nw = 0; ok_ = []; ngt = 0; WS = K.get('subj', set()); GV = []
        gA9, gB9, gd9 = int(getattr(cfg, 'inf_kmc_rulecomp_gA', 1)), int(getattr(cfg, 'inf_kmc_rulecomp_gB', 1)), float(getattr(cfg, 'inf_kmc_rulecomp_gd1', 0.5))
        gate9 = lambda A_, B_, p_: (A_ in WS or not gA9) and (B_ in WS or not gB9) and p_ >= gd9   # the write gate (parts switchable)
        for sl in sorted(set(subj.values())):
            Q = [(A, r3) for (A, r3), s2 in subj.items() if s2 == sl]
            d1s = dist([te(A, rules[r3][0]) for A, r3 in Q], sl, PW)
            for (A, r3), d1 in zip(Q, d1s):
                if not d1: continue
                sc = {}
                for b in sorted(d1, key=d1.get, reverse=True)[:KR]:
                    for n, v in (dist([te(b, rules[r3][1])], sl, PW)[0]).items(): sc[n] = sc.get(n, 0.0) + d1[b] * v
                if not sc: continue
                C = max(sc, key=sc.get); B1 = max(d1, key=d1.get)
                try:                                                           # 18 Nov READOUT, GATE VARIANTS (nothing changed by it): every candidate
                    m9 = {ph['grand']: W.get('parent', {}), ph['skip']: W.get('mgr', {})}.get(r3)   #   with its truth and which gates it passes
                    s2 = sorted(sc.values(), reverse=True); mg = s2[0] - (s2[1] if len(s2) > 1 else 0.0)
                    GV.append((C == m9.get(m9.get(A)) if m9 is not None and A in m9 else None, sc[C], mg, A in WS, B1 in WS, d1[B1], C in WS, C == A or C == B1))
                except Exception: pass
                if sc[C] < thr or C == A or C == B1: continue
                if int(getattr(cfg, 'inf_kmc_rulecomp_wgate', 0)) and not gate9(A, B1, d1[B1]): ngt += 1; continue
                if (r3, A, C) not in RW: RW[(r3, A, C)] = vis; nw += 1
                model.__dict__.setdefault('_ra_bridge', {})[(r3, A, C)] = B1   # 19 Nov: the bridge, for chain-form teaching
                m_ = {ph['grand']: W.get('parent', {}), ph['skip']: W.get('mgr', {})}.get(r3)   # FIX (18 Nov, labelled): a subject with NO true composite (a middle
                if m_ is not None and A in m_: ok_.append(C == m_.get(m_.get(A)))               #   name whose parent is a root) counts as WRONG, as in the relational module's line
        if int(getattr(cfg, 'inf_kmc_rulecomp_reverify', 0)) and RW:           # 18 Nov MODEL CHANGE (combined with the gate, by the user's decision): RE-VERIFY every
            WRm = model.__dict__.get('_rm_written', {}); rt_ = []; kp_ = []        #   earlier route A write at each sleep with today's memory - retracted (no longer taught)
            subj_of = {e['A']: int(S_slot(e['d'])) for e in fs0}                   #   unless the gated two reads still give the same answer
            for (r3, A, C) in list(RW):
                sl = subj_of.get(A)
                if sl is None or r3 not in rules: continue
                d1 = dist([te(A, rules[r3][0])], sl, PW)[0]; ok9 = False
                if d1:
                    B1 = max(d1, key=d1.get)
                    if gate9(A, B1, d1[B1]) and B1 != C:
                        d2 = dist([te(B1, rules[r3][1])], sl, PW)[0]; ok9 = bool(d2) and max(d2, key=d2.get) == C and d1[B1] * d2[C] >= thr
                m9 = {ph['grand']: W.get('parent', {}), ph['skip']: W.get('mgr', {})}.get(r3); tr9 = (C == m9.get(m9.get(A))) if m9 is not None and A in m9 else None
                if ok9: kp_.append(tr9); continue
                RW.pop((r3, A, C), None); WRm.pop((r3, A, C), None); rt_.append(tr9)
            pr = lambda L: f"{np.mean([x for x in L if x is not None]) if any(x is not None for x in L) else float('nan'):.2f}"
            print(f"  [ROUTE A RE-VERIFY | visit {vis + 1}] MODEL CHANGE: earlier route A writes checked with today's gated two reads: KEPT {len(kp_)} (right {pr(kp_)}) | "
                  f"RETRACTED {len(rt_)} (right {pr(rt_)}; no longer taught) | {len(RW)} written composites remain", flush=True)
        try:
            V = [g for g in GV if g[0] is not None and not g[7]]
            def pv(nm, f):
                L = [g[0] for g in V if f(g)]; return f"{nm} {np.mean(L) if L else float('nan'):.2f} ({len(L)})"
            print(f"  [ROUTE A, GATE VARIANTS | visit {vis + 1}] READOUT: every route A candidate this sleep (truth known, n {len(V)}), precision (n passing) under each gate || "
                  + " | ".join([pv('no gate', lambda g: True), pv('score >= 0.5', lambda g: g[1] >= 0.5), pv('score >= 0.8', lambda g: g[1] >= 0.8),
                                pv('subject written', lambda g: g[3]), pv('first-read name written', lambda g: g[4]), pv('both written', lambda g: g[3] and g[4]),
                                pv('both + first read >= 0.5 + score >= 0.5 (kra3)', lambda g: g[3] and g[4] and g[5] >= 0.5 and g[1] >= 0.5),
                                pv('kra3 + margin >= 0.3', lambda g: g[3] and g[4] and g[5] >= 0.5 and g[1] >= 0.5 and g[2] >= 0.3),
                                pv('answer NOT a written subject', lambda g: not g[6]), pv('kra3 + answer not a written subject', lambda g: g[3] and g[4] and g[5] >= 0.5 and g[1] >= 0.5 and not g[6])]), flush=True)
        except Exception as e9: print(f"  [ROUTE A, GATE VARIANTS | visit {vis + 1}] unavailable ({type(e9).__name__}: {str(e9)[:100]})", flush=True)
        print(f"  [ROUTE A WRITES | visit {vis + 1}] MODEL CHANGE, {src}, {'slot-normalised' if int(getattr(cfg, 'inf_kmc_rulecomp_norm', 0)) else 'raw'} pool table: {len(subj)} subjects with an r1 fact and no r3 fact; {nw} new composites written for the relational module's "
              f"write path (composite score >= {thr}; {len(RW)} so far; {ngt} refused by the gate: both subjects written in the memory and the first read's top name >= 0.5) | precision against the rule world's truth {np.mean(ok_) if ok_ else float('nan'):.2f} (n {len(ok_)})", flush=True)
    PN = {s_: {n: v / sum(dd.values()) * np.log(1.0 + len(names) / len(dd)) for n, v in dd.items()} for s_, dd in PTR.items()}   # DIAGNOSTIC: each slot's
    fn, cn = [], []                                                            #   names normalised and weighted by how few names share it (crowded slots count less)
    for sl in sorted({int(S_slot(d)) for d, _, _, _ in tg}):
        L = [t for t in tg if int(S_slot(t[0])) == sl]; d1s = dist([te(l, rules[r3][0]) for d, l, r3, C in L], sl, PN)
        bq = list({(max(d1, key=d1.get), rules[r3][1]) for (d, l, r3, C), d1 in zip(L, d1s) if d1}); bd = dict(zip(bq, dist([te(b_, r_) for b_, r_ in bq], sl, PN)))
        for (d, l, r3, C), d1 in zip(L, d1s):
            b1 = max(d1, key=d1.get) if d1 else None; fn.append(b1 is not None and b1 == par.get(l)); d2 = bd.get((b1, rules[r3][1])) or {}
            cn.append(bool(d2) and max(d2, key=d2.get) == C)
    sz = [len(dd) for dd in PTR.values()]
    print(f"  [ROUTE A, CROWDED SLOTS | visit {vis + 1}] names per pointer slot: median {np.median(sz):.0f}, 90th percentile {np.percentile(sz, 90):.0f}, max {max(sz)} | "
          f"SLOT-NORMALISED table: first hop {np.mean(fn):.2f} (raw {np.mean(first):.2f}) | composite, top B only {np.mean(cn):.2f} (raw {np.mean(dg['top1']):.2f})", flush=True)
    if int(getattr(cfg, 'inf_kmc_rulecomp_clarify', 0)):                      # 18 Nov READOUT, WHAT ROUTE A USES (the correction, tested): if route A is an address lookup over
        try:                                                                   #   recorded facts, (1) a table whose answers are SHUFFLED among the facts collapses it, (2) a plain
            rs8 = np.random.RandomState(7); ans8 = [e['ans'] for e in fs]; perm = rs8.permutation(len(ans8))   # dictionary over the same facts (no model at all) matches it
            PS = {}
            idx8 = {id(e): k for k, e in enumerate(fs)}
            for sl, L in by_d.items():
                for e, x in zip(L, _kma_ptr_reads(model, K, [te(e['A'], e['r']) for e in L], sl)):
                    if x is None: continue
                    a8 = ans8[perm[idx8[id(e)]]]
                    for s_, w_ in zip(*x): d_ = PS.setdefault(s_, {}); d_[a8] = d_.get(a8, 0.0) + w_
            DCT = {(e['A'], str(e['r']).strip()): e['ans'] for e in fs}; sh, dc = [], []
            for sl in sorted({int(S_slot(d)) for d, _, _, _ in tg}):
                L = [t for t in tg if int(S_slot(t[0])) == sl]; d1s = dist([te(l, rules[r3][0]) for d, l, r3, C in L], sl, PS)
                for (d, l, r3, C), d1 in zip(L, d1s):
                    b1 = max(d1, key=d1.get) if d1 else None; d2 = dist([te(b1, rules[r3][1])], sl, PS)[0] if b1 else {}
                    sh.append(bool(d2) and max(d2, key=d2.get) == C)
                    dc.append(DCT.get((DCT.get((l, rules[r3][0])), rules[r3][1])) == C)
            print(f"  [ROUTE A, WHAT IT USES | visit {vis + 1}] the same composites (n {len(tg)}): route A, top B only {np.mean(dg['top1']):.2f} | the SAME reads with the table's answers SHUFFLED among "
                  f"the facts {np.mean(sh):.2f} | a PLAIN DICTIONARY over the same recorded facts, no model at all {np.mean(dc):.2f} || route A never reads the memory's learned values: "
                  f"its answers come from the table", flush=True)
        except Exception as e9: print(f"  [ROUTE A, WHAT IT USES | visit {vis + 1}] unavailable ({type(e9).__name__}: {str(e9)[:100]})", flush=True)
    if int(getattr(cfg, 'inf_kmc_rulecomp_decode', 0)):                       # 18 Nov READOUT, ROUTE D (the honest in-model test): the TRUNK ITSELF answers the composite
        try:                                                                   #   question in two DECODING steps via the learned rule - it decodes B for 'A r1', then C for
            from platypus.inference.distill import _dw_greedy                 #   'B r2' (store and memory off; no pointer table, no recorded facts used at question time)
            rs9 = np.random.RandomState(vis); T9 = [tg[k] for k in rs9.permutation(len(tg))[:int(getattr(cfg, 'inf_kmc_rulecomp_decode', 0))]]
            o9 = {'b': [], 'c': [], 'direct': [], 'pure': [], 'pdirect': []}
            for (d, l, r3, C) in T9:
                sl = int(S_slot(d)); r1, r2 = rules[r3]
                b_ = _dw_greedy(model, te(l, r1), sl, 20); c_ = _dw_greedy(model, te(b_, r2), sl, 20) if b_ else ''
                dr = _dw_greedy(model, te(l, r3), sl, 20)
                o9['b'].append(b_ == par.get(l)); o9['c'].append(c_ == C); o9['direct'].append(dr == C)
                if l in pure: o9['pure'].append(c_ == C); o9['pdirect'].append(dr == C)
            try:
              from platypus.inference.closed_form import _ws_batch, _ws_run
              def decm(pb, sl, n=20):                                            # ROUTE E: greedy decoding WITH the integrated memory's learned values (kmi_logits)
                  o = b''
                  for _ in range(n):
                      Xg, _ = _ws_batch(model, [pb + o + b' ']); zg = _ws_run(model, Xg, sl).float(); zg = kmi_logits(model, K, [pb + o], Xg, zg, sl)
                      q = len(pb) + len(o) - 1
                      if q >= zg.shape[1]: break
                      c = int(zg[0, q].argmax())
                      if c in (46, 10): break
                      o += bytes([c])
                  return o.decode('latin-1', 'replace').strip()
              oE = {'b': [], 'c': [], 'direct': [], 'pure': []}
              for (d, l, r3, C) in T9[:int(getattr(cfg, 'inf_kmc_rulecomp_decode_e', 30))]:
                  sl = int(S_slot(d)); r1, r2 = rules[r3]; b_ = decm(te(l, r1), sl); c_ = decm(te(b_, r2), sl) if b_ else ''
                  oE['b'].append(b_ == par.get(l)); oE['c'].append(c_ == C); oE['direct'].append(decm(te(l, r3), sl) == C)
                  if l in pure: oE['pure'].append(c_ == C)
              print(f"  [ROUTE E: TWO STEPS WITH THE MEMORY'S LEARNED VALUES | visit {vis + 1}] trunk + integrated memory logits (no table): first step B right {m(oE['b'])} | "
                    f"TWO-STEP C right {m(oE['c'])} vs direct {m(oE['direct'])} | pure families two-step {m(oE['pure'])}", flush=True)
            except Exception as e8: print(f"  [ROUTE E | visit {vis + 1}] unavailable ({type(e8).__name__}: {str(e8)[:100]})", flush=True)
            try:                                                               # 18 Nov ROUTE D VARIANTS (the premise bottleneck): the READING wording ('A is a child of ',
                rd = lambda A_, r_: f"{A_} {r_} ".encode('latin-1', 'replace')   #   as the premise was read) instead of the test wording; and the composites whose
                tset = {(e['A'], str(e['r']).strip()) for e in fs0 if e.get('key', ('',))[0] == 'fact'}   #   two premises are TAUGHT pool facts
                oV = {'b': [], 'c': [], 'pure': [], 'bt': [], 'ct': [], 'nt': 0}
                for (d, l, r3, C) in T9:
                    sl = int(S_slot(d)); r1, r2 = rules[r3]; b_ = _dw_greedy(model, rd(l, r1), sl, 20); c_ = _dw_greedy(model, rd(b_, r2), sl, 20) if b_ else ''
                    oV['b'].append(b_ == par.get(l)); oV['c'].append(c_ == C)
                    if l in pure: oV['pure'].append(c_ == C)
                for (d, l, r3, C) in tg:
                    r1, r2 = rules[r3]
                    if (l, r1) in tset and (par.get(l), r2) in tset and len(oV['bt']) < 40:
                        sl = int(S_slot(d)); b_ = _dw_greedy(model, te(l, r1), sl, 20); c_ = _dw_greedy(model, te(b_, r2), sl, 20) if b_ else ''
                        oV['bt'].append(b_ == par.get(l)); oV['ct'].append(c_ == C)
                print(f"  [ROUTE D, PREMISES | visit {vis + 1}] store and memory off || READING wording ('A is a child of '): first step B {m(oV['b'])}, two-step C {m(oV['c'])}, "
                      f"pure families {m(oV['pure'])} || both premises TAUGHT pool facts (test wording): first step B {m(oV['bt'])}, two-step C {m(oV['ct'])}", flush=True)
            except Exception as e7: print(f"  [ROUTE D, PREMISES | visit {vis + 1}] unavailable ({type(e7).__name__}: {str(e7)[:100]})", flush=True)
            print(f"  [ROUTE D: THE TRUNK DECODES TWO STEPS | visit {vis + 1}] {src}; store and memory OFF, nothing looked up: composites never planted (n {len(T9)}): "
                  f"first step B right {m(o9['b'])} | TWO-STEP C right {m(o9['c'])} vs the composite asked DIRECTLY {m(o9['direct'])} || PURE FAMILIES: two-step {m(o9['pure'])} vs direct {m(o9['pdirect'])}", flush=True)
            if int(getattr(cfg, 'inf_kmc_rulecomp_dwrite', 0)) and (src.startswith('LEARNED') or int(getattr(cfg, 'inf_kmc_rulecomp_oracle_write', 0))):
                # 18 Nov MODEL CHANGE (kra4, combined by the user's decision): ROUTE D WRITES - the trunk's own two-step decoding via the learned rule is what
                # the relational module writes into the question pool. LABEL-FREE subjects (a recorded r1 fact, no recorded r3 fact); GATE: B and C must be
                # names the system has recorded (a subject or answer of a pool fact), all three different, and B decoded the same in a second wording.
                known = {e['A'] for e in fs0} | {e['ans'] for e in fs0}; hasr = {(e['A'], str(e['r']).strip()) for e in fs0}; sj = {}
                for e in fs0:
                    for r3, (r1, r2) in rules.items():
                        if str(e['r']).strip() == r1 and (e['A'], r3) not in hasr: sj[(e['A'], r3)] = int(S_slot(e['d']))
                RW = model.__dict__.setdefault('_ra_written', {}); nw = 0; ok_ = []; why = {'B unknown': 0, 'B unstable': 0, 'C unknown': 0, 'same name': 0}
                for (A, r3), sl in list(sj.items())[:int(getattr(cfg, 'inf_kmc_rulecomp_dwrite_n', 150))]:
                    r1, r2 = rules[r3]; b_ = _dw_greedy(model, te(A, r1), sl, 20)
                    if b_ not in known: why['B unknown'] += 1; continue
                    if _dw_greedy(model, f"{A} {r1} ".encode('latin-1', 'replace'), sl, 20) != b_: why['B unstable'] += 1; continue
                    if int(getattr(cfg, 'inf_kmc_rulecomp_dgate', 0)) and (b_, r2) not in hasr: why['B has no recorded r2 fact'] = why.get('B has no recorded r2 fact', 0) + 1; continue
                    c_ = _dw_greedy(model, te(b_, r2), sl, 20)
                    if c_ not in known: why['C unknown'] += 1; continue
                    if len({A, b_, c_}) < 3: why['same name'] += 1; continue
                    if (r3, A, c_) not in RW: RW[(r3, A, c_)] = vis; nw += 1
                    model.__dict__.setdefault('_ra_bridge', {})[(r3, A, c_)] = b_
                    m_ = {ph['grand']: W.get('parent', {}), ph['skip']: W.get('mgr', {})}.get(r3)
                    if m_ is not None and A in m_: ok_.append(c_ == m_.get(m_.get(A)))
                print(f"  [ROUTE D WRITES | visit {vis + 1}] MODEL CHANGE, {src}: {len(sj)} subjects with an r1 fact and no r3 fact (up to {int(getattr(cfg, 'inf_kmc_rulecomp_dwrite_n', 150))} asked); "
                      f"{nw} new composites written ({len(RW)} so far) | refused: " + ", ".join(f"{k} {v}" for k, v in why.items())
                      + f" | precision against the rule world's truth {np.mean(ok_) if ok_ else float('nan'):.2f} (n {len(ok_)}; a subject with no composite counts as wrong)", flush=True)
        except Exception as e9: print(f"  [ROUTE D: THE TRUNK DECODES TWO STEPS | visit {vis + 1}] unavailable ({type(e9).__name__}: {str(e9)[:100]})", flush=True)
    if int(getattr(cfg, 'inf_kmc_rulej', 0)):                                  # 18 Nov ROUTE J, THE JEPA-STYLE RELATIONAL MEMORY: one linear operator per relation on the trunk's
        try:                                                                   #   name codes, fitted (closed form, ridge) on the facts the READER RECORDED (the pool); the
            old9 = K.get('tcode'); K['tcode'] = {}                             #   composite = the learned rule's two operators applied in LATENT space (the bridge B is never
            try:                                                               #   decoded), decoded as the nearest code among ALL recorded names (no fact table at question time)
                rel9 = {}
                if int(getattr(cfg, 'inf_kmc_rulej_recent', 0)):               # 19 Nov kra9: BELIEF REVISION BY RECENCY - each (subject, relation) keeps only the answer read in
                    vo9 = {int(k): i for i, k in enumerate((model.__dict__.get('_t2s') or {}).keys())}   #   the most recently visited domain (visit order of the slots)
                    lt9 = {}
                    for e in fs0:
                        k9 = (e['A'], str(e['r']).strip()); o9 = vo9.get(int(e['d']), -1)
                        if k9 not in lt9 or o9 > lt9[k9][0]: lt9[k9] = (o9, e)
                    fsJ = [v[1] for v in lt9.values()]
                else: fsJ = fs0
                for e in fsJ: rel9.setdefault(str(e['r']).strip(), []).append((e['A'], e['ans']))
                tuJ = [(d, l, r3, C) for (d, sent, k, l, C) in W['held'] if int(d) not in seen and ph.get(k) in rules for r3 in [ph.get(k)]][:200]
                NM = sorted({e['A'] for e in fs0} | {e['ans'] for e in fs0} | {l for _, l, _, _ in tg} | {l for _, l, _, _ in tuJ})
                Z = _kma_tcode(model, K, NM).float(); Z = Z - Z.mean(0); Z = Z / (Z.norm(dim=-1, keepdim=True) + 1e-8); NI = {n: k for k, n in enumerate(NM)}
                dec = {e['A'] for e in fs0} | {e['ans'] for e in fs0}; DI = torch.tensor([NI[n] for n in sorted(dec)], device=Z.device); DN = sorted(dec)
                lam = float(getattr(cfg, 'inf_kmc_rulej_lam', 0.1)); D9 = Z.shape[1]; Id = torch.eye(D9, device=Z.device)
                def fit(pairs, shuffle=False):
                    X = Z[[NI[a] for a, b in pairs]]; Y = Z[[NI[b] for a, b in pairs]]
                    if shuffle: Y = Y[torch.randperm(len(Y), generator=torch.Generator().manual_seed(5)).to(Y.device)]
                    return torch.linalg.solve(X.t() @ X + lam * Id, X.t() @ Y)
                HJ = int(getattr(cfg, 'inf_kmc_rulej_mlp', 0)); SNAP = int(getattr(cfg, 'inf_kmc_rulej_snap', 0)); DP = {n: k for k, n in enumerate(DN)}
                if HJ:                                                         # 19 Nov kra7: a JEPA-style PREDICTOR per relation (MLP, InfoNCE over the recorded names' codes,
                    Zd = Z.detach()                                            #   temperature 0.05) - the linear operator holds about D facts; the synthetic world showed the
                    OBJ = str(getattr(cfg, 'inf_kmc_rulej_obj', 'nce'))       # 23 Nov kra44: 'cos' = NON-CONTRASTIVE predictor (1 - cosine to the answer's code, no
                    def fit(pairs, shuffle=False, seed=11, obj=None):                    #   negatives; the trunk's codes are fixed, so nothing can collapse); 'nce' = InfoNCE (default)
                        g9 = torch.Generator().manual_seed(seed)               #   MLP holds 2,200 (first hop 1.00) and its top-1 minus top-2 margin rejects unrecorded subjects
                        with torch.random.fork_rng(devices=[]):                # a fixed init without touching the run's own random stream
                            torch.manual_seed(seed); net = torch.nn.Sequential(torch.nn.Linear(D9, HJ), torch.nn.GELU(), torch.nn.Linear(HJ, HJ), torch.nn.GELU(), torch.nn.Linear(HJ, D9)).to(Z.device)
                        A_ = torch.tensor([NI[a] for a, b in pairs], device=Z.device); yb = torch.tensor([DP[b] for a, b in pairs], device=Z.device)
                        if shuffle: yb = yb[torch.randperm(len(yb), generator=g9).to(yb.device)]
                        o9 = torch.optim.Adam(net.parameters(), lr=1e-3)
                        with torch.enable_grad():
                            for _ in range(int(getattr(cfg, 'inf_kmc_rulej_mlp_steps', 1500))):
                                k9 = torch.randint(0, len(A_), (min(256, len(A_)),), generator=g9).to(Z.device)
                                if (obj or OBJ) == 'cos': l9 = (1 - (F.normalize(net(Zd[A_[k9]]), dim=-1) * Zd[DI][yb[k9]]).sum(-1)).mean()
                                else: l9 = F.cross_entropy(F.normalize(net(Zd[A_[k9]]), dim=-1) @ Zd[DI].t() / 0.05, yb[k9])
                                o9.zero_grad(); l9.backward(); o9.step()
                        net.eval(); return net
                    app = lambda O_, v: O_(v)
                else: app = lambda O_, v: v @ O_
                OP = {r: fit(p) for r, p in rel9.items() if len(p) >= 5}; ON = {r: fit(p, True) for r, p in rel9.items() if len(p) >= 5}
                def comp(A, r3, O):
                    r1, r2 = rules[r3]
                    if r1 not in O or r2 not in O: return None
                    with torch.no_grad():
                        v1 = app(O[r1], Z[NI[A]]); s1 = (F.normalize(v1, dim=-1) @ Z[DI].t()).sort(descending=True); b1 = DN[int(s1.indices[0])]
                        v2 = app(O[r2], Z[NI[b1]] if SNAP else v1); s2 = (F.normalize(v2, dim=-1) @ Z[DI].t()).sort(descending=True)
                    m2 = float(s2.values[0] - s2.values[1]); m1 = float(s1.values[0] - s1.values[1])
                    return b1, DN[int(s2.indices[0])], (min(m1, m2) if HJ else m2)
                if HJ:                                                         # 19 Nov DIAGNOSTICS for the MLP predictor: memorisation of the recorded facts it was fitted on,
                    try:                                                       #   the targets' first hops split by whether that fact was RECORDED, and the linear operator
                        with torch.no_grad():                                  #   on the same targets (kra6's route J) for a same-run comparison
                            mem9 = [DN[int((F.normalize(OP[r](Z[NI[a]]), dim=-1) @ Z[DI].t()).argmax())] == b for r, p in rel9.items() if r in OP for a, b in p[:200]]
                            hasr0 = {(e['A'], str(e['r']).strip()) for e in fs0}
                            def lfit(pairs):
                                X = Z[[NI[a] for a, b in pairs]]; Y = Z[[NI[b] for a, b in pairs]]; return torch.linalg.solve(X.t() @ X + lam * Id, X.t() @ Y)
                            OL = {r: lfit(p) for r, p in rel9.items() if len(p) >= 5}; dgJ = {'rec': [], 'unrec': [], 'lin_b': [], 'lin_c': [], 'lin_rec': []}
                            for (d, l, r3, C) in tg:
                                r1, r2 = rules[r3]
                                if r1 not in OP or r2 not in OP: continue
                                b1 = DN[int((F.normalize(OP[r1](Z[NI[l]]), dim=-1) @ Z[DI].t()).argmax())]; rc_ = (l, r1) in hasr0
                                dgJ['rec' if rc_ else 'unrec'].append(b1 == par.get(l))
                                v1 = Z[NI[l]] @ OL[r1]; bl = DN[int((F.normalize(v1, dim=-1) @ Z[DI].t()).argmax())]; dgJ['lin_b'].append(bl == par.get(l))
                                if rc_: dgJ['lin_rec'].append(bl == par.get(l))
                                dgJ['lin_c'].append(DN[int((F.normalize(v1 @ OL[r2], dim=-1) @ Z[DI].t()).argmax())] == C)
                        print(f"  [ROUTE J, MLP DIAGNOSTICS | visit {vis + 1}] recorded facts memorised {m(mem9)} || targets' first hop B: fact RECORDED {m(dgJ['rec'])}, "
                              f"NOT recorded {m(dgJ['unrec'])} || LINEAR operator, same targets: first hop {m(dgJ['lin_b'])} (recorded {m(dgJ['lin_rec'])}), composite {m(dgJ['lin_c'])}", flush=True)
                    except Exception as e8: print(f"  [ROUTE J, MLP DIAGNOSTICS | visit {vis + 1}] unavailable ({type(e8).__name__}: {str(e8)[:100]})", flush=True)
                oJ = {'b': [], 'c': [], 'pure': [], 'null': [], 'mr': [], 'mw': []}
                for (d, l, r3, C) in tg:
                    x = comp(l, r3, OP)
                    if x is None: continue
                    oJ['b'].append(x[0] == par.get(l)); oJ['c'].append(x[1] == C); (oJ['mr'] if x[1] == C else oJ['mw']).append(x[2])
                    if l in pure: oJ['pure'].append(x[1] == C)
                    y = comp(l, r3, ON); oJ['null'].append(y is not None and y[1] == C)
                mu = []                                                        # never-read subjects (families the model never read): their margins
                for (d, l, r3, C) in tuJ:
                    x = comp(l, r3, OP); mu.append(x[2] if x else 0.0)
                auc = np.mean([[float(a_ > b_) + 0.5 * (a_ == b_) for b_ in mu] for a_ in oJ['mr']]) if oJ['mr'] and mu else float('nan')
                thr9 = float(getattr(cfg, 'inf_kmc_rulej_thr', 0.05))
                if thr9 < 0:                                                   # 19 Nov: AUTO threshold, label-free - the 5th percentile of the single-hop margins of the
                    sm = []                                                    #   RECORDED facts themselves (keeps 95% of what was recorded, as in the synthetic test)
                    with torch.no_grad():
                        for r, p in rel9.items():
                            if r not in OP: continue
                            na = collections.Counter(a for a, b in p)              # subjects with ONE recorded answer only: one-to-many relations tie at the top
                            for a, b in [x for x in p if na[x[0]] == 1][:200]:
                                s9 = F.normalize(app(OP[r], Z[NI[a]]), dim=-1) @ Z[DI].t(); sv = s9.topk(2)
                                if DN[int(sv.indices[0])] == b: sm.append(float(sv.values[0] - sv.values[1]))   # only facts it reproduces: the recorded answer is known, no label needed
                    thr9 = float(np.quantile(sm, 0.05)) if sm else 0.05
                cov = np.mean([m_ >= thr9 for m_ in oJ['mr'] + oJ['mw']]) if oJ['mr'] or oJ['mw'] else float('nan')
                pg = np.mean([1.0] * sum(m_ >= thr9 for m_ in oJ['mr']) + [0.0] * sum(m_ >= thr9 for m_ in oJ['mw'])) if any(m_ >= thr9 for m_ in oJ['mr'] + oJ['mw']) else float('nan')
                print(f"  [ROUTE J: JEPA-STYLE RELATIONAL MEMORY | visit {vis + 1}] {src}; one operator per relation on the trunk's name codes, fitted on {len(fs0)} RECORDED facts "
                      f"({len(OP)} relations); composites never planted, the bridge kept LATENT, C the nearest of {len(DN)} recorded names (chance {1 / max(1, len(DN)):.3f}) || "
                      f"first hop B {m(oJ['b'])} | COMPOSITE C {m(oJ['c'])} | PURE FAMILIES {m(oJ['pure'])} | NULL (operators fitted on shuffled facts) {m(oJ['null'])} || "
                      f"margin gate >= {thr9}: answers {cov:.2f}, right when answering {pg:.2f} | NEVER-READ families: margin >= gate {np.mean([x >= thr9 for x in mu]) if mu else float('nan'):.2f} (n {len(mu)}), "
                      f"AUC right-vs-never-read {auc:.2f} | predictor {'MLP ' + str(HJ) if HJ else 'linear'}{', B snapped' if SNAP else ''}, gate {thr9:.3f}", flush=True)
                if HJ and int(getattr(cfg, 'inf_kmc_rulej_cmpobj', 0)):       # 23 Nov kra44 READOUT: the OTHER objective's route J on the same targets (InfoNCE vs
                    try:                                                       #   NON-CONTRASTIVE cosine; local lecun_steps A: non-contrastive objectives as good as InfoNCE)
                        ob2 = 'nce' if OBJ == 'cos' else 'cos'; OP2 = {r: fit(p, obj=ob2) for r, p in rel9.items() if len(p) >= 5}; o2 = {'b': [], 'c': [], 'pure': [], 'mr': [], 'mw': []}
                        for (d, l, r3, C) in tg:
                            x = comp(l, r3, OP2)
                            if x is None: continue
                            o2['b'].append(x[0] == par.get(l)); o2['c'].append(x[1] == C); (o2['mr'] if x[1] == C else o2['mw']).append(x[2])
                            if l in pure: o2['pure'].append(x[1] == C)
                        mu2 = [(lambda x: x[2] if x else 0.0)(comp(l, r3, OP2)) for (d, l, r3, C) in tuJ]
                        au2 = np.mean([[float(a_ > b_) + 0.5 * (a_ == b_) for b_ in mu2] for a_ in o2['mr']]) if o2['mr'] and mu2 else float('nan')
                        print(f"  [ROUTE J, OBJECTIVE COMPARISON | visit {vis + 1}] route J's operators refitted with '{ob2}' ({'NON-CONTRASTIVE: 1 - cosine, no negatives' if ob2 == 'cos' else 'InfoNCE'}) on the same "
                              f"recorded facts; the run uses '{OBJ}' (above) || first hop B {m(o2['b'])} | COMPOSITE C {m(o2['c'])} | PURE FAMILIES {m(o2['pure'])} | AUC right-vs-never-read {au2:.2f}", flush=True)
                    except Exception as e9: print(f"  [ROUTE J, OBJECTIVE COMPARISON | visit {vis + 1}] unavailable ({type(e9).__name__}: {str(e9)[:100]})", flush=True)
                NE9 = int(getattr(cfg, 'inf_kmc_rulej_ens', 0))
                if HJ and NE9 > 1:                                             # 23 Nov kra44 READOUT: ENSEMBLE CONTRADICTION (local, lecun_steps B: false claims contradicted
                    try:                                                       #   1.00, true 0.00, never-read names 0.02) - route J refitted with other seeds; a claim
                        OPS9 = [OP] + [{r: fit(p, seed=11 + 100 * k9) for r, p in rel9.items() if len(p) >= 5} for k9 in range(1, NE9)]   #   is CONTRADICTED when the
                        def ens9(Q):                                           #   seeds agree (>= 0.8) on another answer; unknown subjects should not be
                            out9 = []
                            for (d, l, r3, C) in Q:
                                xs = [comp(l, r3, O_) for O_ in OPS9]; ans = [x[1] for x in xs if x is not None]
                                if not ans: out9.append(None); continue
                                top = max(set(ans), key=ans.count); out9.append((top, ans.count(top) / len(OPS9), C))
                            return out9
                        rk9 = np.random.RandomState(13); ek = [x for x in ens9(tg) if x is not None]; eu = [x for x in ens9(tuJ) if x is not None]
                        fc9 = [DN[rk9.randint(len(DN))] for _ in ek]; fc9 = [f if f != C else DN[(DN.index(f) + 1) % len(DN)] for f, (t_, a_, C) in zip(fc9, ek)]
                        ctr = lambda x, claim: x[1] >= 0.8 and x[0] != claim
                        akn = [a_ for t_, a_, C in ek]; aun = [a_ for t_, a_, C in eu]
                        auc9 = np.mean([[float(a_ > b_) + 0.5 * (a_ == b_) for b_ in aun] for a_ in akn]) if akn and aun else float('nan')
                        print(f"  [ROUTE J, ENSEMBLE CONTRADICTION | visit {vis + 1}] {NE9} seeds of route J; READ families' composites (n {len(ek)}): seeds' majority right {m([t_ == C for t_, a_, C in ek])}, "
                              f"seeds agree {np.mean(akn) if akn else float('nan'):.2f} || CLAIMS: FALSE claims CONTRADICTED {m([ctr(x, f) for x, f in zip(ek, fc9)])} | TRUE claims contradicted "
                              f"{m([ctr(x, x[2]) for x in ek])} || NEVER-READ families (n {len(eu)}): seeds agree {np.mean(aun) if aun else float('nan'):.2f}, a claim called contradicted "
                              f"{m([x[1] >= 0.8 for x in eu])} (should be low: unknown, not contradicted) | AUC agreement read-vs-never-read {auc9:.2f}", flush=True)
                    except Exception as e9: print(f"  [ROUTE J, ENSEMBLE CONTRADICTION | visit {vis + 1}] unavailable ({type(e9).__name__}: {str(e9)[:100]})", flush=True)
                if HJ and int(getattr(cfg, 'inf_kmc_rulej_search', 0)):       # 23 Nov kra44 READOUT: LATENT PROGRAM SEARCH (local, latent_search / lecun_steps: 3-4 hops 1.00)
                    try:                                                       #   for each composite relation with recorded examples, every program of 1-2 OTHER relations
                        out9 = []                                              #   is run through route J's operators (bridge snapped) and the best on the examples kept;
                        RL9 = [r for r in OP]                                  #   found program = the learned rule? and its answers on the targets
                        for r3, (r1g, r2g) in rules.items():
                            ex = [(a, b) for a, b in rel9.get(r3, []) if a in NI and b in DP][:60]
                            if len(ex) < 5: continue
                            hs = torch.tensor([NI[a] for a, b in ex], device=Z.device); ans = [b for a, b in ex]; sh = list(np.random.RandomState(3).permutation(ans))
                            best = None; bn = 0.0; n9 = 0
                            with torch.no_grad():
                                for ra in RL9:
                                    if ra == r3: continue
                                    s1 = F.normalize(app(OP[ra], Z[hs]), dim=-1) @ Z[DI].t(); i1 = s1.argmax(-1); v1 = Z[DI][i1]
                                    for prog, vv in [((ra,), None)] + [((ra, rb), rb) for rb in RL9 if rb != r3]:
                                        if vv is None: ii = i1
                                        else: ii = (F.normalize(app(OP[vv], v1), dim=-1) @ Z[DI].t()).argmax(-1)
                                        o9 = [DN[int(i)] for i in ii.tolist()]; n9 += 1
                                        acc = np.mean([o == a for o, a in zip(o9, ans)]); bn = max(bn, np.mean([o == a for o, a in zip(o9, sh)]))
                                        if best is None or acc > best[0]: best = (acc, prog)
                            tg3 = [(l, C) for (d, l, r3_, C) in tg if r3_ == r3 and l in NI]; rt9 = []
                            if tg3 and len(best[1]) == 2:
                                with torch.no_grad():
                                    v = Z[torch.tensor([NI[l] for l, C in tg3], device=Z.device)]
                                    for r_ in best[1]: v = Z[DI][(F.normalize(app(OP[r_], v), dim=-1) @ Z[DI].t()).argmax(-1)]
                                    rt9 = [DN[int(i)] == C for i, (l, C) in zip((F.normalize(v, dim=-1) @ Z[DI].t()).argmax(-1).tolist(), tg3)]
                            out9.append(f"'{r3}': found {' o '.join(best[1])} (right on {len(ex)} recorded examples {best[0]:.2f}, shuffled null {bn:.2f}, {n9} programs) "
                                        f"{'= the learned rule' if tuple(best[1]) == (r1g, r2g) else '!= the learned rule ' + r1g + ' o ' + r2g} | on the targets {m(rt9)}")
                        print(f"  [ROUTE J, LATENT PROGRAM SEARCH | visit {vis + 1}] programs of 1-2 relations searched over route J's operators from each composite's recorded examples (no rule given) || "
                              + (" || ".join(out9) if out9 else "no composite relation with 5+ recorded examples"), flush=True)
                    except Exception as e9: print(f"  [ROUTE J, LATENT PROGRAM SEARCH | visit {vis + 1}] unavailable ({type(e9).__name__}: {str(e9)[:100]})", flush=True)
                if int(getattr(cfg, 'inf_kmc_rulej_write', 0)) and (src.startswith('LEARNED') or int(getattr(cfg, 'inf_kmc_rulecomp_oracle_write', 0))):
                    hasr = {(e['A'], str(e['r']).strip()) for e in fs0}; RW = model.__dict__.setdefault('_ra_written', {}); nw = 0; ok_ = []; why = {'margin': 0, 'same name': 0, 'B has no recorded r2 fact': 0}
                    for e in fs0:
                        for r3, (r1, r2) in rules.items():
                            if str(e['r']).strip() != r1 or (e['A'], r3) in hasr: continue
                            x = comp(e['A'], r3, OP)
                            if x is None: continue
                            if x[2] < thr9: why['margin'] += 1; continue
                            if len({e['A'], x[0], x[1]}) < 3: why['same name'] += 1; continue
                            if (x[0], r2) not in hasr: why['B has no recorded r2 fact'] += 1; continue
                            if (r3, e['A'], x[1]) not in RW: RW[(r3, e['A'], x[1])] = vis; nw += 1
                            model.__dict__.setdefault('_ra_bridge', {})[(r3, e['A'], x[1])] = x[0]
                            m_ = {ph['grand']: W.get('parent', {}), ph['skip']: W.get('mgr', {})}.get(r3)
                            if m_ is not None and e['A'] in m_: ok_.append(x[1] == m_.get(m_.get(e['A'])))
                    print(f"  [ROUTE J WRITES | visit {vis + 1}] MODEL CHANGE, {src}: {nw} new composites written ({len(RW)} so far) | refused: " + ", ".join(f"{k} {v}" for k, v in why.items())
                          + f" | precision against the rule world's truth {np.mean(ok_) if ok_ else float('nan'):.2f} (n {len(ok_)}; a subject with no composite counts as wrong)", flush=True)
                    MV = W.get('moves') or {}
                    def tru9(A, r3, C):                                         # truth after any move READ so far (the moved leaf's new composite)
                        mv9 = MV.get(A)
                        if mv9 and mv9['d2'] in seen: return C == mv9['C_new']
                        m9 = {ph['grand']: W.get('parent', {}), ph['skip']: W.get('mgr', {})}.get(r3)
                        return (C == m9.get(m9.get(A))) if m9 is not None and A in m9 else None
                    if int(getattr(cfg, 'inf_kmc_rulej_reverify', 0)) and RW:  # 19 Nov kra9 MODEL CHANGE: RE-DERIVE every earlier write with today's predictor (synthetic world:
                        WRm = model.__dict__.get('_rm_written', {}); BR = model.__dict__.setdefault('_ra_bridge', {}); kp9, rw9, rt9 = [], [], []   # check the bridge, retract,
                        for (r3, A, C) in list(RW):                             #   re-derive, and write back EVERY checked composite: moved people 0.76-0.90 asked directly)
                            if r3 not in rules or A not in NI: continue
                            x = comp(A, r3, OP); r2 = rules[r3][1]
                            ok9 = x is not None and x[2] >= thr9 and len({A, x[0], x[1]}) == 3 and (x[0], r2) in hasr
                            if ok9 and x[1] == C: BR[(r3, A, C)] = x[0]; kp9.append(tru9(A, r3, C)); continue
                            RW.pop((r3, A, C), None); WRm.pop((r3, A, C), None); BR.pop((r3, A, C), None)
                            if ok9: RW[(r3, A, x[1])] = vis; BR[(r3, A, x[1])] = x[0]; rw9.append((tru9(A, r3, C), tru9(A, r3, x[1])))
                            else: rt9.append(tru9(A, r3, C))
                        pr9 = lambda L: f"{np.mean([v for v in L if v is not None]):.2f}" if any(v is not None for v in L) else 'n/a'
                        print(f"  [ROUTE J RE-DERIVE | visit {vis + 1}] MODEL CHANGE: every earlier route J write re-derived with today's predictor and gate: KEPT {len(kp9)} (right {pr9(kp9)}) | "
                              f"REWRITTEN {len(rw9)} (old right {pr9([a for a, b in rw9])} -> new right {pr9([b for a, b in rw9])}) | RETRACTED {len(rt9)} (were right {pr9(rt9)}) | "
                              f"{len(RW)} written composites now", flush=True)
                    if MV:                                                      # 19 Nov kra9 READOUT: the MOVED leaves (premise changed in a domain read later)
                        hasp = collections.defaultdict(set)
                        for e in fs0: hasp[(e['A'], str(e['r']).strip())].add(e['ans'])
                        g9 = {'read': [], 'ctl': [], 'readok': []}; fd9 = {a_: int(d_) for (d_, s_, k_, a_, b_) in W['planted'] if k_ in ('parent', 'mgr')}
                        for l, mv9 in MV.items():
                            r3 = ph['grand'] if mv9['kind'] == 'parent' else ph['skip']; r1 = ph[mv9['r1']]
                            if int(mv9['d']) not in seen or l not in NI or r3 not in rules: continue
                            x = comp(l, r3, OP); rec = hasp.get((l, r1), set()); wr = [c for (q, a, c) in RW if q == r3 and a == l]
                            if mv9['d2'] in seen and fd9.get(mv9['new']) in seen: g9['readok'].append((mv9['new'] in rec, mv9['old'] in rec, x is not None and x[1] == mv9['C_new'],
                                x is not None and x[1] == mv9['C_old'], x is not None and x[2] >= thr9, mv9['C_new'] in wr, mv9['C_old'] in wr))   # 19 Nov: the new parent's own family read too
                            g9['read' if mv9['d2'] in seen else 'ctl'].append((mv9['new'] in rec, mv9['old'] in rec, x is not None and x[1] == mv9['C_new'], x is not None and x[1] == mv9['C_old'],
                                                                               x is not None and x[2] >= thr9, mv9['C_new'] in wr, mv9['C_old'] in wr))
                        def mm9(L, j): return f"{np.mean([t[j] for t in L]):.2f}" if L else 'n/a'
                        print(f"  [ROUTE J, PREMISE CHANGE | visit {vis + 1}] MOVED leaves (a new parent planted in another domain; truth after the move = the new family's root) || "
                              + " || ".join(f"{nm} (n {len(L)}): reader recorded NEW premise {mm9(L, 0)}, OLD {mm9(L, 1)} | route J composite NEW {mm9(L, 2)}, OLD {mm9(L, 3)}, passes gate {mm9(L, 4)} | "
                                            f"WRITTEN composite NEW {mm9(L, 5)}, OLD (stale) {mm9(L, 6)}" for nm, L in (('MOVE READ', g9['read']), ('MOVE READ AND THE NEW PARENT\'S FAMILY READ', g9['readok']), ('MOVE NOT YET READ (control)', g9['ctl']))), flush=True)
            finally: K['tcode'] = old9 if old9 is not None else {}
        except Exception as e7: print(f"  [ROUTE J | visit {vis + 1}] unavailable ({type(e7).__name__}: {str(e7)[:120]})", flush=True)
    tu = [(d, l, r3, C) for (d, sent, k, l, C) in W['held'] if int(d) not in seen and ph.get(k) in rules for r3 in [ph.get(k)]][:200]
    if tu:                                                                     # FALSE-MEMORY CHECK: the same two reads for composites whose subjects were NEVER READ
        sl = int(S_slot(max(seen))) if seen else 0; uc = []
        d1u = dist([te(l, rules[r3][0]) for d, l, r3, C in tu], sl)
        for (d, l, r3, C), d1 in zip(tu, d1u):
            sc = {}
            for b in sorted(d1, key=d1.get, reverse=True)[:KR]:
                for (d2,) in [dist([te(b, rules[r3][1])], sl)]:
                    for n, v in d2.items(): sc[n] = sc.get(n, 0.0) + d1[b] * v
            uc.append(max(sc.values()) if sc else 0.0)
        WS = K.get('subj', set()); PNu = {s_: {n: v / sum(dd.values()) * np.log(1.0 + len(names) / len(dd)) for n, v in dd.items()} for s_, dd in PTR.items()}
        def gated(Q, sl_):                                                     # the GATE: answer only when the subject AND the first read's top name were written subjects
            o_ = []
            for (d, l, r3, C), d1 in zip(Q, dist([te(l, rules[r3][0]) for d, l, r3, C in Q], sl_, PNu)):
                b1 = max(d1, key=d1.get) if d1 else None
                if b1 is None or l not in WS or b1 not in WS: o_.append(None); continue
                d2 = dist([te(b1, rules[r3][1])], sl_, PNu)[0]; o_.append(max(d2, key=d2.get) == C if d2 else None)
            return o_
        gu = gated(tu, sl); gr = []
        for sl2 in sorted({int(S_slot(d)) for d, _, _, _ in tg}): gr += gated([t for t in tg if int(S_slot(t[0])) == sl2], sl2)
        kr = [x for x in gr if x is not None]
        print(f"  [ROUTE A, GATED | visit {vis + 1}] slot-normalised table + GATE (the subject and the first read's top name both written subjects of the memory): "
              f"READ families answered {np.mean([x is not None for x in gr]):.2f}, right when answered {np.mean(kr) if kr else float('nan'):.2f}, right overall {np.mean([x is True for x in gr]):.2f} (n {len(gr)}) | "
              f"NEVER-READ families given a name {np.mean([x is not None for x in gu]):.2f} (n {len(gu)}) | {len(WS)} subjects written", flush=True)
        rc = [c for c, o in dg['conf'] if o]; auc = np.mean([[float(a_ > b_) + 0.5 * (a_ == b_) for b_ in uc] for a_ in rc]) if rc and uc else float('nan')
        print(f"  [ROUTE A, NEVER-READ SUBJECTS | visit {vis + 1}] {len(tu)} composites whose family was NEVER READ, asked the same way: the memory gives a name {np.mean([c > 0 for c in uc]):.2f} | "
              f"composite score: never read median {np.median(uc):.3f} vs right answers on read families {np.median(rc) if rc else float('nan'):.3f} | AUC right-vs-never-read {auc:.2f} "
              f"(1.0 = a threshold on the composite score abstains on every never-read subject and keeps every right answer)", flush=True)
    print(f"  [ROUTE A, DIAGNOSTICS | visit {vis + 1}] second hop alone (read 2 from the TRUE B) {m(dg['hop2'])} | composite when the first hop was right {m(dg.get(('f1', True), []))}, "
          f"wrong {m(dg.get(('f1', False), []))} | only the top B carried {m(dg['top1'])} (top {KR}: {m(out['all'])}) || BY DOMAIN (composite / second hop alone): "
          + " ".join(f"d{d_} {np.mean(dg[('d', d_)]):.2f}/{np.mean(dg[('dom', d_)]):.2f} (n {len(dg[('d', d_)])})" for d_ in sorted(k[1] for k in dg if isinstance(k, tuple) and k[0] == 'd')), flush=True)


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
                lg = _ws_run(model, X, sl).float(); K['_open'] = {}; lgm = kmi_logits(model, K, [pb] * len(cb), X, lg, sl); gop.append(int(bool(K['_open'].get(0, False))))
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
                        if use: zg = kmi_logits(model, K, [pb], Xg, zg, sl)
                        q = len(pb) + len(o) - 1
                        if q >= zg.shape[1]: break
                        o += bytes([int(zg[0, q].argmax())])
                    outs.append(int(o == tb))
                if outs: ooff.append(outs[0]); oon.append(outs[1])
        if not on: return "- (0)"
        g_ = f", gate open {np.mean(gop):.2f}" if K.get('gate') else ''
        if not oon: return f"WITH - / {np.mean(on):.2f} vs trunk - / {np.mean(off):.2f} ({len(on)}{g_})"
        return f"WITH {np.mean(oon):.2f} / {np.mean(on):.2f} vs trunk {np.mean(ooff):.2f} / {np.mean(off):.2f} ({len(on)}{g_})"
    bw = lambda e: f"{e['A']} {e['r']} "                                       # 15 Nov ALL-TEXT readout: the bare sentence wording, no frame
    te = lambda e: f"It is on record that {e['A']} {e['r']} "; pa = lambda e: f"It is on record that {e['A']} {_DT_PARA[e['r']]} "
    from platypus.inference.premises import is_rti as _is_rti            # 16 Nov: typed writes ('head wording tail') are inferences too
    kind = lambda L, t=None: [e for e in L if (not _is_rti(e) if t == 'fact' else (_is_rti(e) and e.get('truth') is t))]
    rs.shuffle(fs); rs.shuffle(hf)
    lm = K['loss']; ml = f"{np.mean(lm[:50]):.3f} -> {np.mean(lm[-50:]):.3f}" if lm else "-"
    print(f"  [INTEGRATED KEYED MEMORY | visit {vis + 1}] the model's answer = trunk + memory read ({'EVERY byte of any text, written on answer bytes' if K.get('allt') else 'question rows only'}; subject key: {K.get('namekey')}; {K['steps']} memory steps, "
          f"{K['hit']} of {K['rows']} rows addressed, {len(K['touched'])} of {K['n'] ** 2} slots used, memory loss {ml}), test wording, open (greedy exact) / "
          f"rank among 10 (chance 0.10) || taught FACTS {score(kind(fs, 'fact'), te)} | HELD BACK facts {score(kind(hf, 'fact'), te, 30)} | taught inferences "
          f"RIGHT {score(kind(fs, True), te)} / WRONG {score(kind(fs, False), te, 40)} | held back inferences {score(kind(hf, True) + kind(hf, False), te, 30)} | "
          f"PARAPHRASED facts {score([e for e in kind(fs, 'fact') if e['r'] in _DT_PARA], pa, 40)}" +
          (f" | BARE wording 'A r ' facts {score(kind(fs, 'fact'), bw, 40)} || ALL-TEXT address (middle block, in context; wording head "
           f"loss {K.get('hloss', float('nan')):.2f} over {K.get('hrels', 0)} relations; gate weight {K.get('gatew')})" if K.get('allt') else ''), flush=True)
    if K.get('allt'):                                                          # 15 Nov TWO-HOP THROUGH THE MEMORY (REMERGE-style chaining; readout)
        try:
            _kma_twohop(model, cfg, K, S_slot, fs, pool, te, vis)
            if int(getattr(cfg, 'inf_kmc', 0)): _kma_twohop(model, cfg, K, S_slot, fs, pool, te, vis, memonly=True)
            if int(getattr(cfg, 'inf_kmc_keyread', 0)): _kma_twohop_key(model, cfg, K, S_slot, fs, pool, vis)   # 16 Nov route C (readout)
            if int(getattr(cfg, 'inf_kmc_probe2', 0)):
                try: kma_probe_twohop(model, cfg, K, S_slot, fs, vis)
                except Exception as e8: print(f"  [TWO-HOP PROBES THROUGH THE MEMORY | visit {vis + 1}] unavailable ({type(e8).__name__}: {str(e8)[:120]})", flush=True)
            if int(getattr(cfg, 'inf_kmc_rulecomp', 0)):
                try: kma_rule_compose(model, cfg, K, S_slot, fs + (hf if int(getattr(cfg, 'inf_kmc_rulej_allrec', 0)) else []), vis)   # 19 Nov kra11: every RECORDED fact, held-back control too
                except Exception as e8: print(f"  [ROUTE A: COMPOSITION THROUGH THE MEMORY | visit {vis + 1}] unavailable ({type(e8).__name__}: {str(e8)[:120]})", flush=True)
            if int(getattr(cfg, 'inf_kmc_compose', 0)):                        # 17 Nov THE COMPOSING BUILD: composed two-hop conclusions written to the ledger
                try: kma_compose_write(model, cfg, K, S_slot, fs, vis)
                except Exception as e9: print(f"  [COMPOSED WRITES | visit {vis + 1}] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
        except Exception as e9:
            from platypus.common import _fail
            _fail('inference.kmem:twohop'); print(f"  [INTEGRATED KEYED MEMORY, TWO-HOP] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
    if K.get('gate'):                                                          # ABSTENTION: questions about names the system never read
        from platypus.inference.distill import _DT_REFUSE_SYL
        rels = sorted({e['r'] for e in fs}); g9 = np.random.RandomState(int(cfg.seed) + 5454 + vis); op9 = []
        for k in range(60):
            nm = ''.join(g9.choice(_DT_REFUSE_SYL, 2)).title() + ' ' + ''.join(g9.choice(_DT_REFUSE_SYL, 3)).title()
            pb = f"It is on record that {nm} {rels[g9.randint(len(rels))]} ".encode('latin-1', 'replace'); X, _ = _ws_batch(model, [pb + b'Xx.'])
            with torch.no_grad(): lg = _ws_run(model, X, 0).float(); K['_open'] = {}; kmi_logits(model, K, [pb], X, lg, 0)
            op9.append(int(bool(K['_open'].get(0, False))))
        print(f"  [INTEGRATED KEYED MEMORY, GATE | visit {vis + 1}] the memory answers only where its address was written ({len(K.get('subj', ()))} subjects written, "
              f"{int(K['wmask'].sum()) if 'wmask' in K else 0} slots): gate OPEN on questions about NEVER-READ names {np.mean(op9):.2f} (60; the memory abstains on the rest - "
              f"the trunk answers these with a stored name, false-memory probe a)", flush=True)
    try: _kmi_status_readout(model, cfg, K, fs, score, te, vis)              # 15 Nov READOUTS: by consolidation status and age; learning speed as a judge
    except Exception as e9:
        from platypus.common import _fail
        _fail('inference.kmem:status'); print(f"  [INTEGRATED KEYED MEMORY, BY STATUS] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
    if int(getattr(cfg, 'inf_kmi_counter', 0)):                                # COUNTER-WRITE (oracle, on a copy): unlearn the WRONG inferences, re-write the rest
        try: _kmi_counter_readout(model, cfg, K, S_slot, kind(fs, True), kind(fs, False), kind(fs, 'fact'), vis)
        except Exception as e9:
            from platypus.common import _fail
            _fail('inference.kmem:counter'); print(f"  [INTEGRATED KEYED MEMORY, COUNTER-WRITE] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
    K['loss'] = lm[-200:]


def _kmc_byte_edit(model, cfg, K, S_slot, L, rec=None):
    """15 Nov RECONSOLIDATION-STYLE RETRACTION (inf_kmc_mode 'byte' / 'wide'): retrieve the slots each retracted answer byte reads (every
    training frame, every byte of the answer) and lower ONLY THAT BYTE'S LOGIT in those slots until the memory's support (log p with it minus
    without) is below -inf_kmc_margin - no other byte of any slot moves, so a sibling fact of the same subject (other answer bytes) keeps its
    values. 'byte': the 8 slots read, in proportion to the read weight; 'wide': all 64 slots of the product neighbourhood (8 subject keys x
    8 wording keys) by the same amount - so wordings the edit never saw, which read nearby slots, lose it too. After reconsolidation (an
    update of the retrieved trace at prediction error; Sevenster, Beckers & Kindt 2013), not extinction (a competing memory). rec: every
    edit (slot, byte, delta) is recorded under the inference's key, so a retraction can be UNDONE if later reading supports it (forgetting as
    lost access, Ryan & Frankland 2022)."""
    from platypus.inference.distill import _dt_forms
    from platypus.inference.closed_form import _ws_batch, _ws_run
    FR = _dt_forms(cfg); mg = float(getattr(cfg, 'inf_kmc_margin', 1.0)); wide = str(getattr(cfg, 'inf_kmc_mode', 'margin')) == 'wide'; n = K['n']
    rows = [(f.format(A=e['A'], r=e['r']).encode('latin-1', 'replace'), (e['ans'] + '.').encode('latin-1', 'replace'), int(S_slot(e['d'])), (e['A'], e['ans'])) for e in L for f in FR]
    bysl = {}
    for r9 in rows: bysl.setdefault(r9[2], []).append(r9)
    left = 0
    for it9 in range(int(getattr(cfg, 'inf_kmc_passes', 8))):
        left = 0
        for sl, RW in bysl.items():
            for i0 in range(0, len(RW), 64):
                ch = RW[i0:i0 + 64]; X, _ = _ws_batch(model, [a + b for a, b, c, k in ch]); p0s = [len(a) - 1 for a, b, c, k in ch]
                with torch.no_grad():
                    lg = _ws_run(model, X, sl).float(); ns = [min(len(b), lg.shape[1] - p0) for (a, b, c, k), p0 in zip(ch, p0s)]
                    ad = _kma_addr_rows(model, K, X, sl, p0s, ns) if K.get('allt') else _kmi_addr(model, K, [a for a, b, c, k in ch], ns)
                    if ad is None: continue
                    nk, wk, off, rr, pp = ad; mv, idx = _kmi_read(K, nk, wk, off); w = K['_w']
                    z0 = torch.stack([lg[r, p0s[r] + j] for r, j in zip(rr, pp)]); tg = torch.stack([X[r, p0s[r] + 1 + j] for r, j in zip(rr, pp)])
                    sup = F.log_softmax(z0 + mv, -1).gather(-1, tg[:, None])[:, 0] - F.log_softmax(z0, -1).gather(-1, tg[:, None])[:, 0]
                    nm9 = torch.tensor([j < len(ch[r][1]) - 1 for r, j in zip(rr, pp)], device=sup.device)   # the NAME's bytes only (not the final '.', shared by all)
                    bad = (sup > -mg) & nm9
                    if not bool(bad.any()): continue
                    left += int(bad.sum()); ex = (sup[bad] + mg + 0.5)
                    if wide:                                                  # the whole product neighbourhood of the read: 8 x 8 slots
                        q1 = F.normalize(nk[bad] @ K['R1'], dim=-1); w2 = F.normalize(wk[bad] + 0.5 * K['EOFF'][off[bad]], dim=-1)
                        i1 = (q1 @ K['k1'].t()).topk(_KM_KK, -1)[1]; i2 = (w2 @ K['k2'].t()).topk(_KM_KK, -1)[1]
                        ix = (i1[:, :, None] * n + i2[:, None, :]).flatten(1); dl = -ex[:, None].expand(-1, ix.shape[1])
                    else:
                        ix = idx[bad]; wb = w[bad]; dl = -ex[:, None] * wb / (wb * wb).sum(-1, keepdim=True)
                    tb = tg[bad][:, None].expand(-1, ix.shape[1])
                    fl = ix.flatten() * K['V'] + tb.flatten(); u, inv = torch.unique(fl, return_inverse=True)   # one change per (slot, byte): the
                    dmin = torch.zeros(len(u), device=dl.device).scatter_reduce(0, inv, dl.flatten(), 'amin', include_self=True)   #   largest asked, not the sum
                    K['v'].view(-1).index_add_(0, u, dmin)
                    dl = dmin[inv].view_as(dl) * (torch.arange(len(fl), device=dl.device) == torch.zeros(len(u), dtype=torch.long, device=dl.device).scatter_reduce(0, inv, torch.arange(len(fl), device=dl.device), 'amin', include_self=False)[inv]).view_as(dl).float()
                    if rec is not None:
                        bk = [ch[r][3] for r, q in zip(rr, bad.tolist()) if q]
                        for q, kk in enumerate(bk): rec.setdefault(kk, []).append((ix[q].detach().cpu(), int(tb[q, 0]), dl[q].detach().cpu()))
        if left == 0: break
    return left


def kmc_restore(K, rec, key):
    """undo a recorded byte-edit retraction (later reading supports the inference after all)"""
    with torch.no_grad():
        for ix, b, dl in rec.pop(key, []): K['v'].index_put_((ix.to(K['v'].device), torch.full_like(ix, b).to(K['v'].device)), -dl.to(K['v'].device), accumulate=True)


def _kmi_counter_readout(model, cfg, K, S_slot, right, wrong, facts, vis):
    """ORACLE (the labels pick the wrong inferences - the mechanism, not a trigger), on a COPY of the memory: 60 steps driving the memory's
    support for the WRONG inferences' answer bytes (log-probability with it minus without) below -inf_kmc_margin while re-writing the RIGHT ones
    (GPU cg27a_kmg visit 3: driving it only to zero left the wrong ones ranked 0.90) (same subjects share slots) - offline (tools/jepa_mem.py JM_M2) this
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
                lg = _ws_run(model, X, int(S_slot(e['d']))).float(); gt = K.get('gate'); K['gate'] = 0; z = kmi_logits(model, K, [pb] * len(cb), X, lg, int(S_slot(e['d']))); K['gate'] = gt
                lp = F.log_softmax(z, -1); l0 = F.log_softmax(lg, -1); p0 = len(pb) - 1; n0 = min(len(cb[0]), lp.shape[1] - p0)
                if n0 <= 0: continue                                                   # 17 Nov: a long (retyped) prompt past the row's width
                ar = torch.arange(p0, p0 + n0, device=lp.device)
                sc = [float(lp[j, torch.arange(p0, p0 + max(1, min(len(c), lp.shape[1] - p0)), device=lp.device), X[j, p0 + 1:p0 + 1 + max(1, min(len(c), lp.shape[1] - p0))]].mean()) for j, c in enumerate(cb)]
                sup.append(float((lp[0, ar, X[0, p0 + 1:p0 + 1 + n0]] - l0[0, ar, X[0, p0 + 1:p0 + 1 + n0]]).mean()) / math.log(2))
            hit.append(int(sc[0] > max(sc[1:])))
        return (np.mean(hit) if hit else float('nan'), np.mean(sup) if sup else float('nan'))
    b4 = [evalg(x) for x in (wrong, right, facts)]; snap = K['v'].detach().clone(); st0 = K['opt'].state_dict()
    md = str(getattr(cfg, 'inf_kmc_mode', 'margin'))
    if md in ('byte', 'wide'):
        rec = {}; lf = _kmc_byte_edit(model, cfg, K, S_slot, wrong, rec); af = [evalg(x) for x in (wrong, right, facts)]
        e1 = [evalg(x) for x in (wrong,)]
        for kk in list(rec): kmc_restore(K, rec, kk)                            # REVERSIBLE: undo every edit
        rs_ = evalg(wrong)
        with torch.no_grad(): K['v'].copy_(snap)
        print(f"  [INTEGRATED KEYED MEMORY, COUNTER-WRITE ({md.upper()} EDIT) | visit {vis + 1}] ORACLE on a copy: only the WRONG answers' own bytes lowered in the slots "
              f"they read ({'all 64 of the neighbourhood' if md == 'wide' else 'the 8 read, by weight'}), training frames only; rank WITH the memory and its SUPPORT (bits per "
              f"answer byte): WRONG {b4[0][0]:.2f}, {b4[0][1]:+.2f} -> {af[0][0]:.2f}, {af[0][1]:+.2f} | RIGHT {b4[1][0]:.2f}, {b4[1][1]:+.2f} -> {af[1][0]:.2f}, {af[1][1]:+.2f} | "
              f"facts {b4[2][0]:.2f}, {b4[2][1]:+.2f} -> {af[2][0]:.2f}, {af[2][1]:+.2f} | bytes above the margin left {lf} | UNDONE: WRONG back to {rs_[0]:.2f}, {rs_[1]:+.2f}", flush=True)
        return
    for st in range(int(getattr(cfg, 'inf_kmc_steps', 60))):
        for L, ul in ((wrong, True), (right, False)):
            it, X, lg = rows(L, 32); p0s = [len(a) - 1 for a, b, c in it]; ns = [min(len(b), lg.shape[1] - p0) for (a, b, c), p0 in zip(it, p0s)]
            a = _kma_addr_rows(model, K, X, it[0][2], p0s, ns) if K.get('allt') else _kmi_addr(model, K, [a for a, b, c in it], ns)
            if a is None: continue
            nk, wk, off, rr, pp = a; mv, _ = _kmi_read(K, nk, wk, off)
            z0 = torch.stack([lg[r, p0s[r] + j] for r, j in zip(rr, pp)]); z = z0 + mv; tg = torch.stack([X[r, p0s[r] + 1 + j] for r, j in zip(rr, pp)])
            if ul:                                                              # unlearn: the memory's SUPPORT for the wrong answer's bytes pushed below -margin
                mg = float(getattr(cfg, 'inf_kmc_margin', 1.0))
                lo = (F.log_softmax(z, -1).gather(-1, tg[:, None]) - F.log_softmax(z0, -1).gather(-1, tg[:, None]) + mg).clamp(min=0).mean()
            else: lo = F.cross_entropy(z, tg)
            K['opt'].zero_grad(); lo.backward(); K['opt'].step()
    af = [evalg(x) for x in (wrong, right, facts)]
    with torch.no_grad(): K['v'].copy_(snap)
    K['opt'].load_state_dict(st0)
    print(f"  [INTEGRATED KEYED MEMORY, COUNTER-WRITE | visit {vis + 1}] ORACLE on a copy (the labels pick them): {int(getattr(cfg, 'inf_kmc_steps', 60))} steps unlearning the WRONG inferences while "
          f"re-writing the RIGHT ones; rank WITH the memory (chance 0.10) and the MEMORY'S SUPPORT for the answer (bits per answer byte the memory adds; 0 = none): "
          f"WRONG {b4[0][0]:.2f}, {b4[0][1]:+.2f} -> {af[0][0]:.2f}, {af[0][1]:+.2f} | RIGHT {b4[1][0]:.2f}, {b4[1][1]:+.2f} -> {af[1][0]:.2f}, {af[1][1]:+.2f} | "
          f"facts {b4[2][0]:.2f}, {b4[2][1]:+.2f} -> {af[2][0]:.2f}, {af[2][1]:+.2f}", flush=True)



def kmc_retract(model, cfg, S_slot, ents, vis):
    """15 Nov CONSOLIDATION THROUGH THE MEMORY (inf_kmc): inferences the retraction trigger has RETRACTED since the last readout are counter-
    written OUT of the real memory - 60 steps pushing the memory's support for their answers below -inf_kmc_margin while re-writing the kept
    inferences (they share subjects and wording, hence slots). They are no longer taught (inf_rret) and never reached the trunk (inf_kmc)."""
    from platypus.inference.premises import _PM, _RTI_R
    from platypus.inference.distill import _dt_forms
    from platypus.inference.closed_form import _ws_batch, _ws_run
    K = model.__dict__.get('_kmi'); W = _PM.get('rti') or {}
    if K is None or K['head'] is None or not W: return
    new = [(k, w) for k, w in W.items() if w.get('ret') and not w.get('kmc_done')]
    kept = [(k, w) for k, w in W.items() if w.get('judged') is not None and not w.get('flag')]
    pend = [(k, w) for k, w in W.items() if w.get('judged') is None]
    from platypus.inference.premises import rti_ent
    ent = lambda k, w: rti_ent(k, w)
    print(f"  [CONSOLIDATION THROUGH THE MEMORY | visit {vis + 1}] written inferences: {len(pend)} awaiting judgement (memory only), {len(kept)} judged and KEPT "
          f"(now taught to the trunk too: right {np.mean([bool(w['truth']) for k, w in kept]) if kept else float('nan'):.2f}), "
          f"{sum(1 for w in W.values() if w.get('flag'))} flagged (never taught to the trunk; right {np.mean([bool(w['truth']) for w in W.values() if w.get('flag')]) if any(w.get('flag') for w in W.values()) else float('nan'):.2f}), "
          f"{sum(1 for w in W.values() if w.get('ret'))} of them retracted from the memory ({len(new)} now) | WITH AN ORACLE the trunk would receive "
          f"{sum(1 for k, w in kept if w['truth'])} right and 0 wrong; it receives {sum(1 for k, w in kept if w['truth'])} right and {sum(1 for k, w in kept if not w['truth'])} wrong", flush=True)
    try:                                                                       # 15 Nov INFERENCE LEDGER (readout): epistemic state and provenance of each inference
        from platypus.inference.compress import _cg_auc
        stt = lambda w: 'retracted' if w.get('ret') else 'flagged' if w.get('flag') else 'kept (judged)' if w.get('judged') is not None else 'inferred (awaiting)'
        cnt = collections.Counter(stt(w) for w in W.values()); pv = [(k, w) for k, w in W.items() if w.get('prem')]
        ex = " | ".join(f"{k[0]} ~ {k[1]}: premises {', '.join(w['prem'][:3])} ({w['bridges']} bridges), rule {w.get('rule')}, score {w.get('score', float('nan')):.3f}, "
                        f"written at visit {w['vis'] + 1}, {stt(w)}" for k, w in pv[:2])
        yv = np.array([bool(w['truth']) for k, w in pv]); sv = np.array([w.get('score', 0.0) for k, w in pv]); bv = np.array([w['bridges'] for k, w in pv], float)
        au = lambda x: _cg_auc(x[yv], x[~yv]) if len(yv) >= 20 and yv.any() and (~yv).any() else float('nan')
        print(f"  [INFERENCE LEDGER | visit {vis + 1}] {len(W)} inferences: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) + f" | with provenance {len(pv)} "
              f"(premises, rule, score) - e.g. {ex} || provenance as evidence (evaluation only): AUC of the write score {au(sv):.3f}, of the number of bridges {au(bv):.3f}", flush=True)
    except Exception as e9:
        print(f"  [INFERENCE LEDGER] unavailable ({type(e9).__name__}: {e9})"[:200], flush=True)
    if not new: return
    FR = _dt_forms(cfg); g = np.random.RandomState(int(cfg.seed) + 5757 + vis); mg = float(getattr(cfg, 'inf_kmc_margin', 1.0))
    RX = [ent(k, w) for k, w in new]; RT = [ent(k, w) for k, w in kept] or [ent(k, w) for k, w in pend if not w.get('flag')]
    if str(getattr(cfg, 'inf_kmc_mode', 'margin')) in ('byte', 'wide'):       # reconsolidation-style edit, recorded (reversible)
        _kmc_byte_edit(model, cfg, K, S_slot, RX, K.setdefault('rec', {}))
        for k, w in new: w['kmc_done'] = vis
        return
    def step(L, ul):
        it = [(FR[g.randint(len(FR))].format(A=e['A'], r=e['r']).encode('latin-1', 'replace'), (e['ans'] + '.').encode('latin-1', 'replace'), int(S_slot(e['d']))) for e in (L[g.randint(len(L))] for _ in range(32))]
        X, _ = _ws_batch(model, [a + b for a, b, c in it])
        with torch.no_grad(): lg = _ws_run(model, X, it[0][2]).float()
        p0s = [len(a) - 1 for a, b, c in it]; ns = [min(len(b), lg.shape[1] - p0) for (a, b, c), p0 in zip(it, p0s)]
        a = _kma_addr_rows(model, K, X, it[0][2], p0s, ns) if K.get('allt') else _kmi_addr(model, K, [a for a, b, c in it], ns)
        if a is None: return
        nk, wk, off, rr, pp = a; mv, _ = _kmi_read(K, nk, wk, off)
        z0 = torch.stack([lg[r, p0s[r] + j] for r, j in zip(rr, pp)]); z = z0 + mv; tg = torch.stack([X[r, p0s[r] + 1 + j] for r, j in zip(rr, pp)])
        lo = (F.log_softmax(z, -1).gather(-1, tg[:, None]) - F.log_softmax(z0, -1).gather(-1, tg[:, None]) + mg).clamp(min=0).mean() if ul else F.cross_entropy(z, tg)
        K['opt'].zero_grad(); lo.backward(); K['opt'].step()
    for st in range(int(getattr(cfg, 'inf_kmc_steps', 60))):
        step(RX, True)
        if RT: step(RT, False)
    for k, w in new: w['kmc_done'] = vis


def kmfb_readout(model, cfg, S_slot, ents, vis):
    """15 Nov READOUT, MEMORY FEEDBACK FROM READING (inf_kmfb; the trunk's experience grades the hippocampus, label-free). The all-text
    memory predicts bytes wherever its gate opens; real text read since the first writes then CONFIRMS or CONTRADICTS each entry in use:
    spans of the distant documents read since 2/10 ('A <words> B', <= 60 characters) whose first name is a written subject; on the second
    name's bytes, the bits the gated memory saves (trunk - trunk + memory). The entry credited is the subject's entry whose answer's first
    byte the memory's read favours at the name's first byte; CONFIRM = the text's name is that answer, else the gain is the entry's
    CONTRADICTION in that context (the address includes the wording key, so the gate opens only for compatible relations). Per written
    inference: spans, mean gain, confirmations - contradictions; their AUC right vs wrong (labels only score; the current judge 0.64),
    coverage, and how well the write-time features (rank, bridges) predict the feedback (could it guide selection?)."""
    from platypus.inference.premises import _PM, _RTI_R
    from platypus.inference.closed_form import _ws_batch
    from platypus.inference.compress import _cg_auc
    K = model.__dict__.get('_kmi'); W = _PM.get('rti') or {}; R = _PM.get('R') or []
    if K is None or K.get('head') is None or not K.get('allt') or not R:
        print(f"  [MEMORY FEEDBACK FROM READING | visit {vis + 1}] needs the all-text memory and the distant records", flush=True); return
    nv = max(1, int(getattr(cfg, 'inf_premises_visits', 10))); ND = int(_PM.get('ND', 0)); lo_, hi_ = int(round(ND * 0.2)), int(round(ND * min(1.0, (vis + 1) / nv)))
    if hi_ <= lo_:
        print(f"  [MEMORY FEEDBACK FROM READING | visit {vis + 1}] no text read since the first writes yet", flush=True); return
    ent = collections.defaultdict(dict)                                        # subject -> {answer: kind}
    for e in ents:
        if e['key'][0] == 'fact' and not e.get('rti') and e['r'] != _RTI_R: ent[e['A']][e['ans']] = 'fact'
    for (A, C), w in W.items():
        if w.get('tw'): ent[w['th']][w['tt']] = ('inf', bool(w['truth']))         # 16 Nov typed writes: subject = the head, answer = the tail
        else: ent[A][C] = ('inf', bool(w['truth']))
    subj = set(ent) & set(K.get('subj', ()))
    sp = [r for r in R if lo_ <= r[0] < hi_ and r[1] in subj and r[7].startswith(r[1])]
    rs = np.random.RandomState(int(cfg.seed) + 6161 + vis); cap = int(getattr(cfg, 'inf_kmfb_n', 4000))
    sp = [sp[i] for i in rs.choice(len(sp), min(cap, len(sp)), replace=False)] if sp else []
    if not sp:
        print(f"  [MEMORY FEEDBACK FROM READING | visit {vis + 1}] no spans name a written subject (documents {lo_}-{hi_})", flush=True); return
    try: sl = int(S_slot(1))
    except Exception: sl = 0
    fb = collections.defaultdict(lambda: [0, 0.0, 0, 0, 0.0]); opened = 0; tot = collections.Counter(); L2 = math.log(2)
    for i0 in range(0, len(sp), 64):
        ch = sp[i0:i0 + 64]; tb = [(r[7] + '.').encode('latin-1', 'replace') for r in ch]
        X, _ = _ws_batch(model, [b + b' ' for b in tb])
        with torch.no_grad():
            lg, H = _kma_run(model, X, sl); lm, opn, _, _ = kma_logits(model, K, X, lg, H, no_rti=bool(int(getattr(cfg, 'inf_kma_nori', 0))) and not int(getattr(cfg, 'inf_kmc_useful_open', 0)))
            l0 = F.log_softmax(lg.float(), -1); l1 = F.log_softmax(lm, -1); mv = lm - lg.float()
        for k, (r, b) in enumerate(zip(ch, tb)):
            runs = [(m.start(), m.end()) for m in _KMI_RUN.finditer(b)]
            if len(runs) < 2: continue
            s0, e0 = runs[-1]
            if s0 < 1 or e0 > l0.shape[1]: continue
            pos = [i for i in range(s0 - 1, e0 - 1) if opn.get((k, i))]
            if not pos: continue
            opened += 1; ix = torch.tensor(pos, device=l0.device); tg = X[k, ix + 1]
            gain = float((l1[k, ix].gather(-1, tg[:, None]) - l0[k, ix].gather(-1, tg[:, None])).sum()) / L2
            A = r[1]; ans = list(ent[A]); first = mv[k, s0 - 1]
            pick = max(ans, key=lambda a: float(first[a.encode('latin-1', 'replace')[0]])) if ans else None
            if pick is None: continue
            name = b[s0:e0].decode('latin-1').rstrip('.'); hit = name == pick or name.startswith(pick) or pick.startswith(name)
            sA = runs[0][1]; posall = [i for i in range(max(0, sA - 1), e0 - 1) if opn.get((k, i))]   # 16 Nov: the WHOLE span after the subject (wording + name)
            ixa = torch.tensor(posall, device=l0.device); tga = X[k, ixa + 1]
            gall = float((l1[k, ixa].gather(-1, tga[:, None]) - l0[k, ixa].gather(-1, tga[:, None])).sum()) / L2
            f = fb[(A, pick)]; f[0] += 1; f[1] += gain; f[2] += int(hit); f[3] += int(not hit); f[4] += gall
            kd = ent[A][pick]; tot[('fact' if kd == 'fact' else ('inference right' if kd[1] else 'inference wrong'), 'gain')] += gain; tot[(kd if kd == 'fact' else 'inf', 'n')] += 1
    inf = [(k, v) for k, v in fb.items() if isinstance(ent[k[0]].get(k[1]), tuple)]
    y = np.array([ent[k[0]][k[1]][1] for k, v in inf]); mg = np.array([v[1] / v[0] for k, v in inf]); cc = np.array([(v[2] - v[3]) / v[0] for k, v in inf])
    auc = lambda s: _cg_auc(s[y], s[~y]) if len(y) >= 10 and y.any() and (~y).any() else float('nan')
    hk = {((w['th'], w['tt']) if w.get('tw') else (A, C)): (A, C) for (A, C), w in W.items()}   # 16 Nov: typed writes are keyed head -> tail here
    nW = sum(1 for k in hk if k[0] in subj); rk = []
    for k9, v in inf:
        w = W.get(hk.get(k9, k9)) or {}
        if w.get('rank') is not None: rk.append((float(w['rank']), v[1] / v[0]))
        if hk.get(k9) in W:                                                    # 16 Nov USEFULNESS: the memory's mean bits saved on this inference's later text -
            sp9 = int(getattr(cfg, 'inf_kmc_useful_span', 0))                 #   on the second name (default) or on the whole span after the subject
            W[hk[k9]]['use_sum'] = W[hk[k9]].get('use_sum', 0.0) + (v[4] if sp9 else v[1]); W[hk[k9]]['use_nsum'] = W[hk[k9]].get('use_nsum', 0) + v[0]   # 16 Nov CUMULATIVE
            W[hk[k9]]['use'] = (W[hk[k9]]['use_sum'] / W[hk[k9]]['use_nsum']) if int(getattr(cfg, 'inf_kmc_useful_cum', 0)) else (v[4] if sp9 else v[1]) / v[0]   #   (queue 45: summed over visits)
            W[hk[k9]]['use_n'] = v[0]; W[hk[k9]]['use_name'] = v[1] / v[0]; W[hk[k9]]['use_span'] = v[4] / v[0]
    from scipy.stats import spearmanr
    rho = spearmanr([a for a, b in rk], [b for a, b in rk])[0] if len(rk) >= 10 else float('nan')
    print(f"  [MEMORY FEEDBACK FROM READING | visit {vis + 1}]{' (inference slots readable here: the usefulness measure, inf_kmc_useful_open; prediction keeps the gate rule)' if int(getattr(cfg, 'inf_kma_nori', 0)) and int(getattr(cfg, 'inf_kmc_useful_open', 0)) else ''} {len(sp)} spans of documents {lo_}-{hi_} naming a written subject; the gate open on the second name in "
          f"{opened} | bits SAVED by the memory on those names (negative = it hurt): facts {tot[('fact', 'gain')]:+.1f} ({tot[('fact', 'n')]} spans), inferences RIGHT "
          f"{tot[('inference right', 'gain')]:+.1f}, WRONG {tot[('inference wrong', 'gain')]:+.1f} ({tot[('inf', 'n')]} spans) || WRITTEN INFERENCES with feedback: {len(inf)} of {nW} "
          f"({len(inf) / max(1, nW):.2f}); AUC right vs wrong (label-free feedback; the current judge 0.64): mean bits saved {auc(mg):.3f}, confirmations - contradictions "
          f"{auc(cc):.3f} | SELECTION: write-time rank vs feedback, Spearman {rho:+.2f} (n {len(rk)})", flush=True)
    thr = float(getattr(cfg, 'inf_kmc_useful_thr', 0.0)); uw = [w for w in W.values() if w.get('use') is not None]
    if uw:                                                                     # 16 Nov USEFULNESS (the keep rule of inf_kmc_useful; labels only score)
        up = [w for w in uw if w['use'] > thr]; un = [w for w in uw if w['use'] <= thr]; jk = [w for w in up if w.get('judged') is not None and not w.get('flag')]
        f9 = lambda L: f"{np.mean([bool(w['truth']) for w in L]):.3f} ({len(L)})" if L else 'nan (0)'
        sn = [w for w in uw if w.get('use_span', 0) > thr]; nn = [w for w in uw if w.get('use_name', 0) > thr]
        print(f"  [USEFULNESS | visit {vis + 1}] measured on {'the WHOLE SPAN' if int(getattr(cfg, 'inf_kmc_useful_span', 0)) else 'the SECOND NAME'} (second name: useful right {f9(nn)}; "
              f"whole span: useful right {f9(sn)}) || written inferences with feedback {len(uw)} of {len(W)}: USEFUL (mean bits saved > {thr:g}) right {f9(up)}, not useful right {f9(un)}, "
              f"no feedback right {f9([w for w in W.values() if w.get('use') is None])} | useful AND judged-kept right {f9(jk)} | NOT HARMFUL AND judged-kept right {f9([w for w in W.values() if w.get('judged') is not None and not w.get('flag') and not (w.get('use') is not None and w['use'] <= thr)])} | (USEFUL OR TYPED-CONFIRMED) AND judged-kept right {f9([w for w in W.values() if w.get('judged') is not None and not w.get('flag') and ((w.get('use') is not None and w['use'] > thr) or float(w.get('cc', 0.0)) > 0)])} | keep rule {('ON, judge AND not harmful (inf_kmc_useful 2)' if int(getattr(cfg, 'inf_kmc_useful', 0)) == 2 else 'ON, judge AND (useful OR typed-confirmed) (inf_kmc_useful 3)' if int(getattr(cfg, 'inf_kmc_useful', 0)) == 3 else 'ON (inf_kmc_useful)') if int(getattr(cfg, 'inf_kmc_useful', 0)) else 'off (readout)'}", flush=True)
