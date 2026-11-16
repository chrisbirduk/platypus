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
        except Exception as e9:
            from platypus.common import _fail
            _fail('inference.kmem:twohop'); print(f"  [INTEGRATED KEYED MEMORY, TWO-HOP] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
    if K.get('gate'):                                                          # ABSTENTION: questions about names the system never read
        from platypus.inference.distill import _DT_REFUSE_SYL
        rels = sorted({e['r'] for e in fs}); g9 = np.random.RandomState(int(cfg.seed) + 5454 + vis); op9 = []
        for k in range(60):
            nm = ''.join(g9.choice(_DT_REFUSE_SYL, 2)).title() + ' ' + ''.join(g9.choice(_DT_REFUSE_SYL, 3)).title()
            pb = f"It is on record that {nm} {rels[g9.randint(len(rels))]} ".encode('latin-1'); X, _ = _ws_batch(model, [pb + b'Xx.'])
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
                lp = F.log_softmax(z, -1); l0 = F.log_softmax(lg, -1); p0 = len(pb) - 1; n0 = min(len(cb[0]), lp.shape[1] - p0); ar = torch.arange(p0, p0 + n0, device=lp.device)
                sc = [float(lp[j, torch.arange(p0, p0 + min(len(c), lp.shape[1] - p0), device=lp.device), X[j, p0 + 1:p0 + 1 + min(len(c), lp.shape[1] - p0)]].mean()) for j, c in enumerate(cb)]
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
            W[hk[k9]]['use'] = (v[4] if sp9 else v[1]) / v[0]; W[hk[k9]]['use_n'] = v[0]; W[hk[k9]]['use_name'] = v[1] / v[0]; W[hk[k9]]['use_span'] = v[4] / v[0]
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
              f"no feedback right {f9([w for w in W.values() if w.get('use') is None])} | useful AND judged-kept right {f9(jk)} | NOT HARMFUL AND judged-kept right {f9([w for w in W.values() if w.get('judged') is not None and not w.get('flag') and not (w.get('use') is not None and w['use'] <= thr)])} | keep rule {('ON, judge AND not harmful (inf_kmc_useful 2)' if int(getattr(cfg, 'inf_kmc_useful', 0)) == 2 else 'ON (inf_kmc_useful)') if int(getattr(cfg, 'inf_kmc_useful', 0)) else 'off (readout)'}", flush=True)
