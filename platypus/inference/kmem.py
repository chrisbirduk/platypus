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
