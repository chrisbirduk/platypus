"""DISTILLATION IN SLEEP (inf_distill, 10 Nov; off by default). The memory's facts are taught to the trunk while it sleeps - the
complementary-learning-systems loop (the hippocampus teaches the cortex). Each sleep, BEFORE the name memory is rebuilt: inf_distill_steps
steps on the trunk's blocks only (heads, private tables, gates untouched), each step one slot: inf_distill_rows questions about stored
facts and conclusions (the name memory's material: live conclusions and name-to-name facts seen at least twice) in SELF-WRITTEN wordings -
some with text between the name and the answer, so the subject has to be carried - none of them a test wording; target = the stored
answer, loss on the answer bytes only; both memories off. PROTECTION: anchor rows (stored real sentences of the same slot) keep the
model's outputs on ordinary text at what they were before this sleep's distillation (KL, weight inf_distill_anchor); a BRAKE measures
the visited domain's and the old domains' BPC before and after and rolls the whole distillation back if either rises by more than
inf_distill_brake. CONTROL: a fixed fifth of the facts (by a hash of the fact) is never taught. READOUT: recall from the trunk (every
memory off) in the TEST wordings for a sample of the taught facts and of the never-taught control, before and after; across sleeps, [FROM THE WEIGHTS, FIXED SET] (with and without the closed-form edits) is the consolidation measure."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import _fail, _HO_TWO

_DT_FORMS = ("In short, {A} {r} ", "Notes on {A}: {A} {r} ", "{A}, as noted before, {r} ", "As for {A}, people say {A} {r} ",
             "Q: {A} {r} whom? A: ", "{A}, whose name came up earlier in the report, {r} ", "From the notes: {A} {r} ")
_DT_PARA = {'works with': 'is a colleague of', 'lives near': 'is a neighbour of', 'reports to': 'is managed by', 'is a partner of': 'is partnered with',
            'trained under': 'was taught by', 'is connected to': 'is tied to'}
_DT_TEST = ("{A} is tied to ", "The one tied to {A} is ", "It is on record that {A} {r} ")   # never taught (checked below)


def _dt_items(ents, rs, n):
    out = []
    for i in rs.choice(len(ents), min(n, len(ents)), replace=False):
        e = ents[int(i)]; f = _DT_FORMS[rs.randint(len(_DT_FORMS))]
        out.append((f.format(A=e['A'], r=e['r']), e['ans'], e))
    return out


def _dt_recall(model, S_slot, ents):
    """open-ended recall from the trunk (memory and store off) in the test wordings: conclusions the standard held-out wordings, facts a
    new frame and a paraphrased relation"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    isl = lambda b: 65 <= b <= 90 or 97 <= b <= 122; r = {'conclusions': [], 'facts': []}
    for e in ents:
        if e['key'][0] == 'conclusion': qs = [w.format(A=e['A']) for w in _HO_TWO]
        else: qs = [f"It is on record that {e['A']} {e['r']} "] + ([f"{e['A']} {_DT_PARA[e['r']]} "] if e['r'] in _DT_PARA else [])
        for q in qs:
            pb = q.encode('latin-1', 'replace'); tb = (e['ans'] + '.').encode('latin-1', 'replace'); X, _ = _ws_batch(model, [pb + tb])
            with torch.no_grad(): lg = _ws_run(model, X, int(S_slot(e['d'])))[0]
            p0 = len(pb) - 1; ps = torch.arange(p0, p0 + len(tb), device=lg.device)
            if int(ps[-1]) >= lg.shape[0]: continue
            top = lg[ps].argmax(-1); tr = torch.tensor(list(tb), device=lg.device)
            r['conclusions' if e['key'][0] == 'conclusion' else 'facts'].append(bool((top[:-1] == tr[:-1]).all()) and not isl(int(top[-1])))
    return {k: (np.mean(v) if v else float('nan'), len(v)) for k, v in r.items()}


def distill_sleep(model, cfg, S_, S_slot, vals, t, ts, olds_):
    """one sleep's distillation (see the module note)"""
    if not int(getattr(cfg, 'inf_distill', 0)): return
    from platypus.inference.closed_form import _ws_batch, _ws_run, _ws_bpc
    from platypus.inference.namemem import _nm_material, _NM
    from platypus.inference.store import _INFM
    t0 = time.time(); vis = int(model.__dict__.get('_visit', 0))
    allm = _nm_material(model, cfg, S_)
    import zlib                                                                 # CONTROL: a fixed fifth of the facts (by a hash of the fact) is never taught,
    ctl = lambda e: zlib.crc32(repr(e['key']).encode()) % 5 == 0               #   so a rise in the taught facts' recall can be told from general change
    ents = [e for e in allm if not ctl(e)]; held = [e for e in allm if ctl(e)]
    if len(ents) < 16: print(f"  [DISTILLATION | visit {vis + 1}] skipped: {len(ents)} facts", flush=True); return
    assert all('tied to' not in f and 'on record' not in f and 'We read' not in f for f in _DT_FORMS)   # no taught wording is a test wording
    rs = np.random.RandomState(int(cfg.seed) + 9191 + vis)
    probe = [ents[int(i)] for i in rs.choice(len(ents), min(32, len(ents)), replace=False)]
    hprobe = [held[int(i)] for i in rs.choice(len(held), min(32, len(held)), replace=False)] if held else []
    steps = int(getattr(cfg, 'inf_distill_steps', 40)); nrow = int(getattr(cfg, 'inf_distill_rows', 16)); lam = float(getattr(cfg, 'inf_distill_anchor', 1.0))
    brake = float(getattr(cfg, 'inf_distill_brake', 0.01)); lr = float(cfg.lr) * float(getattr(cfg, 'inf_distill_lr', 0.01))
    params = [p for b in model.blocks for p in b.parameters() if p.requires_grad]
    snap = [p.detach().clone() for p in params]
    was_nm = _NM['force']; _NM['force'] = False; had = 'force' in _INFM; fz = _INFM.get('force'); _INFM['force'] = False
    try:
        r0 = _dt_recall(model, S_slot, probe); h0 = _dt_recall(model, S_slot, hprobe) if hprobe else None; b0, o0 = _ws_bpc(model, cfg, vals, t, ts, olds_)
        it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
        bysl = {}
        for e in ents: bysl.setdefault(int(S_slot(e['d'])), []).append(e)
        anc = {}                                                               # anchor rows per slot with the teacher's log-probabilities (before distilling)
        for sl in bysl:
            cand = [x for x in real if int(S_slot(x['d'])) == sl]
            if not cand: continue
            pick = [cand[int(i)] for i in rs.choice(len(cand), min(48, len(cand)), replace=False)]
            X, M = _ws_batch(model, [bytes(x['s']) for x in pick])
            with torch.no_grad(): lp = F.log_softmax(_ws_run(model, X, sl).float(), -1)
            anc[sl] = (X, M, lp)
        slots = list(bysl); tried = []; kept = False; ls = []
        for div in (1.0, 3.0, 9.0):                                            # DOSE RETRIES (as the closed-form edit): a dose that trips the brake is undone
            with torch.no_grad():                                              #   and a third of it tried, then a ninth; the first dose that passes is kept
                for p, s0 in zip(params, snap): p.copy_(s0)
            op = torch.optim.Adam(params, lr=lr / div); rsd = np.random.RandomState(int(cfg.seed) + 9192 + vis); ls = []
            for st in range(steps):
                sl = slots[rsd.randint(len(slots))]; items = _dt_items(bysl[sl], rsd, nrow)
                txt = [(p + a + '.').encode('latin-1', 'replace') for p, a, e in items]; X, M = _ws_batch(model, txt)
                lg = _ws_run(model, X, sl, grad_=True)
                rows, pos = [], []
                for r9, (p, a, e) in enumerate(items):
                    p0 = len(p.encode('latin-1', 'replace')) - 1
                    for j in range(len(a) + 1):
                        if p0 + j < lg.shape[1]: rows.append(r9); pos.append(p0 + j)
                rows = torch.tensor(rows, device=X.device); pos = torch.tensor(pos, device=X.device)
                loss = F.cross_entropy(lg[rows, pos], X[rows, pos + 1])
                if sl in anc and lam > 0:
                    Xa, Ma, lpa = anc[sl]; k = torch.tensor(rsd.choice(len(Xa), min(int(getattr(cfg, 'inf_distill_arows', 16)), len(Xa)), replace=False), device=Xa.device)
                    lga = F.log_softmax(_ws_run(model, Xa[k], sl, grad_=True).float(), -1); m = Ma[k][:, 1:lga.shape[1] + 1]
                    kl = ((lpa[k].exp() * (lpa[k] - lga)).sum(-1) * m).sum() / m.sum().clamp(min=1)
                    loss = loss + lam * kl
                op.zero_grad(); loss.backward(); op.step(); ls.append(float(loss.detach()))
                for p in model.parameters():
                    if p.grad is not None: p.grad = None
            b1, o1 = _ws_bpc(model, cfg, vals, t, ts, olds_)
            rt = _dt_recall(model, S_slot, probe); ht = _dt_recall(model, S_slot, hprobe) if hprobe else None   # what this dose taught (before any rollback)
            ok = (b1 - b0) <= brake and (not (o1 == o1) or (o1 - o0) <= brake)
            tried.append((div, b1, o1, ok, rt, ht, ls[0], ls[-1]))
            if ok: kept = True; break
        if not kept:
            with torch.no_grad():
                for p, s0 in zip(params, snap): p.copy_(s0)
        r1 = _dt_recall(model, S_slot, probe); h1 = _dt_recall(model, S_slot, hprobe) if hprobe else None
        f = lambda r: " | ".join(f"{k} {v[0]:.2f} ({v[1]})" for k, v in r.items()) if r else "none"
        tr = " ; ".join(f"lr x{float(getattr(cfg, 'inf_distill_lr', 0.01)) / d:.4g}: loss {l0:.3f} -> {l1:.3f}, BPC visited {b0:.4f} -> {b1:.4f}, old {o0:.4f} -> {o1:.4f}, "
                        f"{'KEPT' if ok else 'over the brake'}; taught facts then {f(rt)}, control then {f(ht)}" for d, b1, o1, ok, rt, ht, l0, l1 in tried)
        print(f"  [DISTILLATION | visit {vis + 1}] {steps} steps x {nrow} questions ({len(ents)} facts and conclusions, {len(_DT_FORMS)} self-written wordings, "
              f"trunk blocks only, anchors x{lam:g}, brake {brake:g}) || {tr} || {'KEPT' if kept else 'ALL DOSES ROLLED BACK'} | from the trunk, test wordings, "
              f"{len(probe)} taught facts: before {f(r0)} -> now {f(r1)} || CONTROL, {len(hprobe)} never-taught facts: before {f(h0)} -> now {f(h1)} | {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        _fail('inference.distill'); print(f"  [DISTILLATION] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
        with torch.no_grad():
            for p, s in zip(params, snap): p.copy_(s)
    finally:
        _NM['force'] = was_nm
        if had: _INFM['force'] = fz
        else: _INFM.pop('force', None)


# ---------------------------------------------------------------- INTERLEAVED (wake) distillation
# 11 Nov options that pair the question rows with protection (each off by default; see distill_wake_step and distill_wake_sleep):
#   inf_distill_wake_params='mlp_out'   only the blocks' mlp_out weights take the question gradient
#   inf_distill_wake_proj=1             anchor projection (A-GEM) against next-byte loss on stored real sentences
#   inf_distill_wake_ondream=1          question rows only on steps that also train on dream rows, weight x the dream interval (same dose)
#   inf_distill_wake_akl=lam            soft anchor: + lam x KL(the model at the last sleep || now) on stored real sentences of OLD slots
#   inf_distill_repair=N                repair replay at each sleep: N steps of that KL on the old slots (previous sleep's predictions) before
#                                       the new pool is built (lr = cfg.lr x inf_distill_repair_lr)
#   inf_distill_wake_spaced=1           spaced rehearsal: a fact the trunk answers in a taught wording at a sleep has its sampling interval
#                                       doubled (to 16), a fact it misses goes back to 1; rows drawn with weight 1/interval
#   inf_distill_wake_selfkl=1           a fact the trunk answers in a taught wording is held by KL to its own answer distribution at that
#                                       sleep instead of pushed by cross-entropy
_ISL = lambda b: 65 <= b <= 90 or 97 <= b <= 122


def _dw_known(model, S_slot, ents):
    """facts the trunk answers in a TAUGHT wording (the first self-written form, never a test wording): repr(key) -> bool"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    out = {}
    for e in ents:
        pb = _DT_FORMS[0].format(A=e['A'], r=e['r']).encode('latin-1', 'replace'); tb = (e['ans'] + '.').encode('latin-1', 'replace')
        X, _ = _ws_batch(model, [pb + tb])
        with torch.no_grad(): lg = _ws_run(model, X, int(S_slot(e['d'])))[0]
        p0 = len(pb) - 1
        if p0 + len(tb) - 1 >= lg.shape[0]: out[repr(e['key'])] = False; continue
        top = lg[p0:p0 + len(tb)].argmax(-1); tr = torch.tensor(list(tb), device=lg.device)
        out[repr(e['key'])] = bool((top[:-1] == tr[:-1]).all()) and not _ISL(int(top[-1]))
    return out


def _dw_cache(model, cfg, S_, S_slot, slots, vis):
    """32 stored real sentences per slot with the model's log-probabilities now (the anchor targets until the next sleep)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
    ra = np.random.RandomState(int(cfg.seed) + 9696 + vis); by = {}
    for i in ra.permutation(len(real)):
        x = real[int(i)]; sl = int(S_slot(x['d']))
        if sl in slots and len(by.setdefault(sl, [])) < 32: by[sl].append(bytes(x['s']))
    C = {}
    for sl, txt in by.items():
        if len(txt) < 4: continue
        X, M = _ws_batch(model, txt)
        with torch.no_grad(): lp = F.log_softmax(_ws_run(model, X, sl).float(), -1).half()
        C[sl] = (X, M, lp)
    return C


def _dw_kl(model, C, sl, rs, k, grad=True):
    from platypus.inference.closed_form import _ws_run
    X, M, lp = C[sl]
    idx = torch.tensor(rs.choice(len(X), min(k, len(X)), replace=False), device=X.device) if k else torch.arange(len(X), device=X.device)
    if grad: lg = F.log_softmax(_ws_run(model, X[idx], sl, grad_=True).float(), -1)
    else:
        with torch.no_grad(): lg = F.log_softmax(_ws_run(model, X[idx], sl).float(), -1)
    T = lg.shape[1]; m = M[idx][:, 1:T + 1].float(); t_ = lp[idx][:, :T].float()
    return ((t_.exp() * (t_ - lg)).sum(-1) * m).sum() / m.sum().clamp(min=1)


def _dw_params(model, cfg):
    P = [p for p in model.parameters() if p.requires_grad]
    only = str(getattr(cfg, 'inf_distill_wake_params', '') or '')
    if only == 'mlp_out':
        keep = set(id(t) for b in model.blocks for t in (b.mlp_out.weight, b.mlp_out.bias))
        return [p for p in P if id(p) in keep], [p for p in P if id(p) not in keep]
    if only: raise ValueError(f"inf_distill_wake_params {only!r}")
    return P, []


def _dw_repair(model, cfg, D, vslot, oslots, vis):
    """REPAIR REPLAY: N steps of KL to the previous sleep's predictions on stored real sentences of the old slots (not the one just visited)"""
    n = int(getattr(cfg, 'inf_distill_repair', 0)); C = (D or {}).get('cache') or {}
    sls = [s for s in oslots if s in C and s != vslot]
    if not n or not sls: return
    t0 = time.time(); P, frz = _dw_params(model, cfg); allp = list(model.parameters()); g_saved = [p.grad for p in allp]
    for p in allp: p.grad = None
    for p in frz: p.requires_grad_(False)
    rs = np.random.RandomState(int(cfg.seed) + 9797 + vis)
    try:
        b = {s: float(_dw_kl(model, C, s, rs, 0, grad=False)) for s in sls}
        op = torch.optim.Adam(P, lr=float(cfg.lr) * float(getattr(cfg, 'inf_distill_repair_lr', 0.1)))
        for st in range(n):
            sl = sls[rs.randint(len(sls))]; loss = _dw_kl(model, C, sl, rs, 8)
            op.zero_grad(); loss.backward(); op.step()
        for p in allp: p.grad = None
        a = {s: float(_dw_kl(model, C, s, rs, 0, grad=False)) for s in sls}
        print(f"  [DISTILLATION, REPAIR REPLAY | visit {vis + 1}] {n} steps on {len(sls)} old slots ({'mlp_out only' if frz else 'all parameters'}): KL to the previous "
              f"sleep's predictions on their stored real sentences {np.mean(list(b.values())):.4f} -> {np.mean(list(a.values())):.4f} | per slot "
              + " ".join(f"{s}:{b[s]:.4f}->{a[s]:.4f}" for s in sls) + f" | {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        _fail('inference.distill:repair'); print(f"  [DISTILLATION, REPAIR REPLAY] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
    finally:
        for p in frz: p.requires_grad_(True)
        for p, g in zip(allp, g_saved): p.grad = g


def distill_wake_sleep(model, cfg, S_, S_slot, vslot=None, oslots=()):
    """INTERLEAVED DISTILLATION (inf_distill_wake > 0; 10 Nov): at each sleep, report the visit just finished (rows used, the mean
    answer loss, recall from the trunk in test wordings for taught facts against the never-taught control fifth) and rebuild the pool of
    question rows for the next visit; during the visit, distill_wake_step adds inf_distill_wake_rows of them to each real-text training step
    (same optimiser - its Fisher whitening, the real-text loss and the dream rows apply - no burst, no separate optimiser)"""
    import zlib
    from platypus.inference.namemem import _nm_material, _NM
    vis = int(model.__dict__.get('_visit', 0)); D = model.__dict__.get('_dtw')
    allm = _nm_material(model, cfg, S_)
    ctl = lambda e: zlib.crc32(repr(e['key']).encode()) % 5 == 0
    ents = [e for e in allm if not ctl(e)]; held = [e for e in allm if ctl(e)]
    f = lambda x: " | ".join(f"{k} {v[0]:.2f} ({v[1]})" for k, v in x.items()) if x else "none"
    was = _NM['force']; _NM['force'] = False
    try:
        if D is not None and D.get('probe'):
            r = _dt_recall(model, S_slot, D['probe']); h = _dt_recall(model, S_slot, D['hprobe']) if D.get('hprobe') else None
            ls = D['losses']; ak = D.get('akl') or []
            print(f"  [DISTILLATION, INTERLEAVED | visit {vis + 1}] last visit: {D['n']} training steps with {int(getattr(cfg, 'inf_distill_wake_rows', 4))} question rows each "
                  f"(weight {float(cfg.inf_distill_wake):g}), answer loss {np.mean(ls[:20]) if ls else float('nan'):.3f} (first 20) -> {np.mean(ls[-20:]) if ls else float('nan'):.3f} (last 20) | "
                  f"from the trunk, test wordings, after the visit: {len(D['probe'])} taught facts {f(r)} || CONTROL, {len(D['hprobe'])} never-taught {f(h)}"
                  + (f" | soft anchor KL during the visit {np.mean(ak[:20]):.4f} (first 20) -> {np.mean(ak[-20:]):.4f} (last 20)" if ak else ""), flush=True)
        st = (D or {}).get('st') or dict(proj=0, conf=0, cut=[])
        if D is not None and int(getattr(cfg, 'inf_distill_wake_proj', 0)) and st['proj']:
            print(f"  [DISTILLATION, INTERLEAVED | visit {vis + 1}] ANCHOR PROJECTION last visit: {st['proj']} steps, question gradient in conflict with the anchor gradient "
                  f"in {st['conf']} ({st['conf'] / max(1, st['proj']):.2f}); share of the question gradient removed when in conflict: mean {np.mean(st['cut']) if st['cut'] else float('nan'):.3f}", flush=True)
        if vslot is not None: _dw_repair(model, cfg, D, int(vslot), [int(s) for s in oslots], vis)
        sp = int(getattr(cfg, 'inf_distill_wake_spaced', 0)); sk = int(getattr(cfg, 'inf_distill_wake_selfkl', 0))
        known = _dw_known(model, S_slot, ents) if (sp or sk) else {}
        iv = dict((D or {}).get('iv') or {})
        if sp:
            for k_, ok in known.items(): iv[k_] = min(16, 2 * iv.get(k_, 1)) if ok else 1
        rows = []; nt = 0
        from platypus.inference.closed_form import _ws_batch, _ws_run
        for e in ents:
            k_ = repr(e['key']); sl = int(S_slot(e['d']))
            for fm in _DT_FORMS:
                pb = fm.format(A=e['A'], r=e['r']).encode('latin-1', 'replace'); ab = (e['ans'] + '.').encode('latin-1', 'replace'); tl = None
                if sk and known.get(k_):
                    X, _ = _ws_batch(model, [pb + ab])
                    with torch.no_grad(): lg = _ws_run(model, X, sl)[0]
                    p0 = len(pb) - 1; n_ = min(len(ab), lg.shape[0] - p0)
                    if n_ > 0: tl = F.log_softmax(lg[p0:p0 + n_].float(), -1).half(); nt += 1
                rows.append((pb, ab, sl, k_, tl))
        if sp or sk:
            nk = sum(known.values())
            print(f"  [DISTILLATION, INTERLEAVED | visit {vis + 1}] TAUGHT-WORDING CHECK: the trunk answers {nk} of {len(known)} taught facts in a taught wording"
                  + (f" | spaced rehearsal: mean interval {np.mean([iv.get(repr(e['key']), 1) for e in ents]):.2f}, at 16: {sum(1 for e in ents if iv.get(repr(e['key']), 1) >= 16)}" if sp else "")
                  + (f" | held by KL to their own answers: {nt} rows" if sk else ""), flush=True)
        rs = np.random.RandomState(int(cfg.seed) + 9393 + vis)
        probe = [ents[int(i)] for i in rs.choice(len(ents), min(32, len(ents)), replace=False)] if ents else []
        hprobe = [held[int(i)] for i in rs.choice(len(held), min(32, len(held)), replace=False)] if held else []
        bysl = {}
        for rw in rows: bysl.setdefault(rw[2], []).append(rw)
        w = {}
        if sp:
            for sl, pool in bysl.items():
                a_ = np.array([1.0 / iv.get(rw[3], 1) for rw in pool]); w[sl] = a_ / a_.sum()
        anc = {}                                                               # ANCHOR SENTENCES per slot (stored real text) for the projection
        if int(getattr(cfg, 'inf_distill_wake_proj', 0)):
            it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]; ra = np.random.RandomState(int(cfg.seed) + 9595 + vis)
            for x in [real[int(i)] for i in ra.permutation(len(real))]:
                sl = int(S_slot(x['d']))
                if len(anc.setdefault(sl, [])) < 64: anc[sl].append(bytes(x['s']))
        cache = None
        if float(getattr(cfg, 'inf_distill_wake_akl', 0.0)) > 0 or int(getattr(cfg, 'inf_distill_repair', 0)):
            cache = _dw_cache(model, cfg, S_, S_slot, set([int(s) for s in oslots] + ([int(vslot)] if vslot is not None else [])), vis)
        model.__dict__['_dtw'] = dict(bysl=bysl, w=w, iv=iv, rs=np.random.RandomState(int(cfg.seed) + 9494 + vis), n=0, losses=[], akl=[], probe=probe, hprobe=hprobe,
                                      anc=anc, cache=cache, st=dict(proj=0, conf=0, cut=[]))
    finally: _NM['force'] = was


def distill_wake_loss(model, cfg):
    """inf_distill_wake_rows question rows (one slot, drawn from the pool, by spaced-rehearsal weight when on) - cross-entropy on the answer
    bytes, or KL to the trunk's own answer at the last sleep for rows of facts it already answers (inf_distill_wake_selfkl); 0.0 when no pool"""
    D = model.__dict__.get('_dtw')
    if not D or not D['bysl']: return 0.0
    try:
        from platypus.inference.closed_form import _ws_batch, _ws_run
        k = int(getattr(cfg, 'inf_distill_wake_rows', 4)); rs = D['rs']; sls = list(D['bysl']); sl = sls[rs.randint(len(sls))]; pool = D['bysl'][sl]
        pw = (D.get('w') or {}).get(sl)
        items = [pool[int(i)] for i in rs.choice(len(pool), min(k, len(pool)), replace=False, p=pw)]
        X, M = _ws_batch(model, [it[0] + it[1] for it in items]); lg = _ws_run(model, X, sl, grad_=True)
        tot = 0.0; cnt = 0; ce = []
        for r9, it in enumerate(items):
            p0 = len(it[0]) - 1; n_ = min(len(it[1]), lg.shape[1] - p0)
            if n_ <= 0: continue
            z = lg[r9, p0:p0 + n_].float(); tl = it[4] if len(it) > 4 else None
            if tl is not None:
                t_ = tl[:n_].float(); tot = tot + (t_.exp() * (t_ - F.log_softmax(z, -1))).sum()
            else:
                c_ = F.cross_entropy(z, X[r9, p0 + 1:p0 + 1 + n_], reduction='sum'); tot = tot + c_; ce.append(float(c_.detach()) / n_)
            cnt += n_
        if not cnt: return 0.0
        loss = tot / cnt; D['n'] += 1; D['losses'].append(float(np.mean(ce)) if ce else float(loss.detach()))
        return loss
    except Exception as e:
        _fail('inference.distill:wake')
        if not D.get('err'): D['err'] = 1; print(f"  [DISTILLATION, INTERLEAVED] a step failed ({type(e).__name__}: {e})"[:300], flush=True)
        return 0.0


def _dw_anchor_loss(model, cfg, D):
    """next-byte cross-entropy on inf_distill_wake_rows stored real sentences of one slot (the anchor for the projection)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    A = D.get('anc') or {}
    if not A: return None
    rs = D['rs']; sls = list(A); sl = sls[rs.randint(len(sls))]; pool = A[sl]
    X, M = _ws_batch(model, [pool[int(i)] for i in rs.choice(len(pool), min(int(getattr(cfg, 'inf_distill_wake_rows', 4)), len(pool)), replace=False)])
    lg = _ws_run(model, X, sl, grad_=True); T = lg.shape[1]; m = M[:, 1:T + 1].float()
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), X[:, 1:T + 1].reshape(-1), reduction='none').view(m.shape)
    return (ce * m).sum() / m.sum().clamp(min=1)


def distill_wake_step(model, cfg, olds=None, dream=None, dk=1):
    """the interleaved question rows' backward, after the real step's backward (gradients add before the optimiser step); options above"""
    D = model.__dict__.get('_dtw')
    if not D: return
    ond = int(getattr(cfg, 'inf_distill_wake_ondream', 0))
    if ond and not dream: return
    w = float(cfg.inf_distill_wake) * (max(1, int(dk)) if ond else 1)
    proj = int(getattr(cfg, 'inf_distill_wake_proj', 0)); akl = float(getattr(cfg, 'inf_distill_wake_akl', 0.0))
    P, frz = _dw_params(model, cfg)
    for p in frz: p.requires_grad_(False)
    try:
        g0 = None
        if proj: g0 = [p.grad for p in P]; [setattr(p, 'grad', None) for p in P]
        dl_ = distill_wake_loss(model, cfg)
        if not torch.is_tensor(dl_):
            if g0 is not None: [setattr(p, 'grad', g) for p, g in zip(P, g0)]
            return
        tot = w * dl_
        C = D.get('cache') or {}
        if akl > 0 and C and olds:
            sls = [int(s) for s in olds if int(s) in C]
            if sls:
                kl = _dw_kl(model, C, sls[D['rs'].randint(len(sls))], D['rs'], int(getattr(cfg, 'inf_distill_wake_rows', 4)))
                tot = tot + akl * kl; D['akl'].append(float(kl.detach()))
        tot.backward()
        if proj:
            gq = [p.grad for p in P]; [setattr(p, 'grad', None) for p in P]
            la = None
            try: la = _dw_anchor_loss(model, cfg, D)
            except Exception: _fail('inference.distill:anchor')
            if la is not None:
                la.backward(); ga = [p.grad for p in P]
                with torch.no_grad():
                    dot = sum(float((q.float() * a.float()).sum()) for q, a in zip(gq, ga) if q is not None and a is not None)
                    nn_ = sum(float(a.float().pow(2).sum()) for a in ga if a is not None); D['st']['proj'] += 1
                    if dot < 0 and nn_ > 0:
                        c = dot / nn_; nq = sum(float(q.float().pow(2).sum()) for q in gq if q is not None)
                        for i, (q, a) in enumerate(zip(gq, ga)):
                            if q is not None and a is not None: gq[i] = q - c * a
                        D['st']['conf'] += 1; D['st']['cut'].append(min(1.0, (dot * dot / nn_) / max(nq, 1e-30)))   # share of |gq|^2 removed
            for p, g, q in zip(P, g0, gq):
                p.grad = q if g is None else (g if q is None else g + q)
    finally:
        for p in frz: p.requires_grad_(True)
