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
        spd = int(getattr(cfg, 'inf_distill_spindle', 0)) and D is not None and D.get('bysl'); qls = []
        for st in range(n):
            sl = sls[rs.randint(len(sls))]; loss = _dw_kl(model, C, sl, rs, 8)
            if spd:
                ql = distill_wake_loss(model, cfg)
                if torch.is_tensor(ql): loss = loss + float(cfg.inf_distill_wake) * ql; qls.append(float(ql.detach()))
            op.zero_grad(); loss.backward(); op.step()
        for p in allp: p.grad = None
        a = {s: float(_dw_kl(model, C, s, rs, 0, grad=False)) for s in sls}
        print(f"  [DISTILLATION, REPAIR REPLAY | visit {vis + 1}] {n} steps on {len(sls)} old slots ({'mlp_out only' if frz else 'all parameters'}): KL to the previous "
              f"sleep's predictions on their stored real sentences {np.mean(list(b.values())):.4f} -> {np.mean(list(a.values())):.4f} | per slot "
              + " ".join(f"{s}:{b[s]:.4f}->{a[s]:.4f}" for s in sls)
              + (f" | SPINDLE: question rows inside the burst, answer loss {np.mean(qls[:10]):.3f} (first 10) -> {np.mean(qls[-10:]):.3f} (last 10)" if qls else "") + f" | {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        _fail('inference.distill:repair'); print(f"  [DISTILLATION, REPAIR REPLAY] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
    finally:
        for p in frz: p.requires_grad_(True)
        for p, g in zip(allp, g_saved): p.grad = g


def _dw_ever(model, cfg, S_slot, ents, held, D, vis):
    """EVER-TAUGHT LEDGER (readout, 11 Nov): every fact that has ever been in the question pool (and every control fact ever held back) is
    kept with the visit it entered; at each sleep, recall from the trunk in the test wordings over a fixed sample of the ledger (up to
    inf_distill_ever facts, chosen by a hash of the fact so the sample is stable), split by whether the fact is STILL in the pool or has LEFT it
    (crowded out by the memory cap, superseded or retracted) and by age (visits since it entered); with the ledger and pool sizes.
    Recall over the current pool alone can be flattered when the pool keeps the easiest facts."""
    import zlib
    nmax = int(getattr(cfg, 'inf_distill_ever', 256))
    if nmax <= 0: return
    L = model.__dict__.setdefault('_dw_ledger', {})
    now = {repr(e['key']) for e in ents}; nowc = {repr(e['key']) for e in held}
    for grp, es in (('t', ents), ('c', held)):
        for e in es: L.setdefault(repr(e['key']), dict(e=e, v=vis, g=grp))
    if not L: return
    pick = sorted(L, key=lambda k: zlib.crc32(k.encode()))
    tk = [k for k in pick if L[k]['g'] == 't'][:nmax]; ck = [k for k in pick if L[k]['g'] == 'c'][:max(32, nmax // 4)]
    tk1 = [k for k in tk if vis - L[k]['v'] >= 1]                              # entered at an earlier sleep, so trained for at least one visit
    f = lambda x: " | ".join(f"{k_} {v_[0]:.2f} ({v_[1]})" for k_, v_ in x.items() if v_[1]) or "none"
    def grp_(keys, cur):
        out = []
        for nm, sel in (('still in the pool', lambda k: k in cur), ('left the pool', lambda k: k not in cur)):
            ks = [k for k in keys if sel(k)]
            if ks: out.append(f"{nm} ({len(ks)}): {f(_dt_recall(model, S_slot, [L[k]['e'] for k in ks]))}")
        for nm, lo, hi in (('ENTERED AT THIS SLEEP (not yet taught)', 0, 0), ('taught 1 visit', 1, 1), ('2-3 visits', 2, 3), ('4+ visits', 4, 999)):
            ks = [k for k in keys if lo <= vis - L[k]['v'] <= hi]
            if ks: out.append(f"{nm} ({len(ks)}): {f(_dt_recall(model, S_slot, [L[k]['e'] for k in ks]))}")
        return " || ".join(out)
    try:
        nt = sum(1 for v in L.values() if v['g'] == 't'); nc = sum(1 for v in L.values() if v['g'] == 'c')
        print(f"  [DISTILLATION, EVER TAUGHT | visit {vis + 1}] ledger: {nt} facts and conclusions ever taught ({len(now)} in the pool now), {nc} control ever held back "
              f"({len(nowc)} now) | from the trunk, test wordings, sample of {len(tk)} ever in the pool: TAUGHT FOR AT LEAST ONE VISIT ({len(tk1)}) "
              f"{f(_dt_recall(model, S_slot, [L[k]['e'] for k in tk1])) if tk1 else 'none'} || "
              + grp_(tk, now) + f" ||| CONTROL, sample of {len(ck)} ever held back: ALL {f(_dt_recall(model, S_slot, [L[k]['e'] for k in ck]))}", flush=True)
    except Exception as e:
        _fail('inference.distill:ever'); print(f"  [DISTILLATION, EVER TAUGHT] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _dw_conc22(model, cfg, S_, S_slot, ents, held, vis):
    """CONCLUSIONS 2 x 2 (readout, 11 Nov): live conclusions TAUGHT by the question rows vs NEVER TAUGHT (the hashed control fifth), recall
    from the trunk in the held-out wordings with the closed-form edits IN PLACE and SUBTRACTED (the accumulated edit model._cf_E removed
    from blocks[inf_cf_layer].mlp_out for the measurement, as [FROM THE WEIGHTS, FIXED SET]); never-taught conclusions also split by
    whether BOTH their premise facts (A - bridge, bridge - C) are in the taught pool - composition from taught premises. Edits explain
    it if both rows fall to ~0 without them; replay if both survive equally; distillation adds only if the taught row survives better."""
    from platypus.inference.namemem import _nm_layer
    tc = [e for e in ents if e['key'][0] == 'conclusion']; cc = [e for e in held if e['key'][0] == 'conclusion']
    if not tc and not cc: return
    tf = {(e['A'], e['ans']) for e in ents if e['key'][0] == 'fact'}
    br = {}
    for c in S_.get('conc', []):
        if isinstance(c.get('A'), str) and isinstance(c.get('C'), str) and isinstance(c.get('B'), str): br[(c['A'], c['C'])] = c['B']
    def prem(e):
        B = br.get((e['A'], e['ans']))
        return B is not None and ((e['A'], B) in tf or (B, e['A']) in tf) and ((B, e['ans']) in tf or (e['ans'], B) in tf)
    cp = [e for e in cc if prem(e)]; cn = [e for e in cc if not prem(e)]
    E = model.__dict__.get('_cf_E'); mo = _nm_layer(model, cfg)
    g = lambda r: f"{r['conclusions'][0]:.2f} ({r['conclusions'][1]})" if r.get('conclusions', (0, 0))[1] else "-"
    def ask():
        return [g(_dt_recall(model, S_slot, x)) if x else "-" for x in (tc, cc, cp, cn)]
    try:
        a_ = ask(); b_ = None
        if E is not None and mo is not None and tuple(E.shape) == tuple(mo.weight.shape):
            W0 = mo.weight.detach().clone()
            try:
                with torch.no_grad(): mo.weight.copy_((W0.float() - E.float().to(W0.device)).to(W0.dtype))
                b_ = ask()
            finally:
                with torch.no_grad(): mo.weight.copy_(W0)
        nb = b_ if b_ is not None else ["no edits"] * 4
        print(f"  [DISTILLATION, CONCLUSIONS 2x2 | visit {vis + 1}] from the trunk, held-out wordings (n = questions) | edits IN PLACE -> SUBTRACTED: "
              f"TAUGHT {a_[0]} -> {nb[0]} | NEVER TAUGHT {a_[1]} -> {nb[1]} || never taught, BOTH PREMISES TAUGHT {a_[2]} -> {nb[2]} | "
              f"premises not both taught {a_[3]} -> {nb[3]} | conclusions: {len(tc)} taught, {len(cc)} never taught ({len(cp)} with both premises taught)", flush=True)
    except Exception as e:
        _fail('inference.distill:conc22'); print(f"  [DISTILLATION, CONCLUSIONS 2x2] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _dw_pd_probe(model, cfg, S_slot, ents, held, ctx_, vis):
    """SOURCE-IN-CONTEXT PROBE (readout, 11 Nov; inf_distill_pd_probe): does the trunk answer better when the fact's stored source
    sentence comes before the question? (What prompt distillation needs: a teacher better than the student.) Facts only, up to 64 taught
    and 64 never-taught (the control fifth - never trained, so with vs without context is pure reading from context), in a TAUGHT wording
    and a TEST wording (new frame): open-ended greedy answer exact, without -> with the sentence; answer bytes top-1; mean KL(with || without)
    per answer byte (the signal soft targets would carry)."""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    from platypus.inference.jepa import cfg_seq_
    L0 = int(cfg_seq_(model)); rs = np.random.RandomState(int(cfg.seed) + 7777 + vis)
    def pick(es):
        es = [e for e in es if e['key'][0] == 'fact' and repr(e['key']) in ctx_]
        return [es[int(i)] for i in rs.choice(len(es), min(64, len(es)), replace=False)] if es else []
    def one(es, wf):
        r = dict(n=0, ex0=0, ex1=0, b=0, b0=0, b1=0, kl=0.0)
        for e in es:
            q = wf(e).encode('latin-1', 'replace'); a = (e['ans'] + '.').encode('latin-1', 'replace'); cx = ctx_[repr(e['key'])]
            room = L0 - len(q) - len(a) - 1
            if room <= 8: continue
            cx = cx[-room:]
            X, _ = _ws_batch(model, [q + a, cx + q + a])
            with torch.no_grad(): lg = _ws_run(model, X, int(S_slot(e['d']))).float()
            p0 = len(q) - 1; p1 = len(cx) + p0; n_ = len(a)
            if p1 + n_ - 1 >= lg.shape[1]: continue
            z0 = lg[0, p0:p0 + n_]; z1 = lg[1, p1:p1 + n_]; tr = torch.tensor(list(a), device=lg.device)
            t0 = z0.argmax(-1); t1 = z1.argmax(-1); isl = lambda b: 65 <= b <= 90 or 97 <= b <= 122
            r['n'] += 1; r['b'] += n_ - 1
            r['ex0'] += int(bool((t0[:-1] == tr[:-1]).all()) and not isl(int(t0[-1]))); r['ex1'] += int(bool((t1[:-1] == tr[:-1]).all()) and not isl(int(t1[-1])))
            r['b0'] += int((t0[:-1] == tr[:-1]).sum()); r['b1'] += int((t1[:-1] == tr[:-1]).sum())
            l1 = F.log_softmax(z1, -1); r['kl'] += float((l1.exp() * (l1 - F.log_softmax(z0, -1))).sum(-1)[:-1].sum())
        return r
    fmt = lambda r: (f"exact {r['ex0'] / r['n']:.2f} -> {r['ex1'] / r['n']:.2f}, answer bytes {r['b0'] / max(1, r['b']):.2f} -> {r['b1'] / max(1, r['b']):.2f}, KL {r['kl'] / max(1, r['b']):.3f} (n {r['n']})" if r['n'] else "none")
    try:
        tf = lambda e: _DT_FORMS[0].format(A=e['A'], r=e['r']); nf = lambda e: f"It is on record that {e['A']} {e['r']} "
        T, C = pick(ents), pick(held)
        print(f"  [SOURCE IN CONTEXT | visit {vis + 1}] the trunk answers WITHOUT -> WITH the fact's stored source sentence before the question (window {L0}) || "
              f"TAUGHT facts, taught wording: {fmt(one(T, tf))} | test wording: {fmt(one(T, nf))} || NEVER-TAUGHT facts (reading from context alone), "
              f"taught wording: {fmt(one(C, tf))} | test wording: {fmt(one(C, nf))}", flush=True)
    except Exception as e:
        _fail('inference.distill:pdprobe'); print(f"  [SOURCE IN CONTEXT] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


_BRG = {'held': set(), 'anch': set()}


def _dw_bridge(cfg, S_, ents, held, vis):
    """BRIDGE ANCHORING (11 Nov; inf_distill_bridge = N descriptors; after BRACE, NeurIPS 2026): two-hop recall from injected facts fails when
    the bridge entity B is represented differently as the ANSWER of hop 1 (A - B) and as the SUBJECT of hop 2 (B - C); reusing the same
    descriptors of B in both hops' training text aligned it (composition up to 0.78 at 1.7-14B). For each live conclusion's bridge B: up to N
    TAUGHT facts about B whose other name is not a chain partner of B (never A or C of any chain through B, so no row holds both hops of a
    chain), or else stored sentence clips naming B and no partner, fixed per B by a hash; bridges are those of live conclusions AND of
    every chain of two taught facts (the chains the COMPOSITION PROBE asks); every question row with B as answer gets one after the answer (trained bytes), every row with B as
    subject gets one before the question. Never a control fact. The test: [DISTILLATION, CONCLUSIONS 2x2] never-taught conclusions with
    both premises taught, edits subtracted (inf_distill_conc_ctl 1 holds every conclusion back)."""
    import zlib
    n = int(getattr(cfg, 'inf_distill_bridge', 0)); part = {}
    if int(getattr(cfg, 'inf_distill_bridge_syn', 0)):                        # 11 Nov SYNTHETIC, HALF: the bridges of the HELD-OUT chains (inf_kgx_hold: never
        from platypus.inference.probes import inf_probes                      #   concluded, stored, replayed or edited); half of them, by a hash of the name,
        N9 = int(getattr(cfg, 'inf_kgx_hold', 0)); held_b = set()            #   get one fixed INVENTED attribute as descriptor ('B keeps the green ledger.'),
        if N9 > 0:                                                            #   the other half none: anchored vs unanchored held-out chains in one run
            for i9, c9 in enumerate(inf_probes(cfg)):
                if i9 % N9 == N9 - 1:
                    nm9 = list(c9['names'])
                    for k9 in range(1, len(nm9) - 1): held_b.add(nm9[k9])
        adj = ('green', 'silver', 'oak', 'copper', 'amber', 'blue', 'iron', 'linen', 'stone', 'cedar')
        nou = ('ledger', 'lantern', 'compass', 'satchel', 'kettle', 'harp', 'anchor', 'quill', 'clock', 'ribbon')
        anch = {B for B in held_b if zlib.crc32(B.encode('latin-1', 'replace')) % 2 == 0}
        out = {}
        for B in anch:
            h9 = zlib.crc32(('d|' + B).encode('latin-1', 'replace'))
            out[B] = [f"{B} keeps the {adj[h9 % 10]} {nou[(h9 // 10) % 10]}.".encode('latin-1', 'replace')]
        _BRG['held'] = held_b; _BRG['anch'] = anch
        fs = [e for e in ents if e['key'][0] == 'fact']
        h1 = sum(1 for e in fs if e['ans'] in out); h2 = sum(1 for e in fs if e['A'] in out)
        print(f"  [DISTILLATION, BRIDGE | visit {vis + 1}] SYNTHETIC, HALF: {len(held_b)} bridges of held-out chains (every {N9}th chain never concluded), "
              f"{len(anch)} anchored with an invented attribute, {len(held_b) - len(anch)} not | taught facts anchored: {h1} with the bridge as answer, {h2} as subject"
              + (f" | e.g. {sorted(out)[0]}: {out[sorted(out)[0]][0].decode('latin-1')}" if out else ""), flush=True)
        return out
    for c in S_.get('conc', []):
        if c.get('freed') or c.get('retracted'): continue
        if isinstance(c.get('A'), str) and isinstance(c.get('B'), str) and isinstance(c.get('C'), str): part.setdefault(c['B'], set()).update((c['A'], c['C']))
    fs = [e for e in ents if e['key'][0] == 'fact']; out = {}; nsent = 0; nlive = len(part)
    sub = {}
    for e in fs: sub.setdefault(e['A'], set()).add(e['ans'])
    for e in fs:                                                               # ALSO every bridge of a chain of TAUGHT facts (A - B, B - C):
        if e['ans'] in sub:                                                    #   partners are every name that reaches B or that B reaches
            for C in sub[e['ans']]:
                if C != e['A']: part.setdefault(e['ans'], set()).update((e['A'], C))
    real = None; byname = None
    for B, pt in part.items():
        cand = sorted({f"{e['A']} {e['r']} {e['ans']}." for e in fs if (e['A'] == B and e['ans'] not in pt) or (e['ans'] == B and e['A'] not in pt)})
        if not cand:                                                           # FALLBACK: a stored real sentence naming B and no chain partner of B,
            if byname is None:                                                 #   clipped to 48 bytes around B (the local harness's bridges rarely
                import re as _re                                               #   have a third taught fact); one pass over the stored sentences,
                byname = {}; want = set(part)                                  #   indexed by the capitalised words they contain
                for x in S_.get('it', []):
                    if x.get('k') != 'real': continue
                    b9 = x['s']; b9 = bytes(int(v) & 255 for v in b9.tolist()) if hasattr(b9, 'tolist') and not isinstance(b9, (bytes, bytearray)) else (b9.encode('latin-1', 'replace') if isinstance(b9, str) else bytes(b9))
                    t = b9.decode('latin-1')
                    for w in set(_re.findall(r"[A-Z][a-z]+", t)) & want:
                        if len(byname.setdefault(w, [])) < 64: byname[w].append(t)
            bb = []
            for t in byname.get(B, []):
                i9 = t.find(B)
                if i9 < 0 or any(q in t for q in pt): continue
                a9 = max(0, i9 - 12); bb.append(t[a9:a9 + 48].strip())
            cand = sorted(set(bb)); nsent += 1 if cand else 0
        if not cand: continue
        cand.sort(key=lambda t: zlib.crc32((B + '|' + t).encode('latin-1', 'replace')))
        out[B] = [t.encode('latin-1', 'replace') for t in cand[:n]]
    h1 = sum(1 for e in fs if e['ans'] in out); h2 = sum(1 for e in fs if e['A'] in out)
    print(f"  [DISTILLATION, BRIDGE | visit {vis + 1}] {len(part)} bridge names ({nlive} in live conclusions, the rest in chains of taught facts), {len(out)} with a descriptor ({len(out) - nsent} a taught fact about the bridge, "
          f"{nsent} a stored sentence clip; no chain partner; up to {n} each, mean {np.mean([len(v) for v in out.values()]) if out else 0:.2f}) | taught facts anchored: {h1} with the bridge as answer, "
          f"{h2} as subject" + (f" | e.g. {list(out)[0]}: {out[list(out)[0]][0].decode('latin-1')}" if out else ""), flush=True)
    return out


def _dw_ns_probe(model, cfg, S_, S_slot, ents, vslot, oslots, vis):
    """NULL-SPACE PROBE (readout, 11 Nov; inf_distill_ns_probe): could the question rows be confined to directions the old domains do not
    use (AlphaEdit-style null-space projection)? For each block's mlp_out (the weights the question rows write): the covariance of its
    INPUTS on 32 stored real sentences per OLD slot (not the one just visited); the number of input directions holding 90 / 99 / 99.9% of
    that energy (of H); and the share of a question-row gradient (up to 64 taught facts, taught wordings) that lies INSIDE those
    directions - what a hard projection at that cut-off would remove (an isotropic update would lose k / H). Training is not changed."""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    sls = [int(s) for s in oslots if int(s) != int(vslot)] if vslot is not None else []
    fs = [e for e in ents if e['key'][0] == 'fact']
    if not sls or not fs: return
    it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
    ra = np.random.RandomState(int(cfg.seed) + 9898 + vis); by = {}
    for i in ra.permutation(len(real)):
        x = real[int(i)]; sl = int(S_slot(x['d']))
        if sl in sls and len(by.setdefault(sl, [])) < 32: by[sl].append(bytes(x['s']))
    blocks = list(model.blocks); C = [None] * len(blocks); hs = []; cur = {}
    def mk(j):
        def h(mod, inp, out):
            x = inp[0].detach().float(); m = cur.get('m')
            x = x.reshape(-1, x.shape[-1]) if m is None else x[m[:, :x.shape[1]] > 0]
            c = x.t() @ x; C[j] = c if C[j] is None else C[j] + c
        return h
    allp = list(model.parameters()); g_saved = [p.grad for p in allp]; rq = [p.requires_grad for p in allp]
    try:
        for j, b in enumerate(blocks): hs.append(b.mlp_out.register_forward_hook(mk(j)))
        npos = 0
        for sl, txt in by.items():
            if len(txt) < 4: continue
            X, M = _ws_batch(model, txt); cur['m'] = M[:, :-1]; npos += int(M[:, :-1].sum())
            with torch.no_grad(): _ws_run(model, X, sl)
        for h in hs: h.remove()
        hs = []; cur.clear()
        if npos == 0: return
        rs = np.random.RandomState(int(cfg.seed) + 9899 + vis); pick = [fs[int(i)] for i in rs.choice(len(fs), min(64, len(fs)), replace=False)]
        for p in allp: p.grad = None
        W = [b.mlp_out.weight for b in blocks]
        for w in W: w.requires_grad_(True)
        bys = {}
        for e in pick: bys.setdefault(int(S_slot(e['d'])), []).append(e)
        for sl, es in bys.items():
            rows = [(_DT_FORMS[int(rs.randint(len(_DT_FORMS)))].format(A=e['A'], r=e['r']).encode('latin-1', 'replace'), (e['ans'] + '.').encode('latin-1', 'replace')) for e in es]
            X, _ = _ws_batch(model, [a + b for a, b in rows]); lg = _ws_run(model, X, sl, grad_=True); loss = 0.0
            for r9, (a, b) in enumerate(rows):
                p0 = len(a) - 1; n_ = min(len(b), lg.shape[1] - p0)
                if n_ > 0: loss = loss + F.cross_entropy(lg[r9, p0:p0 + n_].float(), X[r9, p0 + 1:p0 + 1 + n_], reduction='sum')
            if torch.is_tensor(loss): loss.backward()
        res = []
        for j, w in enumerate(W):
            if w.grad is None or C[j] is None: continue
            G = w.grad.detach().float(); ev, V = torch.linalg.eigh(C[j].double()); ev = ev.flip(0).clamp_min(0); V = V.flip(1).float()
            cs = torch.cumsum(ev, 0) / ev.sum().clamp_min(1e-12); H = int(ev.numel()); gn = float((G * G).sum()) + 1e-30; r = []
            for q in (0.9, 0.99, 0.999):
                k = int((cs < q).sum()) + 1; GU = G @ V[:, :k]; r.append((k, float((GU * GU).sum()) / gn))
            res.append((j, H, r))
        if not res: return
        fmt = lambda r: " / ".join(f"{k}" for k, _ in r); fs_ = lambda r: " / ".join(f"{v:.2f}" for _, v in r)
        mean = [np.mean([x[2][i][1] for x in res]) for i in range(3)]; iso = [np.mean([x[2][i][0] / x[1] for x in res]) for i in range(3)]
        print(f"  [NULL SPACE PROBE | visit {vis + 1}] mlp_out inputs on old slots {sls} ({npos} byte positions) || per block: input directions holding 90 / 99 / 99.9% "
              f"of the old-domain energy (of H) -> share of a question-row gradient ({len(pick)} taught facts) INSIDE them: "
              + " | ".join(f"block {j}: {fmt(r)} of {H} -> {fs_(r)}" for j, H, r in res)
              + f" || MEAN share inside {mean[0]:.2f} / {mean[1]:.2f} / {mean[2]:.2f} (an isotropic update: {iso[0]:.2f} / {iso[1]:.2f} / {iso[2]:.2f}); "
              f"a hard projection at that cut-off keeps {1 - mean[0]:.2f} / {1 - mean[1]:.2f} / {1 - mean[2]:.2f} of the fact update", flush=True)
    except Exception as e:
        _fail('inference.distill:nsprobe'); print(f"  [NULL SPACE PROBE] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
    finally:
        for h in hs: h.remove()
        for p, g, q in zip(allp, g_saved, rq): p.grad = g; p.requires_grad_(q)


def _fu_init(model, cfg):
    """FACT UNITS (11 Nov; inf_distill_fu = units added per block at each sleep; neurogenesis): a reserve of inf_distill_fu_max units beside each
    block's MLP, none active at first. Unit j: u_j = relu(g_j (cos(hn, K_j) - tau)), output V_j (zero at birth: no disruption). K_j is fixed
    at birth to the mean normalised MLP input at the answer positions of one fact's question (three taught wordings) - the unit fires on
    that fact's context and, by the cosine threshold, little elsewhere; V and g are trained by the question rows ONLY (the trunk's own
    weights take no question gradient in this mode). Kept outside model.parameters(): the main optimiser never sees them."""
    from platypus.layers import _FU
    if model.__dict__.get('_fu') is not None: return model.__dict__['_fu']
    dev = model.embed.weight.device; fm = int(getattr(cfg, 'inf_distill_fu_max', 1024)); g0 = float(getattr(cfg, 'inf_distill_fu_g', 4.0))
    P = {}
    for b in model.blocks:
        d = int(b.d_model)
        P[id(b)] = dict(K=torch.zeros(fm, d, device=dev), g=torch.full((fm,), g0, device=dev), V=torch.zeros(fm, d, device=dev), n=0,
                        mu=torch.zeros(d, device=dev), t=torch.full((fm,), float(getattr(cfg, 'inf_distill_fu_tau', 0.5)), device=dev), mu_set=False)
    _FU['P'] = P; _FU['on'] = True; _FU['tau'] = float(getattr(cfg, 'inf_distill_fu_tau', 0.5))
    ts = [t for v in P.values() for t in (v['V'], v['g'])]
    st = dict(P=P, map={}, opt=torch.optim.Adam(ts, lr=float(cfg.lr) * float(getattr(cfg, 'inf_distill_fu_lr', 3.0))), steps=0, ql=[])
    model.__dict__['_fu'] = st
    return st


def _fu_old_sents(cfg, S_, S_slot, sls, seed, n=32):
    it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
    ra = np.random.RandomState(seed); by = {}
    for i in ra.permutation(len(real)):
        x = real[int(i)]; sl = int(S_slot(x['d']))
        if sl in sls and len(by.setdefault(sl, [])) < n: by[sl].append(bytes(x['s']))
    return by


def _fu_sleep(model, cfg, S_, S_slot, ents, held, vslot, oslots, vis):
    """FACT UNITS at a sleep: report last visit (question steps, answer loss, firing on OLD-DOMAIN text vs on taught questions and the
    never-taught control's questions), then give each new taught fact (no unit yet; up to inf_distill_fu per block) its own unit"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    from platypus.layers import _FU
    st = _fu_init(model, cfg); P = st['P']; blocks = list(model.blocks); fm = int(getattr(cfg, 'inf_distill_fu_max', 1024))
    def fire(groups):
        _FU['st'] = {}
        try:
            for sl, txt in groups:
                if not txt: continue
                X, _ = _ws_batch(model, txt)
                with torch.no_grad(): _ws_run(model, X, sl)
            r = [v for v in _FU['st'].values()]
        finally: _FU['st'] = None
        if not r: return None
        a = sum(x[0] for x in r) / max(1.0, sum(x[2] for x in r)); m = sum(x[1] for x in r) / max(1.0, sum(x[2] for x in r))
        return a, m
    try:
        n0 = P[id(blocks[0])]['n']
        if n0 > 0:
            sls = [int(s) for s in oslots if vslot is None or int(s) != int(vslot)]
            old = fire(list(_fu_old_sents(cfg, S_, S_slot, sls, int(cfg.seed) + 7171 + vis).items())) if sls else None
            vis_ = fire(list(_fu_old_sents(cfg, S_, S_slot, [int(vslot)], int(cfg.seed) + 7172 + vis).items())) if vslot is not None else None
            tq = [e for e in ents if repr(e['key']) in st['map']][:64]; nf = lambda e: f"It is on record that {e['A']} {e['r']} "
            byq = lambda es, wf: [(int(S_slot(e['d'])), [(wf(e) + e['ans'] + '.').encode('latin-1', 'replace')]) for e in es]
            q1 = fire(byq(tq, lambda e: _DT_FORMS[0].format(A=e['A'], r=e['r']))); q2 = fire(byq(tq, nf))
            hq = fire(byq([e for e in held if e['key'][0] == 'fact'][:64], nf))
            fmt = lambda r: f"{r[0]:.3f} of positions (mean drive {r[1]:.4f})" if r else "-"
            ql = st['ql']
            print(f"  [FACT UNITS | visit {vis + 1}] {n0} units per block (cap {fm}), {len(st['map'])} facts with a unit | last visit: {st['steps']} question steps "
                  f"(fact units only; the trunk takes no question gradient), answer loss {np.mean(ql[:20]) if ql else float('nan'):.3f} (first 20) -> {np.mean(ql[-20:]) if ql else float('nan'):.3f} (last 20) || "
                  f"ANY UNIT FIRING: old-domain text {fmt(old)} | the visited domain's text {fmt(vis_)} | taught facts' questions, taught wording {fmt(q1)}, TEST wording {fmt(q2)} | "
                  f"never-taught facts' questions {fmt(hq)} || |V| mean {float(torch.stack([P[id(b)]['V'][:n0].norm(dim=-1).mean() for b in blocks]).mean()):.3f}", flush=True)
        st['steps'] = 0; st['ql'] = []
        new = [e for e in ents if e['key'][0] == 'fact' and repr(e['key']) not in st['map']]
        k = min(int(getattr(cfg, 'inf_distill_fu', 0)), fm - P[id(blocks[0])]['n'], len(new))
        if k <= 0: return
        import zlib
        new.sort(key=lambda e: zlib.crc32(repr(e['key']).encode()))
        new = new[:k]; acc = {id(b): [] for b in blocks}
        real = _fu_old_sents(cfg, S_, S_slot, set(int(S_slot(x['d'])) for x in S_.get('it', [])[:20000] if x.get('k') == 'real'), int(cfg.seed) + 7373 + vis, n=16)
        rcap = {id(b): [] for b in blocks}                                     # MLP inputs on ordinary stored text (centring and thresholds)
        for sl, txt in real.items():
            if not txt: continue
            X, M = _ws_batch(model, txt); _FU['cap'] = {}
            try:
                with torch.no_grad(): _ws_run(model, X, sl)
                cap = dict(_FU['cap'])
            finally: _FU['cap'] = None
            m9 = M[:, :-1] > 0
            for b in blocks:
                h = cap.get(id(b))
                if h is not None: rcap[id(b)].append(h[m9[:, :h.shape[1]]].float())
        for b in blocks:                                                       #   CENTRE once, at the first births: the MLP inputs share a large common
            Pb = P[id(b)]                                                      #   direction, so raw cosines fire everywhere (first local run: 0.36-0.48 of text)
            if not Pb['mu_set'] and rcap[id(b)]:
                Pb['mu'] = torch.cat(rcap[id(b)]).mean(0); Pb['mu_set'] = True
        for e in new:
            txt = [(fm_.format(A=e['A'], r=e['r']) + e['ans'] + '.').encode('latin-1', 'replace') for fm_ in _DT_FORMS[:3]]
            pl = [len(fm_.format(A=e['A'], r=e['r']).encode('latin-1', 'replace')) for fm_ in _DT_FORMS[:3]]
            X, _ = _ws_batch(model, txt); _FU['cap'] = {}
            try:
                with torch.no_grad(): _ws_run(model, X, int(S_slot(e['d'])))
                cap = dict(_FU['cap'])
            finally: _FU['cap'] = None
            na = len(e['ans'])
            for b in blocks:
                h = cap.get(id(b))
                if h is None: continue
                rows = []
                for r9, p0 in enumerate(pl):
                    seg = h[r9, p0 - 1:min(p0 - 1 + na, h.shape[1])].float() - P[id(b)]['mu']
                    if seg.shape[0]: rows.append(F.normalize(seg, dim=-1).mean(0))
                acc[id(b)].append(F.normalize(torch.stack(rows).mean(0), dim=-1) if rows else None)
        q9 = float(getattr(cfg, 'inf_distill_fu_q', 0.99)); tmin = float(getattr(cfg, 'inf_distill_fu_tau', 0.5)); tl9 = []
        for b in blocks:                                                       # PER-UNIT THRESHOLD: the q-quantile of the unit's cosine on ordinary stored
            Pb = P[id(b)]; j0 = Pb['n']                                        #   text (at least tau), so at birth it fires on at most 1 - q of text positions
            R9 = F.normalize(torch.cat(rcap[id(b)]) - Pb['mu'], dim=-1) if rcap[id(b)] else None
            for i, kv in enumerate(acc[id(b)]):
                if kv is None: continue
                Pb['K'][j0 + i] = kv; t9 = tmin
                if R9 is not None and R9.shape[0] > 50: t9 = max(tmin, float(torch.quantile((R9 @ kv).float(), q9)))
                Pb['t'][j0 + i] = t9; tl9.append(t9)
            Pb['n'] = j0 + len(new)
        for i, e in enumerate(new): st['map'][repr(e['key'])] = P[id(blocks[0])]['n'] - len(new) + i
        print(f"  [FACT UNITS | visit {vis + 1}] {len(new)} new units per block for {len(new)} new taught facts (keys from three taught wordings' answer positions); "
              f"{P[id(blocks[0])]['n']} active of {fm} | centred inputs, per-unit thresholds at the {q9:.2f} quantile of cosine on ordinary text: mean {np.mean(tl9) if tl9 else float('nan'):.3f}", flush=True)
    except Exception as e:
        _fail('inference.distill:fu'); print(f"  [FACT UNITS] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _dw_comp_probe(model, cfg, S_, S_slot, ents, held, D, vis):
    """COMPOSITION PROBE (readout, 11 Nov; inf_distill_comp_probe = N): two-fact chains built from the TAUGHT facts themselves - A r1 B and
    B r2 C, both in last visit's question pool (taught for at least one visit), A != C, no live conclusion for A-C or C-A (so never
    edited, replayed or asked) and no stored fact linking A and C directly - asked 'A is connected to' in the held-out conclusion
    wordings. Up to N chains; with the trunk's recall of the two premises in the test wordings, and composition among chains whose
    premises it answers both (the only chains it could compose). The same for chains of never-taught (control) facts."""
    import zlib
    n = int(getattr(cfg, 'inf_distill_comp_probe', 0))
    if not n or not D or not D.get('bysl'): return
    if _BRG['held']: n = max(n, 256)                                           # held-out chains: ask them all
    try:
        prev = {rw[3] for pool in D['bysl'].values() for rw in pool}
        live = set()
        for c in S_.get('conc', []):
            if not c.get('freed') and not c.get('retracted') and isinstance(c.get('A'), str) and isinstance(c.get('C'), str): live |= {(c['A'], c['C']), (c['C'], c['A'])}
        allf = [e for e in ents + held if e['key'][0] == 'fact']; direct = {(e['A'], e['ans']) for e in allf} | {(e['ans'], e['A']) for e in allf}
        def chains(fs):
            byA = {}
            for e in fs: byA.setdefault(e['A'], []).append(e)
            out = []
            for e1 in fs:
                for e2 in byA.get(e1['ans'], []):
                    A, C = e1['A'], e2['ans']
                    if A == C or (A, C) in live or (A, C) in direct: continue
                    out.append((e1, e2))
            out.sort(key=lambda p: zlib.crc32((p[0]['A'] + '|' + p[0]['ans'] + '|' + p[1]['ans']).encode('latin-1', 'replace')))
            return out[:n]
        def run(chs):
            if not chs: return None
            q = [dict(key=('conclusion', e1['A'], 'is connected to', e2['ans']), A=e1['A'], r='is connected to', ans=e2['ans'], d=e1['d']) for e1, e2 in chs]
            comp = [_dt_recall(model, S_slot, [x])['conclusions'] for x in q]
            prem = [(_dt_recall(model, S_slot, [e1])['facts'], _dt_recall(model, S_slot, [e2])['facts']) for e1, e2 in chs]
            ok = [i for i, (a, b) in enumerate(prem) if a[1] and b[1] and a[0] >= 0.5 and b[0] >= 0.5]
            c_all = float(np.nanmean([c[0] for c in comp])) if comp else float('nan')
            c_ok = float(np.nanmean([comp[i][0] for i in ok])) if ok else float('nan')
            return len(chs), len(ok), c_all, c_ok
        T = chains([e for e in ents if e['key'][0] == 'fact' and repr(e['key']) in prev]); H = chains([e for e in held if e['key'][0] == 'fact'])
        rt, rh = run(T), run(H)
        f = lambda r: (f"{r[0]} chains, both premises answered in a test wording {r[1]} | composed (held-out wordings) ALL {r[2]:.2f}, among those with both premises answered "
                       + (f"{r[3]:.2f}" if r[1] else "-")) if r else "none"
        xtra = ""
        if _BRG['held']:
            ta = [p for p in T if p[0]['ans'] in _BRG['anch']]; tu = [p for p in T if p[0]['ans'] in _BRG['held'] and p[0]['ans'] not in _BRG['anch']]
            xtra = f" || HELD-OUT CHAINS, bridge ANCHORED (invented attribute): {f(run(ta))} | bridge NOT anchored: {f(run(tu))}"
        print(f"  [COMPOSITION PROBE | visit {vis + 1}] 'A is connected to C' for chains A-B, B-C of facts, no conclusion or direct fact for A-C (never edited, replayed or asked) || "
              f"TAUGHT premises: {f(rt)} || NEVER-TAUGHT premises: {f(rh)}" + xtra, flush=True)
    except Exception as e:
        _fail('inference.distill:comp'); print(f"  [COMPOSITION PROBE] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _dw_latent_chain(model, cfg, S_, S_slot, ents, held, D, vis):
    """LATENT CHAINING (readout, 11 Nov; inf_distill_lchain = training steps): can a small JEPA-style predictor over the trunk's own name codes do
    rule-free inference? Entity code: the last block's output mean-pooled over the name in 'About NAME.' (frozen, centred, normalised).
    P(e_A, r) -> e_B: an MLP with a learned relation embedding, trained by InfoNCE against every entity code on the TAUGHT facts (one-hop only,
    never a chain). Q(e_A, r1, r2) -> e_C: the same, trained on the chains the SYSTEM ITSELF concluded (live conclusions with both premise facts
    found) - composition learned from examples, no rule. Tested on chains of taught facts with NO conclusion (never concluded, stored,
    replayed or edited; with inf_kgx_hold the held-out chains): the true C among 20 candidate names (chance 0.05), by P twice (chaining
    through the predicted bridge) and by Q in one step; and P on held-back (never-taught) facts as a baseline. Training is not changed."""
    import zlib
    from platypus.inference.closed_form import _ws_batch, _ws_run
    n_st = int(getattr(cfg, 'inf_distill_lchain', 0))
    if not n_st: return
    try:
        fs = [e for e in ents if e['key'][0] == 'fact']; hf = [e for e in held if e['key'][0] == 'fact']
        if len(fs) < 16: return
        names = sorted({e['A'] for e in fs + hf} | {e['ans'] for e in fs + hf}); ix = {n: i for i, n in enumerate(names)}
        rels = sorted({e['r'] for e in fs + hf}); rx = {r: i for i, r in enumerate(rels)}
        dev = model.embed.weight.device; cap = {}
        h9 = model.blocks[-1].register_forward_hook(lambda m_, i_, o_: cap.__setitem__('h', o_.detach()))
        E = []
        try:
            for i0 in range(0, len(names), 64):
                chunk = names[i0:i0 + 64]; txt = [f"About {n}.".encode('latin-1', 'replace') for n in chunk]
                X, _ = _ws_batch(model, txt)
                with torch.no_grad(): _ws_run(model, X, 0)
                h = cap['h'].float()
                for j, n in enumerate(chunk):
                    E.append(h[j, 5:5 + len(n.encode('latin-1', 'replace'))].mean(0))
        finally: h9.remove()
        E = torch.stack(E); E = F.normalize(E - E.mean(0), dim=-1); d = E.shape[1]
        g = torch.Generator(device='cpu').manual_seed(int(cfg.seed) + 4242 + vis)
        def mlp(k_in):
            return torch.nn.Sequential(torch.nn.Linear(k_in, 512), torch.nn.GELU(), torch.nn.Linear(512, d)).to(dev)
        def train(net, emb, X_idx, R_idx, Y_idx):
            opt = torch.optim.Adam(list(net.parameters()) + list(emb.parameters()), lr=1e-3)
            n = len(Y_idx); Xi = torch.tensor(X_idx, device=dev); Ri = torch.tensor(R_idx, device=dev); Yi = torch.tensor(Y_idx, device=dev)
            for st in range(n_st):
                b = torch.randint(0, n, (min(256, n),), generator=g).to(dev)
                z = F.normalize(net(torch.cat([E[Xi[b]], emb(Ri[b]).flatten(1)], -1)), dim=-1)
                loss = F.cross_entropy(z @ E.t() / 0.05, Yi[b]); opt.zero_grad(); loss.backward(); opt.step()
            return float(loss)
        def rank(z, true, rs):                                                 # true among 20 candidates (19 random other names)
            cand = [true] + [int(c) for c in rs.choice([i for i in range(len(names)) if i != true], min(19, len(names) - 1), replace=False)]
            sc = (F.normalize(z, dim=-1) @ E[cand].t()); return int(sc.argmax()) == 0
        torch.manual_seed(int(cfg.seed) + 4343 + vis)
        P = mlp(2 * d); Pe = torch.nn.Embedding(len(rels), d).to(dev)
        lp = train(P, Pe, [ix[e['A']] for e in fs], [[rx[e['r']]] for e in fs], [ix[e['ans']] for e in fs])
        rs = np.random.RandomState(int(cfg.seed) + 4444 + vis)
        with torch.no_grad():
            pz = lambda a, r: P(torch.cat([E[a][None], Pe(torch.tensor([[r]], device=dev)).flatten(1)], -1))[0]
            one = np.mean([rank(pz(ix[e['A']], rx[e['r']]), ix[e['ans']], rs) for e in fs[:256]])
            base = np.mean([rank(pz(ix[e['A']], rx[e['r']]), ix[e['ans']], rs) for e in hf[:256]]) if hf else float('nan')
        # chains: taught facts, no live conclusion, no direct fact (as the composition probe)
        prev = {rw[3] for pool in D['bysl'].values() for rw in pool} if D and D.get('bysl') else {repr(e['key']) for e in fs}
        live = set()
        for c in S_.get('conc', []):
            if not c.get('freed') and not c.get('retracted') and isinstance(c.get('A'), str) and isinstance(c.get('C'), str): live |= {(c['A'], c['C']), (c['C'], c['A'])}
        direct = {(e['A'], e['ans']) for e in fs + hf} | {(e['ans'], e['A']) for e in fs + hf}
        tf = [e for e in fs if repr(e['key']) in prev]; byA = {}
        for e in tf: byA.setdefault(e['A'], []).append(e)
        test = [(e1, e2) for e1 in tf for e2 in byA.get(e1['ans'], []) if e1['A'] != e2['ans'] and (e1['A'], e2['ans']) not in live and (e1['A'], e2['ans']) not in direct]
        allf = {}
        for e in fs + hf: allf.setdefault(e['A'], []).append(e)
        trc = []                                                               # the system's own concluded chains, with their premise facts
        for c in S_.get('conc', []):
            if c.get('freed') or c.get('retracted') or not all(isinstance(c.get(k), str) for k in ('A', 'B', 'C')): continue
            e1 = next((e for e in allf.get(c['A'], []) if e['ans'] == c['B']), None); e2 = next((e for e in allf.get(c['B'], []) if e['ans'] == c['C']), None)
            if e1 is not None and e2 is not None and c['C'] in ix: trc.append((e1, e2))
        res = {}
        with torch.no_grad():
            if test:
                pp = [rank(pz(int((F.normalize(pz(ix[a['A']], rx[a['r']]), dim=-1) @ E.t()).argmax()), rx[b['r']]), ix[b['ans']], rs) for a, b in test[:256]]
                hop1 = [rank(pz(ix[a['A']], rx[a['r']]), ix[a['ans']], rs) and rank(pz(ix[b['A']], rx[b['r']]), ix[b['ans']], rs) for a, b in test[:256]]
                res['pp'] = (np.mean(pp), len(pp), np.mean([x for x, h in zip(pp, hop1) if h]) if any(hop1) else float('nan'), int(sum(hop1)))
        lq = float('nan')
        if test and len(trc) >= 8:
            Q = mlp(3 * d); Qe = torch.nn.Embedding(len(rels), d).to(dev)
            lq = train(Q, Qe, [ix[a['A']] for a, b in trc], [[rx[a['r']], rx[b['r']]] for a, b in trc], [ix[b['ans']] for a, b in trc])
            with torch.no_grad():
                qz = lambda a, r1, r2: Q(torch.cat([E[a][None], Qe(torch.tensor([[r1, r2]], device=dev)).flatten(1)], -1))[0]
                q_tr = np.mean([rank(qz(ix[a['A']], rx[a['r']], rx[b['r']]), ix[b['ans']], rs) for a, b in trc[:256]])
                q_te = [rank(qz(ix[a['A']], rx[a['r']], rx[b['r']]), ix[b['ans']], rs) for a, b in test[:256]]
                res['q'] = (q_tr, np.mean(q_te), len(q_te))
        def split(sel):
            if not _BRG['held'] or not test: return ""
            out = []
            for nm9, keep in (('ANCHORED', lambda a: a['ans'] in _BRG['anch']), ('NOT anchored', lambda a: a['ans'] in _BRG['held'] and a['ans'] not in _BRG['anch'])):
                tt = [(a, b) for a, b in test if keep(a)]
                if tt:
                    with torch.no_grad():
                        v = [rank(pz(int((F.normalize(pz(ix[a['A']], rx[a['r']]), dim=-1) @ E.t()).argmax()), rx[b['r']]), ix[b['ans']], rs) for a, b in tt]
                    out.append(f"{nm9} {np.mean(v):.2f} (n {len(v)})")
            return (" || held-out chains, P twice: " + " | ".join(out)) if out else ""
        pp = res.get('pp'); q = res.get('q')
        print(f"  [LATENT CHAINING | visit {vis + 1}] trunk name codes (last block, centred) for {len(names)} names, {len(rels)} relations; true C among 20 names (chance 0.05) || "
              f"P(e_A, r) -> e_B trained on {len(fs)} taught facts ({n_st} steps, loss {lp:.2f}): one-hop on them {one:.2f} | on never-taught facts {base:.2f} || "
              f"CHAINS with no conclusion ({len(test)}): P twice {pp[0]:.2f} (n {pp[1]}), among chains whose both hops P gets right {pp[2]:.2f} (n {pp[3]})" if pp else
              f"  [LATENT CHAINING | visit {vis + 1}] {len(names)} names, P one-hop {one:.2f} on taught facts, {base:.2f} never taught || no test chains", end="", flush=True)
        if pp:
            print((f" || Q(e_A, r1, r2) -> e_C trained on {len(trc)} chains the system concluded (loss {lq:.2f}): on them {q[0]:.2f} | on the unconcluded chains {q[1]:.2f} (n {q[2]})" if q else
                   f" || Q: too few concluded chains ({len(trc)})") + split(None), flush=True)
        else: print("", flush=True)
    except Exception as e:
        _fail('inference.distill:lchain'); print(f"  [LATENT CHAINING] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _name_codes(model, names):
    """trunk code of each name: the last block's output mean-pooled over the name in 'About NAME.' (centred, normalised)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    cap = {}; h9 = model.blocks[-1].register_forward_hook(lambda m_, i_, o_: cap.__setitem__('h', o_.detach())); E = []
    try:
        for i0 in range(0, len(names), 64):
            chunk = names[i0:i0 + 64]; X, _ = _ws_batch(model, [f"About {n}.".encode('latin-1', 'replace') for n in chunk])
            with torch.no_grad(): _ws_run(model, X, 0)
            h = cap['h'].float()
            for j, n in enumerate(chunk): E.append(h[j, 5:5 + len(n.encode('latin-1', 'replace'))].mean(0))
    finally: h9.remove()
    E = torch.stack(E); return F.normalize(E - E.mean(0), dim=-1)


def _dw_opalg(model, cfg, S_, ents, held, vis):
    """OPERATOR ALGEBRA ON THE RULE WORLD (readout, 11 Nov; inf_distill_opalg): can relation operators over the trunk's name codes compose,
    and discover composition rules, as in tools/toy_jop.py? On the planted RULE WORLD (inf_rule_world) of the domains visited so far:
    a linear operator per relation (child-of, grandchild-of, supervised-by, overseen-by) fitted on the PLANTED facts - (A) in closed form on
    the trunk's frozen name codes, (B) with a small learned projection head trained JEPA-style (InfoNCE + isotropy). Tests on the HELD-OUT
    composite facts (never planted): the composed operator P_r P_r, the learned composite operator, true answer among all rule-world
    names; RULE DISCOVERY: the product's distance to each relation operator, against a NULL (the same fit with the facts' targets shuffled).
    Also how many planted rule-world facts the store's own reader recorded. Training is not changed."""
    from platypus.inference.probes import inf_rule_world
    try:
        W = inf_rule_world(cfg)
        if not W['planted']: return
        seen = set(int(k) for k in (model.__dict__.get('_t2s') or {}).keys())
        P_ = [r for r in W['planted'] if r[0] in seen]; H_ = [r for r in W['held'] if r[0] in seen]
        if len(P_) < 12 or not H_: return
        st = {(e['A'], e['ans']) for e in ents + held if e['key'][0] == 'fact'}; rec = sum(1 for r in P_ if (r[3], r[4]) in st)
        names = sorted({r[3] for r in P_ + H_} | {r[4] for r in P_ + H_}); ix = {n: i for i, n in enumerate(names)}
        E0 = _name_codes(model, names); d = E0.shape[1]; dev = E0.device; N = len(names)
        rels = ('parent', 'grand', 'mgr', 'skip')
        def fit_ridge(E, perm=None):
            out = {}
            for r in rels:
                fr = [x for x in P_ if x[2] == r]
                if len(fr) < 2: continue
                X = E[[ix[x[3]] for x in fr]]; Yi = [ix[x[4]] for x in fr]
                if perm is not None: Yi = list(np.random.RandomState(perm).permutation(Yi))
                Y = E[Yi]; lam = 1e-2 * float((X * X).sum()) / max(1, X.shape[1])
                out[r] = torch.linalg.solve(X.t() @ X + lam * torch.eye(X.shape[1], device=dev), X.t() @ Y)
            return out
        def fit_head(perm=None, steps=600):
            g = torch.Generator(device='cpu').manual_seed(int(cfg.seed) + 515 + (perm or 0))
            Wp = torch.nn.Linear(d, 128, bias=False).to(dev); Pr = torch.nn.ParameterDict({r: torch.nn.Parameter(torch.eye(128, device=dev)) for r in rels})
            opt = torch.optim.Adam(list(Wp.parameters()) + list(Pr.parameters()), lr=3e-3)
            rows = [(r[2], ix[r[3]], ix[r[4]]) for r in P_]
            if perm is not None:
                rs9 = np.random.RandomState(perm); by = {}
                for k, (r, a, b) in enumerate(rows): by.setdefault(r, []).append(k)
                rows2 = list(rows)
                for r, ks in by.items():
                    tg = [rows[k][2] for k in ks]; rs9.shuffle(tg)
                    for k, t in zip(ks, tg): rows2[k] = (r, rows[k][1], t)
                rows = rows2
            for _ in range(steps):
                E = F.normalize(Wp(E0), dim=-1); loss = 0.0
                for r in rels:
                    br = [(a, b) for rr, a, b in rows if rr == r]
                    if not br: continue
                    a = torch.tensor([x for x, _ in br], device=dev); b = torch.tensor([y for _, y in br], device=dev)
                    loss = loss + F.cross_entropy(F.normalize(E[a] @ Pr[r], dim=-1) @ E.t() / 0.1, b)
                z = E - E.mean(0); cov = z.t() @ z / N
                loss = loss + 0.1 * ((cov * N / 128 - torch.eye(128, device=dev)) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): return F.normalize(Wp(E0), dim=-1), {r: Pr[r].detach() for r in rels}
        def score(E, Pd):
            acc = lambda pred, tgt: float((F.normalize(pred, dim=-1) @ E.t()).argmax(-1).eq(tgt).float().mean())
            res = {}
            for r1, r2 in (('parent', 'grand'), ('mgr', 'skip')):
                hh = [x for x in H_ if x[2] == r2]
                if not hh or r1 not in Pd: continue
                a = torch.tensor([ix[x[3]] for x in hh], device=dev); c = torch.tensor([ix[x[4]] for x in hh], device=dev)
                comp = acc(E[a] @ Pd[r1] @ Pd[r1], c); lrn = acc(E[a] @ Pd[r2], c) if r2 in Pd else float('nan')
                M = Pd[r1] @ Pd[r1]; dist = {r: float(torch.norm(Pd[r] - M) / torch.norm(M)) for r in Pd}
                res[r2] = (comp, lrn, len(hh), dist)
            return res
        with torch.no_grad():
            Ra = score(E0, fit_ridge(E0)); Na = score(E0, fit_ridge(E0, perm=int(cfg.seed) + 77))
        Eb, Pb = fit_head(); Rb = score(Eb, Pb)
        Enb, Pnb = fit_head(perm=int(cfg.seed) + 78); Nb = score(Enb, Pnb)
        def fmt(R, Nl):
            out = []
            for r2, (comp, lrn, n, dist) in R.items():
                r1 = 'parent' if r2 == 'grand' else 'mgr'; nr = Nl.get(r2)
                best = min(dist, key=dist.get); oth = sorted(v for k, v in dist.items() if k != r2)
                out.append(f"{'grandchild = child o child' if r2 == 'grand' else 'overseen = supervised o supervised'} (n {n}): composed operator {comp:.2f}, learned composite operator {lrn:.2f} | "
                           f"RULE: product nearest '{best}', distance to the true composite {dist.get(r2, float('nan')):.2f} vs the others {oth[0] if oth else float('nan'):.2f}"
                           + (f"; NULL (targets shuffled): composed {nr[0]:.2f}, distance to the composite {nr[3].get(r2, float('nan')):.2f}" if nr else ""))
            return " | ".join(out) if out else "none"
        print(f"  [OPERATOR ALGEBRA, RULE WORLD | visit {vis + 1}] {len(P_)} planted rule-world facts in visited domains ({rec} recorded by the store's reader), {len(H_)} HELD-OUT "
              f"composite facts, {N} names (chance {1 / N:.3f}) || (A) FROZEN TRUNK CODES, closed-form operators: {fmt(Ra, Na)} || (B) LEARNED PROJECTION HEAD (JEPA-style): {fmt(Rb, Nb)}", flush=True)
    except Exception as e:
        _fail('inference.distill:opalg'); print(f"  [OPERATOR ALGEBRA] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def distill_wake_sleep(model, cfg, S_, S_slot, vslot=None, oslots=()):
    """INTERLEAVED DISTILLATION (inf_distill_wake > 0; 10 Nov): at each sleep, report the visit just finished (rows used, the mean
    answer loss, recall from the trunk in test wordings for taught facts against the never-taught control fifth) and rebuild the pool of
    question rows for the next visit; during the visit, distill_wake_step adds inf_distill_wake_rows of them to each real-text training step
    (same optimiser - its Fisher whitening, the real-text loss and the dream rows apply - no burst, no separate optimiser)"""
    import zlib
    from platypus.inference.namemem import _nm_material, _NM
    vis = int(model.__dict__.get('_visit', 0)); D = model.__dict__.get('_dtw')
    pmx = int(getattr(cfg, 'inf_distill_pool_max', 0))                         # 11 Nov CAPACITY: the question pool no longer stops at the name memory's cap
    allm = _nm_material(model, cfg, S_, nmax=pmx if pmx > 0 else None)          #   (inf_nm_max): every fact the store holds (seen 2+ times, latest object)
    cct = int(getattr(cfg, 'inf_distill_conc_ctl', 0))                        # 11 Nov: 1 = EVERY conclusion held back (never asked) - a powered composition test
    ctl = lambda e: (cct and e['key'][0] == 'conclusion') or zlib.crc32(repr(e['key']).encode()) % 5 == 0   #   and every live conclusion, up to inf_distill_pool_max
    ents = [e for e in allm if not ctl(e)]; held = [e for e in allm if ctl(e)]
    if pmx > 0:
        nmm = len(model.__dict__.get('_nm_ents', []) or [])
        print(f"  [DISTILLATION, CAPACITY | visit {vis + 1}] question pool {len(ents)} facts and conclusions taught ({sum(1 for e in ents if e['key'][0] == 'conclusion')} conclusions), "
              f"{len(held)} held back as control, from {len(allm)} the store holds (budget {pmx}; the name memory keeps {nmm}, cap {int(getattr(cfg, 'inf_nm_max', 384))})", flush=True)
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
        _gate_report(model, cfg, vis)
        if vslot is not None: _dw_repair(model, cfg, D, int(vslot), [int(s) for s in oslots], vis)
        _dw_ever(model, cfg, S_slot, ents, held, D, vis)
        if int(getattr(cfg, 'inf_distill_conc22', 1)): _dw_conc22(model, cfg, S_, S_slot, ents, held, vis)
        if int(getattr(cfg, 'inf_distill_ns_probe', 0)): _dw_ns_probe(model, cfg, S_, S_slot, ents, vslot, oslots, vis)
        if int(getattr(cfg, 'inf_distill_comp_probe', 0)): _dw_comp_probe(model, cfg, S_, S_slot, ents, held, D, vis)
        if int(getattr(cfg, 'inf_distill_lchain', 0)): _dw_latent_chain(model, cfg, S_, S_slot, ents, held, D, vis)
        if int(getattr(cfg, 'inf_distill_opalg', 0)): _dw_opalg(model, cfg, S_, ents, held, vis)
        if int(getattr(cfg, 'inf_distill_fu', 0)): _fu_sleep(model, cfg, S_, S_slot, ents, held, vslot, oslots, vis)
        sp = int(getattr(cfg, 'inf_distill_wake_spaced', 0)); sk = int(getattr(cfg, 'inf_distill_wake_selfkl', 0))
        spc = int(getattr(cfg, 'inf_distill_wake_spaced_conc', 0))             # 11 Nov: 0 = conclusions are NOT spaced (always rehearsed): the taught-wording check
        kn_ents = ents if (sk or spc) else [e for e in ents if e['key'][0] != 'conclusion']   #   runs with the closed-form edits in place, so it would take edit-held
        known = _dw_known(model, S_slot, kn_ents) if (sp or sk) else {}         #   conclusions for learned and stop the rows that give them recall WITHOUT the edits
        iv = dict((D or {}).get('iv') or {})
        if sp:
            for e in ents:
                k_ = repr(e['key'])
                if e['key'][0] == 'conclusion' and not spc: iv[k_] = 1; continue
                if k_ in known: iv[k_] = min(16, 2 * iv.get(k_, 1)) if known[k_] else 1
        rows = []; nt = 0
        from platypus.inference.closed_form import _ws_batch, _ws_run
        ctx_ = {}                                                              # 11 Nov PROMPT DISTILLATION: each fact's / conclusion's stored source sentence(s)
        pdp = int(getattr(cfg, 'inf_distill_pd_probe', 0))
        if int(getattr(cfg, 'inf_distill_wake_pd', 0)) or pdp:
            it9 = S_.get('it', [])
            for e in ents + (held if pdp else []):
                ss = []
                for i9 in (e.get('src') or ())[:2]:
                    if isinstance(i9, (int, np.integer)) and 0 <= int(i9) < len(it9):
                        b9 = it9[int(i9)]['s']
                        b9 = bytes(int(v) & 255 for v in b9.tolist()) if hasattr(b9, 'tolist') and not isinstance(b9, (bytes, bytearray)) else (b9.encode('latin-1', 'replace') if isinstance(b9, str) else bytes(b9))
                        ss.append(b9.strip()[:160])
                if ss: ctx_[repr(e['key'])] = b' '.join(ss) + b' '
            if pdp: _dw_pd_probe(model, cfg, S_slot, ents, held, ctx_, vis)
            H9 = (D or {}).get('pdst')
            if H9 and H9['n']:
                print(f"  [DISTILLATION, PROMPT DISTILLATION | visit {vis + 1}] last visit: {H9['n']} rows with the source sentence in the teacher's context (T {float(getattr(cfg, 'inf_distill_wake_pd_T', 2.0)):g}); "
                      f"answer bytes the TEACHER gets right {H9['t'] / max(1, H9['b']):.2f}, the STUDENT {H9['s'] / max(1, H9['b']):.2f} (of {H9['b']}); rows without a source sentence {H9['nos']} (hard targets)", flush=True)
            if int(getattr(cfg, 'inf_distill_wake_pd', 0)):
                print(f"  [DISTILLATION, PROMPT DISTILLATION | visit {vis + 1}] {len(ctx_)} of {len(ents)} taught facts and conclusions have a stored source sentence for the teacher", flush=True)
        brd = {}
        if int(getattr(cfg, 'inf_distill_bridge', 0)):
            try: brd = _dw_bridge(cfg, S_, ents, held, vis)
            except Exception as e9:
                _fail('inference.distill:bridge'); print(f"  [DISTILLATION, BRIDGE] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
        from platypus.inference.jepa import cfg_seq_
        L09 = int(cfg_seq_(model))
        for e in ents:
            k_ = repr(e['key']); sl = int(S_slot(e['d']))
            own9 = f"{e['A']} {e['r']} {e['ans']}.".encode('latin-1', 'replace')   #   (never the row's own fact as its descriptor)
            d1 = [d for d in brd.get(e['ans'], []) if d != own9] if e['key'][0] == 'fact' else None
            d2 = [d for d in brd.get(e['A'], []) if d != own9] if e['key'][0] == 'fact' else None
            for j9, fm in enumerate(_DT_FORMS):
                pb = fm.format(A=e['A'], r=e['r']).encode('latin-1', 'replace'); ab = (e['ans'] + '.').encode('latin-1', 'replace'); tl = None
                if d2:                                                         #   BRIDGE ANCHORING: the bridge as SUBJECT - its descriptor before the question,
                    rm9 = L09 - 1 - len(pb) - len(ab) - 1                          #   trimmed from the left to keep the answer in the window
                    if rm9 >= 8: pb = d2[j9 % len(d2)][-rm9:] + b' ' + pb
                if d1: ab = ab + b' ' + d1[j9 % len(d1)]                     #   the bridge as ANSWER - the same descriptor pool after it (trained bytes)
                if sk and known.get(k_):
                    X, _ = _ws_batch(model, [pb + ab])
                    with torch.no_grad(): lg = _ws_run(model, X, sl)[0]
                    p0 = len(pb) - 1; n_ = min(len(ab), lg.shape[0] - p0)
                    if n_ > 0: tl = F.log_softmax(lg[p0:p0 + n_].float(), -1).half(); nt += 1
                rows.append((pb, ab, sl, k_, tl, ctx_.get(k_)))
        if sp or sk:
            nk = sum(known.values())
            print(f"  [DISTILLATION, INTERLEAVED | visit {vis + 1}] TAUGHT-WORDING CHECK: the trunk answers {nk} of {len(known)} taught {'facts and conclusions' if (sk or spc) else 'facts (conclusions always rehearsed)'} in a taught wording"
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
        hpool = {}
        if int(getattr(cfg, 'inf_distill_wake_hint', 0)):
            H = (D or {}).get('hst')
            if H and H['steps']:
                print(f"  [DISTILLATION, HINTED | visit {vis + 1}] last visit: {H['steps']} steps, positions where the name memory changes the prediction "
                      f"{H['sel'] / max(1, H['pos']):.4f} of {H['pos']:.0f} (the update's support), KL there {np.mean(H['kl'][:20]) if H['kl'] else float('nan'):.4f} (first 20) -> "
                      f"{np.mean(H['kl'][-20:]) if H['kl'] else float('nan'):.4f} (last 20)", flush=True)
            hpool, hents = _dw_hint_pool(model, cfg, S_, S_slot, ents, held, vis)
            probe = [hents[int(i)] for i in rs.choice(len(hents), min(32, len(hents)), replace=False)] if hents else []
            print(f"  [DISTILLATION, HINTED | visit {vis + 1}] pool for the next visit: {sum(len(v) for v in hpool.values())} stored real sentences naming {len(hents)} taught facts' subjects "
                  f"(sentences naming a control subject left out)", flush=True)
        model.__dict__['_dtw'] = dict(pdst=dict(n=0, b=0, t=0, s=0, nos=0), bysl=bysl, w=w, iv=iv, rs=np.random.RandomState(int(cfg.seed) + 9494 + vis), n=0, losses=[], akl=[], probe=probe, hprobe=hprobe,
                                      anc=anc, cache=cache, hpool=hpool, st=dict(proj=0, conf=0, cut=[]))
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
        pd = int(getattr(cfg, 'inf_distill_wake_pd', 0)); LT = None
        if pd and any(len(it) > 5 and it[5] for it in items):                    # PROMPT DISTILLATION: the teacher reads the source sentence, then the question
            from platypus.inference.jepa import cfg_seq_
            L0_ = int(cfg_seq_(model))                                           # the context is trimmed from the LEFT to fit the window (the fact sits at the
            items = [it[:5] + ((it[5][-max(0, L0_ - len(it[0]) - len(it[1]) - 1):] if L0_ - len(it[0]) - len(it[1]) - 1 > 0 else b''),) if len(it) > 5 and it[5] else it for it in items]   #   sentence's end; local windows are 64 bytes)
            Tt, Mt = _ws_batch(model, [(it[5] if len(it) > 5 and it[5] else b'') + it[0] + it[1] for it in items])
            with torch.no_grad(): LT = _ws_run(model, Tt, sl).float()
            T_ = float(getattr(cfg, 'inf_distill_wake_pd_T', 2.0)); ce_w = float(getattr(cfg, 'inf_distill_wake_pd_ce', 0.0)); H9 = D['pdst']
        for r9, it in enumerate(items):
            p0 = len(it[0]) - 1; n_ = min(len(it[1]), lg.shape[1] - p0)
            if n_ <= 0: continue
            z = lg[r9, p0:p0 + n_].float(); tl = it[4] if len(it) > 4 else None
            cx = it[5] if (LT is not None and len(it) > 5) else None
            if cx:
                q0 = len(cx) + p0; m_ = min(n_, LT.shape[1] - q0)
                if m_ > 0:
                    zt = LT[r9, q0:q0 + m_]; zs = z[:m_]; tgt = X[r9, p0 + 1:p0 + 1 + m_]
                    pt = F.softmax(zt / T_, -1); kl = (pt * (F.log_softmax(zt / T_, -1) - F.log_softmax(zs / T_, -1))).sum() * T_ * T_
                    c_ = F.cross_entropy(zs, tgt, reduction='sum'); tot = tot + kl + ce_w * c_; ce.append(float(c_.detach()) / m_); cnt += m_
                    H9['n'] += 1; H9['b'] += m_; H9['t'] += int((zt.argmax(-1) == tgt).sum()); H9['s'] += int((zs.detach().argmax(-1) == tgt).sum())
                    continue
            if LT is not None: H9['nos'] += 1
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


def distill_wake_step(model, cfg, olds=None, dream=None, dk=1, dkl=None):
    """the interleaved question rows' backward, after the real step's backward (gradients add before the optimiser step); options above"""
    D = model.__dict__.get('_dtw')
    if not D: return
    ond = int(getattr(cfg, 'inf_distill_wake_ondream', 0))
    if ond and not dream: return
    w = float(cfg.inf_distill_wake) * (max(1, int(dk)) if ond else 1)
    if int(getattr(cfg, 'inf_distill_spindle', 0)): return                 # spindle coupling: questions only inside the sleep's replay burst
    if int(getattr(cfg, 'inf_distill_readonly', 0)): return                # CONTROL RUN: the pool and every readout as usual, no question rows trained
    gw = _gate_weight(model, cfg, dkl)
    if gw <= 0: return
    w *= gw
    if int(getattr(cfg, 'inf_distill_fu', 0)) and model.__dict__.get('_fu') is not None:   # FACT UNITS: the question rows train the fact units only
        st = model.__dict__['_fu']; allp = list(model.parameters()); rq = [p.requires_grad for p in allp]
        ts = [t for v in st['P'].values() for t in (v['V'], v['g'])]
        try:
            for p in allp: p.requires_grad_(False)
            for t in ts: t.requires_grad_(True)
            dl_ = distill_wake_loss(model, cfg)
            if torch.is_tensor(dl_) and dl_.requires_grad:
                gs = torch.autograd.grad(w * dl_, ts, allow_unused=True)
                for t, g in zip(ts, gs): t.grad = g
                st['opt'].step(); st['opt'].zero_grad(set_to_none=True); st['steps'] += 1; st['ql'].append(float(dl_.detach()))
                with torch.no_grad():
                    for v in st['P'].values(): v['g'].clamp_(min=0.0)
        except Exception:
            _fail('inference.distill:fu_step')
        finally:
            for t in ts: t.requires_grad_(False); t.grad = None
            for p, q in zip(allp, rq): p.requires_grad_(q)
        return
    proj = int(getattr(cfg, 'inf_distill_wake_proj', 0)); akl = float(getattr(cfg, 'inf_distill_wake_akl', 0.0))
    P, frz = _dw_params(model, cfg)
    for p in frz: p.requires_grad_(False)
    try:
        g0 = None
        if proj: g0 = [p.grad for p in P]; [setattr(p, 'grad', None) for p in P]
        dl_ = distill_wake_hint_loss(model, cfg) if int(getattr(cfg, 'inf_distill_wake_hint', 0)) else distill_wake_loss(model, cfg)
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


# ---------------------------------------------------------------- HINTED SELF-DISTILLATION (inf_distill_wake_hint, 11 Nov)
# After Uemura and Suzuki (arXiv 2610.05200): self-distillation with a hint updates sparsely (learning without overwriting) where
# supervised fine-tuning on correct outputs updates densely (overwriting). Here the HINT is the name memory: on stored real sentences that
# name taught subjects, TEACHER = the trunk with the name memory on (no gradient), STUDENT = the trunk alone; loss = KL(teacher || student)
# averaged over the positions where the memory changes the prediction (KL above 1e-4; elsewhere teacher = student, no gradient). No
# templated wordings, no one-hot targets. CONTROL: sentences naming any subject of a never-taught (control) fact are left out of the pool,
# so the memory cannot teach the control facts; the taught probe is drawn from facts whose subject is in the pool.
def _dw_hint_pool(model, cfg, S_, S_slot, ents, held, vis):
    import re
    csub = set(e['A'] for e in held); tents = [e for e in ents if e['A'] not in csub]
    if not tents: return {}, []
    it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
    rt = re.compile('|'.join(re.escape(n) for n in sorted(set(e['A'] for e in tents), key=len, reverse=True)))
    rc = re.compile('|'.join(re.escape(n) for n in sorted(csub, key=len, reverse=True))) if csub else None
    pool = {}; hit = set(); ra = np.random.RandomState(int(cfg.seed) + 9898 + vis)
    for i in ra.permutation(len(real)):
        x = real[int(i)]; s = bytes(x['s']).decode('latin-1')
        if rc is not None and rc.search(s): continue
        ms = rt.findall(s)
        if not ms: continue
        sl = int(S_slot(x['d']))
        if len(pool.setdefault(sl, [])) < 256: pool[sl].append(bytes(x['s'])); hit.update(ms)
    return pool, [e for e in tents if e['A'] in hit]


def distill_wake_hint_loss(model, cfg):
    D = model.__dict__.get('_dtw')
    if not D or not D.get('hpool'): return 0.0
    from platypus.inference.namemem import _NM
    if not _NM.get('ready'): return 0.0
    try:
        from platypus.inference.closed_form import _ws_batch, _ws_run
        rs = D['rs']; sls = list(D['hpool']); sl = sls[rs.randint(len(sls))]; pool = D['hpool'][sl]
        X, M = _ws_batch(model, [pool[int(i)] for i in rs.choice(len(pool), min(int(getattr(cfg, 'inf_distill_wake_rows', 4)), len(pool)), replace=False)])
        was = _NM['force']
        try:
            _NM['force'] = True
            with torch.no_grad(): lt = F.log_softmax(_ws_run(model, X, sl).float(), -1)
            _NM['force'] = False
            ls = F.log_softmax(_ws_run(model, X, sl, grad_=True).float(), -1)
        finally: _NM['force'] = was
        T = ls.shape[1]; m = M[:, 1:T + 1].float()
        kl = (lt.exp() * (lt - ls)).sum(-1) * m
        sel = (kl.detach() > 1e-4).float(); ns = float(sel.sum())
        H = D.setdefault('hst', dict(steps=0, pos=0.0, sel=0.0, kl=[]))
        H['steps'] += 1; H['pos'] += float(m.sum()); H['sel'] += ns
        if ns < 1: return 0.0
        loss = (kl * sel).sum() / ns; H['kl'].append(float(loss.detach())); D['n'] += 1
        return loss
    except Exception as e:
        _fail('inference.distill:hint')
        if not D.get('err'): D['err'] = 1; print(f"  [DISTILLATION, HINTED] a step failed ({type(e).__name__}: {e})"[:300], flush=True)
        return 0.0


# ---------------------------------------------------------------- RHYTHM GATES (11 Nov; each off by default)
# inf_gate_theta=L      THETA-LIKE PHASE GATE: training alternates ENCODING phases (real text + question rows, no dream rows) and
#                       CONSOLIDATION phases (real text + dream rows, no question rows). Consolidation phases last L steps on average,
#                       encoding phases L x (k - 1), k = the dream interval in force, so the share of dream steps (and the dream dose) is
#                       unchanged; question rows run in encoding phases only, weight x k / (k - 1) (same dose).
#                       inf_gate_theta_jitter=1: phase lengths drawn at random (exponential, mean as above; stochastic);
#                       0: fixed lengths (the control).
# inf_gate_beta=s       BETA-LIKE DRIFT BRAKE: the question rows' weight x min(1, (ref / ema)^s) where ema = running mean of the dream KL
#                       (old domains' drift from the teacher, measured on every dream step) and ref = its median over the previous visit
#                       x inf_gate_beta_ref: writes are throttled while old domains drift more than usual.
# inf_distill_spindle=1 SPINDLE-LIKE COUPLING: no question rows while awake; at each sleep they run INSIDE the repair replay burst
#                       (inf_distill_repair steps: KL to the previous sleep's predictions on old slots + the question loss x the weight).
def gate_dream(model, cfg, step, dk):
    """is this a dream step? (the theta gate when on, else every dk-th step)"""
    L = float(getattr(cfg, 'inf_gate_theta', 0.0)); dk = max(1, int(dk))
    if L <= 0: return step % dk == 0
    G = model.__dict__.get('_gth')
    if G is None:
        G = model.__dict__['_gth'] = dict(step=None, phase='con', left=0, rs=np.random.RandomState(int(cfg.seed) + 5151), n={'enc': 0, 'con': 0}, sw=0, lens=[], k=dk)
    if G['step'] != step:
        G['step'] = step; G['k'] = dk
        if G['left'] <= 0:
            G['phase'] = 'con' if G['phase'] == 'enc' else 'enc'; G['sw'] += 1
            mean = L if G['phase'] == 'con' else L * max(1, dk - 1)
            cap = float(getattr(cfg, 'inf_gate_theta_cap', 0.0))                # 11 Nov: BOUNDED stochastic phases - at most cap x the mean (0 = unbounded)
            ln = G['rs'].exponential(mean) if int(getattr(cfg, 'inf_gate_theta_jitter', 1)) else mean
            if cap > 0: ln = min(ln, cap * mean)
            G['left'] = max(1, int(round(ln)))
            G['lens'].append((G['phase'], G['left']))
        G['left'] -= 1; G['n'][G['phase']] += 1
    return G['phase'] == 'con'


def _gate_weight(model, cfg, dkl):
    """multiplier on the question rows' weight from the theta phase and the beta brake (0 = no question rows this step)"""
    g = 1.0
    if float(getattr(cfg, 'inf_gate_theta', 0.0)) > 0:
        G = model.__dict__.get('_gth')
        if G is not None:
            if G['phase'] != 'enc': return 0.0
            k = max(2, int(G.get('k', 2))); g *= k / (k - 1)
    s = float(getattr(cfg, 'inf_gate_beta', 0.0))
    if s > 0:
        B = model.__dict__.setdefault('_gbt', dict(ema=None, cur=[], ref=None, gs=[], hist=[]))
        if dkl is not None:
            v = float(dkl); B['cur'].append(v); B['ema'] = v if B['ema'] is None else 0.7 * B['ema'] + 0.3 * v
        b = 1.0
        if B['ref'] and B['ema'] is not None and B['ema'] > B['ref']: b = min(1.0, (B['ref'] / B['ema']) ** s)
        B['gs'].append(b); g *= b
    return g


def _gate_report(model, cfg, vis):
    """the gates' readouts for the visit just finished (printed at the sleep), then reset for the next visit"""
    G = model.__dict__.get('_gth')
    if G is not None and float(getattr(cfg, 'inf_gate_theta', 0.0)) > 0:
        le = [l for p, l in G['lens'] if p == 'enc']; lc = [l for p, l in G['lens'] if p == 'con']
        print(f"  [RHYTHM GATE, THETA | visit {vis + 1}] {'stochastic' if int(getattr(cfg, 'inf_gate_theta_jitter', 1)) else 'fixed'} phases: encoding {G['n']['enc']} steps, "
              f"consolidation {G['n']['con']} steps, {G['sw']} switches; phase lengths encoding mean {np.mean(le) if le else float('nan'):.1f} "
              f"(min {min(le) if le else 0}, max {max(le) if le else 0}), consolidation mean {np.mean(lc) if lc else float('nan'):.1f} | dream interval k {G.get('k')}", flush=True)
        G['n'] = {'enc': 0, 'con': 0}; G['sw'] = 0; G['lens'] = []
    B = model.__dict__.get('_gbt')
    if B is not None and float(getattr(cfg, 'inf_gate_beta', 0.0)) > 0:
        print(f"  [RHYTHM GATE, BETA | visit {vis + 1}] dream KL this visit median {np.median(B['cur']) if B['cur'] else float('nan'):.4f} (reference "
              f"{B['ref'] if B['ref'] else float('nan'):.4f}); write gate mean {np.mean(B['gs']) if B['gs'] else float('nan'):.3f}, share of steps throttled "
              f"{np.mean([x < 0.999 for x in B['gs']]) if B['gs'] else float('nan'):.2f}, min {min(B['gs']) if B['gs'] else float('nan'):.3f}", flush=True)
        if B['cur']: B['ref'] = float(np.median(B['cur'])) * float(getattr(cfg, 'inf_gate_beta_ref', 1.0))
        B['cur'] = []; B['gs'] = []; B['ema'] = None
