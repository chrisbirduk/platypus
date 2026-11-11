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
def distill_wake_sleep(model, cfg, S_, S_slot):
    """INTERLEAVED DISTILLATION (inf_distill_wake > 0; 10 Nov): at each sleep, report the visit just finished (rows used, the mean
    answer loss, recall from the trunk in test wordings for taught facts against the never-taught control fifth) and rebuild the pool of
    question rows for the next visit; during the visit, distill_wake_loss adds inf_distill_wake_rows of them to each real-text training step
    (same backward, same optimiser - its Fisher whitening, the real-text loss and the dream rows apply - no burst, no separate optimiser)"""
    import zlib
    from platypus.inference.namemem import _nm_material, _NM
    vis = int(model.__dict__.get('_visit', 0)); D = model.__dict__.get('_dtw')
    allm = _nm_material(model, cfg, S_)
    ctl = lambda e: zlib.crc32(repr(e['key']).encode()) % 5 == 0
    ents = [e for e in allm if not ctl(e)]; held = [e for e in allm if ctl(e)]
    if D is not None and D.get('probe'):
        was = _NM['force']; _NM['force'] = False
        try:
            r = _dt_recall(model, S_slot, D['probe']); h = _dt_recall(model, S_slot, D['hprobe']) if D.get('hprobe') else None
        finally: _NM['force'] = was
        f = lambda x: " | ".join(f"{k} {v[0]:.2f} ({v[1]})" for k, v in x.items()) if x else "none"
        ls = D['losses']
        print(f"  [DISTILLATION, INTERLEAVED | visit {vis + 1}] last visit: {D['n']} training steps with {int(getattr(cfg, 'inf_distill_wake_rows', 4))} question rows each "
              f"(weight {float(cfg.inf_distill_wake):g}), answer loss {np.mean(ls[:20]) if ls else float('nan'):.3f} (first 20) -> {np.mean(ls[-20:]) if ls else float('nan'):.3f} (last 20) | "
              f"from the trunk, test wordings, after the visit: {len(D['probe'])} taught facts {f(r)} || CONTROL, {len(D['hprobe'])} never-taught {f(h)}", flush=True)
    rows = []
    for e in ents:
        for fm in _DT_FORMS:
            rows.append(((fm.format(A=e['A'], r=e['r'])).encode('latin-1', 'replace'), (e['ans'] + '.').encode('latin-1', 'replace'), int(S_slot(e['d']))))
    rs = np.random.RandomState(int(cfg.seed) + 9393 + vis)
    probe = [ents[int(i)] for i in rs.choice(len(ents), min(32, len(ents)), replace=False)] if ents else []
    hprobe = [held[int(i)] for i in rs.choice(len(held), min(32, len(held)), replace=False)] if held else []
    bysl = {}
    for rw in rows: bysl.setdefault(rw[2], []).append(rw)
    model.__dict__['_dtw'] = dict(bysl=bysl, rs=np.random.RandomState(int(cfg.seed) + 9494 + vis), n=0, losses=[], probe=probe, hprobe=hprobe)


def distill_wake_loss(model, cfg):
    """inf_distill_wake_rows question rows (one slot, drawn from the pool) - cross-entropy on the answer bytes only; 0.0 when no pool"""
    D = model.__dict__.get('_dtw')
    if not D or not D['bysl']: return 0.0
    try:
        from platypus.inference.closed_form import _ws_batch, _ws_run
        k = int(getattr(cfg, 'inf_distill_wake_rows', 4)); rs = D['rs']; sls = list(D['bysl']); sl = sls[rs.randint(len(sls))]; pool = D['bysl'][sl]
        items = [pool[int(i)] for i in rs.choice(len(pool), min(k, len(pool)), replace=False)]
        X, M = _ws_batch(model, [p + a for p, a, _ in items]); lg = _ws_run(model, X, sl, grad_=True)
        rows, pos = [], []
        for r9, (p, a, _) in enumerate(items):
            p0 = len(p) - 1
            for j in range(len(a)):
                if p0 + j < lg.shape[1]: rows.append(r9); pos.append(p0 + j)
        rows = torch.tensor(rows, device=X.device); pos = torch.tensor(pos, device=X.device)
        loss = F.cross_entropy(lg[rows, pos], X[rows, pos + 1]); D['n'] += 1; D['losses'].append(float(loss.detach()))
        return loss
    except Exception as e:
        _fail('inference.distill:wake')
        if not D.get('err'): D['err'] = 1; print(f"  [DISTILLATION, INTERLEAVED] a step failed ({type(e).__name__}: {e})"[:300], flush=True)
        return 0.0
