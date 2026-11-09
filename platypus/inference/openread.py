"""OPEN-ENDED READOUTS (inf_open_diag 1; readouts only - every weight change below is measured and undone).
Multiple choice works for held-out wordings (0.16-0.36 of 20) but open-ended greedy does not (0.00), while the fitted wording reaches 0.22
open-ended. These readouts say why, and test four remedies without changing the model:
 DIAGNOSTIC   at every byte of the answer (teacher-forced): is the true byte the model's top choice; at the FIRST byte, its rank, how many
              bits it trails the winner, and whether the winner is a letter (a different name) or not (a generic continuation). Open hit =
              every answer byte top-1 and a non-letter after it (= greedy decoding produces the name). Fitted wording, the other written
              edit wordings, the held-out wordings; edited conclusions and not-yet-edited ones; from the weights (store off).
 (1) KNOWN NAMES FROM EVERYTHING SEEN: in evaluate (inf_eval_trie_inv seen) - the known-name decoder's inventory is every capitalised
              word in the stored sentences, not just the fast store's names.
 (2)+(3) SHADOW EDITS (_cf_shadow, called after a kept closed-form edit; inf_cf_shadow 1): the same conclusions re-fitted as configured,
              with SELF-CONTRAST (each answer byte must beat the model's own top non-answer byte, not only competitors' answers), with SIX
              wordings (two more written wordings), and with both; each applied alone on the pre-edit weights at the kept scale, measured,
              and removed.
 (4) RUNG 3 KEYS: does the held-out question's key retrieve the right conclusion among the edited ones (nearest fitted-wording key)? The
              edit's own keys (MLP input of the edited block) and the trunk's output, raw, centred, and in closed-form SFA directions."""
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import _fail, _CONC_CF, _HO_TWO

_CONC_CF_X = ("{A} can be linked to ", "Starting from {A}, one arrives at ")       # two more WRITTEN wordings for the six-wording shadow edit;
                                                                                  #   none is a test wording (_HO_TWO, _CF_VARY, reworded/context/pronoun)


def _isl(b): return 65 <= b <= 90 or 97 <= b <= 122


def _tf_items(model, items, hook=None):
    """items: (prompt, C, slot). One teacher-forced forward per item (store off). Per item: open hit, per-byte top-1 share, first-byte rank,
    bits behind the winner at the first byte, winner-is-a-letter, and (when hook is a capture dict) the key at the first answer position"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    out = []
    for pr, C, sl in items:
        pb = pr.encode('latin-1', 'replace'); tb = (C + '.').encode('latin-1', 'replace'); X, M = _ws_batch(model, [pb + tb])
        if hook is not None: hook['on'] = True; hook['k'] = []
        with torch.no_grad(): lg = _ws_run(model, X, int(sl))[0]
        if hook is not None: hook['on'] = False
        ps = torch.arange(len(pb) - 1, len(pb) + len(tb) - 1, device=X.device); ps = ps[ps < lg.shape[0]]
        if len(ps) < len(tb): continue
        L = F.log_softmax(lg[ps], -1); tr = torch.tensor(list(tb), device=X.device)
        top = L.argmax(-1); okb = (top[:-1] == tr[:-1]).float(); endok = not _isl(int(top[-1]))
        hit = bool(okb.all()) and endok
        l0 = L[0]; t0 = int(tr[0]); rank = int((l0 > l0[t0]).sum()) + 1; w0 = int(l0.argmax()); gap = float(l0[w0] - l0[t0]) / np.log(2)
        key = None
        if hook is not None and hook['k']: key = (hook['k'][-1][0][ps[0]].detach().float().cpu(), None)
        out.append(dict(hit=hit, top1=float(okb.mean()), rank=rank, gap=gap, wl=_isl(w0), w0=w0 == t0, key=key))
    return out


def _summ(R):
    if not R: return "n 0"
    a = lambda k: np.array([r[k] for r in R], float)
    rk = a('rank'); mis = [r for r in R if not r['w0']]
    return (f"open {a('hit').mean():.2f} | answer bytes top-1 {a('top1').mean():.2f} | first byte top-1 {(rk <= 1).mean():.2f}, top-3 {(rk <= 3).mean():.2f}, "
            f"median rank {np.median(rk):.0f}, trails the winner by {np.median([r['gap'] for r in mis]) if mis else 0.0:.1f} bits (median of misses), "
            f"winner a letter {np.mean([r['wl'] for r in mis]) if mis else float('nan'):.2f} | n {len(R)}")


def open_diag(model, cfg, S_, S_slot, tag, keys=False):
    try:
        lv = [c for c in S_.get('conc', []) if not c.get('freed') and isinstance(c.get('C'), str) and isinstance(c.get('A'), str)]
        if len(lv) < 8: print(f"  [OPEN DIAGNOSTIC | {tag}] skipped: {len(lv)} conclusions", flush=True); return
        rs = np.random.RandomState(int(cfg.seed) + 4141); ed = [c for c in lv if c.get('cfe') is not None]; ne = [c for c in lv if c.get('cfe') is None]
        ed = [ed[i] for i in rs.permutation(len(ed))[:48]]; ne = [ne[i] for i in rs.permutation(len(ne))[:48]]
        nw = max(1, min(len(_CONC_CF), int(getattr(cfg, 'inf_cf_wordings', 1))))
        W = {'fitted wording': [_CONC_CF[0]], f'other written edit wordings ({nw - 1})': list(_CONC_CF[1:nw]), 'HELD-OUT wordings': list(_HO_TWO)}
        for grp, L in (('edited', ed), ('not edited', ne)):
            if not L: continue
            parts = []
            for wn, ws in W.items():
                if not ws: continue
                R = _tf_items(model, [(w.format(A=c['A']), c['C'], S_slot(c['d'])) for c in L for w in ws]); parts.append(f"{wn}: {_summ(R)}")
            print(f"  [OPEN DIAGNOSTIC | {tag} | {grp} conclusions, from the weights] " + " || ".join(parts), flush=True)
        if keys and len(ed) >= 8: _key_readout(model, cfg, S_, S_slot, ed, tag)
    except Exception as e:
        _fail('inference.openread'); print(f"  [OPEN DIAGNOSTIC] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _key_readout(model, cfg, S_, S_slot, ed, tag):
    """(4) the held-out question's key against the fitted-wording keys of all these conclusions: top-1 retrieval of its own conclusion"""
    from platypus.inference.closed_form import _ws_batch, _ws_run, _ws_eager
    from platypus.inference.selcmp import _codes, _fit_linear
    li = int(getattr(cfg, 'inf_cf_layer', -1)); mo = model.blocks[li].mlp_out; back = _ws_eager(model); cap = {'on': False, 'k': []}
    def hk(m, i, o):
        if cap['on']: cap['k'].append(i[0].detach().float())
    h = mo.register_forward_hook(hk)
    try:
        from platypus.conduit import _carry_save, _carry_load
        from platypus.layers import _CARRY
        def keys_of(w):
            K, H, NS, NK = [], [], [], []
            for c in ed:
                pb = w.format(A=c['A']).encode('latin-1', 'replace'); X, M = _ws_batch(model, [pb + (c['C'] + '.').encode('latin-1', 'replace')]); p0 = len(pb) - 1
                cap['on'] = True; cap['k'] = []; _ws_run(model, X, int(S_slot(c['d']))); cap['on'] = False
                K.append(cap['k'][-1][0][p0].cpu()); ab = c['A'].encode('latin-1', 'replace'); i0 = pb.find(ab); i1 = i0 + len(ab)
                NK.append(cap['k'][-1][0][i1 - 1].cpu() if i0 >= 0 else cap['k'][-1][0][p0].cpu())
                sv = _carry_save(); _CARRY.update(on=False, st={}, new={}); was = model.training; model.eval()
                try:
                    with torch.no_grad(): hf = model.features(X[:, :-1])
                finally: _carry_load(sv); model.train(was)
                hf = hf[0] if isinstance(hf, (tuple, list)) else hf
                H.append(hf[0, p0].float().cpu()); NS.append(hf[0, i0:i1].float().mean(0).cpu() if i0 >= 0 else hf[0, p0].float().cpu())
            return torch.stack(K), torch.stack(H), torch.stack(NS), torch.stack(NK)
        Kf, Hf, NSf, NKf = keys_of(_CONC_CF[0]); Kh = [keys_of(w) for w in _HO_TWO]
        it = S_.get('it', []); rs = np.random.RandomState(int(cfg.seed) + 4142)
        ix = [i for i in range(len(it) - 1) if it[i].get('k') == 'real' and it[i + 1].get('k') == 'real' and it[i]['d'] == it[i + 1]['d'] and 20 <= len(it[i]['s']) <= 300 and 20 <= len(it[i + 1]['s']) <= 300]
        sfa = None
        if len(ix) >= 400:
            ix = [ix[int(k)] for k in rs.choice(len(ix), min(3000, len(ix)), replace=False)]
            _, sfa, _, _, _ = _fit_linear(_codes(model, [it[i]['s'] for i in ix]).cpu(), _codes(model, [it[i + 1]['s'] for i in ix]).cpu(), seed=int(cfg.seed))
        def r1(Q, Kb):
            Q = F.normalize(Q, dim=-1); Kb = F.normalize(Kb, dim=-1); return float(((Q @ Kb.T).argmax(1) == torch.arange(len(Q))).float().mean())
        rows = {}
        def sims(Q, Kb, ctr=True):
            if ctr: m = Kb.mean(0); Q = Q - m; Kb = Kb - m
            return F.normalize(Q, dim=-1) @ F.normalize(Kb, dim=-1).T
        def zsum(*S): return sum((x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6) for x in S)
        def acc(S): return float((S.argmax(1) == torch.arange(len(S))).float().mean())
        for (Kq, Hq, NSq, NKq) in Kh:                                             # AMENDED RUNG 3: the NAME CHANNEL and [slow code ; name]
            rows.setdefault('NAME: trunk state over the subject name (centred)', []).append(acc(sims(NSq, NSf)))
            rows.setdefault('NAME: edit key at the name\'s last byte (centred)', []).append(acc(sims(NKq, NKf)))
            rows.setdefault('edit key + NAME state', []).append(acc(zsum(sims(Kq, Kf), sims(NSq, NSf))))
            if sfa is not None:
                mu, P, _ = sfa
                rows.setdefault('[SFA 32 ; NAME state]', []).append(acc(zsum(sims((Hq - mu) @ P, (Hf - mu) @ P, False), sims(NSq, NSf))))
                rows.setdefault('[SFA 32 ; NAME state ; edit key]', []).append(acc(zsum(sims((Hq - mu) @ P, (Hf - mu) @ P, False), sims(NSq, NSf), sims(Kq, Kf))))
        nA = len(set(c['A'] for c in ed))
        for (Kq, Hq, NSq, NKq) in Kh:
            rows.setdefault('edit key raw', []).append(r1(Kq, Kf)); mk = Kf.mean(0); rows.setdefault('edit key centred', []).append(r1(Kq - mk, Kf - mk))
            rows.setdefault('trunk output raw', []).append(r1(Hq, Hf)); mh = Hf.mean(0); rows.setdefault('trunk output centred', []).append(r1(Hq - mh, Hf - mh))
            if sfa is not None:
                mu, P, _ = sfa; rows.setdefault('trunk output in SFA directions (32)', []).append(r1((Hq - mu) @ P, (Hf - mu) @ P))
                for k in (4, 16): rows.setdefault(f'trunk output, slowest {k} SFA directions', []).append(r1((Hq - mu) @ P[:, :k], (Hf - mu) @ P[:, :k]))
        print(f"  [RUNG 3 KEYS | {tag}] held-out question -> its own conclusion by the nearest fitted-wording key, among {len(ed)} edited conclusions with {nA} distinct subjects "
              f"(chance {1 / len(ed):.3f}; mean of {len(_HO_TWO)} held-out wordings): " + " | ".join(f"{n} {np.mean(v):.2f}" for n, v in rows.items()), flush=True)
    finally:
        h.remove(); back()


def _cf_shadow(model, cfg, E_, mo_, cap_, S_slot, W0_, scale, Pm_, C0_, lam_, Cp_):
    """(2)+(3) shadow edits on up to 16 of this sleep's conclusions (NOTE: the shadow fits leave out competitor contrast, inf_cf_contrast, so
    'as configured' has no margin term at all), each applied alone to the pre-edit weights at the kept scale, measured,
    and removed (the kept edit is restored). cap_ is _cf_apply's hook dict (cap_['on'] captures keys, cap_['delta'] adds an output change)."""
    from platypus.inference.closed_form import _ws_batch, _ws_run, _ws_twohop
    Wk = mo_.weight.detach().clone(); E = [c for c in E_][:16]
    if len(E) < 8: return
    lr_ = float(getattr(cfg, 'inf_cf_lr', 0.5)); st_ = int(getattr(cfg, 'inf_cf_steps', 25)); nw = max(1, min(len(_CONC_CF), int(getattr(cfg, 'inf_cf_wordings', 1))))
    ws_self = float(getattr(cfg, 'inf_cf_shadow_self', 1.0)); rs_ = np.random.RandomState(int(cfg.seed) + 4343)
    def fit(words, selfc):
        Ks, Rs = [], []
        for c in E:
            for w in words:
                pr = w.format(A=c['A']).encode('latin-1', 'replace'); tg = (c['C'] + '.').encode('latin-1', 'replace'); X, M = _ws_batch(model, [pr + tg])
                T1 = X.shape[1] - 1; ps = torch.arange(len(pr) - 1, min(T1, len(pr) + len(tg) - 1), device=X.device)
                if ps.numel() == 0: continue
                sl = int(S_slot(c['d'])); cap_['on'] = True; cap_['k'] = []; _ws_run(model, X, sl); cap_['on'] = False; K1 = cap_['k'][-1][0][ps]
                D = mo_.weight.shape[0]; dl = torch.zeros(len(ps), D, device=X.device, requires_grad=True); op = torch.optim.Adam([dl], lr=lr_ * 0.1); yt = X[0, ps + 1]; comp = None
                for s in range(st_):
                    cap_['delta'] = torch.zeros(1, T1, D, device=X.device).index_put((torch.zeros_like(ps), ps), dl)
                    lg = _ws_run(model, X, sl, grad_=True)[0, ps]; l = F.cross_entropy(lg, yt) + 1e-3 * (dl ** 2).sum()
                    if selfc > 0:
                        if comp is None:                                          # the model's own top non-answer byte at each answer position (before fitting)
                            with torch.no_grad(): comp = lg.detach().scatter(1, yt[:, None], -1e9).argmax(1)
                        mg = lg.gather(1, yt[:, None])[:, 0] - lg.gather(1, comp[:, None])[:, 0]; l = l + selfc * F.softplus(-mg).mean()
                    op.zero_grad(); l.backward(); op.step(); cap_['delta'] = None
                Ks.append(K1); Rs.append(dl.detach().float())
        K = torch.cat(Ks); R = torch.cat(Rs); H = K.shape[1]
        if Pm_ is not None:
            Kc = K @ Pm_; A = Kc.T @ Kc + 1e-2 * float((K.T @ K).diagonal().mean()) * torch.eye(H, device=K.device) + lam_ * len(K) * (Pm_ @ C0_ @ Pm_)
            return (torch.linalg.solve(A, Kc.T @ R).T) @ Pm_
        return torch.linalg.solve(K.T @ K + lam_ * len(K) * C0_ + 1e-4 * torch.eye(H, device=K.device), K.T @ R).T
    def measure():
        f_ = lambda w, dec: _ws_twohop(model, cfg, E, Cp_, S_slot, np.random.RandomState(3), w, dec)
        Rf = _tf_items(model, [(_CONC_CF[0].format(A=c['A']), c['C'], S_slot(c['d'])) for c in E])
        Rh = _tf_items(model, [(w.format(A=c['A']), c['C'], S_slot(c['d'])) for c in E for w in _HO_TWO])
        mc = np.nanmean([f_(w, False) for w in _HO_TWO])
        return (f"fitted: open {np.mean([r['hit'] for r in Rf]):.2f} | HELD-OUT: multiple choice (10) {mc:.2f}, open {np.mean([r['hit'] for r in Rh]):.2f}, "
                f"first byte top-1 {np.mean([r['rank'] <= 1 for r in Rh]):.2f}, answer bytes top-1 {np.mean([r['top1'] for r in Rh]):.2f}")
    V = {f'as configured ({nw} wording(s))': (list(_CONC_CF[:nw]), 0.0), f'+ SELF-CONTRAST x{ws_self:g}': (list(_CONC_CF[:nw]), ws_self),
         'SIX wordings': (list(_CONC_CF) + list(_CONC_CF_X), 0.0), f'SIX wordings + SELF-CONTRAST x{ws_self:g}': (list(_CONC_CF) + list(_CONC_CF_X), ws_self)}
    out = []
    try:
        with torch.no_grad(): mo_.weight.copy_(W0_)
        out.append(f"no edit: {measure()}")
        for n, (ws, sc) in V.items():
            with torch.no_grad(): mo_.weight.copy_(W0_)
            dW = fit(ws, sc)
            with torch.no_grad(): mo_.weight.copy_((W0_.float() + scale * dW.float()).to(W0_.dtype))
            out.append(f"{n}: {measure()}")
        print(f"  [CF SHADOW EDITS] {len(E)} of this sleep's conclusions, each variant fitted on them alone and applied to the pre-edit weights at x{scale:g}, measured, removed "
              f"(the kept edit is restored) || " + " || ".join(out), flush=True)
    except Exception as e:
        _fail('inference.openread:shadow'); print(f"  [CF SHADOW EDITS] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
    finally:
        cap_['delta'] = None
        with torch.no_grad(): mo_.weight.copy_(Wk)
