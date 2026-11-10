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
    and removed (the kept edit is restored). cap_ is _cf_apply's hook dict (cap_['on'] captures keys, cap_['delta'] adds an output change).
    NAME-KEYED shadow edits (9 Nov, the amended rung 3 as a build): the same targets solved on keys [own key ; NAME key] - the name key is the
    edited layer's input at the subject name's last byte - and, at test, the name part read from the question's own name (the edit's output
    change at the answer positions = dW_own k_own + dW_name k_name); also the NAME key alone. Plain ridge, so its control is 'own key, plain
    ridge'. (Six wordings were dropped: no gain over three on the GPU, five sleeps.)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run, _ws_twohop
    Wk = mo_.weight.detach().clone(); E = [c for c in E_][:16]
    if len(E) < 8: return
    lr_ = float(getattr(cfg, 'inf_cf_lr', 0.5)); st_ = int(getattr(cfg, 'inf_cf_steps', 25)); nw = max(1, min(len(_CONC_CF), int(getattr(cfg, 'inf_cf_wordings', 1))))
    ws_self = float(getattr(cfg, 'inf_cf_shadow_self', 1.0)); D = mo_.weight.shape[0]
    def name_pos(pr, A, ps0):
        ab = A.encode('latin-1', 'replace'); i0 = pr.find(ab); return (i0 + len(ab) - 1) if i0 >= 0 else int(ps0)
    def fit(words, selfc, infow=0.0, items=None):
        """items: (prompt, answer, domain, subject) to fit; by default this sleep's conclusions in each of the given wordings.
        infow > 0: INFORMATION-WEIGHTED answer bytes (as in replay, inf_cons_infow): each byte's loss weighted by its bits under a neutral cue
        (the subject replaced by 'someone'), divided by the mean over the answer and clipped to [0.25, infow].
        (The ROME-style subject-site fit was removed 10 Nov: negative, see RESULTS.md.)"""
        Ks, Rs, Ns, Js = [], [], [], []
        its = items if items is not None else [(w.format(A=c['A']), c['C'], c['d'], c['A']) for c in E for w in words]
        for prs, Cs, dd, As in its:
                pr = prs.encode('latin-1', 'replace'); tg = (Cs + '.').encode('latin-1', 'replace'); X, M = _ws_batch(model, [pr + tg])
                T1 = X.shape[1] - 1; ps = torch.arange(len(pr) - 1, min(T1, len(pr) + len(tg) - 1), device=X.device)
                if ps.numel() == 0: continue
                cq = cap_; sl = int(S_slot(dd)); cq['on'] = True; cq['k'] = []
                try: _ws_run(model, X, sl)
                finally: cq['on'] = False                                         # never left on: it would keep every later forward's keys
                K1 = cq['k'][-1][0][ps]; nk = cq['k'][-1][0][name_pos(pr, As, ps[0])]
                wb = None
                if infow > 0:
                    prn = prs.replace(As, 'someone').encode('latin-1', 'replace'); Xn, Mn = _ws_batch(model, [prn + tg]); psn = torch.arange(len(prn) - 1, len(prn) - 1 + len(ps), device=X.device)
                    with torch.no_grad():
                        lgn = _ws_run(model, Xn, sl)[0]
                        if int(psn[-1]) < lgn.shape[0]:
                            bn = F.cross_entropy(lgn[psn].float(), Xn[0, psn + 1], reduction='none'); wb = (bn / bn.mean().clamp(min=1e-6)).clamp(0.25, infow)
                pd = ps                                                           # the output change is fitted at the answer positions
                dl = torch.zeros(len(pd), D, device=X.device, requires_grad=True); op = torch.optim.Adam([dl], lr=lr_ * 0.1); yt = X[0, ps + 1]; comp = None
                for s in range(st_):
                    cq['delta'] = torch.zeros(1, T1, D, device=X.device).index_put((torch.zeros_like(pd), pd), dl)
                    lg = _ws_run(model, X, sl, grad_=True)[0, ps]; ce = F.cross_entropy(lg, yt, reduction='none')
                    l = (ce * wb).mean() if wb is not None else ce.mean(); l = l + 1e-3 * (dl ** 2).sum()
                    if selfc > 0:
                        if comp is None:                                          # the model's own top non-answer byte at each answer position (before fitting)
                            with torch.no_grad(): comp = lg.detach().scatter(1, yt[:, None], -1e9).argmax(1)
                        mg = lg.gather(1, yt[:, None])[:, 0] - lg.gather(1, comp[:, None])[:, 0]; l = l + selfc * F.softplus(-mg).mean()
                    op.zero_grad(); l.backward(); op.step(); cq['delta'] = None
                Ks.append(K1.float()); Rs.append(dl.detach().float()); Ns.append(nk.float()[None].expand(len(pd), -1)); Js.append(torch.arange(len(pd), device=X.device))
        return torch.cat(Ks), torch.cat(Rs), torch.cat(Ns), torch.cat(Js)
    def solve_cfg(K, R):
        H = K.shape[1]
        if Pm_ is not None:
            Kc = K @ Pm_; A = Kc.T @ Kc + 1e-2 * float((K.T @ K).diagonal().mean()) * torch.eye(H, device=K.device) + lam_ * len(K) * (Pm_ @ C0_ @ Pm_)
            return (torch.linalg.solve(A, Kc.T @ R).T) @ Pm_
        return ridge(K, R, C0_)
    def ridge(K, R, Cr):
        H = K.shape[1]; return torch.linalg.solve(K.T @ K + lam_ * len(K) * Cr.float() + 1e-4 * torch.eye(H, device=K.device), K.T @ R).T
    def measure():
        f_ = lambda w, dec: _ws_twohop(model, cfg, E, Cp_, S_slot, np.random.RandomState(3), w, dec)
        Rf = _tf_items(model, [(_CONC_CF[0].format(A=c['A']), c['C'], S_slot(c['d'])) for c in E])
        Rh = _tf_items(model, [(w.format(A=c['A']), c['C'], S_slot(c['d'])) for c in E for w in _HO_TWO])
        mc = np.nanmean([f_(w, False) for w in _HO_TWO])
        return (f"fitted: open {np.mean([r['hit'] for r in Rf]):.2f} | HELD-OUT: multiple choice (10) {mc:.2f}, open {np.mean([r['hit'] for r in Rh]):.2f}, "
                f"first byte top-1 {np.mean([r['rank'] <= 1 for r in Rh]):.2f}, answer bytes top-1 {np.mean([r['top1'] for r in Rh]):.2f}")
    def tf_nk(lst, dfun):
        """teacher-forced metrics with the name part of the edit read from each question's own name (weights hold the own-key part)"""
        out = []
        for w, c in lst:
            pr = w.format(A=c['A']).encode('latin-1', 'replace'); tb = (c['C'] + '.').encode('latin-1', 'replace'); X, M = _ws_batch(model, [pr + tb]); sl = int(S_slot(c['d']))
            T1 = X.shape[1] - 1; ps = torch.arange(len(pr) - 1, min(T1, len(pr) + len(tb) - 1), device=X.device)
            if len(ps) < len(tb): continue
            cap_['on'] = True; cap_['k'] = []
            with torch.no_grad(): _ws_run(model, X, sl)
            cap_['on'] = False; nk = cap_['k'][-1][0][name_pos(pr, c['A'], ps[0])].float()
            dv = scale * dfun(nk, len(ps)); cap_['delta'] = torch.zeros(1, T1, D, device=X.device).index_put((torch.zeros_like(ps), ps), dv)
            try:
                with torch.no_grad(): lg = _ws_run(model, X, sl)[0]
            finally: cap_['delta'] = None
            L = F.log_softmax(lg[ps].float(), -1); tr = torch.tensor(list(tb), device=X.device); top = L.argmax(-1)
            okb = (top[:-1] == tr[:-1]).float(); l0 = L[0]; t0 = int(tr[0])
            out.append(dict(hit=bool(okb.all()) and not _isl(int(top[-1])), top1=float(okb.mean()), rank=int((l0 > l0[t0]).sum()) + 1))
        return out
    def measure_nk(dfun):
        Rf = tf_nk([(_CONC_CF[0], c) for c in E], dfun); Rh = tf_nk([(w, c) for c in E for w in _HO_TWO], dfun)
        return (f"fitted: open {np.mean([r['hit'] for r in Rf]):.2f} | HELD-OUT: open {np.mean([r['hit'] for r in Rh]):.2f}, "
                f"first byte top-1 {np.mean([r['rank'] <= 1 for r in Rh]):.2f}, answer bytes top-1 {np.mean([r['top1'] for r in Rh]):.2f}")
    def _source_shadow():
        """SOURCE FACTS through the conclusion write path: recorded one-hop facts (the evidence record: subject, relation phrase, object), fitted
        as the edit fits conclusions, in the source wording ("A <relation> ") alone and with the edit's written wordings; measured in the
        source wording and in the held-out wordings, from the pre-edit weights (before) and after; open-ended = greedy produces the object"""
        from platypus.inference.sentences import _inf_store
        S9 = _inf_store(model); it = S9.get('it', []); rel = S9.get('rel', {}); rs9 = np.random.RandomState(int(cfg.seed) + 4545); fx = []
        isname = lambda b: len(b) >= 3 and 65 <= b[0] <= 90 and all(65 <= x <= 90 or 97 <= x <= 122 for x in b)
        try:                                                                       # 9 Nov: only records between names of the planted fact sentences (the evidence
            from platypus.inference.probes import inf_probes                       #   record also catches junk from prose, e.g. 'Transylvania on December');
            pn9 = {n.encode('latin-1', 'replace') for c in inf_probes(cfg) for n in c.get('names', [])}   #   readout only, nothing is trained on this
        except Exception: pn9 = None
        keys = list(rel.keys()); rs9.shuffle(keys)
        for (a9, r9) in keys:
            for o9, ix9 in rel[(a9, r9)].items():
                if isname(bytes(a9)) and isname(bytes(o9)) and ix9 and ix9[-1] < len(it) and (pn9 is None or (bytes(a9) in pn9 and bytes(o9) in pn9)):
                    fx.append((bytes(a9).decode('latin-1'), bytes(r9).decode('latin-1').strip(), bytes(o9).decode('latin-1'), it[ix9[-1]]['d'])); break
            if len(fx) >= 16: break
        if len(fx) < 8: return f"SOURCE FACTS skipped ({len(fx)} recorded name-to-name facts)"
        src = [(f"{a9} {r9} ", o9, d9, a9) for a9, r9, o9, d9 in fx]; ho = [(w.format(A=a9), o9, d9, a9) for a9, r9, o9, d9 in fx for w in _HO_TWO]
        def meas():
            R1 = _tf_items(model, [(p9, o9, S_slot(d9)) for p9, o9, d9, a9 in src]); R2 = _tf_items(model, [(p9, o9, S_slot(d9)) for p9, o9, d9, a9 in ho])
            return (f"source wording: open {np.mean([r['hit'] for r in R1]):.2f}, answer bytes top-1 {np.mean([r['top1'] for r in R1]):.2f} | HELD-OUT: open "
                    f"{np.mean([r['hit'] for r in R2]):.2f}, first byte top-1 {np.mean([r['rank'] <= 1 for r in R2]):.2f}, answer bytes top-1 {np.mean([r['top1'] for r in R2]):.2f}")
        with torch.no_grad(): mo_.weight.copy_(W0_)
        o9 = [f"no edit: {meas()}"]
        for nm9, its9 in (('source wording only', src), (f'source wording + the {nw} edit wordings', src + [(w.format(A=a9), o9_, d9, a9) for a9, r9, o9_, d9 in fx for w in _CONC_CF[:nw]])):
            K9, R9, N9, J9 = fit(None, ws_self, 0.0, its9); put(solve_cfg(K9, R9)); o9.append(f"{nm9} (+ self-contrast): {meas()}")
        return f"SOURCE FACTS ({len(fx)} recorded one-hop facts between planted names, e.g. '{fx[0][0]} {fx[0][1]} {fx[0][2]}') :: " + " :: ".join(o9)
    def _slot_gate(FITs, put):
        """ANSWER-SLOT GATES (north-star rung 3 without the oracle; readout). The NAME x POSITION memory needs to know WHERE an answer starts
        and WHOSE answer it is; neither is given. SUBJECT REGISTER: the most recent name-like word before the position (capitalised, 3+
        letters: the store's own name pattern); its last byte's key is the name key the memory reads. Two ways to decide where to fire,
        each trained only on the system's own material (this sleep's conclusions in the written edit wordings, recorded name-to-name facts in
        their source wording, ordinary stored sentences); the held-out question wordings are never seen:
          LINEAR: logistic regression on the edited layer's key at the byte (64 principal directions);
          SERAC-STYLE (Mitchell et al. 2022): a small two-layer gate on the same key plus the bytes since the register's name ended, trained on
            MORE FORMS the system writes itself (relation-agnostic: 'Notes on A: ...', 'Q: A <rel> whom? A: ', 'It is recorded that A <rel> ',
            'A <rel> ... '; never a test wording) and HARD NEGATIVES (the 1-3 bytes after a name in ordinary sentences, weighted x3);
          (A WISE-style self-firing memory was removed 10 Nov: negative on both scales, see RESULTS.md.)
        Thresholds for the gates: the 99th percentile on held-in ordinary sentences. Measured on held-out
        wordings of this sleep's conclusions: fires at the slot / fires earlier in the prompt / recall with the memory applied from the FIRST
        firing byte for as many bytes as the longest fitted answer; and on held-out ordinary sentences: fires per 1000 bytes, BPC change with
        the memory applied at the first firing byte of each sentence. Pre-edit weights, store off, the memory at the kept scale."""
        import re as _re
        from platypus.inference.sentences import _inf_store
        K, R, N, J = FITs; H = K.shape[1]; cI = float(C0_.diagonal().mean()) * torch.eye(H, device=C0_.device)
        PJ = {j: ridge(N[J == j], R[J == j], cI) for j in range(int(J.max()) + 1) if int((J == j).sum()) >= 2}
        if not PJ: return "ANSWER-SLOT GATES skipped (no memory)"
        jmax = max(PJ) + 1; put(torch.zeros_like(W0_, dtype=torch.float32))
        S9 = _inf_store(model); it = S9.get('it', []); rel = S9.get('rel', {}); rs9 = np.random.RandomState(int(cfg.seed) + 4747)
        isname = lambda b: len(b) >= 3 and 65 <= b[0] <= 90 and all(65 <= x <= 90 or 97 <= x <= 122 for x in b)
        def keys_of(txt, sl):                                                     # the edited layer's keys at every byte (store off, pre-edit weights)
            X, M = _ws_batch(model, [txt]); cap_['on'] = True; cap_['k'] = []
            with torch.no_grad(): _ws_run(model, X, sl)
            cap_['on'] = False; return X, cap_['k'][-1][0].float()                # (T-1, H): key at t predicts byte t+1
        def regs(txt, T):                                                         # SUBJECT REGISTER per byte: last byte of the latest name-like word ending before t
            out = np.full(T, -1, dtype=np.int64)
            for m in _re.finditer(rb"[A-Za-z]+", txt):
                if isname(m.group(0)): out[m.end():] = m.end() - 1               # bytes after the name (the name's own bytes do not see it yet)
            return out[:T]
        # ---- material
        facts = []
        for (a9, r9) in list(rel.keys())[:4000]:
            if len(facts) >= 64: break
            for o9, ix9 in rel[(a9, r9)].items():
                if isname(bytes(a9)) and isname(bytes(o9)) and ix9 and ix9[-1] < len(it):
                    facts.append((bytes(a9).decode('latin-1'), bytes(r9).decode('latin-1').strip(), bytes(o9).decode('latin-1'), it[ix9[-1]]['d'])); break
        base = [(w.format(A=c['A']), c['C'], c['d']) for c in E for w in words] + [(f"{a} {r} ", o, d) for a, r, o, d in facts]
        FC = ("Notes on {A}: {A} {r} ", "Q: {A} {r} whom? A: ", "It is recorded that {A} {r} ", "{A} {r} ... ")   # relation-agnostic, never a test wording
        more = [(f.format(A=c['A'], r='is connected to'), c['C'], c['d']) for c in E for f in FC] + [(f.format(A=a, r=r), o, d) for a, r, o, d in facts for f in FC]
        real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
        sel = [real[int(k)] for k in rs9.choice(len(real), min(400, len(real)), replace=False)] if real else []
        held_in, held_out = sel[:len(sel) // 2], sel[len(sel) // 2:]
        def collect(items):                                                       # (CPU keys, register per byte, slot byte) per training text
            out = []
            for pr, C, d in items:
                pb = pr.encode('latin-1', 'replace'); txt = pb + (C + '.').encode('latin-1', 'replace'); X, Kt = keys_of(txt, int(S_slot(d)))
                out.append((Kt.cpu(), regs(txt, Kt.shape[0]), len(pb) - 1))
            return out
        TB, TM = collect(base), collect(more)
        TO = []
        for x in held_in:
            txt = bytes(x['s']); X, Kt = keys_of(txt, int(S_slot(x['d']))); TO.append((Kt.cpu(), regs(txt, Kt.shape[0])))
        if len(TB) < 16 or not TO: return f"ANSWER-SLOT GATES skipped ({len(TB)} training texts)"
        Kall = torch.cat([k for k, _, _ in TB] + [k for k, _ in TO]); mu = Kall.mean(0)
        U, Sv, V = torch.linalg.svd(Kall[torch.randperm(len(Kall), generator=torch.Generator().manual_seed(7))[:4000]] - mu, full_matrices=False)
        P = V[:64].T; sd = ((Kall - mu) @ P).std(0) + 1e-6; del Kall
        f = lambda Kt: ((Kt.float().cpu() - mu) @ P) / sd
        sub30 = lambda n, s: torch.randperm(n, generator=torch.Generator().manual_seed(s))[:30]
        def fit_logit(Xp, Xn, wn=None, hidden=0):
            g0 = torch.Generator().manual_seed(11); din = Xp.shape[1]
            W1 = (torch.randn(din, hidden, generator=g0) / din ** 0.5).requires_grad_(True) if hidden else None
            w = torch.zeros(hidden or din, requires_grad=True); b = torch.zeros(1, requires_grad=True)
            ps = [w, b] + ([W1] if hidden else []); op = torch.optim.Adam(ps, lr=0.03 if hidden else 0.05)
            fwd = (lambda Z: F.relu(Z @ W1) @ w + b) if hidden else (lambda Z: Z @ w + b)
            wn = torch.ones(len(Xn)) if wn is None else wn; wp = float(wn.sum()) / len(Xp)
            with torch.enable_grad():
                for _ in range(400 if hidden else 300):
                    loss = -(wp * F.logsigmoid(fwd(Xp)).sum() + (wn * F.logsigmoid(-fwd(Xn))).sum()) / (wp * len(Xp) + float(wn.sum()))
                    loss = loss + 1e-3 * sum((q ** 2).sum() for q in ps if q is not b); op.zero_grad(); loss.backward(); op.step()
            return lambda Z: fwd(Z).detach()
        # ---- LINEAR (as before: base texts + ordinary)
        Lp = torch.stack([f(k[sp:sp + 1])[0] for k, _, sp in TB if sp < len(k)])
        Ln = torch.cat([f(torch.cat([k[:sp], k[sp + 1:]])) for k, _, sp in TB] + [f(k[sub30(len(k), i)]) for i, (k, _) in enumerate(TO)])
        lin = fit_logit(Lp, Ln)
        # ---- SERAC-STYLE (more forms, hard negatives, recency feature, two layers)
        def fz(k, rg):                                                            # key features + log bytes since the register's name + has-register
            t = torch.arange(len(k)); rr = torch.tensor(rg[:len(k)]); hr = (rr >= 0).float()
            return torch.cat([f(k), (torch.log1p((t - rr).clamp(min=0).float()) * hr)[:, None], hr[:, None]], 1)
        Sp, Sn, Sw = [], [], []
        for k, rg, sp in TB + TM:
            Z = fz(k, rg)
            if sp < len(Z): Sp.append(Z[sp]); Sn.append(torch.cat([Z[:sp], Z[sp + 1:]])); Sw.append(torch.ones(len(Z) - 1))
        for i, (k, rg) in enumerate(TO):
            Z = fz(k, rg); ix = sub30(len(Z), i); Sn.append(Z[ix]); Sw.append(torch.ones(len(ix)))
            hard = [t for t in range(1, len(Z)) if rg[t] >= 0 and 1 <= t - rg[t] <= 3]   # HARD NEGATIVES: just after a name in ordinary text
            if hard: Sn.append(Z[hard]); Sw.append(torch.full((len(hard),), 3.0))
        ser = fit_logit(torch.stack(Sp), torch.cat(Sn), torch.cat(Sw), hidden=32)
        # ---- scoring of a text by each gate
        def score(kind, k, rg):
            if kind == 'linear': return lin(f(k))
            return ser(fz(k, rg))
        ordo = []
        for x in held_out:
            txt = bytes(x['s']); sl = int(S_slot(x['d'])); X, Kt = keys_of(txt, sl); ordo.append((txt, sl, X, Kt, regs(txt, Kt.shape[0])))
        qs = []
        for c in E:
            for w9 in _HO_TWO:
                pb = w9.format(A=c['A']).encode('latin-1', 'replace'); tb = (c['C'] + '.').encode('latin-1', 'replace'); txt = pb + tb; sl = int(S_slot(c['d']))
                X, Kt = keys_of(txt, sl); qs.append((txt, sl, X, Kt, regs(txt, Kt.shape[0]), len(pb) - 1, tb))
        held_in_scores = {kind: torch.cat([score(kind, k, rg) for k, rg in TO]) for kind in ('linear', 'serac')}
        def apply_memory(X, Kt, sl, rg, t0):                                      # name-keyed memory bytes from t0 on, keyed by the register at t0
            ne = int(rg[t0]) if t0 < len(rg) else -1
            if ne < 0: return None
            nk = Kt[ne]; T1 = X.shape[1] - 1; ps = torch.arange(t0, min(T1, t0 + jmax), device=X.device)
            dv = scale * torch.stack([(PJ[j] @ nk) if j in PJ else torch.zeros(D, device=X.device) for j in range(len(ps))])
            cap_['delta'] = torch.zeros(1, T1, D, device=X.device).index_put((torch.zeros_like(ps), ps), dv.to(X.device))
            try:
                with torch.no_grad(): return _ws_run(model, X, sl)[0]
            finally: cap_['delta'] = None
        def evaluate(kind, th):
            at, early, hit, fb, ab = [], [], [], [], []
            for txt, sl, X, Kt, rg, sp, tb in qs:
                sc = score(kind, Kt.cpu(), rg); fired = [t for t in range(len(sc)) if float(sc[t]) > th]
                at.append(sp in fired); early.append(any(t < sp for t in fired)); t0 = fired[0] if fired else None
                lg = apply_memory(X, Kt, sl, rg, t0) if t0 is not None and t0 <= sp + 2 else None
                if lg is None:
                    with torch.no_grad(): lg = _ws_run(model, X, sl)[0]
                ps = torch.arange(sp, sp + len(tb), device=X.device); ps = ps[ps < lg.shape[0]]
                if len(ps) < len(tb): continue
                L = F.log_softmax(lg[ps].float(), -1); tr = torch.tensor(list(tb), device=X.device); top = L.argmax(-1); okb = (top[:-1] == tr[:-1]).float()
                hit.append(bool(okb.all()) and not _isl(int(top[-1]))); fb.append(int(top[0]) == int(tr[0])); ab.append(float(okb.mean()))
            nfire, nby, dbits = 0, 0, 0.0
            for txt, sl, X, Kt, rg in ordo:
                sc = score(kind, Kt.cpu(), rg); fired = [t for t in range(len(sc)) if float(sc[t]) > th]; nfire += len(fired); nby += len(sc)
                if not fired: continue
                lg1 = apply_memory(X, Kt, sl, rg, fired[0])
                if lg1 is None: continue
                with torch.no_grad(): l0 = F.cross_entropy(_ws_run(model, X, sl)[0].float(), X[0, 1:], reduction='sum')
                dbits += float(F.cross_entropy(lg1.float(), X[0, 1:], reduction='sum') - l0) / np.log(2)
            return (f"fires AT the slot {np.mean(at):.2f}, earlier {np.mean(early):.2f} | recall (no oracle): open {np.mean(hit):.2f}, first byte {np.mean(fb):.2f}, "
                    f"answer bytes {np.mean(ab):.2f} | ordinary text: {1000 * nfire / max(1, nby):.1f} fires per 1000 bytes, BPC {dbits / max(1, nby):+.4f}")
        res = []
        for kind, nm in (('linear', 'LINEAR'), ('serac', 'SERAC-STYLE (more forms, hard negatives, recency, 2 layers)')):
            th = float(torch.quantile(held_in_scores[kind], 0.99)); res.append(f"{nm} at 1% of ordinary text: {evaluate(kind, th)}")
        return (f"{len(TB)} base + {len(TM)} extra-form training texts ({len(E)} conclusions, {len(facts)} recorded facts), {len(TO)} ordinary sentences; "
                f"memory: NAME x POSITION + self-contrast, {jmax} bytes; held-out questions {len(qs)} || " + " || ".join(res))
    out = []
    try:
        with torch.no_grad(): mo_.weight.copy_(W0_)
        out.append(f"no edit: {measure()}")
        words = list(_CONC_CF[:nw]); FIT = {0.0: fit(words, 0.0), ws_self: fit(words, ws_self)}
        def put(dW):
            with torch.no_grad(): mo_.weight.copy_((W0_.float() + scale * dW.float()).to(W0_.dtype))
        H = FIT[0.0][0].shape[1]; cI = float(C0_.diagonal().mean()) * torch.eye(H, device=C0_.device)
        for sc, tag in ((0.0, ''), (ws_self, f' + SELF-CONTRAST x{ws_self:g}')):
            K, R, N, J = FIT[sc]
            put(solve_cfg(K, R)); out.append(f"as configured ({nw} wording(s)){tag}: {measure()}")
            put(torch.zeros_like(W0_, dtype=torch.float32))
            PJ = {}                                                                # NAME x POSITION: one small memory per answer byte j, keyed by the name
            for j in range(int(J.max()) + 1):
                m = J == j
                if int(m.sum()) >= 2: PJ[j] = ridge(N[m], R[m], cI)
            zero = torch.zeros(D, device=N.device)
            out.append(f"NAME x POSITION (a memory per answer byte, keyed by the name){tag}: "
                       f"{measure_nk(lambda nk, n: torch.stack([(PJ[j] @ nk) if j in PJ else zero for j in range(n)]))}")
        iwc = float(getattr(cfg, 'inf_cons_infow', 0.0)) or 4.0                                                              # INFORMATION-WEIGHTED edit fits (as configured solver)
        for sc, tag in ((0.0, ''), (ws_self, f' + SELF-CONTRAST x{ws_self:g}')):
            K, R, N, J = fit(words, sc, iwc); put(solve_cfg(K, R)); out.append(f"as configured{tag} + INFORMATION-WEIGHTED bytes (cap {iwc:g}): {measure()}")
        try: out.append(_source_shadow())
        except Exception as e9: cap_['delta'] = None; cap_['on'] = False; out.append(f"SOURCE FACTS unavailable ({type(e9).__name__}: {str(e9)[:80]})")
        if int(getattr(cfg, 'inf_slot_gate', 1)):
            try: print(f"  [ANSWER-SLOT GATES | visit {int(model.__dict__.get('_visit', 0)) + 1}] " + _slot_gate(FIT[ws_self], put), flush=True)
            except Exception as e7: cap_['delta'] = None; cap_['on'] = False; _fail('inference.openread:slotgate'); print(f"  [ANSWER-SLOT GATES] unavailable ({type(e7).__name__}: {str(e7)[:160]})", flush=True)
            with torch.no_grad(): mo_.weight.copy_(W0_)
        print(f"  [CF SHADOW EDITS] {len(E)} of this sleep's conclusions, each variant fitted on them alone and applied to the pre-edit weights at x{scale:g}, measured, removed "
              f"(the kept edit is restored) || " + " || ".join(out), flush=True)
    except Exception as e:
        _fail('inference.openread:shadow'); print(f"  [CF SHADOW EDITS] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
    finally:
        cap_['delta'] = None; cap_['on'] = False; cap_['k'] = []                 # 10 Nov: capture can never be left on (it would keep every forward's keys)
        with torch.no_grad(): mo_.weight.copy_(Wk)
