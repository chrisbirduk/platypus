"""NAME MEMORY (inf_name_mem, 10 Nov; on by default). A binding memory beside the weights - a hippocampus to the trunk's cortex - that
answers questions about stored facts in wordings never written. Local evidence (RESULTS.md, 10 Nov): end to end, no oracle, open-ended
recall in held-out wordings 0.71-0.87 (relation in its own wording) and 0.30-0.46 (relation paraphrased) against 0.00-0.08 without it,
at no cost on ordinary text. It never changes the weights; it acts in the forward pass only where the fast store acts (store-on
evaluation; never in training, never in 'weights only' / 'from the weights' readouts), so its effect is reported in the store-on column.

ENTRIES (one per fact): the system's own live conclusions ('A is connected to C', written in inf_cf_wordings edit wordings) and the
  evidence record's name-to-name facts ('A <relation> O', the latest object per subject and relation; seen in at least
  inf_nm_min_seen sentences, most seen first - on the GPU the newest were prose co-occurrences seen once), up to inf_nm_max.
KEY (two parts): the edited layer's input (blocks[inf_cf_layer].mlp_out) at the subject name's last byte, and at the byte where the
  answer starts; each averaged over the entry's written wordings.
VALUE: the entry's own change to that layer's output at the answer bytes, fitted on its written wordings (inf_cf_steps Adam steps,
  self-contrast inf_nm_selfc: each answer byte must beat the model's own best non-answer byte).
EVERY SLEEP (after the closed-form edit): all keys are re-read from the current weights (the trunk keeps learning, so keys drift);
  each kept value is checked (applied at its written wordings, does it still produce the answer?) and refitted only if not; new entries
  are fitted; entries no longer in the material are dropped (retracted / freed conclusions, superseded facts).
READ, at every byte: SUBJECT REGISTER = the latest name-like word ending before the byte (capitalised, 3+ letters); name part = centred
  cosine of the register's name key with each entry's; question part = the same for the byte's own key. GATE: a SERAC-style slot gate
  (Mitchell et al. 2022; key in 64 principal directions + bytes since the register's name; two layers; trained each sleep on the
  written wordings, four more self-written forms and ordinary sentences with hard negatives just after names) fires past its 99th
  percentile on held-in ordinary bytes, AND a stored name is present (name part past its own 99th percentile on ordinary text).
  Ordinary text for these thresholds = stored sentences that name no stored subject (recorded facts come from ordinary text).
  RE-ANCHORING: every firing restarts the memory from that byte; the entry is the best two-part match at the firing byte.
READOUT each sleep: held-out recall with the memory off and on (conclusions in the standard held-out wordings; facts in a new frame
  and with a paraphrased relation), and MISFIRES: firing rate and BPC change on text that names stored subjects without asking about them,
  and on ordinary text."""
import re
import time
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import _fail, _CONC_CF, _HO_TWO

_NM = {'ready': False, 'X': None, 'force': None, 'hooked': None, 'stats': [0, 0], 'mode': 'hidden', 'plan': None}
_NM_PARA = {'works with': 'is a colleague of', 'lives near': 'is a neighbour of', 'reports to': 'is managed by', 'is a partner of': 'is partnered with',
            'trained under': 'was taught by', 'is connected to': 'is tied to'}
_NM_FORMS = ("Notes on {A}: {A} {r} ", "Q: {A} {r} whom? A: ", "In short, {A} {r} ", "{A} {r} ... ")   # self-written, relation-agnostic, never a test wording
_NM_REL = re.compile(r"^[a-z]+( [a-z]+){0,3}$")
_NM_WORD = re.compile(rb"[A-Za-z]+")


def _nm_isname(b): return len(b) >= 3 and 65 <= b[0] <= 90 and all(65 <= x <= 90 or 97 <= x <= 122 for x in b)


def _nm_regs(txt, T):
    """SUBJECT REGISTER per byte: index of the last byte of the latest name-like word ending before it (-1: none yet)"""
    out = np.full(T, -1, dtype=np.int64)
    for m in _NM_WORD.finditer(txt):
        if _nm_isname(m.group(0)): out[m.end():] = m.end() - 1
    return out[:T]


def _nm_layer(model, cfg):
    return model.blocks[int(getattr(cfg, 'inf_cf_layer', -1))].mlp_out


# ---------------------------------------------------------------- the forward pass
def _nm_active():
    if not _NM['ready'] or torch.is_grad_enabled(): return False
    if _NM['force'] is not None: return bool(_NM['force'])
    from platypus.inference.store import _INFM
    return bool(_INFM['force']) if 'force' in _INFM else bool(_INFM.get('on', False))


def _nm_hook(m, inp, out):
    if not _nm_active(): return None
    X = _NM['X']
    if X is None or X.dim() != 2 or out.dim() != 3 or X.shape[0] != out.shape[0] or X.shape[1] != out.shape[1]: return None
    try:
        M = _NM['M']; K = inp[0].float(); B, T, H = K.shape; dev = K.device
        Xc = X.detach().to('cpu', torch.uint8).numpy()
        rg = torch.tensor(np.stack([_nm_regs(bytes(Xc[b].tolist()), T) for b in range(B)]), device=dev)
        ok = rg >= 0; Kn = K.gather(1, rg.clamp(min=0)[..., None].expand(-1, -1, H))
        sn = F.normalize(Kn - M['mN'], dim=-1) @ M['NMc'].T; sq = F.normalize(K - M['mQ'], dim=-1) @ M['QMc'].T
        t = torch.arange(T, device=dev)[None].expand(B, -1); hr = ok.float()
        Z = torch.cat([((K - M['muS']) @ M['PS']) / M['sdS'], (torch.log1p((t - rg).clamp(min=0).float()) * hr)[..., None], hr[..., None]], -1)
        g = F.relu(Z @ M['W1']) @ M['w'] + M['b']
        fire = (g > M['th_s']) & (sn.max(-1).values > M['th_n']) & ok
        if not bool(fire.any()): return None
        item = (sn + sq).argmax(-1)
        last = torch.where(fire, t, torch.full_like(t, -1)).cummax(1).values
        il = item.gather(1, last.clamp(min=0)); off = t - last
        valid = (last >= 0) & (off < M['L'][il])
        dv = M['V'][il, off.clamp(min=0, max=M['V'].shape[1] - 1)] * valid[..., None].float()
        _NM['stats'][0] += int(fire.sum()); _NM['stats'][1] += B * T
        if _NM['mode'] == 'logit':                                             # OUTPUT-LEVEL VALUE: the answer byte is boosted in the logits
            _NM['plan'] = (il, off.clamp(min=0, max=M['ANS'].shape[1] - 1), valid); return None   #   (applied by the Model.forward wrapper)
        return out + dv.to(out.dtype)
    except Exception:
        _fail('inference.namemem:hook'); return None


def _nm_boost(z, pl):
    """OUTPUT-LEVEL VALUE (inf_nm_value logit): at each byte the memory covers, the stored answer byte is raised just enough to beat the
    model's best other byte by inf_nm_margin nats, computed on the fly from the current logits - nothing fitted to the trunk can go stale"""
    try:
        il, off, valid = pl; M = _NM['M']; tg = M['ANS'][il, off]                # (B,T) target byte per position
        zf = z.float(); za = zf.gather(-1, tg[..., None])[..., 0]; zo = zf.scatter(-1, tg[..., None], -1e9).max(-1).values
        b = (zo - za + M['margin']).clamp(min=0) * valid.float()
        return z + (torch.zeros_like(zf).scatter(-1, tg[..., None], b[..., None])).to(z.dtype)
    except Exception:
        _fail('inference.namemem:boost'); return z


def _nm_install(model, cfg):
    """the forward hook on the edited layer and the byte stash on Model.forward (once)"""
    mo = _nm_layer(model, cfg)
    if _NM['hooked'] is not mo:
        if _NM.get('h') is not None: _NM['h'].remove()
        _NM['h'] = mo.register_forward_hook(_nm_hook); _NM['hooked'] = mo
    from platypus.model import Model
    if not getattr(Model.forward, '_nm_stash', False):
        f0 = Model.forward
        def fwd(s, x, domain=0):
            _NM['X'] = x; _NM['plan'] = None
            z = f0(s, x, domain)
            pl = _NM['plan']; _NM['plan'] = None
            if pl is not None and torch.is_tensor(z) and z.dim() == 3 and z.shape[:2] == pl[0].shape: z = _nm_boost(z, pl)
            return z
        fwd._nm_stash = True; Model.forward = fwd


# ---------------------------------------------------------------- material
def _nm_material(model, cfg, S_):
    it = S_.get('it', []); nmax = int(getattr(cfg, 'inf_nm_max', 384)); nw = max(1, min(len(_CONC_CF), int(getattr(cfg, 'inf_cf_wordings', 1))))
    lv = [c for c in S_.get('conc', []) if not c.get('freed') and not c.get('retracted') and isinstance(c.get('A'), str) and isinstance(c.get('C'), str)]
    out, seen = [], set()
    for c in reversed(lv):                                                     # newest first
        k = ('conclusion', c['A'], 'is connected to', c['C'])
        if k in seen: continue
        seen.add(k); out.append(dict(key=k, A=c['A'], r='is connected to', ans=c['C'], d=c['d'], pfs=[w.format(A=c['A']) for w in _CONC_CF[:nw]]))
        if len(out) >= nmax // 2: break
    fx = []; mins = int(getattr(cfg, 'inf_nm_min_seen', 2))
    for (a, r), objs in S_.get('rel', {}).items():
        a = bytes(a); r = bytes(r).decode('latin-1', 'replace').strip()
        if not _nm_isname(a) or not _NM_REL.match(r): continue
        best = None                                                            # the latest object for this subject and relation (an update replaces)
        for o, ix in objs.items():
            o = bytes(o)
            if _nm_isname(o) and ix and ix[-1] < len(it) and (best is None or ix[-1] > best[1]): best = (o, ix[-1], len(set(ix)))
        if best is not None and best[2] >= mins: fx.append((best[2], best[1], a.decode('latin-1'), r, best[0].decode('latin-1'), it[best[1]]['d']))
    for n9, _, A, r, o, d in sorted(fx, reverse=True)[:max(0, nmax - len(out))]:   # MOST SEEN first, then newest (10 Nov, GPU: the newest
        out.append(dict(key=('fact', A, r, o), A=A, r=r, ans=o, d=d, pfs=[f"{A} {r} "], seen=n9))   #   were prose co-occurrences seen once)
    model.__dict__['_nm_cand'] = (len(fx), sum(1 for x in fx if x[0] >= 3))
    return out


# ---------------------------------------------------------------- keys, values, gate (sleep)
def _nm_run(model, txts, sl, cap, delta=None, grad=False):
    from platypus.inference.closed_form import _ws_batch, _ws_run
    X, Mk = _ws_batch(model, txts); cap['on'] = True; cap['k'] = []; cap['delta'] = delta(X) if delta is not None else None
    try: lg = _ws_run(model, X, sl, grad_=grad)
    finally: cap['on'] = False; cap['delta'] = None
    return X, cap['k'][-1], lg


def _nm_texts(ents):
    """per written text: (entry index, prompt bytes, answer bytes, slot position, name end)"""
    T = []
    for e9, e in enumerate(ents):
        tb = (e['ans'] + '.').encode('latin-1', 'replace'); ab = e['A'].encode('latin-1', 'replace')
        for pf in e['pfs']:
            pb = pf.encode('latin-1', 'replace'); i0 = pb.find(ab)
            if i0 < 0: continue
            T.append((e9, pb, tb, len(pb) - 1, i0 + len(ab) - 1))
    return T


def _nm_fit_values(model, cfg, ents, idx, S_slot, cap, D, bs=24):
    """fits V for ents[i], i in idx, on their written wordings; batched, one optimiser per batch of entries, gradients to the values only"""
    lr = float(getattr(cfg, 'inf_cf_lr', 0.5)) * 0.1; st = int(getattr(cfg, 'inf_cf_steps', 25)); selfc = float(getattr(cfg, 'inf_nm_selfc', 1.0))
    bysl = {}
    for i in idx: bysl.setdefault(int(S_slot(ents[i]['d'])), []).append(i)
    for sl, L in bysl.items():
        for c0 in range(0, len(L), bs):
            ch = L[c0:c0 + bs]; TX = _nm_texts([ents[i] for i in ch])
            if not TX: continue
            Lmax = max(len(tb) for _, _, tb, _, _ in TX); dev = next(model.parameters()).device
            DL = torch.zeros(len(ch), Lmax, D, device=dev, requires_grad=True); op = torch.optim.Adam([DL], lr=lr)
            rows, pos, ent, jj, tgt_ok = [], [], [], [], []
            from platypus.inference.closed_form import _ws_batch
            X, _ = _ws_batch(model, [pb + tb for _, pb, tb, _, _ in TX]); T1 = X.shape[1] - 1
            for r9, (e9, pb, tb, p0, ne) in enumerate(TX):
                for j in range(len(tb)):
                    if p0 + j < T1: rows.append(r9); pos.append(p0 + j); ent.append(e9); jj.append(j)
            rows = torch.tensor(rows, device=dev); pos = torch.tensor(pos, device=dev); ent = torch.tensor(ent, device=dev); jj = torch.tensor(jj, device=dev)
            yt = X[rows, pos + 1]; comp = None; txt = [pb + tb for _, pb, tb, _, _ in TX]
            for s in range(st):
                with torch.enable_grad():
                    dlt = lambda X_: torch.zeros(X_.shape[0], X_.shape[1] - 1, D, device=dev).index_put((rows, pos), DL[ent, jj])
                    _, _, lg = _nm_run(model, txt, sl, cap, delta=dlt, grad=True)
                    lg = lg[rows, pos]; l = F.cross_entropy(lg, yt)
                    if selfc > 0:
                        if comp is None: comp = lg.detach().scatter(1, yt[:, None], -1e9).argmax(1)
                        l = l + selfc * F.softplus(-(lg.gather(1, yt[:, None])[:, 0] - lg.gather(1, comp[:, None])[:, 0])).mean()
                    l = l + 1e-3 * (DL ** 2).sum() / len(ch)
                    gr, = torch.autograd.grad(l, [DL])
                op.zero_grad(); DL.grad = gr; op.step()
            for k, i in enumerate(ch): ents[i]['V'] = DL[k, :len(ents[i]['ans']) + 1].detach().clone()


def _nm_keys_and_check(model, ents, S_slot, cap, bs=48):
    """re-reads both keys of every entry from the current weights; for entries with a value, checks whether it still produces the answer"""
    bysl = {}
    for i, e in enumerate(ents): bysl.setdefault(int(S_slot(e['d'])), []).append(i)
    nk = {i: [] for i in range(len(ents))}; qk = {i: [] for i in range(len(ents))}; good = {i: [] for i in range(len(ents))}
    for sl, L in bysl.items():
        TX = [(L[e9], pb, tb, p0, ne) for e9, pb, tb, p0, ne in _nm_texts([ents[i] for i in L])]
        for c0 in range(0, len(TX), bs):
            ch = TX[c0:c0 + bs]; txt = [pb + tb for _, pb, tb, _, _ in ch]
            with torch.no_grad(): X, K, _ = _nm_run(model, txt, sl, cap)
            for r9, (i, pb, tb, p0, ne) in enumerate(ch): nk[i].append(K[r9, ne].float().cpu()); qk[i].append(K[r9, p0].float().cpu())
            hv = [r9 for r9, (i, _, _, _, _) in enumerate(ch) if ents[i].get('V') is not None]
            if not hv: continue
            def dlt(X_):
                d = torch.zeros(X_.shape[0], X_.shape[1] - 1, ents[ch[hv[0]][0]]['V'].shape[1], device=X_.device)
                for r9 in hv:
                    i, pb, tb, p0, ne = ch[r9]; V = ents[i]['V']; n = min(len(V), d.shape[1] - p0)
                    if n > 0: d[r9, p0:p0 + n] = V[:n].to(X_.device)
                return d
            with torch.no_grad(): X, K, lg = _nm_run(model, txt, sl, cap, delta=dlt)
            for r9 in hv:
                i, pb, tb, p0, ne = ch[r9]; ps = torch.arange(p0, p0 + len(tb) - 1, device=lg.device); ps = ps[ps < lg.shape[1]]
                good[i].append(bool((lg[r9, ps].argmax(-1) == X[r9, ps + 1]).all()) if len(ps) == len(tb) - 1 else False)
    for i, e in enumerate(ents):
        if nk[i]: e['nk'] = torch.stack(nk[i]).mean(0); e['qk'] = torch.stack(qk[i]).mean(0)
        e['ok'] = bool(good[i]) and all(good[i])


def _nm_gate(model, cfg, ents, S_, S_slot, cap, seed):
    """SERAC-style slot gate and thresholds (see the module note). Returns the gate tensors, held-out ordinary sentences, and the forms"""
    rs = np.random.RandomState(seed); it = S_.get('it', []); sb = set(e['A'].encode('latin-1', 'replace') for e in ents)
    real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200                 # ORDINARY = sentences naming no stored subject (recorded
            and not (set(m.group(0) for m in _NM_WORD.finditer(bytes(x['s']))) & sb)]        #   facts come from such text; the misfire readout measures those)
    sel = [real[int(k)] for k in rs.choice(len(real), min(320, len(real)), replace=False)] if real else []
    hi, ho = sel[:len(sel) // 2], sel[len(sel) // 2:]
    W = []                                                                     # (keys (T,H) cpu, register, slot) of written texts and extra forms
    bysl = {}
    for i, e in enumerate(ents): bysl.setdefault(int(S_slot(e['d'])), []).append(i)
    fsub = set(int(k) for k in rs.choice(len(ents), min(64, len(ents)), replace=False))
    for sl, L in bysl.items():
        txs = []
        for i in L:
            e = ents[i]; tb = (e['ans'] + '.').encode('latin-1', 'replace')
            for pf in e['pfs'] + ([f.format(A=e['A'], r=e['r']) for f in _NM_FORMS] if i in fsub else []):
                pb = pf.encode('latin-1', 'replace'); txs.append((pb + tb, len(pb) - 1))
        for c0 in range(0, len(txs), 48):
            ch = txs[c0:c0 + 48]
            with torch.no_grad(): X, K, _ = _nm_run(model, [t for t, _ in ch], sl, cap)
            for r9, (t, p0) in enumerate(ch):
                n = min(len(t) - 1, K.shape[1]); W.append((K[r9, :n].float().cpu(), _nm_regs(t, n), p0))
    O = []
    for x in hi:
        t = bytes(x['s'])
        with torch.no_grad(): X, K, _ = _nm_run(model, [t], int(S_slot(x['d'])), cap)
        n = min(len(t) - 1, K.shape[1]); O.append((K[0, :n].float().cpu(), _nm_regs(t, n)))
    Kall = torch.cat([k for k, _, _ in W] + [k for k, _ in O]); muS = Kall.mean(0)
    Vs = torch.linalg.svd(Kall[torch.randperm(len(Kall), generator=torch.Generator().manual_seed(7))[:4000]] - muS, full_matrices=False)[2]
    PS = Vs[:64].T.contiguous(); sdS = ((Kall - muS) @ PS).std(0) + 1e-6; del Kall
    def feat(k, rg):
        t = torch.arange(len(k)); rr = torch.tensor(rg[:len(k)]); hr = (rr >= 0).float()
        return torch.cat([((k - muS) @ PS) / sdS, (torch.log1p((t - rr).clamp(min=0).float()) * hr)[:, None], hr[:, None]], 1)
    Sp, Sn, Sw = [], [], []
    for k, rg, sp in W:
        Z = feat(k, rg)
        if sp < len(Z): Sp.append(Z[sp]); Sn.append(torch.cat([Z[:sp], Z[sp + 1:]])); Sw.append(torch.ones(len(Z) - 1))
    for i9, (k, rg) in enumerate(O):
        Z = feat(k, rg); ix = torch.randperm(len(Z), generator=torch.Generator().manual_seed(i9))[:30]; Sn.append(Z[ix]); Sw.append(torch.ones(len(ix)))
        hard = [t for t in range(1, len(Z)) if rg[t] >= 0 and 1 <= t - rg[t] <= 3]
        if hard: Sn.append(Z[hard]); Sw.append(torch.full((len(hard),), 3.0))
    Xp, Xn, wn = torch.stack(Sp), torch.cat(Sn), torch.cat(Sw); din = Xp.shape[1]; g0 = torch.Generator().manual_seed(11)
    W1 = (torch.randn(din, 32, generator=g0) / din ** 0.5).requires_grad_(True); w = torch.zeros(32, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    op = torch.optim.Adam([w, b, W1], lr=0.03); fwd = lambda Z: F.relu(Z @ W1) @ w + b; wp = float(wn.sum()) / len(Xp)
    with torch.enable_grad():
        for _ in range(400):
            loss = -(wp * F.logsigmoid(fwd(Xp)).sum() + (wn * F.logsigmoid(-fwd(Xn))).sum()) / (wp * len(Xp) + float(wn.sum()))
            loss = loss + 1e-3 * ((w ** 2).sum() + (W1 ** 2).sum()); op.zero_grad(); loss.backward(); op.step()
    with torch.no_grad(): th_s = float(torch.quantile(torch.cat([fwd(feat(k, rg)) for k, rg in O]), 0.99))
    return dict(muS=muS, PS=PS, sdS=sdS, W1=W1.detach(), w=w.detach(), b=b.detach(), th_s=th_s), O, ho


def nm_sleep(model, cfg, S_, S_slot):
    """the end-of-sleep rebuild (see the module note)"""
    t0 = time.time(); _nm_install(model, cfg)
    mo = _nm_layer(model, cfg); D = mo.weight.shape[0]; cap = {'on': False, 'k': [], 'delta': None}
    def hk(m, i, o):
        if cap['on']: cap['k'].append(i[0].detach())
        if cap['delta'] is not None: return o + cap['delta'].to(o.dtype)
    from platypus.inference.closed_form import _ws_eager
    back = _ws_eager(model); h = mo.register_forward_hook(hk); was = _NM['force']; _NM['force'] = False
    try:
        if _NM['ready']:                                                       # STALE: last sleep's memory after this visit's training = what the
            for md9 in ('hidden', 'logit'):                                    #   model's own store-on evaluations used (drift test, 10 Nov)
                try: nm_readout(model, cfg, S_, S_slot, f"visit {int(model.__dict__.get('_visit', 0)) + 1} | STALE (built last sleep, before this rebuild) | value {md9}", mode=md9)
                except Exception as e9: print(f"  [NAME MEMORY READOUT] stale unavailable ({type(e9).__name__}: {e9})"[:200], flush=True)
            _NM['force'] = False
        old = {e['key']: e for e in model.__dict__.get('_nm_ents', [])}
        ents = []
        for e in _nm_material(model, cfg, S_):
            o = old.get(e['key'])
            if o is not None: o['seen'] = e.get('seen')
            ents.append(o if o is not None else e)
        mne = int(getattr(cfg, 'inf_nm_min_entries', 32))                     # a gate trained on a handful of facts fires everywhere (local, 10 Nov:
        if len(ents) < mne: _NM['ready'] = False; print(f"  [NAME MEMORY] {len(ents)} facts (< {mne}): not built yet, memory off", flush=True); return   #   11 entries, 54 fires / 1000 bytes)
        _nm_keys_and_check(model, ents, S_slot, cap)
        nfit = [i for i, e in enumerate(ents) if e.get('V') is None or not e.get('ok')]; nnew = sum(1 for e in ents if e.get('V') is None)
        if nfit: _nm_fit_values(model, cfg, ents, nfit, S_slot, cap, D)
        ents = [e for e in ents if e.get('nk') is not None and e.get('V') is not None]
        model.__dict__['_nm_ents'] = ents
        G, O, ho = _nm_gate(model, cfg, ents, S_, S_slot, cap, int(cfg.seed) + 5151 + int(model.__dict__.get('_visit', 0)))
        NM = torch.stack([e['nk'] for e in ents]); QM = torch.stack([e['qk'] for e in ents]); mN = NM.mean(0); mQ = QM.mean(0)
        NMc = F.normalize(NM - mN, dim=-1); QMc = F.normalize(QM - mQ, dim=-1)
        bn = []
        for k, rg in O:
            ok = torch.tensor(rg >= 0)
            if ok.any(): bn.append((F.normalize(k[torch.tensor(np.maximum(rg, 0))] - mN, dim=-1) @ NMc.T).max(1).values[ok])
        th_n = float(torch.quantile(torch.cat(bn), 0.99)) if bn else 1.0
        pn = (F.normalize(NM - mN, dim=-1) @ NMc.T).diagonal()                  # each entry's own name against the memory (what a true mention scores)
        Lmax = max(len(e['V']) for e in ents); V = torch.zeros(len(ents), Lmax, D)
        for i, e in enumerate(ents): V[i, :len(e['V'])] = e['V'].float().cpu()
        dev = next(model.parameters()).device
        ANS = torch.full((len(ents), Lmax), 46, dtype=torch.long)               # the answer bytes (and the full stop), for the output-level value
        for i, e in enumerate(ents):
            ab9 = list((e['ans'] + '.').encode('latin-1', 'replace'))[:Lmax]; ANS[i, :len(ab9)] = torch.tensor(ab9)
        M = dict(mN=mN, mQ=mQ, NMc=NMc, QMc=QMc, V=V, L=torch.tensor([len(e['V']) for e in ents]), th_n=th_n, ANS=ANS,
                 margin=float(getattr(cfg, 'inf_nm_margin', 1.0)), **G)
        _NM['M'] = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in M.items()}; _NM['ready'] = True; _NM['ho'] = ho
        nf = sum(1 for e in ents if e['key'][0] == 'fact')
        sn9 = [e['seen'] for e in ents if e.get('seen')]; cd9 = model.__dict__.get('_nm_cand', (0, 0))
        try:                                                                    # READOUT ONLY: how many entries are about planted probe names
            from platypus.inference.probes import inf_probes
            pn9 = {n for c in inf_probes(cfg) for n in c.get('names', [])}
            pl9 = f"; planted probe subject and answer: facts {np.mean([e['A'] in pn9 and e['ans'] in pn9 for e in ents if e['key'][0] == 'fact'] or [np.nan]):.2f}, conclusions {np.mean([e['A'] in pn9 and e['ans'] in pn9 for e in ents if e['key'][0] == 'conclusion'] or [np.nan]):.2f} (readout)"
        except Exception: pl9 = ""
        cnt9 = (f"facts chosen by sightings (at least {int(getattr(cfg, 'inf_nm_min_seen', 2))}; {cd9[0]} candidates, {cd9[1]} seen 3+ times): sightings median "
                f"{np.median(sn9) if sn9 else float('nan'):.0f}, min {min(sn9) if sn9 else 0}{pl9}; ")
        print(f"  [NAME MEMORY | visit {int(model.__dict__.get('_visit', 0)) + 1}] {len(ents)} entries ({nf} recorded facts, {len(ents) - nf} conclusions); {cnt9}"
              f"keys re-read from the current weights; values fitted {len(nfit)} ({nnew} new, {len(nfit) - nnew} refitted because they no longer produced "
              f"their answer), kept {len(ents) - len(nfit)}; gate thresholds: slot {G['th_s']:.3f}, stored name {th_n:.3f} (entries' own names score median {float(pn.median()):.3f}, {float((pn > th_n).float().mean()):.2f} past it) | {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        _fail('inference.namemem:sleep'); _NM['ready'] = False; print(f"  [NAME MEMORY] build failed, memory off ({type(e).__name__}: {e})"[:240], flush=True)
    finally:
        _NM['force'] = was; h.remove(); back(); cap['on'] = False; cap['delta'] = None
    _NM['mode'] = str(getattr(cfg, 'inf_nm_value', 'hidden'))
    try:
        for md9 in ('hidden', 'logit'): nm_readout(model, cfg, S_, S_slot, f"visit {int(model.__dict__.get('_visit', 0)) + 1} | FRESH (just rebuilt) | value {md9}", mode=md9)
    except Exception as e:
        _fail('inference.namemem:readout'); print(f"  [NAME MEMORY READOUT] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


# ---------------------------------------------------------------- readout
def nm_readout(model, cfg, S_, S_slot, tag, n=48, mode=None):
    """held-out recall with the memory off / on, through the model's own forward (store off otherwise), and misfires"""
    if not _NM['ready']: return
    md0 = _NM['mode']
    if mode is not None: _NM['mode'] = mode
    try: _nm_readout(model, cfg, S_, S_slot, tag, n)
    finally: _NM['mode'] = md0


def _nm_readout(model, cfg, S_, S_slot, tag, n):
    from platypus.inference.closed_form import _ws_batch, _ws_run
    ents = model.__dict__.get('_nm_ents', []); rs = np.random.RandomState(int(cfg.seed) + 6161)
    isl = lambda b: 65 <= b <= 90 or 97 <= b <= 122
    def tf(pb, tb, sl, on):
        X, _ = _ws_batch(model, [pb + tb]); _NM['force'] = on; _NM['stats'] = [0, 0]
        try:
            with torch.no_grad(): lg = _ws_run(model, X, sl)[0]
        finally: _NM['force'] = None
        p0 = len(pb) - 1; ps = torch.arange(p0, p0 + len(tb), device=lg.device)
        if int(ps[-1]) >= lg.shape[0]: return None
        top = lg[ps].argmax(-1); tr = torch.tensor(list(tb), device=lg.device)
        return bool((top[:-1] == tr[:-1]).all()) and not isl(int(top[-1]))
    def bits(txt, sl, on):
        X, _ = _ws_batch(model, [txt]); _NM['force'] = on; _NM['stats'] = [0, 0]
        try:
            with torch.no_grad(): lg = _ws_run(model, X, sl)[0]
        finally: _NM['force'] = None
        T = min(len(txt) - 1, lg.shape[0]); return float(F.cross_entropy(lg[:T].float(), X[0, 1:T + 1], reduction='sum')) / np.log(2), T, _NM['stats'][0]
    C = [e for e in ents if e['key'][0] == 'conclusion']; Fx = [e for e in ents if e['key'][0] == 'fact']
    C = [C[i] for i in rs.permutation(len(C))[:n]]; Fx = [Fx[i] for i in rs.permutation(len(Fx))[:n]]
    own = [ents[i] for i in rs.permutation(len(ents))[:n]]
    fam = {'OWN WRITTEN WORDING (is the value right?)': [(e['pfs'][0], e) for e in own],
           'conclusions, standard held-out wordings': [(w.format(A=e['A']), e) for e in C for w in _HO_TWO],
           'facts, new frame': [(f"It is on record that {e['A']} {e['r']} ", e) for e in Fx],
           'facts, paraphrased relation': [(f"{e['A']} {_NM_PARA[e['r']]} ", e) for e in Fx if e['r'] in _NM_PARA]}
    parts = []
    for nm, L in fam.items():
        r0, r1 = [], []
        for pf, e in L:
            pb = pf.encode('latin-1', 'replace'); tb = (e['ans'] + '.').encode('latin-1', 'replace'); sl = int(S_slot(e['d']))
            a = tf(pb, tb, sl, False); b = tf(pb, tb, sl, True)
            if a is not None and b is not None: r0.append(a); r1.append(b)
        if r0: parts.append(f"{nm} ({len(r0)}): open {np.mean(r0):.2f} -> {np.mean(r1):.2f}")
    subs = sorted(set(e['A'] for e in ents)); tmpl = ("{A} arrived late to the meeting.", "Yesterday {A} said nothing at all.", "The letter from {A} was short and polite.")
    ms = [(t.format(A=A).encode('latin-1', 'replace'), int(S_slot(e['d']))) for A, e in [(A, next(x for x in ents if x['A'] == A)) for A in [subs[i] for i in rs.permutation(len(subs))[:24]]] for t in tmpl]
    it = S_.get('it', []); sb = set(s.encode('latin-1', 'replace') for s in subs); ans = set(e['ans'].encode('latin-1', 'replace') for e in ents)
    for x in it:
        if len(ms) >= 72 + 100: break
        if x.get('k') != 'real' or not (30 <= len(x['s']) <= 200): continue
        s = bytes(x['s']); ws = set(m.group(0) for m in _NM_WORD.finditer(s))
        if ws & sb and not ws & ans: ms.append((s, int(S_slot(x['d']))))
    def cost(L):
        b0 = b1 = nb = nf = 0.0
        for txt, sl in L:
            x0, T, _ = bits(txt, sl, False); x1, _, f1 = bits(txt, sl, True); b0 += x0; b1 += x1; nb += T; nf += f1
        return f"fires {1000 * nf / max(1, nb):.1f} per 1000 bytes, BPC {(b1 - b0) / max(1, nb):+.4f} ({len(L)} texts)"
    ho = [(bytes(x['s']), int(S_slot(x['d']))) for x in (_NM.get('ho') or [])[:100]]
    print(f"  [NAME MEMORY READOUT | {tag}] held-out questions, memory off -> on (the model's forward, store otherwise off): " + " | ".join(parts)
          + f" || MISFIRES: text naming stored subjects without asking about them: {cost(ms)} | ordinary text: {cost(ho)}", flush=True)
