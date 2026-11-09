"""SLOW HEAD (north-star rung 1; slow_k > 0, off by default). A small linear map z = W (h - m) on the trunk's features, learned ONLINE
during visits from the real text of each training step by three local terms - nothing stored, no pairs, no negatives:
  prediction      mean over units of (z(t + delta) - z(t))^2 with each unit standardised over the batch = 2 (1 - its temporal correlation):
                  scale-free, as in SFA's unit-variance constraint (a first version on raw z shrank the code instead of slowing it)
  variance        relu(1 - std) + relu(std - 2) of every unit (spread held near 1)
  decorrelation   squared off-diagonal correlation / k (units carry different signals)
i.e. VICReg with time, which is linear slow feature analysis written as a loss. STOP-GRADIENT: h is detached, so the trunk, its
training and every BPC figure are unchanged; W has its own optimiser and its own random generator. m is a running mean of h per
(slot, byte class) - letters, digits, spaces, punctuation - so the trivially slow directions (which slot, which kind of byte) are
removed before W sees them. v3 (slow_cf_init 1): pooled segment pairs from the first visit are kept in a small reservoir and, when the
first visit ends, W is SET to the closed-form SFA solution on them (whiten, then the slowest directions of the change covariance); the
online rule then only tracks the trunk's drift from there. v4 (slow_sent N > 0): SENTENCE-scale slowness - instead of byte segments of the
training rows, every slow_sent_every steps N adjacent pairs of real stored sentences (same domain, consecutive in the store = consecutive
in the stream) are read (trunk in eval mode, store off, no grad), their centred features mean-pooled exactly as the readout pools them,
and the same three terms are applied across the pair (v3 reached closed-form slowness at 16 bytes but not closed-form retrieval, which is
fitted on consecutive sentences) (offline on a trained trunk the online loss holds a closed-form start at 0.84,
while random starts need ~8000 steps at lr 3e-2 to reach it - a run gives ~380 per visit). The codes are read out by the selector comparison (inf_selcmp: SLOW (online)) against closed-form SFA."""
import math
import torch
import torch.nn.functional as F
from platypus.common import _fail


def _byte_class(x):
    """0 letter, 1 digit, 2 space, 3 other"""
    c = torch.full_like(x, 3)
    c[((x >= 65) & (x <= 90)) | ((x >= 97) & (x <= 122))] = 0
    c[(x >= 48) & (x <= 57)] = 1
    c[(x == 32) | (x == 10) | (x == 9)] = 2
    return c


def slow_init(model, cfg):
    k = int(getattr(cfg, 'slow_k', 0) or 0)
    if k <= 0: model.__dict__['_slow'] = None; return
    d = int(cfg.d_model); dev = model.embed.weight.device; g = torch.Generator().manual_seed(int(cfg.seed) + 3131)
    W = torch.nn.Parameter((torch.randn(d, k, generator=g) / math.sqrt(d)).to(dev))
    model.__dict__['_slow'] = {'W': W, 'opt': torch.optim.Adam([W], lr=float(getattr(cfg, 'slow_lr', 1e-2))), 'mu': {}, 'n': 0, 'k': k,
                               'delta': int(getattr(cfg, 'slow_delta', 8)), 'every': max(1, int(getattr(cfg, 'slow_every', 1))), 'log': [], 'vis': -1,
                               'seg': int(getattr(cfg, 'slow_seg', 16)), 'iters': max(1, int(getattr(cfg, 'slow_iters', 4))),
                               'sent': int(getattr(cfg, 'slow_sent', 0) or 0), 'sent_every': max(1, int(getattr(cfg, 'slow_sent_every', 2))), 'pix': None, 'pn': -1,
                               'cf': int(getattr(cfg, 'slow_cf_init', 0) or 0), 'res': [], 'nres': 0, 'cfdone': False, 'vis0': None}
    st0 = model.__dict__['_slow']
    print(f"  [SLOW HEAD] on (rung 1): k {k}, " + (f"SEGMENT slowness: pooled {st0['seg']}-byte segments predicting the next" if st0['seg'] > 0 else f"delta {st0['delta']} bytes") + f", {st0['iters']} iteration(s) per step, lr {float(getattr(cfg, 'slow_lr', 1e-2)):g}, "
          f"weights prediction {float(getattr(cfg, 'slow_wp', 1.0)):g} / variance {float(getattr(cfg, 'slow_wv', 1.0)):g} / decorrelation "
          f"{float(getattr(cfg, 'slow_wc', 1.0)):g}; stop-gradient (the trunk is unchanged)" + ("; CLOSED-FORM START at the end of the first visit (v3)" if st0['cf'] else "")
          + (f"; SENTENCE PAIRS (v4): {st0['sent']} adjacent stored sentence pairs every {st0['sent_every']} step(s), replacing byte segments" if st0['sent'] else ""), flush=True)


def _cf_set(st):
    """v3: set W to closed-form SFA on the reservoir of consecutive pooled segments (rows of A predict rows of B)"""
    A = torch.cat([a for a, b in st['res']]); B = torch.cat([b for a, b in st['res']]); Z = torch.cat([A, B]); mu = Z.mean(0)
    U, S, V = torch.linalg.svd(Z - mu, full_matrices=False); kk = min(64, V.shape[0]); Wh = V[:kk].T / (S[:kk] / max(1.0, (len(Z) - 1) ** 0.5) + 1e-3)
    D = (A - B) @ Wh; ev, evec = torch.linalg.eigh(D.T @ D / len(D)); k = st['k']
    with torch.no_grad(): st['W'].data.copy_((Wh @ evec[:, :k]).to(st['W'].device, st['W'].dtype))
    st['opt'].state.clear(); st['cfdone'] = True; st['res'] = []
    print(f"  [SLOW HEAD | CLOSED-FORM START] W set from {len(A)} consecutive pooled segment pairs: slowness (half the variance of the whitened change, "
          f"1 = unrelated) of the {k} directions {float(ev[:k].mean() / 2):.3f} (slowest {float(ev[0] / 2):.3f}); the online rule continues from here", flush=True)


def _centre(st, x, h, dom, update):
    """subtract the running mean per (slot, byte class); update it from this batch when asked"""
    cls = _byte_class(x[:, :h.shape[1]]); out = h.clone()
    for c in range(4):
        m_ = cls == c
        if not m_.any(): continue
        key = (dom, c)
        if update:
            bm = h[m_].mean(0); st['mu'][key] = bm if key not in st['mu'] else 0.99 * st['mu'][key] + 0.01 * bm
        mu = st['mu'].get(key)
        if mu is None:                                                            # unseen slot (readout): the mean of the class over slots
            ms = [v for (d_, c_), v in st['mu'].items() if c_ == c]
            mu = torch.stack(ms).mean(0) if ms else None
        if mu is not None: out[m_] = h[m_] - mu
    return out


def slow_update(model, cfg, x, h, dom):
    """one online step on the real rows of a training step (h detached)"""
    st = model.__dict__.get('_slow')
    if st is None: return
    try:
        st['n'] += 1
        if st['n'] % st['every']: return
        hc = _centre(st, x, h.detach().float(), int(dom), True); P = st['seg']; D = st['delta']   # (also keeps the per-class means current in v4)
        if st['sent'] > 0:
            if st['n'] % st['sent_every'] == 0: _sent_update(model, cfg, st)
            return
        if P > 0:                                                                  # v2 SEGMENT slowness: mean code of P bytes predicts the next segment's (closed form on
            n_ = hc.shape[1] // P                                                  #   pooled segments reaches slowness 0.84 over 32 directions locally, against 0.93
            if n_ < 2: return                                                      #   for single bytes 8 apart; v1 stalled at 0.986)
            hc = hc[:, :n_ * P].reshape(hc.shape[0], n_, P, -1).mean(2); D = 1
            if st['cf'] and not st['cfdone']:                                      # v3: reservoir during the first visit, closed-form start when it ends
                v_ = int(model.__dict__.get('_visit', 0))
                if st['vis0'] is None: st['vis0'] = v_
                if v_ != st['vis0'] and st['nres'] >= 1000: _cf_set(st)
                elif st['nres'] < 6000:
                    a_ = hc[:, :-1].reshape(-1, hc.shape[-1]).detach().cpu(); b_ = hc[:, 1:].reshape(-1, hc.shape[-1]).detach().cpu()
                    st['res'].append((a_, b_)); st['nres'] += len(a_)
        if hc.shape[1] <= D: return
        for _ in range(st['iters']):
          with torch.enable_grad():
            z = hc @ st['W']
            zf = z.reshape(-1, z.shape[-1]); mu = zf.mean(0); std = torch.sqrt(zf.var(0) + 1e-4)
            zn = (z - mu) / std                                                   # SCALE-FREE slowness (as SFA: unit variance): 2 (1 - temporal correlation)
            pred = ((zn[:, D:] - zn[:, :-D]) ** 2).mean()                          #   per unit, so shrinking the code cannot lower it
            vl = F.relu(1 - std).mean() + F.relu(std - 2).mean()                  # spread held near 1
            zc = (zf - mu) / std; cov = zc.T @ zc / max(1, len(zf) - 1); off = cov - torch.diag(torch.diag(cov)); cl = (off ** 2).sum() / st['k']   # correlations
            loss = float(getattr(cfg, 'slow_wp', 1.0)) * pred + float(getattr(cfg, 'slow_wv', 1.0)) * vl + float(getattr(cfg, 'slow_wc', 1.0)) * cl
            st['opt'].zero_grad(); loss.backward(); st['opt'].step()
        with torch.no_grad():
            ratio = float(pred / 2)                                               # 1 = no slower than unrelated positions
            cr = cov
            st['log'].append((float(pred), float(std.mean()), float((cr - torch.diag(torch.diag(cr))).abs().sum() / (st['k'] * (st['k'] - 1))), ratio))
        vis = int(model.__dict__.get('_visit', 0))
        if vis != st['vis'] and len(st['log']) >= 20:
            L = st['log'][-200:]; m = lambda j: sum(r[j] for r in L) / len(L)
            print(f"  [SLOW HEAD | visit {vis}] {st['n']} updates | slowness (change " + (f"to the next {P}-byte segment" if P > 0 else f"over {D} bytes") + f" / unrelated) {m(3):.3f} | unit spread {m(1):.2f} | "
                  f"mean |correlation| between units {m(2):.3f} | prediction {m(0):.3f}", flush=True)
            st['vis'] = vis
    except Exception:
        _fail('inference.slowhead:update')


def _sent_update(model, cfg, st):
    """v4: one update on adjacent pairs of real stored sentences"""
    from platypus.inference.sentences import _inf_store
    it = _inf_store(model).get('it', [])
    if len(it) < 200: return
    if st['pix'] is None or len(it) - st['pn'] > 2000:                          # adjacent real pairs (refreshed as the store grows)
        st['pix'] = [i for i in range(len(it) - 1) if it[i].get('k') == 'real' and it[i + 1].get('k') == 'real' and it[i]['d'] == it[i + 1]['d']
                     and 20 <= len(it[i]['s']) <= 300 and 20 <= len(it[i + 1]['s']) <= 300]; st['pn'] = len(it)
    if len(st['pix']) < 50: return
    g = st.setdefault('g', torch.Generator().manual_seed(int(cfg.seed) + 3232)); ix = torch.randint(0, len(st['pix']), (st['sent'],), generator=g).tolist()
    Pa = _pooled(model, st, [it[st['pix'][j]]['s'] for j in ix]); Pb = _pooled(model, st, [it[st['pix'][j] + 1]['s'] for j in ix]); k = st['k']
    for _ in range(st['iters']):
        with torch.enable_grad():
            za = Pa @ st['W']; zb = Pb @ st['W']; zf = torch.cat([za, zb]); mu = zf.mean(0); sd = torch.sqrt(zf.var(0) + 1e-4)
            pred = (((zb - mu) / sd - (za - mu) / sd) ** 2).mean(); vl = F.relu(1 - sd).mean() + F.relu(sd - 2).mean()
            zc = (zf - mu) / sd; cov = zc.T @ zc / max(1, len(zf) - 1); off = cov - torch.diag(torch.diag(cov)); cl = (off ** 2).sum() / k
            loss = float(getattr(cfg, 'slow_wp', 1.0)) * pred + float(getattr(cfg, 'slow_wv', 1.0)) * vl + float(getattr(cfg, 'slow_wc', 1.0)) * cl
            st['opt'].zero_grad(); loss.backward(); st['opt'].step()
    with torch.no_grad(): st['log'].append((float(pred), float(sd.mean()), float(off.abs().sum() / (k * (k - 1))), float(pred / 2)))
    vis = int(model.__dict__.get('_visit', 0))
    if vis != st['vis'] and len(st['log']) >= 10:
        L = st['log'][-100:]; m = lambda j: sum(r[j] for r in L) / len(L)
        print(f"  [SLOW HEAD | visit {vis}] {len(st['log'])} sentence-pair updates ({len(st['pix'])} adjacent pairs in the store) | slowness (change to the next sentence / unrelated) {m(3):.3f} | "
              f"unit spread {m(1):.2f} | mean |correlation| between units {m(2):.3f}", flush=True)
        st['vis'] = vis


def _pooled(model, st, txts):
    """centred trunk features mean-pooled over each passage (store off, no carried context, eval mode): the readout's pooling before W"""
    from platypus.inference.jepa import cfg_seq_
    from platypus.conduit import _carry_save, _carry_load
    from platypus.layers import _CARRY
    L0 = int(cfg_seq_(model)); dev = model.embed.weight.device; out = []; was = model.training; model.eval()
    try:
        for a in range(0, len(txts), 64):
            bs = [bytes(t)[:L0] for t in txts[a:a + 64]]; L = max(8, max(len(b) for b in bs))
            X = torch.full((len(bs), L), 32, dtype=torch.long, device=dev); M = torch.zeros(len(bs), L, device=dev)
            for j, b in enumerate(bs):
                if b: X[j, :len(b)] = torch.tensor(list(b), device=dev); M[j, :len(b)] = 1.0
            sv = _carry_save(); _CARRY.update(on=False, st={}, new={})
            try:
                with torch.no_grad(): h = model.features(X).float()
            finally: _carry_load(sv)
            hc = _centre(st, X, h, -1, False); out.append((hc * M[..., None]).sum(1) / M.sum(1, keepdim=True).clamp(min=1))
    finally: model.train(was)
    return torch.cat(out)


def slow_codes(model, txts):
    """the slow code of each passage: z averaged over its bytes (store off, no carried context), read with the trunk in eval mode"""
    st = model.__dict__.get('_slow')
    if st is None: return None
    from platypus.inference.jepa import cfg_seq_
    from platypus.conduit import _carry_save, _carry_load
    from platypus.layers import _CARRY
    L0 = int(cfg_seq_(model)); dev = model.embed.weight.device; out = []; was = model.training; model.eval()
    try:
        for a in range(0, len(txts), 64):
            bs = [bytes(t)[:L0] for t in txts[a:a + 64]]; L = max(8, max(len(b) for b in bs))
            X = torch.full((len(bs), L), 32, dtype=torch.long, device=dev); M = torch.zeros(len(bs), L, device=dev)
            for j, b in enumerate(bs):
                if b: X[j, :len(b)] = torch.tensor(list(b), device=dev); M[j, :len(b)] = 1.0
            sv = _carry_save(); _CARRY.update(on=False, st={}, new={})
            try:
                with torch.no_grad(): h = model.features(X).float()
            finally: _carry_load(sv)
            z = _centre(st, X, h, -1, False) @ st['W'].detach()
            out.append((z * M[..., None]).sum(1) / M.sum(1, keepdim=True).clamp(min=1))
    finally: model.train(was)
    return torch.cat(out)
