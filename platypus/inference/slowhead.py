"""SLOW HEAD (north-star rung 1; slow_k > 0, off by default). A small linear map z = W (h - m) on the trunk's features, learned ONLINE
during visits from the real text of each training step by three local terms - nothing stored, no pairs, no negatives:
  prediction      mean over units of (z(t + delta) - z(t))^2 with each unit standardised over the batch = 2 (1 - its temporal correlation):
                  scale-free, as in SFA's unit-variance constraint (a first version on raw z shrank the code instead of slowing it)
  variance        relu(1 - std) + relu(std - 2) of every unit (spread held near 1)
  decorrelation   squared off-diagonal correlation / k (units carry different signals)
i.e. VICReg with time, which is linear slow feature analysis written as a loss. STOP-GRADIENT: h is detached, so the trunk, its
training and every BPC figure are unchanged; W has its own optimiser and its own random generator. m is a running mean of h per
(slot, byte class) - letters, digits, spaces, punctuation - so the trivially slow directions (which slot, which kind of byte) are
removed before W sees them. The codes are read out by the selector comparison (inf_selcmp: SLOW (online)) against closed-form SFA."""
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
                               'seg': int(getattr(cfg, 'slow_seg', 16)), 'iters': max(1, int(getattr(cfg, 'slow_iters', 4)))}
    st0 = model.__dict__['_slow']
    print(f"  [SLOW HEAD] on (rung 1): k {k}, " + (f"SEGMENT slowness: pooled {st0['seg']}-byte segments predicting the next" if st0['seg'] > 0 else f"delta {st0['delta']} bytes") + f", {st0['iters']} iteration(s) per step, lr {float(getattr(cfg, 'slow_lr', 1e-2)):g}, "
          f"weights prediction {float(getattr(cfg, 'slow_wp', 1.0)):g} / variance {float(getattr(cfg, 'slow_wv', 1.0)):g} / decorrelation "
          f"{float(getattr(cfg, 'slow_wc', 1.0)):g}; stop-gradient (the trunk is unchanged)", flush=True)


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
        hc = _centre(st, x, h.detach().float(), int(dom), True); P = st['seg']; D = st['delta']
        if P > 0:                                                                  # v2 SEGMENT slowness: mean code of P bytes predicts the next segment's (closed form on
            n_ = hc.shape[1] // P                                                  #   pooled segments reaches slowness 0.84 over 32 directions locally, against 0.93
            if n_ < 2: return                                                      #   for single bytes 8 apart; v1 stalled at 0.986)
            hc = hc[:, :n_ * P].reshape(hc.shape[0], n_, P, -1).mean(2); D = 1
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
