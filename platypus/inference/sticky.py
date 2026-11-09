"""STICKY ROUTING ON MIXED STREAMS (sticky_router_eval 1; readout at pass ends, the model and its headline are unchanged).

Where text arrives as a stream, the router's posterior over heads can carry from one window to the next ("sticky"), with a reset gate
checked on the first 64 bytes of each window. This is the routing to use for streamed text; it is NOT applied to the label-free headline,
whose windows are grouped by domain (carrying the posterior there would leak the domain).

Streams: NW windows (256 bytes) of domain a, then NW of domain b, for ordered pairs of seen domains (held-out text, second half of each
domain's validation bytes). Each byte is routed by the posterior over heads given the window's bytes so far and a PRIOR:
  none (the current router) | STICKY (the previous window's final posterior, mixed with 1% uniform) | sticky + a RESET GATE.
Gates, each with its threshold set at 2% false resets on windows that did not switch:
  router loss (bits of the first 64 bytes under the carried head minus the best head) | slowest SFA band (cosine distance of the first
  64 bytes' code to the carried domain's prototype) | both, z-scored and summed |
  and THE RULE CASCADE'S OWN SIGNALS on the same windows (train._cdet): representation jump (1 - cosine of the window's mean trunk code to
  the running mean of the stream so far) and loss jump (bits of the first 64 bytes under the carried head minus their running mean).
Reported: token share routed right and mixture BPC, inside a domain / the first two windows after a switch / all; how often each gate fires
at the switch and elsewhere.
"""
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import _fail


def sticky_route(ll, prior0=None, gate=None, thr=None, sticky=True, eps=0.01, warm=64):
    """the router: ll (D, S, K, T) log-likelihood of each byte under each head, for S streams of K windows; returns the log posterior used
    for each byte (D, S, K, T) and which windows reset. gate(k, carried) -> score per stream (fires when > thr)"""
    D, S, K, T = ll.shape; LN = float(np.log(D))
    cum = torch.cat([torch.zeros_like(ll[..., :1]), torch.cumsum(ll, -1)[..., :-1]], -1)
    prior = torch.full((S, D), -LN) if prior0 is None else prior0; out = torch.zeros_like(ll); fired = torch.zeros(S, K)
    for k in range(K):
        p0 = prior if (sticky and k > 0) else torch.full_like(prior, -LN)
        post = F.log_softmax(p0.T[..., None] + cum[:, :, k], 0)
        if gate is not None and k > 0:
            fire = gate(k, prior.argmax(1)) > thr; fired[:, k] = fire.float()
            pu = F.log_softmax(cum[:, :, k] - cum[:, :, k, warm:warm + 1], 0)               # uniform prior from byte `warm` on
            post = torch.where(fire[None, :, None] & (torch.arange(T)[None, None, :] >= warm), pu, post)
        out[:, :, k] = post
        end = F.log_softmax(post[:, :, -1] + ll[:, :, k, -1], 0).T
        prior = torch.log((1 - eps) * end.exp() + eps / D)
    return out, fired


def sticky_router_readout(model, cfg, raw_vals_fn, seen_d, slot_of, tag):
    try:
        from platypus.inference.selcmp import _codes, _whiten
        from platypus.inference.residual import _bits_rows
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            st0 = np.random.get_state()
            try: _readout(model, cfg, raw_vals_fn, seen_d, slot_of, tag, _codes, _whiten, _bits_rows)
            finally: np.random.set_state(st0)
    except Exception as e:
        _fail('inference.sticky'); print(f"  [STICKY ROUTER | {tag}] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _readout(model, cfg, raw_vals_fn, seen_d, slot_of, tag, _codes, _whiten, _bits_rows):
    doms = sorted(set(int(d) for d in seen_d)); D = len(doms)
    if D < 2: return
    rs = np.random.RandomState(int(cfg.seed) + 4141); rv = raw_vals_fn(); T_ = 256; NW = 6; LN2 = float(np.log(2)); W64 = 64
    per = max(1, int(round(180 / (D * (D - 1))))); per = min(per, 3)                           # about 180 streams at most
    pa, pb = [], []                                                                            # SLOWEST BAND: SFA on adjacent 200-byte halves
    for d in doms:
        h = len(rv[d]) // 2
        for i in rs.randint(0, h - 401, 300): pa.append(bytes(rv[d][i:i + 200])); pb.append(bytes(rv[d][i + 200:i + 400]))
    ZA, ZB = _codes(model, pa).cpu(), _codes(model, pb).cpu(); mu, Wh = _whiten(torch.cat([ZA, ZB])); A = (ZA - mu) @ Wh; B = (ZB - mu) @ Wh
    ev, evec = torch.linalg.eigh((A - B).T @ (A - B) / len(A)); P4 = (Wh @ evec)[:, :4]
    proto = torch.stack([((_codes(model, [bytes(rv[d][i:i + 200]) for i in rs.randint(0, len(rv[d]) // 2 - 201, 40)]).cpu() - mu) @ P4).mean(0) for d in doms])
    streams = []
    for a in range(D):
        for b in range(D):
            if a == b: continue
            for _ in range(per):
                ra, rb = rv[doms[a]], rv[doms[b]]
                sa = rs.randint(len(ra) // 2, len(ra) - NW * T_ - 1); sb = rs.randint(len(rb) // 2, len(rb) - NW * T_ - 1)
                streams.append(([bytes(ra[sa + k * T_:sa + (k + 1) * T_]) for k in range(NW)] + [bytes(rb[sb + k * T_:sb + (k + 1) * T_]) for k in range(NW)], [a] * NW + [b] * NW))
    S = len(streams); K = 2 * NW; W = sum([s[0] for s in streams], []); lab = torch.tensor(sum([s[1] for s in streams], [])).reshape(S, K)
    bits = torch.stack([torch.tensor(np.stack(_bits_rows(model, W, int(slot_of(e)), False))) for e in doms]).float().reshape(D, S, K, -1)
    ll = -bits * LN2; T = ll.shape[-1]
    c64 = _codes(model, [w[:W64] for w in W]).cpu(); z64 = F.normalize(((c64 - mu) @ P4), dim=-1).reshape(S, K, 4); pn = F.normalize(proto, dim=-1)
    cfull = F.normalize(c64, dim=-1).reshape(S, K, -1)
    b64 = bits[..., :W64].mean(-1)                                                             # (D, S, K)
    ar = torch.arange(S)
    def g_loss(k, c): return b64[c, ar, k] - b64[:, :, k].min(0).values
    def g_band(k, c): return 1 - (z64[:, k] * pn[c]).sum(-1)
    def g_rep(k, c):                                                                           # CASCADE: representation jump vs the stream's running mean
        m = cfull[:, 0].clone()
        for j in range(1, k): m = 0.9 * m + 0.1 * cfull[:, j]
        return 1 - F.cosine_similarity(cfull[:, k], m, dim=-1)
    def g_ljump(k, c):                                                                         # CASCADE: loss jump under the carried head vs its running mean
        m = b64[c, ar, 0].clone()
        for j in range(1, k): m = 0.9 * m + 0.1 * b64[c, ar, j]
        return b64[c, ar, k] - m
    keep = list(range(1, NW)) + list(range(NW + 2, K))
    def cal(g): return float(torch.quantile(torch.cat([g(k, lab[:, k - 1]) for k in keep]), 0.98))
    def zs(g):
        v = torch.cat([g(k, lab[:, k - 1]) for k in keep]); m, s = v.mean(), v.std() + 1e-6; return lambda k, c: (g(k, c) - m) / s
    zl, zb = zs(g_loss), zs(g_band); g_both = lambda k, c: zl(k, c) + zb(k, c)
    inside = torch.zeros(K, dtype=torch.bool); inside[1:NW] = True; inside[NW + 2:] = True; after = torch.zeros(K, dtype=torch.bool); after[NW:NW + 2] = True
    rows = [('none (current router)', None, False), ('STICKY, no gate', None, True), ('STICKY + gate: router loss', g_loss, True),
            ('STICKY + gate: slowest band', g_band, True), ('STICKY + gate: router loss + slowest band', g_both, True),
            ('STICKY + CASCADE gate: representation jump', g_rep, True), ('STICKY + CASCADE gate: loss jump', g_ljump, True)]
    print(f"  [STICKY ROUTER | {tag}] {S} held-out streams of {NW} + {NW} windows ({T_} bytes) over {D} seen domains ({per} per ordered pair); "
          f"reset gates on the first {W64} bytes at 2% false resets; readout only (not the headline: its windows are grouped by domain)", flush=True)
    for name, g, stk in rows:
        post, fired = sticky_route(ll, gate=g, thr=(cal(g) if g is not None else None), sticky=stk, warm=W64)
        ok = (post.argmax(0) == lab[:, :, None]).float(); mix = -torch.logsumexp(post + ll, 0) / LN2
        print(f"    {name:44s} tokens right: inside {ok[:, inside].mean():.3f}, first 2 after a switch {ok[:, after].mean():.3f}, all {ok.mean():.3f} | "
              f"mixture BPC inside {mix[:, inside].mean():.4f}, after {mix[:, after].mean():.4f}, all {mix.mean():.4f}"
              + (f" | fires at the switch {fired[:, NW].mean():.2f}, elsewhere {fired[:, inside].mean():.3f}" if g is not None else ""), flush=True)
