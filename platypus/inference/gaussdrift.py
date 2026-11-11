"""GAUSSIAN DRIFT READOUT (inf_gauss_drift, 11 Nov; readout only, training untouched). Does replay keep the hidden states' second-order
structure on old domains? At each sleep: a FIXED set of stored real sentences per slot (chosen when the slot is first seen, then kept), the
residual stream entering each block and the final features (store and name memory off, no carried context), a Gaussian (mean, covariance)
fitted per slot and layer over every byte position; reported against the previous sleep's Gaussian on the same sentences: the
2-Wasserstein (Bures) distance squared, split into the mean part |mu1 - mu0|^2 and the covariance part, each RELATIVE to tr(Sigma0)
(0 = unchanged; 1 = moved by the states' whole spread), and the covariance's effective rank. Old slots (not the one just visited) are
averaged; the visited slot is shown apart. Compare with the revisit BPC (forgetting) of the same run and across runs."""
import numpy as np
import torch
from platypus.common import _fail


def _bures(m0, S0, m1, S1):
    """(mean part, covariance part) of W2^2 between N(m0,S0) and N(m1,S1)"""
    e, V = torch.linalg.eigh(S0); r0 = (V * e.clamp(min=0).sqrt()) @ V.T
    M = r0 @ S1 @ r0; em = torch.linalg.eigvalsh((M + M.T) / 2).clamp(min=0).sqrt().sum()
    return float((m1 - m0).pow(2).sum()), float(torch.trace(S0) + torch.trace(S1) - 2 * em)


def _erank(S):
    e = torch.linalg.eigvalsh(S).clamp(min=0); p = e / e.sum().clamp(min=1e-30)
    return float(torch.exp(-(p * (p + 1e-30).log()).sum()))


def gauss_drift(model, cfg, S_, S_slot, vslot, oslots):
    if not int(getattr(cfg, 'inf_gauss_drift', 0)): return
    from platypus.inference.closed_form import _ws_batch, _ws_run
    from platypus.inference.namemem import _NM
    G = model.__dict__.setdefault('_gdr', {'sent': {}, 'prev': {}, 'hist': []}); vis = int(model.__dict__.get('_visit', 0))
    slots = sorted(set(int(s) for s in oslots) | ({int(vslot)} if vslot is not None else set()))
    it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
    for sl in slots:
        if sl in G['sent']: continue
        c = [bytes(x['s']) for x in real if int(S_slot(x['d'])) == sl]
        if len(c) >= 16:
            rs = np.random.RandomState(int(cfg.seed) + 4242 + sl); G['sent'][sl] = [c[int(i)] for i in rs.choice(len(c), min(48, len(c)), replace=False)]
    nL = len(model.blocks); cap = {'on': False, 'h': {}}
    def mk(l):
        def hk(m, a):
            if cap['on']: cap['h'][l] = a[0].detach().float()
        return hk
    hs = [model.blocks[l].register_forward_pre_hook(mk(l)) for l in range(nL)]
    def fk(m, a, o):
        if cap['on']: cap['h'][nL] = (o[0] if isinstance(o, tuple) else o).detach().float()
    was = _NM['force']; _NM['force'] = False; cur = {}
    try:
        try: hs.append(model.blocks[-1].register_forward_hook(fk))
        except Exception: pass
        for sl, txt in G['sent'].items():
            if sl not in slots: continue
            X, M = _ws_batch(model, txt); cap['on'] = True; cap['h'] = {}
            try:
                with torch.no_grad(): _ws_run(model, X, sl)
            finally: cap['on'] = False
            m = M[:, :X.shape[1] - 1].bool()
            st = {}
            for l, h in cap['h'].items():
                Z = h[:, :m.shape[1]][m[:, :h.shape[1]]].double()
                mu = Z.mean(0); C = Z - mu; st[l] = (mu, C.T @ C / max(1, Z.shape[0] - 1))
            cur[sl] = st
        rows = {}
        for sl, st in cur.items():
            if sl not in G['prev']: continue
            for l, (mu, S) in st.items():
                if l not in G['prev'][sl]: continue
                m0, S0 = G['prev'][sl][l]; a, b = _bures(m0, S0, mu, S); tr = float(torch.trace(S0)) or 1e-30
                rows.setdefault('old' if sl != vslot else 'visited', {}).setdefault(l, []).append((a / tr, b / tr, _erank(S)))
        def fmt(d):
            return " ".join(f"L{l}{'(out)' if l == nL else ''} {np.mean([x[0] + x[1] for x in v]):.4f} (mean {np.mean([x[0] for x in v]):.4f}, cov {np.mean([x[1] for x in v]):.4f}, erank {np.mean([x[2] for x in v]):.1f})"
                            for l, v in sorted(d.items()))
        if rows:
            G['hist'].append((vis + 1, {k: {l: float(np.mean([x[0] + x[1] for x in v])) for l, v in d.items()} for k, d in rows.items()}))
            print(f"  [GAUSSIAN DRIFT | visit {vis + 1}] W2^2 to the previous sleep / tr(Sigma prev), residual stream into each block, fixed stored sentences "
                  f"(store and memory off) | OLD SLOTS ({len([s for s in cur if s in G['prev'] and s != vslot])}): {fmt(rows.get('old', {})) or 'none'} || "
                  f"VISITED SLOT: {fmt(rows.get('visited', {})) or 'none'}", flush=True)
        G['prev'] = cur
    except Exception as e:
        _fail('inference.gaussdrift'); print(f"  [GAUSSIAN DRIFT] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
    finally:
        for h in hs: h.remove()
        _NM['force'] = was
