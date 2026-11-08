"""TIMESCALE BANK READOUTS on the training clock (bank_drift 1; readout only, the model is unchanged).
Item 2, FORGETTING WITHOUT A VALIDATION SET: each seen domain has a fixed set of reference windows (validation text, first half); its
PROTOTYPE is the mean code of those windows. The prototype is recorded at the end of the domain's first visit; after every later visit
its DRIFT is 1 - cosine(now, then), also relative to the mean drift of all domains (the bank's own drift). Beside it, the domain's bits
per byte on the same windows (slot given) against the same moment: the forgetting the drift should predict. At pass ends and at the
end, the rank correlation of drift with the BPC rise over all (domain, visit) points, for trunk codes and (when on) the slow head's codes.
Item 3, TRAINING-SLOWNESS: the mean cosine of each reference window's code with its own code when first recorded - how much a code moves
under learning (the property drift-free keys need), trunk against slow head.
ROUTING BY PROTOTYPE (at summaries): held-out windows (second half of each domain's validation text) go to the nearest domain
prototype by cosine - with prototypes as they are now, and with the STALE prototypes recorded at each domain's first visit (can a key
written once still route later, with no refresh). Trunk codes against the slow head's; chance = 1 / seen domains."""
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import _fail


def _bk_spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or a.std() == 0 or b.std() == 0: return float('nan')
    ra, rb = a.argsort().argsort(), b.argsort().argsort(); return float(np.corrcoef(ra, rb)[0, 1])


def bank_drift(model, cfg, raw_vals_fn, seen_d, slot_of, vis, tag, summary=False):
    from platypus.inference.selcmp import _codes
    from platypus.inference.residual import _bits_rows
    try:
        B = model.__dict__.setdefault('_bank', {'ref': {}, 'p0': {}, 'c0': {}, 'b0': {}, 'pts': []})
        if not B['ref']:
            rv = raw_vals_fn(); rs = np.random.RandomState(int(cfg.seed) + 2221)
            for d in range(len(rv)):
                v = rv[d]; h = len(v) // 2; B['ref'][d] = [bytes(v[i:i + 200]) for i in rs.randint(0, max(1, h - 201), 24)]
        has_slow = model.__dict__.get('_slow') is not None
        if has_slow: from platypus.inference.slowhead import slow_codes
        cur = {}
        for d in seen_d:
            W = B['ref'][d]; zt = F.normalize(_codes(model, W).float(), dim=-1); zs = F.normalize(slow_codes(model, W).float(), dim=-1) if has_slow else None
            bt = float(np.mean([b.mean() for b in _bits_rows(model, W, int(slot_of(d)), False)]))
            cur[d] = (zt, zs, bt)
            if d not in B['p0']: B['p0'][d] = (zt.mean(0), zs.mean(0) if zs is not None else None); B['c0'][d] = (zt, zs); B['b0'][d] = bt
        dr = {}
        for d, (zt, zs, bt) in cur.items():
            p0t, p0s = B['p0'][d]; c0t, c0s = B['c0'][d]
            dt = 1 - float(F.cosine_similarity(zt.mean(0), p0t, dim=0)); ds = (1 - float(F.cosine_similarity(zs.mean(0), p0s, dim=0))) if zs is not None else float('nan')
            st = float((zt * c0t).sum(-1).mean()); ss = float((zs * c0s).sum(-1).mean()) if zs is not None else float('nan')
            dr[d] = (dt, ds, bt - B['b0'][d], st, ss)
        mt = np.mean([v[0] for v in dr.values()]); ms = np.nanmean([v[1] for v in dr.values()]) if has_slow else float('nan')
        for d, v in dr.items():
            if v[2] != 0.0 or B['b0'][d] != cur[d][2]: B['pts'].append((vis, d, v[0], v[0] - mt, v[1], v[1] - ms if has_slow else float('nan'), v[2]))
        print(f"  [BANK DRIFT | {tag}] per seen domain: prototype drift trunk" + ("/slow" if has_slow else "") + ", BPC rise since its first visit, "
              f"own-code stability trunk" + ("/slow" if has_slow else "") + " | " + " ".join(
              f"d{d} {v[0]:.4f}" + (f"/{v[1]:.4f}" if has_slow else "") + f" {v[2]:+.3f} {v[3]:.3f}" + (f"/{v[4]:.3f}" if has_slow else "") for d, v in sorted(dr.items())), flush=True)
        if summary and len(cur) > 1:
            if not B.get('held'):
                rv = raw_vals_fn(); rs2 = np.random.RandomState(int(cfg.seed) + 2222); B['held'] = {}
                for d in range(len(rv)):
                    v = rv[d]; h = len(v) // 2; B['held'][d] = [bytes(v[i:i + 200]) for i in h + rs2.randint(0, max(1, len(v) - h - 201), 32)]
            ds_ = sorted(cur); res = {}
            for nm, j in (('trunk', 0), ('slow', 1)):
                if j == 1 and not has_slow: continue
                Pn = F.normalize(torch.stack([cur[d][j].mean(0) for d in ds_]), dim=-1); Ps = F.normalize(torch.stack([B['p0'][d][j] for d in ds_]), dim=-1)
                an, as_ = [], []
                for i, d in enumerate(ds_):
                    Z = F.normalize((_codes(model, B['held'][d]) if j == 0 else slow_codes(model, B['held'][d])).float(), dim=-1)
                    an.append(float(((Z @ Pn.T).argmax(1) == i).float().mean())); as_.append(float(((Z @ Ps.T).argmax(1) == i).float().mean()))
                res[nm] = (np.mean(an), np.mean(as_), an)
            print(f"  [BANK ROUTING | {tag}] held-out windows to the nearest domain prototype over {len(ds_)} seen domains (chance {1/len(ds_):.2f}): "
                  + " | ".join(f"{nm} now {a:.2f}, stale (first-visit) {b:.2f} [" + " ".join(f"d{d} {x:.2f}" for d, x in zip(ds_, c)) + "]" for nm, (a, b, c) in res.items()), flush=True)
        if summary and B['pts']:
            P = np.array([p[2:] for p in B['pts']], float)
            print(f"  [BANK DRIFT | {tag}] ITEM 2 over {len(P)} (domain, visit) points: rank correlation of prototype drift with the BPC rise - trunk {_bk_spearman(P[:, 0], P[:, 4]):+.2f} "
                  f"(relative to the bank {_bk_spearman(P[:, 1], P[:, 4]):+.2f})" + (f", slow head {_bk_spearman(P[:, 2], P[:, 4]):+.2f} (relative {_bk_spearman(P[:, 3], P[:, 4]):+.2f})" if has_slow else "")
                  + " | + = drift predicts forgetting", flush=True)
    except Exception as e:
        _fail('inference.bankdrift'); print(f"  [BANK DRIFT] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
