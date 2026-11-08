"""GIST TEST (gist_test 1; readout at the end of the run, the model is restored and unchanged).
Does a learned gist code of a chunk carry information the trunk does not already have? Measured as a code length:
    bits(gist) + bits(chunk | gist, trunk)   against   bits(chunk | trunk)
with the slot given (so the per-slot FiLM code and private parts are already in the baseline). The gist of a chunk is the JEPA index
target code of the chunk (arm G: names masked, as the gist spec; arm V: names kept, as a reference), mean over its 128-byte pieces,
through a learned k-dimensional bottleneck quantised to 8 bits per dimension (cost 8k bits per chunk), injected into the trunk's
input embedding through a learned map and a zero-initialised gate. Only the bottleneck, the map and the gate are trained, on the first
half of each seen domain's validation text (never trained on by the trunk); every number is on the second half. Controls: the same
trained injector fed ANOTHER chunk's gist (shuffled), and the baseline with no gist. Chunk lengths: gist_test_len."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from platypus.common import _fail


def _gist_codes(cfg, chunks, arm, dev):
    """mean of the arm's target codes over each chunk's 128-byte pieces"""
    from platypus.inference.jindex import _JX, _jx_batch, _jx_ids
    X9 = _JX[arm]; L = int(getattr(cfg, 'inf_jx_len', 128)); out = []
    with torch.no_grad():
        for c in chunks:
            ps = [c[i:i + L] for i in range(0, len(c), L) if len(c[i:i + L]) >= 16]
            out.append(X9['tg'](_jx_batch([_jx_ids(p, X9['gen']) for p in ps], L, dev)).mean(0))
    return torch.stack(out)


def gist_test(model, cfg, raw_vals, seen_d, slot_of):
    from platypus.inference.jindex import _JX
    from platypus.conduit import _carry_save, _carry_load
    from platypus.layers import _CARRY
    from platypus.inference.store import _INFM
    from platypus.private import _XDG
    from platypus.ngrams import _COMPILED
    arms = [a for a in ('G', 'V') if a in _JX]
    if not arms: print("  [GIST TEST] skipped: the JEPA index (inf_jx, arms V and G) is off", flush=True); return
    dev = model.embed.weight.device; rs = np.random.RandomState(int(cfg.seed) + 515)
    lens = [int(x) for x in (getattr(cfg, 'gist_test_len', None) or (128, 1024))]
    ks = [int(x) for x in (getattr(cfg, 'gist_test_k', None) or (8, 32, 128))]
    steps = int(getattr(cfg, 'gist_test_steps', 300)); B = int(getattr(cfg, 'gist_test_batch', 16)); NT = int(getattr(cfg, 'gist_test_n', 64))
    was = model.training; model.eval(); comp0 = _COMPILED.get('failed'); _COMPILED['failed'] = True       # eager forward, so the embedding hook runs
    req = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in req: p.requires_grad_(False)
    sv_carry = _carry_save(); _CARRY.update(on=False, st={}, new={}); sv_inf = _INFM.get('on'); _INFM['on'] = False; sv_x = _XDG.get('cur')
    inj = {'z': None}
    hk = model.embed.register_forward_hook(lambda m_, i_, o_: o_ if inj['z'] is None else o_ + inj['z'][:, None, :].to(o_.dtype))
    try:
        for Lc in lens:
            tr, te = [], []
            for d in seen_d:
                v = raw_vals[d]; h = len(v) // 2
                for _ in range(max(B * 4, 64)):
                    i = rs.randint(0, h - Lc - 1); tr.append((d, v[i:i + Lc]))
                for _ in range(NT):
                    i = rs.randint(h, len(v) - Lc - 1); te.append((d, v[i:i + Lc]))
            def bits(rows, z):
                """mean bits per byte of each row, slot given; z (n, d_model) added to the input embedding or None"""
                out = []
                for a in range(0, len(rows), 16):
                    rr = rows[a:a + 16]
                    for d in sorted(set(r[0] for r in rr)):
                        ii = [j for j, r in enumerate(rr) if r[0] == d]
                        X = torch.tensor([list(rr[j][1]) for j in ii], dtype=torch.long, device=dev)
                        inj['z'] = None if z is None else z[[a + j for j in ii]]
                        sl = int(slot_of(d)); _XDG['cur'] = sl
                        lp = F.log_softmax(model(X[:, :-1], sl).float(), -1)
                        nll = -lp.gather(-1, X[:, 1:, None]).squeeze(-1).mean(1) / math.log(2)
                        for jj, v_ in zip(ii, nll.tolist()): out.append((a + jj, v_))
                inj['z'] = None
                return np.array([v for _, v in sorted(out)])
            with torch.no_grad(): base = bits(te, None)
            res = {}
            for arm in arms:
                Gtr = _gist_codes(cfg, [c for _, c in tr], arm, dev); Gte = _gist_codes(cfg, [c for _, c in te], arm, dev)
                mu, sd = Gtr.mean(0), Gtr.std(0) + 1e-6
                for k in (ks if arm == 'G' else ks[-1:]):
                    torch.manual_seed(int(cfg.seed) + 77 * k)
                    A = nn.Linear(Gtr.shape[1], k).to(dev); Bm = nn.Linear(k, model.embed.weight.shape[1], bias=False).to(dev)
                    gate = nn.Parameter(torch.zeros(1, device=dev)); op = torch.optim.Adam(list(A.parameters()) + list(Bm.parameters()) + [gate], lr=1e-3)
                    def code(G, train):
                        c = torch.tanh(A((G - mu) / sd))                                      # in (-1, 1): 8 bits per dimension
                        q = torch.round((c + 1) * 127.5) / 127.5 - 1
                        c = c + (q - c).detach() if train else q                             # straight-through quantisation
                        return gate * Bm(c)
                    for st in range(steps):
                        ix = rs.randint(0, len(tr), B)
                        loss = 0.0
                        for d in sorted(set(tr[i][0] for i in ix)):
                            jj = [i for i in ix if tr[i][0] == d]
                            X = torch.tensor([list(tr[i][1]) for i in jj], dtype=torch.long, device=dev)
                            inj['z'] = code(Gtr[jj], True); _XDG['cur'] = int(slot_of(d))
                            lp = F.log_softmax(model(X[:, :-1], int(slot_of(d))).float(), -1)
                            loss = loss + (-lp.gather(-1, X[:, 1:, None]).squeeze(-1).mean()) * len(jj) / B
                        inj['z'] = None; op.zero_grad(); loss.backward(); op.step()
                    with torch.no_grad():
                        own = bits(te, code(Gte, False))
                        perm = np.roll(np.arange(len(te)), len(te) // 2)                     # another chunk's gist (half the list away: mostly another domain)
                        shuf = bits(te, code(Gte[perm], False))
                    res[(arm, k)] = (own, shuf, float(gate.detach().abs()))
            nB = Lc - 1
            print(f"  [GIST TEST | chunks of {Lc} bytes, slot given, {len(te)} held-out chunks over {len(seen_d)} domains] baseline (trunk + slot code, no gist) "
                  f"{base.mean():.4f} bits/byte = {base.mean() * nB:.1f} bits per chunk", flush=True)
            for (arm, k), (own, shuf, g) in res.items():
                sv = (base - own).mean() * nB; ss = (base - shuf).mean() * nB; cost = 8 * k
                print(f"    gist arm {arm} ({'names masked' if arm == 'G' else 'names kept'}), {k} dims x 8 bits = {cost} bits: with the chunk's own gist "
                      f"{own.mean():.4f} bits/byte, SAVES {sv:+.1f} bits per chunk (net of the gist's cost {sv - cost:+.1f}) | another chunk's gist "
                      f"{shuf.mean():.4f}, saves {ss:+.1f} | gate {g:.3f} | chunks helped {np.mean(own < base):.2f}", flush=True)
    except Exception as e:
        _fail('inference.gisttest'); print(f"  [GIST TEST] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
    finally:
        hk.remove(); inj['z'] = None
        for p, r in req: p.requires_grad_(r)
        _carry_load(sv_carry); _INFM['on'] = sv_inf; _XDG['cur'] = sv_x
        if comp0 is None: _COMPILED.pop('failed', None)
        else: _COMPILED['failed'] = comp0
        model.train(was)
