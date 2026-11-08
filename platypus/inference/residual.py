"""MEMORY RESIDUAL and GRADUATION readouts (cg27a; readout only, the model is restored and unchanged).

MEMORY RESIDUAL (inf_resid 1): what the trunk still cannot predict of what the fast store holds, measured as a code length with no gist:
for a sample of stored real sentences, bits per byte from the trunk alone (store off) and with the store on, on all bytes and on name
bytes, grouped by visits since the sentence was stored (what has not graduated into the weights yet). SCHEMA SUPPORT from the JEPA index
arm G (names masked): the mean cosine of a sentence's generic code to its 5 nearest other stored sentences; residual by tercile of support
(a sentence that fits a common frame should leave only its names as residual). TRANSFER: held-out validation text of each seen domain,
store on minus store off, on all bytes and on name bytes (does what the store holds help text it never read?).

GRADUATION (grad_eval 1): per slot, held-out BPC of its domain with every part as trained, then with the slot's private trunk units off,
its private n-gram tables off, its FiLM code zeroed, and all three off. A small rise means the domain's knowledge lives in the shared
trunk (it has graduated); a large rise means it is held privately."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import _fail
from platypus.config import PILE_DOMAINS


def _bits_rows(model, rows, slot, store_on):
    """per row: bits of every byte after the first (list of numpy arrays); rows are bytes, scored as ' '+row with the slot given"""
    from platypus.conduit import _carry_save, _carry_load
    from platypus.layers import _CARRY
    from platypus.private import _XDG
    from platypus.inference.store import _INFM
    dev = model.embed.weight.device; out = []
    for a in range(0, len(rows), 32):
        rr = [b' ' + r for r in rows[a:a + 32]]; L = max(len(r) for r in rr)
        X = torch.full((len(rr), L), 32, dtype=torch.long, device=dev)
        for j, r in enumerate(rr): X[j, :len(r)] = torch.tensor(list(r), device=dev)
        sv = _carry_save(); _CARRY.update(on=False, st={}, new={}); sc = _XDG['cur']; _XDG['cur'] = slot; si = _INFM.get('on'); _INFM['on'] = bool(store_on) and _INFM.get('M') is not None
        try:
            with torch.no_grad(): lp = F.log_softmax(model(X[:, :-1], slot).float(), -1)
        finally: _carry_load(sv); _XDG['cur'] = sc; _INFM['on'] = si
        nl = (-lp.gather(-1, X[:, 1:, None]).squeeze(-1) / math.log(2)).cpu().numpy()
        for j, r in enumerate(rr): out.append(nl[j, :len(r) - 1])
    return out


def _name_mask(row):
    from platypus.inference.units import _INF_NAME
    m = np.zeros(len(row), dtype=bool)
    for x in _INF_NAME.finditer(bytes(row)): m[x.start():x.end()] = True
    return m


def resid_readout(model, cfg, S_, raw_vals, seen_d, slot_of, tag):
    """see the module docstring"""
    try:
        rs = np.random.RandomState(int(cfg.seed) + 6161); vis = int(model.__dict__.get('_visit', 0)); n = int(getattr(cfg, 'inf_resid_n', 240))
        it = [x for x in S_.get('it', []) if x.get('k') in ('real', 'ver') and not x.get('gone') and 30 <= len(x['s']) <= 400 and 'v' in x]
        if not it: print(f"  [inference MEMORY RESIDUAL | {tag}] no stored sentences", flush=True); return
        it = [it[int(i)] for i in rs.choice(len(it), min(n, len(it)), replace=False)]
        rec = []                                                                 # (age, trunk bits/byte, store bits/byte, trunk name bits/byte, store name bits/byte)
        for sl in sorted(set(int(slot_of(x['d'])) for x in it)):
            grp = [x for x in it if int(slot_of(x['d'])) == sl]; rows = [bytes(x['s']) for x in grp]
            bt = _bits_rows(model, rows, sl, False); bs = _bits_rows(model, rows, sl, True)
            for x, r, a, b in zip(grp, rows, bt, bs):
                nm = _name_mask(r)[:len(a)]                                     # bits[k] is byte k of the row (' ' + row shifts by one)
                rec.append((vis - int(x['v']), float(a.mean()), float(b.mean()), float(a[nm].mean()) if nm.any() else float('nan'), float(b[nm].mean()) if nm.any() else float('nan'), x, a, b))
        def line(R):
            if not R: return "n 0"
            A = np.array([r[:5] for r in R], dtype=float)
            return (f"n {len(R)}: trunk alone {A[:, 1].mean():.3f} bits/byte (names {np.nanmean(A[:, 3]):.3f}), with the store {A[:, 2].mean():.3f} "
                    f"(names {np.nanmean(A[:, 4]):.3f}) -> residual the store still carries {A[:, 1].mean() - A[:, 2].mean():+.3f} (names {np.nanmean(A[:, 3]) - np.nanmean(A[:, 4]):+.3f})")
        ag = {'0-1': [r for r in rec if r[0] <= 1], '2-3': [r for r in rec if 2 <= r[0] <= 3], '4-9': [r for r in rec if 4 <= r[0] <= 9], '10+': [r for r in rec if r[0] >= 10]}
        print(f"  [inference MEMORY RESIDUAL | {tag}] stored real sentences, bits per byte, slot given | ALL {line(rec)} || by visits since stored: "
              + " | ".join(f"{k} {line(v)}" for k, v in ag.items() if v), flush=True)
        # WHERE THE RESIDUAL IS: (1) the HIGH-SURPRISE bytes of the same sentences (the trunk's top 10% per sentence); (2) the ANSWER bytes of the
        # system's own recorded one-hop facts (the object after 'A r ', facts the reader recorded from text - never the probe answer key)
        hs_t, hs_s, rest_t, rest_s = [], [], [], []
        for r in rec:
            a, b = r[6], r[7]; m = a >= np.quantile(a, 0.9)
            hs_t.append(float(a[m].mean())); hs_s.append(float(b[m].mean())); rest_t.append(float(a[~m].mean()) if (~m).any() else float('nan')); rest_s.append(float(b[~m].mean()) if (~m).any() else float('nan'))
        fx = [f for f in S_.get('facts', []) if not f.get('freed') and all(isinstance(f.get(k), str) for k in ('A', 'r', 'B'))]
        fx = [fx[int(i)] for i in rs.choice(len(fx), min(n, len(fx)), replace=False)] if fx else []
        ft, fs_ = [], []
        for sl in sorted(set(int(slot_of(f['d'])) for f in fx)):
            grp = [f for f in fx if int(slot_of(f['d'])) == sl]
            rows = [f"{f['A']} {f['r']} {f['B']}.".encode('latin-1', 'replace') for f in grp]
            bt = _bits_rows(model, rows, sl, False); bs = _bits_rows(model, rows, sl, True)
            for f, a, b in zip(grp, bt, bs):
                p0 = len(f"{f['A']} {f['r']} ".encode('latin-1', 'replace')); p1 = p0 + len(f['B'].encode('latin-1', 'replace'))
                ft.append(float(a[p0:p1].sum())); fs_.append(float(b[p0:p1].sum()))
        print(f"  [inference MEMORY RESIDUAL | {tag}] WHERE IT IS: high-surprise bytes (trunk top 10% per sentence) trunk {np.mean(hs_t):.3f} vs store {np.mean(hs_s):.3f} bits/byte "
              f"(saves {np.mean(hs_t) - np.mean(hs_s):+.3f}); the other 90% {np.nanmean(rest_t):.3f} vs {np.nanmean(rest_s):.3f} ({np.nanmean(rest_t) - np.nanmean(rest_s):+.3f})"
              + (f" | ANSWER bytes of {len(ft)} recorded facts (object after 'A r ', the system's own records): trunk {np.mean(ft):.2f} vs store {np.mean(fs_):.2f} bits per answer "
                 f"(saves {np.mean(ft) - np.mean(fs_):+.2f}; store below 1 bit for {np.mean(np.array(fs_) < 1.0):.2f}, trunk for {np.mean(np.array(ft) < 1.0):.2f})" if ft else " | no recorded facts"), flush=True)
        from platypus.inference.jindex import _JX, _jx_batch, _jx_ids
        if 'G' in _JX and len(rec) >= 12:                                         # SCHEMA SUPPORT from the generic (names masked) arm
            X9 = _JX['G']; L = int(getattr(cfg, 'inf_jx_len', 128)); dev = model.embed.weight.device
            with torch.no_grad(): Z = F.normalize(X9['tg'](_jx_batch([_jx_ids(r[5]['s'], X9['gen']) for r in rec], L, dev)).float(), dim=-1)
            C = Z @ Z.T; C.fill_diagonal_(-1.0); sup = C.topk(min(5, len(rec) - 1), -1).values.mean(-1).cpu().numpy()
            q = np.quantile(sup, [1 / 3, 2 / 3]); T = [[r for r, s in zip(rec, sup) if s <= q[0]], [r for r, s in zip(rec, sup) if q[0] < s <= q[1]], [r for r, s in zip(rec, sup) if s > q[1]]]
            nb = lambda R: np.nanmean([r[1] for r in R]) - np.nanmean([r[3] for r in R]) if R else float('nan')
            print(f"  [inference MEMORY RESIDUAL | {tag}] by SCHEMA SUPPORT (arm G, names masked: mean cosine to the 5 nearest stored sentences; terciles "
                  f"{sup.min():.2f}-{q[0]:.2f}-{q[1]:.2f}-{sup.max():.2f}): LOW {line(T[0])} | MID {line(T[1])} | HIGH {line(T[2])} || trunk alone, all bytes minus name bytes "
                  f"(frame vs bindings): low {nb(T[0]):+.3f}, high {nb(T[2]):+.3f}", flush=True)
        if _INFM_present():                                                      # TRANSFER to text the store never read
            tr = []
            for d in seen_d:
                v = raw_vals[d]; h = len(v) // 2; sl = int(slot_of(d))
                rows = [bytes(v[i:i + 256]) for i in rs.randint(h, len(v) - 257, 12)]
                a = _bits_rows(model, rows, sl, False); b = _bits_rows(model, rows, sl, True)
                for r, x, y in zip(rows, a, b):
                    nm = _name_mask(r)[:len(x)]; tr.append((d, float(x.mean() - y.mean()), float(x[nm].mean() - y[nm].mean()) if nm.any() else float('nan')))
            print(f"  [inference MEMORY RESIDUAL | {tag}] TRANSFER (held-out validation text, store off minus on; + = the store helps text it never read): all bytes "
                  f"{np.mean([t[1] for t in tr]):+.4f}, name bytes {np.nanmean([t[2] for t in tr]):+.4f} bits/byte | per domain "
                  + " ".join(f"{PILE_DOMAINS[d][:6]} {np.mean([t[1] for t in tr if t[0] == d]):+.4f}" for d in seen_d), flush=True)
    except Exception as e:
        _fail('inference.residual'); print(f"  [inference MEMORY RESIDUAL] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _INFM_present():
    from platypus.inference.store import _INFM
    return _INFM.get('M') is not None


def grad_readout(model, cfg, vals, seen_d, slot_of, tag, lite=False):
    """see the module docstring; vals[d] is the validation getter bpc() takes; lite: only every part on and all off (the per-visit series)"""
    from platypus.inference.store import bpc
    from platypus.private import _PUOFF
    from platypus.ngrams import _COMPILED
    try:
        comp0 = _COMPILED.get('failed'); _COMPILED['failed'] = True              # eager: the switches below are read at run time
        res = []
        try:
            for d in seen_d:
                sl = int(slot_of(d)); r = {'on': bpc(model, vals[d], sl)}
                T = model.__dict__.get('_ngp'); fc = getattr(model, 'film_codes', None); has_f = fc is not None and fc.shape[1] > 0
                if not lite:
                  _PUOFF['d'] = sl
                  try: r['units off'] = bpc(model, vals[d], sl)
                  finally: _PUOFF['d'] = None
                if T is not None and not lite:
                    sv = T[sl]; T[sl] = [_ZeroEmb(E) for E in sv]
                    try: r['n-grams off'] = bpc(model, vals[d], sl)
                    finally: T[sl] = sv
                if has_f and not lite:
                    with torch.no_grad(): c0 = fc[sl].clone(); fc[sl].zero_()
                    try: r['FiLM off'] = bpc(model, vals[d], sl)
                    finally:
                        with torch.no_grad(): fc[sl].copy_(c0)
                _PUOFF['d'] = sl
                if T is not None: sv = T[sl]; T[sl] = [_ZeroEmb(E) for E in sv]
                if has_f:
                    with torch.no_grad(): c0 = fc[sl].clone(); fc[sl].zero_()
                try: r['all off'] = bpc(model, vals[d], sl)
                finally:
                    _PUOFF['d'] = None
                    if T is not None: T[sl] = sv
                    if has_f:
                        with torch.no_grad(): fc[sl].copy_(c0)
                res.append((d, sl, r))
        finally:
            if comp0 is None: _COMPILED.pop('failed', None)
            else: _COMPILED['failed'] = comp0
        ks = [k for k in ('units off', 'n-grams off', 'FiLM off', 'all off') if any(k in r for _, _, r in res)]
        print(f"  [GRADUATION | {tag}] held-out BPC per slot with its private parts switched off (rise = knowledge held privately, ~0 = graduated to the shared trunk): mean rise "
              + ", ".join(f"{k} {np.mean([r[k] - r['on'] for _, _, r in res if k in r]):+.4f}" for k in ks) + " | per domain "
              + " ".join(f"{PILE_DOMAINS[d][:6]} (slot {sl}) {r['on']:.3f}: " + "/".join(f"{r[k] - r['on']:+.3f}" for k in ks if k in r) for d, sl, r in res), flush=True)
    except Exception as e:
        _fail('inference.residual:grad'); print(f"  [GRADUATION] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


class _ZeroEmb(torch.nn.Module):
    """a private n-gram table switched off: same lookup interface, all zeros"""
    def __init__(self, E):
        super().__init__(); self.num_embeddings = E.num_embeddings; self.embedding_dim = E.embedding_dim; self.weight = torch.zeros(1, E.embedding_dim, device=E.weight.device, dtype=E.weight.dtype).expand(E.num_embeddings, E.embedding_dim)   # a view: no memory
    def forward(self, ix): return torch.zeros(*ix.shape, self.embedding_dim, device=self.weight.device, dtype=self.weight.dtype)


def pair_judge_readout(model, cfg, slot_of, tag):
    """RELATION JUDGE ON REAL PROSE (inf_rdr_judge 1; readout only): is there a relational signal in the trunk for pairs of entities that
    co-occur in Re-DocRED paragraphs (the embedded benchmark's gold entities; 29% of clause-level pairs are gold relations)?
    (a) SUBSTITUTION, pairs in one clause: bits on Y's name with X as written, minus the mean over up to 3 other entities of the same
        paragraph put in X's place (negative = the true X makes Y easier to predict);
    (b) MUTUAL COMPRESSION, pairs in different sentences: bits on Y's sentence after X's sentence, minus after a random other sentence of
        the same paragraph without Y.
    Reported as the AUC of gold-relation pairs against the other co-occurring pairs (0.5 = no relational signal) - the separation a
    model-based relation detector built on this judge would start from."""
    import re
    from platypus.inference.probes import _bench
    try:
        class _C: inf_bench = 'embedded'; inf_bench_maxamb = 1
        G = _bench(_C()); t2e = {t.encode('utf-8'): next(iter(es)) for t, es in G['_t2e'].items()}
        REL = {(min(a, b), max(a, b)) for a, b, _, _ in G['rel']}
        rs = np.random.RandomState(int(cfg.seed) + 4401); WR = re.compile(rb"[A-Za-z0-9][A-Za-z0-9'-]*")
        sub, mc = [], []                                                         # (gold, score)
        for dk, docs in sorted(G['docs'].items()):
            sl = int(slot_of(int(dk))) if callable(slot_of) else 0
            for doc in docs:
                b = doc.encode('utf-8', 'replace')[:3000]; sents = [s for s in re.split(rb'(?<=[.!?])\s+', b) if s]
                ments = [[(m.start(), m.end(), t2e[m.group(0)]) for m in WR.finditer(s) if m.group(0) in t2e] for s in sents]
                ents = sorted({e for ms in ments for _, _, e in ms}); names = {}
                for ms, s in zip(ments, sents):
                    for a, c, e in ms: names.setdefault(e, s[a:c])
                rows, meta = [], []
                for si, (s, ms) in enumerate(zip(sents, ments)):                  # (a) substitution within a clause
                    for cl in re.finditer(rb'[^.,;:!?()"]+', s):
                        inn = [(a, c, e) for a, c, e in ms if cl.start() <= a and c <= cl.end()]
                        for i in range(len(inn)):
                            for j in range(i + 1, len(inn)):
                                (xa, xc, ex), (ya, yc, ey) = inn[i], inn[j]
                                if ex == ey: continue
                                alt = [e for e in ents if e not in (ex, ey)]
                                if not alt: continue
                                for e2 in [alt[k] for k in rs.choice(len(alt), min(3, len(alt)), replace=False)]:
                                    s2 = s[:xa] + names[e2] + s[xc:]; d = len(names[e2]) - (xc - xa)
                                    rows += [s, s2]; meta += [('a', (min(ex, ey), max(ex, ey)), ya, yc, ya + d, yc + d)]
                if rows:
                    bits = _bits_rows(model, rows, sl, False); acc = {}
                    for k, mt in enumerate(meta):
                        _, p, y0, y1, y0b, y1b = mt; t = float(bits[2 * k][y0:y1].sum()); f = float(bits[2 * k + 1][y0b:y1b].sum())
                        acc.setdefault((p, y0, sents.index(rows[2 * k]) if rows[2 * k] in sents else -1), []).append((t, f))
                    for (p, _, _), v in acc.items(): sub.append((p in REL, v[0][0] - float(np.mean([f for _, f in v]))))
                ys = {}                                                           # (b) mutual compression across sentences
                for si, ms in enumerate(ments):
                    for _, _, e in ms: ys.setdefault(e, set()).add(si)
                rows, meta = [], []
                for ex in ents:
                    for ey in ents:
                        if ex >= ey: continue
                        for xi in sorted(ys[ex])[:1]:
                            for yi in sorted(ys[ey] - ys[ex])[:1]:
                                if xi == yi: continue
                                oth = [k for k in range(len(sents)) if k not in (xi, yi) and k not in ys[ey]]
                                if not oth: continue
                                ri = oth[rs.randint(len(oth))]; n0 = len(sents[xi]) + 1; n1 = len(sents[ri]) + 1
                                rows += [sents[xi] + b' ' + sents[yi], sents[ri] + b' ' + sents[yi]]; meta.append(((ex, ey), n0, n1))
                if rows:
                    bits = _bits_rows(model, rows, sl, False)
                    for k, (p, n0, n1) in enumerate(meta): mc.append((p in REL, float(bits[2 * k][n0:].mean() - bits[2 * k + 1][n1:].mean())))
        def auc(L):
            g = np.array([s for y, s in L if y]); o = np.array([s for y, s in L if not y])
            if not len(g) or not len(o): return float('nan')
            return float(((g[:, None] < o[None, :]).mean() + 0.5 * (g[:, None] == o[None, :]).mean()))   # lower score (more compression) = predicted related
        print(f"  [inference RELATION JUDGE ON PROSE | {tag}] Re-DocRED paragraphs, trunk alone: (a) SUBSTITUTION in a clause: {len(sub)} pairs, gold {np.mean([y for y, _ in sub]) if sub else float('nan'):.2f}, "
              f"AUC {auc(sub):.3f} (mean bits saved by the true X: gold {-np.mean([s for y, s in sub if y]) if any(y for y, _ in sub) else float('nan'):+.2f}, other {-np.mean([s for y, s in sub if not y]) if any(not y for y, _ in sub) else float('nan'):+.2f}) | "
              f"(b) MUTUAL COMPRESSION across sentences: {len(mc)} pairs, gold {np.mean([y for y, _ in mc]) if mc else float('nan'):.2f}, AUC {auc(mc):.3f} | 0.5 = no relational signal", flush=True)
    except Exception as e:
        _fail('inference.residual:judge'); print(f"  [inference RELATION JUDGE ON PROSE] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
