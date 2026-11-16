"""KEYED CONTINUATION UNITS (inf_ku; 16 Nov 2026): NAME, DATE and NUMBER, each an indexable key with its own memory of what followed it in
the text the system has read; when a key fires again, the stored continuations consistent with the bytes read since predict the next byte,
mixed with the trunk by a LEARNED GATE (a softmax over the trunk and each firing unit, its logit read from a table keyed by unit, consistent
continuations, bytes into the continuation and the trunk's entropy; one online gradient step on -log p after each byte is scored).
Firings are CAUSAL: every capitalised word fires NAME with the capitalised run so far (scored from the byte after the word); every word
starting with a digit (DATE: or a capital) fires DATE (key: the sentence's last name) and NUMBER (key: the last name + the word before) at its
first byte and is scored from its second; continuations are stored only for real years, months and numbers.
Offline (tools/keyed_units.py RU_CAUSAL 1, Re-DocRED, stand-in trunk): NAME alone saves 67 bits per 1,000 bytes with the gate (2.7% of the
trunk's bits), DATE + NUMBER 3.6 more on top (+5%); FACT and PLACE add nothing. A prediction-time component: the trunk's training is untouched.
Memory: filled at each visit from the visited domain's training text (the first inf_ku_fill bytes of its shard), inf_ku_cap continuations
per key. Readout [KEYED UNITS] at every visit: held-out bits per byte, trunk vs trunk + NAME vs trunk + NAME + DATE + NUMBER, per domain."""
import re, math, time, collections
import numpy as np
import torch
import torch.nn.functional as F

_TOK = re.compile(rb"[A-Za-z][A-Za-z'\-]*|[0-9][0-9,.]*|[^\sA-Za-z0-9]")
_MONTHS = {b'January', b'February', b'March', b'April', b'May', b'June', b'July', b'August', b'September', b'October', b'November', b'December'}
_SKIP = {b'The', b'A', b'In', b'He', b'She', b'It', b'His', b'Her', b'This', b'They', b'An', b'On', b'At', b'As', b'For', b'But', b'And', b'We', b'I'}
_END = {b'.', b'!', b'?'}
_isyear = re.compile(rb'1[0-9]{3}|20[0-9]{2}').fullmatch; _isnum = re.compile(rb'[0-9][0-9,.]*').fullmatch
CL = 24


def ku_fires(b, word=False):
    """causal firings of a byte string: (fires, stores) as lists of (unit, key, position where the continuation starts); word=True adds the
    GENERIC units W1 / W2 (16 Nov, inf_ku_word): a firing at EVERY word end, key = the sentence's last 1 / 2 words, no hand rule"""
    fires, store = [], []; run = []; last = None; prev = None; ti = 0; gap = 0; pw = None
    for m in _TOK.finditer(b):
        w = m.group(0); a, e = m.start(), m.end()
        if w in _END:
            run = []; last = None; prev = w; ti = 0; gap = 0; pw = None; continue
        if word and w[:1].isalnum():
            fires.append(('W1', w, e)); store.append(('W1', w, e))
            if pw is not None: k2 = pw + b' ' + w; fires.append(('W2', k2, e)); store.append(('W2', k2, e))
            pw = w
        c0 = w[:1]
        if (c0.isdigit() or c0.isupper()) and last is not None:
            fires.append(('DATE', last + b' |date', a))
            if _isyear(w) or w in _MONTHS: store.append(('DATE', last + b' |date', a))
            if c0.isdigit() and prev is not None:
                fires.append(('NUMBER', last + b' |' + prev, a))
                if _isnum(w): store.append(('NUMBER', last + b' |' + prev, a))
        if c0.isupper() and not (ti == 0 and w in _SKIP):
            run = run + [w] if (run and gap == 0) else [w]; gap = 0; key = b' '.join(run)
            fires.append(('NAME', key, e)); store.append(('NAME', key, e)); last = key
        else: gap += 1
        prev = w; ti += 1
    return fires, store


def _st(model):
    st = model.__dict__.get('_ku')
    if st is None:
        st = dict(mem={u: {} for u in ('NAME', 'DATE', 'NUMBER', 'W1', 'W2')}, gate={c: {} for c in ('NAME', 'ALL', 'WORD', 'WORD+HAND')}, nfill=0, bytes=0)
        model.__dict__['_ku'] = st
    return st


def ku_fill(model, cfg, raw):
    """store the continuations of the visited domain's training text (the first inf_ku_fill bytes)"""
    t0 = time.time(); st = _st(model); cap = int(getattr(cfg, 'inf_ku_cap', 32)); n = int(getattr(cfg, 'inf_ku_fill', 8_000_000))
    b = bytes(raw[:n]); _, store = ku_fires(b); M = st['mem']
    for u, k, p in store:
        d = M[u].get(k)
        if d is None: d = M[u][k] = collections.deque(maxlen=cap)
        d.append(b[p:p + CL])
    if int(getattr(cfg, 'inf_ku_word', 0)):                                     # 16 Nov GENERIC WORD UNITS: a smaller fill (every word end fires);
        bw = b[:int(getattr(cfg, 'inf_ku_word_fill', 2_000_000))]; wcap = int(getattr(cfg, 'inf_ku_word_cap', 16))   #   plain lists, capped
        for u, k, p in ku_fires(bw, word=True)[1]:
            if u not in ('W1', 'W2'): continue
            d = M[u].get(k)
            if d is None: d = M[u][k] = []
            d.append(bw[p:p + CL])
            if len(d) > wcap: del d[0]
    st['nfill'] += len(store); st['bytes'] += len(b)
    return time.time() - t0


_nbk = lambda n: 0 if n <= 1 else 1 if n == 2 else 2 if n <= 4 else 3 if n <= 8 else 4
_obk = lambda o: 0 if o == 0 else 1 if o <= 3 else 2 if o <= 11 else 3
_ebk = lambda e: 0 if e < 0.5 else 1 if e < 1.5 else 2 if e < 3 else 3


def _row_bits(st, b, pt, ent, units, combo, lr, mask=None):
    """bits saved on one row by the units (learned gate, prequential: each byte scored before the gate learns from it)"""
    fires, _ = ku_fires(b, word=('W1' in units)); M = st['mem']; best = {}
    for u, k, p in fires:
        if u not in units: continue
        conts = M[u].get(k)
        if not conts: continue
        for off in range(1, min(CL, len(b) - p)):
            seen = b[p:p + off]; cons = [c for c in conts if len(c) > off and c[:off] == seen]
            if not cons: break
            nn_ = len(cons); hit = sum(1 for c in cons if c[off] == b[p + off]); pm = (hit + 1.0 / 256) / (nn_ + 1.0)
            q = best.setdefault(p + off, {}).get(u)
            if q is None or nn_ > q[1]: best[p + off][u] = (pm, nn_, off)
    G = st['gate'][combo]; saved = 0.0
    for pos in sorted(best):
        d = best[pos]; ks = [(u, _nbk(v[1]), _obk(v[2]), _ebk(float(ent[pos]))) for u, v in d.items()]
        lg = np.array([0.0] + [G.setdefault(k, math.log(max(d[k[0]][1], 1) / 2.0)) for k in ks]); w = np.exp(lg - lg.max()); w /= w.sum()
        ps = np.array([pt[pos]] + [d[k[0]][0] for k in ks]); p_ = float((w * ps).sum())
        if mask is None or mask[pos]: saved += -math.log2(max(pt[pos], 1e-12)) + math.log2(max(p_, 1e-12))
        for j, k in enumerate(ks): G[k] += lr * w[j + 1] * (ps[j + 1] - p_) / max(p_, 1e-12)
    return saved, len(best)


@torch.no_grad()
def ku_readout(model, cfg, vals, seen_d, next_d, slot_of, tag, nb=8):
    """[KEYED UNITS] held-out bits per byte: the trunk alone, + NAME, + NAME + DATE + NUMBER (learned gates, prequential), per domain"""
    st = model.__dict__.get('_ku')
    if not st or not st['nfill']: return
    try:
        lr = float(getattr(cfg, 'inf_ku_lr', 0.05)); Ts = int(model.cfg.seq_len); parts = []; tot = [0.0, 0.0, 0.0, 0, 0.0, 0.0]
        wd = int(getattr(cfg, 'inf_ku_word', 0)) and bool(st['mem']['W1'])
        doms = [(d, 'read') for d in seen_d] + ([(next_d, 'NEXT')] if next_d is not None and next_d not in seen_d else [])
        was = model.training; model.eval()
        try:
            for d, kind in doms:
                g = vals[d]; cur = model.__dict__.get('_cur_dom'); sl = int(slot_of(d)) if kind == 'read' or cur is None else int(slot_of(int(cur)))
                if hasattr(g, 'reset'): g.reset()
                acc = [0.0, 0.0, 0.0, 0, 0.0, 0.0]
                for _ in range(nb):
                    x, y = g(); lg = model(x, sl)[:, -Ts:].float(); yy = y[:, -Ts:]
                    lp = F.log_softmax(lg, -1); pt = lp.gather(-1, yy[..., None])[..., 0].exp().cpu().numpy(); ent = (-(lp.exp() * lp).sum(-1) / math.log(2)).cpu().numpy()
                    yb = yy.cpu().numpy().astype(np.uint8)
                    for r in range(yb.shape[0]):
                        b = bytes(yb[r]); acc[0] += float(-np.log2(np.maximum(pt[r], 1e-12)).sum()); acc[3] += len(b)
                        acc[1] += _row_bits(st, b, pt[r], ent[r], ('NAME',), 'NAME', lr)[0]; acc[2] += _row_bits(st, b, pt[r], ent[r], ('NAME', 'DATE', 'NUMBER'), 'ALL', lr)[0]
                        if wd:
                            acc[4] += _row_bits(st, b, pt[r], ent[r], ('W1', 'W2'), 'WORD', lr)[0]
                            acc[5] += _row_bits(st, b, pt[r], ent[r], ('NAME', 'DATE', 'NUMBER', 'W1', 'W2'), 'WORD+HAND', lr)[0]
                for i in range(6): tot[i] += acc[i]
                n = max(1, acc[3]); parts.append(f"domain {d} ({kind}): trunk {acc[0] / n:.4f}, + NAME {-acc[1] / n:+.4f}, + NAME + DATE + NUMBER {-acc[2] / n:+.4f}"
                                                 + (f", GENERIC WORDS {-acc[4] / n:+.4f}, WORDS + NAME + DATE + NUMBER {-acc[5] / n:+.4f}" if wd else ''))
        finally: model.train(was)
        n = max(1, tot[3]); M = st['mem']
        print(f"  [KEYED UNITS | {tag}] change in held-out bits per byte (negative = the units help; learned gate, prequential) || ALL DOMAINS: trunk {tot[0] / n:.4f}, + NAME "
              f"{-tot[1] / n:+.4f}, + NAME + DATE + NUMBER {-tot[2] / n:+.4f} (DATE + NUMBER on top of NAME {-(tot[2] - tot[1]) / n:+.4f}) || " + " | ".join(parts)
              + f" || memory: {st['bytes'] / 1e6:.1f} MB read, keys NAME {len(M['NAME'])}, DATE {len(M['DATE'])}, NUMBER {len(M['NUMBER'])}", flush=True)
        if wd:
            print(f"  [GENERIC KEYED UNITS | {tag}] every word end keyed by its last 1-2 words (no hand rule; learned gate, prequential), change in held-out "
                  f"bits per byte || ALL DOMAINS: GENERIC WORDS {-tot[4] / n:+.4f} vs hand NAME + DATE + NUMBER {-tot[2] / n:+.4f}; WORDS + HAND {-tot[5] / n:+.4f} "
                  f"(hand units on top of words {-(tot[5] - tot[4]) / n:+.4f}) || keys W1 {len(M['W1'])}, W2 {len(M['W2'])}", flush=True)
    except Exception as e:
        from platypus.common import _fail
        _fail('inference.kunits'); print(f"  [KEYED UNITS] readout unavailable ({type(e).__name__}: {e})"[:240], flush=True)
