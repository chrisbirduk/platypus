"""28 Nov 2026 kra62e WORD-FREE READOUT (inf_wordfree; readout only, the model is not changed).

Un-parked from the local tests 12d-12g (units_lex.py). Each sleep the stored real sentences are lower-cased and their spaces REMOVED;
units are then found with no word boundary given:
  1. candidate units: substrings (up to LMAX bytes) seen MINF+ times, grown a byte at a time;
  2. EDGE UNITS: strings seen as a span's first bytes and as a span's last bytes (2+ times each), spans being the stretches between
     non-letter bytes (punctuation, digits) - those bytes mark edges as a record separator does (local test 11k on a byte stream);
  3. a LEXICON by minimum description length (de Marcken 1996; unigram LM as in SentencePiece; hard EM): each sentence cut by Viterbi
     into the cheapest units (-log2 p(unit)); a unit is kept if what it saves on the text exceeds SPELL bits a byte to spell it;
     edge units are always kept.
Scored against the removed spaces (word boundaries) and, as a stand-in for names, runs of capitalised words in the original sentence
(evaluation only): boundary precision / recall / F1, name starts and ends found, cuts inside names, names cut EXACTLY as one unit,
against a random cut at the same rate. Local (Wikipedia, 1,500 documents): F1 0.760, names whole 0.306 (gold NER names).
"""
import math, re, time, collections
import numpy as np

LMAX, MINF, IT = 14, 3, 6


def _cands(T):
    FR = collections.Counter()
    for s in T:
        for ch in s: FR[ch] += 1
    for k in range(2, LMAX + 1):
        ck = collections.Counter()
        for s in T:
            for i in range(len(s) - k + 1):
                w = s[i:i + k]
                if FR.get(w[:-1], 0) >= MINF and FR.get(w[1:], 0) >= MINF: ck[w] += 1
        n0 = len(FR); FR.update({w: c for w, c in ck.items() if c >= MINF})
        if len(FR) == n0: break
    return FR


def _edge(T):
    cp, cs = collections.Counter(), collections.Counter()
    for s in T:
        for x in re.split(r'[^a-z]+', s):
            if len(x) < 3: continue
            for k in range(3, min(LMAX, len(x)) + 1): cp[x[:k]] += 1; cs[x[-k:]] += 1
    return {w: min(cp[w], cs[w]) for w in cp if cp[w] >= 2 and cs.get(w, 0) >= 2}


def _vit(s, C, unk, whole_ok=True):
    n = len(s); B = [0.0] + [1e18] * n; BK = [0] * (n + 1)
    for i in range(1, n + 1):
        for k in range(1, min(LMAX, i) + 1):
            if not whole_ok and k == n: continue
            w = s[i - k:i]; c = C.get(w)
            if c is None:
                if k > 1: continue
                c = unk
            if B[i - k] + c < B[i]: B[i] = B[i - k] + c; BK[i] = k
    cu, i = [], n
    while i > 0: i -= BK[i]; cu.append(i)
    return {c for c in cu if c > 0}, B[n]


def _costs(lex):
    N = sum(lex.values()) + 256; return {w: -math.log2(c / N) for w, c in lex.items()}, -math.log2(1 / N)


def _segs(s, cuts):
    b = [0] + sorted(cuts) + [len(s)]; return [s[x:y] for x, y in zip(b, b[1:])]


def _score(D, seg):
    c = collections.Counter()
    for (s, wb, g), pb in zip(D, seg):
        c['tp'] += len(pb & wb); c['fp'] += len(pb - wb); c['fn'] += len(wb - pb)
        c['ns'] += sum(a in pb for a, b in g if a > 0); c['nsn'] += sum(1 for a, b in g if a > 0)
        c['ne'] += sum(b in pb for a, b in g if b < len(s)); c['nen'] += sum(1 for a, b in g if b < len(s))
        c['exact'] += sum((a == 0 or a in pb) and (b == len(s) or b in pb) and not any(a < i < b for i in pb) for a, b in g); c['n'] += len(g)
        c['inside'] += sum(len([i for i in pb if a < i < b]) for a, b in g)
    pr = c['tp'] / max(1, c['tp'] + c['fp']); rc = c['tp'] / max(1, c['tp'] + c['fn']); f1 = 2 * pr * rc / max(1e-9, pr + rc)
    return (f"boundaries P/R/F1 {pr:.3f}/{rc:.3f}/{f1:.3f} | name starts / ends found {c['ns'] / max(1, c['nsn']):.2f} / {c['ne'] / max(1, c['nen']):.2f} | "
            f"cuts inside names {c['inside'] / max(1, c['n']):.2f} a name | names cut EXACTLY as one unit {c['exact'] / max(1, c['n']):.3f} (n {c['n']})")


def wordfree_readout(model, cfg, S_):
    """the readout; S_ is the inference store (its stored real sentences)"""
    if not int(getattr(cfg, 'inf_wordfree', 0)): return
    v_ = int(model.__dict__.get('_visit', 0))
    if v_ % max(1, int(getattr(cfg, 'inf_wordfree_every', 1))): return
    t0 = time.time()
    try:
        its = [x.get('s') for x in S_.get('it', []) if x.get('s') and not x.get('gone')][-int(getattr(cfg, 'inf_wordfree_n', 6000)):]
        D = []
        for s0 in its:
            if isinstance(s0, (bytes, bytearray)): s0 = bytes(s0).decode('latin-1', 'replace')   # the store keeps sentences as bytes (kra62e GPU visit 1)
            s0 = s0.encode('ascii', 'replace').decode('ascii'); toks = s0.split()
            if len(toks) < 3: continue
            st, p = [], 0
            for t in toks: st.append(p); p += len(t)
            if p > 220: continue
            g, i = [], 0                                                       # name stand-in: a run of capitalised words, not the sentence's first word alone
            while i < len(toks):
                if toks[i][:1].isupper() and i > 0:
                    j = i
                    while j + 1 < len(toks) and toks[j + 1][:1].isupper(): j += 1
                    g.append((st[i], st[j] + len(toks[j]))); i = j + 1
                else: i += 1
            D.append((''.join(toks).lower(), set(st[1:]), g))
        if len(D) < 200:
            print(f"  [WORD-FREE READOUT | visit {v_ + 1}] {len(D)} usable stored sentences: too few", flush=True); return
        T = [s for s, _, _ in D]; FR = _cands(T); BYTES = {w for w in FR if len(w) == 1}; EU = _edge(T)
        SPELL = float(getattr(cfg, 'inf_wordfree_spell', 4.0)); lex = dict(FR)
        for w, c in EU.items(): lex[w] = max(lex.get(w, 0), c)
        seg = None
        for it in range(IT):
            C, unk = _costs(lex); new = collections.Counter(); seg = []
            for s in T:
                cu, _ = _vit(s, C, unk); seg.append(cu); new.update(_segs(s, cu))
            C2, unk2 = _costs(new); keep = {}
            for w, c in new.items():
                if len(w) == 1 or w in EU or c * (_vit(w, C2, unk2, False)[1] - C2[w]) > SPELL * (len(w) + 1): keep[w] = c
            for b_ in BYTES: keep.setdefault(b_, 1)
            lex = keep
        rate = np.mean([len(c) / max(1, len(s) - 1) for s, c in zip(T, seg)]); rs = np.random.RandomState(0)
        rnd = [{i for i in range(1, len(s)) if rs.rand() < rate} for s in T]
        top = sorted((w for w in lex if len(w) >= 6), key=lambda w: -lex[w])[:12]
        print(f"  [WORD-FREE READOUT | visit {v_ + 1}] {len(D)} stored sentences, spaces removed, lower-cased; no word boundary given: {len(EU)} edge units, "
              f"lexicon {len(lex)} units (MDL, {SPELL:g} bits a byte to spell) || LEARNED CUT: {_score(D, seg)} || RANDOM CUT, same rate: {_score(D, rnd)} || "
              f"names = runs of capitalised words (evaluation only) | longest-used units: {', '.join(top)} ({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  [WORD-FREE READOUT | visit {v_ + 1}] unavailable ({type(e).__name__}: {str(e)[:160]})", flush=True)
