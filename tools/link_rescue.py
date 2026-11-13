"""CAN COMPRESSION LINKING BE RESCUED? (13 Nov 2026; tools/link_rescue.py) - offline, no effect on the package.
On the GPU (cg27a_link) compression links at a fixed 3 bits were 0 of 6 on benchmark entities while letter links were 201 of 201: compression
ranks same-entity pairs well (AUC 0.74-0.84 in the model) but same-entity pairs are rare among candidates, so a fixed threshold is imprecise.
Two rescues, scored on the Re-DocRED dev mentions with the stand-in trunk of tools/trunk_guides.py (weaker than the model's trunk: its link
AUC was 0.56 - compare the rows with each other, not with the GPU):
  BASELINE   compression links on pairs letters do not link, at the RATE the GPU linked (1.5% of candidates) - the fixed-threshold policy
  RESCUE 1   CALIBRATED ON LETTER LINKS: letter-linked pairs are free positives (201 of 201 right on the GPU); a compression link needs a gain
             at least the median (or the 75th percentile) of the letter-linked pairs' gains - the system sets its own threshold from what it knows
  RESCUE 2   NARROWED CANDIDATES: each later mention links at most ONE earlier mention - its best by gain, by a margin over the runner-up -
             and only capitalised names (proper names; the store's mentions)
  BOTH       1 and 2 together
Each row: links made, precision (same entity), how many of them are HARD (no shared word - what letters can never find) and right.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/link_rescue.py [lm steps] [documents]"""
import os
import re
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_tg = open(os.path.join(HERE, 'trunk_guides.py')).read()
exec(compile(_tg[:_tg.index('# ---------------------------------------------------------------- T1 boundaries')], 'trunk_guides', 'exec'))
_cg = open(os.path.join(HERE, 'compress_gain.py')).read()
exec(compile(_cg[_cg.index("TH = b'Thing'"):_cg.index('# ---------------------------------------------------------------- (a) units')], 'compress_gain', 'exec'))
def toks_of(s): return [w.lower() for w in re.findall(r"[A-Za-z0-9]+", s)]
def c3(s):
    s = re.sub(r'[^a-z0-9]', '', s.lower()); return {s[i:i + 3] for i in range(max(0, len(s) - 2))}
def initials(s): return ''.join(w[0] for w in re.findall(r"[A-Za-z]+", s) if w[0].isupper()).lower()
P(f"\n================ LINK RESCUE  ({time.time() - T0:.0f}s)")
_b = _cg[_cg.index('# ---------------------------------------------------------------- (b) links'):_cg.index('def gain_b(m):')]
exec(compile(_b.replace('P(f"(b) LINKS', 'P(f"LINKS'), 'compress_gain_b', 'exec'))
NAMES = []
for D in DOCS:
    for j, mj in enumerate(D['ments']):
        for i in [i for i in range(j) if D['ments'][i]['b'] <= mj['a']][-12:]: NAMES.append((D['ments'][i]['name'], mj['name'], D['ments'][i]['sid'] == mj['sid']))
assert len(NAMES) == len(PY)


def letter(a, b):
    """the store's letter rule (tools: store_link): one name is the tail of the other, or an all-caps name is the other's initials"""
    ta, tb = toks_of(a), toks_of(b)
    if not ta or not tb: return False
    s, l = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(s) < len(l) and l[-len(s):] == s and (a if len(ta) <= len(tb) else b)[:1].isupper(): return True
    for x, y in ((a, b), (b, a)):
        if re.fullmatch(r'[A-Z]{2,}', x.strip()) and initials(y) == x.strip().lower(): return True
    return False


LET = np.array([letter(a, b) for a, b, _ in NAMES]); CAP = np.array([a[:1].isupper() and b[:1].isupper() for a, b, _ in NAMES])
SAME = np.array([s for _, _, s in NAMES]); HARD = PF[:, 5] == 1
LETC = LET & CAP & ~SAME                                                      # calibration positives: capitalised letter links across sentences
def gain_b(m):
    bw = bits(m, [s_ for s_, p_ in PW], [(p_, len(s_)) for s_, p_ in PW]); bo = bits(m, [s_ for s_, p_ in PO], [(p_, len(s_)) for s_, p_ in PO]); return bo - bw
t1 = time.time(); gb = gain_b(lm)
P(f"gains computed ({time.time() - t1:.0f}s) | candidate pairs {len(PY)}, same entity {PY.mean():.3f} | LETTER links {LET.sum()}, same entity "
  f"{PY[LET].mean():.3f} | pairs letters do not link {(~LET).sum()}, same entity among them {PY[~LET].mean():.3f} (HARD same-entity {int((PY & HARD & ~LET).sum())})")
P(f"   calibration positives (capitalised letter links across sentences) {LETC.sum()}, same entity {PY[LETC].mean():.3f}")
P(f"   gain (bits): letter-linked pairs median {np.median(gb[LET]):.2f}, 75th pct {np.percentile(gb[LET], 75):.2f} | other same-entity pairs median "
  f"{np.median(gb[PY & ~LET]):.2f} | different-entity pairs median {np.median(gb[~PY]):.2f}, 98.5th pct {np.percentile(gb[~PY & ~LET], 98.5):.2f}")
best = {}; second = {}
for k in range(len(PY)):
    if LET[k]: continue
    key = PK[k]
    if key not in best or gb[k] > gb[best[key]]: second[key] = gb[best[key]] if key in best else -1e9; best[key] = k
    elif gb[k] > second.get(key, -1e9): second[key] = gb[k]
ISBEST = np.zeros(len(PY), bool); MARG = np.full(len(PY), -1e9)
for key, k in best.items(): ISBEST[k] = True; MARG[k] = gb[k] - second.get(key, -1e9)


def row(tag, sel):
    n = int(sel.sum()); P(f"   {tag:62s} links {n:6d} | same entity {PY[sel].mean() if n else float('nan'):.3f} | right {int((PY & sel).sum()):5d} "
                          f"| HARD right {int((PY & sel & HARD).sum()):4d} of {int((sel & HARD).sum()):5d}")


cand = ~LET
P("Compression links on pairs letters do not link (the letter links themselves: " + f"{LET.sum()} at {PY[LET].mean():.3f}):")
row("chance (every candidate pair)", cand)
thr_rate = np.percentile(gb[cand], 98.5); row(f"BASELINE: top 1.5% by gain (the GPU link rate; {thr_rate:.2f} bits)", cand & (gb >= thr_rate))
for q in (50, 75, 90):
    th = np.percentile(gb[LETC], q); row(f"RESCUE 1: gain >= letter links' {q}th pct ({th:.2f} bits)", cand & (gb >= th))
for m in (0.0, 1.0, 2.0):
    row(f"RESCUE 2: best candidate, margin >= {m:.0f} bits, capitalised", cand & ISBEST & (MARG >= m) & CAP)
    row(f"RESCUE 2 + top 1.5%: margin >= {m:.0f}", cand & ISBEST & (MARG >= m) & CAP & (gb >= thr_rate))
for q in (50, 75):
    th = np.percentile(gb[LETC], q)
    for m in (0.0, 1.0, 2.0): row(f"BOTH: letters' {q}th pct + best, margin >= {m:.0f}, capitalised", cand & ISBEST & (MARG >= m) & CAP & (gb >= th))
P(f"DONE ({time.time() - T0:.0f}s)")
