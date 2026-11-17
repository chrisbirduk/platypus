"""READER DIAGNOSIS AND FIXES (17 Nov 2026; tools/reader_diag.py) - offline. Real-text composition is limited by the READER: which
co-occurrences are facts (tools/frame_compose.py: every co-occurrence composes at 0.15 vs base 0.13; facts only, 0.45). Stages:
  R1 WHERE GOLD IS LOST (Re-DocRED): share of gold relations visible as a mention pair within GAP tokens in one sentence, anywhere in one
     sentence, and after resolving pronoun-led sentences to the document's title / the most recent name of a compatible type; with the
     precision of each candidate set (gold-related / candidate pairs).
  R2 CROSS-DOCUMENT REDUNDANCY (distant set): a pair's number of distinct documents with a co-occurrence (any frame / the same coarse
     frame); precision by the distant labels per level.
  R3 SELF-TRAINED READER: logistic regression over a record's frame (words between, 1-2 grams), type pair, gap and order, trained on
     PSEUDO-LABELS from the distant set's training documents: positive = the pair recurs in 3+ distinct documents with the same coarse
     frame, negative = the pair occurs in one document only (no labels used). UPPER BOUND: the same reader trained on the distant labels.
     Scored (a) on held-out distant documents by the distant labels, (b) on Re-DocRED gold pairs (another corpus: AUC, precision of the top
     5 / 10 / 25%), and (c) as the premise filter of composition: rules over coarse keys on the reader's top band, precision of composed
     pairs on Re-DocRED (tools/frame_compose.py's measure; base 0.132, every co-occurrence 0.155, gold facts 0.45).
Usage: python3 tools/reader_diag.py <distant.json.gz>   env RD_GAP (12) RD_TOP (0.25) RD_MAXREC (1500000)"""
import os, sys, json, math, time, types, collections, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
T0 = time.time(); P = lambda s: print(s, flush=True)
GAP = int(os.environ.get('RD_GAP', 12)); TOP = float(os.environ.get('RD_TOP', 0.25))
TR = json.load(open(SCR + '/bench_dl/train_revised.json')); DV = json.load(open(SCR + '/bench_dl/dev_revised.json')) + json.load(open(SCR + '/bench_dl/test_revised.json'))
PRON = {'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their', 'the'}

# ---------------- R1: where gold is lost (Re-DocRED)
def mentions(d):
    bys = collections.defaultdict(list)
    for e, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], e))
    return {s: sorted(v) for s, v in bys.items()}
def cand_sets(d):
    ms = mentions(d); ty = [vs[0]['type'] for vs in d['vertexSet']]; near, same = set(), set()
    for s, L in ms.items():
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                a, b = L[i][2], L[j][2]
                if a == b: continue
                same.add(frozenset((a, b)))
                if L[i][1] <= L[j][0] and L[j][0] - L[i][1] <= GAP: near.add(frozenset((a, b)))
    title = 0                                                                  # entity 0 is usually the title entity in Re-DocRED
    res = set(same)
    for s, toks in enumerate(d['sents']):
        if not toks or toks[0].lower() not in PRON: continue
        for (_, _, e) in ms.get(s, []):
            if e != title: res.add(frozenset((title, e)))
        prev = [x for x in ms.get(s - 1, [])]
        for (_, _, p) in prev:
            for (_, _, e) in ms.get(s, []):
                if p != e and ty[p] == ty[title]: res.add(frozenset((p, e)))
    return near, same, res
gold_n = 0; hit = collections.Counter(); ncand = collections.Counter(); rel = collections.Counter()
for d in DV:
    g = {frozenset((l['h'], l['t'])) for l in d['labels'] if l['h'] != l['t']}; gold_n += len(g)
    for nm, cs in zip(('within %d tokens' % GAP, 'same sentence', '+ pronoun-led sentences resolved'), cand_sets(d)):
        hit[nm] += len(g & cs); ncand[nm] += len(cs); rel[nm] += len(g & cs)
P(f"R1 WHERE GOLD IS LOST (Re-DocRED held-out {len(DV)} docs, {gold_n} gold related pairs):")
for nm in ('within %d tokens' % GAP, 'same sentence', '+ pronoun-led sentences resolved'):
    P(f"   {nm:34s} gold visible {hit[nm] / gold_n:.3f} | candidates {ncand[nm]}, precision {rel[nm] / max(1, ncand[nm]):.3f}")

# ---------------- R2 / R3 on the distant set
import platypus.inference.premises as pm
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, TY, LAB, ND = pm._PM['R'], pm._PM['TY'], pm._PM['LAB'], pm._PM['ND']
MAXR = int(os.environ.get('RD_MAXREC', 1500000))
if len(R) > MAXR: R = R[:MAXR]
cut = int(ND * 0.7)
def coarse(r): return (TY.get(r[1], 'X'), r[4], TY.get(r[2], 'X'))
docs_any = collections.defaultdict(set); docs_fr = collections.defaultdict(set)
for r in R:
    if r[0] >= cut: continue
    k = frozenset((r[1], r[2])); docs_any[k].add(r[0]); docs_fr[(k, coarse(r))].add(r[0])
P(f"R2 CROSS-DOCUMENT REDUNDANCY (distant set: {ND} docs, {len(R)} records, training docs < {cut}; {len(docs_any)} pairs)  ({time.time() - T0:.0f}s)")
for nm, D_ in (('any frame', {k: len(v) for k, v in docs_any.items()}), ('same coarse frame', None)):
    if D_ is None:
        D_ = collections.defaultdict(int)
        for (k, f), v in docs_fr.items(): D_[k] = max(D_[k], len(v))
    lv = collections.defaultdict(list)
    for k, n in D_.items(): lv['1' if n == 1 else '2' if n == 2 else '3-4' if n <= 4 else '5+'].append(k in LAB)
    P(f"   {nm:18s} " + " | ".join(f"{l}: {np.mean(lv[l]):.3f} ({len(lv[l])})" for l in ('1', '2', '3-4', '5+') if lv[l]))
# R3: the reader
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
FH = FeatureHasher(n_features=2 ** 18, input_type='string', alternate_sign=False)
def feats(mid_words, ta, tb, order):
    w = mid_words[-6:]; f = [f"t:{ta}|{tb}|{order}", f"g:{min(len(mid_words), 8)}", f"l:{w[-1] if w else ''}", f"f:{w[0] if w else ''}"]
    f += [f"w:{x}" for x in w] + [f"b:{a}_{b}" for a, b in zip(w, w[1:])] + [f"tl:{ta}|{w[-1] if w else ''}|{tb}"]
    return f
fr_max = collections.defaultdict(int)
for (k, f), v in docs_fr.items(): fr_max[k] = max(fr_max[k], len(v))
Xtr, ypl, ylab = [], [], []; rs = np.random.RandomState(0)
for r in R:
    if r[0] >= cut or r[5] > GAP: continue
    k = frozenset((r[1], r[2])); n = len(docs_any[k]); m = fr_max[k]
    pl = 1 if m >= 3 else (0 if n == 1 else None)
    if pl is None or (pl == 0 and rs.rand() > 0.25): continue                  # subsample the many singletons
    Xtr.append(feats(r[3].split(), TY.get(r[1], 'X'), TY.get(r[2], 'X'), 'f')); ypl.append(pl); ylab.append(int(k in LAB))
P(f"R3 SELF-TRAINED READER: {len(Xtr)} training records (pseudo-positive {np.mean(ypl):.3f}; their label rate: pseudo-pos {np.mean([l for l, p in zip(ylab, ypl) if p]):.3f}, "
  f"pseudo-neg {np.mean([l for l, p in zip(ylab, ypl) if not p]):.3f})  ({time.time() - T0:.0f}s)")
Xh = FH.transform(Xtr)
clf = LogisticRegression(max_iter=300, C=1.0).fit(Xh, ypl); up = LogisticRegression(max_iter=300, C=1.0).fit(Xh, ylab)
Xho, yho = [], []
for r in R:
    if r[0] < cut or r[5] > GAP: continue
    Xho.append(feats(r[3].split(), TY.get(r[1], 'X'), TY.get(r[2], 'X'), 'f')); yho.append(int(frozenset((r[1], r[2])) in LAB))
    if len(Xho) >= 200000: break
Hh = FH.transform(Xho); yho = np.array(yho)
def bands(s, y):
    o = np.argsort(-s); return " / ".join(f"{y[o[:max(1, int(len(y) * q))]].mean():.3f}" for q in (0.05, 0.10, 0.25))
for nm, m in (('SELF-TRAINED (pseudo-labels)', clf), ('UPPER BOUND (distant labels)', up)):
    s = m.predict_proba(Hh)[:, 1]
    P(f"   {nm:30s} held-out distant records ({len(yho)}, base {yho.mean():.3f}): AUC {roc_auc_score(yho, s):.3f}, top 5 / 10 / 25% {bands(s, yho)}")
# on Re-DocRED gold
recs = []
for di, d in enumerate(DV):
    g = {frozenset((l['h'], l['t'])) for l in d['labels']}; ty = [vs[0]['type'] for vs in d['vertexSet']]
    for s, L in mentions(d).items():
        toks = d['sents'][s]
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                (a0, a1, ea), (b0, b1, eb) = L[i], L[j]
                if ea == eb or a1 > b0 or b0 - a1 > GAP: continue
                recs.append((di, ea, eb, [w.lower() for w in toks[a1:b0]], ty[ea], ty[eb], int(frozenset((ea, eb)) in g)))
Hr = FH.transform([feats(x[3], x[4], x[5], 'f') for x in recs]); yr = np.array([x[6] for x in recs])
SC = {}
for nm, m in (('SELF-TRAINED', clf), ('UPPER BOUND', up)):
    s = m.predict_proba(Hr)[:, 1]; SC[nm] = s
    P(f"   {nm:30s} Re-DocRED mention pairs ({len(yr)}, base {yr.mean():.3f}): AUC {roc_auc_score(yr, s):.3f}, top 5 / 10 / 25% {bands(s, yr)}")
# (c) as the premise filter for composition (coarse keys = type pair + last word; rules mined on Re-DocRED TRAIN docs read the same way)
def doc_recs(docs):
    out = []
    for di, d in enumerate(docs):
        ty = [vs[0]['type'] for vs in d['vertexSet']]
        for s, L in mentions(d).items():
            toks = d['sents'][s]
            for i in range(len(L)):
                for j in range(i + 1, len(L)):
                    (a0, a1, ea), (b0, b1, eb) = L[i], L[j]
                    if ea == eb or a1 > b0 or b0 - a1 > GAP: continue
                    out.append((di, ea, eb, [w.lower() for w in toks[a1:b0]], ty[ea], ty[eb]))
    return out
def wlb(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d_ = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d_
trr = doc_recs(TR)
s_tr = {nm: m.predict_proba(FH.transform([feats(x[3], x[4], x[5], 'f') for x in trr]))[:, 1] for nm, m in (('SELF-TRAINED', clf), ('UPPER BOUND', up))}
def premises(rr, sc, thr):
    by = collections.defaultdict(lambda: collections.defaultdict(set))
    for x, v in zip(rr, sc):
        if v < thr: continue
        k = (x[4], x[3][-1] if x[3] else '', x[5]); by[x[0]][(x[1], x[2])].add(k); by[x[0]][(x[2], x[1])].add((x[5], x[3][-1] if x[3] else '', x[4], 'r'))
    return by
def chains(prem):
    byh = collections.defaultdict(list)
    for (a, b), ks in prem.items():
        for k in ks: byh[a].append((k, b))
    for (a, b), ks in prem.items():
        for k1 in ks:
            for k2, c in byh.get(b, ()):
                if c != a and c != b: yield a, k1, b, k2, c
# R4: pair-level redundancy from the distant corpus as the reader of Re-DocRED (names matched by their most common mention text)
nm_ = lambda d, e: collections.Counter(m['name'] for m in d['vertexSet'][e]).most_common(1)[0][0]
red = np.array([max(len(docs_any.get(frozenset((nm_(DV[x[0]], x[1]), nm_(DV[x[0]], x[2]))), ())), 0) for x in recs])
P(f"   R4 PAIR REDUNDANCY from the distant corpus on Re-DocRED pairs: pairs seen in 1+ distant docs {np.mean(red > 0):.3f}; gold rate by docs " + " | ".join(
  f"{l}: {yr[m].mean():.3f} ({int(m.sum())})" for l, m in (('0', red == 0), ('1', red == 1), ('2', red == 2), ('3+', red >= 3)) if m.sum()))
SC['REDUNDANCY'] = red + 1e-3 * SC['SELF-TRAINED']; s_tr['REDUNDANCY'] = np.array([len(docs_any.get(frozenset((nm_(TR[x[0]], x[1]), nm_(TR[x[0]], x[2]))), ())) for x in trr]) + 1e-3 * s_tr['SELF-TRAINED']
P(f"   COMPOSITION over the reader's premises (rules over coarse keys mined on {len(TR)} training docs; composed pair gold-related on {len(DV)} held-out docs):")
for nm in ('SELF-TRAINED', 'REDUNDANCY', 'UPPER BOUND'):
    for q in (0.10, 0.25, 0.50, 1.0):
        thr_tr = np.quantile(s_tr[nm], 1 - q) if q < 1 else -1; thr_dv = np.quantile(SC[nm], 1 - q) if q < 1 else -1
        PT = premises(trr, s_tr[nm], thr_tr); cnt = collections.Counter(); hit = collections.Counter()
        for di, prem in PT.items():
            seen = set()
            for a, k1, b, k2, c in chains(prem):
                if (a, k1, k2, c) in seen: continue
                seen.add((a, k1, k2, c)); cnt[(k1, k2)] += 1; hit[(k1, k2)] += bool(prem.get((a, c)) or prem.get((c, a)))
        rules = {k for k, n in cnt.items() if hit[k] >= 10 and wlb(hit[k], n) >= 0.5}
        PD = premises(recs, SC[nm], thr_dv); comp = rl = 0; prem_rel = []
        for di, prem in PD.items():
            g = {(l['h'], l['t']) for l in DV[di]['labels']} | {(l['t'], l['h']) for l in DV[di]['labels']}
            prem_rel += [((a, b) in g) for (a, b) in prem]
            out = {(a, c) for a, k1, b, k2, c in chains(prem) if (k1, k2) in rules and not ((a, c) in prem or (c, a) in prem)}
            comp += len(out); rl += sum(p in g for p in out)
        P(f"      {nm:13s} top {q:.0%}: premise precision {np.mean(prem_rel) if prem_rel else float('nan'):.3f} | rules {len(rules)} | composed {comp}, precision {rl / max(1, comp):.3f} (base 0.132)  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
