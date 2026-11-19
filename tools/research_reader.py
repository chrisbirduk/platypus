"""TWO READER FIXES FROM THE LITERATURE (19 Nov 2026; tools/research_reader.py) - offline, distant set, LABEL-FREE (own-document labels for
evaluation only).
A. MULTI-INSTANCE ('at least one') TRAINING (Riedel et al. 2010; Hoffmann et al. 2011; Surdeanu et al. 2012): usefulness labels a PAIR, not
   a sentence, so every comma-listed mention of a related pair became a positive. Here a positive pair is a bag of mentions of which only
   the ONE the current reader scores highest is taken as a positive (hard EM, MA_ITERS rounds); negative pairs keep all their mentions.
B. RELATION-PHRASE CONSTRAINTS (ReVerb, Fader et al. 2011): a relation phrase is a verb, or a verb plus a preposition, or a verb plus
   nouns / adjectives ending in a preposition (spaCy part-of-speech tags on the words between the names), AND it must occur with at least
   MA_PAIRS distinct name pairs (the lexical constraint). Reported: the precision of records whose words between satisfy both, against
   other records of the same length; then used as a teacher: such records join the usefulness positives (unless usefulness says -1).
Arms (trained on the first 80% of documents, scored on the rest): USEFUL (now) | MULTI-INSTANCE | REVERB-TAUGHT | MULTI-INSTANCE + REVERB.
Reported: record AUC, top 10% precision, PHRASE recall (2-12 words, not a list) in the top 10%, and pair-level AUC (noisy-or).
Usage: python3 tools/research_reader.py <distant_50k.json.gz>   env MA_DOCS (30000), MA_ITERS (2), MA_PAIRS (20)"""
import os, sys, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=1))
ND = min(pm._PM['ND'], int(os.environ.get('MA_DOCS', 30000))); R = [r for r in pm._PM['R'] if r[0] < ND]
y = np.array([bool(r[6]) for r in R]); doc = np.array([r[0] for r in R]); te = doc >= int(ND * 0.8); tr = ~te
key = [frozenset((r[1], r[2])) for r in R]; dist = np.array([r[5] for r in R]); lst = np.array([pm._rret_listlike(r) for r in R])
phrase = (dist >= 2) & ~lst & np.array(['(' not in r[3] for r in R])
pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for i in np.where(tr)[0]: r = R[i]; pd[key[i]].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0])
def u_(i):
    r = R[i]; a_ = len(pd[key[i]] - {r[0]}); nA, nB = len(nd[r[1]] - {r[0]}), len(nd[r[2]] - {r[0]})
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); return max(hA * np.log(ND / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(ND / (nA + 2)) - (1 - hB) * 0.5)
U = np.full(len(R), np.nan); I = np.where(tr)[0]; U[I] = [u_(i) for i in I]; ok = tr & ~np.isnan(U); hi, lo = np.quantile(U[ok], 0.8), np.quantile(U[ok], 0.5)
ps, ng = np.where(ok & (U >= hi))[0], np.where(ok & (U <= lo))[0]
P(f"{ND} documents, {len(R)} records (phrase records {phrase.mean():.2f}, related {y[phrase].mean():.3f}; others {y[~phrase].mean():.3f}); usefulness +1 {len(ps)} (right {y[ps].mean():.3f}), -1 {len(ng)}  ({time.time() - T0:.0f}s)")
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3}" for r in R])
def fit(pos, neg):
    c = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); c.fit(X[np.r_[pos, neg]], np.r_[np.ones(len(pos)), np.zeros(len(neg))]); return c.decision_function(X)
# ---- B. ReVerb constraints on the words between
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])
mids = sorted({r[3] for r, p in zip(R, phrase) if p}); tags = {}
for m, dd in zip(mids, nlp.pipe(mids, batch_size=2048)): tags[m] = [t.pos_ for t in dd]
def reverb_syntax(tg):
    if not tg or tg[0] not in ('VERB', 'AUX'): return False                    # V | V P | V W* P   (W = noun, adjective, adverb, pronoun, determiner)
    if len(tg) == 1: return True
    if tg[-1] == 'ADP' and all(t in ('NOUN', 'ADJ', 'ADV', 'PRON', 'DET', 'VERB', 'AUX', 'PROPN', 'PART') for t in tg[1:-1]): return True
    return len(tg) == 2 and tg[1] in ('ADP', 'PART')
pairs_by_mid = collections.defaultdict(set)
for i in np.where(tr & phrase)[0]: pairs_by_mid[R[i][3]].add(key[i])
KP = int(os.environ.get('MA_PAIRS', 20))
rv = np.array([bool(p) and reverb_syntax(tags.get(r[3], [])) and len(pairs_by_mid.get(r[3], ())) >= KP for r, p in zip(R, phrase)])
syn = np.array([bool(p) and reverb_syntax(tags.get(r[3], [])) for r, p in zip(R, phrase)])
P(f"B. REVERB on {len(mids)} distinct phrases ({time.time() - T0:.0f}s): phrase records passing the syntactic constraint {syn[phrase].mean():.3f} (related {y[syn].mean():.3f}), "
  f"and the lexical one (>= {KP} distinct pairs) {rv[phrase].mean():.3f} (related {y[rv].mean():.3f}); other phrase records related {y[phrase & ~rv].mean():.3f}")
top_rv = collections.Counter(r[3] for r, v in zip(R, rv) if v).most_common(15); P("   commonest passing phrases: " + ", ".join(f"'{m}' {n}" for m, n in top_rv))
rvpos = np.where(tr & rv & ~np.isin(np.arange(len(R)), ng))[0]
# ---- A. multi-instance
def multi_instance(pos0, neg, iters=int(os.environ.get('MA_ITERS', 2))):
    s = fit(pos0, neg); bags = collections.defaultdict(list)
    for i in pos0: bags[key[i]].append(i)
    for _ in range(iters):
        sel = np.array([max(b, key=lambda j: s[j]) for b in bags.values()]); s = fit(sel, neg)
    return s, sel
def rep(nm, s):
    m = te; o = np.argsort(-s[m]); top = np.zeros(m.sum(), bool); top[o[:m.sum() // 10]] = True; yy = y[m]; phm = phrase[m]
    g = collections.defaultdict(list)
    for i in np.where(m)[0]: g[key[i]].append(i)
    ks = list(g); yp = np.array([any(y[i] for i in g[k]) for k in ks]); sp = np.array([1 - np.prod(1 - 1 / (1 + np.exp(-s[g[k]]))) for k in ks])
    P(f"   {nm:30s} record AUC {roc_auc_score(yy, s[m]):.3f} | top 10% {yy[top].mean():.3f} | phrase recall {top[yy & phm].mean():.3f} | phrase-record AUC {roc_auc_score(yy[phm], s[m][phm]):.3f} | pair AUC (noisy-or) {roc_auc_score(yp, sp):.3f}")
P("ARMS (held-out documents):")
sU = fit(ps, ng); rep('USEFUL (now)', sU)
sM, sel = multi_instance(ps, ng); P(f"   (multi-instance kept {len(sel)} mentions of {len(ps)}: right {y[sel].mean():.3f} vs {y[ps].mean():.3f}; phrase share {phrase[sel].mean():.2f} vs {phrase[ps].mean():.2f})"); rep('MULTI-INSTANCE', sM)
sR = fit(np.unique(np.r_[ps, rvpos]), ng); P(f"   (ReVerb positives added {len(rvpos)}, right {y[rvpos].mean():.3f})"); rep('REVERB-TAUGHT', sR)
sMR, sel2 = multi_instance(np.unique(np.r_[ps, rvpos]), ng); rep('MULTI-INSTANCE + REVERB', sMR)
P(f"DONE ({time.time() - T0:.0f}s)")
