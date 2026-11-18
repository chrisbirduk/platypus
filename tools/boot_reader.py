"""BOOTSTRAPPED READER (18 Nov 2026; tools/boot_reader.py) - offline, distant set, LABEL-FREE (distant labels for evaluation only).
The reader's records are mention pairs within 12 tokens of one sentence; label-free it ranks them at AUC 0.62 (reader_diag). Snowball /
DIPRE-style weak supervision without labels: a few GENERIC, high-precision connective templates mark seed records ('X , born in Y',
'X ( born ... Y', 'X is located in Y', 'X , the capital of Y', 'X was founded in Y', 'X is a member of Y' ...); negatives are pairs joined only
by list glue ('X , Y and Z' style: the text between is ',' / 'and' / 'or') or far apart with no word between them that is not glue. A logistic
classifier over features of the text BETWEEN the names (words, last word, first word, length), the type pair and the distance learns from
the seeds; each round its most confident records join the positives (and its least confident the negatives), BS_ROUNDS rounds.
Measured: AUC and precision of the top records by the distant labels (a pair is right if any Wikidata relation links it in its document),
against distance alone and the last-word heuristic, on held-out documents.
Usage: python3 tools/boot_reader.py <distant.json.gz>   env BS_ROUNDS (3) BS_ADD (2000)"""
import os, sys, time, types, collections, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, TY, LAB, ND = pm._PM['R'], pm._PM['TY'], pm._PM['LAB'], pm._PM['ND']
R = [r for r in R if r[5] <= 12 and r[1] != r[2]]
GLUE = {',', 'and', 'or', 'the', 'a', 'an', '(', ')', '-', '"', "'", ';', ':'}
SEEDS = [r'^, born in$', r'^\( born .* in$', r'^was born in$', r'^is located in$', r'^, the capital of$', r'^is the capital of$', r'^was founded in$',
         r'^is a member of$', r'^is a city in$', r'^is a town in$', r'^is a village in$', r'^, a city in$', r'^plays for$', r'^played for$', r'^was educated at$',
         r'^studied at$', r'^is headquartered in$', r'^, located in$', r'^died in$', r'^, who was born in$', r'^is a river in$', r'^is an album by$', r'^is a song by$']
SRX = [re.compile(x) for x in SEEDS]
def feats(r):
    w = r[3].lower().split(); f = ['ty=' + TY.get(r[1], 'X') + '>' + TY.get(r[2], 'X'), 'd=' + str(min(r[5], 12)), 'n=' + str(min(len(w), 8))]
    f += ['w=' + x for x in set(w)] + (['last=' + w[-1], 'first=' + w[0], 'bi=' + ' '.join(w[-2:])] if w else ['empty'])
    f += ['ty_last=' + TY.get(r[1], 'X') + '>' + TY.get(r[2], 'X') + ':' + (w[-1] if w else '')]
    return f
vocab = {}; rows, cols = [], []
for i, r in enumerate(R):
    for f in feats(r): rows.append(i); cols.append(vocab.setdefault(f, len(vocab)))
X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(R), len(vocab)))
y = np.array([frozenset((r[1], r[2])) in LAB for r in R], dtype=float)
doc = np.array([r[0] for r in R]); test = doc >= int(ND * 0.8); train = ~test
txt = [' '.join(r[3].lower().split()) for r in R]
seed = np.array([any(rx.search(t) for rx in SRX) for t in txt]); neg0 = np.array([bool(t.split()) and all(x in GLUE for x in t.split()) for t in txt]) | np.array([r[5] >= 9 and all(x in GLUE for x in t.split()) for r, t in zip(R, txt)])
same = np.array([TY.get(r[1], 'X') == TY.get(r[2], 'X') for r in R])          # BS_NEG sametype: list glue only between names of the SAME type ('A , B and C' lists)
neg = neg0 & same if os.environ.get('BS_NEG', '') == 'sametype' else neg0
P(f"{ND} documents, {len(R)} records ({train.sum()} training, {test.sum()} held out); {len(vocab)} features | seeds: {(seed & train).sum()} positives "
  f"(labelled {y[seed & train].mean():.3f}), {(neg & train).sum()} negatives (labelled {y[neg & train].mean():.3f}) | base rate {y[test].mean():.3f}  ({time.time() - T0:.0f}s)")
def report(nm, s):
    st = s[test]; yt = y[test]; o = np.argsort(-st)
    P(f"   {nm:44s} AUC {roc_auc_score(yt, st):.3f} | precision top 1% {yt[o[:len(o) // 100]].mean():.3f}, top 5% {yt[o[:len(o) // 20]].mean():.3f}, top 20% {yt[o[:len(o) // 5]].mean():.3f}")
report('distance only (nearer = better)', -np.array([r[5] for r in R], dtype=float))
rel = np.array([float(bool(t.split()) and t.split()[-1].isalpha() and t.split()[-1] not in GLUE) for t in txt]) - 0.01 * np.array([r[5] for r in R])
report('last word is a relational word', rel)
pos, ng = seed & train, neg & train
for rd in range(int(os.environ.get('BS_ROUNDS', 3)) + 1):
    idx = np.where(pos | ng)[0]; lab = pos[idx].astype(int)
    clf = LogisticRegression(C=0.5, max_iter=300, class_weight='balanced').fit(X[idx], lab); s = clf.decision_function(X)
    report(f'BOOTSTRAPPED round {rd} ({pos.sum()} pos / {ng.sum()} neg)', s)
    tr = np.where(train & ~pos & ~ng)[0]; o = tr[np.argsort(-s[tr])]; k = int(os.environ.get('BS_ADD', 2000))
    P(f"      added this round: top {k} labelled {y[o[:k]].mean():.3f}, bottom {k} labelled {y[o[-k:]].mean():.3f}")
    pos = pos.copy(); pos[o[:k]] = True; ng = ng.copy(); ng[o[-k:]] = True
P(f"DONE ({time.time() - T0:.0f}s)")
