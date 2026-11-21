"""THE SEEDED SELECTOR ON THE PACKAGE'S OWN CANDIDATES (20 Nov 2026; tools/rcode_select_pkg.py) - offline, the GPU code path.
tools/rcode_select.py compared rankings on its own candidate set (record pairs, 128-dim codes from every record) and found kra14's rule far
weaker there (top 100 0.17-0.27) than the package reports (0.72), so the comparison is redone on the package's candidates: premises._rcode_step
as kra14 runs it (512-dim codes, the graph from the SEEDED reader's top half - inf_rcode_reader 0.5, every name pair co-occurring in a held-out
document, distant labels for evaluation), its write-time features saved (save_cand). Candidates are split in half by a hash of the pair:
half A teaches the selector (its labels: the seed; RP_N of them, 0 = all), half B is scored. On half B: kra14's rule (code + Adamic-Adar rank
sum among |log degree ratio| > 1), the code, and the selector (logistic regression / gradient-boosted trees on the code, Adamic-Adar,
resource allocation, both degrees and their ratio), alone and under the same asymmetry filter: AUC and precision of the top 100 / 200 /
400 / 600 (kra14 writes 600 a readout).
Usage: python3 tools/rcode_select_pkg.py <distant_50k.json.gz>   env RP_FRAC (0.4: documents read), RP_N (0)"""
import os, sys, types, time, zlib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time()
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa', inf_rcode_asym=1.0,
                            inf_rcode_selreport=1, inf_rti_max=600, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rcode_reader=0.5, inf_premises_seed_ctx=1)
pm._load_distant(cfg); ND = pm._PM['ND']; CX = pm._PM.get('CTX') or {}
RD = [r for r in pm._PM['R'] if r[0] < int(round(ND * float(os.environ.get('RP_FRAC', 0.4))))]; del pm._PM['R']
y = np.array([bool(r[6]) for r in RD]); hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(int(r[5]), 12) // 3} " + ' '.join('C_' + w for w in CX.get(id(r), '').split()) for r in RD])
sd = np.random.RandomState(4 + 707).permutation(len(RD))[:20000]                # the SEEDED READER as kra14 (20,000 labelled records, context on)
c = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); c.fit(X[sd], y[sd]); pm._PM['rdscore'] = c.decision_function(X); del X
P(f"{len(RD)} records read ({ND} documents in the set); seeded reader on 20000 records  ({time.time() - T0:.0f}s)")
pm._PM['rti'] = {}; pm._PM['save_cand'] = True; pm._rcode_step(None, cfg, lambda d: 0, [1], RD, 3); C = pm._PM['cand']
pr = C['pairs']; yy = np.asarray(C['y'], bool); code, aa, ra, dA, dB, rule = (np.asarray(C[k], float) for k in ('code', 'aa', 'ra', 'dA', 'dB', 'rule'))
asy = np.abs(np.log((dA + 1) / (dB + 1))) > 1.0
F = np.c_[code, aa, ra, np.log1p(dA), np.log1p(dB), np.log1p(np.minimum(dA, dB)), np.log1p(dA + dB), np.abs(np.log((dA + 1) / (dB + 1)))]
half = np.array([zlib.crc32(('|'.join(p)).encode('utf-8', 'replace')) % 2 for p in pr]); A, B = half == 0, half == 1
ia = np.where(A)[0]; NR = int(os.environ.get('RP_N', 0))
if NR: ia = ia[np.random.RandomState(5).permutation(len(ia))[:NR]]
P(f"package candidates {len(pr)} (labelled {yy.mean():.3f}; asymmetric {asy.mean():.2f}, labelled {yy[asy].mean():.3f}); selector taught on {len(ia)} of half A "
  f"(labelled {yy[ia].mean():.3f}), scored on half B ({int(B.sum())})  ({time.time() - T0:.0f}s)")
sc = StandardScaler().fit(F[ia]); lr = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(sc.transform(F[ia]), yy[ia])
gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(F[ia], yy[ia])
lv, gv = lr.decision_function(sc.transform(F)), gb.predict_proba(F)[:, 1]
S = {'kra14 rule (code + AA, asym > 1)': rule, 'code': code, 'SELECTOR, logistic': lv, 'SELECTOR, trees': gv,
     'SELECTOR, logistic, asym > 1': np.where(asy, lv, -1e9), 'SELECTOR, trees, asym > 1': np.where(asy, gv, -9.0)}
yb = yy[B]
for nm, s in S.items():
    sb = np.nan_to_num(s[B], neginf=-1e12); o = np.argsort(-sb, kind='stable')
    P(f"   {nm:34s} AUC {roc_auc_score(yb, sb):.3f} | top 100 / 200 / 400 / 600 " + " / ".join(f"{yb[o[:n]].mean():.3f}" for n in (100, 200, 400, 600)))
P("   logistic weights (standardised): " + ", ".join(f"{a} {b:+.2f}" for a, b in zip(['code', 'AA', 'RA', 'log dA', 'log dB', 'log deg min', 'log deg sum', 'deg ratio'], lr.coef_[0])))
P(f"DONE ({time.time() - T0:.0f}s)")
