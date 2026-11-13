"""DOCUMENT CONTEXT WITHOUT LABELS (13 Nov 2026; tools/doc_topics.py) - offline, no effect on the package.
A slow, document-level context for keyed compression (tools/record_mdl.py RM_CTX): every document of Re-DocRED (train + dev, in that order)
and of DocRED's distant set is a bag of its lower-cased words (hashed, tf-idf), reduced by SVD (64) and clustered (k-means, k = 8 and 32).
No labels. Written to DT_OUT as {'redoc_<k>': topic per Re-DocRED document, 'dist_<k>': topic per distant document}.
Usage: python3 tools/doc_topics.py <train_distant.json.gz>   env DT_OUT (default <scratchpad>/distant/topics.npz)"""
import os
import sys
import gzip
import json
import time
import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
P = lambda *a: print(*a, flush=True); T0 = time.time()
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
OUT = os.environ.get('DT_OUT', SCR + '/distant/topics.npz')
HV = HashingVectorizer(n_features=2 ** 18, alternate_sign=False, norm=None, token_pattern=r"[a-z][a-z]+", lowercase=True)
txt = lambda d: ' '.join(' '.join(s) for s in d['sents'])
R = json.load(open(SCR + '/bench_dl/train_revised.json')) + json.load(open(SCR + '/bench_dl/dev_revised.json'))
XR = HV.transform([txt(d) for d in R]); nR = len(R); del R
D = json.load(gzip.open(sys.argv[1], 'rt')); nD = len(D); parts = []
for i in range(0, nD, 10000):
    parts.append(HV.transform([txt(D[j]) for j in range(i, min(nD, i + 10000))]))
    for j in range(i, min(nD, i + 10000)): D[j] = None
del D
X = vstack([XR] + parts).tocsr(); del parts; P(f"documents: Re-DocRED {nR}, distant {nD} ({time.time() - T0:.0f}s)")
X = TfidfTransformer(sublinear_tf=True).fit_transform(X); Z = normalize(TruncatedSVD(64, random_state=0).fit_transform(X)); del X
out = {}
for k in (8, 32):
    lab = MiniBatchKMeans(k, random_state=0, batch_size=4096, n_init=3).fit_predict(Z); out[f'redoc_{k}'] = lab[:nR]; out[f'dist_{k}'] = lab[nR:]
    P(f"k {k}: document shares " + ' '.join(f"{c:.3f}" for c in np.sort(np.bincount(lab, minlength=k) / len(lab))[::-1][:10]) + f" ... ({time.time() - T0:.0f}s)")
np.savez(OUT, **out); P(f"written {OUT} ({time.time() - T0:.0f}s)")
