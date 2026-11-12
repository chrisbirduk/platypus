"""A random sample of the fast channel's records from the benchmark sentences (adjacent and document-centre records), each with its
sentence, for judging by hand whether it states a real relation between the two fillers. Usage: python3 tools/precision_sample.py [n] [seed]"""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'reader_tests.py')).read()
exec(compile(_src[:_src.index('# ---------------- T1 REDUNDANCY')], 'reader_tests', 'exec'))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 100; rs = np.random.RandomState(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
recs = [r for r in BR if r[4] in (0, 1)]
for i in rs.choice(len(recs), n, replace=False):
    a, f, b, (_, di, si), k = recs[i]; g = gp(a, b)
    print(f"#{i} | {' '.join(a)} || {f} || {' '.join(b)} | {'centre' if k == 0 else 'adjacent'} | gold {('yes' if g else 'no') if g is not None else 'n/a'}\n    {' '.join(bdocs[di][si])[:300]}")
