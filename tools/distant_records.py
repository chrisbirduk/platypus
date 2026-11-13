"""STORE RECORDS FROM THE DISTANT DocRED TRAINING SET (13 Nov 2026; tools/distant_records.py) - offline, no effect on the package.
DocRED's distantly supervised training set (101,873 Wikipedia documents, entity mentions with automatic types; no overlap with Re-DocRED's
documents) read into store records exactly as tools/consistency_teacher.py does (two names in one sentence, at most 12 words apart, the span
at most 60 characters): (document, name A, name B, words between, last word between, number of words between). Also each name's mention
types (automatic; evaluation only). Relation labels are not kept. Written to DR_OUT for tools/record_mdl.py RM_EXTRA.
Usage: python3 tools/distant_records.py <train_distant.json.gz>   env DR_OUT (default <scratchpad>/distant/records.pkl)"""
import os
import sys
import gzip
import json
import time
import pickle
import collections
P = lambda *a: print(*a, flush=True); T0 = time.time()
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
OUT = os.environ.get('DR_OUT', SCR + '/distant/records.pkl')
D = json.load(gzip.open(sys.argv[1], 'rt')); P(f"documents {len(D)} ({time.time() - T0:.0f}s)")
REC = []; TY = collections.defaultdict(collections.Counter)
for di in range(len(D)):
    d = D[di]; D[di] = None
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei)); TY[cn[ei]][m.get('type', 'X')] += 1
    for si, ms in bys.items():
        ms = sorted(ms); toks = d['sents'][si]
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                (a0, a1), (b0, b1) = ms[x][:2], ms[y][:2]; ea, eb = ms[x][2], ms[y][2]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb]: continue
                if b0 - a1 <= 12 and len(' '.join(toks[a0:b1])) <= 60:
                    mid = toks[a1:b0]; REC.append((di, cn[ea], cn[eb], ' '.join(mid).lower(), mid[-1].lower() if mid else '', len(mid)))
    if di % 20000 == 0: P(f"   {di} documents, {len(REC)} records ({time.time() - T0:.0f}s)")
del D
TY = {n: c.most_common(1)[0][0] for n, c in TY.items()}
pickle.dump(dict(rec=REC, types=TY), open(OUT, 'wb'), protocol=4)
P(f"records {len(REC)}, names {len(TY)}; written {OUT} ({time.time() - T0:.0f}s)")
