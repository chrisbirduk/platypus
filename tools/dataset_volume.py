"""DOES MORE REAL TEXT UNLOCK RULES? (13 Nov 2026; tools/dataset_volume.py) - offline, no effect on the package.
The relational module's miner (_rm_mine, every guard) on GOLD facts of Re-DocRED (train 3,053 + dev 500 documents, local copy), by volume:
the ceiling where reading and relation identity are both solved, so only the AMOUNT of text varies. Earlier (realtext_mine.py ORACLE,
262 documents through the reader): 0 rules. Per volume: 20% of the facts are held out at random; rules found (r1 o r2 -> r3, functional
accuracy, n); proposals checked against ALL gold facts on pairs Re-DocRED annotates (both names in one document: exhaustive there), plus the
held-out facts recovered. Names are entity names (the commonest mention), so one name across documents is one entity - as the store sees it.
Usage: python3 tools/dataset_volume.py [volumes, default 262,800,1600,3553]   env DV_DIR (folder with train_revised.json, dev_revised.json)"""
import os
import sys
import json
import time
import random
import collections
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from platypus.inference.distill import _rm_mine  # noqa: E402
torch.set_num_threads(int(os.environ.get('DV_THREADS', 1)))
P = lambda *a: print(*a, flush=True); T0 = time.time()
DIR = os.environ.get('DV_DIR', '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad/bench_dl')
docs = json.load(open(DIR + '/train_revised.json')) + json.load(open(DIR + '/dev_revised.json'))
REL = dict(x.split(' ', 1) for x in 'P17 country|P131 located in|P27 citizenship|P150 contains|P569 born on|P570 died on|P577 published|P175 performer|P495 country of origin|P527 has part|P361 part of|P19 birthplace|P20 deathplace|P36 capital|P1376 capital of|P54 team|P30 continent|P123 publisher|P264 label|P740 formed in|P159 headquarters|P276 location|P50 author|P57 director|P161 cast|P86 composer|P170 creator|P178 developer|P102 party|P108 employer|P69 educated at|P26 spouse|P22 father|P25 mother|P40 child|P3373 sibling|P571 founded|P580 start|P582 end|P585 point in time|P206 by water|P205 basin country|P403 mouth|P710 participant|P1441 in work|P674 characters|P400 platform|P449 network|P6 head of govt|P35 head of state|P37 language|P463 member of|P488 chair|P118 league|P127 owned by|P749 parent org|P355 subsidiary|P166 award|P800 notable work|P1001 jurisdiction|P194 legislature|P162 producer|P272 production co|P39 position|P140 religion|P1412 speaks|P551 residence|P607 conflict|P364 orig language|P1344 took part in|P241 branch|P1336 claimed by|P1198 unemployment|P155 follows|P156 followed by|P676 lyrics by|P58 screenwriter|P179 series|P137 operator|P112 founded by|P1056 product|P937 work location|P286 head coach|P121 item operated|P39 position held'.split('|'))
rn = lambda c: f"{c} {REL.get(c, '')}".strip()
random.Random(0).shuffle(docs)
VOL = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '262,800,1600,3553').split(',')]


def doc_facts(d):
    nm = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    F = {(nm[l['h']], l['r'], nm[l['t']]) for l in d['labels'] if nm[l['h']] != nm[l['t']]}
    A = {(a, b) for a in set(nm) for b in set(nm) if a != b}
    return F, A


ALLF, ALLA = set(), set()
per = [doc_facts(d) for d in docs]
for F, A in per: ALLF |= F; ALLA |= A
P(f"documents {len(docs)} | gold facts {len(ALLF)} | relation types {len({r for _, r, _ in ALLF})} | names {len({x for a, _, b in ALLF for x in (a, b)})} "
  f"| commonest {[(rn(r), n) for r, n in collections.Counter(r for _, r, _ in ALLF).most_common(6)]}")
for V in VOL:
    t1 = time.time(); F = set()
    for f, _ in per[:V]: F |= f
    F = sorted(F); rng = random.Random(1); held = set(rng.sample(F, len(F) // 5)); obs = [x for x in F if x not in held]
    pairs = collections.defaultdict(set)
    for a, r, b in obs: pairs[r].add((a, b))
    R = _rm_mine(dict(pairs), 800, 927)
    D = R.get('cdiag', []); ok = [x for x in D if x[6] >= 3]
    P(f"\n== {V} documents: {len(F)} facts ({len(obs)} observed, {len(held)} held out), {len({x for a, _, b in obs for x in (a, b)})} names | relations mined "
      f"{len(R['rels'])} | candidates {R['cand']}, testable {len(ok)} | RULES {len(R['rules'])} | {time.time() - t1:.0f}s")
    for t in R['rules'][:12]: P(f"     {rn(t[0])} o {rn(t[1])} -> {rn(t[2])}: predicts {t[6]:.2f} of {t[7]}")
    for x in sorted(ok, key=lambda x: -x[5])[:6]:
        if (x[0], x[1], x[2]) not in {t[:3] for t in R['rules']}: P(f"     (not a rule) {rn(x[0])} o {rn(x[1])} -> {rn(x[2])}: {x[5]:.2f} of {x[6]}")
    pr = [(p[0], p[1], p[2], p[6]) for p in R['props']]
    sc = [((a, r, c) in ALLF) for a, r, c, w in pr if (a, c) in ALLA]; scw = [((a, r, c) in ALLF) for a, r, c, w in pr if w and (a, c) in ALLA]
    rec = sum(1 for a, r, c, w in pr if (a, r, c) in held)
    P(f"   proposals {len(pr)}: on annotated pairs {len(sc)}, gold {np.mean(sc) if sc else float('nan'):.2f} | would be written (rule 0.9+, stored "
      f"premises) {sum(1 for x in pr if x[3])}, on annotated pairs {len(scw)}, gold {np.mean(scw) if scw else float('nan'):.2f} | HELD-OUT facts recovered {rec}")
P(f"DONE in {time.time() - T0:.0f}s")
