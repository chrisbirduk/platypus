"""REAL-TEXT COMPOSITION RULES (17 Nov 2026; tools/rt_rules.py) - offline, the distant set: can composition rules r1 o r2 -> r3 be MINED from
the system's own real-text records (no relation labels), and do the conclusions they license hold in documents read LATER?

Relation of a record 'A ... B' (label-free): its TYPED FRAME = (type of A, the words between when 1-3 of them, else the last word, type of B);
types are the store's automatic name types (as the typed writes use). Training = records of the first TRAIN share of documents; held out =
the rest. Rules: for every chain A r1 B, B r2 C in training (B's fan-out capped), the frames r3 that also link A to C in training; a rule
is kept when it predicts at least MIN_SUP stated A-C pairs, with a Wilson 95% lower bound of its confidence >= MIN_LB, and it beats the
same count on chains whose C is shuffled among r2's objects (null). Then every training chain with a kept rule and an A-C pair NEVER
STATED in training is a COMPOSED conclusion 'A r3 C', checked by (a) the pair stated in a held-out document (later reading), (b) a
distant label (Wikidata relation between A and C; undercounts), and the same for ALL chains (no rule) and random pairs of the same types.
v2 options: RR_WORDS 1 = frames must end in a word (no punctuation-only or 'and'/'or' list frames); RR_XDOC 1 = the two hops come from
DIFFERENT documents and a rule's evidence (A-C stated) must come from a third document (composition across reading, not one list).
Usage: python3 tools/rt_rules.py <distant.json.gz>   env RR_TRAIN (0.7) RR_FAN (30) RR_MINSUP (5) RR_MINLB (0.3) RR_SEED (4) RR_WORDS (0) RR_XDOC (0)"""
import os, sys, types, collections, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm

TR = float(os.environ.get('RR_TRAIN', 0.7)); FAN = int(os.environ.get('RR_FAN', 30)); MINSUP = int(os.environ.get('RR_MINSUP', 5))
MINLB = float(os.environ.get('RR_MINLB', 0.3)); WORDS = int(os.environ.get('RR_WORDS', 0)); XDOC = int(os.environ.get('RR_XDOC', 0)); rs = np.random.RandomState(int(os.environ.get('RR_SEED', 4)))
t0 = time.time()
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, TY, LAB, ND = pm._PM['R'], pm._PM['TY'], pm._PM['LAB'], pm._PM['ND']
cut = int(ND * TR)


STOP = {'and', 'or', 'the', 'a', 'an', ',', '(', ')', ';', ':', '-', '"', "'s"}
def frame(r):
    mid = r[3].split()
    w = r[3] if 1 <= len(mid) <= 3 else (r[4] if mid else '')
    if WORDS and (not mid or not mid[-1].isalpha() or mid[-1] in STOP): return None
    return (TY.get(r[1], 'X'), w, TY.get(r[2], 'X'))


tr_f = collections.defaultdict(dict); tr_pair = collections.defaultdict(set); ho_pair = set()
for r in R:
    if r[0] < cut:
        tr_pair[frozenset((r[1], r[2]))].add(r[0])
        if r[5] > 6: continue
        f = frame(r)
        if f is not None: tr_f[(r[1], r[2])].setdefault(f, set()).add(r[0])
    else: ho_pair.add(frozenset((r[1], r[2])))
out_ = collections.defaultdict(list); fr_obj = collections.defaultdict(list)
for (a, b), fs in tr_f.items():
    for f, ds in fs.items(): out_[a].append((f, b, ds)); fr_obj[f].append(b)
print(f"distant docs {ND} (train {cut}), records {len(R)}, training (A, B) pairs {len(tr_f)}, frames {len(fr_obj)}, read {time.time() - t0:.0f}s", flush=True)


def chains(shuffle=False):
    for a, L in out_.items():
        for f1, b, d1 in L:
            L2 = out_.get(b) or []
            if len(L2) > FAN: L2 = [L2[i] for i in rs.choice(len(L2), FAN, replace=False)]
            for f2, c, d2 in L2:
                if XDOC and not (d2 - d1 or d1 - d2): continue                 # the hops must be readable in different documents
                if shuffle:
                    ob = fr_obj[f2]; c = ob[rs.randint(len(ob))]
                if c == a or c == b: continue
                yield a, f1, b, f2, c, d1 | d2


def mine(shuffle):
    cnt = collections.Counter(); hit = collections.Counter()
    for a, f1, b, f2, c, dd in chains(shuffle):
        cnt[(f1, f2)] += 1
        for f3, d3 in tr_f.get((a, c), {}).items():
            if XDOC and not (d3 - dd): continue                                # evidence from a third document
            hit[(f1, f2, f3)] += 1
    return cnt, hit


cnt, hit = mine(False); ncnt, nhit = mine(True)
def wlb(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
rules = {}
for (f1, f2, f3), k in hit.items():
    n = cnt[(f1, f2)]
    if k < MINSUP: continue
    lb = wlb(k, n); nk = nhit.get((f1, f2, f3), 0); nn = max(1, ncnt.get((f1, f2), 0))
    if lb >= MINLB and k / n > 2 * (nk / nn):
        if (f1, f2) not in rules or lb > rules[(f1, f2)][1]: rules[(f1, f2)] = (f3, lb, k, n, nk / nn)
print(f"chain pairs (r1, r2) {len(cnt)}; candidate rules with support >= {MINSUP}: {sum(1 for x, k in hit.items() if k >= MINSUP)}; KEPT {len(rules)} "
      f"(Wilson LB >= {MINLB}, > 2x the shuffled null)", flush=True)
for (f1, f2), (f3, lb, k, n, nl) in sorted(rules.items(), key=lambda x: -x[1][1])[:15]:
    print(f"  {f1} o {f2} -> {f3}: conf {k / n:.2f} (LB {lb:.2f}, {k} of {n}; null {nl:.3f})", flush=True)


def check(P):
    P = list(P)
    if not P: return "n 0"
    ho = np.mean([frozenset((a, c)) in ho_pair for a, c in P]); lb = np.mean([frozenset((a, c)) in LAB for a, c in P])
    return f"n {len(P)}: stated in a LATER document {ho:.3f}, distant label {lb:.3f}"


comp = set(); allc = set()
for a, f1, b, f2, c, dd in chains(False):
    if frozenset((a, c)) in tr_pair: continue
    allc.add((a, c))
    if (f1, f2) in rules: comp.add((a, c))
names = list({x for p in allc for x in p}); byty = collections.defaultdict(list)
for n in names: byty[TY.get(n, 'X')].append(n)
rnd = set()
for a, c in list(comp)[:20000]:
    L = byty[TY.get(c, 'X')]; c2 = L[rs.randint(len(L))]
    if c2 != a and frozenset((a, c2)) not in tr_pair: rnd.add((a, c2))
sub = list(allc); sub = [sub[i] for i in rs.choice(len(sub), min(50000, len(sub)), replace=False)] if sub else []
print(f"COMPOSED by kept rules (never stated in training): {check(comp)}", flush=True)
print(f"ALL CHAINS (no rule; sample): {check(sub)}", flush=True)
print(f"RANDOM same-type pairs (control): {check(rnd)}", flush=True)
byr = collections.defaultdict(set)
for a, f1, b, f2, c, dd in chains(False):
    if (f1, f2) in rules and frozenset((a, c)) not in tr_pair: byr[(f1, f2)].add((a, c))
for k_, P in sorted(byr.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"  rule {k_[0]} o {k_[1]} -> {rules[k_][0]}: {check(P)}", flush=True)
print(f"done {time.time() - t0:.0f}s", flush=True)
