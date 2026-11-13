"""THE STORE TEACHING ITSELF WHAT TO EXTRACT (13 Nov 2026; tools/consistency_teacher.py) - offline, no trunk, no effect on the package.
tools/extract_learn.py: the trunk's prediction feedback is a weak teacher (learner top 10% 0.33-0.35 related vs 0.24 base; the model's own trunk
at chance per record, GPU cg27a_xgain visits 2-4). Here the teacher is CONSISTENCY among the records themselves - label-free, from counting:
  T1 CORROBORATION   the same two names joined by the same wording in another document (later reading confirms it)
  T2 FUNCTIONAL      the wording maps a subject to ONE object: among subjects it joins to 2+ records, the share whose objects all agree
                     ('born in', 'the capital of' are functional; list glue ',' 'and' joins a subject to many names)
  T3 SELECTIVE       the wording's subjects (and objects) share a shape (a person-like / place-like / number name by capitalisation, digits and
                     length): low entropy of the shapes it joins (selectional preference, Resnik 1996)
  T4 RULES           the record takes part in a chain of a rule the COUNTING MINER verifies over the candidate records themselves (wordings with
                     20+ records as relations; n >= 10 subjects, Wilson z 3 >= 0.5): a premise of a verified rule, or a conclusion it predicts
Every statistic is computed on the TRAINING documents only. Scored on the same held-out documents as tools/extract_learn.py (a fifth, 15,609
candidates, 24.2% related): each teacher's score applied directly (wordings seen in training), and a LEARNER (ridge, the same no-gold features)
trained on the teachers' combined score and applied to every held-out candidate. Compare: trunk-feedback learner top 10% 0.33-0.35; ceiling with
gold labels 0.50 (same amount of data) / 0.71 (all training candidates).
Usage: python3 tools/consistency_teacher.py"""
import os
import sys
import json
import math
import time
import collections
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
P = lambda *a: print(*a, flush=True); T0 = time.time()
DIR = os.environ.get('DV_DIR', '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad/bench_dl')
docs = json.load(open(DIR + '/train_revised.json')) + json.load(open(DIR + '/dev_revised.json'))


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    _u, _i, _c = np.unique(np.concatenate([pos, neg]), return_inverse=True, return_counts=True); r = (np.cumsum(_c) - (_c - 1) / 2.0)[_i]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


rs0 = np.random.RandomState(0); dperm = rs0.permutation(len(docs)); TEST = set(dperm[:len(docs) // 5].tolist())   # as tools/extract_learn.py
C = []
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    gold = {(l['h'], l['t']) for l in d['labels']} | {(l['t'], l['h']) for l in d['labels']}
    bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms); toks = d['sents'][si]
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                (a0, a1), (b0, b1) = ms[x][:2], ms[y][:2]; ea, eb = ms[x][2], ms[y][2]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb]: continue
                if b0 - a1 <= 12 and len(' '.join(toks[a0:b1])) <= 60:
                    C.append(dict(di=di, gold=(ea, eb) in gold, mid=toks[a1:b0], A=toks[a0:a1], B=toks[b0:b1], cA=cn[ea], cB=cn[eb], test=di in TEST,
                                  f=' '.join(toks[a1:b0]).lower()))
y = np.array([c['gold'] for c in C]); te = np.array([c['test'] for c in C]); TR = np.where(~te)[0]; TE = np.where(te)[0]; yt = y[TE]
FRAC = float(os.environ.get('CT_FRAC', 1.0))                                   # 13 Nov: share of the TRAINING documents read (volume curve)
if FRAC < 1.0:
    trd = sorted({C[i]['di'] for i in TR}); kd = set(np.random.RandomState(7).permutation(trd)[:int(len(trd) * FRAC)].tolist())
    TR = np.array([i for i in TR if C[i]['di'] in kd])
P(f"candidates {len(C)} (related {y.mean():.3f}); training {len(TR)}, held-out {len(TE)} (related {yt.mean():.3f})  ({time.time() - T0:.0f}s)")


def shape(toks):
    w = ' '.join(toks)
    if any(ch.isdigit() for ch in w): return 'num'
    caps = sum(t[:1].isupper() for t in toks)
    return f"cap{min(caps, 3)}" if caps else 'low'


# ---- statistics from TRAINING documents only
trip = collections.defaultdict(set); fsub = collections.defaultdict(lambda: collections.defaultdict(set)); fcnt = collections.Counter()
fshA = collections.defaultdict(collections.Counter); fshB = collections.defaultdict(collections.Counter)
for i in TR:
    c = C[i]; trip[(c['cA'], c['f'], c['cB'])].add(c['di']); fsub[c['f']][c['cA']].add(c['cB']); fcnt[c['f']] += 1
    fshA[c['f']][shape(c['A'])] += 1; fshB[c['f']][shape(c['B'])] += 1
def func(f):
    subs = [o for o in fsub[f].values()] if f in fsub else []
    k = [len(o) == 1 for o in subs if len(o) >= 1]
    multi = [o for a, o in fsub[f].items()] if f in fsub else []
    n2 = sum(1 for a in fsub.get(f, {}) for _ in [0])
    # subjects with 2+ RECORDS under f: objects agree?
    return None
recs_per_sub = collections.defaultdict(lambda: collections.defaultdict(list))
for i in TR: recs_per_sub[C[i]['f']][C[i]['cA']].append(C[i]['cB'])
FUNC = {}
for f, d_ in recs_per_sub.items():
    ss = [v for v in d_.values() if len(v) >= 2]
    if len(ss) >= 3: FUNC[f] = float(np.mean([len(set(v)) == 1 for v in ss]))
def ent(cn_):
    n = sum(cn_.values()); return -sum(v / n * math.log2(v / n) for v in cn_.values()) if n else 0.0
SEL = {f: -(ent(fshA[f]) + ent(fshB[f])) for f in fcnt if fcnt[f] >= 5}
# T4 rules by counting over wordings with 20+ records (and their inverses)
sub = collections.defaultdict(lambda: collections.defaultdict(set))
for i in TR:
    c = C[i]
    if fcnt[c['f']] >= 20: sub[c['f']][c['cA']].add(c['cB']); sub['~' + c['f']][c['cB']].add(c['cA'])
rels = list(sub); tgt = [r for r in rels if not r.startswith('~')]; RULES = []
def wil(p, n, z=3.0): return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n) if n else 0.0
for r1 in rels:
    for r2 in rels:
        if r2 == '~' + r1 or r1 == '~' + r2: continue
        ends = {}
        for a, bs in sub[r1].items():
            cs = collections.Counter(x for b in bs for x in sub[r2].get(b, ()) if x != a)
            if cs: ends[a] = cs
        if len(ends) < 10: continue
        for r3 in tgt:
            if r3 in (r1.lstrip('~'), r2.lstrip('~')): continue
            tst = [a for a in ends if a in sub[r3]]
            if len(tst) < 10: continue
            acc = sum(1 for a in tst if ends[a].most_common(1)[0][0] in sub[r3][a]) / len(tst)
            if wil(acc, len(tst)) >= 0.5: RULES.append((r1, r2, r3, acc, len(tst), ends))
INRULE = set()                                                                 # (subject, wording, object) records a verified rule uses or predicts
for r1, r2, r3, acc, n, ends in RULES:
    for a, cs in ends.items():
        x = cs.most_common(1)[0][0]
        INRULE.add((a, r3, x))
        for b in sub[r1].get(a, ()):
            if x in sub[r2].get(b, ()):
                INRULE.add((a, r1.lstrip('~'), b) if not r1.startswith('~') else (b, r1[1:], a)); INRULE.add((b, r2.lstrip('~'), x) if not r2.startswith('~') else (x, r2[1:], b))
P(f"training statistics: wordings {len(fcnt)}, functional-scored {len(FUNC)}, selectivity-scored {len(SEL)}, counting-miner RULES over candidate records "
  f"{len(RULES)}" + "".join(f"\n     '{r1}' o '{r2}' -> '{r3}': {acc:.2f} of {n}" for r1, r2, r3, acc, n, _ in sorted(RULES, key=lambda r: -r[4])[:8]) + f"  ({time.time() - T0:.0f}s)")


def teach(ix):
    t1 = np.array([float(len(trip.get((C[i]['cA'], C[i]['f'], C[i]['cB']), set()) - {C[i]['di']}) > 0) for i in ix])
    t2 = np.array([FUNC.get(C[i]['f'], np.nan) for i in ix]); t3 = np.array([SEL.get(C[i]['f'], np.nan) for i in ix])
    t4 = np.array([float((C[i]['cA'], C[i]['f'], C[i]['cB']) in INRULE) for i in ix])
    return t1, t2, t3, t4


def rep(tag, sc, yy=None):
    yy = yt if yy is None else yy; sc = np.where(np.isnan(sc), np.nanmedian(sc) if np.isfinite(sc).any() else 0.0, sc)
    o = np.argsort(-sc + 1e-9 * np.random.RandomState(5).rand(len(sc))); k1, k2 = len(sc) // 10, len(sc) // 4
    P(f"   {tag:56s} AUC {auc(sc[yy], sc[~yy]):.3f} | top 10%: precision {yy[o[:k1]].mean():.3f}, recall {yy[o[:k1]].sum() / max(1, yy.sum()):.3f} | "
      f"top 25%: precision {yy[o[:k2]].mean():.3f}, recall {yy[o[:k2]].sum() / max(1, yy.sum()):.3f}")


a1, a2, a3, a4 = teach(TR); ytr = y[TR]
P(f"TEACHERS on training records (how well each label-free teacher marks related pairs; coverage = share of records it can score):")
for nm_, t_ in (('T1 corroborated in another document', a1), ('T2 functional wording', a2), ('T3 selective wording', a3), ('T4 in a verified rule chain', a4)):
    ok = np.isfinite(t_); P(f"   {nm_:40s} coverage {ok.mean():.3f} | AUC {auc(t_[ok & ytr], t_[ok & ~ytr]):.3f} | related when the teacher is high (top quarter of scored) "
                           f"{ytr[ok][np.argsort(-t_[ok])[:max(1, ok.sum() // 4)]].mean():.3f} (base {ytr.mean():.3f})")
b1, b2, b3, b4 = teach(TE)
P(f"HELD-OUT DOCUMENTS ({len(TE)} candidates, related {yt.mean():.3f}); the teachers applied directly (wordings seen in training):")
rep('random', np.random.RandomState(3).rand(len(TE)))
for nm_, t_ in (('T2 functional wording', b2), ('T3 selective wording', b3)): rep(nm_, t_)
rk = lambda x: (lambda z: (np.argsort(np.argsort(z)) / len(z)))(np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(x).any() else 0, x))
comb_te = rk(b2) + rk(b3)
rep('T2 + T3 (rank sum)', comb_te)
HV = HashingVectorizer(n_features=2 ** 16, ngram_range=(1, 2), alternate_sign=False, token_pattern=r"[A-Za-z]+|[^\sA-Za-z]", norm='l2')
fq = collections.Counter(c['f'] for c in C)
def feats(ix):
    W = HV.transform([' '.join(C[i]['mid']) for i in ix])
    ends = HV.transform([('FIRST_' + (C[i]['mid'][0] if C[i]['mid'] else 'none') + ' LAST_' + (C[i]['mid'][-1] if C[i]['mid'] else 'none')).lower() for i in ix])
    D = np.array([[len(C[i]['mid']), float(',' in C[i]['mid']), float('(' in C[i]['mid'] or ')' in C[i]['mid']), float(any(w[:1].isdigit() for w in C[i]['A'] + C[i]['B'])),
                   float(any(w[:1].isupper() for w in C[i]['mid'])), np.log1p(fq[C[i]['f']]), float(len(C[i]['mid']) == 0)] for i in ix], float)
    D = (D - D.mean(0)) / (D.std(0) + 1e-6); return hstack([W, ends, csr_matrix(D)]).tocsr()
Xtr, Xte = feats(TR), feats(TE)
teacher_tr = rk(a2) + rk(a3) + 2 * a1 + 2 * a4                                 # combined label-free teacher score per training record
for nm_, lab in (('combined teacher (T1-T4)', teacher_tr), ('T2 + T3 only', rk(a2) + rk(a3)), ('T1 corroboration only', a1), ('T4 rule chains only', a4)):
    if np.std(lab) == 0: P(f"   LEARNER from {nm_}: teacher constant, skipped"); continue
    rep(f"LEARNER from {nm_} (ridge)", Ridge(alpha=10.0).fit(Xtr, lab).predict(Xte))
    hi = lab >= np.percentile(lab, 75)
    if hi.any() and (~hi).any(): rep(f"LEARNER from {nm_} (top quarter as positives)", LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, hi).decision_function(Xte))
rep('CEILING: same learner on GOLD labels, all training candidates', LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr).decision_function(Xte))
P(f"DONE ({time.time() - T0:.0f}s)")

# ---- SELF-TRAINING ROUNDS (13 Nov): the learner's own top records rebuild the teachers' statistics (functional / selective wordings counted
#      only over records it keeps), the teachers relabel, the learner retrains - the store bootstrapping what to extract, still no gold
P("SELF-TRAINING: each round recounts T2 / T3 over the training records the learner keeps (top 30%), relabels, retrains")
lr_ = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, teacher_tr >= np.percentile(teacher_tr, 75)); s_tr = lr_.decision_function(Xtr)
for rnd in range(1, 4):
    keep = TR[s_tr >= np.percentile(s_tr, 70)]
    rps = collections.defaultdict(lambda: collections.defaultdict(list)); shA = collections.defaultdict(collections.Counter); shB = collections.defaultdict(collections.Counter); fc_ = collections.Counter()
    for i in keep: c = C[i]; rps[c['f']][c['cA']].append(c['cB']); shA[c['f']][shape(c['A'])] += 1; shB[c['f']][shape(c['B'])] += 1; fc_[c['f']] += 1
    FUNC = {f: float(np.mean([len(set(v)) == 1 for v in d_.values() if len(v) >= 2])) for f, d_ in rps.items() if sum(1 for v in d_.values() if len(v) >= 2) >= 3}
    SEL = {f: -(ent(shA[f]) + ent(shB[f])) for f in fc_ if fc_[f] >= 5}
    a1, a2, a3, a4 = teach(TR); lab = rk(a2) + rk(a3) + 2 * a1 + 2 * a4 + rk(s_tr)
    lr_ = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, lab >= np.percentile(lab, 75)); s_tr = lr_.decision_function(Xtr)
    rep(f"round {rnd}: LEARNER (kept {len(keep)} training records; functional {len(FUNC)}, selective {len(SEL)} wordings)", lr_.decision_function(Xte))
P(f"DONE ({time.time() - T0:.0f}s)")
