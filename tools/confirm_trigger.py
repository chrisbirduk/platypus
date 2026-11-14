"""WHICH WRITTEN INFERENCES ARE WRONG? A RETRACTION TRIGGER FROM LATER READING (14 Nov 2026; tools/confirm_trigger.py) - offline.
tools/retract_diag.py: unlikelihood (weight 0.1) unlearns a taught wrong inference cleanly; the trigger is the open part. Re-ranking a write
by the store's own score after more reading (tools/fact_compress.py FC_SAVE / FC_LOAD) flags wrong ones with a likelihood ratio of only
about 2 (bottom 10-20%), which at the GPU's base (a third wrong) retracts as many right as wrong. Here: the 200 writes of stage 1
(FC_SAVE files, code + Adamic-Adar after 5,000 distant documents) against what the next documents read (distant 5,000-30,000) say about
each pair, label-free: documents and records stating the pair, their wording's FUNCTIONALITY (share of a wording's subjects with a single
partner; the best label-free relation teacher in tools/consistency_teacher.py), the gap between the names, and the pair's PMI over
documents (co-mention beyond what the two names' frequencies predict). For each: AUC right vs wrong, and the 'least confirmed' 10 / 20 / 30%
as the retraction set: wrong share, likelihood ratio (wrong caught / right lost) and the implied wrong share at the GPU's base of 0.34.
Usage: python3 tools/confirm_trigger.py <ret_s0.json> [<ret_s1.json> ...]   env CT_LO (5000), CT_HI (30000)"""
import os, sys, json, pickle, math, collections
import numpy as np
from sklearn.metrics import roc_auc_score
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
LO = int(os.environ.get('CT_LO', 5000)); HI = int(os.environ.get('CT_HI', 30000))
R = [r for r in pickle.load(open(SCR + '/distant/records.pkl', 'rb'))['rec'] if r[0] < HI]
print(f"distant records read up to document {HI}: {len(R)}; new reading = documents {LO}-{HI}", flush=True)
subj = collections.defaultdict(lambda: collections.defaultdict(set))                 # wording -> subject -> partners (all reading so far)
for r in R: subj[r[3]][r[1]].add(r[2])
fun = {w: np.mean([1.0 / len(v) for v in d.values()]) for w, d in subj.items() if sum(len(v) for v in d.values()) >= 5}
new = [r for r in R if r[0] >= LO]; nd = HI - LO
dfn = collections.Counter(); seen = set()
for r in new:
    for n in (r[1], r[2]):
        if (r[0], n) not in seen: seen.add((r[0], n)); dfn[n] += 1
byp = collections.defaultdict(list)
for r in new: byp[frozenset((r[1], r[2]))].append(r)
def feats(A, B):
    rs = byp.get(frozenset((A, B)), []); docs = {r[0] for r in rs}
    fs = [fun.get(r[3], 0.0) for r in rs]; gaps = [r[5] for r in rs]
    pmi = math.log((len(docs) + 0.5) * nd / ((dfn[A] + 1) * (dfn[B] + 1)))
    return dict(docs=len(docs), records=len(rs), short=sum(g <= 3 for g in gaps), fun_max=max(fs, default=0.0), fun_mean=float(np.mean(fs)) if fs else 0.0,
                fun_docs=len({r[0] for r, f in zip(rs, fs) if f >= 0.5}), pmi=pmi, popularity=-math.log((dfn[A] + 1) * (dfn[B] + 1)))
for path in sys.argv[1:]:
    W = json.load(open(path)); y = np.array([w[2] for w in W], bool); F = [feats(w[0], w[1]) for w in W]
    print(f"== {os.path.basename(path)}: {len(W)} writes, wrong share {1 - y.mean():.2f}")
    keys = list(F[0]) + ['stage-1 rank (code + AA)', 'pmi + stage-1 rank', 'short + pmi', 'short + pmi + fun_docs', 'short + pmi + stage-1 rank']
    cols = {k: np.array([f[k] for f in F], float) for k in F[0]}; r1 = np.array([w[3] for w in W]); cols['stage-1 rank (code + AA)'] = r1
    rk = lambda x: np.argsort(np.argsort(x)) / (len(x) - 1); cols['pmi + stage-1 rank'] = rk(cols['pmi']) + rk(r1); cols['short + pmi'] = rk(cols['short']) + rk(cols['pmi'])
    cols['short + pmi + fun_docs'] = cols['short + pmi'] + rk(cols['fun_docs']); cols['short + pmi + stage-1 rank'] = cols['short + pmi'] + rk(r1)
    for k in keys:
        v = cols[k] + 1e-9 * np.random.RandomState(0).rand(len(y)); out = []
        for q in (0.1, 0.2, 0.3):
            f = v <= np.quantile(v, q); c = (f & ~y).sum() / max(1, (~y).sum()); l = (f & y).sum() / max(1, y.sum()); lr = c / max(l, 1e-9)
            o = 0.34 / 0.66 * lr; out.append(f"bottom {int(q * 100)}%: wrong {1 - y[f].mean():.2f}, LR {lr:4.1f}, at base 0.34 -> {o / (1 + o):.2f}")
        print(f"   {k:26s} AUC right vs wrong {roc_auc_score(y, v):.3f} || " + " | ".join(out))
