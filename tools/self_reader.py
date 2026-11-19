"""SELF-CONSISTENCY READER (19 Nov 2026; tools/self_reader.py) - offline, distant set, LABEL-FREE (distant labels for evaluation only).
The reader's records are name pairs within 12 words of one sentence; label-free it ranks them at AUC 0.62. Two ways for what else was read
to judge a record, and route J's abstention on real text:
1. CROSS-DOCUMENT AGREEMENT: how many OTHER documents record the same pair (and with a non-glue link word); AUC on held-out documents
   against distance alone, and combined.
2. ROUTE J ON REAL RECORDS: free name codes plus one MLP predictor per link frame (the last word between the names; the 60 commonest),
   trained with InfoNCE on the records of the TRAINING documents only (no labels). A held-out document's record is scored by how well
   its pair fits everything else read: the predictor's log-probability of B given (A, frame) and of A given (B, frame), among all names.
   AUC on held-out records whose two names both occur in training, against distance and agreement on the same records.
3. INFERENCE AND ABSTENTION: for training subjects, the predictor's top object per frame that was NEVER recorded with the subject anywhere:
   precision by the distant labels (the pair is labelled in some document), by margin bin, against random never-recorded pairs; the gate
   (5th percentile of the margins of recorded facts the predictor reproduces) and its precision and yield.
Usage: python3 tools/self_reader.py <distant.json.gz>   env SR_DIM (128) SR_HID (512) SR_STEPS (4000) SR_FRAMES (60) SR_THREADS (2)"""
import os, sys, time, types, collections
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.metrics import roc_auc_score
T0 = time.time(); P = lambda s: print(s, flush=True); torch.manual_seed(0); torch.set_num_threads(int(os.environ.get('SR_THREADS', 2)))
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, TY, LAB, ND = pm._PM['R'], pm._PM['TY'], pm._PM['LAB'], pm._PM['ND']
GLUE = {',', 'and', 'or', 'the', 'a', 'an', '(', ')', '-', '"', "'", ';', ':', ''}
y = np.array([frozenset((r[1], r[2])) in LAB for r in R], float); doc = np.array([r[0] for r in R]); test = doc >= int(ND * 0.8); train = ~test
dist = np.array([r[5] for r in R], float)
P(f"{ND} documents, {len(R)} records ({train.sum()} training, {test.sum()} held out); base rate held out {y[test].mean():.3f}  ({time.time() - T0:.0f}s)")
def rep(nm, s, mask):
    st, yt = s[mask], y[mask]; o = np.argsort(-st)
    P(f"   {nm:58s} AUC {roc_auc_score(yt, st):.3f} | precision top 5% {yt[o[:max(1, len(o) // 20)]].mean():.3f}, top 20% {yt[o[:max(1, len(o) // 5)]].mean():.3f}  (n {mask.sum()})")
# ---- 1. cross-document agreement
pd = collections.defaultdict(set); pdl = collections.defaultdict(set)
for r in R:
    k = frozenset((r[1], r[2])); pd[k].add(r[0])
    if r[4] not in GLUE: pdl[k].add(r[0])
agr = np.array([len(pd[frozenset((r[1], r[2]))] - {r[0]}) for r in R], float); agl = np.array([len(pdl[frozenset((r[1], r[2]))] - {r[0]}) for r in R], float)
P("1. CROSS-DOCUMENT AGREEMENT (held-out records):")
rep('distance only (nearer = better)', -dist, test)
rep('other documents recording the pair', agr, test)
rep('other documents recording it with a link word', agl, test)
rep('agreement + distance (log1p(docs) - 0.05 x distance)', np.log1p(agr) - 0.05 * dist, test)
P(f"   share of held-out records with any other document: {(agr[test] > 0).mean():.3f}")
# ---- 2. route J on the training records
fr = collections.Counter(r[4] for r, t in zip(R, train) if t and r[4] not in GLUE); FR = [f for f, _ in fr.most_common(int(os.environ.get('SR_FRAMES', 60)))]; FX = {f: k for k, f in enumerate(FR)}
names = sorted({x for r, t in zip(R, train) if t for x in (r[1], r[2])}); NX = {n: k for k, n in enumerate(names)}
trp = sorted({(NX[r[1]], FX[r[4]], NX[r[2]]) for r, t in zip(R, train) if t and r[4] in FX})
P(f"2. ROUTE J ON REAL RECORDS: {len(names)} names, {len(FR)} frames, {len(trp)} training facts (unlabelled)  ({time.time() - T0:.0f}s)")
D_, H_ = int(os.environ.get('SR_DIM', 128)), int(os.environ.get('SR_HID', 512))
class PJ(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.E = torch.nn.Embedding(len(names), D_); s.Rf = torch.nn.Embedding(len(FR) * 2, D_)
        s.net = torch.nn.Sequential(torch.nn.Linear(2 * D_, H_), torch.nn.GELU(), torch.nn.Linear(H_, H_), torch.nn.GELU(), torch.nn.Linear(H_, D_))
        torch.nn.init.normal_(s.E.weight, std=0.1)
    def logits(s, a, f, back=False):
        q = F.normalize(s.net(torch.cat([s.E(a), s.Rf(f + (len(FR) if back else 0))], -1)), dim=-1)
        return q @ F.normalize(s.E.weight, dim=-1).t() / 0.05
m = PJ(); op = torch.optim.Adam(m.parameters(), lr=2e-3); T = torch.tensor(trp); g = torch.Generator().manual_seed(1)
for st in range(int(os.environ.get('SR_STEPS', 4000))):
    b = T[torch.randint(0, len(T), (512,), generator=g)]
    l = F.cross_entropy(m.logits(b[:, 0], b[:, 1]), b[:, 2]) + F.cross_entropy(m.logits(b[:, 2], b[:, 1], True), b[:, 0])
    op.zero_grad(); l.backward(); op.step()
    if (st + 1) % 1000 == 0:
        with torch.no_grad(): acc = (m.logits(T[:2000, 0], T[:2000, 1]).argmax(-1) == T[:2000, 2]).float().mean()
        P(f"   step {st + 1}: loss {float(l):.3f} | training facts reproduced {float(acc):.3f}  ({time.time() - T0:.0f}s)")
m.eval()
ok = np.array([t and r[1] in NX and r[2] in NX for r, t in zip(R, test)])
idx = np.where(ok)[0]; sJ = np.full(len(R), np.nan); sJf = np.full(len(R), np.nan)
with torch.no_grad():
    for i0 in range(0, len(idx), 2048):
        ch = idx[i0:i0 + 2048]; a = torch.tensor([NX[R[i][1]] for i in ch]); bb = torch.tensor([NX[R[i][2]] for i in ch])
        fs = torch.tensor([FX.get(R[i][4], -1) for i in ch]); has = fs >= 0; fs = fs.clamp(min=0)
        la = F.log_softmax(m.logits(a, fs), -1).gather(1, bb[:, None])[:, 0] + F.log_softmax(m.logits(bb, fs, True), -1).gather(1, a[:, None])[:, 0]
        best = torch.stack([F.log_softmax(m.logits(a, torch.full_like(a, k)), -1).gather(1, bb[:, None])[:, 0] for k in range(len(FR))], 1).max(1).values
        sJf[ch] = np.where(has.numpy(), la.numpy(), -1e9); sJ[ch] = best.numpy()
P(f"   held-out records with both names known from training: {ok.sum()} of {test.sum()} (base rate {y[ok].mean():.3f})")
rep('distance only, same records', -dist, ok)
rep('cross-document agreement, same records', np.log1p(agr), ok)
rep('ROUTE J: fit with everything else read (best frame)', sJ, ok)
okf = ok & (sJf > -1e8)
rep('ROUTE J: fit under the record\'s own link frame', sJf, okf)
rep('ROUTE J (best frame) + agreement + distance (ranks summed)', np.nan_to_num(np.argsort(np.argsort(np.nan_to_num(sJ, nan=-1e9))) + np.argsort(np.argsort(np.log1p(agr))) + np.argsort(np.argsort(-dist)), nan=0.0), ok)
# ---- 3. inference and abstention
rec = collections.defaultdict(set)
for r in R: rec[r[1]].add(r[2]); rec[r[2]].add(r[1])
rs = np.random.RandomState(0); subj = rs.choice(len(names), min(3000, len(names)), replace=False); out = []; mrec = []
with torch.no_grad():
    for i0 in range(0, len(subj), 500):
        a = torch.tensor(subj[i0:i0 + 500])
        for k in range(len(FR)):
            pr = F.softmax(m.logits(a, torch.full_like(a, k)), -1); tv = pr.topk(2)
            for j in range(len(a)):
                A, B = names[int(a[j])], names[int(tv.indices[j, 0])]
                mg = float(tv.values[j, 0] - tv.values[j, 1])
                if B != A and B not in rec[A]: out.append((mg, frozenset((A, B)) in LAB, k))
    for i0 in range(0, min(len(trp), 20000), 2000):
        b = T[i0:i0 + 2000]; pr = F.softmax(m.logits(b[:, 0], b[:, 1]), -1); tv = pr.topk(2)
        good = tv.indices[:, 0] == b[:, 2]; mrec += (tv.values[:, 0] - tv.values[:, 1])[good].tolist()
thr = float(np.quantile(mrec, 0.05)) if mrec else 0.0
mg = np.array([o[0] for o in out]); yy = np.array([o[1] for o in out], float)
base = []
while len(base) < 20000:
    A, B = names[rs.randint(len(names))], names[rs.randint(len(names))]
    if A != B and B not in rec[A]: base.append(frozenset((A, B)) in LAB)
P(f"3. INFERENCE (never-recorded pairs proposed by route J; {len(subj)} subjects x {len(FR)} frames = {len(out)} proposals): base (random never-recorded pairs) {np.mean(base):.4f}")
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)):
    sel = (mg >= lo) & (mg < hi); P(f"   margin {lo:.1f}-{hi:.1f}: precision {yy[sel].mean() if sel.any() else float('nan'):.4f} (n {sel.sum()})")
sel = mg >= thr
P(f"   GATE (5th percentile of reproduced recorded facts' margins = {thr:.3f}): passes {sel.mean():.3f} of proposals ({sel.sum()}), precision {yy[sel].mean() if sel.any() else float('nan'):.4f} "
  f"vs {yy.mean():.4f} for all proposals (random base {np.mean(base):.4f}, n {len(base)})")
P(f"DONE ({time.time() - T0:.0f}s)")
