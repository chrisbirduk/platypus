"""IS RELATION ORTHOGONAL TO THE TRUNK'S MAIN STRUCTURE? (14 Nov 2026; tools/relation_ortho.py) - offline, no effect on the package.
Hypothesis (the author): what a relation word contributes lies in directions orthogonal to the trunk's dominant structure (topic, domain,
frequency), so the trunk's own state identifies relations poorly (tools/relation_free.py: AUC 0.58) because the large directions swamp it.
Records as tools/relation_free.py (Re-DocRED, both names in one sentence 1-12 words apart, names masked as 'Thing' / 'Stuff', 3 words of
context each side; held-out documents; 600 queries x 30 candidates from other documents; same gold relation or not). The stand-in trunk
(TG_CKPT) reads each record; the state is averaged over the WORDING between the names, at each layer. Then:
  REMOVE THE TOP k principal components of ORDINARY TEXT states (wiki windows, the same layer), k = 0, 5, 10, 20, 50, 100: relation identity
  (nearest candidate of the same relation; AUC same vs not). If the hypothesis holds, identity RISES as the main structure is removed.
  ALIGNMENT: the share of the between-relation variance (class means of the 20 commonest relations) that falls in the top k ordinary-text
  components, against the share of all record variance there (ratio < 1: relation differences lie off the main structure).
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/relation_ortho.py"""
import os, sys, json, time, collections
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get('RF_THREADS', 1))); P = lambda *a: print(*a, flush=True); T0 = time.time()
DIR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad/bench_dl'
docs = json.load(open(DIR + '/train_revised.json')) + json.load(open(DIR + '/dev_revised.json'))
rs0 = np.random.RandomState(0); dperm = rs0.permutation(len(docs)); TEST = set(dperm[:len(docs) // 5].tolist())
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    _u, _i, _c = np.unique(np.concatenate([pos, neg]), return_inverse=True, return_counts=True); r = (np.cumsum(_c) - (_c - 1) / 2.0)[_i]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
te = []
for di in sorted(TEST):
    d = docs[di]; cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    rel = collections.defaultdict(set)
    for l in d['labels']: rel[(l['h'], l['t'])].add(l['r']); rel[(l['t'], l['h'])].add(l['r'] + '^-1')
    bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms); toks = d['sents'][si]
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y]
                if ea == eb or a1 > b0 or not 1 <= b0 - a1 <= 12 or cn[ea] == cn[eb] or not rel.get((ea, eb)): continue
                pre = ' '.join(toks[max(0, a0 - 3):a0] + ['Thing']) + ' '; mid = ' '.join(toks[a1:b0]); post = ' Stuff ' + ' '.join(toks[b1:b1 + 3])
                te.append(dict(di=di, pre=pre, mid=mid, post=post, rs=rel[(ea, eb)]))
rq = np.random.RandomState(1)
qs = rq.choice(len(te), min(600, len(te)), replace=False); PAIRS = []
for q in qs:
    cs = [int(c) for c in rq.choice(len(te), 40, replace=False) if te[int(c)]['di'] != te[q]['di']][:30]; PAIRS += [(int(q), c) for c in cs]
PAIRS = np.array(PAIRS); same = np.array([bool(te[i]['rs'] & te[j]['rs']) for i, j in PAIRS])
used = sorted(set(PAIRS.ravel().tolist()))
P(f"held-out gold-relation records {len(te)}; {len(qs)} queries x 30 candidates; same relation {same.mean():.3f}  ({time.time() - T0:.0f}s)")
sys.argv = [sys.argv[0], '6000', '5']; _tg = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trunk_guides.py')).read()
g = {}; exec(compile(_tg[:_tg.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'), g); lm, L, PAD = g['lm'], g['L'], g['PAD']
NLAY = len(lm.b)
@torch.no_grad()
def states(items, bs=64):
    """items: list of (bytes, lo, hi) - mean state over byte positions lo..hi at every layer (after each block; the last after the norm)"""
    out = [[] for _ in range(NLAY)]
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; X = torch.full((len(ch), L), PAD, dtype=torch.long); m = torch.zeros(len(ch), L)
        for j, (b, lo, hi) in enumerate(ch): X[j, :len(b)] = torch.tensor(list(b)); m[j, lo:hi] = 1
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); T_ = Xi.shape[1]; am = torch.triu(torch.ones(T_, T_, dtype=torch.bool), 1)
        h = lm.emb(Xi) + lm.pos[:T_]
        for li, blk in enumerate(lm.b):
            h = blk(h, am); hh = lm.n(h) if li == NLAY - 1 else h
            out[li].append(((hh * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)).numpy())
    return [np.concatenate(o) for o in out]
items = []
for r in te:
    a = r['pre'].encode('utf-8', 'replace'); b = r['mid'].encode('utf-8', 'replace'); c = r['post'].encode('utf-8', 'replace')
    full = (a + b + c)[:L]; lo = min(len(a), L - 1); hi = min(len(a) + max(1, len(b)), L)
    items.append((full, lo, hi))
H = states(items); P(f"record states at {NLAY} layers  ({time.time() - T0:.0f}s)")
wiki = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin', 'rb').read().replace(b'\n', b' ')
rw = np.random.RandomState(5); win = []
for i0 in rw.randint(0, len(wiki) - L - 1, 1500):
    b = wiki[i0:i0 + L]; s0 = rw.randint(8, L - 24); win.append((b, s0, s0 + rw.randint(4, 20)))   # ordinary text, a short span averaged like a wording
Hw = states(win); P(f"ordinary-text states: {len(win)} spans  ({time.time() - T0:.0f}s)")
def report(name, Z):
    Z = Z - Z.mean(0); Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9); sim = np.einsum('ij,ij->i', Z[PAIRS[:, 0]], Z[PAIRS[:, 1]])
    top, ch = [], []
    for n0 in range(0, len(PAIRS), 30):
        blk = np.arange(n0, min(n0 + 30, len(PAIRS))); b = blk[np.argmax(sim[blk])]; top.append(same[b]); ch.append(same[blk].mean())
    return f"{name}: nearest same {np.mean(top):.3f} (chance {np.mean(ch):.3f}), AUC {auc(sim[same], sim[~same]):.3f}"
rel1 = [next(iter(sorted(r['rs']))) for r in te]; cnt = collections.Counter(rel1); top20 = [x for x, _ in cnt.most_common(20)]
for li in range(NLAY):
    X = H[li]; W = Hw[li] - Hw[li].mean(0); U, S, Vt = np.linalg.svd(W, full_matrices=False)          # ordinary-text principal directions
    msg = []
    for k in (0, 5, 10, 20, 50, 100):
        if k >= Vt.shape[0]: continue
        Xc = X - Hw[li].mean(0); Xp = Xc - (Xc @ Vt[:k].T) @ Vt[:k] if k else Xc
        msg.append(report(f"top {k} removed", Xp))
    Xc = X - X.mean(0); mu = np.array([X[[i for i, r_ in enumerate(rel1) if r_ == c]].mean(0) for c in top20]); muc = mu - mu.mean(0)
    al = []
    for k in (5, 10, 20, 50):
        Vk = Vt[:k]; tot = (np.linalg.norm(Xc @ Vk.T) ** 2) / (np.linalg.norm(Xc) ** 2); btw = (np.linalg.norm(muc @ Vk.T) ** 2) / (np.linalg.norm(muc) ** 2)
        al.append(f"top {k}: relation {btw:.2f} vs all {tot:.2f} (ratio {btw / max(tot, 1e-9):.2f})")
    P(f"LAYER {li + 1} | " + " | ".join(msg))
    P(f"   ALIGNMENT (share of variance in the top-k ordinary-text directions): " + " | ".join(al))
    Ur, Sr, Vr = np.linalg.svd(Xc, full_matrices=False)                                                   # the records' own top directions
    P(f"   records' OWN top directions removed: " + " | ".join(report(f"k {k}", Xc - (Xc @ Vr[:k].T) @ Vr[:k] if k else Xc) for k in (0, 5, 10, 20)))
P(f"DONE ({time.time() - T0:.0f}s)")
