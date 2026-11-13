"""THE SURPRISED STORE: STORE WHAT THE TRUNK DOES NOT ALREADY PREDICT (13 Nov 2026; tools/surprise_store.py) - offline, no effect on the package.
SLIMM (van Kesteren et al. 2012): input that fits what the cortex already predicts is learned by the cortex and the hippocampus is suppressed;
input that does not fit is stored by the hippocampus as a separate episode. Here the stand-in trunk (tools/trunk_guides.py) reads Re-DocRED
TRAINING documents it has never seen (it was trained on WikiText and the dev documents); every sentence and every record (two names in one
sentence, the words between) gets the trunk's SURPRISE:
  sentence   bits per byte, read after the previous sentence
  record     bits per byte over the record | over the second name only | over the words between only
and whether the trunk ALREADY KNOWS a related record's fact: the true second name costs the fewest bits among it and up to 9 other names of the
same document put in its place (after the first name and the words between).
QUESTIONS: (1) does surprise pick out sentences / records that hold real relations (gold), or junk? (2) among real facts, does it keep the ones
the trunk does NOT already know - the ones only the store can hold? (3) per record kept, how many NEW true facts does a surprise-gated store
hold, against keeping at random, keeping the predictable, and the learned extractor's closeness baseline?
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/surprise_store.py [documents, default 600]"""
import os
import sys
import json
import collections
HERE = os.path.dirname(os.path.abspath(__file__))
ND_ = int(sys.argv[1]) if len(sys.argv) > 1 else 600
sys.argv = [sys.argv[0], '6000', '5']
_tg = open(os.path.join(HERE, 'trunk_guides.py')).read()
exec(compile(_tg[:_tg.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'))
P(f"\n================ THE SURPRISED STORE  ({time.time() - T0:.0f}s)")
docs = json.load(open(SCR + '/bench_dl/train_revised.json'))
docs = [docs[i] for i in np.random.RandomState(0).permutation(len(docs))[:ND_]]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    _u, _i, _c = np.unique(np.concatenate([pos, neg]), return_inverse=True, return_counts=True); r = (np.cumsum(_c) - (_c - 1) / 2.0)[_i]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.no_grad()
def perbyte(seqs, bs=96):
    """bits of every byte of each sequence (read from its start, last L bytes)"""
    out = []
    for i in range(0, len(seqs), bs):
        ch = [s[-L:] for s in seqs[i:i + bs]]; X = torch.full((len(ch), L), PAD, dtype=torch.long)
        for j, c in enumerate(ch): X[j, :len(c)] = torch.tensor(list(c))
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(lm(Xi), -1)
        nl = (-lp.gather(-1, X.clamp(max=V - 1)[..., None])[..., 0] / math.log(2)).numpy()
        for j, (s, c) in enumerate(zip(seqs[i:i + bs], ch)): out.append((len(s) - len(c), nl[j, :len(c)]))
    return out


# ---- sentences and their surprise
SENT, REC = [], []
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    gold = {(l['h'], l['t']) for l in d['labels']} | {(l['t'], l['h']) for l in d['labels']}
    prev = b''
    for si, toks in enumerate(d['sents']):
        offs = []; b = b''
        for w in toks:
            if b: b += b' '
            offs.append((len(b), len(b) + len(w.encode('utf-8', 'replace')))); b += w.encode('utf-8', 'replace')
        ctx = prev[-50:] + (b' ' if prev else b''); SENT.append(dict(di=di, si=si, ctx=ctx, b=b, offs=offs)); prev = b
        ms = sorted((m['pos'][0], m['pos'][1], ei) for ei, vs in enumerate(d['vertexSet']) for m in vs if m['sent_id'] == si)
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y]
                if ea == eb or a1 > b0 or not 1 <= b0 - a1 <= 12 or cn[ea] == cn[eb]: continue
                REC.append(dict(s=len(SENT) - 1, a=(offs[a0][0], offs[a1 - 1][1]), m=(offs[a1 - 1][1], offs[b0][0]), bb=(offs[b0][0], offs[b1 - 1][1]),
                                gold=(ea, eb) in gold, dist=b0 - a1, eb=eb, di=di, others=[cn[e] for e in range(len(cn)) if e not in (ea, eb)]))
t1 = time.time(); PB = perbyte([s['ctx'] + s['b'] for s in SENT])
for s, (cut, nl) in zip(SENT, PB):
    off = len(s['ctx']) - cut; s['nl'] = nl[max(0, off):]; s['bpb'] = float(np.mean(s['nl'])) if len(s['nl']) else np.nan
P(f"{len(docs)} unseen training documents, {len(SENT)} sentences, {len(REC)} records (related {np.mean([r['gold'] for r in REC]):.3f}) | trunk read them "
  f"({time.time() - t1:.0f}s): {np.nanmean([s['bpb'] for s in SENT]):.2f} bits per byte")
def span(r, k):
    nl = SENT[r['s']]['nl']; a, b = r[k]; v = nl[a:b] if b <= len(nl) else np.array([]); return float(np.mean(v)) if len(v) else np.nan
for r in REC:
    r['rec'] = span(dict(r, rr=(r['a'][0], r['bb'][1])), 'rr'); r['obj'] = span(r, 'bb'); r['mid'] = span(r, 'm')
    r['objsum'] = r['obj'] * (r['bb'][1] - r['bb'][0]) if np.isfinite(r['obj']) else np.nan
R = [r for r in REC if np.isfinite(r['rec']) and np.isfinite(r['obj']) and np.isfinite(r['mid'])]; y = np.array([r['gold'] for r in R])
# ---- does the trunk already KNOW a related record's fact? (true second name first among up to 10 names of the document)
rq = np.random.RandomState(1); K = [i for i, r in enumerate(R) if len(r['others']) >= 3]
K = [K[int(i)] for i in rq.permutation(len(K))[:4000]]; seqs, spans, own = [], [], []
for i in K:
    r = R[i]; s = SENT[r['s']]; pre = s['ctx'] + s['b'][:r['bb'][0]]; tb = s['b'][r['bb'][0]:r['bb'][1]]
    cands = [tb] + [o.encode('utf-8', 'replace') for o in r['others'][:9]]
    for k, c in enumerate(cands): seqs.append(pre + c); spans.append(len(pre)); own.append((i, k, len(c)))
t1 = time.time(); PK = perbyte(seqs); best = {}
for (i, k, n), (cut, nl), sp in zip(own, PK, spans):
    v = float(np.mean(nl[sp - cut:sp - cut + n])) if sp - cut + n <= len(nl) and sp - cut >= 0 else np.inf
    best.setdefault(i, []).append((v, k))
KNOWS = {i: min(v)[1] == 0 for i, v in best.items() if np.isfinite(min(v)[0])}
kn_rel = [KNOWS[i] for i in KNOWS if R[i]['gold']]; kn_un = [KNOWS[i] for i in KNOWS if not R[i]['gold']]
P(f"the trunk ALREADY KNOWS the second name (first of up to 10 names of the document; {time.time() - t1:.0f}s): related records "
  f"{np.mean(kn_rel):.3f} (n {len(kn_rel)}), unrelated {np.mean(kn_un):.3f} (n {len(kn_un)}), chance about {np.mean([1 / min(10, 1 + len(R[i]['others'])) for i in KNOWS]):.3f}")
# ---- (1) sentences: surprise vs the real relations they hold
nrel = collections.Counter(r['s'] for r in REC if r['gold']); sb = np.array([s['bpb'] for s in SENT]); ok = np.isfinite(sb)
q1, q3 = np.nanpercentile(sb, 25), np.nanpercentile(sb, 75)
P(f"(1) SENTENCES by surprise: related records per sentence - most predictable quarter {np.mean([nrel[k] for k in np.where(ok & (sb <= q1))[0]]):.2f}, "
  f"middle half {np.mean([nrel[k] for k in np.where(ok & (sb > q1) & (sb < q3))[0]]):.2f}, most surprising quarter {np.mean([nrel[k] for k in np.where(ok & (sb >= q3))[0]]):.2f}")
# ---- (2) records: which surprise picks real relations, and new ones
def row(tag, sc, share=0.25):
    o = np.argsort(-sc + 1e-9 * rq.rand(len(sc))); k = int(len(sc) * share); keep = o[:k]
    kk = [i for i in keep if i in KNOWS and R[i]['gold']]; new = [i for i in kk if not KNOWS[i]]
    rel_keep = y[keep].sum(); newrate = len(new) / max(1, len(kk))
    P(f"   {tag:52s} AUC related {auc(sc[y], sc[~y]):.3f} | keep {share:.0%}: related {y[keep].mean():.3f}, recall {rel_keep / y.sum():.3f} | of kept related facts the trunk "
      f"does NOT know {newrate:.3f} | NEW true facts per 100 kept {100 * y[keep].mean() * newrate:.1f}")
P(f"(2)+(3) RECORDS ({len(R)}, related {y.mean():.3f}); keep a quarter by:")
rec = np.array([r['rec'] for r in R]); obj = np.array([r['obj'] for r in R]); mid = np.array([r['mid'] for r in R]); dist = -np.array([r['dist'] for r in R], float)
row('random', rq.rand(len(R)))
row('SURPRISE over the record', rec); row('SURPRISE on the second name', obj); row('SURPRISE on the words between', mid)
row('PREDICTABLE record (least surprise)', -rec); row('PREDICTABLE words between', -mid)
row('closeness of the names (no trunk)', dist)
rk = lambda x: np.argsort(np.argsort(x)) / len(x)
row('surprising name + predictable words between', rk(obj) + rk(-mid)); row('closeness + surprising second name', rk(dist) + rk(obj))
P(f"   all records kept: related {y.mean():.3f}, trunk does not know {1 - np.mean(kn_rel):.3f} of related -> NEW true facts per 100 kept {100 * y.mean() * (1 - np.mean(kn_rel)):.1f}")
P(f"DONE ({time.time() - T0:.0f}s)")
