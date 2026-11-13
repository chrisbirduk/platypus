"""OVERLAPPING REPLAY (13 Nov 2026; tools/overlap_replay.py) - offline, no effect on the package.
Lewis & Durrant 2011 (overlapping replay during sleep builds schemas); Ellenbogen et al. 2007 (sleep helps relational inference). Does
replaying memories that SHARE an element together teach the trunk relational structure? The stand-in trunk (TG_CKPT) is trained further, from
the same start, on Re-DocRED TRAINING sentences (the held-out fifth of tools/relation_free.py never used) in two arms with the same steps:
  OVERLAP   two sentence windows from DIFFERENT documents that name the same entity, read one after the other
  RANDOM    two random sentence windows (each around some name), read one after the other
Each arm's trunk is saved (OR_OUT_<arm>.pt) and scored by tools/relation_free.py's trunk row (does the trunk's state over a name-masked record
tell which relation it expresses? start: 0.58) and by held-out bits per byte (no damage).
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/overlap_replay.py [steps, default 1500]"""
import os
import sys
import json
import collections
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS_ = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
sys.argv = [sys.argv[0], '6000', '5']
_tg = open(os.path.join(HERE, 'trunk_guides.py')).read()
exec(compile(_tg[:_tg.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'))
docs = json.load(open(SCR + '/bench_dl/train_revised.json')) + json.load(open(SCR + '/bench_dl/dev_revised.json'))
TEST = set(np.random.RandomState(0).permutation(len(docs))[:len(docs) // 5].tolist())
WIN = collections.defaultdict(list); ALLW = []                                   # entity name -> windows (doc, bytes) around its mentions
HELD = []
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    if di in TEST:
        HELD.append(b' '.join(' '.join(s).encode('utf-8', 'replace') for s in d['sents'][:4])); continue
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs:
            toks = d['sents'][m['sent_id']]; a, b = m['pos']
            w = ' '.join(toks[max(0, a - 6):b + 6]).encode('utf-8', 'replace')[:60]
            WIN[cn[ei]].append((di, w)); ALLW.append(w)
shared = [n for n, v in WIN.items() if len({d_ for d_, _ in v}) >= 2]
P(f"windows {len(ALLW)}, names in 2+ training documents {len(shared)} | held-out passages {len(HELD)}")
held = np.frombuffer(b' '.join(HELD[:300]), dtype=np.uint8)
@torch.no_grad()
def heldbits(m):
    m.eval(); tot = 0.0; n = 0
    for i in range(0, len(held) - L, L):
        X = torch.tensor(held[i:i + L].astype(np.int64))[None]; Xi = torch.cat([torch.full((1, 1), PAD), X[:, :-1]], 1)
        tot += float(F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1), reduction='sum')); n += L
    m.train(); return tot / n / math.log(2)
P(f"start: held-out {heldbits(lm):.3f} bits per byte")
for arm in ('OVERLAP', 'RANDOM'):
    m = LM(); m.load_state_dict(torch.load(CK)); m.train(); op = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); r = np.random.RandomState(7); ls = []
    for st in range(STEPS_):
        seqs = []
        for _ in range(32):
            if arm == 'OVERLAP':
                n = shared[r.randint(len(shared))]; v = WIN[n]; i = r.randint(len(v)); j = r.randint(len(v))
                tries = 0
                while v[j][0] == v[i][0] and tries < 10: j = r.randint(len(v)); tries += 1
                s = v[i][1] + b' ... ' + v[j][1]
            else:
                s = ALLW[r.randint(len(ALLW))] + b' ... ' + ALLW[r.randint(len(ALLW))]
            seqs.append(s[:L])
        X = torch.full((32, L), PAD, dtype=torch.long)
        for k, s in enumerate(seqs): X[k, :len(s)] = torch.tensor(list(s))
        Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); lo = m(Xi).reshape(-1, V); tg_ = X.reshape(-1)
        l = F.cross_entropy(lo[tg_ != PAD], tg_[tg_ != PAD]); op.zero_grad(); l.backward(); op.step(); ls.append(l.item())
        if (st + 1) % 500 == 0: P(f"   {arm} step {st + 1}: train {np.mean(ls[-200:]) / math.log(2):.3f} bits per byte  ({time.time() - T0:.0f}s)")
    out = os.environ.get('OR_OUT', SCR + '/or') + f"_{arm.lower()}.pt"; torch.save(m.state_dict(), out)
    P(f"{arm}: held-out {heldbits(m):.3f} bits per byte | saved {out}")
P(f"DONE ({time.time() - T0:.0f}s)")
