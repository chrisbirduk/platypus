"""DO WRONG INFERENCES FADE WHILE RIGHT ONES STAY? (13 Nov 2026; tools/false_fade.py) - offline, no effect on the package.
GPU cg27a_rti visit 4: written inferences recalled from the trunk - RIGHT 0.09 vs WRONG 0.02 (visit 3: 0.07 vs 0.13): right ones retained, wrong
ones fading. Is it the reading after the write? The stand-in trunk (TG_CKPT) is taught 120 RIGHT inferences (A and C gold-related in a
Re-DocRED training document) and 120 WRONG ones (A with a same-type name from another document, never related) as 'A is connected to C.',
then trained on more text in two arms, everything else equal:
  EVIDENCE   the training documents that name each right pair (and each wrong inference's A, with its own true partners)
  OTHER      as many other Re-DocRED training documents, none naming the inferences' names
Recall (C first among C and 9 same-type names after 'It is on record that A is connected to ', bits per byte) for RIGHT and WRONG at the
end of the teaching and after 100 / 300 / 600 steps of reading. Two seeds.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/false_fade.py"""
import os
import sys
import json
import collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], '6000', '5']
_tg = open(os.path.join(HERE, 'trunk_guides.py')).read()
exec(compile(_tg[:_tg.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'))
docs = json.load(open(SCR + '/bench_dl/train_revised.json'))
cn_ = lambda d: [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
tp_ = lambda d: [vs[0].get('type', 'X') for vs in d['vertexSet']]
byt = collections.defaultdict(set); ndoc = collections.defaultdict(set); REL = set()
for di, d in enumerate(docs):
    cn, tp = cn_(d), tp_(d)
    for n, t in zip(cn, tp): byt[t].add(n); ndoc[n].add(di)
    for l in d['labels']: REL.add(frozenset((cn[l['h']], cn[l['t']])))
dtext = lambda di: ' '.join(' '.join(s) for s in docs[di]['sents']).encode('utf-8', 'replace')
for seed in (0, 1):
    r = np.random.RandomState(seed); right, wrong = [], []
    for di in r.permutation(len(docs)):
        d = docs[int(di)]; cn, tp = cn_(d), tp_(d)
        for l in d['labels']:
            A, Cn = cn[l['h']], cn[l['t']]
            if len(right) < 120 and 3 <= len(A) <= 30 and 3 <= len(Cn) <= 30 and len(ndoc[A]) <= 5 and all(A != x[0] for x in right):
                right.append((A, Cn, tp[l['t']], int(di))); break
        if len(right) >= 120: break
    used = {x[0] for x in right}
    for di in r.permutation(len(docs)):
        d = docs[int(di)]; cn, tp = cn_(d), tp_(d)
        for k, A in enumerate(cn):
            if len(wrong) >= 120: break
            if A in used or not 3 <= len(A) <= 30 or len(ndoc[A]) > 5: continue
            pool = [n for n in byt[tp[k]] if not ndoc[n] & ndoc[A] and frozenset((A, n)) not in REL and 3 <= len(n) <= 30]
            if pool: wrong.append((A, pool[r.randint(len(pool))], tp[k], int(di))); used.add(A); break
        if len(wrong) >= 120: break
    names = {x[0] for x in right + wrong} | {x[1] for x in right + wrong}
    ev = sorted({x[3] for x in right + wrong} | {dd for x in right for dd in ndoc[x[0]] & ndoc[x[1]]})
    oth = [di for di in r.permutation(len(docs)) if not any(n in names for n in cn_(docs[int(di)]))][:len(ev)]
    EVT = np.frombuffer(b' '.join(dtext(di) for di in ev), dtype=np.uint8); OTT = np.frombuffer(b' '.join(dtext(int(di)) for di in oth), dtype=np.uint8)
    facts = [f"{A} is connected to {Cn}.".encode('utf-8', 'replace') for A, Cn, t, di in right + wrong]
    @torch.no_grad()
    def recall(m, items):
        m.eval(); hit = []
        for A, Cn, t, di in items:
            cands = [Cn] + [n for n in r.permutation(sorted(byt[t]))[:30] if n != Cn and 3 <= len(n) <= 30][:9]; pre = f"It is on record that {A} is connected to ".encode('utf-8', 'replace')
            seqs = [pre + c.encode('utf-8', 'replace') + b'.' for c in cands]; X = torch.full((len(seqs), L), PAD, dtype=torch.long)
            for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
            Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1); nl = -F.log_softmax(m(Xi), -1).gather(-1, X[..., None])[..., 0]
            sc = [float(nl[k, len(pre):min(L, len(q)) - 1].mean()) for k, q in enumerate(seqs)]; hit.append(int(np.argmin(sc) == 0))
        m.train(); return float(np.mean(hit))
    def batch(src, nfact, rr):
        seqs = [facts[rr.randint(len(facts))] for _ in range(nfact)]
        for _ in range(32 - nfact): i0 = rr.randint(0, len(src) - L - 1); seqs.append(bytes(src[i0:i0 + L]))
        X = torch.full((32, L), PAD, dtype=torch.long)
        for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
        return X
    def step(m, op, X):
        Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); lo = m(Xi).reshape(-1, V); tg_ = X.reshape(-1)
        l = F.cross_entropy(lo[tg_ != PAD], tg_[tg_ != PAD]); op.zero_grad(); l.backward(); op.step()
    m0 = LM(); m0.load_state_dict(torch.load(CK)); m0.train(); op = torch.optim.AdamW(m0.parameters(), lr=3e-4, weight_decay=0.01); rr = np.random.RandomState(seed + 5)
    P(f"seed {seed}: {len(right)} right, {len(wrong)} wrong inferences; evidence documents {len(ev)}, other documents {len(oth)} | before teaching: right {recall(m0, right):.2f}, wrong {recall(m0, wrong):.2f}")
    for s_ in range(300): step(m0, op, batch(OTT, 8, rr))                       # teach the inferences (8 fact rows + 24 ordinary text windows)
    st = {k: v.clone() for k, v in m0.state_dict().items()}
    P(f"   taught (300 steps): right {recall(m0, right):.2f}, wrong {recall(m0, wrong):.2f}  ({time.time() - T0:.0f}s)")
    for arm, src in (('EVIDENCE', EVT), ('OTHER', OTT)):
        m = LM(); m.load_state_dict(st); m.train(); op2 = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); rr2 = np.random.RandomState(seed + 9); line = []
        for s_ in range(1, 601):
            step(m, op2, batch(src, 0, rr2))
            if s_ in (100, 300, 600): line.append(f"{s_}: right {recall(m, right):.2f} / wrong {recall(m, wrong):.2f}")
        P(f"   reading {arm:8s} - " + " | ".join(line) + f"  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
