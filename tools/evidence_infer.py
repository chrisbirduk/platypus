"""DOES READING THE PREMISES MAKE A NOVEL INFERENCE STICK? (14 Nov 2026; tools/evidence_infer.py) - offline, no effect on the package.
The offline analogue of GPU cg27a_rti2 (evidence replay). Chains from Re-DocRED training documents (gold, used to build the test only):
A-B gold-related in one document, B-C gold-related in another, A and C NEVER in the same document anywhere (novel pairs). Against them,
UNSUPPORTED pairs A-C' (C' of C's type, no chain, never together with A). The stand-in trunk (TG_CKPT), three arms, everything else equal:
  READ      the evidence documents only (the chain's two documents; for an unsupported pair, A's own documents): does reading the premises
            alone raise recall of C for A? (latent two-hop from reading; cg27a_rti2's held-back arm)
  TEACH+EV  taught 'A is connected to C.' (supported and unsupported alike), then the evidence documents (cg27a_rti2's taught arm)
  TEACH+OT  taught, then as many other documents naming none of the names (cg27a_rti's taught arm)
Recall: C first among C and 9 same-type names, in two wordings never taught ('It is on record that A is connected to ' and 'Records
show a link between A and '), before, after teaching and after 100 / 300 / 600 steps of reading. Two seeds.
v2 (14 Nov): the unsupported pairs' evidence includes C's own documents (C equally familiar; v1 read only A's, so reading could favour
supported C by familiarity alone), and a third measure, the GAIN FROM A (C's bits after 'It is on record that someone is connected to '
minus after the A wording, ranked among the 10 names), which familiarity cannot move.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/evidence_infer.py"""
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
byt = collections.defaultdict(set); ndoc = collections.defaultdict(set); ntype = {}; GR = collections.defaultdict(list)
for di, d in enumerate(docs):
    cn, tp = cn_(d), tp_(d)
    for n, t in zip(cn, tp): byt[t].add(n); ndoc[n].add(di); ntype[n] = t
    for l in d['labels']:
        a, b = cn[l['h']], cn[l['t']]
        if a != b: GR[a].append((b, di)); GR[b].append((a, di))
ok = lambda n: 3 <= len(n) <= 30 and len(ndoc[n]) <= 5
dtext = lambda di: ' '.join(' '.join(s) for s in docs[di]['sents']).encode('utf-8', 'replace')
WORD = [("It is on record that {A} is connected to ", ""), ("Records show a link between {A} and ", "")]
NEUT = "It is on record that someone is connected to "                     # v2: GAIN FROM A = C's bits after NEUT minus after WORD[0], ranked among
                                                                             #   the 10 names (familiarity of C cancels)
for seed in (0, 1):
    r = np.random.RandomState(seed); sup, uns, used = [], [], set()
    names = [n for n in r.permutation(sorted(GR)) if ok(n)]
    for A in names:                                                            # SUPPORTED: A-B (d1), B-C (d2), A and C never together
        if len(sup) >= 120: break
        if A in used: continue
        for k_ in r.permutation(len(GR[A])):
            B, d1 = GR[A][int(k_)]
            if len(ndoc[B]) > 30: continue
            cs = [(C, d2) for C, d2 in GR[B] if d2 != d1 and C != A and ok(C) and not ndoc[A] & ndoc[C] and C not in used]
            if cs:
                C, d2 = cs[r.randint(len(cs))]; sup.append((A, C, ntype.get(C, 'X'), (d1, d2))); used |= {A, C}; break
    for A in names:                                                            # UNSUPPORTED: A-C' of a supported C's type, no chain
        if len(uns) >= 120: break
        if A in used: continue
        t = sup[len(uns) % len(sup)][2]
        reach = {x for B, _ in GR[A] for x, _ in GR[B]}
        pool = [n for n in byt[t] if ok(n) and n != A and n not in used and n not in reach and not ndoc[A] & ndoc[n]]
        if pool:                                                               # v2 (14 Nov): C's own documents are read too, so C is as familiar as a
            C = pool[r.randint(len(pool))]; uns.append((A, C, t, tuple(sorted(ndoc[A]))[:2] + tuple(sorted(ndoc[C]))[:2])); used |= {A, C}   # supported C

    allnames = {x[0] for x in sup + uns} | {x[1] for x in sup + uns}
    ev = sorted({d for x in sup + uns for d in x[3]})
    oth = [di for di in r.permutation(len(docs)) if not any(n in allnames for n in cn_(docs[int(di)]))][:len(ev)]
    EVT = np.frombuffer(b' '.join(dtext(di) for di in ev), dtype=np.uint8); OTT = np.frombuffer(b' '.join(dtext(int(di)) for di in oth), dtype=np.uint8)
    facts = [f"{A} is connected to {C}.".encode('utf-8', 'replace') for A, C, t, _ in sup + uns]
    CANDS = {}
    for A, C, t, _ in sup + uns: CANDS[(A, C)] = [C] + [n for n in r.permutation(sorted(byt[t]))[:40] if n != C and n != A and ok(n)][:9]
    SC0 = {}
    @torch.no_grad()
    def recall(m, items):
        m.eval(); out = []
        for w, _ in WORD:
            hit = []
            for A, C, t, _ in items:
                pre = w.format(A=A).encode('utf-8', 'replace'); seqs = [pre + c.encode('utf-8', 'replace') + b'.' for c in CANDS[(A, C)]]
                X = torch.full((len(seqs), L), PAD, dtype=torch.long)
                for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
                Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1); nl = -F.log_softmax(m(Xi), -1).gather(-1, X[..., None])[..., 0]
                sc = [float(nl[k, len(pre):min(L, len(q)) - 1].mean()) for k, q in enumerate(seqs)]; hit.append(int(np.argmin(sc) == 0))
                if w == WORD[0][0]: SC0[(A, C)] = sc
            out.append(float(np.mean(hit)))
        hit = []
        for A, C, t, _ in items:                                               # GAIN FROM A
            pre = NEUT.encode('utf-8', 'replace'); seqs = [pre + c.encode('utf-8', 'replace') + b'.' for c in CANDS[(A, C)]]
            X = torch.full((len(seqs), L), PAD, dtype=torch.long)
            for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
            Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1); nl = -F.log_softmax(m(Xi), -1).gather(-1, X[..., None])[..., 0]
            g0 = np.array([float(nl[k, len(pre):min(L, len(q)) - 1].mean()) for k, q in enumerate(seqs)]) - np.array(SC0[(A, C)])
            hit.append(int(np.argmax(g0) == 0))
        out.append(float(np.mean(hit)))
        m.train(); return out
    fmt = lambda s, u: f"supported {s[0]:.2f}/{s[1]:.2f} gain {s[2]:.2f}, unsupported {u[0]:.2f}/{u[1]:.2f} gain {u[2]:.2f}"
    def batch(src, nfact, rr):
        seqs = [facts[rr.randint(len(facts))] for _ in range(nfact)]
        for _ in range(32 - nfact): i0 = rr.randint(0, len(src) - L - 1); seqs.append(bytes(src[i0:i0 + L]))
        X = torch.full((32, L), PAD, dtype=torch.long)
        for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
        return X
    def step(m, op, X):
        Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); lo = m(Xi).reshape(-1, V); tg_ = X.reshape(-1)
        l = F.cross_entropy(lo[tg_ != PAD], tg_[tg_ != PAD]); op.zero_grad(); l.backward(); op.step()
    def read(st, src, tag):
        m = LM(); m.load_state_dict(st); m.train(); op2 = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); rr2 = np.random.RandomState(seed + 9); line = []
        for s_ in range(1, 601):
            step(m, op2, batch(src, 0, rr2))
            if s_ in (100, 300, 600): line.append(f"{s_}: " + fmt(recall(m, sup), recall(m, uns)))
        P(f"   {tag:9s} - " + " | ".join(line) + f"  ({time.time() - T0:.0f}s)")
    m0 = LM(); m0.load_state_dict(torch.load(CK)); m0.train(); st0 = {k: v.clone() for k, v in m0.state_dict().items()}
    P(f"seed {seed}: {len(sup)} supported (A-B, B-C in two documents, A and C never together), {len(uns)} unsupported; evidence documents {len(ev)}, "
      f"other {len(oth)} | recall in two untaught wordings, chance 0.10 | before: " + fmt(recall(m0, sup), recall(m0, uns)))
    read(st0, EVT, 'READ')
    op = torch.optim.AdamW(m0.parameters(), lr=3e-4, weight_decay=0.01); rr = np.random.RandomState(seed + 5)
    for s_ in range(300): step(m0, op, batch(OTT, 8, rr))
    st = {k: v.clone() for k, v in m0.state_dict().items()}
    P(f"   taught (300 steps): " + fmt(recall(m0, sup), recall(m0, uns)) + f"  ({time.time() - T0:.0f}s)")
    read(st, EVT, 'TEACH+EV'); read(st, OTT, 'TEACH+OT')
P(f"DONE ({time.time() - T0:.0f}s)")
