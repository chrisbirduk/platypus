"""SCHEMA-CONGRUENT FACTS CONSOLIDATE FASTER? (13 Nov 2026; tools/schema_consol.py) - offline, no effect on the package.
Tse et al. 2007: new associations that fit an existing schema become cortical (hippocampus-independent) fast; SLIMM (van Kesteren et al. 2012):
congruent input is learned by the cortex directly. Does the same hold for the trunk? The stand-in trunk (TG_CKPT) learns 100 new facts
'<invented person> was born in <place>.' mixed into ordinary Re-DocRED text, in two arms with everything else equal:
  CONGRUENT     the place is a real place name that follows 'born in' in the text the trunk has read (fits what it predicts)
  INCONGRUENT   the place is a real name that never follows 'born in' (organisations, works: violates what it predicts)
Measured at steps 25 / 50 / 100 / 200 / 400: fact recall (the true object first among 10 of its arm's objects, after 'X was born in'), the
object's bits, and held-out bits per byte on other text (interference). Two seeds of the assignment.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/schema_consol.py"""
import os
import re
import sys
import json
import collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], '6000', '5']
_tg = open(os.path.join(HERE, 'trunk_guides.py')).read()
exec(compile(_tg[:_tg.index("for D in DOCS: D['S'], D['H'] = run(lm, D['b'])")], 'trunk_guides', 'exec'))
docs = json.load(open(SCR + '/bench_dl/dev_revised.json'))                    # the text the stand-in has read
after = collections.Counter(); every = collections.Counter(); typ = {}
for d in docs:
    for vs in d['vertexSet']:
        for m in vs:
            toks = d['sents'][m['sent_id']]; a, b = m['pos']; n = ' '.join(toks[a:b]); every[n] += 1; typ[n] = m.get('type', 'X')
            if a >= 2 and toks[a - 2].lower() == 'born' and toks[a - 1] == 'in': after[n] += 1
cong = [n for n, c in after.most_common() if typ.get(n) == 'LOC' and 4 <= len(n) <= 20][:60]
inc = [n for n, c in every.most_common() if typ.get(n) in ('ORG', 'MISC') and n not in after and 4 <= len(n) <= 20 and not any(ch.isdigit() for ch in n)][:60]
SY = ['ba', 'ko', 'ri', 'tel', 'man', 'vor', 'su', 'ne', 'dal', 'gri', 'po', 'lin', 'ka', 'mer', 'tu', 'zen', 'ho', 'fi', 'rad', 'ul']
text = np.frombuffer(b' '.join(' '.join(' '.join(s) for s in d['sents']).encode('utf-8', 'replace') for d in docs[:300]), dtype=np.uint8)
held = np.frombuffer(b' '.join(' '.join(' '.join(s) for s in d['sents']).encode('utf-8', 'replace') for d in docs[300:360]), dtype=np.uint8)
P(f"congruent places {len(cong)} (e.g. {cong[:5]}), incongruent names {len(inc)} (e.g. {inc[:5]})")
@torch.no_grad()
def heldbits(m):
    m.eval(); tot = 0.0; n = 0
    for i in range(0, min(len(held), 60 * L) - L, L):
        X = torch.tensor(held[i:i + L].astype(np.int64))[None]; Xi = torch.cat([torch.full((1, 1), PAD), X[:, :-1]], 1)
        tot += float(F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1), reduction='sum')); n += L
    m.train(); return tot / n / math.log(2)
@torch.no_grad()
def recall(m, facts, pool):
    m.eval(); hits = []; bits_ = []
    for s, o in facts:
        cands = [o] + [c for c in pool if c != o][:9]; seqs = [f"{s} was born in {c}.".encode() for c in cands]; pre = len(f"{s} was born in ".encode())
        X = torch.full((len(seqs), L), PAD, dtype=torch.long)
        for k, q in enumerate(seqs): X[k, :len(q)] = torch.tensor(list(q))
        Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(m(Xi), -1)
        nl = -lp.gather(-1, X[..., None])[..., 0] / math.log(2)
        sc = [float(nl[k, pre:len(q) - 1].mean()) for k, q in enumerate(seqs)]; hits.append(int(np.argmin(sc) == 0)); bits_.append(sc[0])
    m.train(); return float(np.mean(hits)), float(np.mean(bits_))
for seed in (0, 1):
    r = np.random.RandomState(seed)
    people = list({(''.join(r.choice(SY) for _ in range(2)).capitalize() + ' ' + ''.join(r.choice(SY) for _ in range(3)).capitalize()) for _ in range(400)})[:200]
    for arm, pool in (('CONGRUENT', cong), ('INCONGRUENT', inc)):
        facts = [(people[i + (0 if arm == 'CONGRUENT' else 100)], pool[r.randint(len(pool))]) for i in range(100)]
        fb = [f"{s} was born in {o}.".encode() for s, o in facts]
        m = LM(); m.load_state_dict(torch.load(CK)); m.train(); op = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); rr = np.random.RandomState(seed + 9)
        a0, b0 = recall(m, facts, pool); line = [f"step 0: recall {a0:.2f}, object bits {b0:.2f}, held-out {heldbits(m):.3f}"]
        for st in range(1, 401):
            seqs = [fb[rr.randint(len(fb))] for _ in range(8)]                    # 8 fact rows + 24 ordinary text windows per step
            for _ in range(24): i0 = rr.randint(0, len(text) - L - 1); seqs.append(bytes(text[i0:i0 + L]))
            X = torch.full((32, L), PAD, dtype=torch.long)
            for k, q in enumerate(seqs): X[k, :len(q)] = torch.tensor(list(q[:L]))
            Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); lo = m(Xi).reshape(-1, V); tg_ = X.reshape(-1)
            l = F.cross_entropy(lo[tg_ != PAD], tg_[tg_ != PAD]); op.zero_grad(); l.backward(); op.step()
            if st in (25, 50, 100, 200, 400):
                a, b = recall(m, facts, pool); line.append(f"{st}: {a:.2f} / {b:.2f} / {heldbits(m):.3f}")
        P(f"seed {seed} {arm:11s} recall / object bits / held-out bits per byte - " + " | ".join(line) + f"  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
