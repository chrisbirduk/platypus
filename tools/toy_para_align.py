"""TOY PARAPHRASE ALIGNMENT (12 Nov 2026; tools/toy_para_align.py) - the weak point after replay: a question with a PARAPHRASED relation
('A is a colleague of') lands far from the stored key ('A works with': GPU cosine ~0.65 vs ~0.88 for a new frame). Small causal transformer
as tools/toy_align2.py. Each relation has two phrasings (tokens ra, rb); every fact is declared 'A ra B .'; 20% are ALSO declared 'A rb B .'
(the evidence a relational module merges paraphrases from); 70% taught as questions '<prefix> A ra ?' -> B (7 prefixes). Test: unseen
prefixes, question with ra and with the PARAPHRASE rb, for taught and never-taught facts.
  off          no alignment
  as built     question (ra) -> declaration 'A ra' state, taught facts
  + para       + question in the paraphrase (rb) -> the same 'A ra' declaration state, taught facts
  + para, all  + the same for the NEVER-ASKED facts too (label-free)
Usage: python3 tools/toy_para_align.py [seed] [steps]"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.argv = [sys.argv[0], 'X'] + sys.argv[1:]                                 # reuse toy_align2's model and training (no tests run: 'X')
HERE = os.path.dirname(os.path.abspath(__file__))
exec(compile(open(os.path.join(HERE, 'toy_align2.py')).read(), 'toy_align2', 'exec'))
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
rs = np.random.RandomState(seed + 11)
NRB = NR // 2                                                                 # 3 relations x 2 phrasings (REL(i), REL(i + 3))
RA = lambda i: REL(i); RB = lambda i: REL(i + NRB)
facts = [(ENT(rs.randint(NE)), rs.randint(NRB), ENT(rs.randint(NE))) for _ in range(600)]
taught = facts[:420]; un = facts[420:]; both = set(rs.choice(600, 120, replace=False))
mk2 = lambda: [WRD(w) for w in rs.choice(NW, rs.randint(1, 4), replace=False)]
pref2 = [mk2() for _ in range(7)]; tp = [mk2() for _ in range(10)]
seqs = [[A, RA(r), B, DOT] for A, r, B in facts] + [[A, RB(r), B, DOT] for i, (A, r, B) in enumerate(facts) if i in both]
seqs += [p + [A, RA(r), QM, B] for A, r, B in taught for p in pref2]
al_b = [(p + [A, RA(r), QM], [A, RA(r), B, DOT]) for A, r, B in taught for p in pref2]
al_p = [(p + [A, RB(r), QM], [A, RA(r), B, DOT]) for A, r, B in taught for p in pref2]
al_pu = [(p + [A, RB(r), QM], [A, RA(r), B, DOT]) for A, r, B in un for p in pref2] + [(p + [A, RA(r), QM], [A, RA(r), B, DOT]) for A, r, B in un for p in pref2]
q = lambda F_, R_: ([p + [A, R_(r), QM] for A, r, B in F_ for p in tp], [B for A, r, B in F_ for p in tp])
P(f"PARAPHRASE: 600 facts, 3 relations x 2 phrasings, 20% also declared in the paraphrase, 420 taught (phrasing a only)")
for tag, al in (('off', None), ('as built', al_b), ('+ para', al_b + al_p), ('+ para, all (label-free)', al_b + al_p + al_pu)):
    m = train(seqs, [1.0] * len(seqs), STEPS, seed + 6, al)
    P(f"   {tag}: TAUGHT phrasing a {acc(m, *q(taught, RA)):.2f}, PARAPHRASE b {acc(m, *q(taught, RB)):.2f} | NEVER TAUGHT phrasing a "
      f"{acc(m, *q(un, RA)):.2f}, PARAPHRASE b {acc(m, *q(un, RB)):.2f}")
P("DONE")
