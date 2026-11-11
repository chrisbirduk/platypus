"""TOY: can a JEPA-style relation predictor learn INFERENCE STRUCTURE without rules? (11 Nov 2026; tools/toy_jop.py)

A world with real composition rules - a family tree (grandparent = parent o parent) and an org chart (skip-manager = manager o manager) -
plus random relations with no structure (lives_near, met). Entities get embeddings; each relation r a linear predictor P_r. Training is
JEPA-style: the predictor maps the source's embedding to the target's (InfoNCE over all entities, plus a SIGReg-like isotropy term).
Only HALF of the grandparent and skip-manager facts are trained; the other half is held out.

Tests (true answer ranked among all entities):
  T1  COMPOSED OPERATOR   P_parent P_parent e_A -> grandparent, on held-out people (no grandparent fact of theirs seen), and the same
                          with an intermediate snap to the nearest entity (two lookups)
  T2  LEARNED RELATION    P_grandparent on held-out people - does the trained rule-carrying relation generalise?
  T3  SCHEMA DISCOVERY    nearest single relation operator to P_r2 P_r1 for each relation pair: does parent o parent pick grandparent,
                          manager o manager skip_manager, and do random relations give no clear match?
  T4  TIED vs UNTIED      the same with separate source and target embeddings (as attention's keys vs values): the composed operator
                          then has to cross spaces
  T5  PATH TRAINING       T1/T2 when a few two-step paths are trained as well (compositional training, Guu et al. 2015)
Usage: python3 tools/toy_jop.py [seed] [steps]"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
torch.set_num_threads(1)
torch.manual_seed(seed); rs = np.random.RandomState(seed)
N_FAM, N_ORG, D = 300, 300, 64
# family: 5 generations of 60; each person in gen g>0 has one parent in gen g-1
par = {}
for g in range(1, 5):
    for i in range(60):
        par[g * 60 + i] = (g - 1) * 60 + int(rs.randint(60))
# org: a tree of 300 people (ids 300..599), each non-root has a manager with a lower index
mgr = {}
for i in range(1, N_ORG):
    mgr[N_FAM + i] = N_FAM + int(rs.randint(max(1, i // 2), i)) if i > 1 else N_FAM
N = N_FAM + N_ORG
R = ['parent', 'grandparent', 'manager', 'skip_manager', 'lives_near', 'met']
facts = {r: [] for r in R}
for c, p in par.items():
    facts['parent'].append((c, p))
    if p in par: facts['grandparent'].append((c, par[p]))
for e, m in mgr.items():
    facts['manager'].append((e, m))
    if m in mgr: facts['skip_manager'].append((e, mgr[m]))
for r in ('lives_near', 'met'):
    facts[r] = [(int(rs.randint(N)), int(rs.randint(N))) for _ in range(300)]
held = {}
for r in ('grandparent', 'skip_manager'):                                   # hold out half of the composite facts (by source)
    rs.shuffle(facts[r]); k = len(facts[r]) // 2; held[r] = facts[r][k:]; facts[r] = facts[r][:k]


def train(tied=True, paths=0, steps=None):
    steps = steps or STEPS
    E = torch.nn.Parameter(torch.randn(N, D) * 0.1)
    E2 = E if tied else torch.nn.Parameter(torch.randn(N, D) * 0.1)          # untied: separate target (value) embeddings
    P = torch.nn.ParameterDict({r: torch.nn.Parameter(torch.eye(D) + 0.01 * torch.randn(D, D)) for r in R})
    ps = [E] + ([] if tied else [E2]) + list(P.values())
    opt = torch.optim.Adam(ps, lr=3e-3)
    rows = [(r, a, b) for r in R for a, b in facts[r]]
    pth = []                                                                  # two-step paths: parent-parent and manager-manager chains
    if paths:
        for (c, p) in facts['parent']:
            if p in par: pth.append(('parent', 'parent', c, par[p]))
        for (e, m) in facts['manager']:
            if m in mgr: pth.append(('manager', 'manager', e, mgr[m]))
        held_src = {a for r in held for a, _ in held[r]}
        pth = [x for x in pth if x[2] not in held_src][:paths]
    for st in range(steps):
        b = [rows[i] for i in rs.randint(len(rows), size=256)]
        zs = F.normalize(E, dim=-1); zt = F.normalize(E2, dim=-1)
        loss = 0.0
        for r in R:
            br = [(a, c) for rr, a, c in b if rr == r]
            if not br: continue
            a = torch.tensor([x for x, _ in br]); c = torch.tensor([y for _, y in br])
            pred = F.normalize(zs[a] @ P[r], dim=-1)
            loss = loss + F.cross_entropy(pred @ zt.t() / 0.1, c, reduction='sum')
        if pth:
            bp = [pth[i] for i in rs.randint(len(pth), size=64)]
            for r1 in ('parent', 'manager'):
                bb = [x for x in bp if x[0] == r1]
                if not bb: continue
                a = torch.tensor([x[2] for x in bb]); c = torch.tensor([x[3] for x in bb])
                pred = F.normalize(F.normalize(zs[a] @ P[r1], dim=-1) @ P[r1], dim=-1) if tied else F.normalize(zs[a] @ P[r1] @ P[r1], dim=-1)
                loss = loss + F.cross_entropy(pred @ zt.t() / 0.1, c, reduction='sum')
        z = zs - zs.mean(0); cov = z.t() @ z / N                               # isotropy (SIGReg-like)
        loss = loss / 256 + 0.1 * ((cov * N / D - torch.eye(D)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return E.detach(), E2.detach(), {r: P[r].detach() for r in R}


def evaluate(E, E2, P, tag):
    zs = F.normalize(E, dim=-1); zt = F.normalize(E2, dim=-1)
    acc = lambda pred, tgt: float((F.normalize(pred, dim=-1) @ zt.t()).argmax(-1).eq(tgt).float().mean())
    out = {}
    for r_c, r_b in (('grandparent', 'parent'), ('skip_manager', 'manager')):
        a = torch.tensor([x for x, _ in held[r_c]]); c = torch.tensor([y for _, y in held[r_c]])
        out[r_c] = dict(
            composed=acc(zs[a] @ P[r_b] @ P[r_b], c),
            snapped=acc(zt[(F.normalize(zs[a] @ P[r_b], dim=-1) @ zt.t()).argmax(-1)] @ P[r_b] if False else zs[(F.normalize(zs[a] @ P[r_b], dim=-1) @ zt.t()).argmax(-1)] @ P[r_b], c),
            learned=acc(zs[a] @ P[r_c], c),
            trained_half=acc(zs[torch.tensor([x for x, _ in facts[r_c]])] @ P[r_c], torch.tensor([y for _, y in facts[r_c]])))
    one = {r: acc(zs[torch.tensor([x for x, _ in facts[r]])] @ P[r], torch.tensor([y for _, y in facts[r]])) for r in R}
    sch = []                                                                  # schema discovery: nearest single operator to each product
    for r1 in R:
        for r2 in R:
            M = P[r1] @ P[r2]; d = sorted(((float(torch.norm(P[r] - M) / torch.norm(M)), r) for r in R))
            sch.append((r1, r2, d[0][1], d[0][0], d[1][0]))
    print(f"== {tag} | chance {1 / N:.4f} | one-hop on trained facts: " + ", ".join(f"{r} {v:.2f}" for r, v in one.items()))
    for r_c in out:
        o = out[r_c]
        print(f"   {r_c:13s} HELD-OUT sources: composed operator {o['composed']:.2f} | two lookups (snap) {o['snapped']:.2f} | learned {r_c} operator {o['learned']:.2f} (on its trained half {o['trained_half']:.2f})")
    for r1, r2, best, d1, d2 in sch:
        if (r1, r2) in (('parent', 'parent'), ('manager', 'manager'), ('lives_near', 'lives_near'), ('met', 'lives_near'), ('parent', 'manager')):
            print(f"   SCHEMA {r1} o {r2} -> nearest {best} (relative distance {d1:.2f}, next {d2:.2f})")


for tied, paths, tag in ((True, 0, 'TIED (JEPA: one space)'), (False, 0, 'UNTIED (attention-like: keys vs values)'),
                         (True, 150, 'TIED + 150 trained two-step paths')):
    E, E2, P = train(tied=tied, paths=paths)
    evaluate(E, E2, P, tag)
    sys.stdout.flush()
