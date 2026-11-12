"""TOY 2b (12 Nov; tools/toy_rules_constrain.py): toy_rules.py + RULE CONSTRAINTS (after RiJEPA, 2026): once a rule r1 o r2 -> r is
discovered, the operators are refitted with an energy term pulling P_r toward P_r1 P_r2 - does the composite operator then transfer
to new families directly (0.00 without)? The original description follows.

TOY 2: rule-free inference that TRANSFERS to new entities (11 Nov 2026; tools/toy_rules.py) - step 2 of the JEPA rule test.

tools/toy_jop.py held out composite facts about people whose other facts were all known. Here whole FAMILIES and ORG CHARTS are new at test
time: only their one-hop facts (child-of, supervised-by) are ever seen. The system must (1) DISCOVER the rule from the training families,
with no rule given - grandchild = child o child, overseen = supervised o supervised - by finding which single relation operator the product
of two operators matches, judged against a NULL of fits on shuffled facts; then (2) APPLY it to the new families: compose the operators on
a new person's code, snap to the nearest entity, and compare with their true grandparent / skip level. Also: the learned composite
operator applied directly to new people; a ONE-TO-MANY relation (works_with, 3 colleagues each, symmetric - colleague-of-colleague is not
a relation, so no rule may be found for it); and NOISE (a share of child-of facts pointing to a wrong parent).
Usage: python3 tools/toy_rules.py [seed] [steps] [noise]"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
NOISE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
torch.set_num_threads(1); torch.manual_seed(seed); rs = np.random.RandomState(seed)
NF, D = 40, 64                                                                # 40 families and 40 org charts, 7 people each
ents = 0; facts = []; groups = []                                             # facts: (rel, a, b, group, train?)
for kind, r1, r2 in (('fam', 'child', 'grand'), ('org', 'sup', 'over')):
    for f in range(NF):
        root = ents; mids = [ents + 1, ents + 2]; leaves = [ents + 3, ents + 4, ents + 5, ents + 6]; ents += 7
        train = f < int(NF * 0.7); groups.append((kind, list(range(root, root + 7)), train))
        for m in mids: facts.append((r1, m, root, train))
        for i, l in enumerate(leaves): facts.append((r1, l, mids[i // 2], train))
        for i, l in enumerate(leaves): facts.append((r2, l, root, train))
N = ents
for a in range(N):                                                            # one-to-many, symmetric: 3 colleagues each
    for b in rs.choice(N, 3, replace=False):
        if b != a: facts.append(('works_with', a, int(b), True))
if NOISE > 0:                                                                 # wrong parents on a share of child-of facts
    for k, (r, a, b, t) in enumerate(facts):
        if r == 'child' and rs.rand() < NOISE: facts[k] = (r, a, int(rs.randint(N)), t)
R = ['child', 'grand', 'sup', 'over', 'works_with']
train_rows = [(r, a, b) for r, a, b, t in facts if t or r in ('child', 'sup', 'works_with')]   # new groups: one-hop facts only
test = {r2: [(a, b) for r, a, b, t in facts if r == r2 and not t] for r2 in ('grand', 'over')}


def fit(rows, steps=STEPS, sd=0, cons=(), lam=1.0):
    g = torch.Generator().manual_seed(seed * 100 + sd)
    E = torch.nn.Parameter(torch.randn(N, D, generator=g) * 0.1)
    P = torch.nn.ParameterDict({r: torch.nn.Parameter(torch.eye(D) + 0.01 * torch.randn(D, D, generator=g)) for r in R})
    opt = torch.optim.Adam([E] + list(P.values()), lr=3e-3); rr = np.random.RandomState(seed * 100 + sd)
    for _ in range(steps):
        b = [rows[i] for i in rr.randint(len(rows), size=256)]; z = F.normalize(E, dim=-1); loss = 0.0
        for r in R:
            br = [(a, c) for q, a, c in b if q == r]
            if not br: continue
            a = torch.tensor([x for x, _ in br]); c = torch.tensor([y for _, y in br])
            loss = loss + F.cross_entropy(F.normalize(z[a] @ P[r], dim=-1) @ z.t() / 0.1, c, reduction='sum')
        zc = z - z.mean(0); cov = zc.t() @ zc / N
        loss = loss / 256 + 0.1 * ((cov * N / D - torch.eye(D)) ** 2).mean()
        for r1, r2, r in cons:                                                 # RULE CONSTRAINT: P_r ~ P_r1 P_r2 on the entities' codes
            M = z @ P[r1] @ P[r2]; Q = z @ P[r]
            loss = loss + lam * ((F.normalize(M, dim=-1) - F.normalize(Q, dim=-1)) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return F.normalize(E.detach(), dim=-1), {r: P[r].detach() for r in R}


def products(P):
    out = {}
    for r1 in R:
        for r2 in R:
            M = P[r1] @ P[r2]; out[(r1, r2)] = {r: float(torch.norm(P[r] - M) / torch.norm(M)) for r in R}
    return out


def shuffled(rows, sd):
    rr = np.random.RandomState(1000 + sd); by = {}
    for k, (r, a, b) in enumerate(rows): by.setdefault(r, []).append(k)
    out = list(rows)
    for r, ks in by.items():
        tg = [rows[k][2] for k in ks]; rr.shuffle(tg)
        for k, t in zip(ks, tg): out[k] = (r, rows[k][1], t)
    return out


E, P = fit(train_rows)
pr = products(P)
nulls = [products(fit(shuffled(train_rows, s), sd=s + 1)[1]) for s in range(3)]
print(f"== seed {seed}, {STEPS} steps, noise {NOISE:g} | {N} entities, {len(train_rows)} training facts | new groups: {sum(1 for g in groups if not g[2])} "
      f"(only their one-hop facts seen) | chance {1 / N:.4f}")
print("   RULE DISCOVERY: for each product r1 o r2, the nearest single relation (relative distance) and the same distance in 3 shuffled-fact nulls")
found = {}
for (r1, r2), d in sorted(pr.items()):
    if r1 != r2 and not (r1, r2) in (('child', 'sup'),): continue
    best = min(d, key=d.get); nd = [n[(r1, r2)][best] for n in nulls]; z = (np.mean(nd) - d[best]) / (np.std(nd) + 1e-6)
    sig = d[best] < min(nd) - 0.05 and best not in (r1, r2)
    print(f"     {r1:10s} o {r2:10s} -> {best:10s} {d[best]:.2f} | null {np.mean(nd):.2f} (min {min(nd):.2f}) | z {z:+.1f} | {'RULE' if sig else 'no rule'}")
    if sig: found[best] = (r1, r2)
acc = lambda pred, tgt: float((F.normalize(pred, dim=-1) @ E.t()).argmax(-1).eq(tgt).float().mean())
for r2 in ('grand', 'over'):
    a = torch.tensor([x for x, _ in test[r2]]); c = torch.tensor([y for _, y in test[r2]]); r1 = 'child' if r2 == 'grand' else 'sup'
    snap = E[(F.normalize(E[a] @ P[r1], dim=-1) @ E.t()).argmax(-1)] @ P[r1]
    use = found.get(r2)
    print(f"   NEW GROUPS, {r2} (n {len(a)}): discovered rule {use[0] + ' o ' + use[1] if use else 'none'} applied (compose, snap, compose) "
          f"{acc(snap, c) if use else float('nan'):.2f} | composed without snap {acc(E[a] @ P[r1] @ P[r1], c):.2f} | learned {r2} operator directly {acc(E[a] @ P[r2], c):.2f}")

cons = [(r1, r2, r) for r, (r1, r2) in found.items()]
for lam in (0.3, 1.0, 3.0):
    E, P = fit(train_rows, cons=cons, lam=lam)
    print(f"   RULE CONSTRAINTS {cons}, weight {lam:g}:")
    for r2 in ('grand', 'over'):
        a = torch.tensor([x for x, _ in test[r2]]); c = torch.tensor([y for _, y in test[r2]]); r1 = 'child' if r2 == 'grand' else 'sup'
        print(f"     NEW GROUPS {r2} (n {len(a)}): learned {r2} operator directly {acc(E[a] @ P[r2], c):.2f} | composed {acc(E[a] @ P[r1] @ P[r1], c):.2f}")
