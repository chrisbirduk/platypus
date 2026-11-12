"""TOY: DOES REFUSAL TRANSFER, OR IS IT A SHORTCUT? (12 Nov 2026; tools/toy_refuse.py) - offline, no effect on the package.
GPU cg27a_refuse: question rows about fresh names answered 'unknown' were learned (loss 0.14) but the trunk refused only 3-5% of fresh names
in the untrained test wording. Suspects: (1) the refusal-row names come from their own syllables, so 'names that sound like this -> unknown'
is enough; (2) a handful of trained wordings, so refusal is tied to them. A small causal byte transformer reads fact sentences about invented
people ('Kemora works with Talvin.'), is taught question rows for 3/4 of the facts (the rest only read) and, per condition:
  none        no refusal rows (the control: what the GPU builds do today)
  distinct    refusal rows about fresh names from DIFFERENT syllables than the stored names (as cg27a_refuse)
  same        refusal rows about fresh names from the SAME syllables as the stored names
  same+forms  'same' with 12 question wordings instead of 4
Scored greedily in a HELD-OUT wording ('It is on record that A r ') and in a taught wording: refusal ('unknown') on fresh names from the
stored names' syllables (never read), on fresh names from a THIRD syllable set (as the GPU probe's), and on KNOWN facts (over-refusal);
known facts answered right; facts only read answered right (learning from reading).
Usage: python3 tools/toy_refuse.py [steps] [seed]"""
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
torch.set_num_threads(1); P = lambda *a: print(*a, flush=True); T0 = time.time()
L = 96; PAD = 256; V = 257
S1 = ['ke', 'mo', 'ra', 'tal', 'vin', 'so', 'ri', 'den', 'lu', 'fa', 'gor', 'pi', 'ne', 'zu', 'ba', 'lor']   # stored names
S2 = ['qua', 'xi', 'oth', 'wem', 'ji', 'yul', 'cro', 'phe', 'ux', 'sha', 'vro', 'kli']                   # 'distinct' refusal names
S3 = ['dov', 'ein', 'hap', 'tus', 'mek', 'bri', 'gal', 'nod', 'ost', 'wir', 'cav', 'sim']                 # third set (the probe's)
RELS = ['works with', 'lives near', 'reports to', 'trained under', 'is a partner of', 'is connected to']
F4 = ['Q: {A} {r} whom? A: ', 'In short, {A} {r} ', 'From the notes: {A} {r} ', 'As for {A}, people say {A} {r} ']
F12 = F4 + ['Remember: {A} {r} ', 'Fact: {A} {r} ', 'Plainly, {A} {r} ', 'Some say that {A} {r} ', 'According to the file, {A} {r} ',
            'It turns out {A} {r} ', '{A}, it seems, {r} ', 'Going back to {A}: {A} {r} ']
TEST = 'It is on record that {A} {r} '
assert all('on record' not in f for f in F12)
def mkname(r, S, taken):
    while True:
        n = ''.join(r.choice(S, r.randint(2, 4))).capitalize()
        if n not in taken: taken.add(n); return n
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        h = s.n1(x); x = x + s.a(h, h, h, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class LM(nn.Module):
    def __init__(s, d=128, nl=3):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(L, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl))
        s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V)
    def forward(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for b in s.b: x = b(x, am)
        return s.o(s.n(x))
def enc(t): b = list(t.encode())[:L]; return b + [PAD] * (L - len(b))
@torch.no_grad()
def greedy(m, prompts, n=10):
    out = []
    for i in range(0, len(prompts), 128):
        bs = [bytearray(p.encode()) for p in prompts[i:i + 128]]
        for _ in range(n):
            X = torch.full((len(bs), L), PAD, dtype=torch.long)
            for j, b in enumerate(bs): X[j, :len(b)] = torch.tensor(list(b[:L]))
            Xi = torch.cat([torch.full((len(bs), 1), PAD), X[:, :-1]], 1); lg = m(Xi)
            for j, b in enumerate(bs):
                if len(b) < L and (len(b) == 0 or b[-1] != 46): b.append(int(lg[j, len(b)].argmax()))
        out += [bytes(b[len(p.encode()):]).decode('utf-8', 'replace') for b, p in zip(bs, prompts[i:i + 128])]
    return out
def run(cond, seed):
    r = np.random.RandomState(seed); torch.manual_seed(seed); taken = set()
    people = [mkname(r, S1, taken) for _ in range(120)]
    facts = [(people[r.randint(120)], RELS[r.randint(len(RELS))], people[r.randint(120)]) for _ in range(300)]
    facts = list({(a, rl): (a, rl, b) for a, rl, b in facts if a != b}.values())
    taught = [f for i, f in enumerate(facts) if i % 4]; readonly = [f for i, f in enumerate(facts) if not i % 4]
    known = {(a, rl) for a, rl, b in facts}
    forms = F12 if cond == 'same+forms' else F4
    fresh_tr = [mkname(r, S2 if cond == 'distinct' else S1, taken) for _ in range(400)]
    fresh_te1 = [mkname(r, S1, taken) for _ in range(60)]; fresh_te3 = [mkname(r, S3, taken) for _ in range(60)]
    def sample(n):
        out = []
        for _ in range(n):
            u = r.rand()
            if u < 0.4: a, rl, b = facts[r.randint(len(facts))]; out.append(f"{a} {rl} {b}. ")
            elif u < 0.8 or cond == 'none': a, rl, b = taught[r.randint(len(taught))]; out.append(forms[r.randint(len(forms))].format(A=a, r=rl) + b + '.')
            else: out.append(forms[r.randint(len(forms))].format(A=fresh_tr[r.randint(len(fresh_tr))], r=RELS[r.randint(len(RELS))]) + 'unknown.')
        return torch.tensor([enc(t) for t in out])
    m = LM(); op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01); ls = []
    for st in range(STEPS):
        X = sample(32); Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); msk = (X != PAD).float()
        l = (F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1), reduction='none') * msk.reshape(-1)).sum() / msk.sum()
        op.zero_grad(); l.backward(); op.step(); ls.append(l.item())
    m.eval(); res = {}
    for wname, fm in (('held-out wording', TEST), ('taught wording', forms[1])):
        q = lambda A, rl: fm.format(A=A, r=rl)
        rf = lambda outs: np.mean([o.strip().lower().startswith('unknown') for o in outs])
        o1 = greedy(m, [q(n, RELS[i % 6]) for i, n in enumerate(fresh_te1)]); o3 = greedy(m, [q(n, RELS[i % 6]) for i, n in enumerate(fresh_te3)])
        kt = taught[:80]; ok = greedy(m, [q(a, rl) for a, rl, b in kt]); ro = greedy(m, [q(a, rl) for a, rl, b in readonly[:60]])
        res[wname] = (rf(o1), rf(o3), rf(ok), np.mean([o.strip().startswith(b) for o, (a, rl, b) in zip(ok, kt)]),
                      np.mean([o.strip().startswith(b) for o, (a, rl, b) in zip(ro, readonly[:60])]))
    return np.mean(ls[-100:]) / math.log(2), res
P(f"toy_refuse: {STEPS} steps, seed {SEED}")
for cond in ('none', 'distinct', 'same', 'same+forms'):
    bpb, res = run(cond, SEED)
    P(f"  {cond:11s} (train {bpb:.2f} bits/byte, {time.time() - T0:.0f}s) | " + " || ".join(
        f"{w}: REFUSE fresh (stored syllables) {a:.2f}, fresh (third syllables) {b:.2f}, KNOWN {c:.2f} | known right {d:.2f}, only-read right {e:.2f}"
        for w, (a, b, c, d, e) in res.items()))
P(f"DONE ({time.time() - T0:.0f}s)")
