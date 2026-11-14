"""SPARSE MEMORY FINETUNING ON THE STAND-IN TRUNK (14 Nov 2026; tools/sparse_mem.py) - offline, no effect on the package.
Lin et al. 2025 (FAIR, 'Continual Learning via Sparse Memory Finetuning'): a memory layer (product keys, value slots) in the model; new facts
update ONLY the slots they access most relative to background text (TF-IDF over slot accesses). On their QA streams: forgetting 89% (full
finetuning), 71% (LoRA), 11% (sparse memory), at the same learning. Here, small:
  1. The stand-in trunk (TG_CKPT, 3 blocks x TG_D) gets a product-key memory layer after block 2 (SM_SLOTS = n x n slots, top 16 of n x n per
     byte, values of width d) and continues on wiki text SM_PRE steps so the memory is used (saved to SM_CKPT; resumable).
  2. Facts from Re-DocRED documents this trunk never read, in the gold relation's phrase, taught in the system's seven frames (no test
     wording) in batches of FACTS ONLY (no replay - the stress test of the paper): set A (60 facts), then set B (60 facts).
  3. Methods, each from the same memory model, two learning rates:
       FULL        every parameter
       LORA        rank-16 adapters on every block's two MLP matrices
       MLP_OUT     only the blocks' MLP output matrices (the GPU build's inf_distill_wake_params mlp_out)
       MEM_ALL     every memory value
       MEM_SPARSE  only the SM_T memory values with the highest TF-IDF in the batch (accesses in the batch x log(background batches / batches
                   using the slot)), chosen afresh each step
  Readouts: recall in the TEST wording ('It is on record that A <relation> C') - open-ended exact and rank among 10 of the same type - of A
  after A, of B after B and of A after B (retention); held-out bits per byte on wiki and on Re-DocRED dev text (read in training) before
  and after: FORGETTING = the bits added.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/sparse_mem.py   env SM_PRE (1500), SM_STEPS (150 per set), SM_T (256), SM_RUNS"""
import os, sys, json, math, time, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
P = lambda *a: print(*a, flush=True); T0 = time.time()
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
DW = int(os.environ.get('TG_D', 192)); NL = 3; CK = os.environ['TG_CKPT']; torch.set_num_threads(int(os.environ.get('SM_THREADS', 1)))
L = 128; PAD = 256; V = 257; rs = np.random.RandomState(0); torch.manual_seed(0)
NS = int(os.environ.get('SM_N', 128)); KK = 16; DK = 64
PRE = int(os.environ.get('SM_PRE', 1500)); STEPS = int(os.environ.get('SM_STEPS', 150)); TOPT = int(os.environ.get('SM_T', 256))
SMCK = os.environ.get('SM_CKPT', SCR + f'/sm_mem_{DW}_{NS}_{PRE}.pt')
FORMS = ("In short, {A} {r} ", "Notes on {A}: {A} {r} ", "{A}, as noted before, {r} ", "As for {A}, people say {A} {r} ",
         "Q: {A} {r} whom? A: ", "{A}, whose name came up earlier in the report, {r} ", "From the notes: {A} {r} ")
REL = {'P17': 'is in the country', 'P131': 'is located in', 'P27': 'is a citizen of', 'P19': 'was born in', 'P20': 'died in', 'P108': 'worked for',
       'P54': 'played for', 'P463': 'is a member of', 'P102': 'is a member of the party', 'P175': 'was performed by', 'P495': 'comes from',
       'P159': 'is based in', 'P50': 'was written by', 'P57': 'was directed by', 'P361': 'is part of', 'P150': 'contains', 'P527': 'includes',
       'P740': 'was formed in', 'P276': 'took place in', 'P26': 'was married to', 'P69': 'was educated at'}
tr_docs = json.load(open(SCR + '/bench_dl/train_revised.json')); dv_docs = json.load(open(SCR + '/bench_dl/dev_revised.json'))
cn_ = lambda d: [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
tp_ = lambda d: [vs[0].get('type', 'X') for vs in d['vertexSet']]
dev_names = {n for d in dv_docs for n in cn_(d)}
FACTS, BYT, used = [], collections.defaultdict(set), set()
for d in tr_docs:
    for n, t in zip(cn_(d), tp_(d)): BYT[t].add(n)
for di in rs.permutation(len(tr_docs)):
    d = tr_docs[int(di)]; cn, tp = cn_(d), tp_(d)
    for l in d['labels']:
        A, C = cn[l['h']], cn[l['t']]
        if l['r'] not in REL or A in used or not (3 <= len(A) <= 30 and 3 <= len(C) <= 30) or not all(ord(ch) < 256 for ch in A + C): continue
        if A in dev_names and C in dev_names: continue
        FACTS.append(dict(A=A, C=C, rel=REL[l['r']], t=tp[l['t']])); used.add(A); break
    if len(FACTS) >= 120: break
SET_A, SET_B = FACTS[:60], FACTS[60:120]
wiki = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin', 'rb').read().replace(b'\n', b' '); WK = np.frombuffer(wiki, dtype=np.uint8)
HO = len(WK) - 200000                                                         # the last 200 KB of wiki: never sampled for training here
dev_txt = ' '.join(' '.join(' '.join(s) for s in d['sents']) for d in dv_docs[:200]).encode('latin-1', 'replace')
def hold_seqs(buf, n, off):
    return [bytes(buf[off + k * L: off + (k + 1) * L]) for k in range(n)]
EV_W = hold_seqs(WK, 96, HO); EV_D = hold_seqs(np.frombuffer(dev_txt, np.uint8), 96, 0)

class Mem(nn.Module):
    """product-key memory (Lample et al. 2019): query -> two half-queries scored against n sub-keys each; top 16 of the n x n combinations"""
    def __init__(s, d):
        super().__init__(); s.n = nn.LayerNorm(d); s.q = nn.Linear(d, 2 * DK); s.k1 = nn.Parameter(torch.randn(NS, DK) / DK ** 0.5)
        s.k2 = nn.Parameter(torch.randn(NS, DK) / DK ** 0.5); s.v = nn.Embedding(NS * NS, d); nn.init.normal_(s.v.weight, 0, 0.02); s.last = None
    def forward(s, x):
        q = s.q(s.n(x)); q1, q2 = q[..., :DK], q[..., DK:]
        a1, i1 = (q1 @ s.k1.t()).topk(KK, -1); a2, i2 = (q2 @ s.k2.t()).topk(KK, -1)
        sc = (a1[..., :, None] + a2[..., None, :]).flatten(-2); ix = (i1[..., :, None] * NS + i2[..., None, :]).flatten(-2)
        top, j = sc.topk(KK, -1); idx = ix.gather(-1, j); w = F.softmax(top, -1); s.last = idx
        return (w[..., None] * s.v(idx)).sum(-2)
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        h = s.n1(x); x = x + s.a(h, h, h, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class LM(nn.Module):
    def __init__(s, d=DW, nl=NL):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(L, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl))
        s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V); s.mem = Mem(d); s.mem_at = 1
    def forward(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for k, b in enumerate(s.b):
            x = b(x, am)
            if k == s.mem_at: x = x + s.mem(x)
        return s.o(s.n(x))
class LoRA(nn.Module):
    def __init__(s, lin, r=16):
        super().__init__(); s.lin = lin; s.A = nn.Parameter(torch.randn(r, lin.in_features) * 0.01); s.B = nn.Parameter(torch.zeros(lin.out_features, r))
    def forward(s, x): return s.lin(x) + (x @ s.A.t()) @ s.B.t()
def tens(seqs):
    X = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
    return X
def xin(X): return torch.cat([torch.full((X.shape[0], 1), PAD), X[:, :-1]], 1)
def loss_of(m, seqs):
    X = tens(seqs); lo = m(xin(X)).reshape(-1, V); tg = X.reshape(-1); return F.cross_entropy(lo[tg != PAD], tg[tg != PAD])
@torch.no_grad()
def bits(m, seqs):
    m.eval(); t = 0.0
    for k in range(0, len(seqs), 32): t += float(loss_of(m, seqs[k:k + 32])) * len(seqs[k:k + 32])
    m.train(); return t / len(seqs) / math.log(2)
@torch.no_grad()
def recall(m, facts):
    m.eval(); oe, rk = [], []; rr = np.random.RandomState(5)
    for f in facts:
        pb = f"It is on record that {f['A']} {f['rel']} ".encode('latin-1'); tb = (f['C'] + '.').encode('latin-1')
        X = tens([pb + tb]); lg = m(xin(X))[0]; ps = torch.arange(len(pb), len(pb) + len(tb)); top = lg[ps].argmax(-1)
        oe.append(int(bool((top[:-1] == X[0, ps][:-1]).all())))
        alts = sorted(n for n in BYT[f['t']] if n != f['C'] and 3 <= len(n) <= 30 and all(ord(ch) < 256 for ch in n))
        cands = [f['C']] + [alts[j] for j in rr.choice(len(alts), 9, replace=False)]
        seqs = [pb + c.encode('latin-1') + b'.' for c in cands]; X = tens(seqs)
        nl = -F.log_softmax(m(xin(X)), -1).gather(-1, X[..., None])[..., 0]; sc = [float(nl[k, len(pb):len(q) - 1].mean()) for k, q in enumerate(seqs)]
        rk.append(int(np.argmin(sc) == 0))
    m.train(); return np.mean(oe), np.mean(rk)
def wiki_batch(rr, n=32): return [bytes(WK[i0:i0 + L]) for i0 in rr.randint(0, HO - L - 1, n)]

# 1. the memory model: the trunk + a memory layer, continued on wiki (resumable)
m = LM(); sd = torch.load(CK); miss = m.load_state_dict(sd, strict=False)
P(f"facts {len(SET_A)} + {len(SET_B)} (never read by this trunk) | memory {NS}x{NS} = {NS * NS} slots of width {DW}, top {KK} per byte; "
  f"trunk parameters {sum(p.numel() for n, p in m.named_parameters() if not n.startswith('mem.')):,}, memory {sum(p.numel() for p in m.mem.parameters()):,}")
b_tr = bits(m, EV_W)
if os.path.exists(SMCK): m.load_state_dict(torch.load(SMCK)); P(f"memory model loaded from {SMCK}")
else:
    pm = list(m.mem.parameters()); po = [p for n, p in m.named_parameters() if not n.startswith('mem.')]
    op = torch.optim.AdamW([{'params': pm, 'lr': 1e-3}, {'params': po, 'lr': 1e-4}], weight_decay=0.0); rr = np.random.RandomState(1)
    for st in range(PRE):
        l = loss_of(m, wiki_batch(rr)); op.zero_grad(); l.backward(); op.step()
        if (st + 1) % 500 == 0: P(f"   memory pretraining step {st + 1}: train {float(l) / math.log(2):.3f} bits per byte  ({time.time() - T0:.0f}s)")
    torch.save(m.state_dict(), SMCK)
# background slot use (document frequency over wiki batches) and how much the model leans on its memory
cnt_bg = np.zeros(NS * NS); NBG = 100; rr = np.random.RandomState(2)
with torch.no_grad():
    m.eval()
    for _ in range(NBG): m(xin(tens(wiki_batch(rr)))); cnt_bg += np.bincount(np.unique(m.mem.last.numpy()), minlength=NS * NS) > 0
    at = m.mem_at; m.mem_at = -1; b_nomem = bits(m, EV_W); m.mem_at = at; m.train()
IDF = torch.tensor(np.log((NBG + 1.0) / (cnt_bg + 1.0)), dtype=torch.float32)
b0w, b0d = bits(m, EV_W), bits(m, EV_D)
P(f"held-out wiki bits per byte: trunk {b_tr:.3f} | memory model {b0w:.3f} (memory switched off {b_nomem:.3f}) | dev text {b0d:.3f} | "
  f"slots used by background {int((cnt_bg > 0).sum())} of {NS * NS}  ({time.time() - T0:.0f}s)")
BASE = {k: v.clone() for k, v in m.state_dict().items()}

def teach(mm, facts, params, lr, sparse, rr):
    op = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
    for st in range(STEPS):
        seqs = [(FORMS[rr.randint(len(FORMS))].format(A=f['A'], r=f['rel']) + f['C'] + '.').encode('latin-1') for f in (facts[rr.randint(len(facts))] for _ in range(32))]
        l = loss_of(mm, seqs); op.zero_grad(); l.backward()
        if sparse:                                                            # TF-IDF over slot accesses in this batch; only the top TOPT keep their gradient
            tf = torch.bincount(mm.mem.last.reshape(-1), minlength=NS * NS).float(); keep = (tf * IDF).topk(TOPT).indices
            mask = torch.zeros(NS * NS, 1); mask[keep] = 1.0; mm.mem.v.weight.grad *= mask
        op.step()
        if sparse: mm._touched.update(keep.tolist())
def run(tag, method, lr):
    mm = LM(); mm.load_state_dict(BASE); mm._touched = set(); rr = np.random.RandomState(11)
    for p in mm.parameters(): p.requires_grad_(False)
    if method == 'FULL': ps = list(mm.parameters())
    elif method == 'MLP_OUT': ps = [b.m[2].weight for b in mm.b] + [b.m[2].bias for b in mm.b]
    elif method in ('MEM_ALL', 'MEM_SPARSE'): ps = [mm.mem.v.weight]
    elif method == 'LORA':
        ps = []
        for b in mm.b:
            for j in (0, 2): b.m[j] = LoRA(b.m[j]); ps += [b.m[j].A, b.m[j].B]
    for p in ps: p.requires_grad_(True)
    sp = method == 'MEM_SPARSE'
    teach(mm, SET_A, ps, lr, sp, rr); ra = recall(mm, SET_A); w1, d1 = bits(mm, EV_W), bits(mm, EV_D)
    teach(mm, SET_B, ps, lr, sp, rr); rb = recall(mm, SET_B); raa = recall(mm, SET_A); w2, d2 = bits(mm, EV_W), bits(mm, EV_D)
    tch = f" | slots changed {len(mm._touched)}" if sp else ''
    P(f"   {tag:24s} lr {lr:g}: A after A open {ra[0]:.2f} rank {ra[1]:.2f} | B after B open {rb[0]:.2f} rank {rb[1]:.2f} | A after B open {raa[0]:.2f} "
      f"rank {raa[1]:.2f} || FORGETTING (bits added) wiki {w1 - b0w:+.3f} -> {w2 - b0w:+.3f}, dev {d1 - b0d:+.3f} -> {d2 - b0d:+.3f}{tch}  ({time.time() - T0:.0f}s)")
r0a, r0b = recall(m, SET_A), recall(m, SET_B)
P(f"before teaching: A open {r0a[0]:.2f} rank {r0a[1]:.2f} | B open {r0b[0]:.2f} rank {r0b[1]:.2f} | {STEPS} steps of 32 fact rows per set "
  f"(about {STEPS * 32 / 60:.0f} rows per fact), facts only")
RUNS = [('FULL', 'FULL', 1e-4), ('FULL', 'FULL', 3e-4), ('LORA', 'LORA', 1e-3), ('LORA', 'LORA', 3e-3), ('MLP_OUT', 'MLP_OUT', 1e-4),
        ('MLP_OUT', 'MLP_OUT', 3e-4), ('MEM_ALL', 'MEM_ALL', 1e-3), ('MEM_ALL', 'MEM_ALL', 1e-2), ('MEM_SPARSE', 'MEM_SPARSE', 1e-3),
        ('MEM_SPARSE', 'MEM_SPARSE', 1e-2), (f'MEM_SPARSE t={4 * TOPT}', 'MEM_SPARSE', 1e-2)]
SEL = os.environ.get('SM_RUNS', '')
for k, (tag, meth, lr) in enumerate(RUNS):
    if SEL and str(k + 1) not in SEL.split(','): continue
    if 't=' in tag: TOPT *= 4
    run(tag, meth, lr)
    if 't=' in tag: TOPT //= 4
P(f"DONE ({time.time() - T0:.0f}s)")
