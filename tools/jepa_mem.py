"""A KEY-TO-KEY SPARSE MEMORY ADDRESSED BY NAME + JEPA WORDING KEY (14 Nov 2026; tools/jepa_mem.py) - offline, no effect on the package.
The author's idea: a sparse memory (Lin et al. 2025) whose address is a JEPA key and whose VALUE is a key too, so one lookup's output can be
the next lookup's query (composition in key space; bytes only at the end). Here, on the frozen stand-in trunk (TG_CKPT):
  NAME KEY     the trunk's state over a name's bytes (last block, mean, centred, unit length)
  WORDING KEY  the trunk's state over the question prefix with the subject replaced by 'someone' ('In short, someone was born in '),
               mapped by a small head trained JEPA/InfoNCE-style so the system's SEVEN TRAINING FRAMES of one relation phrase meet (the
               system knows which rows it wrote as one fact); the TEST WORDING ('It is on record that someone ... ') is never seen
  MEMORY       product keys: half 1 = a fixed projection of the subject's NAME KEY, half 2 = the WORDING KEY; NS x NS slots, top KK per read;
               each slot's VALUE lives in name-key space. Writing = gradient on the values of the slots a fact's query reaches (sparse by
               construction: nothing else changes); the loss makes the read nearest the object's name key among all known names
  READ         test-wording query -> value -> the nearest name: RANK among the object and 9 others of its type (chance 0.10; as
               tools/sparse_mem.py, the same facts) and OPEN among every known name
Arms: K-JEPA (the trained wording head), K-RAW (the raw prefix state, randomly projected: no JEPA training), K-NAME (no wording half: the
slot is the name alone). Lifelong order, one memory per arm: set A (60 facts; = sparse_mem's A) -> set B (60) -> M (subjects with two
facts of different relations: the wording half has to separate them) -> CHAINS (A r1 B and B r2 C from one document, both hops taught,
the chain A -> C never): two-hop by LATENT composition (the first read's value used directly as the next name key) and DECODED (the first
read decoded to a known name, its key used). Retention of A, B and M is re-read at the end.
Second version (14 Nov, late): JM_SEED (fact order, memory and head seeds), and after the lifelong sequence, per arm:
  ABSTAIN    never-taught subjects (names in the inventory, never a subject) vs taught ones: AUC of the read's best cosine to a known name
  RETRACT    30 facts of set A unlearned in the memory only (-log(1 - p(object)) on their training-frame queries, 30 steps; the slots they
             reach are the only ones that change): their recall, and the bystanders' (the rest of A, B, M)
  CAPACITY   JM_CAP more single facts in batches of 200 (one per subject, unused documents): recall of each batch at the end and of A
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/jepa_mem.py   env JM_N (64), JM_KK (8), JM_STEPS (150 per set), JM_LR (0.05), JM_SEED (0),
  JM_CAP (600), JM_ARMS (K-JEPA,K-RAW,K-NAME; add K-TEXT with JM_TEXT), JM_TEXT (a DocRED distant .json.gz: K-TEXT, a wording head
  trained on REAL TEXT only - two different wordings that link the same pair of names in one sentence are one statement; InfoNCE; the
  frames and the test wording never seen), JM_PAIRS (20000; with JM_TEXT also arm K-BOTH: the head trained on the text pairs AND the frames), JM_RX (30: retraction steps), JM_NULL (0: rows a write step that push reads for names never
  used as subjects - a disjoint set from the abstention test's - toward zero), JM_REPLAY (0: share of each
  write batch spent re-writing earlier facts, as sleep replays the store), JM_RXMODE (ul: -log(1 - p); margin: the object pushed 0.1 below the best other name; erase: the
  slots the facts' training-frame queries reach set to zero); ABSTAIN also by the read's norm (unwritten
  slots hold zeros)"""
import os, json, math, time, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
P = lambda *a: print(*a, flush=True); T0 = time.time()
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
DW = int(os.environ.get('TG_D', 192)); CK = os.environ['TG_CKPT']; torch.set_num_threads(1)
L = 128; PAD = 256; V = 257; SEED = int(os.environ.get('JM_SEED', 0)); rs = np.random.RandomState(SEED); torch.manual_seed(SEED)
NS = int(os.environ.get('JM_N', 64)); KK = int(os.environ.get('JM_KK', 8)); DK = 32; STEPS = int(os.environ.get('JM_STEPS', 150)); LR = float(os.environ.get('JM_LR', 0.05))
FORMS = ("In short, {A} {r} ", "Notes on {A}: {A} {r} ", "{A}, as noted before, {r} ", "As for {A}, people say {A} {r} ",
         "Q: {A} {r} whom? A: ", "{A}, whose name came up earlier in the report, {r} ", "From the notes: {A} {r} ")
TEST = "It is on record that {A} {r} "
REL = {'P17': 'is in the country', 'P131': 'is located in', 'P27': 'is a citizen of', 'P19': 'was born in', 'P20': 'died in', 'P108': 'worked for',
       'P54': 'played for', 'P463': 'is a member of', 'P102': 'is a member of the party', 'P175': 'was performed by', 'P495': 'comes from',
       'P159': 'is based in', 'P50': 'was written by', 'P57': 'was directed by', 'P361': 'is part of', 'P150': 'contains', 'P527': 'includes',
       'P740': 'was formed in', 'P276': 'took place in', 'P26': 'was married to', 'P69': 'was educated at'}
RELS = sorted(set(REL.values()))
tr_docs = json.load(open(SCR + '/bench_dl/train_revised.json')); dv_docs = json.load(open(SCR + '/bench_dl/dev_revised.json'))
cn_ = lambda d: [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
tp_ = lambda d: [vs[0].get('type', 'X') for vs in d['vertexSet']]
dev_names = {n for d in dv_docs for n in cn_(d)}
ok = lambda n: 3 <= len(n) <= 30 and all(ord(ch) < 256 for ch in n)
BYT = collections.defaultdict(set)
for d in tr_docs:
    for n, t in zip(cn_(d), tp_(d)): BYT[t].add(n)
# set A / B exactly as tools/sparse_mem.py (same seed, same order)
FACTS, used, used_docs = [], set(), set()
for di in rs.permutation(len(tr_docs)):
    d = tr_docs[int(di)]; cn, tp = cn_(d), tp_(d)
    for l in d['labels']:
        A, C = cn[l['h']], cn[l['t']]
        if l['r'] not in REL or A in used or not (ok(A) and ok(C)): continue
        if A in dev_names and C in dev_names: continue
        FACTS.append(dict(A=A, C=C, rel=REL[l['r']], t=tp[l['t']])); used.add(A); used_docs.add(int(di)); break
    if len(FACTS) >= 120: break
SET_A, SET_B = FACTS[:60], FACTS[60:120]
# M: subjects with two facts of different relations and different objects
SET_M, CHAINS = [], []
for di in np.random.RandomState(SEED + 1).permutation(len(tr_docs)):
    di = int(di)
    if di in used_docs: continue
    d = tr_docs[di]; cn, tp = cn_(d), tp_(d)
    byh = collections.defaultdict(list)
    for l in d['labels']:
        if l['r'] in REL and ok(cn[l['h']]) and ok(cn[l['t']]) and cn[l['h']] != cn[l['t']]: byh[l['h']].append(l)
    if len(SET_M) < 80:
        for h, ls in byh.items():
            if cn[h] in used: continue
            two = {}
            for l in ls:
                if REL[l['r']] not in two and cn[l['t']] not in [cn[x['t']] for x in two.values()]: two[REL[l['r']]] = l
            if len(two) >= 2:
                for l in list(two.values())[:2]: SET_M.append(dict(A=cn[h], C=cn[l['t']], rel=REL[l['r']], t=tp[l['t']]))
                used.add(cn[h]); used_docs.add(di); break
        if di in used_docs: continue
    if len(CHAINS) < 60:
        for h, ls in byh.items():
            for l1 in ls:
                m_ = l1['t']; nxt = [l2 for l2 in byh.get(m_, []) if l2['t'] != h and cn[l2['t']] != cn[h]]
                if nxt and cn[h] not in used and cn[m_] not in used:
                    l2 = nxt[0]; CHAINS.append((dict(A=cn[h], C=cn[m_], rel=REL[l1['r']], t=tp[m_]), dict(A=cn[m_], C=cn[l2['t']], rel=REL[l2['r']], t=tp[l2['t']])))
                    used.update([cn[h], cn[m_]]); used_docs.add(di); break
            if di in used_docs: break
    if len(SET_M) >= 80 and len(CHAINS) >= 60: break
HOPS = [f for c in CHAINS for f in c]
CAPF = []; NCAP = int(os.environ.get('JM_CAP', 600))
for di in np.random.RandomState(SEED + 2).permutation(len(tr_docs)):
    di = int(di)
    if di in used_docs or len(CAPF) >= NCAP: continue
    d = tr_docs[di]; cn, tp = cn_(d), tp_(d)
    for l in d['labels']:
        A, C = cn[l['h']], cn[l['t']]
        if l['r'] in REL and A not in used and A != C and ok(A) and ok(C): CAPF.append(dict(A=A, C=C, rel=REL[l['r']], t=tp[l['t']])); used.add(A); used_docs.add(di); break
UNSEEN = []                                                                    # subjects never taught: for abstention
for di in np.random.RandomState(SEED + 3).permutation(len(tr_docs)):
    di = int(di)
    if di in used_docs or len(UNSEEN) >= 120: continue
    d = tr_docs[di]; cn, tp = cn_(d), tp_(d)
    for l in d['labels']:
        A, C = cn[l['h']], cn[l['t']]
        if l['r'] in REL and A not in used and ok(A) and ok(C): UNSEEN.append(dict(A=A, C=C, rel=REL[l['r']], t=tp[l['t']])); used.add(A); break
P(f"facts: A {len(SET_A)}, B {len(SET_B)}, M {len(SET_M)} ({len(SET_M) // 2} subjects x 2 relations), chains {len(CHAINS)} ({len(HOPS)} hop facts)")

class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        h = s.n1(x); x = x + s.a(h, h, h, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class LM(nn.Module):
    def __init__(s, d=DW, nl=3):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(L, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl))
        s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V)
    def states(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for b in s.b: x = b(x, am)
        return s.n(x)
lm = LM(); lm.load_state_dict(torch.load(CK)); lm.eval()
@torch.no_grad()
def enc(texts, spans):
    """trunk states: mean over the byte span [a, b) of each text, and the last byte's state"""
    out_m, out_l = [], []
    for k in range(0, len(texts), 128):
        tb = [t.encode('latin-1') for t in texts[k:k + 128]]; X = torch.full((len(tb), L), PAD, dtype=torch.long)
        for j, q in enumerate(tb): X[j, :len(q)] = torch.tensor(list(q))
        Xi = torch.cat([torch.full((len(tb), 1), PAD), X[:, :-1]], 1); H = lm.states(Xi)
        for j, (a, b) in enumerate(spans[k:k + 128]): out_m.append(H[j, a:b].mean(0)); out_l.append(H[j, b - 1].clone())
    return torch.stack(out_m), torch.stack(out_l)
# name inventory: every name in the sets and in their candidate types
rr = np.random.RandomState(SEED + 5); CAND = {}
def cands(f):
    key = (f['A'], f['rel'], f['C'])
    if key not in CAND:
        alts = sorted(n for n in BYT[f['t']] if n != f['C'] and ok(n)); CAND[key] = [f['C']] + [alts[j] for j in rr.choice(len(alts), 9, replace=False)]
    return CAND[key]
ALLF = SET_A + SET_B + SET_M + HOPS + CAPF + UNSEEN
NAMES = sorted({n for f in ALLF for n in [f['A']] + cands(f)})
NI = {n: k for k, n in enumerate(NAMES)}
nm_m, _ = enc(["The name " + n + "." for n in NAMES], [(9, 9 + len(n)) for n in NAMES])
NK = nm_m - nm_m.mean(0); NK = F.normalize(NK, dim=-1)
P(f"name keys for {len(NAMES)} known names  ({time.time() - T0:.0f}s)")
# wording prefixes, subject masked: 7 training frames x relations, and the test wording
FR = FORMS + (TEST,)
ptxt = [fr.format(A='someone', r=r) for r in RELS for fr in FR]; pm, pl = enc(ptxt, [(0, len(t)) for t in ptxt])
PC = torch.cat([pm, pl], -1); PC = (PC - PC.mean(0)) / (PC.std(0) + 1e-6)
PIDX = {(r, fi): k for k, (r, fi) in enumerate((r, fi) for r in RELS for fi in range(len(FR)))}
TI = len(FR) - 1
PARA = {'was born in': 'has as birthplace', 'died in': 'passed away in', 'is located in': 'lies in', 'is a citizen of': 'holds citizenship of',
        'worked for': 'was employed by', 'played for': 'was a player for', 'was written by': 'was authored by', 'was directed by': 'had as director',
        'is part of': 'belongs to', 'was educated at': 'studied at', 'is in the country': 'is situated in the country', 'comes from': 'originates from',
        'is based in': 'has headquarters in', 'was married to': 'was the spouse of', 'took place in': 'happened in', 'is a member of': 'belongs to the group',
        'was performed by': 'was sung by', 'contains': 'has within it', 'includes': 'comprises', 'was formed in': 'was founded in',
        'is a member of the party': 'belongs to the party'}
pp = [TEST.format(A='someone', r=PARA[r]) for r in RELS]; ppm, ppl = enc(pp, [(0, len(t)) for t in pp])
PPC = (torch.cat([ppm, ppl], -1) - torch.cat([pm, pl], -1).mean(0)) / (torch.cat([pm, pl], -1).std(0) + 1e-6)
# the JEPA wording head: InfoNCE over relations, positives = two different TRAINING frames of the same relation phrase
head = nn.Sequential(nn.Linear(PC.shape[1], 128), nn.GELU(), nn.Linear(128, DK)); oph = torch.optim.Adam(head.parameters(), lr=1e-3); hr = np.random.RandomState(SEED + 7)
for st in range(1500):
    f1 = hr.randint(0, TI, len(RELS)); f2 = (f1 + hr.randint(1, TI, len(RELS))) % TI
    z1 = F.normalize(head(PC[[PIDX[(r, int(a))] for r, a in zip(RELS, f1)]]), dim=-1); z2 = F.normalize(head(PC[[PIDX[(r, int(b))] for r, b in zip(RELS, f2)]]), dim=-1)
    lo = F.cross_entropy(z1 @ z2.t() / 0.1, torch.arange(len(RELS))); oph.zero_grad(); lo.backward(); oph.step()
with torch.no_grad():
    WJ = F.normalize(head(PC), dim=-1); RP = torch.randn(PC.shape[1], DK) / PC.shape[1] ** 0.5; WR = F.normalize(PC @ RP, dim=-1)
    WJP = F.normalize(head(PPC), dim=-1); WRP = F.normalize(PPC @ RP, dim=-1)                     # paraphrased test wording, per relation
    def waccp(W, WP):
        tr = torch.stack([W[[PIDX[(r, fi)] for fi in range(TI)]].mean(0) for r in RELS]); return float(((WP @ tr.t()).argmax(-1) == torch.arange(len(RELS))).float().mean())
    def wacc(W):
        tr = torch.stack([W[[PIDX[(r, fi)] for fi in range(TI)]].mean(0) for r in RELS]); te = W[[PIDX[(r, TI)] for r in RELS]]
        return float(((te @ tr.t()).argmax(-1) == torch.arange(len(RELS))).float().mean())
P(f"wording keys: the TEST wording's nearest relation (by the training frames' mean key) - JEPA head {wacc(WJ):.2f}, raw state {wacc(WR):.2f} "
  f"(chance {1 / len(RELS):.2f}, {len(RELS)} relations) | PARAPHRASED test wording ('It is on record that someone passed away in '): JEPA head "
  f"{waccp(WJ, WJP):.2f}, raw state {waccp(WR, WRP):.2f}  ({time.time() - T0:.0f}s)")
R1 = torch.randn(DW, DK) / DW ** 0.5
TXT = os.environ.get('JM_TEXT', '')                                            # K-TEXT: the wording head learned from REAL TEXT, no frames
if TXT:
    import gzip
    dd = json.load(gzip.open(TXT, 'rt')); NPAIR = int(os.environ.get('JM_PAIRS', 20000)); W_ = collections.defaultdict(dict)
    for d in dd:
        cn = cn_(d)
        for i, vi in enumerate(d['vertexSet']):
            for j, vj in enumerate(d['vertexSet']):
                if i == j or not ok(cn[i]) or not ok(cn[j]): continue
                for mi in vi:
                    for mj in vj:
                        if mi['sent_id'] != mj['sent_id'] or not (0 < mj['pos'][0] - mi['pos'][1] <= 8): continue
                        tk = d['sents'][mi['sent_id']]; btw = ' '.join(tk[mi['pos'][1]:mj['pos'][0]])
                        if not all(ord(ch) < 256 for ch in btw) or not any(ch.isalpha() for ch in btw): continue
                        ctx = ' '.join(tk[max(0, mi['pos'][0] - 5):mi['pos'][0]])
                        if all(ord(ch) < 256 for ch in ctx): W_[(cn[i], cn[j])].setdefault(btw, (ctx + ' someone ' + btw + ' ').strip() + ' ')
    del dd
    multi = [(k, list(v.values())[:3]) for k, v in W_.items() if len(v) >= 2]; np.random.RandomState(SEED + 9).shuffle(multi); multi = multi[:NPAIR]; del W_
    tp_txt = [t[-L + 1:] for k, ws in multi for t in ws]; owner = np.array([n for n, (k, ws) in enumerate(multi) for t in ws])
    P(f"K-TEXT: {len(multi)} name pairs stated in 2+ different wordings ({len(tp_txt)} prefixes) from {TXT.split('/')[-1]}  ({time.time() - T0:.0f}s)")
    tm_, tl_ = enc(tp_txt, [(0, len(t)) for t in tp_txt]); TC = torch.cat([tm_, tl_], -1); TMU, TSD = TC.mean(0), TC.std(0) + 1e-6; TC = (TC - TMU) / TSD
    P(f"   text prefixes encoded  ({time.time() - T0:.0f}s)")
    head_t = nn.Sequential(nn.Linear(TC.shape[1], 128), nn.GELU(), nn.Linear(128, DK)); opt_ = torch.optim.Adam(head_t.parameters(), lr=1e-3); tr_ = np.random.RandomState(SEED + 8)
    byo = collections.defaultdict(list)
    for k, o in enumerate(owner): byo[int(o)].append(k)
    keys_ = list(byo)
    for st in range(3000):
        ps_ = [keys_[i] for i in tr_.choice(len(keys_), 256, replace=False)]; a_ = [byo[p_][0] for p_ in ps_]; b_ = [byo[p_][1 + tr_.randint(len(byo[p_]) - 1)] for p_ in ps_]
        z1 = F.normalize(head_t(TC[a_]), dim=-1); z2 = F.normalize(head_t(TC[b_]), dim=-1)
        lo = F.cross_entropy(z1 @ z2.t() / 0.1, torch.arange(256)); opt_.zero_grad(); lo.backward(); opt_.step()
        if (st + 1) % 1000 == 0: P(f"   text head step {st + 1}: InfoNCE {float(lo.detach()):.3f}  ({time.time() - T0:.0f}s)")
    with torch.no_grad():
        raw_ = lambda m_, l_: (torch.cat([m_, l_], -1) - TMU) / TSD
        WT = F.normalize(head_t(raw_(pm, pl)), dim=-1); WTP = F.normalize(head_t(raw_(ppm, ppl)), dim=-1)
    P(f"   K-TEXT wording keys: the TEST wording's relation by the frames' keys {wacc(WT):.2f} | PARAPHRASED {waccp(WT, WTP):.2f} "
      f"(frame-trained JEPA head {wacc(WJ):.2f} / {waccp(WJ, WJP):.2f}; raw {wacc(WR):.2f} / {waccp(WR, WRP):.2f})")
    head_b = nn.Sequential(nn.Linear(TC.shape[1], 128), nn.GELU(), nn.Linear(128, DK)); opb = torch.optim.Adam(head_b.parameters(), lr=1e-3)
    with torch.no_grad(): FRN = raw_(pm, pl)                                   # K-BOTH: real-text pairs AND the system's frames
    for st in range(3000):
        ps_ = [keys_[i] for i in tr_.choice(len(keys_), 256, replace=False)]; a_ = [byo[p_][0] for p_ in ps_]; b_ = [byo[p_][1 + tr_.randint(len(byo[p_]) - 1)] for p_ in ps_]
        z1 = F.normalize(head_b(TC[a_]), dim=-1); z2 = F.normalize(head_b(TC[b_]), dim=-1); lo = F.cross_entropy(z1 @ z2.t() / 0.1, torch.arange(256))
        f1 = tr_.randint(0, TI, len(RELS)); f2 = (f1 + tr_.randint(1, TI, len(RELS))) % TI
        y1 = F.normalize(head_b(FRN[[PIDX[(r, int(a))] for r, a in zip(RELS, f1)]]), dim=-1); y2 = F.normalize(head_b(FRN[[PIDX[(r, int(b))] for r, b in zip(RELS, f2)]]), dim=-1)
        lo = lo + F.cross_entropy(y1 @ y2.t() / 0.1, torch.arange(len(RELS))); opb.zero_grad(); lo.backward(); opb.step()
    with torch.no_grad(): WB = F.normalize(head_b(FRN), dim=-1); WBP = F.normalize(head_b(raw_(ppm, ppl)), dim=-1)
    P(f"   K-BOTH wording keys (text pairs + frames): the TEST wording {wacc(WB):.2f} | PARAPHRASED {waccp(WB, WBP):.2f}  ({time.time() - T0:.0f}s)")

class KMem:
    def __init__(s, arm):
        g = torch.Generator().manual_seed(SEED + 3); s.arm = arm; s.k1 = F.normalize(torch.randn(NS, DK, generator=g), dim=-1); s.k2 = F.normalize(torch.randn(NS, DK, generator=g), dim=-1)
        s.v = torch.zeros(NS * NS, DW, requires_grad=True); s.opt = torch.optim.Adam([s.v], lr=LR); s.touched = set()
    def wkey(s, r, fi):
        if s.arm == 'K-NAME': return torch.zeros(DK)
        if fi == 'P': return {'K-JEPA': WJP, 'K-RAW': WRP, 'K-TEXT': WTP if TXT else None, 'K-BOTH': WBP if TXT else None}[s.arm][RELS.index(r)]
        return {'K-JEPA': WJ, 'K-RAW': WR, 'K-TEXT': WT if TXT else None, 'K-BOTH': WB if TXT else None}[s.arm][PIDX[(r, fi)]]
    def read(s, nk, wk):
        """nk: (B, DW) subject name keys; wk: (B, DK) wording keys -> (B, DW) read value, slot indices"""
        q1 = F.normalize(nk @ R1, dim=-1); a1, i1 = (q1 @ s.k1.t()).topk(KK, -1)
        if s.arm == 'K-NAME': a2 = torch.zeros(len(nk), KK); i2 = torch.zeros(len(nk), KK, dtype=torch.long)
        else: a2, i2 = (wk @ s.k2.t()).topk(KK, -1)
        sc = (a1[:, :, None] + a2[:, None, :]).flatten(1); ix = (i1[:, :, None] * NS + i2[:, None, :]).flatten(1)
        top, j = sc.topk(KK, -1); idx = ix.gather(-1, j); w = F.softmax(top / 0.05, -1)
        return (w[..., None] * s.v[idx]).sum(1), idx
    def teach(s, facts, rr):
        hist = getattr(s, 'hist', []); nrp = int(32 * float(os.environ.get('JM_REPLAY', 0))) if hist else 0
        for st in range(STEPS):
            fb = [facts[rr.randint(len(facts))] for _ in range(32 - nrp)] + [hist[rr.randint(len(hist))] for _ in range(nrp)]; fis = rr.randint(0, TI, 32)
            nk = NK[[NI[f['A']] for f in fb]]; wk = torch.stack([s.wkey(f['rel'], int(fi)) for f, fi in zip(fb, fis)])
            val, idx = s.read(nk, wk); lg = F.normalize(val, dim=-1) @ NK.t() / 0.05
            lo = F.cross_entropy(lg, torch.tensor([NI[f['C']] for f in fb]))
            if NUL > 0:                                                         # 15 Nov: names never taught as subjects should read ~nothing
                nn_ = NULLN[rr.randint(len(NULLN), size=NUL)]; wn = torch.stack([s.wkey(RELS[rr.randint(len(RELS))], int(rr.randint(TI))) for _ in range(NUL)])
                vn, _ = s.read(NK[nn_], wn); lo = lo + vn.pow(2).sum(-1).mean()
            s.opt.zero_grad(); lo.backward(); s.opt.step(); s.touched.update(idx.flatten().tolist())
        s.hist = hist + [f for f in facts if f not in hist]
    @torch.no_grad()
    def recall(s, facts, fi=None):
        nk = NK[[NI[f['A']] for f in facts]]; wk = torch.stack([s.wkey(f['rel'], TI if fi is None else fi) for f in facts]); val, _ = s.read(nk, wk)
        return s.score(val, facts)
    def score(s, val, facts):
        vn = F.normalize(val, dim=-1); op = (vn @ NK.t()).argmax(-1); oe = np.mean([NAMES[int(op[k])] == f['C'] for k, f in enumerate(facts)])
        rk = np.mean([int(np.argmax([float(vn[k] @ NK[NI[c]]) for c in cands(f)]) == 0) for k, f in enumerate(facts)])
        return oe, rk
    def retract(s, facts, steps=30):
        if os.environ.get('JM_RXMODE', 'ul') == 'erase':                     # reset the slots the facts' training-frame queries reach
            with torch.no_grad():
                nk = NK[[NI[f['A']] for f in facts for fi in range(TI)]]; wk = torch.stack([s.wkey(f['rel'], fi) for f in facts for fi in range(TI)])
                _, idx = s.read(nk, wk); ids = torch.unique(idx.flatten()); s.v[ids] = 0.0; s.erased = len(ids)
            return
        for st in range(steps):
            nk = NK[[NI[f['A']] for f in facts for fi in range(TI)]]; wk = torch.stack([s.wkey(f['rel'], fi) for f in facts for fi in range(TI)])
            val, idx = s.read(nk, wk); cs = F.normalize(val, dim=-1) @ NK.t(); ci = torch.tensor([NI[f['C']] for f in facts for fi in range(TI)])
            if os.environ.get('JM_RXMODE', 'ul') == 'margin':               # push the object below the best other name by a margin
                cc = cs[torch.arange(len(nk)), ci]; co = cs.scatter(1, ci[:, None], -9.0).max(-1).values; lo = F.relu(cc - co + 0.1).mean()
            else:
                pc = F.softmax(cs / 0.05, -1)[torch.arange(len(nk)), ci]; lo = -torch.log1p(-pc.clamp(max=0.999)).mean()
            s.opt.zero_grad(); lo.backward(); s.opt.step()
    @torch.no_grad()
    def confid(s, facts):
        nk = NK[[NI[f['A']] for f in facts]]; wk = torch.stack([s.wkey(f['rel'], TI) for f in facts]); val, _ = s.read(nk, wk)
        return (F.normalize(val, dim=-1) @ NK.t()).max(-1).values.numpy(), val.norm(dim=-1).numpy()
    @torch.no_grad()
    def two_hop(s):
        f1s = [c[0] for c in CHAINS]; f2s = [c[1] for c in CHAINS]
        nk = NK[[NI[f['A']] for f in f1s]]; v1, _ = s.read(nk, torch.stack([s.wkey(f['rel'], TI) for f in f1s]))
        w2 = torch.stack([s.wkey(f['rel'], TI) for f in f2s]); fin = [dict(A=a['A'], C=b['C'], rel=b['rel'], t=b['t']) for a, b in CHAINS]
        lat, _ = s.read(F.normalize(v1, dim=-1), w2)                                              # latent: the value IS the next name key
        dec_n = NK[(F.normalize(v1, dim=-1) @ NK.t()).argmax(-1)]; dec, _ = s.read(dec_n, w2)       # decoded to a known name first
        for k, (a, b) in enumerate(CHAINS): CAND[(a['A'], b['rel'], b['C'])] = cands(b)              # the final answer's candidates = hop 2's
        return s.score(lat, fin), s.score(dec, fin)
fmt = lambda x: f"open {x[0]:.2f} rank {x[1]:.2f}"
NUL = int(os.environ.get('JM_NULL', 0))                                       # rows a step that teach 'no fact here' on inventory names never used as subjects
_subj = {f['A'] for f in SET_A + SET_B + SET_M + HOPS + CAPF + UNSEEN}
NULLN = np.array([NI[n] for n in NAMES if n not in _subj])
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float); return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
P(f"seed {SEED}; capacity facts {len(CAPF)}; never-taught subjects {len(UNSEEN)}")
for arm in os.environ.get('JM_ARMS', 'K-JEPA,K-RAW,K-NAME').split(','):
    M = KMem(arm); rr = np.random.RandomState(11)
    M.teach(SET_A, rr); ra = M.recall(SET_A); M.teach(SET_B, rr); rb = M.recall(SET_B); raa = M.recall(SET_A)
    M.teach(SET_M, rr); rm = M.recall(SET_M); M.teach(HOPS, rr); rh = M.recall(HOPS); lat, dec = M.two_hop()
    P(f"   {arm:7s} A after A {fmt(ra)} | B after B {fmt(rb)} | A after B {fmt(raa)} | M (two relations per subject) {fmt(rm)} | chain hops {fmt(rh)}")
    P(f"           TWO-HOP, chain never taught: LATENT {fmt(lat)} | DECODED {fmt(dec)} || at the end: A {fmt(M.recall(SET_A))}, B {fmt(M.recall(SET_B))}, "
      f"M {fmt(M.recall(SET_M))} | slots written {len(M.touched)} of {NS * NS}  ({time.time() - T0:.0f}s)")
    taught = SET_A + SET_B + SET_M + HOPS
    P(f"           PARAPHRASED test wording (relation phrase never taught): A {fmt(M.recall(SET_A, 'P'))}, B {fmt(M.recall(SET_B, 'P'))}, M {fmt(M.recall(SET_M, 'P'))} "
      f"(the test wording: A {fmt(M.recall(SET_A))}, B {fmt(M.recall(SET_B))}, M {fmt(M.recall(SET_M))})")
    ct, cu = M.confid(taught), M.confid(UNSEEN)
    P(f"           ABSTAIN: taught vs never-taught subjects, AUC of the best cosine to a known name {auc(ct[0], cu[0]):.3f}, of the read's norm {auc(ct[1], cu[1]):.3f} "
      f"(never-taught read 'right' by chance: {fmt(M.recall(UNSEEN))})")
    snap = M.v.detach().clone(); RX, KEEP = SET_A[:30], SET_A[30:]; b4 = [M.recall(x) for x in (RX, KEEP, SET_B, SET_M)]
    M.retract(RX, int(os.environ.get('JM_RX', 30))); af = [M.recall(x) for x in (RX, KEEP, SET_B, SET_M)]
    P(f"           RETRACT 30 of A (memory only, {os.environ.get('JM_RX', 30)} steps, {os.environ.get('JM_RXMODE', 'ul')}): retracted " + f"{fmt(b4[0])} -> {fmt(af[0])} | bystanders: rest of A {fmt(b4[1])} -> {fmt(af[1])}, "
      f"B {fmt(b4[2])} -> {fmt(af[2])}, M {fmt(b4[3])} -> {fmt(af[3])}" + (f" | slots erased {M.erased}" if hasattr(M, 'erased') else ''))
    with torch.no_grad(): M.v.copy_(snap)
    if CAPF:
        bs = [CAPF[k:k + 200] for k in range(0, len(CAPF), 200)]
        for b_ in bs: M.teach(b_, rr)
        P("           CAPACITY: +" + f"{len(CAPF)} facts in batches of 200 -> each batch at the end " + " | ".join(fmt(M.recall(b_)) for b_ in bs)
          + f" || A {fmt(M.recall(SET_A))}, B {fmt(M.recall(SET_B))}, M {fmt(M.recall(SET_M))} | slots written {len(M.touched)} of {NS * NS}  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
