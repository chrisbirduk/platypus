"""AN INTEGRATED KEYED SPARSE MEMORY: addressed by SUBJECT NAME x JEPA WORDING KEY, read INSIDE the trunk's forward pass (15 Nov 2026;
tools/int_mem.py) - offline, no effect on the package.

The author's design, integrated: a product-key memory whose address is computed by the model from its own context at every byte, and
whose read is added to the trunk's last residual (before the final norm and head), so it changes the bytes the model predicts.
  SUBJECT      the latest COMPLETED capitalised run of 3+ bytes before the current position (causal: a run counts once the byte after it has
               been read); its NAME KEY is the trunk's last block over the name in 'About NAME.' (mean, centred; cached per name)
  WORDING      the bytes from the subject's end up to the current position - or, inside a capitalised run that started after the subject
               (an answer being spelled), up to that run's start; its JEPA KEY is a small head over the trunk's middle-layer states on
               those bytes (mean + last), trained JEPA/InfoNCE-style so the system's seven TRAINING frames of one relation phrase meet
               (different subjects, different frames); OFFSET = bytes of the answer run read so far (0 at its first byte), added to the
               wording key, so each byte of an answer has its own slots
  MEMORY       NS x NS slots, top 8; half 1 = a fixed projection of the subject's name key, half 2 = wording key + offset; a VALUE (width d)
               is added to the last residual where a subject is active (wording 1-48 bytes, offset <= 31). Values start at zero: before
               writing, the model IS the trunk. Writing = the LM loss, only the memory values learn (only the slots read change)
Test as tools/sparse_mem.py's mixed setting (its numbers for comparison): 120 never-read Re-DocRED facts, set A then set B, 8 fact rows +
24 wiki rows a step, 400 steps a set; recall through the WHOLE MODEL in the TEST wording ('It is on record that A r ', never a training
frame): RANK (the answer's bytes most likely among 10 of its type, by mean bits; chance 0.10) and OPEN (greedy bytes = the exact name);
the PARAPHRASED relation phrase; FORGETTING = bits added on held-out wiki and Re-DocRED dev text (the memory active); ERASE: the slots
read where 30 facts' training rows predict their answers set to zero - their recall and the bystanders'. Arms: ALL ROWS WRITE (wiki rows' gradients reach the
memory too: the memory also learns from text) and FACT ROWS WRITE (wiki rows read but do not write).
IM_GATE=1 (15 Nov): at evaluation the memory is read only where its address was written - the subject is one the fact rows wrote for AND
half the read weight is on slots they read (= wrote) - on ALL text: does the gate remove the cost on ordinary text?
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/int_mem.py   env IM_N (128), IM_STEPS (400), IM_LR (0.01,0.03), IM_SEED (0), IM_ARMS, IM_GATE, IM_LGATE (1: a LEARNED gate trained by the trunk's loss on all rows), IM_LGREG (0.02)"""
import os, re, json, math, time, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
P = lambda *a: print(*a, flush=True); T0 = time.time()
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
DW = int(os.environ.get('TG_D', 192)); CK = os.environ['TG_CKPT']; torch.set_num_threads(int(os.environ.get('IM_THREADS', 1)))
L = 128; PAD = 256; V = 257; SEED = int(os.environ.get('IM_SEED', 0)); rs = np.random.RandomState(SEED); torch.manual_seed(SEED)
LGATE = int(os.environ.get('IM_LGATE', 0)); LGREG = float(os.environ.get('IM_LGREG', 0.02)); NS = int(os.environ.get('IM_N', 128)); KK = 8; DK = 32; STEPS = int(os.environ.get('IM_STEPS', 400)); MAXW, MAXO = 48, 31
FORMS = ("In short, {A} {r} ", "Notes on {A}: {A} {r} ", "{A}, as noted before, {r} ", "As for {A}, people say {A} {r} ",
         "Q: {A} {r} whom? A: ", "{A}, whose name came up earlier in the report, {r} ", "From the notes: {A} {r} ")
TEST = "It is on record that {A} {r} "
REL = {'P17': 'is in the country', 'P131': 'is located in', 'P27': 'is a citizen of', 'P19': 'was born in', 'P20': 'died in', 'P108': 'worked for',
       'P54': 'played for', 'P463': 'is a member of', 'P102': 'is a member of the party', 'P175': 'was performed by', 'P495': 'comes from',
       'P159': 'is based in', 'P50': 'was written by', 'P57': 'was directed by', 'P361': 'is part of', 'P150': 'contains', 'P527': 'includes',
       'P740': 'was formed in', 'P276': 'took place in', 'P26': 'was married to', 'P69': 'was educated at'}
PARA = {'was born in': 'has as birthplace', 'died in': 'passed away in', 'is located in': 'lies in', 'is a citizen of': 'holds citizenship of',
        'worked for': 'was employed by', 'played for': 'was a player for', 'was written by': 'was authored by', 'was directed by': 'had as director',
        'is part of': 'belongs to', 'was educated at': 'studied at', 'is in the country': 'is situated in the country', 'comes from': 'originates from',
        'is based in': 'has headquarters in', 'was married to': 'was the spouse of', 'took place in': 'happened in', 'is a member of': 'belongs to the group',
        'was performed by': 'was sung by', 'contains': 'has within it', 'includes': 'comprises', 'was formed in': 'was founded in',
        'is a member of the party': 'belongs to the party'}
RELS = sorted(set(REL.values()))
tr_docs = json.load(open(SCR + '/bench_dl/train_revised.json')); dv_docs = json.load(open(SCR + '/bench_dl/dev_revised.json'))
cn_ = lambda d: [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
tp_ = lambda d: [vs[0].get('type', 'X') for vs in d['vertexSet']]
dev_names = {n for d in dv_docs for n in cn_(d)}
FACTS, BYT, used = [], collections.defaultdict(set), set()
for d in tr_docs:
    for n, t in zip(cn_(d), tp_(d)): BYT[t].add(n)
for di in rs.permutation(len(tr_docs)):                                    # = tools/sparse_mem.py's facts (same seed, same rule)
    d = tr_docs[int(di)]; cn, tp = cn_(d), tp_(d)
    for l in d['labels']:
        A, C = cn[l['h']], cn[l['t']]
        if l['r'] not in REL or A in used or not (3 <= len(A) <= 30 and 3 <= len(C) <= 30) or not all(ord(ch) < 256 for ch in A + C): continue
        if A in dev_names and C in dev_names: continue
        FACTS.append(dict(A=A, C=C, rel=REL[l['r']], t=tp[l['t']])); used.add(A); break
    if len(FACTS) >= 120: break
RUN = re.compile(rb"[A-Z][A-Za-z0-9'\-\.]*(?: [A-Z][A-Za-z0-9'\-\.]*)*")
NF0 = len(FACTS); FACTS = [f for f in FACTS if RUN.fullmatch(f['A'].encode('latin-1'))]   # the subject must be readable as a capitalised run
SET_A, SET_B = FACTS[:len(FACTS) // 2], FACTS[len(FACTS) // 2:]
wiki = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin', 'rb').read().replace(b'\n', b' '); WK = np.frombuffer(wiki, dtype=np.uint8)
HO = len(WK) - 200000
dev_txt = ' '.join(' '.join(' '.join(s) for s in d['sents']) for d in dv_docs[:200]).encode('latin-1', 'replace')
EV_W = [bytes(WK[HO + k * L: HO + (k + 1) * L]) for k in range(96)]; EV_D = [dev_txt[k * L:(k + 1) * L] for k in range(96)]

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
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]; hs = []
        for b in s.b: x = b(x, am); hs.append(x)
        return hs
lm = LM(); lm.load_state_dict(torch.load(CK)); lm.eval()
for p in lm.parameters(): p.requires_grad_(False)
def tens(seqs):
    X = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
    return X
def xin(X): return torch.cat([torch.full((X.shape[0], 1), PAD), X[:, :-1]], 1)

RUN = re.compile(rb"[A-Z][A-Za-z0-9'\-\.]*(?: [A-Z][A-Za-z0-9'\-\.]*)*")
def parse(b):
    """per output position t (the model has read b[:t]): (subject name, wording start, wording end, offset) or None"""
    runs = [(m.start(), m.end()) for m in RUN.finditer(b[:L])]; out = [None] * L
    for t in range(1, min(L, len(b) + 1)):
        subj = None; cur = None
        for (s0, e0) in runs:
            if s0 >= t: break
            if t >= e0 + 2 and e0 - s0 >= 3: subj = (s0, e0)                 # completed (the byte after it has been read)
            elif s0 < t < e0 + 2: cur = s0                                   # a run in progress
        if subj is None: continue
        we = cur if (cur is not None and cur > subj[1]) else t; off = (t - cur) if (cur is not None and cur > subj[1]) else 0
        ws = subj[1]
        if not (1 <= we - ws <= MAXW) or off > MAXO: continue
        out[t] = (b[subj[0]:subj[1]].decode('latin-1'), ws, we, off)
    return out
NKC = {}; NMU = None
@torch.no_grad()
def name_keys(names):
    new = [n for n in names if n not in NKC]
    for i0 in range(0, len(new), 128):
        ch = new[i0:i0 + 128]; X = tens([f"About {n}.".encode('latin-1', 'replace') for n in ch]); H = lm.states(xin(X))[-1]
        for j, n in enumerate(ch): NKC[n] = H[j, 7:7 + len(n.encode('latin-1', 'replace'))].mean(0)   # output index 7..: read 'About ' + name bytes
    K = torch.stack([NKC[n] for n in names]); return F.normalize(K - NMU, dim=-1) if NMU is not None else K
NMU = torch.stack([name_keys([n])[0] for n in sorted({f['A'] for f in FACTS} | {f['C'] for f in FACTS})]).mean(0)
NMU = torch.stack(list(NKC.values())).mean(0)

@torch.no_grad()
def feats(seqs):
    """trunk states and, per gated position, (row, t, subject, middle-layer window mean + last state, offset)"""
    X = tens(seqs); hs = lm.states(xin(X)); H1 = hs[1]; G = []
    for r, b in enumerate(seqs):
        cs = torch.cumsum(H1[r], 0)
        for t, a in enumerate(parse(b)):
            if a is None: continue
            nm, ws, we, off = a; lo, hi = ws + 1, we                       # output indices whose state has read the window's bytes
            mean = (cs[hi] - (cs[lo - 1] if lo >= 1 else 0)) / (hi - lo + 1); G.append((r, t, nm, torch.cat([mean, H1[r, hi]]), off))
    return X, hs, G
# the JEPA wording head: the seven training frames of one relation phrase meet (different subjects); the test wording never seen
pre = [(f, k, FORMS[k].format(A=f['A'], r=f['rel']).encode('latin-1')) for f in FACTS for k in range(len(FORMS))]
_, _, G0 = feats([p for _, _, p in pre])
last = {}
for r, t, nm, ft, off in G0: last[r] = ft                                  # the last gated position of each prefix = the full wording
FT = torch.stack([last[i] for i in range(len(pre))]); FMU, FSD = FT.mean(0), FT.std(0) + 1e-6
head = nn.Sequential(nn.Linear(FT.shape[1], 128), nn.GELU(), nn.Linear(128, DK)); oh = torch.optim.Adam(head.parameters(), lr=1e-3)
byrel = collections.defaultdict(list)
for i, (f, k, p) in enumerate(pre): byrel[f['rel']].append(i)
rl = [r for r in RELS if len(byrel[r]) >= 2]; hr = np.random.RandomState(SEED + 7)
for st in range(2000):
    a_, b_ = [], []
    for r in rl:
        i = byrel[r][hr.randint(len(byrel[r]))]; cand = [j for j in byrel[r] if pre[j][1] != pre[i][1] and pre[j][0] is not pre[i][0]]
        a_.append(i); b_.append(cand[hr.randint(len(cand))] if cand else i)
    z1 = F.normalize(head((FT[a_] - FMU) / FSD), dim=-1); z2 = F.normalize(head((FT[b_] - FMU) / FSD), dim=-1)
    lo = F.cross_entropy(z1 @ z2.t() / 0.1, torch.arange(len(rl))); oh.zero_grad(); lo.backward(); oh.step()
for p in head.parameters(): p.requires_grad_(False)
@torch.no_grad()
def wacc(form):
    tf = [(f, form.format(A=f['A'], r=(PARA[f['rel']] if form == 'PARA' else f['rel'])).encode('latin-1')) for f in FACTS] if form != 'PARA' else \
         [(f, TEST.format(A=f['A'], r=PARA[f['rel']]).encode('latin-1')) for f in FACTS]
    _, _, Gt = feats([p for _, p in tf]); lt = {}
    for r, t, nm, ft, off in Gt: lt[r] = ft
    Z = F.normalize(head((torch.stack([lt[i] for i in range(len(tf))]) - FMU) / FSD), dim=-1)
    ZR = F.normalize(head((FT - FMU) / FSD), dim=-1); cen = torch.stack([ZR[byrel[r]].mean(0) if byrel[r] else torch.zeros(DK) for r in RELS])
    return float(np.mean([RELS[int((Z[i] @ cen.t()).argmax())] == f['rel'] for i, (f, p) in enumerate(tf)]))
P(f"facts A {len(SET_A)}, B {len(SET_B)} ({NF0 - len(FACTS)} of sparse_mem's {NF0} dropped: subject not a capitalised run); memory {NS}x{NS} slots, top {KK}; JEPA wording head (in-context, subjects not masked): the relation of the "
  f"TEST wording {wacc(TEST):.2f}, of the PARAPHRASED wording {wacc('PARA'):.2f} (chance {1 / len(rl):.2f})  ({time.time() - T0:.0f}s)")

g = torch.Generator().manual_seed(SEED + 3)
K1 = F.normalize(torch.randn(NS, DK, generator=g), dim=-1); K2 = F.normalize(torch.randn(NS, DK, generator=g), dim=-1)
R1 = torch.randn(DW, DK, generator=g) / DW ** 0.5; EOFF = F.normalize(torch.randn(MAXO + 1, DK, generator=g), dim=-1)
class IMem:
    def __init__(s, lr):
        s.v = torch.zeros(NS * NS, DW, requires_grad=True); s.opt = torch.optim.Adam([s.v], lr=lr)
        s.gate = False; s.wmask = torch.zeros(NS * NS, dtype=torch.bool); s.subj = set(); s.opened = [0, 0]   # 15 Nov GATE (IM_GATE)
        s.lg = None                                                        # 15 Nov LEARNED GATE (IM_LGATE): features of the read -> open (0..1), trained by the
        if LGATE:                                                          #   trunk's own loss on ALL rows (does reading here help the prediction?)
            s.lg = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1)); nn.init.constant_(s.lg[2].bias, -2.0)
            s.lgo = torch.optim.Adam(s.lg.parameters(), lr=3e-3); s.gsum = []
    def logits(s, seqs, write_rows=None, rec=None):
        X, hs, G = feats(seqs); xl = hs[-1].clone()
        if G:
            nk = name_keys([g_[2] for g_ in G]); wk = F.normalize(head((torch.stack([g_[3] for g_ in G]) - FMU) / FSD), dim=-1)
            wk = F.normalize(wk + 0.5 * EOFF[[g_[4] for g_ in G]], dim=-1); q1 = F.normalize(nk @ R1, dim=-1)
            a1, i1 = (q1 @ K1.t()).topk(KK, -1); a2, i2 = (wk @ K2.t()).topk(KK, -1)
            sc = (a1[:, :, None] + a2[:, None, :]).flatten(1); ix = (i1[:, :, None] * NS + i2[:, None, :]).flatten(1)
            top, j = sc.topk(KK, -1); idx = ix.gather(-1, j); w = F.softmax(top / 0.05, -1)
            vv = s.v[idx]
            if write_rows is not None:                                     # what the writing rows read is what they write: the gate's index
                for k, g_ in enumerate(G):
                    if g_[0] in write_rows: s.wmask[idx[k]] = True; s.subj.add(g_[2])
            if write_rows is not None:                                     # rows that may not write: their reads carry no gradient
                wm = torch.tensor([float(g_[0] in write_rows) for g_ in G])[:, None, None]; vv = vv * wm + vv.detach() * (1 - wm)
            val = (w[..., None] * vv).sum(1)
            if s.lg is not None:                                           # LEARNED GATE: written-slot weight, subject written, read confidence, offset
                wf = (w * s.wmask[idx].float()).sum(-1); sw = torch.tensor([float(g_[2] in s.subj) for g_ in G])
                fe = torch.stack([wf, sw, w.max(-1)[0], top.mean(-1), torch.tensor([g_[4] / MAXO for g_ in G], dtype=torch.float32), wf * sw], -1)
                gv = torch.sigmoid(s.lg(fe.detach()))[:, 0]; val = val * gv[:, None]; s.gsum.append(gv)
                if rec is not None and rec and isinstance(rec[0], str): pass
            elif s.gate:                                                   # read only where the address was written: the subject, and half the weight
                op = torch.tensor([float(g_[2] in s.subj) for g_ in G]) * ((w * s.wmask[idx].float()).sum(-1) >= 0.5).float()
                val = val * op[:, None]; s.opened[0] += int(op.sum()); s.opened[1] += len(G)
            if rec is not None: rec.extend((g_[0], g_[1], idx[k].tolist()) for k, g_ in enumerate(G))
            rows = torch.tensor([g_[0] for g_ in G]); ts = torch.tensor([g_[1] for g_ in G])
            xl = xl.index_put((rows, ts), val, accumulate=True)
        return X, lm.o(lm.n(xl))
    def loss(s, seqs, write_rows=None):
        X, lo = s.logits(seqs, write_rows); tg = X.reshape(-1); lo = lo.reshape(-1, V); return F.cross_entropy(lo[tg != PAD], tg[tg != PAD])
@torch.no_grad()
def bits(M, seqs):
    t = 0.0
    for k in range(0, len(seqs), 32): t += float(M.loss(seqs[k:k + 32])) * len(seqs[k:k + 32])
    return t / len(seqs) / math.log(2)
@torch.no_grad()
def recall(M, facts, form=TEST, para=False):
    rr = np.random.RandomState(SEED + 5); oe, rk = [], []
    for f in facts:
        pb = form.format(A=f['A'], r=(PARA[f['rel']] if para else f['rel'])).encode('latin-1'); tb = (f['C'] + '.').encode('latin-1')
        alts = sorted(n for n in BYT[f['t']] if n != f['C'] and 3 <= len(n) <= 30 and all(ord(ch) < 256 for ch in n))
        cands = [f['C']] + [alts[j] for j in rr.choice(len(alts), 9, replace=False)]
        seqs = [pb + c.encode('latin-1') + b'.' for c in cands]; X, lg = M.logits(seqs)
        nl = -F.log_softmax(lg, -1).gather(-1, X[..., None])[..., 0]; sc = [float(nl[k, len(pb):len(q)].mean()) for k, q in enumerate(seqs)]
        rk.append(int(sc[0] < min(sc[1:])))
        out = b''                                                          # greedy, byte by byte, through the whole model
        for _ in range(len(tb)):
            X, lg = M.logits([pb + out]); out += bytes([int(lg[0, len(pb) + len(out)].argmax())])
        oe.append(int(out[:-1] == tb[:-1]))
    return float(np.mean(oe)), float(np.mean(rk))
wb = lambda rr, n: [bytes(WK[i0:i0 + L]) for i0 in rr.randint(0, HO - L - 1, n)]
GATE = int(os.environ.get('IM_GATE', 0))
_bits0, _recall0 = bits, recall
def bits(M, seqs):
    M.gate = bool(GATE); M.opened = [0, 0]
    try: return _bits0(M, seqs)
    finally: M.gate = False
def recall(M, facts, form=TEST, para=False):
    M.gate = bool(GATE)
    try: return _recall0(M, facts, form, para)
    finally: M.gate = False
def teach(M, facts, rr, fact_only):
    for st in range(STEPS):
        fb = [facts[rr.randint(len(facts))] for _ in range(8)]
        seqs = [(FORMS[rr.randint(len(FORMS))].format(A=f['A'], r=f['rel']) + f['C'] + '.').encode('latin-1') for f in fb] + wb(rr, 24)
        if M.lg is not None: M.gsum = []
        lo = M.loss(seqs, write_rows=set(range(8)) if fact_only else None)
        if M.lg is not None and M.gsum: lo = lo + LGREG * torch.cat(M.gsum).mean()   # a small price for opening: open only where it pays
        M.opt.zero_grad()
        if M.lg is not None: M.lgo.zero_grad()
        lo.backward(); M.opt.step()
        if M.lg is not None: M.lgo.step()
NEVER = None
@torch.no_grad()
def abstain(M, n=60):
    """questions about NEVER-READ names (syllables) in the test wording: the share of answer-start reads the gate opens (learned: g > 0.5)"""
    rr = np.random.RandomState(SEED + 77); syl = ['ka', 'lo', 'mi', 'ner', 'tas', 'vol', 'ru', 'den', 'sa', 'por', 'el', 'ith']; op = []
    for k in range(n):
        nm = (''.join(rr.choice(syl, 2)) + ' ' + ''.join(rr.choice(syl, 3))).title(); f = FACTS[rr.randint(len(FACTS))]
        pb = TEST.format(A=nm, r=f['rel']).encode('latin-1'); rec = []
        if M.lg is not None:
            M.gsum = []; M.logits([pb + b'Xx.'], rec=rec)
            if M.gsum and rec: op.append(float(M.gsum[-1][[i for i, (r_, t_, ix_) in enumerate(rec) if t_ == len(pb)] or [0]].max() > 0.5))
        else:
            M.gate = True; M.opened = [0, 0]; M.logits([pb + b'Xx.']); op.append(float(M.opened[0] > 0)); M.gate = False
    return float(np.mean(op)) if op else float('nan')
fmt = lambda x: f"open {x[0]:.2f} rank {x[1]:.2f}"
M0 = IMem(0.01); b0w, b0d = bits(M0, EV_W), bits(M0, EV_D)
P(f"before writing (the memory empty = the trunk): A {fmt(recall(M0, SET_A[:30]))} | held-out bits wiki {b0w:.3f}, dev {b0d:.3f}  ({time.time() - T0:.0f}s)")
ARMS = [(a, float(lr)) for a in os.environ.get('IM_ARMS', 'FACT ROWS WRITE,ALL ROWS WRITE').split(',') for lr in os.environ.get('IM_LR', '0.01,0.03').split(',')]
for arm, lr in ARMS:
    M = IMem(lr); rr = np.random.RandomState(SEED + 11); fo = arm.startswith('FACT') or GATE
    _bits, _recall = bits, recall
    teach(M, SET_A, rr, fo); ra = recall(M, SET_A); w1, d1 = bits(M, EV_W), bits(M, EV_D)
    teach(M, SET_B, rr, fo); rb = recall(M, SET_B); raa = recall(M, SET_A); w2, d2 = bits(M, EV_W), bits(M, EV_D)
    pa = recall(M, SET_A + SET_B, para=True)
    P(f"   {arm:16s} lr {lr:g}: A after A {fmt(ra)} | B after B {fmt(rb)} | A after B {fmt(raa)} | PARAPHRASED (A+B) {fmt(pa)} || FORGETTING (bits added) "
      f"wiki {w1 - b0w:+.3f} -> {w2 - b0w:+.3f}, dev {d1 - b0d:+.3f} -> {d2 - b0d:+.3f} | slots written {int((M.v.detach().abs().sum(1) > 0).sum())} of {NS * NS}"
      + (f" | GATE: open on {M.opened[0] / max(1, M.opened[1]):.3f} of the dev-text positions with a subject ({len(M.subj)} subjects, {int(M.wmask.sum())} slots written)" if GATE and not LGATE else '')
      + (f" | LEARNED GATE: never-read names open {abstain(M):.2f}" if LGATE else (f" | never-read names open {abstain(M):.2f}" if GATE else ''))
      + f"  ({time.time() - T0:.0f}s)")
    RX, KEEP = SET_A[:30], SET_A[30:]; snap = M.v.detach().clone(); b4 = [recall(M, x) for x in (RX, KEEP, SET_B)]; rec = []
    with torch.no_grad():
        ids = set()
        for f in RX:                                                       # the slots read where each training row's ANSWER is predicted
            rows = [(FORMS[k].format(A=f['A'], r=f['rel']) + f['C'] + '.').encode('latin-1') for k in range(len(FORMS))]; rec = []
            M.logits(rows, rec=rec); ln = [len(FORMS[k].format(A=f['A'], r=f['rel']).encode('latin-1')) for k in range(len(FORMS))]
            for r_, t_, ix_ in rec:
                if ln[r_] <= t_ <= ln[r_] + len(f['C']): ids.update(ix_)
        ids = torch.tensor(sorted(ids)); M.v[ids] = 0.0
    af = [recall(M, x) for x in (RX, KEEP, SET_B)]
    P(f"      ERASE 30 of A ({len(ids)} slots read where their training rows' answers are predicted, set to zero): retracted {fmt(b4[0])} -> {fmt(af[0])} | rest of A {fmt(b4[1])} -> "
      f"{fmt(af[1])}, B {fmt(b4[2])} -> {fmt(af[2])} | wiki bits {w2 - b0w:+.3f} -> {bits(M, EV_W) - b0w:+.3f}  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
