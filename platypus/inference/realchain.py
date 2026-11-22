"""22 Nov REAL-TEXT CHAINS (inf_rchain): the rule world's composition test moved to real text.

Containment chains are built from the records the separate record store reads in DocRED's distant set (premises.py: 'A <in-wording> B',
'B <in-wording> C', the most frequent wording and partner of each head; optionally only pairs carrying given gold relations - an oracle reader).
The trunk never reads the distant documents, so it knows only what is taught here, by the interleaved question rows:
  - PREMISES of every chain (train and test), each in inf_rchain_frames wordings ('A in B', 'Records show that A in B'): the overnight
    synthetic finding - a premise learned in one wording is keyed on that wording and is not retrieved inside a chain; two wordings suffice;
  - CHAINS of the train split only, in the arrow form 'A in B > C.' (the form that composes through new bridges; the spelled form does not);
  - optional JEPA STEP-2 ALIGNMENT (inf_rchain_align): the state where 'A in B > ' predicts C pulled to the state of 'B <w2> ' (stop-gradient).
Split by BRIDGE: chains whose bridge hashes into the test share, and that never appears in any taught chain, are never taught as chains.
READOUT [REAL-TEXT CHAINS, THE TRUNK'S OWN]: greedy 'A <composite relation> ' (store and memory off) parsed 'B > C' - chain format, bridge right, C right - for
taught chains and for TEST chains (composition through bridges never in a taught chain), the direct answer ('A in ' starting with C), and the
premises asked alone (first and second hop)."""
import collections, zlib
import numpy as np
import torch
import torch.nn.functional as F

_RC = {}
_CW = {'in', 'is in', 'located in', 'is located in', 'part of', 'is part of', 'city in', 'town in', 'village in', 'region of', 'province of',
       'capital of', 'state of'}


def _hash(s, salt):
    return zlib.crc32((s + '|' + salt).encode('utf-8', 'replace')) % 1000


def rchain_build(cfg, vis):
    """chains from the records of the distant documents read so far -> (train, test); cached per visit"""
    if _RC.get('vis') == vis: return _RC['train'], _RC['test']
    from platypus.inference.premises import _PM, _load_distant, _rnorm
    _load_distant(cfg); R = _PM.get('R') or []; ND = int(_PM.get('ND', 0)); PRD = _PM.get('PRD') or {}
    nv = max(1, int(getattr(cfg, 'inf_premises_visits', 10))); ndv = int(round(ND * min(1.0, (vis + 1) / nv)))
    orc = set(x for x in str(getattr(cfg, 'inf_rchain_oracle', '') or '').split(',') if x)
    HT = collections.defaultdict(collections.Counter); TY0 = _PM.get('TY') or {}; loc0 = int(getattr(cfg, 'inf_rchain_loc', 1))
    for r in R:
        if r[0] >= ndv or not (1 <= r[5] <= 8) or r[1] == r[2]: continue
        if loc0 and (TY0.get(r[2]) != 'LOC' or TY0.get(r[1]) in ('TIME', 'NUM', 'PER')): continue   # the partner a place, the head not a date, number or person
        w = _rnorm(r[3])
        if w in _CW and (not orc or (PRD.get((r[1], r[2])) or set()) & orc): HT[r[1]][(w, r[2])] += 1
    CL = []; TY = _PM.get('TY') or {}; loc = int(getattr(cfg, 'inf_rchain_loc', 1))   # bridge and end must be places (the record names' automatic type):
    for A_, c in HT.items():
        (w1, B_), _ = c.most_common(1)[0]
        if B_ not in HT: continue
        (w2, C_), _ = HT[B_].most_common(1)[0]
        if C_ in (A_, B_) or len(f"Records show that {A_} {w1} {B_} > {C_}.") > 110: continue
        if loc and not (TY.get(B_) == 'LOC' and TY.get(C_) == 'LOC'): continue
        CL.append((A_, w1, B_, w2, C_))
    tq = float(getattr(cfg, 'inf_rchain_testq', 0.3))
    BT = {c[2] for c in CL if _hash(c[2], 'rcb') < tq * 1000}
    trn = sorted([c for c in CL if c[2] not in BT], key=lambda c: _hash(c[0], 'rct'))[:int(getattr(cfg, 'inf_rchain_ntr', 400))]
    inch = {x for c in trn for x in (c[0], c[2], c[4])}
    tst = sorted([c for c in CL if c[2] in BT and c[2] not in inch], key=lambda c: _hash(c[0], 'rce'))[:int(getattr(cfg, 'inf_rchain_test', 150))]
    _RC.update(vis=vis, train=trn, test=tst, pairs={(a, t) for a, c in HT.items() for (w, t) in c})   # every recorded containment link (any wording)
    gl = [bool((PRD.get((c[0], c[2])) or set()) & {'P131', 'P17', 'P30', 'P361', 'P276', 'P150'}) and bool((PRD.get((c[2], c[4])) or set()) & {'P131', 'P17', 'P30', 'P361', 'P276', 'P150'}) for c in tst]
    print(f"  [REAL-TEXT CHAINS | visit {vis + 1}] {ndv} distant documents read: {len(CL)} containment chains{' (oracle ' + ','.join(sorted(orc)) + ')' if orc else ''} "
          f"-> TAUGHT {len(trn)} (arrow form) | TEST {len(tst)} (bridge never in a taught chain; premises taught, chain never) "
          f"({np.mean(gl) if gl else float('nan'):.2f} of test chains have both premises gold-labelled as containment; evaluation only) "
          f"e.g. '{trn[0][0]} in {trn[0][2]} > {trn[0][4]}.', '{trn[1][0]} in {trn[1][2]} > {trn[1][4]}.', '{trn[2][0]} in {trn[2][2]} > {trn[2][4]}.'" if len(trn) > 2 else f"  [REAL-TEXT CHAINS | visit {vis + 1}] {len(CL)} chains, none taught yet", flush=True)
    return trn, tst


def _crel(cfg):
    return str(getattr(cfg, 'inf_rchain_crel', '') or 'in')                   # the composite's relation ('in' = kra32)


_PARAW = ['is located in', 'lies in', 'is part of', 'is situated in', 'is a place in', 'belongs to']   # 22 Nov kra38: containment paraphrases for premise rows
# (kra36 visits 3-5: inside the chain 'A is within ' the bridge is right 0.26-0.29 while the first premise asked alone is right 0.60-0.69 -
#  the premise is retrieved only in the wordings it was learned in; never 'is within' itself, the composite relation's own words)


def _frames(A, w, B, k):
    fs = [f"{A} {w} ", f"Records show that {A} {w} ", f"It is known that {A} {w} "][:max(1, k)]
    return [(f.encode('latin-1', 'replace'), (B + '.').encode('latin-1', 'replace')) for f in fs]


def rchain_rows(cfg, vis, sl):
    """question rows for the next visit: premise rows (all chains, several wordings), chain rows (train), alignment pairs (train)"""
    trn, tst = rchain_build(cfg, vis); k = int(getattr(cfg, 'inf_rchain_frames', 2)); P, C, A = [], [], []
    seen = set()
    s1 = int(getattr(cfg, 'inf_rchain_s1frame', 0)); cr1 = _crel(cfg)
    for (A_, w1, B_, w2, C_) in trn + tst:
        if s1 and (A_, B_) not in seen:                                        # 22 Nov kra41 STEP-1 FRAME: the first premise also taught in the chain's own
            seen.add((A_, B_))                                                 #   step-1 context, ending in the arrow ('A is within ' -> 'B >'): the bridge
            P.extend([(f"{A_} {cr1} ".encode('latin-1', 'replace'), f"{B_} >".encode('latin-1', 'replace'), sl)] * s1)   #   retrieved where the chain needs it,
                                                                               #   the chain format kept (kra32's clash ended in '.'), C never shown (kra36:
                                                                               #   bridge in the chain 0.26-0.32 vs the premise alone 0.60-0.78)
        for (h, w, t) in ((A_, w1, B_), (B_, w2, C_)):
            if (h, w, t) in seen: continue
            seen.add((h, w, t)); P.extend((p, a, sl) for p, a in _frames(h, w, t, k))
            P.extend([(f"{h} > ".encode('latin-1', 'replace'), (t + '.').encode('latin-1', 'replace'), sl)] * int(getattr(cfg, 'inf_rchain_arrowprem', 0)))   # 22 Nov arrow premise frame
            for pw in [x for x in _PARAW if x != _crel(cfg).strip()][:int(getattr(cfg, 'inf_rchain_paraw', 0))]:   # 22 Nov kra38 PARAPHRASED CONTAINMENT WORDINGS
                P.append((f"{h} {pw} ".encode('latin-1', 'replace'), (t + '.').encode('latin-1', 'replace'), sl))   #   (the rule world's premw: 1 wording 0.15-0.19, 4 -> 0.55-0.67)
    for (A_, w1, B_, w2, C_) in trn:                                           # 22 Nov kra32 visits 4-5: with the chain asked as 'A in ' - the SAME prompt as a first premise
        r1 = _crel(cfg)                                                        #   worded 'in' - test heads answer their premise ('A in B.', format 0.20) and never chain;
                                                                               #   the composite gets its own relation (inf_rchain_crel, e.g. 'is within'), as the rule world's
        C.append((f"{A_} {r1} ".encode('latin-1', 'replace'), f"{B_} > {C_}.".encode('latin-1', 'replace'), sl))
        A.append((f"{A_} {r1} {B_} > ".encode('latin-1', 'replace'), f"{B_} {w2} ".encode('latin-1', 'replace')))
        if int(getattr(cfg, 'inf_rchain_align_step', 2)) == 3:                # 22 Nov: STEP 1 too - 'A is within ' (predicts B) ~ 'A <w1> ' (its premise)
            A.append((f"{A_} {r1} ".encode('latin-1', 'replace'), f"{A_} {w1} ".encode('latin-1', 'replace')))
    return dict(prem=P, chain=C, align=A, sl=sl)


def rchain_loss(model, cfg, R, rs):
    """this step's rows: half premise rows, half chain rows; plus the alignment term. Returns (loss or None, ce, align)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    from platypus.inference.distill import _ws_hidden
    n = int(getattr(cfg, 'inf_rchain_rows', 8)); P, C = R.get('prem') or [], R.get('chain') or []
    if not P: return None, None, None
    it = [P[int(i)] for i in rs.choice(len(P), min(n - n // 2 if C else n, len(P)), replace=False)]
    if C: it += [C[int(i)] for i in rs.choice(len(C), min(n // 2, len(C)), replace=False)]
    sl = int(R['sl']); X, _ = _ws_batch(model, [p + a for p, a, _s in it]); lg = _ws_run(model, X, sl, grad_=True); t, c = 0.0, 0
    for j, (p, a, _s) in enumerate(it):
        p0 = len(p) - 1; n_ = min(len(a), lg.shape[1] - p0)
        if n_ > 0: t = t + F.cross_entropy(lg[j, p0:p0 + n_].float(), X[j, p0 + 1:p0 + 1 + n_], reduction='sum'); c += n_
    if not c: return None, None, None
    loss = t / c; ce = float(loss.detach()); al = None
    aw = float(getattr(cfg, 'inf_rchain_align', 0.0)); AL = R.get('align') or []
    if aw > 0 and AL:
        vv = [AL[int(i)] for i in rs.choice(len(AL), min(8, len(AL)), replace=False)]
        Xc, _ = _ws_batch(model, [q for q, p in vv]); Xp, _ = _ws_batch(model, [p for q, p in vv])
        _, Hc = _ws_hidden(model, lambda: _ws_run(model, Xc, sl, grad_=True))
        with torch.no_grad(): _, Hp = _ws_hidden(model, lambda: _ws_run(model, Xp, sl))
        ok = [j for j, (q, p) in enumerate(vv) if Hc is not None and Hp is not None and len(q) - 1 < Hc.shape[1] and len(p) - 1 < Hp.shape[1]]
        if ok:
            hc = torch.stack([Hc[j, len(vv[j][0]) - 1] for j in ok]).float(); hp = torch.stack([Hp[j, len(vv[j][1]) - 1] for j in ok]).float()
            a_ = align_loss(hc, hp, cfg); loss = loss + aw * a_; al = float(a_.detach())
    return loss, ce, al


def align_loss(hc, hp, cfg):
    """22 Nov JEPA step-2 alignment loss between chain states hc (grad) and premise states hp (stop-gradient), by inf_align_mode:
    cos - 1 - cosine (as before); nce - contrastive (InfoNCE over the batch, temperature 0.1): each chain state must pick ITS premise's
    state among the batch's, so the identity directions are matched rather than the shared high-variance ones (synthetic, 150 countries:
    new bridge 0.68 / 0.77 / 0.64 vs cosine 0.61 / 0.69 / 0.53); res - residual: both views centred on their batch mean first"""
    m = str(getattr(cfg, 'inf_align_mode', 'cos'))
    if m == 'nce' and len(hc) > 1:
        sim = F.normalize(hc, dim=-1) @ F.normalize(hp, dim=-1).T / 0.1
        return F.cross_entropy(sim, torch.arange(len(hc), device=hc.device))
    if m == 'res' and len(hc) > 1:
        hc = hc - hc.mean(0, keepdim=True); hp = hp - hp.mean(0, keepdim=True)
    return (1 - F.cosine_similarity(hc, hp, dim=-1)).mean()

def _greedy_batch(model, prompts, sl, n=60):
    from platypus.inference.closed_form import _ws_batch, _ws_run
    bs = [bytearray(p) for p in prompts]; done = [False] * len(bs)
    for _ in range(n):
        live = [j for j in range(len(bs)) if not done[j]]
        if not live: break
        X, _m = _ws_batch(model, [bytes(bs[j]) + b' ' for j in live])
        with torch.no_grad(): lg = _ws_run(model, X, sl)
        for r, j in enumerate(live):
            p = len(bs[j]) - 1
            if p >= lg.shape[1]: done[j] = True; continue
            ch = int(lg[r, p].argmax())
            if ch in (46, 10): done[j] = True
            else: bs[j].append(ch)
    return [bytes(b[len(p):]).decode('latin-1', 'replace').strip() for b, p in zip(bs, prompts)]


def rchain_readout(model, cfg, vis, sl, D=None):
    """the trunk's own chain on taught and TEST real-text chains (store and memory off)"""
    trn, tst = _RC.get('train') or [], _RC.get('test') or []
    if not (trn or tst): return
    was = model.training; model.eval()
    try:
        def score(L):
            if not L: return "0", []
            g = _greedy_batch(model, [f"{c[0]} {_crel(cfg)} ".encode('latin-1', 'replace') for c in L], sl, 70)
            fm, br, cr, dr = [], [], [], []
            for x, c in zip(g, L):
                pt = [s.strip() for s in x.split(' > ')]
                fm.append(len(pt) == 2 and all(pt)); br.append(len(pt) == 2 and pt[0] == c[2]); cr.append(len(pt) == 2 and pt[1] == c[4]); dr.append(x.startswith(c[4]))
            h1 = _greedy_batch(model, [f"{c[0]} {c[1]} ".encode('latin-1', 'replace') for c in L], sl, 40)
            h2 = _greedy_batch(model, [f"{c[2]} {c[3]} ".encode('latin-1', 'replace') for c in L], sl, 40)
            p1 = np.mean([x.split(' > ')[0].strip() == c[2] for x, c in zip(h1, L)]); p2 = np.mean([x.split(' > ')[0].strip() == c[4] for x, c in zip(h2, L)])
            mc = collections.Counter(c[4] for c in (_RC.get('train') or [])).most_common(1); mcv = np.mean([c[4] == mc[0][0] for c in L]) if mc else float('nan')
            cb = [y for y, b_ in zip(cr, br) if b_]
            from platypus.inference.premises import _PM as _PM8                # 22 Nov GOLD: chains whose conclusion A -> C is a gold containment relation (Wikidata labels
            PRD8 = _PM8.get('PRD') or {}; GC8 = {'P131', 'P17', 'P30', 'P361', 'P276', 'P150', 'P706'}   #   in DocRED; evaluation only): right answers that are TRUE
            gt = [bool((PRD8.get((c[0], c[4])) or set()) & GC8) for c in L]; gcr = [y for y, g_ in zip(cr, gt) if g_]
            gold_ = f"; GOLD-TRUE conclusions {len(gcr)}: C right {np.mean(gcr):.2f}" if gcr else "; GOLD-TRUE conclusions 0"
            try:
                PRS = _RC.get('pairs') or set(); va, vc, wg, wn = [], [], [], 0                # 22 Nov kra39: ANY VALID CHAIN - test heads often have several recorded
                for x, c in zip(g, L):                                                 #   containers (e.g. 'Vancouver International Airport -> British Columbia >
                    pt = [s_.strip() for s_ in x.split(' > ')]                         #   Canada' counted wrong against the record's 'Richmond'); a written chain is
                    if len(pt) != 2 or not all(pt): continue                           #   VALID when both of its links are recorded (any wording), and its
                    wn += 1; b_ok = (c[0], pt[0]) in PRS; va.append(b_ok); vc.append(b_ok and (pt[0], pt[1]) in PRS)   #   conclusion GOLD-TRUE when Wikidata
                    g8 = PRD8.get((c[0], pt[1])) or set()                              #   labels A -> C as containment (evaluation only)
                    if g8 or (c[0], pt[1]) in PRD8: wg.append(bool(g8 & GC8))
                gold_ += (f"; ANY VALID CHAIN (both links recorded, any wording; of {wn} well-formed): first link recorded {np.mean(va) if va else float('nan'):.2f}, "
                          f"whole chain recorded {np.mean(vc) if vc else float('nan'):.2f}, written conclusion gold-labelled {len(wg)}: true containment {np.mean(wg) if wg else float('nan'):.2f}")
            except Exception as e_:                                                # readout only: never stop the run
                gold_ += f"; ANY VALID CHAIN readout failed ({type(e_).__name__}: {e_})"
            return (f"{len(L)}: chain format {np.mean(fm):.2f}, bridge right {np.mean(br):.2f}, C right {np.mean(cr):.2f} (given the right bridge {np.mean(cb) if cb else float('nan'):.2f}; "
                    f"guessing the commonest taught C {mcv:.2f}; C first, direct {np.mean(dr):.2f}{gold_}) | "
                    f"premises asked alone: first hop {p1:.2f}, second hop {p2:.2f}"), g
        st, _g1 = score(trn[:80]); se, g2 = score(tst[:150])
        ex = " | ".join(f"{c[0]} -> {x[:60]} [{c[2]} > {c[4]}]" for c, x in list(zip(tst, g2))[:4])
        cel = (D or {}).get('rchl') or []; all_ = (D or {}).get('rcha') or []
        print(f"  [REAL-TEXT CHAINS, THE TRUNK'S OWN | visit {vis + 1}] greedy 'A {_crel(cfg)} ' (store and memory off), parsed 'B > C': TAUGHT {st} || "
              f"TEST, bridge never in a taught chain {se}"
              + (f" || rows this visit: loss {np.mean(cel[:max(1, len(cel) // 10)]):.3f} -> {np.mean(cel[-max(1, len(cel) // 10):]):.3f} ({len(cel)} steps)" if cel else "")
              + (f", JEPA alignment 1 - cos {np.mean(all_[:max(1, len(all_) // 10)]):.3f} -> {np.mean(all_[-max(1, len(all_) // 10):]):.3f}" if all_ else ""), flush=True)
        print(f"      TEST e.g. {ex}", flush=True)
    finally:
        model.train(was)
