"""DISTILLATION IN SLEEP (inf_distill, 10 Nov; off by default). The memory's facts are taught to the trunk while it sleeps - the
complementary-learning-systems loop (the hippocampus teaches the cortex). Each sleep, BEFORE the name memory is rebuilt: inf_distill_steps
steps on the trunk's blocks only (heads, private tables, gates untouched), each step one slot: inf_distill_rows questions about stored
facts and conclusions (the name memory's material: live conclusions and name-to-name facts seen at least twice) in SELF-WRITTEN wordings -
some with text between the name and the answer, so the subject has to be carried - none of them a test wording; target = the stored
answer, loss on the answer bytes only; both memories off. PROTECTION: anchor rows (stored real sentences of the same slot) keep the
model's outputs on ordinary text at what they were before this sleep's distillation (KL, weight inf_distill_anchor); a BRAKE measures
the visited domain's and the old domains' BPC before and after and rolls the whole distillation back if either rises by more than
inf_distill_brake. CONTROL: a fixed fifth of the facts (by a hash of the fact) is never taught. READOUT: recall from the trunk (every
memory off) in the TEST wordings for a sample of the taught facts and of the never-taught control, before and after; across sleeps, [FROM THE WEIGHTS, FIXED SET] (with and without the closed-form edits) is the consolidation measure."""
import time
import re
import collections
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import _fail, _HO_TWO

_DT_FORMS = ("In short, {A} {r} ", "Notes on {A}: {A} {r} ", "{A}, as noted before, {r} ", "As for {A}, people say {A} {r} ",
             "Q: {A} {r} whom? A: ", "{A}, whose name came up earlier in the report, {r} ", "From the notes: {A} {r} ")
_DT_PARA = {'works with': 'is a colleague of', 'lives near': 'is a neighbour of', 'reports to': 'is managed by', 'is a partner of': 'is partnered with',
            'trained under': 'was taught by', 'is connected to': 'is tied to'}
_DT_REFUSE = 'unknown'                                                   # 12 Nov REFUSAL ROWS (inf_distill_refuse): the answer to a question about a name
_DT_REFUSE_SYL = ['bo', 'ne', 'ri', 'so', 'pa', 'gu', 'lin', 'mor', 'tes', 'vad', 'hel', 'zo']   #   never read; fresh names from syllables the false-memory
_DT_TEST = ("{A} is tied to ", "The one tied to {A} is ", "It is on record that {A} {r} ")   # never taught (checked below)
_DT_FORMS2 = ("Earlier we saw that {A} {r} ", "It turns out {A} {r} ", "Remember: {A} {r} ", "According to the file, {A} {r} ",
              "Of all people, {A} {r} ", "Ask anyone and they will say {A} {r} ", "{A} (mentioned above) {r} ", "Fact: {A} {r} ",
              "The notes about {A} say {A} {r} ", "{A}; in other words, {A} {r} ", "Some say that {A} {r} ", "Plainly, {A} {r} ",
              "Going back to {A}: {A} {r} ", "{A}, it seems, {r} ")   # 12 Nov: inf_distill_forms > 7 adds these (none a test wording)


def _dt_forms(cfg):
    """the self-written question wordings in use: the original 7, + up to 14 more (inf_distill_forms)"""
    n = max(1, min(len(_DT_FORMS) + len(_DT_FORMS2), int(getattr(cfg, 'inf_distill_forms', 7))))
    fs = (_DT_FORMS + _DT_FORMS2)[:n]
    assert all('tied to' not in f and 'on record' not in f and 'We read' not in f for f in fs)   # no taught wording is a test wording
    return fs


def _dw_src_bytes(S_, e, n=2):
    """the fact's stored source sentence(s) as bytes (up to n)"""
    it9 = S_.get('it', []); ss = []
    for i9 in (e.get('src') or ())[:n]:
        if isinstance(i9, (int, np.integer)) and 0 <= int(i9) < len(it9):
            b9 = it9[int(i9)]['s']
            b9 = bytes(int(v) & 255 for v in b9.tolist()) if hasattr(b9, 'tolist') and not isinstance(b9, (bytes, bytearray)) else (b9.encode('latin-1', 'replace') if isinstance(b9, str) else bytes(b9))
            ss.append(b9.strip())
    return ss


def _dw_decl(S_, e, L0, src=True):
    """12 Nov VIEW ALIGNMENT: the fact's DECLARATION and the position whose state predicts the answer's first byte: the stored source
    sentence cut after the answer (the answer found after a space, after the subject when the subject is there; trimmed from the left to
    the window), else the plain 'A r ans.'. Returns (bytes, position, 'src' | 'plain')"""
    ab = e['ans'].encode('latin-1', 'replace'); Ab = e['A'].encode('latin-1', 'replace')
    if src:
        for sb in _dw_src_bytes(S_, e):
            i = sb.find(b' ' + ab, max(0, sb.find(Ab)))
            if i < 1 or (Ab not in sb[:i + 1] and sb.find(Ab) >= 0): continue
            d = sb[:i + 1 + len(ab)]; cut = max(0, len(d) - (L0 - 2)); d = d[cut:]; pos = i - cut
            if pos >= 1: return d, pos, 'src'
    pb = f"{e['A']} {e['r']} ".encode('latin-1', 'replace'); return pb + ab + b'.', len(pb) - 1, 'plain'


def _ws_hidden(model, fn):
    """(fn(), the last block's output states during it) - the trunk's state that the view alignment compares"""
    H = {}

    def hk(m_, i_, o_): H['h'] = o_[0] if isinstance(o_, tuple) else o_
    h_ = model.blocks[-1].register_forward_hook(hk)
    try: out = fn()
    finally: h_.remove()
    return out, H.get('h')


@torch.no_grad()
def _dw_align_probe(model, S_, S_slot, es, src):
    """for taught facts and the never-taught control (never aligned): cosine between the last-block state where a TEST-wording question
    predicts its answer and the state where the fact's declaration predicts it - 'It is on record that A r' (it ends in the declaration's
    own 'A r ', so it is near 1 by construction) and the paraphrased relation ('A is a colleague of', a different ending: the informative
    one); and RECALL IN THE READING FORM: the trunk completes the fact's own declaration (every answer byte top-1 after its source
    sentence's prefix) - whether reading put the fact in the weights at all, which alignment can only route to, not create"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    from platypus.inference.jepa import cfg_seq_
    L0 = int(cfg_seq_(model)); c1, c2, rd, c0, pl = [], [], [], [], []; prev = None   # c0: the 'on record' question vs ANOTHER fact's declaration
    for e in es:
        if e['key'][0] == 'conclusion': continue
        sl = int(S_slot(e['d'])); d, pos, _ = _dw_decl(S_, e, L0, src); ab = e['ans'].encode('latin-1', 'replace')
        Xd, _ = _ws_batch(model, [d]); lg, hd = _ws_hidden(model, lambda: _ws_run(model, Xd, sl))
        if hd is None or pos >= hd.shape[1]: continue
        n_ = min(len(ab), lg.shape[1] - pos)
        if n_ == len(ab): rd.append(bool((lg[0, pos:pos + n_].argmax(-1) == torch.tensor(list(ab), device=lg.device)).all()))
        pp = f"{e['A']} {e['r']} ".encode('latin-1', 'replace'); Xp, _ = _ws_batch(model, [pp + ab + b'.']); lp = _ws_run(model, Xp, sl)   # PLAIN 'A r ' (no sentence context)
        if len(pp) - 1 + len(ab) <= lp.shape[1]: pl.append(bool((lp[0, len(pp) - 1:len(pp) - 1 + len(ab)].argmax(-1) == torch.tensor(list(ab), device=lp.device)).all()))
        qs = [(c1, f"It is on record that {e['A']} {e['r']} ")] + ([(c2, f"{e['A']} {_DT_PARA[e['r']]} ")] if e['r'] in _DT_PARA else [])
        for acc, q in qs:
            pb = q.encode('latin-1', 'replace'); Xq, _ = _ws_batch(model, [pb + ab + b'.']); _, hq = _ws_hidden(model, lambda: _ws_run(model, Xq, sl))
            if hq is not None and len(pb) - 1 < hq.shape[1]:
                acc.append(float(F.cosine_similarity(hq[0, len(pb) - 1].float(), hd[0, pos].float(), dim=0)))
                if acc is c1 and prev is not None: c0.append(float(F.cosine_similarity(hq[0, len(pb) - 1].float(), prev, dim=0)))
        prev = hd[0, pos].float()
    m = lambda v: f"{np.mean(v):.3f} ({len(v)})" if v else "nan (0)"
    return (f"cosine, 'on record' wording {m(c1)} (vs ANOTHER fact's declaration {m(c0)}), PARAPHRASED relation {m(c2)}; "
            f"recall completing its own declaration {m(rd)}, completing plain 'A r' {m(pl)}")


def _dt_recall_para(model, S_slot, es):
    """open-ended recall from the trunk in the PARAPHRASED test wording only ('A is a colleague of'; facts whose relation has one)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    isl = lambda b: 65 <= b <= 90 or 97 <= b <= 122; ok = []
    for e in es:
        if e['r'] not in _DT_PARA: continue
        pb = f"{e['A']} {_DT_PARA[e['r']]} ".encode('latin-1', 'replace'); tb = (e['ans'] + '.').encode('latin-1', 'replace'); X, _ = _ws_batch(model, [pb + tb])
        with torch.no_grad(): lg = _ws_run(model, X, int(S_slot(e['d'])))[0]
        p0 = len(pb) - 1; ps = torch.arange(p0, p0 + len(tb), device=lg.device)
        if int(ps[-1]) >= lg.shape[0]: continue
        top = lg[ps].argmax(-1); tr = torch.tensor(list(tb), device=lg.device)
        ok.append(bool((top[:-1] == tr[:-1]).all()) and not isl(int(top[-1])))
    return (float(np.mean(ok)) if ok else float('nan')), len(ok)


_BR_T = {'P19': "{h} was born in ", 'P20': "{h} died in ", 'P27': "{h} is a citizen of ", 'P69': "{h} was educated at ", 'P54': "{h} played for ",
         'P463': "{h} is a member of ", 'P50': "{h} was written by ", 'P57': "{h} was directed by ", 'P175': "{h} was performed by ",
         'P264': "{h} was released on the label ", 'P571': "{h} was founded in ", 'P577': "{h} was first published in ", 'P569': "{h} was born on ",
         'P570': "{h} died on ", 'P131': "{h} is located in ", 'P17': "{h} is in the country ", 'P150': "{h} contains ", 'P36': "The capital of {h} is ",
         'P102': "{h} is a member of the party ", 'P108': "{h} worked for ", 'P26': "{h} was married to ", 'P40': "A child of {h} is ",
         'P22': "The father of {h} is ", 'P3373': "A sibling of {h} is ", 'P1344': "{h} took part in ", 'P361': "{h} is part of ",
         'P527': "{h} includes ", 'P159': "The headquarters of {h} are in ", 'P740': "{h} was formed in ", 'P276': "{h} took place in "}


@torch.no_grad()
def _dw_cand_mc(model, sl, prompt, cands):
    """index of the candidate with the best MEAN log-probability per byte after PROMPT (store and memory off)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    pb = prompt.encode('latin-1', 'replace'); cb = [(c + '.').encode('latin-1', 'replace') for c in cands]
    X, _ = _ws_batch(model, [pb + c for c in cb]); lp = F.log_softmax(_ws_run(model, X, sl).float(), -1); sc = []
    for j, c in enumerate(cb):
        p0 = len(pb) - 1; n_ = min(len(c), lp.shape[1] - p0)
        sc.append(float(lp[j, torch.arange(p0, p0 + n_, device=lp.device), X[j, p0 + 1:p0 + 1 + n_]].mean()) if n_ > 0 else -1e9)
    return int(np.argmax(sc))


def _rtx_rep(d, di):
    """12 Nov REAL-TEXT REPLAY: which benchmark documents are replayed (half, by a hash of domain and document index); the rest are the control"""
    import zlib
    return zlib.crc32(f"rtx|{int(d)}|{int(di)}".encode()) % 2 == 0


def _rtx_slot(cfg, S_slot, d, seen, ask=False):
    """13 Nov (inf_distill_rtx_slot): the slot benchmark documents are replayed through AND asked through. -1 = the slot of the domain the
    document was assigned to (the benchmark's documents are Wikipedia text split across domains 0-4; cg27a_rtxt: Gutenberg's slot replayed 53 of
    them, Gutenberg forgetting +0.224); k >= 0 = always domain k's slot (1 = Wikipedia), None while k is not yet visited (no replay then)"""
    k = int(getattr(cfg, 'inf_distill_rtx_slot', -1))
    if k < 0 or (ask and k not in seen): return int(S_slot(d))              #   (questions fall back to the document's own slot until k is visited)
    return int(S_slot(k)) if k in seen else None


def _dw_bench_recall(model, cfg, S_slot, vis):
    """12 Nov READOUT: REAL-TEXT RECALL FROM THE WEIGHTS - Re-DocRED gold relations of the benchmark documents the model has READ (planted
    once in pass 1, in visited domains), whose two names stand in one sentence, asked in a templated test wording per relation type
    ('X was born in'); the true object against up to 9 other entity names of the same type from the same domain's documents (mean
    log-prob per byte; chance ~0.1-0.2), and the same question with the subject replaced by a fresh name (the relation prior alone).
    With inf_distill_rtx (real-text replay of half the documents), split into REPLAYED documents and the never-replayed CONTROL half."""
    from platypus.inference.probes import _bench
    G = _bench(cfg); seen = set(int(k) for k in (model.__dict__.get('_t2s') or {}).keys()); rs = np.random.RandomState(int(cfg.seed) + 4242 + vis)
    nm = G['ent_name']; ty = G.get('ent_type', {}); items = []; rtx = int(getattr(cfg, 'inf_distill_rtx', 0)) > 0
    for d in sorted(seen):
        dl = G['docs'].get(str(d), [])
        if not dl: continue
        inD = set(e for e, n in nm.items() if any(n in t for t in dl))
        for di, doc in enumerate(dl):
            sents = re.split(r'(?<=[.!?])\s+', doc); inX = set(e for e in inD if nm[e] in doc)
            for h, t, fl, ps in G['rel']:
                if str(h) not in inX or str(t) not in inX: continue
                pc = next((p for p in ps if p in _BR_T), None)
                if pc is None: continue
                H, T = nm[str(h)], nm[str(t)]
                if len(H) < 3 or len(T) < 3 or not any(H in s_ and T in s_ for s_ in sents): continue
                same = [nm[e] for e in inD if e not in (str(h), str(t)) and ty.get(e) == ty.get(str(t)) and len(nm[e]) >= 3]
                if len(same) >= 3: items.append((d, pc, H, T, same, _rtx_rep(d, di) if rtx else False))
    if not items: print(f"  [REAL-TEXT RECALL | visit {vis + 1}] no readable benchmark relations in the visited domains yet", flush=True); return
    items = list({(x[0], x[2], x[3]): x for x in items}.values())
    fresh = lambda: ''.join(rs.choice(['ka', 'to', 'mi', 'ra', 've', 'lo', 'du', 'sen', 'tar', 'bel'], 3)).capitalize()
    def score(L):
        L = [L[int(i)] for i in rs.permutation(len(L))[:64]]; ok = []; pri = []
        for d, pc, H, T, same, rp in L:
            oth = [same[int(i)] for i in rs.permutation(len(same))[:9]]; cands = [T] + oth; sl = _rtx_slot(cfg, S_slot, d, seen, ask=True)
            ok.append(_dw_cand_mc(model, sl, _BR_T[pc].format(h=H), cands) == 0); pri.append(_dw_cand_mc(model, sl, _BR_T[pc].format(h=fresh()), cands) == 0)
        ch = np.mean([1.0 / (1 + min(9, len(x[4]))) for x in L]) if L else float('nan')
        return (f"the true object among up to 10 same-type names {np.mean(ok) if ok else float('nan'):.2f} | with a FRESH subject (relation prior alone) "
                f"{np.mean(pri) if pri else float('nan'):.2f} | chance {ch:.2f} ({len(L)} asked)"), L
    pcs = collections.Counter(x[1] for x in items).most_common(4)
    if rtx:
        a, _ = score([x for x in items if x[5]]); b, _ = score([x for x in items if not x[5]])
        print(f"  [REAL-TEXT RECALL | visit {vis + 1}] benchmark relations in documents READ (two names in one sentence; commonest {pcs}): FROM THE WEIGHTS, "
              f"REPLAYED documents: {a} || CONTROL documents (read once, never replayed): {b}", flush=True)
    else:
        a, _ = score(items)
        print(f"  [REAL-TEXT RECALL | visit {vis + 1}] benchmark relations in documents READ (two names in one sentence; commonest {pcs}): FROM THE WEIGHTS, {a}", flush=True)


_RTX_PRON = ('He ', 'She ', 'It ', 'They ', 'His ', 'Her ', 'Its ', 'Their ', 'The ')


def _dw_bench_diag(model, cfg, S_slot, vis):
    """12 Nov READOUT: REAL-TEXT RECALL DIAGNOSTICS - why recall sits near the prior. Benchmark relations of documents read, from ONE
    sentence: NAMED (both names in the sentence, subject first) or PRONOUN-LED (the subject is the document's title, the sentence opens with
    He / She / It / They / The ...). Per group (replayed / control documents x named / pronoun-led): STORAGE - the trunk completes the
    object after the sentence's own prefix, as read (every byte top-1); OWN-FRAME recall - the question in the sentence's own words with the
    subject named ('Thatcher, a native of'); TEMPLATE recall ('Thatcher was born in') and the FRESH-subject prior; the last three among the
    same 10 candidates (true object + 9 same-type names of the domain)."""
    from platypus.inference.probes import _bench
    from platypus.inference.closed_form import _ws_batch, _ws_run
    from platypus.inference.jepa import cfg_seq_
    G = _bench(cfg); seen = set(int(k) for k in (model.__dict__.get('_t2s') or {}).keys()); rs = np.random.RandomState(int(cfg.seed) + 4646 + vis)
    nm = G['ent_name']; ty = G.get('ent_type', {}); L0 = int(cfg_seq_(model)); rtx = int(getattr(cfg, 'inf_distill_rtx', 0)) > 0; items = []
    PR = ('He ', 'She ', 'It ', 'They ', 'His ', 'Her ', 'Its ', 'Their ')
    for d in sorted(seen):
        dl = G['docs'].get(str(d), [])
        inD = set(e for e, n in nm.items() if any(n in t for t in dl))
        for di, doc in enumerate(dl):
            sents = re.split(r'(?<=[.!?])\s+', doc); inX = set(e for e in inD if nm[e] in doc)
            for h, t, fl, ps in G['rel']:
                if str(h) not in inX or str(t) not in inX: continue
                pc = next((p for p in ps if p in _BR_T), None)
                if pc is None: continue
                H, T = nm[str(h)], nm[str(t)]
                if len(H) < 3 or len(T) < 3: continue
                same = [nm[e] for e in inD if e not in (str(h), str(t)) and ty.get(e) == ty.get(str(t)) and len(nm[e]) >= 3]
                if len(same) < 3: continue
                got = None
                for s_ in sents:
                    if T not in s_: continue
                    j = s_.index(T)
                    if H in s_ and s_.index(H) + len(H) < j:
                        mid = s_[s_.index(H) + len(H):j]
                        if 0 < len(mid.split()) <= 15: got = ('named', s_[:j], H + mid); break
                    elif H not in s_ and doc.startswith(H) and (s_.startswith(PR) or s_.startswith('The ')) and j > 0:
                        cut = s_.index(' ') if s_.startswith(PR) else s_.index(' ', 4) if ' ' in s_[4:] else -1
                        if 0 < cut < j and len(s_[cut:j].split()) <= 15: got = ('pronoun-led', s_[:j], H + s_[cut:j]); break
                if got: items.append((d, pc, H, T, same, _rtx_rep(d, di) if rtx else False, got[0], got[1], got[2]))
    if not items: return
    items = list({(x[0], x[2], x[3]): x for x in items}.values())
    fresh = lambda: ''.join(rs.choice(['ka', 'to', 'mi', 'ra', 've', 'lo', 'du', 'sen', 'tar', 'bel'], 3)).capitalize()
    def score(L):
        L = [L[int(i)] for i in rs.permutation(len(L))[:48]]; st, own, tpl, pri = [], [], [], []
        for d, pc, H, T, same, rp, kd, pre, oq in L:
            sl = _rtx_slot(cfg, S_slot, d, seen, ask=True); oth = [same[int(i)] for i in rs.permutation(len(same))[:9]]; cands = [T] + oth
            pb = pre.encode('latin-1', 'replace')[-(L0 - 2 - len(T)):]; tb = T.encode('latin-1', 'replace'); X, _ = _ws_batch(model, [pb + tb + b' '])
            with torch.no_grad(): lg = _ws_run(model, X, sl)[0]
            p0 = len(pb) - 1
            if p0 + len(tb) <= lg.shape[0]: st.append(bool((lg[p0:p0 + len(tb)].argmax(-1) == torch.tensor(list(tb), device=lg.device)).all()))
            own.append(_dw_cand_mc(model, sl, oq.rstrip() + ' ', cands) == 0); tpl.append(_dw_cand_mc(model, sl, _BR_T[pc].format(h=H), cands) == 0)
            pri.append(_dw_cand_mc(model, sl, _BR_T[pc].format(h=fresh()), cands) == 0)
        m = lambda v: f"{np.mean(v):.2f}" if v else "nan"
        return f"storage {m(st)} | own-frame {m(own)} | template {m(tpl)} | prior {m(pri)} ({len(L)})"
    parts = []
    for rp in ((True, False) if rtx else (False,)):
        for kd in ('named', 'pronoun-led'):
            L = [x for x in items if x[5] == rp and x[6] == kd]
            if L: parts.append(f"{'REPLAYED' if rp else ('CONTROL' if rtx else 'ALL')}, {kd}: {score(L)}")
    print(f"  [REAL-TEXT RECALL, DIAGNOSTICS | visit {vis + 1}] from the weights; storage = completes the object after its own sentence; own-frame / template / prior "
          f"= the true object among up to 10 same-type names (chance ~0.10) || " + " || ".join(parts), flush=True)
_RTX_LEAD = {'In', 'On', 'At', 'After', 'During', 'Before', 'Since', 'From', 'For', 'By', 'As', 'A', 'An', 'When', 'While', 'Although', 'However', 'Following', 'Between', 'Despite', 'Under', 'With', 'Of', 'This', 'These', 'That', 'There'}
_RTX_FORMS = ("In short, ", "From the notes: ", "As noted before, ", "It turns out ")


def _rtx_views(sent, title, L0, maxw=12):
    """12 Nov REAL-TEXT VIEWS: (question, declaration, position) pairs cut from one sentence - subject = the opening capitalised run, or
    the document's TITLE for a sentence led by a pronoun / 'The'; object = the next capitalised run within maxw words; declaration = the
    sentence up to the object (its state predicts the object); question = a self-written lead-in + subject + the words between (never a
    test template: always a lead-in)"""
    out = []
    caps = [(m.start(), m.end()) for m in re.finditer(r"(?:[A-Z][\w'\-]*)(?:\s+(?:of|de|the|and|von|van|la|le)?\s*[A-Z][\w'\-]*)*", sent)]
    if not caps: return out
    if caps[0][0] == 0 and not sent.startswith(_RTX_PRON):
        subj = sent[caps[0][0]:caps[0][1]]; rest0 = caps[0][1]; objs = caps[1:]
        if subj.split()[0] in _RTX_LEAD: return out                         #   'In 1979 ...', 'After the war ...': no subject
    elif sent.startswith(_RTX_PRON) and title: subj = title; rest0 = sent.index(' ') if sent.startswith(_RTX_PRON[:4]) else sent.index(' ', 4) if ' ' in sent[4:] else 0; objs = [c for c in caps if c[0] > rest0]
    else: return out
    for k, (o0, o1) in enumerate(objs[:3]):
        mid = sent[rest0:o0].strip()
        if not mid or len(mid.split()) > maxw or len(mid) < 2: continue
        decl = sent[:o0].encode('latin-1', 'replace')
        if len(decl) < 8: continue
        decl = decl[-(L0 - 2):]; pos = len(decl) - 1
        if decl[-1:] != b' ': continue
        q = (_RTX_FORMS[k % len(_RTX_FORMS)] + subj + ' ' + mid + ' ').encode('latin-1', 'replace')
        out.append((q, decl, pos))
    return out


def _rtx_weights(model, rtx, alpha, vis, bs=64):
    """12 Nov REPLAY BY STORAGE CHECK (inf_distill_rtx_space): the trunk's mean next-byte loss (store and memory off) on every replayed
    real sentence of each slot; replay probability ~ clip(loss, q10, q90) ** alpha, so sentences the trunk already predicts (stored) are
    replayed less and the ones it does not, more (prioritised replay; Schaul et al. 2016, Mattar & Daw 2018). Prints the loss spread, the
    share of replay going to the hardest and the easiest quarter (uniform 0.25), and how last visit's hardest quarter moved."""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    H = model.__dict__.setdefault('_rtx_hist', {}); W = {}; allL = []; easy = hard = 0.0; nsl = 0; mv = []
    prevhard = model.__dict__.get('_rtx_hard') or set(); newhard = set()
    for sl, ss in rtx.items():
        Ls = []
        with torch.no_grad():
            for i in range(0, len(ss), bs):
                X, M = _ws_batch(model, ss[i:i + bs]); lg = _ws_run(model, X, sl); T = lg.shape[1]; m = M[:, 1:T + 1].float()
                ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), X[:, 1:T + 1].reshape(-1), reduction='none').view(m.shape)
                Ls.extend(((ce * m).sum(1) / m.sum(1).clamp(min=1)).cpu().tolist())
        Ls = np.asarray(Ls, float); lo, hi = np.quantile(Ls, 0.1), np.quantile(Ls, 0.9); w = np.clip(Ls, lo, hi) ** float(alpha)
        w = w / w.sum() if w.sum() > 0 else np.full(len(Ls), 1.0 / len(Ls)); W[sl] = w; allL.extend(Ls.tolist()); nsl += 1
        o = np.argsort(Ls); q = max(1, len(Ls) // 4); hard += float(w[o[-q:]].sum()); easy += float(w[o[:q]].sum())
        for j in o[-q:]: newhard.add(hash(ss[int(j)]))
        for j, x in enumerate(ss):
            k = hash(x)
            if k in prevhard and k in H: mv.append(Ls[j] - H[k])
            H[k] = float(Ls[j])
    model.__dict__['_rtx_hard'] = newhard
    if allL:
        print(f"  [DISTILLATION, REPLAY BY STORAGE CHECK | visit {vis + 1}] the trunk's loss on {len(allL)} replayed sentences (nats per byte): median {np.median(allL):.3f}, "
              f"q10 {np.quantile(allL, 0.1):.3f}, q90 {np.quantile(allL, 0.9):.3f} | replay share to the HARDEST quarter {hard / max(1, nsl):.2f}, to the EASIEST "
              f"{easy / max(1, nsl):.2f} (uniform 0.25; alpha {alpha}) | last visit's hardest quarter: loss change {np.mean(mv) if mv else float('nan'):+.3f} ({len(mv)} sentences)", flush=True)
    return W


def _dw_false_probe(model, cfg, S_slot, ents, held, vis):
    """12 Nov READOUT: FALSE MEMORIES - questions about facts NEVER read: (a) a FRESH name with a known relation ('It is on record that
    Kamisen works with'), (b) a KNOWN subject with a relation it has no fact for. Greedy answer from the trunk (store and memory off): the
    share that is a STORED name (a confabulation drawn from memory), for (b) also the share that is the subject's object for ANOTHER
    relation (leakage across relations), and the mean probability of the answer's first byte (confidence)."""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    rs = np.random.RandomState(int(cfg.seed) + 4343 + vis); fs = [e for e in ents + held if e['key'][0] == 'fact']
    if len(fs) < 8: return
    known = set(e['A'] for e in fs) | set(e['ans'] for e in fs); rels = sorted(set(e['r'] for e in fs)); bysub = collections.defaultdict(dict)
    for e in fs: bysub[e['A']][e['r']] = e['ans']
    def ask(A, r, d):
        sl = int(S_slot(d)); pb = f"It is on record that {A} {r} ".encode('latin-1', 'replace'); a = _dw_greedy(model, pb, sl, 20)
        X, _ = _ws_batch(model, [pb + b'x'])
        with torch.no_grad(): lg = _ws_run(model, X, sl)[0, len(pb) - 1].float()
        return a, float(F.softmax(lg, -1).max())
    nq = int(getattr(cfg, 'inf_distill_false_probe_n', 32)); ref = lambda a: a.lower().startswith(_DT_REFUSE)
    A_ = []; Aa = []                                                          # (answer strings kept for the ABSTENTION readout below)
    for _ in range(nq):
        e = fs[rs.randint(len(fs))]; nm = ''.join(rs.choice(['ka', 'to', 'mi', 'ra', 've', 'lo', 'du', 'sen', 'tar', 'bel', 'nor', 'qui'], rs.randint(2, 4))).capitalize()
        if nm in known: continue
        a, c = ask(nm, e['r'], e['d']); A_.append((a in known, c)); Aa.append(a)
    B_ = []; Ba = []
    subs = [s for s in bysub if len(set(rels) - set(bysub[s]))]
    for i in rs.permutation(len(subs))[:nq]:
        A = subs[int(i)]; r = rels[rs.randint(len(rels))]
        if r in bysub[A]: continue
        d = next(e['d'] for e in fs if e['A'] == A); a, c = ask(A, r, d); B_.append((a in known, a in bysub[A].values(), c)); Ba.append(a)
    K_ = []; Ka = []                                                          # (c) KNOWN facts (taught): is the trunk MORE CONFIDENT where it knows? - the
    tf = [e for e in ents if e['key'][0] == 'fact']                           #   same question form and the same confidence measure (first answer byte)
    for i in rs.permutation(len(tf))[:nq]:
        e = tf[int(i)]; a, c = ask(e['A'], e['r'], e['d']); K_.append((a == e['ans'], c)); Ka.append(a)
    kinfo = ""
    if K_:
        kc = [c for ok_, c in K_ if ok_]; kw = [c for ok_, c in K_ if not ok_]; un = [x[1] for x in A_] + [x[2] for x in B_]
        au = float('nan')
        if kc and un:
            r_ = np.concatenate([kc, un]).argsort().argsort().astype(float) + 1; au = (r_[:len(kc)].sum() - len(kc) * (len(kc) + 1) / 2) / (len(kc) * len(un))
        kinfo = (f" || (c) KNOWN (taught) facts ({len(K_)}): answered right {np.mean([ok_ for ok_, c in K_]):.2f}, confidence when RIGHT "
                 f"{np.mean(kc) if kc else float('nan'):.2f}, when wrong {np.mean(kw) if kw else float('nan'):.2f} | CONFIDENCE TELLS KNOWN-RIGHT FROM NEVER-READ: AUC {au:.2f}"
                 f" (0.5 = no; the trunk could abstain on its own only if this is high)")
    f = lambda L, k: f"{np.mean([x[k] for x in L]):.2f}" if L else "nan"
    print(f"  [DISTILLATION, FALSE MEMORY PROBE | visit {vis + 1}] questions about facts NEVER read, from the trunk: (a) FRESH name, known relation ({len(A_)}): answer is a "
          f"STORED name {f(A_, 0)}, confidence {f(A_, 1)} || (b) KNOWN subject, a relation it has no fact for ({len(B_)}): answer is a STORED name {f(B_, 0)}, "
          f"its object for ANOTHER relation {f(B_, 1)}, confidence {f(B_, 2)}" + kinfo, flush=True)
    if K_ and (A_ or B_):                                                     # 12 Nov ABSTENTION (readout): could the trunk refuse by itself? It ABSTAINS when
        try:                                                                  #   it answers the refusal word, or when its confidence is below a threshold t.
            kr = [(ok_, c, ref(a)) for (ok_, c), a in zip(K_, Ka)]            #   t is set to keep a share of the known facts it answers RIGHT; reported: what
            un = [(st, c, ref(a)) for (st, c), a in zip(A_, Aa)] + [(x[0], x[2], ref(a)) for x, a in zip(B_, Ba)]   # share of never-read questions still get
            kc = np.array([c for ok_, c, rf in kr if ok_ and not rf])          #   a stored name (false memories let through), and the known facts' accuracy
            parts = []                                                         #   among those answered (selective accuracy)
            for keep in (0.9, 0.75, 0.5):
                if not len(kc): break
                t = float(np.quantile(kc, 1 - keep))
                ans_k = [(ok_, c) for ok_, c, rf in kr if not rf and c >= t]; leak = np.mean([bool(st) and not rf and c >= t for st, c, rf in un])
                parts.append(f"keep {keep:.0%} of known-right (t {t:.2f}): known answered {len(ans_k) / len(kr):.2f}, accuracy when answered "
                             f"{np.mean([ok_ for ok_, c in ans_k]) if ans_k else float('nan'):.2f}, NEVER-READ still given a stored name {leak:.2f}")
            print(f"  [DISTILLATION, ABSTENTION | visit {vis + 1}] from the trunk, known facts ({len(kr)}) vs never-read questions ({len(un)}; (a) + (b) above): "
                  f"EXPLICIT REFUSAL ('{_DT_REFUSE}') on known facts {np.mean([rf for ok_, c, rf in kr]):.2f}, on (a) fresh names "
                  f"{np.mean([ref(a) for a in Aa]) if Aa else float('nan'):.2f}, on (b) missing relations {np.mean([ref(a) for a in Ba]) if Ba else float('nan'):.2f} | "
                  f"WITHOUT ABSTENTION: never-read given a stored name {np.mean([bool(st) and not rf for st, c, rf in un]):.2f}, known right "
                  f"{np.mean([ok_ for ok_, c, rf in kr]):.2f} || CONFIDENCE THRESHOLD: " + " | ".join(parts), flush=True)
        except Exception as e9:
            _fail('inference.distill:abstain'); print(f"  [DISTILLATION, ABSTENTION] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
        try:                                                                  # 12 Nov GATED ANSWERS (readout of a decision rule; no training change): the trunk
            gq = float(getattr(cfg, 'inf_distill_gate_q', 0.75))              #   answers only when its first-byte confidence >= t, else it abstains (defers to
            kr2 = [(ok_, c, ref(a)) for (ok_, c), a in zip(K_, Ka)]          #   the store). t is CALIBRATED at the previous sleep (the threshold that kept a
            un2 = [(st, c, ref(a)) for (st, c), a in zip(A_, Aa)] + [(x[0], x[2], ref(a)) for x, a in zip(B_, Ba)]   # share gq of the known facts it answered
            kc2 = np.array([c for ok_, c, rf in kr2 if ok_ and not rf])       #   right) and APPLIED to this sleep's new questions: out of sample
            tp = model.__dict__.get('_gate_t'); tn = float(np.quantile(kc2, 1 - gq)) if len(kc2) >= 4 else tp
            if tp is not None and kr2 and un2:
                ak = [(ok_, c) for ok_, c, rf in kr2 if not rf and c >= tp]; ru = [bool(st) for st, c, rf in un2 if not rf and c >= tp]
                print(f"  [DISTILLATION, GATED ANSWERS | visit {vis + 1}] rule: the trunk answers only when its confidence >= t = {tp:.2f} (calibrated at the previous sleep to keep "
                      f"{gq:.0%} of its right answers), else abstains; on THIS sleep's new questions: KNOWN facts ({len(kr2)}): answered {len(ak) / len(kr2):.2f}, right when answered "
                      f"{np.mean([ok_ for ok_, c in ak]) if ak else float('nan'):.2f}, right overall {sum(ok_ for ok_, c in ak) / len(kr2):.2f} (ungated {np.mean([ok_ for ok_, c, rf in kr2]):.2f}) | "
                      f"NEVER READ ({len(un2)}): given a stored name {sum(ru) / len(un2):.2f} (ungated {np.mean([bool(st) and not rf for st, c, rf in un2]):.2f}), abstained "
                      f"{1 - len(ru) / len(un2):.2f} | next t {tn if tn is not None else float('nan'):.2f}", flush=True)
            te = model.__dict__.get('_gate_te')                                 # 13 Nov SMOOTHED: the threshold averaged across sleeps (half last smoothed,
            if te is not None and kr2 and un2:                                  #   half this sleep's calibration) - the raw one swung 0.14 -> 0.53
                ak = [(ok_, c) for ok_, c, rf in kr2 if not rf and c >= te]; ru = [bool(st) for st, c, rf in un2 if not rf and c >= te]
                print(f"  [DISTILLATION, GATED ANSWERS SMOOTHED | visit {vis + 1}] t = {te:.2f} (averaged across sleeps): KNOWN answered {len(ak) / len(kr2):.2f}, right when answered "
                      f"{np.mean([ok_ for ok_, c in ak]) if ak else float('nan'):.2f}, right overall {sum(ok_ for ok_, c in ak) / len(kr2):.2f} | NEVER READ given a stored name "
                      f"{sum(ru) / len(un2):.2f}", flush=True)
            if tn is not None: model.__dict__['_gate_t'] = tn; model.__dict__['_gate_te'] = tn if te is None else 0.5 * te + 0.5 * tn
        except Exception as e9:
            _fail('inference.distill:gate'); print(f"  [DISTILLATION, GATED ANSWERS] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)


def _dt_items(ents, rs, n):
    out = []
    for i in rs.choice(len(ents), min(n, len(ents)), replace=False):
        e = ents[int(i)]; f = _DT_FORMS[rs.randint(len(_DT_FORMS))]
        out.append((f.format(A=e['A'], r=e['r']), e['ans'], e))
    return out


def _dt_recall(model, S_slot, ents):
    """open-ended recall from the trunk (memory and store off) in the test wordings: conclusions the standard held-out wordings, facts a
    new frame and a paraphrased relation"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    isl = lambda b: 65 <= b <= 90 or 97 <= b <= 122; r = {'conclusions': [], 'facts': []}
    for e in ents:
        if e['key'][0] == 'conclusion': qs = [w.format(A=e['A']) for w in _HO_TWO]
        else: qs = [f"It is on record that {e['A']} {e['r']} "] + ([f"{e['A']} {_DT_PARA[e['r']]} "] if e['r'] in _DT_PARA else [])
        for q in qs:
            pb = q.encode('latin-1', 'replace'); tb = (e['ans'] + '.').encode('latin-1', 'replace'); X, _ = _ws_batch(model, [pb + tb])
            with torch.no_grad(): lg = _ws_run(model, X, int(S_slot(e['d'])))[0]
            p0 = len(pb) - 1; ps = torch.arange(p0, p0 + len(tb), device=lg.device)
            if int(ps[-1]) >= lg.shape[0]: continue
            top = lg[ps].argmax(-1); tr = torch.tensor(list(tb), device=lg.device)
            r['conclusions' if e['key'][0] == 'conclusion' else 'facts'].append(bool((top[:-1] == tr[:-1]).all()) and not isl(int(top[-1])))
    return {k: (np.mean(v) if v else float('nan'), len(v)) for k, v in r.items()}


def distill_sleep(model, cfg, S_, S_slot, vals, t, ts, olds_):
    """one sleep's distillation (see the module note)"""
    if not int(getattr(cfg, 'inf_distill', 0)): return
    from platypus.inference.closed_form import _ws_batch, _ws_run, _ws_bpc
    from platypus.inference.namemem import _nm_material, _NM
    from platypus.inference.store import _INFM
    t0 = time.time(); vis = int(model.__dict__.get('_visit', 0))
    allm = _nm_material(model, cfg, S_)
    import zlib                                                                 # CONTROL: a fixed fifth of the facts (by a hash of the fact) is never taught,
    ctl = lambda e: zlib.crc32(repr(e['key']).encode()) % 5 == 0               #   so a rise in the taught facts' recall can be told from general change
    ents = [e for e in allm if not ctl(e)]; held = [e for e in allm if ctl(e)]
    if len(ents) < 16: print(f"  [DISTILLATION | visit {vis + 1}] skipped: {len(ents)} facts", flush=True); return
    assert all('tied to' not in f and 'on record' not in f and 'We read' not in f for f in _DT_FORMS)   # no taught wording is a test wording
    rs = np.random.RandomState(int(cfg.seed) + 9191 + vis)
    probe = [ents[int(i)] for i in rs.choice(len(ents), min(32, len(ents)), replace=False)]
    hprobe = [held[int(i)] for i in rs.choice(len(held), min(32, len(held)), replace=False)] if held else []
    steps = int(getattr(cfg, 'inf_distill_steps', 40)); nrow = int(getattr(cfg, 'inf_distill_rows', 16)); lam = float(getattr(cfg, 'inf_distill_anchor', 1.0))
    brake = float(getattr(cfg, 'inf_distill_brake', 0.01)); lr = float(cfg.lr) * float(getattr(cfg, 'inf_distill_lr', 0.01))
    params = [p for b in model.blocks for p in b.parameters() if p.requires_grad]
    snap = [p.detach().clone() for p in params]
    was_nm = _NM['force']; _NM['force'] = False; had = 'force' in _INFM; fz = _INFM.get('force'); _INFM['force'] = False
    try:
        r0 = _dt_recall(model, S_slot, probe); h0 = _dt_recall(model, S_slot, hprobe) if hprobe else None; b0, o0 = _ws_bpc(model, cfg, vals, t, ts, olds_)
        it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
        bysl = {}
        for e in ents: bysl.setdefault(int(S_slot(e['d'])), []).append(e)
        anc = {}                                                               # anchor rows per slot with the teacher's log-probabilities (before distilling)
        for sl in bysl:
            cand = [x for x in real if int(S_slot(x['d'])) == sl]
            if not cand: continue
            pick = [cand[int(i)] for i in rs.choice(len(cand), min(48, len(cand)), replace=False)]
            X, M = _ws_batch(model, [bytes(x['s']) for x in pick])
            with torch.no_grad(): lp = F.log_softmax(_ws_run(model, X, sl).float(), -1)
            anc[sl] = (X, M, lp)
        slots = list(bysl); tried = []; kept = False; ls = []
        for div in (1.0, 3.0, 9.0):                                            # DOSE RETRIES (as the closed-form edit): a dose that trips the brake is undone
            with torch.no_grad():                                              #   and a third of it tried, then a ninth; the first dose that passes is kept
                for p, s0 in zip(params, snap): p.copy_(s0)
            op = torch.optim.Adam(params, lr=lr / div); rsd = np.random.RandomState(int(cfg.seed) + 9192 + vis); ls = []
            for st in range(steps):
                sl = slots[rsd.randint(len(slots))]; items = _dt_items(bysl[sl], rsd, nrow)
                txt = [(p + a + '.').encode('latin-1', 'replace') for p, a, e in items]; X, M = _ws_batch(model, txt)
                lg = _ws_run(model, X, sl, grad_=True)
                rows, pos = [], []
                for r9, (p, a, e) in enumerate(items):
                    p0 = len(p.encode('latin-1', 'replace')) - 1
                    for j in range(len(a) + 1):
                        if p0 + j < lg.shape[1]: rows.append(r9); pos.append(p0 + j)
                rows = torch.tensor(rows, device=X.device); pos = torch.tensor(pos, device=X.device)
                loss = F.cross_entropy(lg[rows, pos], X[rows, pos + 1])
                if sl in anc and lam > 0:
                    Xa, Ma, lpa = anc[sl]; k = torch.tensor(rsd.choice(len(Xa), min(int(getattr(cfg, 'inf_distill_arows', 16)), len(Xa)), replace=False), device=Xa.device)
                    lga = F.log_softmax(_ws_run(model, Xa[k], sl, grad_=True).float(), -1); m = Ma[k][:, 1:lga.shape[1] + 1]
                    kl = ((lpa[k].exp() * (lpa[k] - lga)).sum(-1) * m).sum() / m.sum().clamp(min=1)
                    loss = loss + lam * kl
                op.zero_grad(); loss.backward(); op.step(); ls.append(float(loss.detach()))
                for p in model.parameters():
                    if p.grad is not None: p.grad = None
            b1, o1 = _ws_bpc(model, cfg, vals, t, ts, olds_)
            rt = _dt_recall(model, S_slot, probe); ht = _dt_recall(model, S_slot, hprobe) if hprobe else None   # what this dose taught (before any rollback)
            ok = (b1 - b0) <= brake and (not (o1 == o1) or (o1 - o0) <= brake)
            tried.append((div, b1, o1, ok, rt, ht, ls[0], ls[-1]))
            if ok: kept = True; break
        if not kept:
            with torch.no_grad():
                for p, s0 in zip(params, snap): p.copy_(s0)
        r1 = _dt_recall(model, S_slot, probe); h1 = _dt_recall(model, S_slot, hprobe) if hprobe else None
        f = lambda r: " | ".join(f"{k} {v[0]:.2f} ({v[1]})" for k, v in r.items()) if r else "none"
        tr = " ; ".join(f"lr x{float(getattr(cfg, 'inf_distill_lr', 0.01)) / d:.4g}: loss {l0:.3f} -> {l1:.3f}, BPC visited {b0:.4f} -> {b1:.4f}, old {o0:.4f} -> {o1:.4f}, "
                        f"{'KEPT' if ok else 'over the brake'}; taught facts then {f(rt)}, control then {f(ht)}" for d, b1, o1, ok, rt, ht, l0, l1 in tried)
        print(f"  [DISTILLATION | visit {vis + 1}] {steps} steps x {nrow} questions ({len(ents)} facts and conclusions, {len(_DT_FORMS)} self-written wordings, "
              f"trunk blocks only, anchors x{lam:g}, brake {brake:g}) || {tr} || {'KEPT' if kept else 'ALL DOSES ROLLED BACK'} | from the trunk, test wordings, "
              f"{len(probe)} taught facts: before {f(r0)} -> now {f(r1)} || CONTROL, {len(hprobe)} never-taught facts: before {f(h0)} -> now {f(h1)} | {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        _fail('inference.distill'); print(f"  [DISTILLATION] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
        with torch.no_grad():
            for p, s in zip(params, snap): p.copy_(s)
    finally:
        _NM['force'] = was_nm
        if had: _INFM['force'] = fz
        else: _INFM.pop('force', None)


# ---------------------------------------------------------------- INTERLEAVED (wake) distillation
# 11 Nov options that pair the question rows with protection (each off by default; see distill_wake_step and distill_wake_sleep):
#   inf_distill_wake_params='mlp_out'   only the blocks' mlp_out weights take the question gradient
#   inf_distill_wake_proj=1             anchor projection (A-GEM) against next-byte loss on stored real sentences
#   inf_distill_wake_ondream=1          question rows only on steps that also train on dream rows, weight x the dream interval (same dose)
#   inf_distill_wake_akl=lam            soft anchor: + lam x KL(the model at the last sleep || now) on stored real sentences of OLD slots
#   inf_distill_repair=N                repair replay at each sleep: N steps of that KL on the old slots (previous sleep's predictions) before
#                                       the new pool is built (lr = cfg.lr x inf_distill_repair_lr)
#   inf_distill_wake_spaced=1           spaced rehearsal: a fact the trunk answers in a taught wording at a sleep has its sampling interval
#                                       doubled (to 16), a fact it misses goes back to 1; rows drawn with weight 1/interval
#   inf_distill_wake_selfkl=1           a fact the trunk answers in a taught wording is held by KL to its own answer distribution at that
#                                       sleep instead of pushed by cross-entropy
_ISL = lambda b: 65 <= b <= 90 or 97 <= b <= 122


def _dw_known(model, S_slot, ents):
    """facts the trunk answers in a TAUGHT wording (the first self-written form, never a test wording): repr(key) -> bool"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    out = {}
    for e in ents:
        pb = _DT_FORMS[0].format(A=e['A'], r=e['r']).encode('latin-1', 'replace'); tb = (e['ans'] + '.').encode('latin-1', 'replace')
        X, _ = _ws_batch(model, [pb + tb])
        with torch.no_grad(): lg = _ws_run(model, X, int(S_slot(e['d'])))[0]
        p0 = len(pb) - 1
        if p0 + len(tb) - 1 >= lg.shape[0]: out[repr(e['key'])] = False; continue
        top = lg[p0:p0 + len(tb)].argmax(-1); tr = torch.tensor(list(tb), device=lg.device)
        out[repr(e['key'])] = bool((top[:-1] == tr[:-1]).all()) and not _ISL(int(top[-1]))
    return out


def _dw_cache(model, cfg, S_, S_slot, slots, vis):
    """32 stored real sentences per slot with the model's log-probabilities now (the anchor targets until the next sleep)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
    ra = np.random.RandomState(int(cfg.seed) + 9696 + vis); by = {}
    for i in ra.permutation(len(real)):
        x = real[int(i)]; sl = int(S_slot(x['d']))
        if sl in slots and len(by.setdefault(sl, [])) < 32: by[sl].append(bytes(x['s']))
    C = {}
    for sl, txt in by.items():
        if len(txt) < 4: continue
        X, M = _ws_batch(model, txt)
        with torch.no_grad(): lp = F.log_softmax(_ws_run(model, X, sl).float(), -1).half()
        C[sl] = (X, M, lp)
    return C


def _dw_kl(model, C, sl, rs, k, grad=True):
    from platypus.inference.closed_form import _ws_run
    X, M, lp = C[sl]
    idx = torch.tensor(rs.choice(len(X), min(k, len(X)), replace=False), device=X.device) if k else torch.arange(len(X), device=X.device)
    if grad: lg = F.log_softmax(_ws_run(model, X[idx], sl, grad_=True).float(), -1)
    else:
        with torch.no_grad(): lg = F.log_softmax(_ws_run(model, X[idx], sl).float(), -1)
    T = lg.shape[1]; m = M[idx][:, 1:T + 1].float(); t_ = lp[idx][:, :T].float()
    return ((t_.exp() * (t_ - lg)).sum(-1) * m).sum() / m.sum().clamp(min=1)


def _dw_params(model, cfg):
    P = [p for p in model.parameters() if p.requires_grad]
    only = str(getattr(cfg, 'inf_distill_wake_params', '') or '')
    if only == 'mlp_out':
        keep = set(id(t) for b in model.blocks for t in (b.mlp_out.weight, b.mlp_out.bias))
        return [p for p in P if id(p) in keep], [p for p in P if id(p) not in keep]
    if only: raise ValueError(f"inf_distill_wake_params {only!r}")
    return P, []


def _dw_repair(model, cfg, D, vslot, oslots, vis):
    """REPAIR REPLAY: N steps of KL to the previous sleep's predictions on stored real sentences of the old slots (not the one just visited)"""
    n = int(getattr(cfg, 'inf_distill_repair', 0)); C = (D or {}).get('cache') or {}
    sls = [s for s in oslots if s in C and s != vslot]
    if not n or not sls: return
    t0 = time.time(); P, frz = _dw_params(model, cfg); allp = list(model.parameters()); g_saved = [p.grad for p in allp]
    for p in allp: p.grad = None
    for p in frz: p.requires_grad_(False)
    rs = np.random.RandomState(int(cfg.seed) + 9797 + vis)
    try:
        b = {s: float(_dw_kl(model, C, s, rs, 0, grad=False)) for s in sls}
        op = torch.optim.Adam(P, lr=float(cfg.lr) * float(getattr(cfg, 'inf_distill_repair_lr', 0.1)))
        spd = int(getattr(cfg, 'inf_distill_spindle', 0)) and D is not None and D.get('bysl'); qls = []
        for st in range(n):
            sl = sls[rs.randint(len(sls))]; loss = _dw_kl(model, C, sl, rs, 8)
            if spd:
                ql = distill_wake_loss(model, cfg)
                if torch.is_tensor(ql): loss = loss + float(cfg.inf_distill_wake) * ql; qls.append(float(ql.detach()))
            op.zero_grad(); loss.backward(); op.step()
        for p in allp: p.grad = None
        a = {s: float(_dw_kl(model, C, s, rs, 0, grad=False)) for s in sls}
        print(f"  [DISTILLATION, REPAIR REPLAY | visit {vis + 1}] {n} steps on {len(sls)} old slots ({'mlp_out only' if frz else 'all parameters'}): KL to the previous "
              f"sleep's predictions on their stored real sentences {np.mean(list(b.values())):.4f} -> {np.mean(list(a.values())):.4f} | per slot "
              + " ".join(f"{s}:{b[s]:.4f}->{a[s]:.4f}" for s in sls)
              + (f" | SPINDLE: question rows inside the burst, answer loss {np.mean(qls[:10]):.3f} (first 10) -> {np.mean(qls[-10:]):.3f} (last 10)" if qls else "") + f" | {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        _fail('inference.distill:repair'); print(f"  [DISTILLATION, REPAIR REPLAY] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
    finally:
        for p in frz: p.requires_grad_(True)
        for p, g in zip(allp, g_saved): p.grad = g


def _dw_ever(model, cfg, S_slot, ents, held, D, vis):
    """EVER-TAUGHT LEDGER (readout, 11 Nov): every fact that has ever been in the question pool (and every control fact ever held back) is
    kept with the visit it entered; at each sleep, recall from the trunk in the test wordings over a fixed sample of the ledger (up to
    inf_distill_ever facts, chosen by a hash of the fact so the sample is stable), split by whether the fact is STILL in the pool or has LEFT it
    (crowded out by the memory cap, superseded or retracted) and by age (visits since it entered); with the ledger and pool sizes.
    Recall over the current pool alone can be flattered when the pool keeps the easiest facts."""
    import zlib
    nmax = int(getattr(cfg, 'inf_distill_ever', 256))
    if nmax <= 0: return
    L = model.__dict__.setdefault('_dw_ledger', {})
    now = {repr(e['key']) for e in ents}; nowc = {repr(e['key']) for e in held}
    for grp, es in (('t', ents), ('c', held)):
        for e in es: L.setdefault(repr(e['key']), dict(e=e, v=vis, g=grp))
    if not L: return
    pick = sorted(L, key=lambda k: zlib.crc32(k.encode()))
    tk = [k for k in pick if L[k]['g'] == 't'][:nmax]; ck = [k for k in pick if L[k]['g'] == 'c'][:max(32, nmax // 4)]
    tk1 = [k for k in tk if vis - L[k]['v'] >= 1]                              # entered at an earlier sleep, so trained for at least one visit
    f = lambda x: " | ".join(f"{k_} {v_[0]:.2f} ({v_[1]})" for k_, v_ in x.items() if v_[1]) or "none"
    def grp_(keys, cur):
        out = []
        for nm, sel in (('still in the pool', lambda k: k in cur), ('left the pool', lambda k: k not in cur)):
            ks = [k for k in keys if sel(k)]
            if ks: out.append(f"{nm} ({len(ks)}): {f(_dt_recall(model, S_slot, [L[k]['e'] for k in ks]))}")
        for nm, lo, hi in (('ENTERED AT THIS SLEEP (not yet taught)', 0, 0), ('taught 1 visit', 1, 1), ('2-3 visits', 2, 3), ('4+ visits', 4, 999)):
            ks = [k for k in keys if lo <= vis - L[k]['v'] <= hi]
            if ks: out.append(f"{nm} ({len(ks)}): {f(_dt_recall(model, S_slot, [L[k]['e'] for k in ks]))}")
        return " || ".join(out)
    try:
        nt = sum(1 for v in L.values() if v['g'] == 't'); nc = sum(1 for v in L.values() if v['g'] == 'c')
        print(f"  [DISTILLATION, EVER TAUGHT | visit {vis + 1}] ledger: {nt} facts and conclusions ever taught ({len(now)} in the pool now), {nc} control ever held back "
              f"({len(nowc)} now) | from the trunk, test wordings, sample of {len(tk)} ever in the pool: TAUGHT FOR AT LEAST ONE VISIT ({len(tk1)}) "
              f"{f(_dt_recall(model, S_slot, [L[k]['e'] for k in tk1])) if tk1 else 'none'} || "
              + grp_(tk, now) + f" ||| CONTROL, sample of {len(ck)} ever held back: ALL {f(_dt_recall(model, S_slot, [L[k]['e'] for k in ck]))}", flush=True)
    except Exception as e:
        _fail('inference.distill:ever'); print(f"  [DISTILLATION, EVER TAUGHT] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _dw_conc22(model, cfg, S_, S_slot, ents, held, vis):
    """CONCLUSIONS 2 x 2 (readout, 11 Nov): live conclusions TAUGHT by the question rows vs NEVER TAUGHT (the hashed control fifth), recall
    from the trunk in the held-out wordings with the closed-form edits IN PLACE and SUBTRACTED (the accumulated edit model._cf_E removed
    from blocks[inf_cf_layer].mlp_out for the measurement, as [FROM THE WEIGHTS, FIXED SET]); never-taught conclusions also split by
    whether BOTH their premise facts (A - bridge, bridge - C) are in the taught pool - composition from taught premises. Edits explain
    it if both rows fall to ~0 without them; replay if both survive equally; distillation adds only if the taught row survives better."""
    from platypus.inference.namemem import _nm_layer
    tc = [e for e in ents if e['key'][0] == 'conclusion']; cc = [e for e in held if e['key'][0] == 'conclusion']
    if not tc and not cc: return
    tf = {(e['A'], e['ans']) for e in ents if e['key'][0] == 'fact'}
    br = {}
    for c in S_.get('conc', []):
        if isinstance(c.get('A'), str) and isinstance(c.get('C'), str) and isinstance(c.get('B'), str): br[(c['A'], c['C'])] = c['B']
    def prem(e):
        B = br.get((e['A'], e['ans']))
        return B is not None and ((e['A'], B) in tf or (B, e['A']) in tf) and ((B, e['ans']) in tf or (e['ans'], B) in tf)
    cp = [e for e in cc if prem(e)]; cn = [e for e in cc if not prem(e)]
    E = model.__dict__.get('_cf_E'); mo = _nm_layer(model, cfg)
    g = lambda r: f"{r['conclusions'][0]:.2f} ({r['conclusions'][1]})" if r.get('conclusions', (0, 0))[1] else "-"
    def ask():
        return [g(_dt_recall(model, S_slot, x)) if x else "-" for x in (tc, cc, cp, cn)]
    try:
        a_ = ask(); b_ = None
        if E is not None and mo is not None and tuple(E.shape) == tuple(mo.weight.shape):
            W0 = mo.weight.detach().clone()
            try:
                with torch.no_grad(): mo.weight.copy_((W0.float() - E.float().to(W0.device)).to(W0.dtype))
                b_ = ask()
            finally:
                with torch.no_grad(): mo.weight.copy_(W0)
        nb = b_ if b_ is not None else ["no edits"] * 4
        print(f"  [DISTILLATION, CONCLUSIONS 2x2 | visit {vis + 1}] from the trunk, held-out wordings (n = questions) | edits IN PLACE -> SUBTRACTED: "
              f"TAUGHT {a_[0]} -> {nb[0]} | NEVER TAUGHT {a_[1]} -> {nb[1]} || never taught, BOTH PREMISES TAUGHT {a_[2]} -> {nb[2]} | "
              f"premises not both taught {a_[3]} -> {nb[3]} | conclusions: {len(tc)} taught, {len(cc)} never taught ({len(cp)} with both premises taught)", flush=True)
    except Exception as e:
        _fail('inference.distill:conc22'); print(f"  [DISTILLATION, CONCLUSIONS 2x2] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _dw_pd_probe(model, cfg, S_slot, ents, held, ctx_, vis):
    """SOURCE-IN-CONTEXT PROBE (readout, 11 Nov; inf_distill_pd_probe): does the trunk answer better when the fact's stored source
    sentence comes before the question? (What prompt distillation needs: a teacher better than the student.) Facts only, up to 64 taught
    and 64 never-taught (the control fifth - never trained, so with vs without context is pure reading from context), in a TAUGHT wording
    and a TEST wording (new frame): open-ended greedy answer exact, without -> with the sentence; answer bytes top-1; mean KL(with || without)
    per answer byte (the signal soft targets would carry)."""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    from platypus.inference.jepa import cfg_seq_
    L0 = int(cfg_seq_(model)); rs = np.random.RandomState(int(cfg.seed) + 7777 + vis)
    def pick(es):
        es = [e for e in es if e['key'][0] == 'fact' and repr(e['key']) in ctx_]
        return [es[int(i)] for i in rs.choice(len(es), min(64, len(es)), replace=False)] if es else []
    def one(es, wf):
        r = dict(n=0, ex0=0, ex1=0, b=0, b0=0, b1=0, kl=0.0)
        for e in es:
            q = wf(e).encode('latin-1', 'replace'); a = (e['ans'] + '.').encode('latin-1', 'replace'); cx = ctx_[repr(e['key'])]
            room = L0 - len(q) - len(a) - 1
            if room <= 8: continue
            cx = cx[-room:]
            X, _ = _ws_batch(model, [q + a, cx + q + a])
            with torch.no_grad(): lg = _ws_run(model, X, int(S_slot(e['d']))).float()
            p0 = len(q) - 1; p1 = len(cx) + p0; n_ = len(a)
            if p1 + n_ - 1 >= lg.shape[1]: continue
            z0 = lg[0, p0:p0 + n_]; z1 = lg[1, p1:p1 + n_]; tr = torch.tensor(list(a), device=lg.device)
            t0 = z0.argmax(-1); t1 = z1.argmax(-1); isl = lambda b: 65 <= b <= 90 or 97 <= b <= 122
            r['n'] += 1; r['b'] += n_ - 1
            r['ex0'] += int(bool((t0[:-1] == tr[:-1]).all()) and not isl(int(t0[-1]))); r['ex1'] += int(bool((t1[:-1] == tr[:-1]).all()) and not isl(int(t1[-1])))
            r['b0'] += int((t0[:-1] == tr[:-1]).sum()); r['b1'] += int((t1[:-1] == tr[:-1]).sum())
            l1 = F.log_softmax(z1, -1); r['kl'] += float((l1.exp() * (l1 - F.log_softmax(z0, -1))).sum(-1)[:-1].sum())
        return r
    fmt = lambda r: (f"exact {r['ex0'] / r['n']:.2f} -> {r['ex1'] / r['n']:.2f}, answer bytes {r['b0'] / max(1, r['b']):.2f} -> {r['b1'] / max(1, r['b']):.2f}, KL {r['kl'] / max(1, r['b']):.3f} (n {r['n']})" if r['n'] else "none")
    try:
        tf = lambda e: _DT_FORMS[0].format(A=e['A'], r=e['r']); nf = lambda e: f"It is on record that {e['A']} {e['r']} "
        T, C = pick(ents), pick(held)
        print(f"  [SOURCE IN CONTEXT | visit {vis + 1}] the trunk answers WITHOUT -> WITH the fact's stored source sentence before the question (window {L0}) || "
              f"TAUGHT facts, taught wording: {fmt(one(T, tf))} | test wording: {fmt(one(T, nf))} || NEVER-TAUGHT facts (reading from context alone), "
              f"taught wording: {fmt(one(C, tf))} | test wording: {fmt(one(C, nf))}", flush=True)
    except Exception as e:
        _fail('inference.distill:pdprobe'); print(f"  [SOURCE IN CONTEXT] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


_BRG = {'held': set(), 'anch': set()}


def _dw_bridge(cfg, S_, ents, held, vis):
    """BRIDGE ANCHORING (11 Nov; inf_distill_bridge = N descriptors; after BRACE, NeurIPS 2026): two-hop recall from injected facts fails when
    the bridge entity B is represented differently as the ANSWER of hop 1 (A - B) and as the SUBJECT of hop 2 (B - C); reusing the same
    descriptors of B in both hops' training text aligned it (composition up to 0.78 at 1.7-14B). For each live conclusion's bridge B: up to N
    TAUGHT facts about B whose other name is not a chain partner of B (never A or C of any chain through B, so no row holds both hops of a
    chain), or else stored sentence clips naming B and no partner, fixed per B by a hash; bridges are those of live conclusions AND of
    every chain of two taught facts (the chains the COMPOSITION PROBE asks); every question row with B as answer gets one after the answer (trained bytes), every row with B as
    subject gets one before the question. Never a control fact. The test: [DISTILLATION, CONCLUSIONS 2x2] never-taught conclusions with
    both premises taught, edits subtracted (inf_distill_conc_ctl 1 holds every conclusion back)."""
    import zlib
    n = int(getattr(cfg, 'inf_distill_bridge', 0)); part = {}
    if int(getattr(cfg, 'inf_distill_bridge_syn', 0)):                        # 11 Nov SYNTHETIC, HALF: the bridges of the HELD-OUT chains (inf_kgx_hold: never
        from platypus.inference.probes import inf_probes                      #   concluded, stored, replayed or edited); half of them, by a hash of the name,
        N9 = int(getattr(cfg, 'inf_kgx_hold', 0)); held_b = set()            #   get one fixed INVENTED attribute as descriptor ('B keeps the green ledger.'),
        if N9 > 0:                                                            #   the other half none: anchored vs unanchored held-out chains in one run
            for i9, c9 in enumerate(inf_probes(cfg)):
                if i9 % N9 == N9 - 1:
                    nm9 = list(c9['names'])
                    for k9 in range(1, len(nm9) - 1): held_b.add(nm9[k9])
        adj = ('green', 'silver', 'oak', 'copper', 'amber', 'blue', 'iron', 'linen', 'stone', 'cedar')
        nou = ('ledger', 'lantern', 'compass', 'satchel', 'kettle', 'harp', 'anchor', 'quill', 'clock', 'ribbon')
        anch = {B for B in held_b if zlib.crc32(B.encode('latin-1', 'replace')) % 2 == 0}
        out = {}
        for B in anch:
            h9 = zlib.crc32(('d|' + B).encode('latin-1', 'replace'))
            out[B] = [f"{B} keeps the {adj[h9 % 10]} {nou[(h9 // 10) % 10]}.".encode('latin-1', 'replace')]
        _BRG['held'] = held_b; _BRG['anch'] = anch
        fs = [e for e in ents if e['key'][0] == 'fact']
        h1 = sum(1 for e in fs if e['ans'] in out); h2 = sum(1 for e in fs if e['A'] in out)
        print(f"  [DISTILLATION, BRIDGE | visit {vis + 1}] SYNTHETIC, HALF: {len(held_b)} bridges of held-out chains (every {N9}th chain never concluded), "
              f"{len(anch)} anchored with an invented attribute, {len(held_b) - len(anch)} not | taught facts anchored: {h1} with the bridge as answer, {h2} as subject"
              + (f" | e.g. {sorted(out)[0]}: {out[sorted(out)[0]][0].decode('latin-1')}" if out else ""), flush=True)
        return out
    for c in S_.get('conc', []):
        if c.get('freed') or c.get('retracted'): continue
        if isinstance(c.get('A'), str) and isinstance(c.get('B'), str) and isinstance(c.get('C'), str): part.setdefault(c['B'], set()).update((c['A'], c['C']))
    fs = [e for e in ents if e['key'][0] == 'fact']; out = {}; nsent = 0; nlive = len(part)
    sub = {}
    for e in fs: sub.setdefault(e['A'], set()).add(e['ans'])
    for e in fs:                                                               # ALSO every bridge of a chain of TAUGHT facts (A - B, B - C):
        if e['ans'] in sub:                                                    #   partners are every name that reaches B or that B reaches
            for C in sub[e['ans']]:
                if C != e['A']: part.setdefault(e['ans'], set()).update((e['A'], C))
    real = None; byname = None
    for B, pt in part.items():
        cand = sorted({f"{e['A']} {e['r']} {e['ans']}." for e in fs if (e['A'] == B and e['ans'] not in pt) or (e['ans'] == B and e['A'] not in pt)})
        if not cand:                                                           # FALLBACK: a stored real sentence naming B and no chain partner of B,
            if byname is None:                                                 #   clipped to 48 bytes around B (the local harness's bridges rarely
                import re as _re                                               #   have a third taught fact); one pass over the stored sentences,
                byname = {}; want = set(part)                                  #   indexed by the capitalised words they contain
                for x in S_.get('it', []):
                    if x.get('k') != 'real': continue
                    b9 = x['s']; b9 = bytes(int(v) & 255 for v in b9.tolist()) if hasattr(b9, 'tolist') and not isinstance(b9, (bytes, bytearray)) else (b9.encode('latin-1', 'replace') if isinstance(b9, str) else bytes(b9))
                    t = b9.decode('latin-1')
                    for w in set(_re.findall(r"[A-Z][a-z]+", t)) & want:
                        if len(byname.setdefault(w, [])) < 64: byname[w].append(t)
            bb = []
            for t in byname.get(B, []):
                i9 = t.find(B)
                if i9 < 0 or any(q in t for q in pt): continue
                a9 = max(0, i9 - 12); bb.append(t[a9:a9 + 48].strip())
            cand = sorted(set(bb)); nsent += 1 if cand else 0
        if not cand: continue
        cand.sort(key=lambda t: zlib.crc32((B + '|' + t).encode('latin-1', 'replace')))
        out[B] = [t.encode('latin-1', 'replace') for t in cand[:n]]
    h1 = sum(1 for e in fs if e['ans'] in out); h2 = sum(1 for e in fs if e['A'] in out)
    print(f"  [DISTILLATION, BRIDGE | visit {vis + 1}] {len(part)} bridge names ({nlive} in live conclusions, the rest in chains of taught facts), {len(out)} with a descriptor ({len(out) - nsent} a taught fact about the bridge, "
          f"{nsent} a stored sentence clip; no chain partner; up to {n} each, mean {np.mean([len(v) for v in out.values()]) if out else 0:.2f}) | taught facts anchored: {h1} with the bridge as answer, "
          f"{h2} as subject" + (f" | e.g. {list(out)[0]}: {out[list(out)[0]][0].decode('latin-1')}" if out else ""), flush=True)
    return out


def _dw_ns_probe(model, cfg, S_, S_slot, ents, vslot, oslots, vis):
    """NULL-SPACE PROBE (readout, 11 Nov; inf_distill_ns_probe): could the question rows be confined to directions the old domains do not
    use (AlphaEdit-style null-space projection)? For each block's mlp_out (the weights the question rows write): the covariance of its
    INPUTS on 32 stored real sentences per OLD slot (not the one just visited); the number of input directions holding 90 / 99 / 99.9% of
    that energy (of H); and the share of a question-row gradient (up to 64 taught facts, taught wordings) that lies INSIDE those
    directions - what a hard projection at that cut-off would remove (an isotropic update would lose k / H). Training is not changed."""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    sls = [int(s) for s in oslots if int(s) != int(vslot)] if vslot is not None else []
    fs = [e for e in ents if e['key'][0] == 'fact']
    if not sls or not fs: return
    it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
    ra = np.random.RandomState(int(cfg.seed) + 9898 + vis); by = {}
    for i in ra.permutation(len(real)):
        x = real[int(i)]; sl = int(S_slot(x['d']))
        if sl in sls and len(by.setdefault(sl, [])) < 32: by[sl].append(bytes(x['s']))
    blocks = list(model.blocks); C = [None] * len(blocks); hs = []; cur = {}
    def mk(j):
        def h(mod, inp, out):
            x = inp[0].detach().float(); m = cur.get('m')
            x = x.reshape(-1, x.shape[-1]) if m is None else x[m[:, :x.shape[1]] > 0]
            c = x.t() @ x; C[j] = c if C[j] is None else C[j] + c
        return h
    allp = list(model.parameters()); g_saved = [p.grad for p in allp]; rq = [p.requires_grad for p in allp]
    try:
        for j, b in enumerate(blocks): hs.append(b.mlp_out.register_forward_hook(mk(j)))
        npos = 0
        for sl, txt in by.items():
            if len(txt) < 4: continue
            X, M = _ws_batch(model, txt); cur['m'] = M[:, :-1]; npos += int(M[:, :-1].sum())
            with torch.no_grad(): _ws_run(model, X, sl)
        for h in hs: h.remove()
        hs = []; cur.clear()
        if npos == 0: return
        rs = np.random.RandomState(int(cfg.seed) + 9899 + vis); pick = [fs[int(i)] for i in rs.choice(len(fs), min(64, len(fs)), replace=False)]
        for p in allp: p.grad = None
        W = [b.mlp_out.weight for b in blocks]
        for w in W: w.requires_grad_(True)
        bys = {}
        for e in pick: bys.setdefault(int(S_slot(e['d'])), []).append(e)
        for sl, es in bys.items():
            rows = [(_DT_FORMS[int(rs.randint(len(_DT_FORMS)))].format(A=e['A'], r=e['r']).encode('latin-1', 'replace'), (e['ans'] + '.').encode('latin-1', 'replace')) for e in es]
            X, _ = _ws_batch(model, [a + b for a, b in rows]); lg = _ws_run(model, X, sl, grad_=True); loss = 0.0
            for r9, (a, b) in enumerate(rows):
                p0 = len(a) - 1; n_ = min(len(b), lg.shape[1] - p0)
                if n_ > 0: loss = loss + F.cross_entropy(lg[r9, p0:p0 + n_].float(), X[r9, p0 + 1:p0 + 1 + n_], reduction='sum')
            if torch.is_tensor(loss): loss.backward()
        res = []
        for j, w in enumerate(W):
            if w.grad is None or C[j] is None: continue
            G = w.grad.detach().float(); ev, V = torch.linalg.eigh(C[j].double()); ev = ev.flip(0).clamp_min(0); V = V.flip(1).float()
            cs = torch.cumsum(ev, 0) / ev.sum().clamp_min(1e-12); H = int(ev.numel()); gn = float((G * G).sum()) + 1e-30; r = []
            for q in (0.9, 0.99, 0.999):
                k = int((cs < q).sum()) + 1; GU = G @ V[:, :k]; r.append((k, float((GU * GU).sum()) / gn))
            res.append((j, H, r))
        if not res: return
        fmt = lambda r: " / ".join(f"{k}" for k, _ in r); fs_ = lambda r: " / ".join(f"{v:.2f}" for _, v in r)
        mean = [np.mean([x[2][i][1] for x in res]) for i in range(3)]; iso = [np.mean([x[2][i][0] / x[1] for x in res]) for i in range(3)]
        print(f"  [NULL SPACE PROBE | visit {vis + 1}] mlp_out inputs on old slots {sls} ({npos} byte positions) || per block: input directions holding 90 / 99 / 99.9% "
              f"of the old-domain energy (of H) -> share of a question-row gradient ({len(pick)} taught facts) INSIDE them: "
              + " | ".join(f"block {j}: {fmt(r)} of {H} -> {fs_(r)}" for j, H, r in res)
              + f" || MEAN share inside {mean[0]:.2f} / {mean[1]:.2f} / {mean[2]:.2f} (an isotropic update: {iso[0]:.2f} / {iso[1]:.2f} / {iso[2]:.2f}); "
              f"a hard projection at that cut-off keeps {1 - mean[0]:.2f} / {1 - mean[1]:.2f} / {1 - mean[2]:.2f} of the fact update", flush=True)
    except Exception as e:
        _fail('inference.distill:nsprobe'); print(f"  [NULL SPACE PROBE] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
    finally:
        for h in hs: h.remove()
        for p, g, q in zip(allp, g_saved, rq): p.grad = g; p.requires_grad_(q)


def _fu_init(model, cfg):
    """FACT UNITS (11 Nov; inf_distill_fu = units added per block at each sleep; neurogenesis): a reserve of inf_distill_fu_max units beside each
    block's MLP, none active at first. Unit j: u_j = relu(g_j (cos(hn, K_j) - tau)), output V_j (zero at birth: no disruption). K_j is fixed
    at birth to the mean normalised MLP input at the answer positions of one fact's question (three taught wordings) - the unit fires on
    that fact's context and, by the cosine threshold, little elsewhere; V and g are trained by the question rows ONLY (the trunk's own
    weights take no question gradient in this mode). Kept outside model.parameters(): the main optimiser never sees them."""
    from platypus.layers import _FU
    if model.__dict__.get('_fu') is not None: return model.__dict__['_fu']
    dev = model.embed.weight.device; fm = int(getattr(cfg, 'inf_distill_fu_max', 1024)); g0 = float(getattr(cfg, 'inf_distill_fu_g', 4.0))
    P = {}
    for b in model.blocks:
        d = int(b.d_model)
        P[id(b)] = dict(K=torch.zeros(fm, d, device=dev), g=torch.full((fm,), g0, device=dev), V=torch.zeros(fm, d, device=dev), n=0,
                        mu=torch.zeros(d, device=dev), t=torch.full((fm,), float(getattr(cfg, 'inf_distill_fu_tau', 0.5)), device=dev), mu_set=False)
    _FU['P'] = P; _FU['on'] = True; _FU['tau'] = float(getattr(cfg, 'inf_distill_fu_tau', 0.5))
    ts = [t for v in P.values() for t in (v['V'], v['g'])]
    st = dict(P=P, map={}, opt=torch.optim.Adam(ts, lr=float(cfg.lr) * float(getattr(cfg, 'inf_distill_fu_lr', 3.0))), steps=0, ql=[])
    model.__dict__['_fu'] = st
    return st


def _fu_old_sents(cfg, S_, S_slot, sls, seed, n=32):
    it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
    ra = np.random.RandomState(seed); by = {}
    for i in ra.permutation(len(real)):
        x = real[int(i)]; sl = int(S_slot(x['d']))
        if sl in sls and len(by.setdefault(sl, [])) < n: by[sl].append(bytes(x['s']))
    return by


def _fu_sleep(model, cfg, S_, S_slot, ents, held, vslot, oslots, vis):
    """FACT UNITS at a sleep: report last visit (question steps, answer loss, firing on OLD-DOMAIN text vs on taught questions and the
    never-taught control's questions), then give each new taught fact (no unit yet; up to inf_distill_fu per block) its own unit"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    from platypus.layers import _FU
    st = _fu_init(model, cfg); P = st['P']; blocks = list(model.blocks); fm = int(getattr(cfg, 'inf_distill_fu_max', 1024))
    def fire(groups):
        _FU['st'] = {}
        try:
            for sl, txt in groups:
                if not txt: continue
                X, _ = _ws_batch(model, txt)
                with torch.no_grad(): _ws_run(model, X, sl)
            r = [v for v in _FU['st'].values()]
        finally: _FU['st'] = None
        if not r: return None
        a = sum(x[0] for x in r) / max(1.0, sum(x[2] for x in r)); m = sum(x[1] for x in r) / max(1.0, sum(x[2] for x in r))
        return a, m
    try:
        n0 = P[id(blocks[0])]['n']
        if n0 > 0:
            sls = [int(s) for s in oslots if vslot is None or int(s) != int(vslot)]
            old = fire(list(_fu_old_sents(cfg, S_, S_slot, sls, int(cfg.seed) + 7171 + vis).items())) if sls else None
            vis_ = fire(list(_fu_old_sents(cfg, S_, S_slot, [int(vslot)], int(cfg.seed) + 7172 + vis).items())) if vslot is not None else None
            tq = [e for e in ents if repr(e['key']) in st['map']][:64]; nf = lambda e: f"It is on record that {e['A']} {e['r']} "
            byq = lambda es, wf: [(int(S_slot(e['d'])), [(wf(e) + e['ans'] + '.').encode('latin-1', 'replace')]) for e in es]
            q1 = fire(byq(tq, lambda e: _DT_FORMS[0].format(A=e['A'], r=e['r']))); q2 = fire(byq(tq, nf))
            hq = fire(byq([e for e in held if e['key'][0] == 'fact'][:64], nf))
            fmt = lambda r: f"{r[0]:.3f} of positions (mean drive {r[1]:.4f})" if r else "-"
            ql = st['ql']
            print(f"  [FACT UNITS | visit {vis + 1}] {n0} units per block (cap {fm}), {len(st['map'])} facts with a unit | last visit: {st['steps']} question steps "
                  f"(fact units only; the trunk takes no question gradient), answer loss {np.mean(ql[:20]) if ql else float('nan'):.3f} (first 20) -> {np.mean(ql[-20:]) if ql else float('nan'):.3f} (last 20) || "
                  f"ANY UNIT FIRING: old-domain text {fmt(old)} | the visited domain's text {fmt(vis_)} | taught facts' questions, taught wording {fmt(q1)}, TEST wording {fmt(q2)} | "
                  f"never-taught facts' questions {fmt(hq)} || |V| mean {float(torch.stack([P[id(b)]['V'][:n0].norm(dim=-1).mean() for b in blocks]).mean()):.3f}", flush=True)
        st['steps'] = 0; st['ql'] = []
        new = [e for e in ents if e['key'][0] == 'fact' and repr(e['key']) not in st['map']]
        k = min(int(getattr(cfg, 'inf_distill_fu', 0)), fm - P[id(blocks[0])]['n'], len(new))
        if k <= 0: return
        import zlib
        new.sort(key=lambda e: zlib.crc32(repr(e['key']).encode()))
        new = new[:k]; acc = {id(b): [] for b in blocks}
        real = _fu_old_sents(cfg, S_, S_slot, set(int(S_slot(x['d'])) for x in S_.get('it', [])[:20000] if x.get('k') == 'real'), int(cfg.seed) + 7373 + vis, n=16)
        rcap = {id(b): [] for b in blocks}                                     # MLP inputs on ordinary stored text (centring and thresholds)
        for sl, txt in real.items():
            if not txt: continue
            X, M = _ws_batch(model, txt); _FU['cap'] = {}
            try:
                with torch.no_grad(): _ws_run(model, X, sl)
                cap = dict(_FU['cap'])
            finally: _FU['cap'] = None
            m9 = M[:, :-1] > 0
            for b in blocks:
                h = cap.get(id(b))
                if h is not None: rcap[id(b)].append(h[m9[:, :h.shape[1]]].float())
        for b in blocks:                                                       #   CENTRE once, at the first births: the MLP inputs share a large common
            Pb = P[id(b)]                                                      #   direction, so raw cosines fire everywhere (first local run: 0.36-0.48 of text)
            if not Pb['mu_set'] and rcap[id(b)]:
                Pb['mu'] = torch.cat(rcap[id(b)]).mean(0); Pb['mu_set'] = True
        for e in new:
            txt = [(fm_.format(A=e['A'], r=e['r']) + e['ans'] + '.').encode('latin-1', 'replace') for fm_ in _DT_FORMS[:3]]
            pl = [len(fm_.format(A=e['A'], r=e['r']).encode('latin-1', 'replace')) for fm_ in _DT_FORMS[:3]]
            X, _ = _ws_batch(model, txt); _FU['cap'] = {}
            try:
                with torch.no_grad(): _ws_run(model, X, int(S_slot(e['d'])))
                cap = dict(_FU['cap'])
            finally: _FU['cap'] = None
            na = len(e['ans'])
            for b in blocks:
                h = cap.get(id(b))
                if h is None: continue
                rows = []
                for r9, p0 in enumerate(pl):
                    seg = h[r9, p0 - 1:min(p0 - 1 + na, h.shape[1])].float() - P[id(b)]['mu']
                    if seg.shape[0]: rows.append(F.normalize(seg, dim=-1).mean(0))
                acc[id(b)].append(F.normalize(torch.stack(rows).mean(0), dim=-1) if rows else None)
        q9 = float(getattr(cfg, 'inf_distill_fu_q', 0.99)); tmin = float(getattr(cfg, 'inf_distill_fu_tau', 0.5)); tl9 = []
        for b in blocks:                                                       # PER-UNIT THRESHOLD: the q-quantile of the unit's cosine on ordinary stored
            Pb = P[id(b)]; j0 = Pb['n']                                        #   text (at least tau), so at birth it fires on at most 1 - q of text positions
            R9 = F.normalize(torch.cat(rcap[id(b)]) - Pb['mu'], dim=-1) if rcap[id(b)] else None
            for i, kv in enumerate(acc[id(b)]):
                if kv is None: continue
                Pb['K'][j0 + i] = kv; t9 = tmin
                if R9 is not None and R9.shape[0] > 50: t9 = max(tmin, float(torch.quantile((R9 @ kv).float(), q9)))
                Pb['t'][j0 + i] = t9; tl9.append(t9)
            Pb['n'] = j0 + len(new)
        for i, e in enumerate(new): st['map'][repr(e['key'])] = P[id(blocks[0])]['n'] - len(new) + i
        print(f"  [FACT UNITS | visit {vis + 1}] {len(new)} new units per block for {len(new)} new taught facts (keys from three taught wordings' answer positions); "
              f"{P[id(blocks[0])]['n']} active of {fm} | centred inputs, per-unit thresholds at the {q9:.2f} quantile of cosine on ordinary text: mean {np.mean(tl9) if tl9 else float('nan'):.3f}", flush=True)
    except Exception as e:
        _fail('inference.distill:fu'); print(f"  [FACT UNITS] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _dw_comp_probe(model, cfg, S_, S_slot, ents, held, D, vis):
    """COMPOSITION PROBE (readout, 11 Nov; inf_distill_comp_probe = N): two-fact chains built from the TAUGHT facts themselves - A r1 B and
    B r2 C, both in last visit's question pool (taught for at least one visit), A != C, no live conclusion for A-C or C-A (so never
    edited, replayed or asked) and no stored fact linking A and C directly - asked 'A is connected to' in the held-out conclusion
    wordings. Up to N chains; with the trunk's recall of the two premises in the test wordings, and composition among chains whose
    premises it answers both (the only chains it could compose). The same for chains of never-taught (control) facts."""
    import zlib
    n = int(getattr(cfg, 'inf_distill_comp_probe', 0))
    if not n or not D or not D.get('bysl'): return
    if _BRG['held']: n = max(n, 256)                                           # held-out chains: ask them all
    try:
        prev = {rw[3] for pool in D['bysl'].values() for rw in pool}
        live = set()
        for c in S_.get('conc', []):
            if not c.get('freed') and not c.get('retracted') and isinstance(c.get('A'), str) and isinstance(c.get('C'), str): live |= {(c['A'], c['C']), (c['C'], c['A'])}
        allf = [e for e in ents + held if e['key'][0] == 'fact']; direct = {(e['A'], e['ans']) for e in allf} | {(e['ans'], e['A']) for e in allf}
        def chains(fs):
            byA = {}
            for e in fs: byA.setdefault(e['A'], []).append(e)
            out = []
            for e1 in fs:
                for e2 in byA.get(e1['ans'], []):
                    A, C = e1['A'], e2['ans']
                    if A == C or (A, C) in live or (A, C) in direct: continue
                    out.append((e1, e2))
            out.sort(key=lambda p: zlib.crc32((p[0]['A'] + '|' + p[0]['ans'] + '|' + p[1]['ans']).encode('latin-1', 'replace')))
            return out[:n]
        def run(chs):
            if not chs: return None
            q = [dict(key=('conclusion', e1['A'], 'is connected to', e2['ans']), A=e1['A'], r='is connected to', ans=e2['ans'], d=e1['d']) for e1, e2 in chs]
            comp = [_dt_recall(model, S_slot, [x])['conclusions'] for x in q]
            prem = [(_dt_recall(model, S_slot, [e1])['facts'], _dt_recall(model, S_slot, [e2])['facts']) for e1, e2 in chs]
            ok = [i for i, (a, b) in enumerate(prem) if a[1] and b[1] and a[0] >= 0.5 and b[0] >= 0.5]
            c_all = float(np.nanmean([c[0] for c in comp])) if comp else float('nan')
            c_ok = float(np.nanmean([comp[i][0] for i in ok])) if ok else float('nan')
            return len(chs), len(ok), c_all, c_ok
        T = chains([e for e in ents if e['key'][0] == 'fact' and repr(e['key']) in prev]); H = chains([e for e in held if e['key'][0] == 'fact'])
        rt, rh = run(T), run(H)
        f = lambda r: (f"{r[0]} chains, both premises answered in a test wording {r[1]} | composed (held-out wordings) ALL {r[2]:.2f}, among those with both premises answered "
                       + (f"{r[3]:.2f}" if r[1] else "-")) if r else "none"
        xtra = ""
        if _BRG['held']:
            ta = [p for p in T if p[0]['ans'] in _BRG['anch']]; tu = [p for p in T if p[0]['ans'] in _BRG['held'] and p[0]['ans'] not in _BRG['anch']]
            xtra = f" || HELD-OUT CHAINS, bridge ANCHORED (invented attribute): {f(run(ta))} | bridge NOT anchored: {f(run(tu))}"
        print(f"  [COMPOSITION PROBE | visit {vis + 1}] 'A is connected to C' for chains A-B, B-C of facts, no conclusion or direct fact for A-C (never edited, replayed or asked) || "
              f"TAUGHT premises: {f(rt)} || NEVER-TAUGHT premises: {f(rh)}" + xtra, flush=True)
    except Exception as e:
        _fail('inference.distill:comp'); print(f"  [COMPOSITION PROBE] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _dw_latent_chain(model, cfg, S_, S_slot, ents, held, D, vis):
    """LATENT CHAINING (readout, 11 Nov; inf_distill_lchain = training steps): can a small JEPA-style predictor over the trunk's own name codes do
    rule-free inference? Entity code: the last block's output mean-pooled over the name in 'About NAME.' (frozen, centred, normalised).
    P(e_A, r) -> e_B: an MLP with a learned relation embedding, trained by InfoNCE against every entity code on the TAUGHT facts (one-hop only,
    never a chain). Q(e_A, r1, r2) -> e_C: the same, trained on the chains the SYSTEM ITSELF concluded (live conclusions with both premise facts
    found) - composition learned from examples, no rule. Tested on chains of taught facts with NO conclusion (never concluded, stored,
    replayed or edited; with inf_kgx_hold the held-out chains): the true C among 20 candidate names (chance 0.05), by P twice (chaining
    through the predicted bridge) and by Q in one step; and P on held-back (never-taught) facts as a baseline. Training is not changed."""
    import zlib
    from platypus.inference.closed_form import _ws_batch, _ws_run
    n_st = int(getattr(cfg, 'inf_distill_lchain', 0))
    if not n_st: return
    try:
        fs = [e for e in ents if e['key'][0] == 'fact']; hf = [e for e in held if e['key'][0] == 'fact']
        if len(fs) < 16: return
        names = sorted({e['A'] for e in fs + hf} | {e['ans'] for e in fs + hf}); ix = {n: i for i, n in enumerate(names)}
        rels = sorted({e['r'] for e in fs + hf}); rx = {r: i for i, r in enumerate(rels)}
        dev = model.embed.weight.device; cap = {}
        h9 = model.blocks[-1].register_forward_hook(lambda m_, i_, o_: cap.__setitem__('h', o_.detach()))
        E = []
        try:
            for i0 in range(0, len(names), 64):
                chunk = names[i0:i0 + 64]; txt = [f"About {n}.".encode('latin-1', 'replace') for n in chunk]
                X, _ = _ws_batch(model, txt)
                with torch.no_grad(): _ws_run(model, X, 0)
                h = cap['h'].float()
                for j, n in enumerate(chunk):
                    E.append(h[j, 5:5 + len(n.encode('latin-1', 'replace'))].mean(0))
        finally: h9.remove()
        E = torch.stack(E); E = F.normalize(E - E.mean(0), dim=-1); d = E.shape[1]
        g = torch.Generator(device='cpu').manual_seed(int(cfg.seed) + 4242 + vis)
        def mlp(k_in):
            return torch.nn.Sequential(torch.nn.Linear(k_in, 512), torch.nn.GELU(), torch.nn.Linear(512, d)).to(dev)
        def train(net, emb, X_idx, R_idx, Y_idx):
            opt = torch.optim.Adam(list(net.parameters()) + list(emb.parameters()), lr=1e-3)
            n = len(Y_idx); Xi = torch.tensor(X_idx, device=dev); Ri = torch.tensor(R_idx, device=dev); Yi = torch.tensor(Y_idx, device=dev)
            for st in range(n_st):
                b = torch.randint(0, n, (min(256, n),), generator=g).to(dev)
                z = F.normalize(net(torch.cat([E[Xi[b]], emb(Ri[b]).flatten(1)], -1)), dim=-1)
                loss = F.cross_entropy(z @ E.t() / 0.05, Yi[b]); opt.zero_grad(); loss.backward(); opt.step()
            return float(loss)
        def rank(z, true, rs):                                                 # true among 20 candidates (19 random other names)
            cand = [true] + [int(c) for c in rs.choice([i for i in range(len(names)) if i != true], min(19, len(names) - 1), replace=False)]
            sc = (F.normalize(z, dim=-1) @ E[cand].t()); return int(sc.argmax()) == 0
        torch.manual_seed(int(cfg.seed) + 4343 + vis)
        P = mlp(2 * d); Pe = torch.nn.Embedding(len(rels), d).to(dev)
        lp = train(P, Pe, [ix[e['A']] for e in fs], [[rx[e['r']]] for e in fs], [ix[e['ans']] for e in fs])
        rs = np.random.RandomState(int(cfg.seed) + 4444 + vis)
        with torch.no_grad():
            pz = lambda a, r: P(torch.cat([E[a][None], Pe(torch.tensor([[r]], device=dev)).flatten(1)], -1))[0]
            one = np.mean([rank(pz(ix[e['A']], rx[e['r']]), ix[e['ans']], rs) for e in fs[:256]])
            base = np.mean([rank(pz(ix[e['A']], rx[e['r']]), ix[e['ans']], rs) for e in hf[:256]]) if hf else float('nan')
        # chains: taught facts, no live conclusion, no direct fact (as the composition probe)
        prev = {rw[3] for pool in D['bysl'].values() for rw in pool} if D and D.get('bysl') else {repr(e['key']) for e in fs}
        live = set()
        for c in S_.get('conc', []):
            if not c.get('freed') and not c.get('retracted') and isinstance(c.get('A'), str) and isinstance(c.get('C'), str): live |= {(c['A'], c['C']), (c['C'], c['A'])}
        direct = {(e['A'], e['ans']) for e in fs + hf} | {(e['ans'], e['A']) for e in fs + hf}
        tf = [e for e in fs if repr(e['key']) in prev]; byA = {}
        for e in tf: byA.setdefault(e['A'], []).append(e)
        test = [(e1, e2) for e1 in tf for e2 in byA.get(e1['ans'], []) if e1['A'] != e2['ans'] and (e1['A'], e2['ans']) not in live and (e1['A'], e2['ans']) not in direct]
        allf = {}
        for e in fs + hf: allf.setdefault(e['A'], []).append(e)
        trc = []                                                               # the system's own concluded chains, with their premise facts
        for c in S_.get('conc', []):
            if c.get('freed') or c.get('retracted') or not all(isinstance(c.get(k), str) for k in ('A', 'B', 'C')): continue
            e1 = next((e for e in allf.get(c['A'], []) if e['ans'] == c['B']), None); e2 = next((e for e in allf.get(c['B'], []) if e['ans'] == c['C']), None)
            if e1 is not None and e2 is not None and c['C'] in ix: trc.append((e1, e2))
        res = {}
        with torch.no_grad():
            if test:
                pp = [rank(pz(int((F.normalize(pz(ix[a['A']], rx[a['r']]), dim=-1) @ E.t()).argmax()), rx[b['r']]), ix[b['ans']], rs) for a, b in test[:256]]
                hop1 = [rank(pz(ix[a['A']], rx[a['r']]), ix[a['ans']], rs) and rank(pz(ix[b['A']], rx[b['r']]), ix[b['ans']], rs) for a, b in test[:256]]
                res['pp'] = (np.mean(pp), len(pp), np.mean([x for x, h in zip(pp, hop1) if h]) if any(hop1) else float('nan'), int(sum(hop1)))
        lq = float('nan')
        if test and len(trc) >= 8:
            Q = mlp(3 * d); Qe = torch.nn.Embedding(len(rels), d).to(dev)
            lq = train(Q, Qe, [ix[a['A']] for a, b in trc], [[rx[a['r']], rx[b['r']]] for a, b in trc], [ix[b['ans']] for a, b in trc])
            with torch.no_grad():
                qz = lambda a, r1, r2: Q(torch.cat([E[a][None], Qe(torch.tensor([[r1, r2]], device=dev)).flatten(1)], -1))[0]
                q_tr = np.mean([rank(qz(ix[a['A']], rx[a['r']], rx[b['r']]), ix[b['ans']], rs) for a, b in trc[:256]])
                q_te = [rank(qz(ix[a['A']], rx[a['r']], rx[b['r']]), ix[b['ans']], rs) for a, b in test[:256]]
                res['q'] = (q_tr, np.mean(q_te), len(q_te))
        def split(sel):
            if not _BRG['held'] or not test: return ""
            out = []
            for nm9, keep in (('ANCHORED', lambda a: a['ans'] in _BRG['anch']), ('NOT anchored', lambda a: a['ans'] in _BRG['held'] and a['ans'] not in _BRG['anch'])):
                tt = [(a, b) for a, b in test if keep(a)]
                if tt:
                    with torch.no_grad():
                        v = [rank(pz(int((F.normalize(pz(ix[a['A']], rx[a['r']]), dim=-1) @ E.t()).argmax()), rx[b['r']]), ix[b['ans']], rs) for a, b in tt]
                    out.append(f"{nm9} {np.mean(v):.2f} (n {len(v)})")
            return (" || held-out chains, P twice: " + " | ".join(out)) if out else ""
        pp = res.get('pp'); q = res.get('q')
        print(f"  [LATENT CHAINING | visit {vis + 1}] trunk name codes (last block, centred) for {len(names)} names, {len(rels)} relations; true C among 20 names (chance 0.05) || "
              f"P(e_A, r) -> e_B trained on {len(fs)} taught facts ({n_st} steps, loss {lp:.2f}): one-hop on them {one:.2f} | on never-taught facts {base:.2f} || "
              f"CHAINS with no conclusion ({len(test)}): P twice {pp[0]:.2f} (n {pp[1]}), among chains whose both hops P gets right {pp[2]:.2f} (n {pp[3]})" if pp else
              f"  [LATENT CHAINING | visit {vis + 1}] {len(names)} names, P one-hop {one:.2f} on taught facts, {base:.2f} never taught || no test chains", end="", flush=True)
        if pp:
            print((f" || Q(e_A, r1, r2) -> e_C trained on {len(trc)} chains the system concluded (loss {lq:.2f}): on them {q[0]:.2f} | on the unconcluded chains {q[1]:.2f} (n {q[2]})" if q else
                   f" || Q: too few concluded chains ({len(trc)})") + split(None), flush=True)
        else: print("", flush=True)
    except Exception as e:
        _fail('inference.distill:lchain'); print(f"  [LATENT CHAINING] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _name_codes(model, names):
    """trunk code of each name: the last block's output mean-pooled over the name in 'About NAME.' (centred, normalised)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    cap = {}; h9 = model.blocks[-1].register_forward_hook(lambda m_, i_, o_: cap.__setitem__('h', o_.detach())); E = []
    try:
        for i0 in range(0, len(names), 64):
            chunk = names[i0:i0 + 64]; X, _ = _ws_batch(model, [f"About {n}.".encode('latin-1', 'replace') for n in chunk])
            with torch.no_grad(): _ws_run(model, X, 0)
            h = cap['h'].float()
            for j, n in enumerate(chunk): E.append(h[j, 5:5 + len(n.encode('latin-1', 'replace'))].mean(0))
    finally: h9.remove()
    E = torch.stack(E); return F.normalize(E - E.mean(0), dim=-1)


def _dw_opalg(model, cfg, S_, ents, held, vis):
    """OPERATOR ALGEBRA ON THE RULE WORLD (readout, 11 Nov; inf_distill_opalg): can relation operators over the trunk's name codes compose,
    and discover composition rules, as in tools/toy_jop.py? On the planted RULE WORLD (inf_rule_world) of the domains visited so far:
    a linear operator per relation (child-of, grandchild-of, supervised-by, overseen-by) fitted on the PLANTED facts - (A) in closed form on
    the trunk's frozen name codes, (B) with a small learned projection head trained JEPA-style (InfoNCE + isotropy). Tests on the HELD-OUT
    composite facts (never planted): the composed operator P_r P_r, the learned composite operator, true answer among all rule-world
    names; RULE DISCOVERY: the product's distance to each relation operator, against a NULL (the same fit with the facts' targets shuffled).
    Also how many planted rule-world facts the store's own reader recorded. Training is not changed."""
    from platypus.inference.probes import inf_rule_world
    try:
        W = inf_rule_world(cfg)
        if not W['planted']: return
        seen = set(int(k) for k in (model.__dict__.get('_t2s') or {}).keys())
        P_ = [r for r in W['planted'] if r[0] in seen]; H_ = [r for r in W['held'] if r[0] in seen]
        if len(P_) < 12 or not H_: return
        st = {(e['A'], e['ans']) for e in ents + held if e['key'][0] == 'fact'}; rec = sum(1 for r in P_ if (r[3], r[4]) in st)
        names = sorted({r[3] for r in P_ + H_} | {r[4] for r in P_ + H_}); ix = {n: i for i, n in enumerate(names)}
        E0 = _name_codes(model, names); d = E0.shape[1]; dev = E0.device; N = len(names)
        rels = ('parent', 'grand', 'mgr', 'skip')
        def fit_ridge(E, perm=None):
            out = {}
            for r in rels:
                fr = [x for x in P_ if x[2] == r]
                if len(fr) < 2: continue
                X = E[[ix[x[3]] for x in fr]]; Yi = [ix[x[4]] for x in fr]
                if perm is not None: Yi = list(np.random.RandomState(perm).permutation(Yi))
                Y = E[Yi]; lam = 1e-2 * float((X * X).sum()) / max(1, X.shape[1])
                out[r] = torch.linalg.solve(X.t() @ X + lam * torch.eye(X.shape[1], device=dev), X.t() @ Y)
            return out
        def fit_head(perm=None, steps=600):
            g = torch.Generator(device='cpu').manual_seed(int(cfg.seed) + 515 + (perm or 0))
            Wp = torch.nn.Linear(d, 128, bias=False).to(dev); Pr = torch.nn.ParameterDict({r: torch.nn.Parameter(torch.eye(128, device=dev)) for r in rels})
            opt = torch.optim.Adam(list(Wp.parameters()) + list(Pr.parameters()), lr=3e-3)
            rows = [(r[2], ix[r[3]], ix[r[4]]) for r in P_]
            if perm is not None:
                rs9 = np.random.RandomState(perm); by = {}
                for k, (r, a, b) in enumerate(rows): by.setdefault(r, []).append(k)
                rows2 = list(rows)
                for r, ks in by.items():
                    tg = [rows[k][2] for k in ks]; rs9.shuffle(tg)
                    for k, t in zip(ks, tg): rows2[k] = (r, rows[k][1], t)
                rows = rows2
            for _ in range(steps):
                E = F.normalize(Wp(E0), dim=-1); loss = 0.0
                for r in rels:
                    br = [(a, b) for rr, a, b in rows if rr == r]
                    if not br: continue
                    a = torch.tensor([x for x, _ in br], device=dev); b = torch.tensor([y for _, y in br], device=dev)
                    loss = loss + F.cross_entropy(F.normalize(E[a] @ Pr[r], dim=-1) @ E.t() / 0.1, b)
                z = E - E.mean(0); cov = z.t() @ z / N
                loss = loss + 0.1 * ((cov * N / 128 - torch.eye(128, device=dev)) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): return F.normalize(Wp(E0), dim=-1), {r: Pr[r].detach() for r in rels}
        def score(E, Pd):
            acc = lambda pred, tgt: float((F.normalize(pred, dim=-1) @ E.t()).argmax(-1).eq(tgt).float().mean())
            res = {}
            for r1, r2 in (('parent', 'grand'), ('mgr', 'skip')):
                hh = [x for x in H_ if x[2] == r2]
                if not hh or r1 not in Pd: continue
                a = torch.tensor([ix[x[3]] for x in hh], device=dev); c = torch.tensor([ix[x[4]] for x in hh], device=dev)
                comp = acc(E[a] @ Pd[r1] @ Pd[r1], c); lrn = acc(E[a] @ Pd[r2], c) if r2 in Pd else float('nan')
                M = Pd[r1] @ Pd[r1]; dist = {r: float(torch.norm(Pd[r] - M) / torch.norm(M)) for r in Pd}
                res[r2] = (comp, lrn, len(hh), dist)
            return res
        with torch.no_grad():
            Ra = score(E0, fit_ridge(E0)); Na = score(E0, fit_ridge(E0, perm=int(cfg.seed) + 77))
        Eb, Pb = fit_head(); Rb = score(Eb, Pb)
        Enb, Pnb = fit_head(perm=int(cfg.seed) + 78); Nb = score(Enb, Pnb)
        def fmt(R, Nl):
            out = []
            for r2, (comp, lrn, n, dist) in R.items():
                r1 = 'parent' if r2 == 'grand' else 'mgr'; nr = Nl.get(r2)
                best = min(dist, key=dist.get); oth = sorted(v for k, v in dist.items() if k != r2)
                out.append(f"{'grandchild = child o child' if r2 == 'grand' else 'overseen = supervised o supervised'} (n {n}): composed operator {comp:.2f}, learned composite operator {lrn:.2f} | "
                           f"RULE: product nearest '{best}', distance to the true composite {dist.get(r2, float('nan')):.2f} vs the others {oth[0] if oth else float('nan'):.2f}"
                           + (f"; NULL (targets shuffled): composed {nr[0]:.2f}, distance to the composite {nr[3].get(r2, float('nan')):.2f}" if nr else ""))
            return " | ".join(out) if out else "none"
        print(f"  [OPERATOR ALGEBRA, RULE WORLD | visit {vis + 1}] {len(P_)} planted rule-world facts in visited domains ({rec} recorded by the store's reader), {len(H_)} HELD-OUT "
              f"composite facts, {N} names (chance {1 / N:.3f}) || (A) FROZEN TRUNK CODES, closed-form operators: {fmt(Ra, Na)} || (B) LEARNED PROJECTION HEAD (JEPA-style): {fmt(Rb, Nb)}", flush=True)
    except Exception as e:
        _fail('inference.distill:opalg'); print(f"  [OPERATOR ALGEBRA] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _rm_fit(rows, N, R, steps, seed, D=64, ortho=()):
    """relational module fit: a free code per name (N x D, unit norm) and a linear operator per relation (R x D x D), trained JEPA-style:
    the operator maps the subject's code to the object's (InfoNCE over every name, temperature 0.1) plus an isotropy term (SIGReg-like);
    relations in ORTHO (one-to-one: each subject one object, each object one subject) are also kept near-orthogonal, so their products
    stay well-conditioned over several hops (many-to-one relations must stay singular and are left free)"""
    g = torch.Generator().manual_seed(seed); rr = np.random.RandomState(seed)
    E = torch.nn.Parameter(torch.randn(N, D, generator=g) * 0.1)
    P = torch.nn.Parameter(torch.eye(D).repeat(R, 1, 1) + 0.01 * torch.randn(R, D, D, generator=g))
    opt = torch.optim.Adam([E, P], lr=3e-3); T = torch.tensor(rows, dtype=torch.long); oi = torch.tensor(sorted(ortho), dtype=torch.long)
    for _ in range(steps):
        b = T[torch.from_numpy(rr.randint(len(rows), size=min(256, len(rows))))]; z = F.normalize(E, dim=-1)
        pred = F.normalize(torch.einsum('bd,bde->be', z[b[:, 1]], P[b[:, 0]]), dim=-1)
        zc = z - z.mean(0); cov = zc.t() @ zc / N
        loss = F.cross_entropy(pred @ z.t() / 0.1, b[:, 2]) + 0.1 * ((cov * N / D - torch.eye(D)) ** 2).mean()
        if len(oi): Po = P[oi]; loss = loss + 0.1 * ((Po @ Po.transpose(1, 2) - torch.eye(D)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return F.normalize(E.detach(), dim=-1), P.detach()


def _rm_mine(pairs, steps, seed, D=64, hist=None, merge=True, data_dist=True, xself=True, ortho=True, derived=None, guards=True):
    """THE RELATIONAL MODULE'S MINER (11 Nov; used by _dw_relmod and tools/relmod_hard.py). pairs = {relation phrase: set of (a, b)}.
    The fixes from the failure analysis, each switchable for the comparison: MERGE - relation phrases whose objects agree on 80%+ of 3+
    shared subjects are one relation (a paraphrase split halves each one's facts and hides rules that cross the two phrasings);
    DATA_DIST - a product's distance to each operator is measured on the CODES OF REAL SUBJECTS of its first relation, not over the whole
    matrix (off the data every operator stays near its identity start, which squeezes the contrast between relations); XSELF - the
    subject and the middle name are excluded at each snap ('except self': a matrix cannot express it, the snap can); ORTHO - one-to-one
    relations kept near-orthogonal (_rm_fit). Always: snap to the nearest name at EVERY hop (resets the residual instead of compounding
    it); one-to-many facts are sets (a proposal is right if it is one of the true objects); a rule must beat 3 shuffled-fact nulls by
    0.05 AND predict >= 0.5 of 3+ of its composite facts by the chain, with a Wilson 95% lower bound >= 0.5 (the functional check). GUARDS (from the hard rule world): the
    check counts only chains whose snapped middle is a known subject of r2 (no snapping through a missing hop), and at least half of the
    checked subjects must have an OBSERVED chain ending on someone other than themselves (with 'except self', r o r-inverse would
    otherwise become 'the most similar person', and grandchild o grandchild 'the nearest root'; a distance-to-identity test removed the
    true sibling rule too: child o parent-of is half identity);
    WRITES (the step-4 policy, readout here): only from a rule that predicts 0.9+ of its observed composite facts AND through STORED
    premises (observed or written before) - the latent chain finds and verifies rules but never writes alone (it snaps through missing
    hops: 57 of 206 latent-only round-2 writes in the hard world were wrong, every stored-premise write right).
    DERIVED = facts the module itself proposed earlier (step 4): used as hops for later chains (three-hop closure), never as evidence -
    not fitted, not merged, not a target of the check (written partial-rule facts would otherwise confirm their own rule). Returns a dict."""
    out = dict(rels=[], merged=[], rules=[], cand=0, cand_alt=0, npairs=0, one=float('nan'), props=[], near=None, N=0, nfacts=0)
    derived = derived or set(); allp = pairs; pairs = {r: {x for x in v if (r,) + tuple(x) not in derived} for r, v in allp.items()}
    rels0 = sorted(pairs, key=lambda r: (-len(pairs[r]), r))
    sub0 = {r: {} for r in rels0}
    for r in rels0:
        for a, b in pairs[r]: sub0[r].setdefault(a, set()).add(b)
    par = {r: r for r in rels0}
    def find(r):
        while par[r] != r: r = par[r]
        return r
    if merge:
        for i, r in enumerate(rels0):
            for r2 in rels0[i + 1:]:
                sh = [a for a in sub0[r2] if a in sub0[r]]
                if len(sh) >= 3 and sum(1 for a in sh if sub0[r][a] & sub0[r2][a]) >= 0.8 * len(sh) and find(r) != find(r2):
                    par[find(r2)] = find(r); out['merged'].append((r2, r))
    cls = {}; cla = {}
    for r in rels0: cls.setdefault(find(r), set()).update(pairs[r]); cla.setdefault(find(r), set()).update(allp[r])
    rels = [r for _, r in sorted(((len(v), r) for r, v in cls.items() if len(v) >= 4), key=lambda x: (-x[0], x[1]))[:24]]
    out['rels'] = rels
    if len(rels) < 2: return out
    sub = {r: {} for r in rels}; obj = {r: {} for r in rels}; suba = {r: {} for r in rels}
    for r in rels:
        for a, b in cls[r]: sub[r].setdefault(a, set()).add(b); obj[r].setdefault(b, set()).add(a)
        for a, b in cla[r]: suba[r].setdefault(a, set()).add(b)
    names = sorted({n for r in rels for a, b in cla[r] for n in (a, b)}); ix = {n: i for i, n in enumerate(names)}; N = len(names)
    rows = sorted((k, ix[a], ix[b]) for k, r in enumerate(rels) for a, b in cls[r]); out['N'] = N; out['nfacts'] = len(rows)
    oneone = [k for k, r in enumerate(rels) if ortho and np.mean([len(v) == 1 for v in sub[r].values()]) >= 0.9 and np.mean([len(v) == 1 for v in obj[r].values()]) >= 0.9]
    out['ortho'] = [rels[k] for k in oneone]
    E, P = _rm_fit(rows, N, len(rels), steps, seed, D, oneone)
    def shuf(s):
        rs = np.random.RandomState(seed + 31 * s); o = []
        for k in range(len(rels)):
            kr = [x for x in rows if x[0] == k]; tg = [x[2] for x in kr]; rs.shuffle(tg); o += [(k, x[1], t) for x, t in zip(kr, tg)]
        return o
    nulls = [_rm_fit(shuf(s), N, len(rels), steps, seed + s + 1, D, oneone) for s in range(3)]
    S = [torch.tensor([ix[a] for a in sub[r]], dtype=torch.long) for r in rels]
    def dists(Em, Pm, i, j, on_data):
        M = Pm[i] @ Pm[j]                                                      # last entry: the IDENTITY (the guard's competitor)
        if not on_data: return np.append((torch.norm(Pm - M, dim=(1, 2)) / torch.norm(M)).numpy(), float(torch.norm(torch.eye(M.shape[0]) - M) / torch.norm(M)))
        Z = Em[S[i]]; Y = Z @ M; A = torch.einsum('nd,kde->kne', Z, Pm)
        return np.append((torch.norm(A - Y, dim=(1, 2)) / torch.norm(Y)).numpy(), float(torch.norm(Z - Y) / torch.norm(Y)))
    def chain(srcs, i, j):                                                     # compose, snap, compose, snap (excluding A and B: 'except self')
        a_ = torch.tensor([ix[a] for a in srcs]); ar = torch.arange(len(srcs))
        s1 = F.normalize(E[a_] @ P[i], dim=-1) @ E.t()
        if xself: s1[ar, a_] = -9
        t1 = s1.topk(2, dim=-1); B = t1.indices[:, 0]
        s2 = F.normalize(E[B] @ P[j], dim=-1) @ E.t()
        if xself: s2[ar, B] = -9; s2[ar, a_] = -9
        t2 = s2.topk(2, dim=-1)
        return [names[int(x)] for x in B], [names[int(x)] for x in t2.indices[:, 0]], torch.minimum(t1.values[:, 0] - t1.values[:, 1], t2.values[:, 0] - t2.values[:, 1])
    def fcheck(i, j, b):                                                       # evidence: OBSERVED facts only (derived ones never count)
        te = [a for a in sub[rels[b]] if a in sub[rels[i]]]
        if not te: return 0.0, 0
        if guards:                                                             # GUARD: the chain must EXIST in the observed facts and end
            ex = sum(1 for a in te if any(c != a for x in sub[rels[i]][a] for c in sub[rels[j]].get(x, ())))   #   on someone other than
            if ex < 0.5 * len(te): return 0.0, -1                              #   the subject (r o r-inverse; missing generations); -1 = guard
        Bs, Cs, _ = chain(te, i, j)
        ok = [(a, c) for a, B, c in zip(te, Bs, Cs) if not guards or B in sub[rels[j]]]    # GUARD: no snapping through a missing hop
        return (sum(1 for a, c in ok if c in sub[rels[b]][a]) / len(ok) if ok else 0.0), len(ok)
    with torch.no_grad():
        pr_ = (F.normalize(torch.einsum('bd,bde->be', E[[x[1] for x in rows]], P[[x[0] for x in rows]]), dim=-1) @ E.t()).argmax(-1).tolist()
        out['one'] = float(np.mean([names[p] in sub[rels[k]][names[a]] for (k, a, _), p in zip(rows, pr_)]))   # right if ANY true object
        near = []
        for i in range(len(rels)):
            for j in range(len(rels)):
                out['npairs'] += 1
                for on_data in (data_dist, not data_dist):
                    d = dists(E, P, i, j, on_data); dn = [dists(En, Pn, i, j, on_data) for En, Pn in nulls]; dI = d[-1]; d = d[:-1]; d[[i, j]] = np.inf
                    best = int(np.argmin(d)); nb = [x[best] for x in dn]; margin = min(nb) - d[best]
                    if on_data != data_dist:                                   # the other metric: counted only, for the comparison
                        out['cand_alt'] += sum(1 for k in range(len(rels)) if np.isfinite(d[k]) and min(x[k] for x in dn) - d[k] > 0.05); continue
                    near.append((margin, i, j, best))
                    for k in np.argsort(d):                                    # EVERY relation beating the null, not only the nearest: one
                        k = int(k)                                             #   product can fit two composites (a strict and a partial rule)
                        if not np.isfinite(d[k]): break
                        nk = [x[k] for x in dn]
                        if min(nk) - d[k] <= 0.05: continue
                        out['cand'] += 1
                        fc, fn = fcheck(i, j, k)
                        out.setdefault('cdiag', []).append((rels[i], rels[j], rels[k], float(d[k]), float(min(nk)), fc, fn))   # diagnostic: every candidate's check
                        n_, p_ = fn, fc; lb_ = (p_ + 1.92 / n_ - 1.96 * np.sqrt(p_ * (1 - p_) / n_ + 0.96 / n_ ** 2)) / (1 + 3.84 / n_) if n_ else 0.0
                        if fn >= 3 and fc >= 0.5 and lb_ >= 0.5:                 # WILSON 95% lower bound >= 0.5 (GPU sleep 3: a FALSE rule passed on 2 of 4)
                            out['rules'].append((rels[i], rels[j], rels[k], float(d[k]), float(np.mean(nk)), float((np.mean(nk) - d[k]) / (np.std(nk) + 1e-6)), fc, fn))
        rk = {(a, b, c) for a, b, c, *_ in out['rules']}
        rej = [x for x in sorted(near, reverse=True) if (rels[x[1]], rels[x[2]], rels[x[3]]) not in rk]
        out['near'] = (rels[rej[0][1]], rels[rej[0][2]], rels[rej[0][3]], rej[0][0]) if rej else None
        if hist is not None:
            for k in list(hist):
                if k not in rk: hist.pop(k)
            for k in rk: hist[k] = hist.get(k, 0) + 1
        for r1, r2, r, *_ in out['rules']:
            i, j = rels.index(r1), rels.index(r2)
            srcs = [a for a in suba[r1] if a not in suba[r]]
            if not srcs: continue
            Bl, Cl, mg = chain(srcs, i, j)
            for k, a in enumerate(srcs):
                if Bl[k] not in suba[r2]: continue                             # the snapped middle must be a known subject of r2
                st = sorted(c for b in suba[r1][a] for c in suba[r2].get(b, ()) if c != a)
                fc_ = [x for x in out['rules'] if x[:3] == (r1, r2, r)][0][6]       # WRITE (step 4 policy): a rule verified at 0.9+ AND both
                dep = 1 if any(c != a for x in sub[r1].get(a, ()) for c in sub[r2].get(x, ())) else 2   # depth: 2 = rests on an inferred fact
                out['props'].append((a, r, Cl[k], float(mg[k]), st, fc_, fc_ >= 0.9 and bool(st), dep))   #   (noise compounds; corroboration by a
                                                                               #   second rule did not help - both share the corrupted premise)
    out['sub'] = suba
    return out


def _dw_relmod(model, cfg, ents, held, vis):
    """RELATIONAL MODULE (step 3 of the rule test, 11 Nov; inf_distill_relmod = training steps; READOUT ONLY, nothing is written).
    tools/toy_rules.py found the true composition rules against a shuffled null and applied them to new families (1.00), while a composite
    relation trained directly did not transfer (0.00). Here the same module (_rm_mine, with the fixes from the failure analysis) runs in
    sleep on the facts the STORE'S READER recorded (every relation phrase - the rule world's and the arbitrary probe relations alike; no
    relation is marked as composable): a free code per name (the entorhinal-style structure map, separate from the trunk, whose learned
    head chose the wrong rule on the GPU) and an operator per relation. Rules found at consecutive sleeps are counted. PROPOSALS: for each
    name A with an r1 fact and no recorded r fact, compose - snap - compose - snap, and propose 'A r C'; scored against the rule world's
    truth (grandchild = parent's parent, overseen = manager's manager) and RECALL of the HELD-OUT composite facts (never planted) in the
    domains visited. Also the same proposals by two STORED hops (A r1 B, B r2 C both recorded)."""
    from platypus.inference.probes import inf_rule_world
    try:
        t0 = time.time(); steps = int(getattr(cfg, 'inf_distill_relmod', 0)); W = inf_rule_world(cfg)
        pairs = {}
        for e in ents + held:
            if e['key'][0] == 'fact': pairs.setdefault(str(e['r']).strip(), set()).add((e['A'], e['ans']))
        WR = model.__dict__.setdefault('_rm_written', {})                     # STEP 4 (inf_distill_relmod_write): facts written at earlier sleeps -
        for (r9, a9, c9) in WR: pairs.setdefault(r9, set()).add((a9, c9))      #   hops for later chains, never evidence (provenance)
        R = _rm_mine(pairs, steps, int(cfg.seed) * 100 + 917, hist=model.__dict__.setdefault('_rm_hist', {}), derived=set(WR))
        rels = R['rels']; mg9 = model.__dict__.setdefault('_rm_merged', set())   # 12 Nov: the paraphrase merges, kept across sleeps (paraphrase alignment)
        for a9, b9 in R['merged']: mg9.add((str(a9), str(b9)))
        if len(rels) < 2: print(f"  [RELATIONAL MODULE | visit {vis + 1}] too few relations recorded ({len(rels)})", flush=True); return
        rw = W.get('rel', {}); par, mgr = W.get('parent', {}), W.get('mgr', {})
        cl = {r: r for r in rels}
        for a, b in R['merged']: cl[a] = cl.get(b, b)
        tr = {}
        for k, m in (('grand', par), ('skip', mgr)):
            if rw.get(k): tr[cl.get(rw[k], rw[k])] = m
        true_rules = {tuple(cl.get(rw.get(x), rw.get(x)) for x in t) for t in (('parent', 'parent', 'grand'), ('mgr', 'mgr', 'skip'))}
        def truth(A, r):
            m = tr.get(r)                                                      # rule-world names: the true composite, or '-' (none exists)
            return (m.get(m.get(A)) or '-') if m is not None and A in m else None
        seen = set(int(k) for k in (model.__dict__.get('_t2s') or {}).keys())
        H_ = [h for h in W.get('held', []) if h[0] in seen]; hk = {(h[3], cl.get(rw.get(h[2]), rw.get(h[2]))): h[4] for h in H_}
        props = R['props']; hi = [p for p in props if p[3] >= 0.1]; st2 = [(p[0], p[1], p[4][0], p[3]) for p in props if p[4]]
        wr = [(p[0], p[1], c, p[3]) for p in props if p[6] for c in p[4]]
        agree = sum(1 for p in props if p[6] and p[2] in p[4]) / max(1, sum(1 for p in props if p[6]))
        def sc(pp):
            kn = [(p, truth(p[0], p[1])) for p in pp]; kn = [(p, t) for p, t in kn if t is not None]
            return len(pp), len(kn), (sum(1 for p, t in kn if p[2] == t) / len(kn) if kn else float('nan'))
        rec = lambda pp: len({(p[0], p[1]) for p in pp if hk.get((p[0], p[1])) == p[2]})
        hist = model.__dict__.get('_rm_hist', {})
        f_rule = lambda t: (f"'{t[0]}' o '{t[1]}' -> '{t[2]}' {t[3]:.2f} vs null {t[4]:.2f} (z {t[5]:+.0f}; predicts {t[6]:.2f} of its {t[7]} recorded facts; "
                            f"{hist.get(t[:3], 0)} sleeps running){'' if t[:3] in true_rules else ' FALSE RULE'}")
        nf = sum(1 for t in R['rules'] if t[:3] not in true_rules)
        miss = [t for t in true_rules if None not in t and all(x in rels for x in t) and t not in {x[:3] for x in R['rules']}]
        a1, k1, p1 = sc(props); a2, k2, p2 = sc(hi); a3, k3, p3 = sc(st2); a4, k4, p4 = sc(wr); nr = R['near']
        nw9 = 0
        if int(getattr(cfg, 'inf_distill_relmod_write', 0)):                   # STEP 4: record this sleep's writes (a rule verified 0.9+, premises stored);
            emA = {}                                                           #   distill_wake_sleep puts them in the question pool (_dw_relmod_write)
            for e in ents + held:
                if e['key'][0] == 'fact': emA.setdefault(e['A'], e)
            for a9, r9, c9, _ in wr:
                if (r9, a9, c9) in WR or a9 not in emA: continue
                WR[(r9, a9, c9)] = dict(d=emA[a9]['d'], src=tuple(emA[a9].get('src') or ()), vis=vis); nw9 += 1
        print(f"  [RELATIONAL MODULE | visit {vis + 1}] {R['N']} names, {len(rels)} relations, {R['nfacts']} facts read by the store's reader"
              + (f" (paraphrases merged: {'; '.join(f'{a} = {b}' for a, b in R['merged'][:4])})" if R['merged'] else "")
              + f"; codes 64-d, {steps} steps, one-hop on its facts {R['one']:.2f} || RULES ({R['npairs']} ordered pairs; {R['cand']} beat 3 shuffled-fact nulls on the "
              f"subjects' codes [{R['cand_alt']} over the whole matrix], kept if the chain also predicts >= 0.5 of 3+ recorded composite facts, Wilson 95% lower bound >= 0.5): {len(R['rules'])} found, {nf} FALSE"
              + (f" | true rules NOT found: {'; '.join(f'{a} o {b} -> {c}' for a, b, c in miss)}" if miss else "") + " | "
              + (" | ".join(f_rule(t) for t in R['rules']) if R['rules'] else "none")
              + (f" | nearest rejected '{nr[0]}' o '{nr[1]}' -> '{nr[2]}' margin {nr[3]:+.2f}" if nr else "")
              + f" || PROPOSALS (compose, snap, compose, snap; A without that fact): {a1}, rule-world truth known for {k1}, precision {p1:.2f} | snap margin >= 0.10: {a2} ({k2} known), precision {p2:.2f}"
              f" | by two STORED hops: {a3} ({k3} known), precision {p3:.2f} | WOULD BE WRITTEN (rule verified 0.9+, premises stored): {a4} ({k4} known), precision {p4:.2f}, "
              f"latent chain agrees {agree:.2f} || HELD-OUT composites in visited domains (never planted, n {len(H_)}): proposed correctly {rec(props)}"
              f" (margin >= 0.10: {rec(hi)}; two stored hops: {rec(st2)}; would be written: {rec(wr)})"
              + (f" || WRITTEN this sleep {nw9} (in all {len(WR)})" if int(getattr(cfg, 'inf_distill_relmod_write', 0)) else "") + f" | {time.time() - t0:.0f}s", flush=True)
        if props:
            ex = [p for p in props if truth(p[0], p[1]) is not None][:2] + [p for p in props if truth(p[0], p[1]) is None][:1]
            print("    e.g. " + " || ".join(f"PROPOSED '{p[0]} {p[1]} {p[2]}.' margin {p[3]:.2f} ({'true' if truth(p[0], p[1]) == p[2] else ('wrong, true ' + str(truth(p[0], p[1])) if truth(p[0], p[1]) else 'truth unknown')})" for p in ex), flush=True)
    except Exception as e:
        _fail('inference.distill:relmod'); print(f"  [RELATIONAL MODULE] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _dw_relmod_real(model, cfg, S_, vis):
    """RELATIONAL MODULE ON REAL TEXT (11 Nov; with inf_distill_relmod and inf_reader_chunk; READOUT ONLY): the same miner (_rm_mine, every
    guard) on the fast sequence channel's records of the real text read (S_['relc']: fillers and the frames between them), the 60 commonest
    frames and the 4,000 best-connected fillers; rules found (frame o frame -> frame) and their proposals scored against the gold relations
    of the embedded Re-DocRED documents where both arguments name benchmark entities (the records themselves: about 0.26 gold offline)"""
    from platypus.inference.probes import _bench
    try:
        t0 = time.time(); RC = S_.get('relc') or {}
        if not RC: print(f"  [RELATIONAL MODULE, REAL TEXT | visit {vis + 1}] no fast-channel records", flush=True); return
        dec = lambda w: w.decode('utf-8', 'ignore') if isinstance(w, (bytes, bytearray)) else str(w)
        fc_ = {}
        for (a, r), ob in RC.items(): fc_[r] = fc_.get(r, 0) + len(ob)
        top = set(sorted(fc_, key=lambda r: -fc_[r])[:60]); deg = {}
        for (a, r), ob in RC.items():
            if r not in top: continue
            for b in ob: deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
        keep = set(sorted(deg, key=lambda n: -deg[n])[:4000]); pairs = {}; nf = 0
        for (a, r), ob in RC.items():
            if r not in top or a not in keep: continue
            for b in ob:
                if b in keep and b != a and nf < 20000: pairs.setdefault(dec(r), set()).add((dec(a), dec(b))); nf += 1
        R = _rm_mine(pairs, min(800, int(getattr(cfg, 'inf_distill_relmod', 800))), int(cfg.seed) * 100 + 919, hist=model.__dict__.setdefault('_rm_hist_real', {}))
        G = _bench(cfg); t2e = G['_t2e']; gold = {(min(h, t), max(h, t)) for h, t, _, _ in G['rel']}
        def ents(x):
            es = set()
            for w in str(x).split(): es |= t2e.get(w, set())
            return es
        def isgold(a, c):
            ea, ec = ents(a), ents(c)
            if not ea or not ec: return None
            return any(x != y and (min(x, y), max(x, y)) in gold for x in ea for y in ec)
        pr = [(p[0], p[1], c) for p in R['props'] for c in (p[4] or [p[2]])]; wr = [(p[0], p[1], c) for p in R['props'] if p[6] for c in p[4]]
        g1 = [isgold(a, c) for a, r, c in pr]; g1 = [x for x in g1 if x is not None]; g2 = [isgold(a, c) for a, r, c in wr]; g2 = [x for x in g2 if x is not None]
        print(f"  [RELATIONAL MODULE, REAL TEXT | visit {vis + 1}] {len(RC)} fast-channel record keys; mined {R['nfacts']} facts, {R['N']} fillers, {len(R['rels'])} frames"
              + (f" (paraphrases merged: {'; '.join(f'{a} = {b}' for a, b in R['merged'][:4])})" if R['merged'] else "")
              + f" | RULES {len(R['rules'])} ({R['cand']} beat the null): " + (" | ".join(f"'{t[0]}' o '{t[1]}' -> '{t[2]}' predicts {t[6]:.2f} of {t[7]}" for t in R['rules'][:6]) or "none")
              + f" || PROPOSALS {len(pr)}, on benchmark entities {len(g1)}, GOLD {np.mean(g1) if g1 else float('nan'):.2f} | would be written {len(wr)}, on benchmark entities {len(g2)}, "
              f"GOLD {np.mean(g2) if g2 else float('nan'):.2f} (nothing written) | {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        _fail('inference.distill:relmod_real'); print(f"  [RELATIONAL MODULE, REAL TEXT] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _dw_relmod_write(model, cfg, S_, S_slot, ents, held, ctl, vis):
    """STEP 4 OF THE RULE TEST (11 Nov; inf_distill_relmod_write 1, with inf_distill_relmod; ONE MODEL CHANGE): the relational module's
    WRITTEN facts (a rule found against the null, verified on 0.9+ of its observed composite facts with a Wilson bound, applied through
    STORED premises only - tools/relmod_hard.py: 746 + 92 writes, all right) join the question pool as facts, taught to the trunk by the
    interleaved rows like any recorded fact; a fifth is held back by the same hash as the never-taught control. Provenance: the module
    keeps them apart (hops for later chains, never evidence for a rule), and a fact the reader later records itself replaces the written
    one. READOUT: how many written, their precision against the rule world's truth, and RECALL FROM THE TRUNK (test wording, store and
    memory off) of facts written at an earlier sleep - taught vs held back, all and the composites that were NEVER PLANTED anywhere."""
    import zlib
    from platypus.inference.probes import inf_rule_world
    from platypus.inference.namemem import _nm_layer
    WR = model.__dict__.get('_rm_written') or {}
    if not WR:
        print(f"  [RELATIONAL MODULE WRITES | visit {vis + 1}] none yet", flush=True); return ents, held
    have = {repr(e['key']) for e in ents + held}; haveAr = {(e['A'], e['r']) for e in ents + held if e['key'][0] == 'fact'}
    W = inf_rule_world(cfg); rw = W.get('rel', {}); par, mgr = W.get('parent', {}), W.get('mgr', {})
    never = {(h[3], rw.get(h[2])) for h in W.get('held', [])}
    tmap = {rw.get('grand'): par, rw.get('skip'): mgr}
    def truth(A, r):
        m = tmap.get(r)
        return (m.get(m.get(A)) or '-') if m is not None and A in m else None
    T_, H_ = [], []; rep = 0
    for (r, A, C), w in WR.items():
        e = dict(key=('fact', A, r, C), A=A, r=r, ans=C, d=w['d'], pfs=[f"{A} {r} "], seen=2, src=w['src'], rm=w['vis'])
        if repr(e['key']) in have or (A, r) in haveAr: rep += 1; continue    # recorded by the reader meanwhile: the observation wins
        wc9 = float(getattr(cfg, 'inf_distill_relmod_write_ctl', 0.0))        # 12 Nov POWERED STEP 4: a separate, larger held-back share for written facts
        held9 = (zlib.crc32((repr(e['key']) + '|w').encode()) % 1000 < wc9 * 1000) if wc9 > 0 else ctl(e)
        (H_ if held9 else T_).append(e)
    kn = [(k, truth(k[1], k[0])) for k in WR]; kn = [(k, t) for k, t in kn if t is not None]
    prec = sum(1 for k, t in kn if k[2] == t) / len(kn) if kn else float('nan')
    old = lambda L: [e for e in L if e['rm'] < vis]; nv = lambda L: [e for e in L if (e['A'], e['r']) in never]
    f = lambda x: f"{x['facts'][0]:.2f} ({x['facts'][1]})"
    cAC = {(c.get('A'), c.get('C')) for c in S_.get('conc', []) if c.get('A') is not None}   # GPU sleep 4: the held-back control answered too - the
    gen = lambda L: [e for e in L if (e['A'], e['ans']) in cAC]; nog = lambda L: [e for e in L if (e['A'], e['ans']) not in cAC]   # sleep layer's own
    def ask():                                                                 #   generic conclusions ('A is connected to C') are taught by edits
        return [_dt_recall(model, S_slot, L) for L in (old(T_), old(H_), nv(old(T_)), nv(old(H_)), nog(old(T_)), nog(old(H_)))]
    rT, rH, rTn, rHn, rTg, rHg = ask(); sub = None
    E = model.__dict__.get('_cf_E'); mo = _nm_layer(model, cfg) if E is not None else None
    if E is not None and mo is not None and tuple(E.shape) == tuple(mo.weight.shape):
        W0 = mo.weight.detach().clone()
        try:
            with torch.no_grad(): mo.weight.copy_((W0.float() - E.float().to(W0.device)).to(W0.dtype))
            sub = ask()
        finally:
            with torch.no_grad(): mo.weight.copy_(W0)
    print(f"  [RELATIONAL MODULE WRITES | visit {vis + 1}] {len(WR)} facts written so far ({sum(1 for w in WR.values() if w['vis'] == vis)} this sleep; rule-world truth known for {len(kn)}, "
          f"precision {prec:.2f}; {sum(1 for k in WR if (k[1], k[0]) in never)} of them composites NEVER PLANTED; {rep} since recorded by the reader) -> question pool: "
          f"{len(T_)} taught, {len(H_)} held back (control); {len(gen(old(T_)))} / {len(gen(old(H_)))} of the earlier ones also concluded by the sleep layer as 'A is connected to C' || "
          f"FROM THE TRUNK, test wording, written at an earlier sleep, closed-form edits IN PLACE: TAUGHT {f(rT)} vs HELD BACK {f(rH)} | NEVER PLANTED: TAUGHT {f(rTn)} vs HELD BACK {f(rHn)} | "
          f"NOT ALSO A SLEEP CONCLUSION: TAUGHT {f(rTg)} vs HELD BACK {f(rHg)}"
          + (f" || edits SUBTRACTED: TAUGHT {f(sub[0])} vs HELD BACK {f(sub[1])} | NEVER PLANTED: TAUGHT {f(sub[2])} vs HELD BACK {f(sub[3])} | NOT ALSO A SLEEP CONCLUSION: "
             f"TAUGHT {f(sub[4])} vs HELD BACK {f(sub[5])}" if sub else " || no closed-form edits"), flush=True)
    try: _dw_relmod_answers(model, S_slot, old(T_), old(H_), par, mgr, truth, rw, vis)
    except Exception as e9:
        _fail('inference.distill:relmod_answers'); print(f"  [RELATIONAL MODULE WRITES, WHAT THE TRUNK ANSWERS] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
    return ents + T_, held + H_


@torch.no_grad()
def _dw_stored(model, S_, S_slot, es, L0, bs=32):
    """12 Nov SPACED REPLAY: repr(key) -> True when the trunk completes the fact's OWN source sentence (every answer byte top-1 after the
    sentence's prefix; store and memory off); facts whose answer is not found in a stored sentence are left out (always replayed)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    out = {}; by = {}
    for e in es:
        d, pos, kd = _dw_decl(S_, e, L0, 1)
        if kd == 'src': by.setdefault(int(S_slot(e['d'])), []).append((repr(e['key']), d, pos, e['ans'].encode('latin-1', 'replace')))
    for sl, L in by.items():
        for i in range(0, len(L), bs):
            ch = L[i:i + bs]; X, _ = _ws_batch(model, [d for k, d, p, a in ch]); lg = _ws_run(model, X, sl)
            for j, (k, d, p, a) in enumerate(ch):
                n_ = min(len(a), lg.shape[1] - p)
                out[k] = n_ == len(a) and bool((lg[j, p:p + n_].argmax(-1) == torch.tensor(list(a), device=lg.device)).all())
    return out


@torch.no_grad()
def _dw_greedy(model, prompt, sl, n=24):
    """the trunk's greedy continuation of PROMPT (store and memory off), up to n bytes or a full stop"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    b = bytearray(prompt)
    for _ in range(n):
        X, _m = _ws_batch(model, [bytes(b) + b' ']); lg = _ws_run(model, X, sl)
        p = len(b) - 1
        if p >= lg.shape[1]: break
        c = int(lg[0, p].argmax())
        if c in (46, 10): break
        b.append(c)
    return bytes(b[len(prompt):]).decode('latin-1', 'replace').strip()


def _dw_relmod_answers(model, S_slot, T_, H_, par, mgr, truth, rw, vis):
    """12 Nov READOUT (after the GPU sleep-6 finding: held-back written composites answered about as often as taught ones): WHAT the trunk
    answers in the test wording for written rule-world facts, taught vs held back - RIGHT, the MIDDLE hop (the parent / manager: one hop
    instead of two), ANOTHER member of the subject's family, the relation's commonest written answer (a HUB), or other; with the mean family
    size (chance of a right guess within the family ~ 1 / size). Guessing within small families predicts right ~ 1 / size and hubs or
    family members for the rest; composition predicts right well above that."""
    def fam(m, A):
        r_ = A; seen = {A}
        while r_ in m and m[r_] not in seen: r_ = m[r_]; seen.add(r_)
        root = r_; out = set()
        for x in set(m) | set(m.values()):
            y = x; sn = {x}
            while y in m and m[y] not in sn: y = m[y]; sn.add(y)
            if y == root: out.add(x)
        return out
    hub = {}
    for e in T_ + H_: hub.setdefault(e['r'], collections.Counter())[e['ans']] += 1
    def cls(L):
        c = collections.Counter(); fs = []; ex = []
        for e in L:
            m = par if e['r'] == rw.get('grand') else mgr if e['r'] == rw.get('skip') else None
            if m is None: continue
            a = _dw_greedy(model, f"It is on record that {e['A']} {e['r']} ".encode('latin-1', 'replace'), int(S_slot(e['d']))); t = truth(e['A'], e['r'])
            F_ = fam(m, e['A']) - {e['A']}; fs.append(max(1, len(F_)))
            k = 'right' if a == t else 'middle hop' if a == m.get(e['A']) else 'family' if a in F_ else 'hub' if hub[e['r']] and a == hub[e['r']].most_common(1)[0][0] else 'other'
            c[k] += 1
            if len(ex) < 3: ex.append(f"{e['A']} {e['r']} -> {a or '(nothing)'} [{t}]")
        n = sum(c.values())
        return (" ".join(f"{k} {c[k]}" for k in ('right', 'middle hop', 'family', 'hub', 'other')) + f" of {n}, family size {np.mean(fs) if fs else float('nan'):.1f}"
                + (" | e.g. " + "; ".join(ex) if ex else ""))
    print(f"  [RELATIONAL MODULE WRITES, WHAT THE TRUNK ANSWERS | visit {vis + 1}] written at an earlier sleep, test wording, edits in place || TAUGHT: {cls(T_)} || "
          f"HELD BACK: {cls(H_)}", flush=True)


def distill_wake_sleep(model, cfg, S_, S_slot, vslot=None, oslots=()):
    """INTERLEAVED DISTILLATION (inf_distill_wake > 0; 10 Nov): at each sleep, report the visit just finished (rows used, the mean
    answer loss, recall from the trunk in test wordings for taught facts against the never-taught control fifth) and rebuild the pool of
    question rows for the next visit; during the visit, distill_wake_step adds inf_distill_wake_rows of them to each real-text training step
    (same optimiser - its Fisher whitening, the real-text loss and the dream rows apply - no burst, no separate optimiser)"""
    import zlib
    from platypus.inference.namemem import _nm_material, _NM
    vis = int(model.__dict__.get('_visit', 0)); D = model.__dict__.get('_dtw')
    pmx = int(getattr(cfg, 'inf_distill_pool_max', 0))                         # 11 Nov CAPACITY: the question pool no longer stops at the name memory's cap
    allm = _nm_material(model, cfg, S_, nmax=pmx if pmx > 0 else None)          #   (inf_nm_max): every fact the store holds (seen 2+ times, latest object)
    cct = int(getattr(cfg, 'inf_distill_conc_ctl', 0))                        # 11 Nov: 1 = EVERY conclusion held back (never asked) - a powered composition test
    ctl = lambda e: (cct and e['key'][0] == 'conclusion') or zlib.crc32(repr(e['key']).encode()) % 5 == 0   #   and every live conclusion, up to inf_distill_pool_max
    ents = [e for e in allm if not ctl(e)]; held = [e for e in allm if ctl(e)]
    if pmx > 0:
        nmm = len(model.__dict__.get('_nm_ents', []) or [])
        print(f"  [DISTILLATION, CAPACITY | visit {vis + 1}] question pool {len(ents)} facts and conclusions taught ({sum(1 for e in ents if e['key'][0] == 'conclusion')} conclusions), "
              f"{len(held)} held back as control, from {len(allm)} the store holds (budget {pmx}; the name memory keeps {nmm}, cap {int(getattr(cfg, 'inf_nm_max', 384))})", flush=True)
    f = lambda x: " | ".join(f"{k} {v[0]:.2f} ({v[1]})" for k, v in x.items()) if x else "none"
    was = _NM['force']; _NM['force'] = False
    try:
        if D is not None and D.get('probe'):
            r = _dt_recall(model, S_slot, D['probe']); h = _dt_recall(model, S_slot, D['hprobe']) if D.get('hprobe') else None
            ls = D['losses']; ak = D.get('akl') or []
            print(f"  [DISTILLATION, INTERLEAVED | visit {vis + 1}] last visit: {D['n']} training steps with {int(getattr(cfg, 'inf_distill_wake_rows', 4))} question rows each "
                  f"(weight {float(cfg.inf_distill_wake):g}), answer loss {np.mean(ls[:20]) if ls else float('nan'):.3f} (first 20) -> {np.mean(ls[-20:]) if ls else float('nan'):.3f} (last 20) | "
                  f"from the trunk, test wordings, after the visit: {len(D['probe'])} taught facts {f(r)} || CONTROL, {len(D['hprobe'])} never-taught {f(h)}"
                  + (f" | soft anchor KL during the visit {np.mean(ak[:20]):.4f} (first 20) -> {np.mean(ak[-20:]):.4f} (last 20)" if ak else ""), flush=True)
        st = (D or {}).get('st') or dict(proj=0, conf=0, cut=[])
        if D is not None and int(getattr(cfg, 'inf_distill_wake_proj', 0)) and st['proj']:
            print(f"  [DISTILLATION, INTERLEAVED | visit {vis + 1}] ANCHOR PROJECTION last visit: {st['proj']} steps, question gradient in conflict with the anchor gradient "
                  f"in {st['conf']} ({st['conf'] / max(1, st['proj']):.2f}); share of the question gradient removed when in conflict: mean {np.mean(st['cut']) if st['cut'] else float('nan'):.3f}", flush=True)
        _gate_report(model, cfg, vis)
        if vslot is not None: _dw_repair(model, cfg, D, int(vslot), [int(s) for s in oslots], vis)
        _dw_ever(model, cfg, S_slot, ents, held, D, vis)
        if int(getattr(cfg, 'inf_distill_conc22', 1)): _dw_conc22(model, cfg, S_, S_slot, ents, held, vis)
        if int(getattr(cfg, 'inf_distill_false_probe', 1)):
            try: _dw_false_probe(model, cfg, S_slot, ents, held, vis)
            except Exception as e9: _fail('inference.distill:false_probe'); print(f"  [DISTILLATION, FALSE MEMORY PROBE] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
        if int(getattr(cfg, 'inf_distill_bench_recall', 1)) and str(getattr(cfg, 'inf_bench', '') or ''):
            try: _dw_bench_recall(model, cfg, S_slot, vis)
            except Exception as e9: _fail('inference.distill:bench_recall'); print(f"  [REAL-TEXT RECALL] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
            try: _dw_bench_diag(model, cfg, S_slot, vis)
            except Exception as e9: _fail('inference.distill:bench_diag'); print(f"  [REAL-TEXT RECALL, DIAGNOSTICS] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
        cgn9 = int(getattr(cfg, 'inf_compress_gain', 0))
        if cgn9 > 0 and (vis + 1) % cgn9 == 0 and str(getattr(cfg, 'inf_bench', '') or ''):   # 12 Nov COMPRESSION GAIN readout (inference.compress)
            from platypus.inference.compress import compress_gain
            compress_gain(model, cfg, S_slot, vis, S_)
        if int(getattr(cfg, 'inf_sstore', 0)):                                 # 13 Nov SURPRISE-GATED STORE (one model change): what it admitted
            try:
                from platypus.inference.sentences import sstore_report
                sstore_report(model, cfg, vis)
            except Exception as e9: _fail('inference.sstore:report'); print(f"  [SURPRISE GATE] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
        pm9 = int(getattr(cfg, 'inf_premises', 0))
        if pm9 and (vis + 1) % max(1, int(getattr(cfg, 'inf_premises_every', 2))) == 0:   # 13 Nov PREMISES AT SCALE readout (inference.premises)
            try:
                from platypus.inference.premises import premises_readout
                dv9 = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=dv9): premises_readout(model, cfg, S_slot, vis)
            except Exception as e9: _fail('inference.premises'); print(f"  [PREMISES AT SCALE] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
        if int(getattr(cfg, 'inf_store_link', 0)):                             # 13 Nov STORE LINKING (inference.compress; one model change)
            from platypus.inference.compress import store_link
            store_link(model, cfg, S_, S_slot, vis)
        if int(getattr(cfg, 'inf_distill_ns_probe', 0)): _dw_ns_probe(model, cfg, S_, S_slot, ents, vslot, oslots, vis)
        if int(getattr(cfg, 'inf_distill_comp_probe', 0)): _dw_comp_probe(model, cfg, S_, S_slot, ents, held, D, vis)
        dv_ = [torch.cuda.current_device()] if torch.cuda.is_available() else []   # 11 Nov: these readouts build small nets (global-RNG inits);
        with torch.random.fork_rng(devices=dv_):                               #   a forked RNG leaves training's random stream untouched
            if int(getattr(cfg, 'inf_distill_lchain', 0)): _dw_latent_chain(model, cfg, S_, S_slot, ents, held, D, vis)   # (local: runs with
            if int(getattr(cfg, 'inf_distill_opalg', 0)): _dw_opalg(model, cfg, S_, ents, held, vis)                       #  opalg diverged
            if int(getattr(cfg, 'inf_distill_relmod', 0)): _dw_relmod(model, cfg, ents, held, vis)                         #  after sleep 1)
            if int(getattr(cfg, 'inf_distill_relmod', 0)) and int(getattr(cfg, 'inf_reader_chunk', 0)): _dw_relmod_real(model, cfg, S_, vis)
        if int(getattr(cfg, 'inf_rti_write', 0)) and int(getattr(cfg, 'inf_premises', 0)):   # 13 Nov END-TO-END REAL-TEXT INFERENCE (one model change)
            try:
                from platypus.inference.premises import rti_write
                ents, held = rti_write(model, cfg, S_slot, ents, held, ctl, vis)
            except Exception as e9: _fail('inference.premises:rti_write'); print(f"  [REAL-TEXT INFERENCE WRITES] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
        if int(getattr(cfg, 'inf_distill_relmod', 0)) and int(getattr(cfg, 'inf_distill_relmod_write', 0)):
            try: ents, held = _dw_relmod_write(model, cfg, S_, S_slot, ents, held, ctl, vis)
            except Exception as e9:
                _fail('inference.distill:relmod_write'); print(f"  [RELATIONAL MODULE WRITES] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
        if int(getattr(cfg, 'inf_distill_fu', 0)): _fu_sleep(model, cfg, S_, S_slot, ents, held, vslot, oslots, vis)
        sp = int(getattr(cfg, 'inf_distill_wake_spaced', 0)); sk = int(getattr(cfg, 'inf_distill_wake_selfkl', 0))
        spc = int(getattr(cfg, 'inf_distill_wake_spaced_conc', 0))             # 11 Nov: 0 = conclusions are NOT spaced (always rehearsed): the taught-wording check
        kn_ents = ents if (sk or spc) else [e for e in ents if e['key'][0] != 'conclusion']   #   runs with the closed-form edits in place, so it would take edit-held
        known = _dw_known(model, S_slot, kn_ents) if (sp or sk) else {}         #   conclusions for learned and stop the rows that give them recall WITHOUT the edits
        iv = dict((D or {}).get('iv') or {})
        if sp:
            for e in ents:
                k_ = repr(e['key'])
                if e['key'][0] == 'conclusion' and not spc: iv[k_] = 1; continue
                if k_ in known: iv[k_] = min(16, 2 * iv.get(k_, 1)) if known[k_] else 1
        rows = []; nt = 0
        from platypus.inference.closed_form import _ws_batch, _ws_run
        ctx_ = {}                                                              # 11 Nov PROMPT DISTILLATION: each fact's / conclusion's stored source sentence(s)
        pdp = int(getattr(cfg, 'inf_distill_pd_probe', 0))
        if int(getattr(cfg, 'inf_distill_wake_pd', 0)) or pdp:
            it9 = S_.get('it', [])
            for e in ents + (held if pdp else []):
                ss = []
                for i9 in (e.get('src') or ())[:2]:
                    if isinstance(i9, (int, np.integer)) and 0 <= int(i9) < len(it9):
                        b9 = it9[int(i9)]['s']
                        b9 = bytes(int(v) & 255 for v in b9.tolist()) if hasattr(b9, 'tolist') and not isinstance(b9, (bytes, bytearray)) else (b9.encode('latin-1', 'replace') if isinstance(b9, str) else bytes(b9))
                        ss.append(b9.strip()[:160])
                if ss: ctx_[repr(e['key'])] = b' '.join(ss) + b' '
            if pdp: _dw_pd_probe(model, cfg, S_slot, ents, held, ctx_, vis)
            H9 = (D or {}).get('pdst')
            if H9 and H9['n']:
                print(f"  [DISTILLATION, PROMPT DISTILLATION | visit {vis + 1}] last visit: {H9['n']} rows with the source sentence in the teacher's context (T {float(getattr(cfg, 'inf_distill_wake_pd_T', 2.0)):g}); "
                      f"answer bytes the TEACHER gets right {H9['t'] / max(1, H9['b']):.2f}, the STUDENT {H9['s'] / max(1, H9['b']):.2f} (of {H9['b']}); rows without a source sentence {H9['nos']} (hard targets)", flush=True)
            if int(getattr(cfg, 'inf_distill_wake_pd', 0)):
                print(f"  [DISTILLATION, PROMPT DISTILLATION | visit {vis + 1}] {len(ctx_)} of {len(ents)} taught facts and conclusions have a stored source sentence for the teacher", flush=True)
        brd = {}
        if int(getattr(cfg, 'inf_distill_bridge', 0)):
            try: brd = _dw_bridge(cfg, S_, ents, held, vis)
            except Exception as e9:
                _fail('inference.distill:bridge'); print(f"  [DISTILLATION, BRIDGE] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True)
        from platypus.inference.jepa import cfg_seq_
        L09 = int(cfg_seq_(model)); FORMS9 = _dt_forms(cfg)
        aw9 = float(getattr(cfg, 'inf_distill_align', 0.0)); decl9 = {}; asrc9 = int(getattr(cfg, 'inf_distill_align_src', 1))
        lf9 = aw9 > 0 and int(getattr(cfg, 'inf_distill_align_lf', 0))
        held_lf9 = [e for e in held if e['key'][0] == 'fact' and zlib.crc32((repr(e['key']) + '|lf').encode()) % 2 == 0] if lf9 else []
        lfrows9 = {}; pararows9 = {}; para9 = aw9 > 0 and int(getattr(cfg, 'inf_distill_align_para', 0))
        for e in held_lf9:
            for fm in FORMS9: lfrows9.setdefault(int(S_slot(e['d'])), []).append((fm.format(A=e['A'], r=e['r']).encode('latin-1', 'replace'), repr(e['key'])))
        if aw9 > 0 or int(getattr(cfg, 'inf_distill_replay', 0)) > 0:          # 12 Nov VIEW ALIGNMENT (readout also for replay without alignment): each TAUGHT fact's declaration (control never aligned)
            A9 = (D or {}).get('al') or []
            try: ct9 = _dw_align_probe(model, S_, S_slot, (D or {}).get('probe') or [], asrc9); ch9 = _dw_align_probe(model, S_, S_slot, (D or {}).get('hprobe') or [], asrc9)
            except Exception: _fail('inference.distill:align_probe'); ct9 = ch9 = 'unavailable'
            for e in ents:
                if e['key'][0] == 'fact': decl9[repr(e['key'])] = _dw_decl(S_, e, L09, asrc9)
            if lf9:                                                            # 12 Nov LABEL-FREE ALIGNMENT: half the never-asked facts (by a second hash)
                L1 = (D or {}).get('al_lf') or []                              #   are aligned (question -> declaration, no answer label); the other half
                try:                                                           #   is the pure control (never asked, never aligned)
                    rl1 = _dt_recall(model, S_slot, (D or {}).get('lfp') or []); rp1 = _dt_recall(model, S_slot, (D or {}).get('pcp') or [])
                    al1 = _dw_align_probe(model, S_, S_slot, (D or {}).get('lfp') or [], asrc9); ap1 = _dw_align_probe(model, S_, S_slot, (D or {}).get('pcp') or [], asrc9)
                except Exception: _fail('inference.distill:align_lf_probe'); rl1 = rp1 = {}; al1 = ap1 = 'unavailable'
                ff = lambda x: " | ".join(f"{k} {v[0]:.2f} ({v[1]})" for k, v in x.items()) if x else "none"
                for e in held_lf9: decl9[repr(e['key'])] = _dw_decl(S_, e, L09, asrc9)
                print(f"  [DISTILLATION, LABEL-FREE ALIGNMENT | visit {vis + 1}] last visit: alignment loss on never-asked facts {np.mean(L1[:20]) if L1 else float('nan'):.3f} (first 20) -> "
                      f"{np.mean(L1[-20:]) if L1 else float('nan'):.3f} (last 20) | FROM THE TRUNK, test wordings: ALIGNED, NEVER ASKED {ff(rl1)} || PURE CONTROL, never asked or aligned "
                      f"{ff(rp1)} || states: aligned {al1} || pure control {ap1} | next visit: {len(held_lf9)} never-asked facts aligned, {len(held) - len(held_lf9)} pure control", flush=True)
            if para9:                                                          # 12 Nov PARAPHRASE ALIGNMENT: taught facts asked with a paraphrase of their relation
                mg = model.__dict__.get('_rm_merged') or set(); eq = {}        #   that the relational module merged (from the text) -> the stored declaration's state
                for a9, b9 in mg: eq.setdefault(a9, set()).add(b9); eq.setdefault(b9, set()).add(a9)
                for _ in range(3):
                    for r9 in list(eq): eq[r9] |= set().union(*[eq.get(x, set()) for x in eq[r9]]) - {r9}
                for e in ents:
                    if e['key'][0] != 'fact' or repr(e['key']) not in decl9: continue
                    for r2 in sorted(eq.get(str(e['r']).strip(), ())):
                        for fm in FORMS9: pararows9.setdefault(int(S_slot(e['d'])), []).append((fm.format(A=e['A'], r=r2).encode('latin-1', 'replace'), repr(e['key'])))
                LP = (D or {}).get('al_pa') or []
                try:
                    prevk = set(((D or {}).get('decl') or {}).keys()); rq8 = np.random.RandomState(int(cfg.seed) + 9797 + vis)   # 12 Nov: a LARGER sample - every
                    tp1 = [e for e in ents if e['key'][0] == 'fact' and e['r'] in _DT_PARA and repr(e['key']) in prevk]           #   taught fact of the last visit and
                    cp1 = [e for e in held if e['key'][0] == 'fact' and e['r'] in _DT_PARA]                                     #   every never-asked fact with a test
                    tp1 = [tp1[int(i)] for i in rq8.permutation(len(tp1))[:96]]; cp1 = [cp1[int(i)] for i in rq8.permutation(len(cp1))[:96]]   #   paraphrase, up to 96 each
                    inm = lambda e: _DT_PARA[e['r']] in eq.get(str(e['r']).strip(), ())
                    rr = [_dt_recall_para(model, S_slot, L) for L in ([e for e in tp1 if inm(e)], [e for e in tp1 if not inm(e)], [e for e in cp1 if inm(e)], [e for e in cp1 if not inm(e)])]
                except Exception: _fail('inference.distill:para_probe'); rr = [(float('nan'), 0)] * 4
                fr = lambda x: f"{x[0]:.2f} ({x[1]})"
                print(f"  [DISTILLATION, PARAPHRASE ALIGNMENT | visit {vis + 1}] last visit: alignment loss on paraphrased questions {np.mean(LP[:20]) if LP else float('nan'):.3f} (first 20) -> "
                      f"{np.mean(LP[-20:]) if LP else float('nan'):.3f} (last 20) | FROM THE TRUNK, the PARAPHRASED test wording ('A is a colleague of'): TAUGHT facts, the test paraphrase "
                      f"WAS merged by the module {fr(rr[0])}, was NOT {fr(rr[1])} || CONTROL (never asked or aligned) merged {fr(rr[2])}, not merged {fr(rr[3])} | merges known: "
                      f"{sorted(mg)[:8]} | next visit: {sum(len(v) for v in pararows9.values())} paraphrase rows", flush=True)
            print(f"  [DISTILLATION, VIEW ALIGNMENT | visit {vis + 1}] last visit: alignment loss (1 - cosine, weight {aw9:g}, "
                  f"{'declaration a stop-gradient target' if int(getattr(cfg, 'inf_distill_align_detach', 1)) else 'both sides trained'}) {np.mean(A9[:20]) if A9 else float('nan'):.3f} (first 20) -> "
                  f"{np.mean(A9[-20:]) if A9 else float('nan'):.3f} (last 20) | NOW, a TEST-wording question's state vs the declaration's: TAUGHT facts {ct9} | "
                  f"CONTROL, never taught or aligned: {ch9} | next visit: {len(decl9)} taught facts with a declaration, "
                  f"{sum(1 for v in decl9.values() if v[2] == 'src')} from the stored source sentence ({len(FORMS9)} question wordings)", flush=True)
        elif len(FORMS9) != len(_DT_FORMS):
            print(f"  [DISTILLATION, WORDINGS | visit {vis + 1}] {len(FORMS9)} self-written question wordings", flush=True)
        for e in ents:
            k_ = repr(e['key']); sl = int(S_slot(e['d']))
            own9 = f"{e['A']} {e['r']} {e['ans']}.".encode('latin-1', 'replace')   #   (never the row's own fact as its descriptor)
            d1 = [d for d in brd.get(e['ans'], []) if d != own9] if e['key'][0] == 'fact' else None
            d2 = [d for d in brd.get(e['A'], []) if d != own9] if e['key'][0] == 'fact' else None
            for j9, fm in enumerate(FORMS9):
                pb = fm.format(A=e['A'], r=e['r']).encode('latin-1', 'replace'); ab = (e['ans'] + '.').encode('latin-1', 'replace'); tl = None
                if d2:                                                         #   BRIDGE ANCHORING: the bridge as SUBJECT - its descriptor before the question,
                    rm9 = L09 - 1 - len(pb) - len(ab) - 1                          #   trimmed from the left to keep the answer in the window
                    if rm9 >= 8: pb = d2[j9 % len(d2)][-rm9:] + b' ' + pb
                if d1: ab = ab + b' ' + d1[j9 % len(d1)]                     #   the bridge as ANSWER - the same descriptor pool after it (trained bytes)
                if sk and known.get(k_):
                    X, _ = _ws_batch(model, [pb + ab])
                    with torch.no_grad(): lg = _ws_run(model, X, sl)[0]
                    p0 = len(pb) - 1; n_ = min(len(ab), lg.shape[0] - p0)
                    if n_ > 0: tl = F.log_softmax(lg[p0:p0 + n_].float(), -1).half(); nt += 1
                rows.append((pb, ab, sl, k_, tl, ctx_.get(k_)))
        if sp or sk:
            nk = sum(known.values())
            print(f"  [DISTILLATION, INTERLEAVED | visit {vis + 1}] TAUGHT-WORDING CHECK: the trunk answers {nk} of {len(known)} taught {'facts and conclusions' if (sk or spc) else 'facts (conclusions always rehearsed)'} in a taught wording"
                  + (f" | spaced rehearsal: mean interval {np.mean([iv.get(repr(e['key']), 1) for e in ents]):.2f}, at 16: {sum(1 for e in ents if iv.get(repr(e['key']), 1) >= 16)}" if sp else "")
                  + (f" | held by KL to their own answers: {nt} rows" if sk else ""), flush=True)
        rf9 = float(getattr(cfg, 'inf_distill_refuse', 0.0))                   # 12 Nov REFUSAL ROWS: questions about FRESH names never read, in the taught
        if 0 < rf9 < 1 and rows:                                               #   wordings, answered with the refusal word - a share rf9 of each slot's pool
            RL9 = (D or {}).get('refl') or []; known9 = {e['A'] for e in ents + held} | {e['ans'] for e in ents + held}
            rr9 = np.random.RandomState(int(cfg.seed) + 9797 + vis); n0 = collections.Counter(rw[2] for rw in rows); rel9 = collections.defaultdict(list); nref9 = 0
            for e in ents: rel9[int(S_slot(e['d']))].append(e['r'])
            allr9 = sorted({e['r'] for e in ents})
            for sl, n_ in n0.items():
                for _ in range(int(round(n_ * rf9 / (1 - rf9)))):
                    nm9 = ''.join(rr9.choice(_DT_REFUSE_SYL, rr9.randint(2, 4))).capitalize()
                    if nm9 in known9: continue
                    rl_ = rel9.get(sl) or allr9; r9 = rl_[rr9.randint(len(rl_))]; fm9 = FORMS9[rr9.randint(len(FORMS9))]
                    rows.append((fm9.format(A=nm9, r=r9).encode('latin-1', 'replace'), (_DT_REFUSE + '.').encode('latin-1', 'replace'), sl, f"refuse|{nm9}|{r9}", None, None)); nref9 += 1
            print(f"  [DISTILLATION, REFUSAL ROWS | visit {vis + 1}] last visit: loss on refusal rows {np.mean(RL9[:20]) if RL9 else float('nan'):.3f} (first 20) -> "
                  f"{np.mean(RL9[-20:]) if RL9 else float('nan'):.3f} (last 20), {len(RL9)} rows | next visit: {nref9} rows about fresh names never read (share {rf9:.2f} of the pool; "
                  f"answer '{_DT_REFUSE}'; names from syllables the false-memory probe never uses; known subjects with a missing relation are NOT trained - the probe's (b) tests transfer)", flush=True)
        rs = np.random.RandomState(int(cfg.seed) + 9393 + vis)
        probe = [ents[int(i)] for i in rs.choice(len(ents), min(32, len(ents)), replace=False)] if ents else []
        hprobe = [held[int(i)] for i in rs.choice(len(held), min(32, len(held)), replace=False)] if held else []
        bysl = {}
        for rw in rows: bysl.setdefault(rw[2], []).append(rw)
        w = {}
        if sp:
            for sl, pool in bysl.items():
                a_ = np.array([1.0 / iv.get(rw[3], 1) for rw in pool]); w[sl] = a_ / a_.sum()
        anc = {}                                                               # ANCHOR SENTENCES per slot (stored real text) for the projection
        if int(getattr(cfg, 'inf_distill_wake_proj', 0)):
            it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]; ra = np.random.RandomState(int(cfg.seed) + 9595 + vis)
            for x in [real[int(i)] for i in ra.permutation(len(real))]:
                sl = int(S_slot(x['d']))
                if len(anc.setdefault(sl, [])) < 64: anc[sl].append(bytes(x['s']))
        cache = None
        if float(getattr(cfg, 'inf_distill_wake_akl', 0.0)) > 0 or int(getattr(cfg, 'inf_distill_repair', 0)):
            cache = _dw_cache(model, cfg, S_, S_slot, set([int(s) for s in oslots] + ([int(vslot)] if vslot is not None else [])), vis)
        hpool = {}
        if int(getattr(cfg, 'inf_distill_wake_hint', 0)):
            H = (D or {}).get('hst')
            if H and H['steps']:
                print(f"  [DISTILLATION, HINTED | visit {vis + 1}] last visit: {H['steps']} steps, positions where the name memory changes the prediction "
                      f"{H['sel'] / max(1, H['pos']):.4f} of {H['pos']:.0f} (the update's support), KL there {np.mean(H['kl'][:20]) if H['kl'] else float('nan'):.4f} (first 20) -> "
                      f"{np.mean(H['kl'][-20:]) if H['kl'] else float('nan'):.4f} (last 20)", flush=True)
            hpool, hents = _dw_hint_pool(model, cfg, S_, S_slot, ents, held, vis)
            probe = [hents[int(i)] for i in rs.choice(len(hents), min(32, len(hents)), replace=False)] if hents else []
            print(f"  [DISTILLATION, HINTED | visit {vis + 1}] pool for the next visit: {sum(len(v) for v in hpool.values())} stored real sentences naming {len(hents)} taught facts' subjects "
                  f"(sentences naming a control subject left out)", flush=True)
        rp9 = int(getattr(cfg, 'inf_distill_replay', 0)); rpl9 = {}; riv9 = {}; rw9 = {}             # 12 Nov SOURCE REPLAY: the stored source sentences of the pool's facts AND
        if rp9 > 0:                                                            #   of the control fifth (read, never asked), replayed as plain text
            R9 = (D or {}).get('rl') or []; nt9 = nc9 = 0; sp9 = int(getattr(cfg, 'inf_distill_replay_space', 0)); riv9 = dict((D or {}).get('riv') or {}); rw9 = {}
            cand9 = []
            for grp9, es9 in ((0, ents), (1, held if int(getattr(cfg, 'inf_distill_replay_ctl', 1)) else [])):
                for e in es9:
                    if e['key'][0] != 'fact': continue
                    ss9 = [x[-(L09 - 1):] for x in _dw_src_bytes(S_, e) if len(x) >= 8]
                    if ss9: cand9.append((grp9, e, ss9))
            st9 = {}
            if sp9 > 1:                                                        # 12 Nov SPACED REPLAY (retire what is stored): a fact whose own source sentence
                try: st9 = _dw_stored(model, S_, S_slot, [e for g_, e, ss_ in cand9], L09)   #   the trunk already completes has its replay interval doubled
                except Exception: _fail('inference.distill:replay_space'); st9 = {}   #   (up to inf_distill_replay_space); one it fails resets to 1
                for g_, e, ss_ in cand9:
                    k9 = repr(e['key'])
                    if k9 in st9: riv9[k9] = min(sp9, 2 * riv9.get(k9, 1)) if st9[k9] else 1
            for g_, e, ss_ in cand9:
                sl9 = int(S_slot(e['d'])); w9 = 1.0 / riv9.get(repr(e['key']), 1) if sp9 > 1 else 1.0
                rpl9.setdefault(sl9, []).extend(ss_); rw9.setdefault(sl9, []).extend([w9] * len(ss_)); nt9 += g_ == 0; nc9 += g_ == 1
            sinfo9 = ""
            if sp9 > 1 and st9:
                stT = [v for (g_, e, ss_) in cand9 for v in [st9.get(repr(e['key']))] if g_ == 0 and v is not None]
                stC = [v for (g_, e, ss_) in cand9 for v in [st9.get(repr(e['key']))] if g_ == 1 and v is not None]
                ivs = [riv9.get(repr(e['key']), 1) for g_, e, ss_ in cand9]; wt = sum(1.0 / x for x in ivs)
                sinfo9 = (f" | SPACED (max interval {sp9}): own sentence completed by taught {np.mean(stT) if stT else float('nan'):.2f} ({len(stT)}), control "
                          f"{np.mean(stC) if stC else float('nan'):.2f} ({len(stC)}); mean interval {np.mean(ivs):.2f}, at {sp9}: {sum(1 for x in ivs if x >= sp9)}; "
                          f"share of replay on facts NOT yet stored {sum(1.0 / riv9.get(repr(e['key']), 1) for g_, e, ss_ in cand9 if not st9.get(repr(e['key']))) / max(wt, 1e-9):.2f}")
            print(f"  [DISTILLATION, SOURCE REPLAY | visit {vis + 1}] last visit: {len(R9)} steps x {rp9} stored source sentences, next-byte loss "
                  f"{np.mean(R9[:20]) if R9 else float('nan'):.3f} (first 20) -> {np.mean(R9[-20:]) if R9 else float('nan'):.3f} (last 20) | next visit: sentences of "
                  f"{nt9} taught facts and {nc9} CONTROL facts (read, never asked), {sum(len(v) for v in rpl9.values())} sentences" + sinfo9, flush=True)
        rtxN9 = int(getattr(cfg, 'inf_distill_rtx', 0)); rtx9 = {}; rtxv9 = {}; rtw9 = {}   # 12 Nov REAL-TEXT REPLAY: the sentences of HALF the benchmark documents read (by
        if rtxN9 > 0 and str(getattr(cfg, 'inf_bench', '') or ''):          #   hash; the other half is the control) replayed as plain text, and alignment
            try:                                                               #   views cut from them (question: lead-in + subject + words up to an object;
                from platypus.inference.probes import _bench                   #   declaration: the sentence up to that object) - no labels, no templates
                Gb = _bench(cfg); R1 = (D or {}).get('rtxl') or []; A1 = (D or {}).get('rtxa') or []; nd9 = nv9 = nt9 = 0
                ttl9 = int(getattr(cfg, 'inf_distill_rtx_title', 0))
                vis9_ = set(int(k) for k in (model.__dict__.get('_t2s') or {}).keys())
                for d9 in sorted(vis9_):
                    sl9 = _rtx_slot(cfg, S_slot, d9, vis9_)
                    if sl9 is None: continue
                    for di9, doc9 in enumerate(Gb['docs'].get(str(d9), [])):
                        if not _rtx_rep(d9, di9): continue
                        ss9 = [x for x in re.split(r'(?<=[.!?])\s+', doc9) if len(x) >= 20]
                        if not ss9: continue
                        m9 = re.match(r"(?:[A-Z][\w'\-]*)(?:\s+(?:of|de|the|and)?\s*[A-Z][\w'\-]*)*", ss9[0]); tt9 = m9.group(0) if m9 else None
                        if ttl9 and tt9:                                       # 13 Nov TITLE FOR PRONOUN (inf_distill_rtx_title): a replayed sentence that opens
                            ss9b = []                                          #   with He / She / It / They (His / Her / Its / Their) is replayed with the
                            for x in ss9:                                      #   document's title in its place ('Title's' for the possessives) - Wikipedia
                                p9 = next((p for p in _RTX_PRON[:8] if x.startswith(p)), None)   # openings are about their title (offline: the title is
                                if p9 is None or x is ss9[0]: ss9b.append(x); continue           # the pronoun's antecedent 0.55, chance 0.21)
                                ss9b.append(tt9 + ("'s " if p9 in ('His ', 'Her ', 'Its ', 'Their ') else ' ') + x[len(p9):]); nt9 += 1
                            ss9 = ss9b
                        rtx9.setdefault(sl9, []).extend(x.encode('latin-1', 'replace')[:L09 - 1] for x in ss9); nd9 += 1
                        if int(getattr(cfg, 'inf_distill_rtx_align', 1)) and aw9 > 0:
                            for x in ss9:
                                v9 = _rtx_views(x, tt9, L09); rtxv9.setdefault(sl9, []).extend(v9); nv9 += len(v9)
                print(f"  [DISTILLATION, REAL-TEXT REPLAY | visit {vis + 1}] last visit: {len(R1)} steps x {rtxN9} benchmark sentences, next-byte loss "
                      f"{np.mean(R1[:20]) if R1 else float('nan'):.3f} (first 20) -> {np.mean(R1[-20:]) if R1 else float('nan'):.3f} (last 20); alignment on real-text views "
                      f"{np.mean(A1[:20]) if A1 else float('nan'):.3f} -> {np.mean(A1[-20:]) if A1 else float('nan'):.3f} | next visit: {nd9} documents replayed (half; the other half "
                      f"read once, the control), {sum(len(v) for v in rtx9.values())} sentences, {nv9} views"
                      + (f", {nt9} pronoun-led sentences replayed with the document's title for the pronoun" if ttl9 else ''), flush=True)
                if int(getattr(cfg, 'inf_rti_evidence', 0)):                  # 13 Nov EVIDENCE REPLAY (inference.premises; cg27a_rti2's one model change)
                    from platypus.inference.premises import rti_evidence
                    ev9 = rti_evidence(cfg, S_slot, vis9_)
                    for sl9_, v9_ in ev9.items(): rtx9.setdefault(sl9_, []).extend(x[:L09 - 1] for x in v9_)
                    print(f"  [REAL-TEXT INFERENCE EVIDENCE | visit {vis + 1}] {sum(len(v) for v in ev9.values())} premise statements of written inferences join the "
                          f"real-text replay pool (slots {sorted(ev9)})", flush=True)
                spx9 = float(getattr(cfg, 'inf_distill_rtx_space', 0.0))
                if spx9 > 0 and rtx9:                                          # 12 Nov REPLAY BY STORAGE CHECK (inf_distill_rtx_space = alpha): before each
                    rtw9 = _rtx_weights(model, rtx9, spx9, vis)                #   visit the trunk reads every replayed sentence; sentences it already predicts
            except Exception as e9:
                _fail('inference.distill:rtx'); print(f"  [DISTILLATION, REAL-TEXT REPLAY] unavailable ({type(e9).__name__}: {e9})"[:240], flush=True); rtx9 = {}; rtxv9 = {}
        lfk9 = set(repr(e['key']) for e in held_lf9); pc9 = [e for e in held if e['key'][0] == 'fact' and repr(e['key']) not in lfk9]; rq9 = np.random.RandomState(int(cfg.seed) + 9696 + vis)
        lfp9 = [held_lf9[int(i)] for i in rq9.choice(len(held_lf9), min(32, len(held_lf9)), replace=False)] if held_lf9 else []
        pcp9 = [pc9[int(i)] for i in rq9.choice(len(pc9), min(32, len(pc9)), replace=False)] if (lf9 and pc9) else []
        rtir9 = []                                                             # 14 Nov RTI DOSE (inf_rti_rows): the taught real-text inferences' own rows
        if int(getattr(cfg, 'inf_rti_rows', 0)):
            try:
                from platypus.inference.premises import _PM as _PM9, _RTI_R as _RR9
                kk9 = {repr(('fact', A9, _RR9, C9)) for (A9, C9) in (_PM9.get('rti') or {})}
                rtir9 = [rw for rw in rows if rw[3] in kk9]
            except Exception: _fail('inference.distill:rti_rows')
            print(f"  [REAL-TEXT INFERENCE DOSE | visit {vis + 1}] next visit: {len(rtir9)} rows of taught real-text inferences, {int(getattr(cfg, 'inf_rti_rows', 0))} of them "
                  f"added to every question-row step (on top of the pool's own rows)", flush=True)
        model.__dict__['_dtw'] = dict(rtir=rtir9, rtil=[], rtx=rtx9, rtxw=rtw9, rtxv=rtxv9, rtxl=[], rtxa=[], para=pararows9, al_pa=[], riv=riv9, rpw=rw9, lf=lfrows9, al_lf=[], lfp=lfp9, pcp=pcp9, rpl=rpl9, rl=[], decl=decl9, al=[], pdst=dict(n=0, b=0, t=0, s=0, nos=0), bysl=bysl, w=w, iv=iv, rs=np.random.RandomState(int(cfg.seed) + 9494 + vis), n=0, losses=[], akl=[], probe=probe, hprobe=hprobe,
                                      anc=anc, cache=cache, hpool=hpool, st=dict(proj=0, conf=0, cut=[]))
    finally: _NM['force'] = was


def distill_wake_loss(model, cfg):
    """inf_distill_wake_rows question rows (one slot, drawn from the pool, by spaced-rehearsal weight when on) - cross-entropy on the answer
    bytes, or KL to the trunk's own answer at the last sleep for rows of facts it already answers (inf_distill_wake_selfkl); 0.0 when no pool"""
    D = model.__dict__.get('_dtw')
    if not D or not D['bysl']: return 0.0
    try:
        from platypus.inference.closed_form import _ws_batch, _ws_run
        k = int(getattr(cfg, 'inf_distill_wake_rows', 4)); rs = D['rs']; sls = list(D['bysl']); sl = sls[rs.randint(len(sls))]; pool = D['bysl'][sl]
        pw = (D.get('w') or {}).get(sl)
        items = [pool[int(i)] for i in rs.choice(len(pool), min(k, len(pool)), replace=False, p=pw)]
        aw = float(getattr(cfg, 'inf_distill_align', 0.0)) if D.get('decl') else 0.0
        X, M = _ws_batch(model, [it[0] + it[1] for it in items])
        if aw > 0: lg, Hq = _ws_hidden(model, lambda: _ws_run(model, X, sl, grad_=True))
        else: lg = _ws_run(model, X, sl, grad_=True); Hq = None
        tot = 0.0; cnt = 0; ce = []
        pd = int(getattr(cfg, 'inf_distill_wake_pd', 0)); LT = None
        if pd and any(len(it) > 5 and it[5] for it in items):                    # PROMPT DISTILLATION: the teacher reads the source sentence, then the question
            from platypus.inference.jepa import cfg_seq_
            L0_ = int(cfg_seq_(model))                                           # the context is trimmed from the LEFT to fit the window (the fact sits at the
            items = [it[:5] + ((it[5][-max(0, L0_ - len(it[0]) - len(it[1]) - 1):] if L0_ - len(it[0]) - len(it[1]) - 1 > 0 else b''),) if len(it) > 5 and it[5] else it for it in items]   #   sentence's end; local windows are 64 bytes)
            Tt, Mt = _ws_batch(model, [(it[5] if len(it) > 5 and it[5] else b'') + it[0] + it[1] for it in items])
            with torch.no_grad(): LT = _ws_run(model, Tt, sl).float()
            T_ = float(getattr(cfg, 'inf_distill_wake_pd_T', 2.0)); ce_w = float(getattr(cfg, 'inf_distill_wake_pd_ce', 0.0)); H9 = D['pdst']
        for r9, it in enumerate(items):
            p0 = len(it[0]) - 1; n_ = min(len(it[1]), lg.shape[1] - p0)
            if n_ <= 0: continue
            z = lg[r9, p0:p0 + n_].float(); tl = it[4] if len(it) > 4 else None
            cx = it[5] if (LT is not None and len(it) > 5) else None
            if cx:
                q0 = len(cx) + p0; m_ = min(n_, LT.shape[1] - q0)
                if m_ > 0:
                    zt = LT[r9, q0:q0 + m_]; zs = z[:m_]; tgt = X[r9, p0 + 1:p0 + 1 + m_]
                    pt = F.softmax(zt / T_, -1); kl = (pt * (F.log_softmax(zt / T_, -1) - F.log_softmax(zs / T_, -1))).sum() * T_ * T_
                    c_ = F.cross_entropy(zs, tgt, reduction='sum'); tot = tot + kl + ce_w * c_; ce.append(float(c_.detach()) / m_); cnt += m_
                    H9['n'] += 1; H9['b'] += m_; H9['t'] += int((zt.argmax(-1) == tgt).sum()); H9['s'] += int((zs.detach().argmax(-1) == tgt).sum())
                    continue
            if LT is not None: H9['nos'] += 1
            if tl is not None:
                t_ = tl[:n_].float(); tot = tot + (t_.exp() * (t_ - F.log_softmax(z, -1))).sum()
            else:
                c_ = F.cross_entropy(z, X[r9, p0 + 1:p0 + 1 + n_], reduction='sum'); tot = tot + c_; ce.append(float(c_.detach()) / n_)
                if str(it[3]).startswith('refuse|'): D.setdefault('refl', []).append(float(c_.detach()) / n_)
            cnt += n_
        if not cnt: return 0.0
        loss = tot / cnt; D['n'] += 1; D['losses'].append(float(np.mean(ce)) if ce else float(loss.detach()))
        rpk = int(getattr(cfg, 'inf_distill_replay', 0)); rpp = (D.get('rpl') or {}).get(sl)
        if rpk > 0 and rpp:                                                    # SOURCE REPLAY: next-byte loss on stored source sentences of this slot
            pw_ = (D.get('rpw') or {}).get(sl); pw_ = (np.asarray(pw_, float) / float(np.sum(pw_))) if pw_ is not None and len(pw_) == len(rpp) and np.sum(pw_) > 0 else None
            ss = [rpp[int(i)] for i in rs.choice(len(rpp), min(rpk, len(rpp), int(np.count_nonzero(pw_)) if pw_ is not None else len(rpp)), replace=False, p=pw_)]
            Xs, Ms = _ws_batch(model, ss); lgs = _ws_run(model, Xs, sl, grad_=True); T9 = lgs.shape[1]; ms = Ms[:, 1:T9 + 1].float()
            cs = F.cross_entropy(lgs.reshape(-1, lgs.shape[-1]).float(), Xs[:, 1:T9 + 1].reshape(-1), reduction='none').view(ms.shape)
            lr_ = (cs * ms).sum() / ms.sum().clamp(min=1); loss = loss + float(getattr(cfg, 'inf_distill_replay_w', 1.0)) * lr_; D['rl'].append(float(lr_.detach()))
        if aw > 0 and Hq is not None:                                          # VIEW ALIGNMENT: the question's state where it predicts the answer ~ the
            dec = [D['decl'].get(it[3]) for it in items]                       #   declaration's state where it predicts it (taught facts only)
            ix = [j for j, d_ in enumerate(dec) if d_ is not None and len(items[j][0]) - 1 < Hq.shape[1]]
            if ix:
                Xd, _ = _ws_batch(model, [dec[j][0] for j in ix])
                if int(getattr(cfg, 'inf_distill_align_detach', 1)):
                    with torch.no_grad(): _, Hd = _ws_hidden(model, lambda: _ws_run(model, Xd, sl))
                else: _, Hd = _ws_hidden(model, lambda: _ws_run(model, Xd, sl, grad_=True))
                ok = [(j, n) for n, j in enumerate(ix) if Hd is not None and dec[j][1] < Hd.shape[1]]
                if ok:
                    hq = torch.stack([Hq[j, len(items[j][0]) - 1] for j, n in ok]).float(); hd = torch.stack([Hd[n, dec[j][1]] for j, n in ok]).float()
                    al = (1 - F.cosine_similarity(hq, hd, dim=-1)).mean(); loss = loss + aw * al; D['al'].append(float(al.detach()))
        rtk = int(getattr(cfg, 'inf_distill_rtx', 0)); rtp = (D.get('rtx') or {}).get(sl)
        if rtk > 0 and rtp:                                                    # REAL-TEXT REPLAY: next-byte loss on replayed benchmark sentences of this slot
            pw3 = (D.get('rtxw') or {}).get(sl); pw3 = pw3 if pw3 is not None and len(pw3) == len(rtp) else None
            ss = [rtp[int(i)] for i in rs.choice(len(rtp), min(rtk, len(rtp)), replace=False, p=pw3)]
            Xs, Ms = _ws_batch(model, ss); lgs = _ws_run(model, Xs, sl, grad_=True); T9 = lgs.shape[1]; ms = Ms[:, 1:T9 + 1].float()
            cs = F.cross_entropy(lgs.reshape(-1, lgs.shape[-1]).float(), Xs[:, 1:T9 + 1].reshape(-1), reduction='none').view(ms.shape)
            lr_ = (cs * ms).sum() / ms.sum().clamp(min=1); loss = loss + float(getattr(cfg, 'inf_distill_replay_w', 1.0)) * lr_; D['rtxl'].append(float(lr_.detach()))
        rk2 = int(getattr(cfg, 'inf_rti_rows', 0)); rp2 = D.get('rtir')
        if rk2 > 0 and rp2:                                                    # 14 Nov RTI DOSE: the written real-text inferences' own question rows
            it2 = [rp2[int(i)] for i in rs.choice(len(rp2), min(rk2, len(rp2)), replace=False)]; sl2 = int(it2[0][2])
            X2, _ = _ws_batch(model, [it[0] + it[1] for it in it2]); lg2 = _ws_run(model, X2, sl2, grad_=True); t2 = 0.0; c2 = 0
            for r9, it in enumerate(it2):
                p0 = len(it[0]) - 1; n_ = min(len(it[1]), lg2.shape[1] - p0)
                if n_ > 0: t2 = t2 + F.cross_entropy(lg2[r9, p0:p0 + n_].float(), X2[r9, p0 + 1:p0 + 1 + n_], reduction='sum'); c2 += n_
            if c2: l2 = t2 / c2; loss = loss + l2; D['rtil'].append(float(l2.detach()))
        rtv = (D.get('rtxv') or {}).get(sl) if aw > 0 else None
        if rtv:                                                                # REAL-TEXT ALIGNMENT: a view's question state -> its declaration's (no labels)
            vv = [rtv[int(i)] for i in rs.choice(len(rtv), min(k, len(rtv)), replace=False)]
            Xq3, _ = _ws_batch(model, [q for q, d_, p_ in vv]); Xd3, _ = _ws_batch(model, [d_ for q, d_, p_ in vv])
            _, Hq3 = _ws_hidden(model, lambda: _ws_run(model, Xq3, sl, grad_=True))
            with torch.no_grad(): _, Hd3 = _ws_hidden(model, lambda: _ws_run(model, Xd3, sl))
            ok3 = [j for j, (q, d_, p_) in enumerate(vv) if Hq3 is not None and Hd3 is not None and len(q) - 1 < Hq3.shape[1] and p_ < Hd3.shape[1]]
            if ok3:
                hq3 = torch.stack([Hq3[j, len(vv[j][0]) - 1] for j in ok3]).float(); hd3 = torch.stack([Hd3[j, vv[j][2]] for j in ok3]).float()
                a3 = (1 - F.cosine_similarity(hq3, hd3, dim=-1)).mean(); loss = loss + aw * a3; D['rtxa'].append(float(a3.detach()))
        for pk_, lk_ in (('lf', 'al_lf'), ('para', 'al_pa')):                 # LABEL-FREE ALIGNMENT (never-asked facts) and PARAPHRASE ALIGNMENT (taught facts asked
            lfp_ = (D.get(pk_) or {}).get(sl) if aw > 0 else None             #   with a merged paraphrase of their relation): questions with no answer bytes and
            if not lfp_: continue                                              #   no cross-entropy -> the stored declaration's state
            qs = [lfp_[int(i)] for i in rs.choice(len(lfp_), min(k, len(lfp_)), replace=False)]
            dq = [(q, D['decl'].get(k9)) for q, k9 in qs if D['decl'].get(k9) is not None]
            if dq:
                Xq2, _ = _ws_batch(model, [q for q, d_ in dq]); Xd2, _ = _ws_batch(model, [d_[0] for q, d_ in dq])
                _, Hq2 = _ws_hidden(model, lambda: _ws_run(model, Xq2, sl, grad_=True))
                with torch.no_grad(): _, Hd2 = _ws_hidden(model, lambda: _ws_run(model, Xd2, sl))
                ok2 = [j for j, (q, d_) in enumerate(dq) if Hq2 is not None and Hd2 is not None and len(q) - 1 < Hq2.shape[1] and d_[1] < Hd2.shape[1]]
                if ok2:
                    hq2 = torch.stack([Hq2[j, len(dq[j][0]) - 1] for j in ok2]).float(); hd2 = torch.stack([Hd2[j, dq[j][1][1]] for j in ok2]).float()
                    a2 = (1 - F.cosine_similarity(hq2, hd2, dim=-1)).mean(); loss = loss + aw * a2; D.setdefault(lk_, []).append(float(a2.detach()))
        return loss
    except Exception as e:
        _fail('inference.distill:wake')
        if not D.get('err'): D['err'] = 1; print(f"  [DISTILLATION, INTERLEAVED] a step failed ({type(e).__name__}: {e})"[:300], flush=True)
        return 0.0


def _dw_anchor_loss(model, cfg, D):
    """next-byte cross-entropy on inf_distill_wake_rows stored real sentences of one slot (the anchor for the projection)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    A = D.get('anc') or {}
    if not A: return None
    rs = D['rs']; sls = list(A); sl = sls[rs.randint(len(sls))]; pool = A[sl]
    X, M = _ws_batch(model, [pool[int(i)] for i in rs.choice(len(pool), min(int(getattr(cfg, 'inf_distill_wake_rows', 4)), len(pool)), replace=False)])
    lg = _ws_run(model, X, sl, grad_=True); T = lg.shape[1]; m = M[:, 1:T + 1].float()
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), X[:, 1:T + 1].reshape(-1), reduction='none').view(m.shape)
    return (ce * m).sum() / m.sum().clamp(min=1)


def distill_wake_step(model, cfg, olds=None, dream=None, dk=1, dkl=None):
    """the interleaved question rows' backward, after the real step's backward (gradients add before the optimiser step); options above"""
    D = model.__dict__.get('_dtw')
    if not D: return
    ond = int(getattr(cfg, 'inf_distill_wake_ondream', 0))
    if ond and not dream: return
    w = float(cfg.inf_distill_wake) * (max(1, int(dk)) if ond else 1)
    if int(getattr(cfg, 'inf_distill_spindle', 0)): return                 # spindle coupling: questions only inside the sleep's replay burst
    if int(getattr(cfg, 'inf_distill_readonly', 0)): return                # CONTROL RUN: the pool and every readout as usual, no question rows trained
    gw = _gate_weight(model, cfg, dkl)
    if gw <= 0: return
    w *= gw
    if int(getattr(cfg, 'inf_distill_fu', 0)) and model.__dict__.get('_fu') is not None:   # FACT UNITS: the question rows train the fact units only
        st = model.__dict__['_fu']; allp = list(model.parameters()); rq = [p.requires_grad for p in allp]
        ts = [t for v in st['P'].values() for t in (v['V'], v['g'])]
        try:
            for p in allp: p.requires_grad_(False)
            for t in ts: t.requires_grad_(True)
            dl_ = distill_wake_loss(model, cfg)
            if torch.is_tensor(dl_) and dl_.requires_grad:
                gs = torch.autograd.grad(w * dl_, ts, allow_unused=True)
                for t, g in zip(ts, gs): t.grad = g
                st['opt'].step(); st['opt'].zero_grad(set_to_none=True); st['steps'] += 1; st['ql'].append(float(dl_.detach()))
                with torch.no_grad():
                    for v in st['P'].values(): v['g'].clamp_(min=0.0)
        except Exception:
            _fail('inference.distill:fu_step')
        finally:
            for t in ts: t.requires_grad_(False); t.grad = None
            for p, q in zip(allp, rq): p.requires_grad_(q)
        return
    proj = int(getattr(cfg, 'inf_distill_wake_proj', 0)); akl = float(getattr(cfg, 'inf_distill_wake_akl', 0.0))
    P, frz = _dw_params(model, cfg)
    for p in frz: p.requires_grad_(False)
    try:
        g0 = None
        if proj: g0 = [p.grad for p in P]; [setattr(p, 'grad', None) for p in P]
        dl_ = distill_wake_hint_loss(model, cfg) if int(getattr(cfg, 'inf_distill_wake_hint', 0)) else distill_wake_loss(model, cfg)
        if not torch.is_tensor(dl_):
            if g0 is not None: [setattr(p, 'grad', g) for p, g in zip(P, g0)]
            return
        tot = w * dl_
        C = D.get('cache') or {}
        if akl > 0 and C and olds:
            sls = [int(s) for s in olds if int(s) in C]
            if sls:
                kl = _dw_kl(model, C, sls[D['rs'].randint(len(sls))], D['rs'], int(getattr(cfg, 'inf_distill_wake_rows', 4)))
                tot = tot + akl * kl; D['akl'].append(float(kl.detach()))
        tot.backward()
        if proj:
            gq = [p.grad for p in P]; [setattr(p, 'grad', None) for p in P]
            la = None
            try: la = _dw_anchor_loss(model, cfg, D)
            except Exception: _fail('inference.distill:anchor')
            if la is not None:
                la.backward(); ga = [p.grad for p in P]
                with torch.no_grad():
                    dot = sum(float((q.float() * a.float()).sum()) for q, a in zip(gq, ga) if q is not None and a is not None)
                    nn_ = sum(float(a.float().pow(2).sum()) for a in ga if a is not None); D['st']['proj'] += 1
                    if dot < 0 and nn_ > 0:
                        c = dot / nn_; nq = sum(float(q.float().pow(2).sum()) for q in gq if q is not None)
                        for i, (q, a) in enumerate(zip(gq, ga)):
                            if q is not None and a is not None: gq[i] = q - c * a
                        D['st']['conf'] += 1; D['st']['cut'].append(min(1.0, (dot * dot / nn_) / max(nq, 1e-30)))   # share of |gq|^2 removed
            for p, g, q in zip(P, g0, gq):
                p.grad = q if g is None else (g if q is None else g + q)
    finally:
        for p in frz: p.requires_grad_(True)


# ---------------------------------------------------------------- HINTED SELF-DISTILLATION (inf_distill_wake_hint, 11 Nov)
# After Uemura and Suzuki (arXiv 2610.05200): self-distillation with a hint updates sparsely (learning without overwriting) where
# supervised fine-tuning on correct outputs updates densely (overwriting). Here the HINT is the name memory: on stored real sentences that
# name taught subjects, TEACHER = the trunk with the name memory on (no gradient), STUDENT = the trunk alone; loss = KL(teacher || student)
# averaged over the positions where the memory changes the prediction (KL above 1e-4; elsewhere teacher = student, no gradient). No
# templated wordings, no one-hot targets. CONTROL: sentences naming any subject of a never-taught (control) fact are left out of the pool,
# so the memory cannot teach the control facts; the taught probe is drawn from facts whose subject is in the pool.
def _dw_hint_pool(model, cfg, S_, S_slot, ents, held, vis):
    import re
    csub = set(e['A'] for e in held); tents = [e for e in ents if e['A'] not in csub]
    if not tents: return {}, []
    it = S_.get('it', []); real = [x for x in it if x.get('k') == 'real' and 30 <= len(x['s']) <= 200]
    rt = re.compile('|'.join(re.escape(n) for n in sorted(set(e['A'] for e in tents), key=len, reverse=True)))
    rc = re.compile('|'.join(re.escape(n) for n in sorted(csub, key=len, reverse=True))) if csub else None
    pool = {}; hit = set(); ra = np.random.RandomState(int(cfg.seed) + 9898 + vis)
    for i in ra.permutation(len(real)):
        x = real[int(i)]; s = bytes(x['s']).decode('latin-1')
        if rc is not None and rc.search(s): continue
        ms = rt.findall(s)
        if not ms: continue
        sl = int(S_slot(x['d']))
        if len(pool.setdefault(sl, [])) < 256: pool[sl].append(bytes(x['s'])); hit.update(ms)
    return pool, [e for e in tents if e['A'] in hit]


def distill_wake_hint_loss(model, cfg):
    D = model.__dict__.get('_dtw')
    if not D or not D.get('hpool'): return 0.0
    from platypus.inference.namemem import _NM
    if not _NM.get('ready'): return 0.0
    try:
        from platypus.inference.closed_form import _ws_batch, _ws_run
        rs = D['rs']; sls = list(D['hpool']); sl = sls[rs.randint(len(sls))]; pool = D['hpool'][sl]
        X, M = _ws_batch(model, [pool[int(i)] for i in rs.choice(len(pool), min(int(getattr(cfg, 'inf_distill_wake_rows', 4)), len(pool)), replace=False)])
        was = _NM['force']
        try:
            _NM['force'] = True
            with torch.no_grad(): lt = F.log_softmax(_ws_run(model, X, sl).float(), -1)
            _NM['force'] = False
            ls = F.log_softmax(_ws_run(model, X, sl, grad_=True).float(), -1)
        finally: _NM['force'] = was
        T = ls.shape[1]; m = M[:, 1:T + 1].float()
        kl = (lt.exp() * (lt - ls)).sum(-1) * m
        sel = (kl.detach() > 1e-4).float(); ns = float(sel.sum())
        H = D.setdefault('hst', dict(steps=0, pos=0.0, sel=0.0, kl=[]))
        H['steps'] += 1; H['pos'] += float(m.sum()); H['sel'] += ns
        if ns < 1: return 0.0
        loss = (kl * sel).sum() / ns; H['kl'].append(float(loss.detach())); D['n'] += 1
        return loss
    except Exception as e:
        _fail('inference.distill:hint')
        if not D.get('err'): D['err'] = 1; print(f"  [DISTILLATION, HINTED] a step failed ({type(e).__name__}: {e})"[:300], flush=True)
        return 0.0


# ---------------------------------------------------------------- RHYTHM GATES (11 Nov; each off by default)
# inf_gate_theta=L      THETA-LIKE PHASE GATE: training alternates ENCODING phases (real text + question rows, no dream rows) and
#                       CONSOLIDATION phases (real text + dream rows, no question rows). Consolidation phases last L steps on average,
#                       encoding phases L x (k - 1), k = the dream interval in force, so the share of dream steps (and the dream dose) is
#                       unchanged; question rows run in encoding phases only, weight x k / (k - 1) (same dose).
#                       inf_gate_theta_jitter=1: phase lengths drawn at random (exponential, mean as above; stochastic);
#                       0: fixed lengths (the control).
# inf_gate_beta=s       BETA-LIKE DRIFT BRAKE: the question rows' weight x min(1, (ref / ema)^s) where ema = running mean of the dream KL
#                       (old domains' drift from the teacher, measured on every dream step) and ref = its median over the previous visit
#                       x inf_gate_beta_ref: writes are throttled while old domains drift more than usual.
# inf_distill_spindle=1 SPINDLE-LIKE COUPLING: no question rows while awake; at each sleep they run INSIDE the repair replay burst
#                       (inf_distill_repair steps: KL to the previous sleep's predictions on old slots + the question loss x the weight).
def gate_dream(model, cfg, step, dk):
    """is this a dream step? (the theta gate when on, else every dk-th step)"""
    L = float(getattr(cfg, 'inf_gate_theta', 0.0)); dk = max(1, int(dk))
    if L <= 0: return step % dk == 0
    G = model.__dict__.get('_gth')
    if G is None:
        G = model.__dict__['_gth'] = dict(step=None, phase='con', left=0, rs=np.random.RandomState(int(cfg.seed) + 5151), n={'enc': 0, 'con': 0}, sw=0, lens=[], k=dk)
    if G['step'] != step:
        G['step'] = step; G['k'] = dk
        if G['left'] <= 0:
            G['phase'] = 'con' if G['phase'] == 'enc' else 'enc'; G['sw'] += 1
            mean = L if G['phase'] == 'con' else L * max(1, dk - 1)
            cap = float(getattr(cfg, 'inf_gate_theta_cap', 0.0))                # 11 Nov: BOUNDED stochastic phases - at most cap x the mean (0 = unbounded)
            ln = G['rs'].exponential(mean) if int(getattr(cfg, 'inf_gate_theta_jitter', 1)) else mean
            if cap > 0: ln = min(ln, cap * mean)
            G['left'] = max(1, int(round(ln)))
            G['lens'].append((G['phase'], G['left']))
        G['left'] -= 1; G['n'][G['phase']] += 1
    return G['phase'] == 'con'


def _gate_weight(model, cfg, dkl):
    """multiplier on the question rows' weight from the theta phase and the beta brake (0 = no question rows this step)"""
    g = 1.0
    if float(getattr(cfg, 'inf_gate_theta', 0.0)) > 0:
        G = model.__dict__.get('_gth')
        if G is not None:
            if G['phase'] != 'enc': return 0.0
            k = max(2, int(G.get('k', 2))); g *= k / (k - 1)
    s = float(getattr(cfg, 'inf_gate_beta', 0.0))
    if s > 0:
        B = model.__dict__.setdefault('_gbt', dict(ema=None, cur=[], ref=None, gs=[], hist=[]))
        if dkl is not None:
            v = float(dkl); B['cur'].append(v); B['ema'] = v if B['ema'] is None else 0.7 * B['ema'] + 0.3 * v
        b = 1.0
        if B['ref'] and B['ema'] is not None and B['ema'] > B['ref']: b = min(1.0, (B['ref'] / B['ema']) ** s)
        B['gs'].append(b); g *= b
    return g


def _gate_report(model, cfg, vis):
    """the gates' readouts for the visit just finished (printed at the sleep), then reset for the next visit"""
    G = model.__dict__.get('_gth')
    if G is not None and float(getattr(cfg, 'inf_gate_theta', 0.0)) > 0:
        le = [l for p, l in G['lens'] if p == 'enc']; lc = [l for p, l in G['lens'] if p == 'con']
        print(f"  [RHYTHM GATE, THETA | visit {vis + 1}] {'stochastic' if int(getattr(cfg, 'inf_gate_theta_jitter', 1)) else 'fixed'} phases: encoding {G['n']['enc']} steps, "
              f"consolidation {G['n']['con']} steps, {G['sw']} switches; phase lengths encoding mean {np.mean(le) if le else float('nan'):.1f} "
              f"(min {min(le) if le else 0}, max {max(le) if le else 0}), consolidation mean {np.mean(lc) if lc else float('nan'):.1f} | dream interval k {G.get('k')}", flush=True)
        G['n'] = {'enc': 0, 'con': 0}; G['sw'] = 0; G['lens'] = []
    B = model.__dict__.get('_gbt')
    if B is not None and float(getattr(cfg, 'inf_gate_beta', 0.0)) > 0:
        print(f"  [RHYTHM GATE, BETA | visit {vis + 1}] dream KL this visit median {np.median(B['cur']) if B['cur'] else float('nan'):.4f} (reference "
              f"{B['ref'] if B['ref'] else float('nan'):.4f}); write gate mean {np.mean(B['gs']) if B['gs'] else float('nan'):.3f}, share of steps throttled "
              f"{np.mean([x < 0.999 for x in B['gs']]) if B['gs'] else float('nan'):.2f}, min {min(B['gs']) if B['gs'] else float('nan'):.3f}", flush=True)
        if B['cur']: B['ref'] = float(np.median(B['cur'])) * float(getattr(cfg, 'inf_gate_beta_ref', 1.0))
        B['cur'] = []; B['gs'] = []; B['ema'] = None
