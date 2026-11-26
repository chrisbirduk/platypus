"""26 Nov MQuAKE PORT (inf_mquake = N cases, 0 = off; MODEL CHANGE when on: the cases' facts are planted in the text it reads).

Real multi-hop knowledge edits from MQuAKE-CF-3k-v2 (Zhong et al., EMNLP 2023; MIT licence; github.com/princeton-nlp/MQuAKE):
each case is a 2-4 hop chain of Wikidata facts (37 relations: citizenship, head of state, author, capital, ...), one to four of its
facts are EDITED (counterfactual), and the multi-hop answer must change with them ('Who is the head of state of the country Ellie
Kemper is a citizen of?' Donald Trump -> after 'Ellie Kemper is a citizen of Croatia.', Kolinda Grabar-Kitarovic). Here, unlike the
rule world, the relations, entities, edits and sentence frames are MQuAKE's, not ours.

The model is trained from scratch, so it knows none of these facts: it READS them.
  - Case selection (no supervision of the model; a consistent world): cases are taken in a seeded order and kept only if every
    fact agrees with the facts already kept (before and after the edits) and no fact a case edits appears in any other kept case.
    Every string must be latin-1 (the reader's bytes); the subject must sit inside each hop's cloze.
  - Planting: each case gets a domain d (its world: every unedited fact of its old and new chains, and the OLD value of each edited
    fact) and a LATER domain d2 (its edits). Each fact is planted as MQuAKE's own cloze + object ('Ellie Kemper is a citizen of
    Croatia.') in inf_mquake_k frames (the cloze alone, 'Records show that ...', ...), inf_mquake_rep copies each. From the second
    pass on, the old value of an edited fact is no longer planted (the edit is the truth).
READOUTS (each sleep; cases whose domain d has been read):
  - [MQuAKE PORT, THE TRUNK'S OWN HOPS] step by step from the subject, each hop MQuAKE's cloze for that relation, greedy, the answer
    becoming the next hop's subject. MULTI-HOP NEW (the edited answer, MQuAKE's multi-hop accuracy) / OLD (stale) - split by whether
    the edit has been read (control: OLD is right before it is), and by 2 / 3 / 4 hops; the EDITED FACT ALONE (single-hop edit
    success); EACH HOP given the true previous entity (what it knows, apart from chaining). The decomposition into relations is given
    (as in MQuAKE's chain-of-thought setting), not the question parsed.
  - [MQuAKE PORT, JEPA MEMORY] (with inf_jmem) the keyed-memory JEPA, trained each sleep on the facts read (an edit, once read,
    replaces the old value), answers by its own hops '<entity> <relation id> ?'.
"""
import collections, json, os
import numpy as np

_MQ = {}
_URL = 'https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k-v2.json'
_FR = ['{c} {o}.', 'Records show that {c} {o}.', 'It is well known that {c} {o}.', 'According to the records, {c} {o}.']


def _l1(s):
    try:
        s.encode('latin-1'); return True
    except UnicodeEncodeError:
        return False


def _load(cfg):
    p = str(getattr(cfg, 'inf_mquake_path', '') or '')
    if p and os.path.exists(p): return json.load(open(p, encoding='utf-8'))
    p2 = os.path.join(os.environ.get('PILE_CACHE', '.'), 'MQuAKE-CF-3k-v2.json')
    if not os.path.exists(p2):
        import urllib.request
        os.makedirs(os.path.dirname(os.path.abspath(p2)), exist_ok=True); urllib.request.urlretrieve(_URL, p2)
    return json.load(open(p2, encoding='utf-8'))


def _frame(c, o, j):
    c2 = c if j == 0 or c.split(' ', 1)[0] not in ('The', 'A', 'An') else c[0].lower() + c[1:]
    return _FR[j % len(_FR)].format(c=c2, o=o)


def mq_build(cfg):
    """the kept cases and the facts to plant (built once)"""
    if 'cases' in _MQ: return _MQ
    _MQ['cases'] = []; _MQ['facts'] = []
    n = int(getattr(cfg, 'inf_mquake', 0))
    if n <= 0: return _MQ
    from platypus.config import PILE_DOMAINS
    nd = len(PILE_DOMAINS); D = _load(cfg); rs = np.random.RandomState(int(cfg.seed) + 7171); keep = []
    for c in D:
        o = c['orig']; T = [tuple(x) for x in o['triples_labeled']]; N = [tuple(x) for x in o['new_triples_labeled']]
        H, HN = c['single_hops'], c['new_single_hops']
        if len(H) != len(T) or len(HN) != len(N) or len(T) != len(N): continue
        if not all(_l1(x) for t in T + N for x in t) or not all(_l1(h['cloze']) for h in H + HN): continue
        if not all(t[0] in h['cloze'] and not h['cloze'].strip().endswith(t[0]) for t, h in zip(T + N, H + HN)): continue
        keep.append(c)
    WF, OLD, NEW, EK, CK = {}, {}, {}, set(), set()
    for i in rs.permutation(len(keep)):
        c = keep[i]; o = c['orig']; T = [tuple(x) for x in o['triples_labeled']]; N = [tuple(x) for x in o['new_triples_labeled']]
        eid = {tuple(x) for x in o['edit_triples']}; ek = {(N[j][0], N[j][1]) for j, x in enumerate(o['new_triples']) if tuple(x) in eid}
        if not ek: continue
        world = {(s, r): ob for (s, r, ob) in T + N if (s, r) not in ek}
        oldv = {(s, r): ob for (s, r, ob) in T if (s, r) in ek}; newv = {(s, r): ob for (s, r, ob) in N if (s, r) in ek}
        if len(world) != len({(s, r, ob) for (s, r, ob) in T + N if (s, r) not in ek}): continue      # an unedited fact with two values
        if any(WF.get(k, v) != v for k, v in world.items()): continue
        if (set(world) | set(oldv)) & EK or ek & CK: continue                                         # an edited fact belongs to one case only
        WF.update(world); OLD.update(oldv); NEW.update(newv); EK |= ek; CK |= set(world)
        d = int(rs.randint(nd - 1)); d2 = int(d + 1 + rs.randint(nd - 1 - d))
        tpl = [h['cloze'].replace(t[0], '{}', 1) for t, h in zip(T, c['single_hops'])]
        tpn = [h['cloze'].replace(t[0], '{}', 1) for t, h in zip(N, c['new_single_hops'])]
        rid = [x[1] for x in o['triples']]
        al = lambda a, xs: {a.lower()} | {x.lower() for x in xs if _l1(x)}
        _MQ['cases'].append(dict(id=c['case_id'], d=d, d2=d2, T=T, N=N, ek=ek, tpl=tpl, tpn=tpn, rid=rid,
                                 old=al(c['answer'], c.get('answer_alias', [])), new=al(c['new_answer'], c.get('new_answer_alias', [])), q=c['questions'][0], na=c['new_answer']))
        for j, (s, r, ob) in enumerate(T):
            _MQ['facts'].append(((s, r) in ek and 'old' or 'world', d, d2, s, r, ob, tpl[j], rid[j]))
        for j, (s, r, ob) in enumerate(N):
            _MQ['facts'].append(((s, r) in ek and 'edit' or 'world', d, d2, s, r, ob, tpn[j], rid[j]))
        if len(_MQ['cases']) >= n: break
    seen_ = set(); F = []
    for f in _MQ['facts']:
        if (f[0], f[1], f[3], f[4], f[5]) not in seen_: seen_.add((f[0], f[1], f[3], f[4], f[5])); F.append(f)
    _MQ['facts'] = F
    nh = collections.Counter(len(c['T']) for c in _MQ['cases']); ne = collections.Counter(len(c['ek']) for c in _MQ['cases'])
    print(f"  [MQuAKE PORT] {len(_MQ['cases'])} of {len(keep)} usable MQuAKE-CF-3k-v2 cases kept (a consistent world); hops "
          + ', '.join(f'{k}: {v}' for k, v in sorted(nh.items())) + ' | edits a case ' + ', '.join(f'{k}: {v}' for k, v in sorted(ne.items()))
          + f" | {sum(f[0] == 'world' for f in F)} world facts, {sum(f[0] == 'old' for f in F)} edited facts (old value read in the case's domain, "
          f"the edit in a later one) | {int(getattr(cfg, 'inf_mquake_k', 3))} frames x {int(getattr(cfg, 'inf_mquake_rep', 1))}", flush=True)
    return _MQ


def mq_sentences(cfg, d, p):
    """the MQuAKE sentences planted in domain d, pass p"""
    if int(getattr(cfg, 'inf_mquake', 0)) <= 0: return []
    M = mq_build(cfg); k = max(1, int(getattr(cfg, 'inf_mquake_k', 3))); rep = max(1, int(getattr(cfg, 'inf_mquake_rep', 1))); out = []
    for (kind, d1, d2, s, r, ob, tpl, rid) in M['facts']:
        here = (kind == 'world' and d1 == d) or (kind == 'old' and d1 == d and int(p) == 0) or (kind == 'edit' and (d2 == d or (int(p) >= 1 and d1 == d)))
        if here: out += [_frame(tpl.format(s), ob, j) for j in range(k) for _ in range(rep)]
    return out


def _read(seen):
    """facts read so far: (s, r) -> (object, domain); an edit, once its domain is read, replaces the old value"""
    R = {}
    for (kind, d1, d2, s, r, ob, tpl, rid) in _MQ.get('facts', []):
        if kind in ('world', 'old') and d1 in seen and not (kind == 'old' and d2 in seen): R.setdefault((s, rid), (ob, d1))
        if kind == 'edit' and d2 in seen: R[(s, rid)] = (ob, d2)
    return R


def _hit(a, S):
    a = (a or '').strip().rstrip('.').lower()
    return bool(a) and (a in S or any('.' in x and x.split('.')[0].strip() == a for x in S))


def mq_trunk_readout(model, cfg, vis, S_slot, seen):
    """step-by-step multi-hop answers from the trunk (greedy on MQuAKE's clozes)"""
    if int(getattr(cfg, 'inf_mquake', 0)) <= 0: return
    from platypus.inference.realchain import _greedy_batch
    M = mq_build(cfg); CS = [c for c in M['cases'] if c['d'] in seen][:int(getattr(cfg, 'inf_mquake_eval', 200))]
    if not CS: print(f"  [MQuAKE PORT, THE TRUNK'S OWN HOPS | visit {vis + 1}] no case's domain read yet", flush=True); return
    def ask(prompts, slots):
        out = [''] * len(prompts); by = collections.defaultdict(list)
        for i, s in enumerate(slots): by[s].append(i)
        for s, ix in by.items():
            for j in range(0, len(ix), 64):
                ch = ix[j:j + 64]; a = _greedy_batch(model, [prompts[i].encode('latin-1', 'replace') for i in ch], int(S_slot(s)))
                for i, x in zip(ch, a): out[i] = x
        return out
    def dom(c, s_, r_):                                                         # the domain the fact was last read in (its slot)
        return c['d2'] if (s_, r_) in c['ek'] and c['d2'] in seen else c['d']
    cur = [c['T'][0][0] for c in CS]
    for h in range(4):                                                         # step by step: the answer is the next hop's subject
        L = [i for i, c in enumerate(CS) if h < len(c['T'])]
        if not L: break
        ans = ask([CS[i]['tpl'][h].format(cur[i]) for i in L], [dom(CS[i], CS[i]['T'][h][0], CS[i]['T'][h][1]) for i in L])
        for i, a in zip(L, ans): cur[i] = a.strip().rstrip('.')
    tr = [(c['N'] if c['d2'] in seen else c['T']) for c in CS]                 # each hop given the TRUE previous entity
    P, Sl, Tr = [], [], []
    for c, ch in zip(CS, tr):
        for h, (s_, r_, ob) in enumerate(ch): P.append((c['tpn'] if c['d2'] in seen else c['tpl'])[h].format(s_)); Sl.append(dom(c, s_, r_)); Tr.append(ob)
    hop = ask(P, Sl); hk = collections.defaultdict(list); k = 0
    for i, (c, ch) in enumerate(zip(CS, tr)):
        for _ in ch: hk[i].append(_hit(hop[k], {Tr[k].lower()})); k += 1
    E = [(c, j) for c in CS for j, (s_, r_, ob) in enumerate(c['N']) if (s_, r_) in c['ek']]   # the EDITED fact alone
    ea = ask([c['tpn'][j].format(c['N'][j][0]) for c, j in E], [dom(c, c['N'][j][0], c['N'][j][1]) for c, j in E]); ed = collections.defaultdict(list)
    for (c, j), a in zip(E, ea):
        ov = next((ob for (s_, r_, ob) in c['T'] if (s_, r_) == (c['N'][j][0], c['N'][j][1])), None)   # None: the edited subject is not on the old chain
        ed[id(c)].append((_hit(a, {c['N'][j][2].lower()}), _hit(a, {ov.lower()}) if ov else False))
    rows = [(c, _hit(a, c['new']), _hit(a, c['old']), all(hk[i]), np.mean(hk[i]), ed[id(c)]) for i, (c, a) in enumerate(zip(CS, cur))]
    f = lambda L, j: f"{np.mean([r[j] for r in L]):.2f}" if L else 'n/a'
    fe = lambda L, j: f"{np.mean([e[j] for r in L for e in r[5]]):.2f}" if any(r[5] for r in L) else 'n/a'
    rd = [r for r in rows if r[0]['d2'] in seen]; ct = [r for r in rows if r[0]['d2'] not in seen]
    byh = ' | '.join(f"{h} hops (n {len(L)}): NEW {f(L, 1)}, OLD {f(L, 2)}" for h in (2, 3, 4) for L in [[r for r in rd if len(r[0]['T']) == h]] if L)
    ex = next((r for r in rd if r[1]), None) or (rd[0] if rd else None)
    print(f"  [MQuAKE PORT, THE TRUNK'S OWN HOPS | visit {vis + 1}] {len(CS)} cases read; step by step from the subject on MQuAKE's clozes (relations given) || "
          f"EDIT READ (n {len(rd)}): MULTI-HOP NEW {f(rd, 1)}, OLD (stale) {f(rd, 2)} | the EDITED FACT alone NEW {fe(rd, 0)}, OLD {fe(rd, 1)} | each hop with the true "
          f"previous entity {f(rd, 4)} (all hops right {f(rd, 3)}) || {byh or 'no hops'} || EDIT NOT YET READ (control: OLD is right; n {len(ct)}): MULTI-HOP OLD {f(ct, 2)}, "
          f"NEW {f(ct, 1)} | each hop {f(ct, 4)}" + (f" || e.g. '{ex[0]['q']}' -> {cur[rows.index(ex)]!r} (new {ex[0]['na']!r})" if ex else ''), flush=True)
    model.__dict__.setdefault('_mq_hist', {})[vis] = dict(new=float(np.mean([r[1] for r in rd])) if rd else None, old=float(np.mean([r[2] for r in rd])) if rd else None, n=len(rd))


def mq_jmem_items(cfg, seen):
    """the facts read so far, for the JEPA memory: '<s> <relation id> <o>.'"""
    if int(getattr(cfg, 'inf_mquake', 0)) <= 0: return []
    mq_build(cfg); return [(f"{s} {r} {ob}.", (s, ob)) for (s, r), (ob, _) in sorted(_read(seen).items())]


def mq_jmem_readout(J, cfg, vis, seen):
    """the JEPA memory's own hops on the cases read"""
    if int(getattr(cfg, 'inf_mquake', 0)) <= 0 or J is None: return
    M = mq_build(cfg); CS = [c for c in M['cases'] if c['d'] in seen][:int(getattr(cfg, 'inf_mquake_eval', 200))]
    if not CS: return
    cur = [c['T'][0][0] for c in CS]; mg = [9.0] * len(CS)
    for h in range(4):
        L = [i for i, c in enumerate(CS) if h < len(c['T'])]
        if not L: break
        a, g = J.hopw([cur[i] for i in L], [CS[i]['rid'][h] for i in L])
        for i, x, y in zip(L, a, g): cur[i] = x; mg[i] = min(mg[i], y)
    rows = [(c, _hit(a, c['new']), _hit(a, c['old'])) for c, a in zip(CS, cur)]
    rd = [r for r in rows if r[0]['d2'] in seen]; ct = [r for r in rows if r[0]['d2'] not in seen]
    f = lambda L, j: f"{np.mean([r[j] for r in L]):.2f}" if L else 'n/a'
    byh = ' | '.join(f"{h} hops (n {len(L)}): NEW {f(L, 1)}" for h in (2, 3, 4) for L in [[r for r in rd if len(r[0]['T']) == h]] if L)
    print(f"  [MQuAKE PORT, JEPA MEMORY | visit {vis + 1}] {len(CS)} cases read; the keyed-memory JEPA's own hops '<entity> <relation id> ?' on the facts read "
          f"(an edit, once read, replaces the old value) || EDIT READ (n {len(rd)}): MULTI-HOP NEW {f(rd, 1)}, OLD (stale) {f(rd, 2)} || {byh or 'no hops'} || "
          f"EDIT NOT YET READ (control; n {len(ct)}): OLD {f(ct, 2)}, NEW {f(ct, 1)}", flush=True)
