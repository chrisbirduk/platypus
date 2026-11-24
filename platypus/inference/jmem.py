"""24 Nov kra50: THE KEYED-MEMORY JEPA AS A PARALLEL MEMORY (inf_jmem). Local tests (tools/jepa_store.py): a JEPA that predicts, from a sentence with
one name masked, the SLOT of that name (one learned slot per name, the slot an EMA of the target encoder's states of the name) stores knowledge
without a generative objective - one-hop recall 0.98-1.00 and 2-3 hop chaining by its own hops 0.99-1.00 on three synthetic worlds; protected
across domains by replay of its own recall, verified by independent views.
Here it runs beside the trunk, on the facts the model has READ: each sleep it trains on the premise sentences of the domains read so far (rule
world: the planted parent / manager facts; real text: the recorded containment premises of the real-text chains), then answers by its own hops,
for comparison with the trunk. Readouts only: nothing it predicts is written or taught.
Own small byte encoder (context + EMA target), predictor, slot table; kept in model.__dict__ (not a submodule: never in the trunk's optimizer or
checkpoint)."""
import time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

L_, V_, PAD_ = 64, 256, 0


class _Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)

    def forward(s, x, kp):
        h = s.n1(x); x = x + s.a(h, h, h, key_padding_mask=kp, need_weights=False)[0]; return x + s.m(s.n2(x))


class _Enc(nn.Module):
    def __init__(s, d, nl):
        super().__init__(); s.emb = nn.Embedding(V_, d); s.pos = nn.Parameter(torch.randn(L_, d) * 0.02); s.b = nn.ModuleList(_Blk(d, 4) for _ in range(nl)); s.n = nn.LayerNorm(d)

    def forward(s, X):
        kp = X == PAD_; x = s.emb(X) + s.pos[:X.shape[1]]
        for b in s.b: x = b(x, kp)
        return s.n(x)


def _tens(seqs, dev):
    X = torch.full((len(seqs), L_), PAD_, dtype=torch.long)
    for k, q in enumerate(seqs): q = q[:L_]; X[k, :len(q)] = torch.tensor(list(q), dtype=torch.long)
    return X.to(dev)


_span_mean = lambda H, sp: torch.stack([H[k, a:min(b, H.shape[1])].mean(0) for k, (a, b) in enumerate(sp)])


class JMem:
    def __init__(s, cfg, dev):
        d = int(getattr(cfg, 'inf_jmem_dim', 192)); s.d = d; s.dev = dev
        g = torch.Generator().manual_seed(17)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            s.ctx = _Enc(d, int(getattr(cfg, 'inf_jmem_layers', 3))).to(dev); s.tgt = _Enc(d, int(getattr(cfg, 'inf_jmem_layers', 3))).to(dev)
            s.pred = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, d)).to(dev)
        s.tgt.load_state_dict(s.ctx.state_dict())
        for p in s.tgt.parameters(): p.requires_grad = False
        s.opt = torch.optim.AdamW(list(s.ctx.parameters()) + list(s.pred.parameters()), lr=float(getattr(cfg, 'inf_jmem_lr', 1e-3)), weight_decay=0.01)
        s.keys = []; s.ki = {}; s.SL = torch.zeros(0, d, device=dev); s.rs = np.random.RandomState(5); s.steps = 0
        s.lr0 = float(getattr(cfg, 'inf_jmem_lr', 1e-3)); s.warm = int(getattr(cfg, 'inf_jmem_warm', 0))

    def _slot(s, k):
        if k not in s.ki:
            s.ki[k] = len(s.keys); s.keys.append(k)
            v = F.normalize(torch.randn(1, s.d, generator=torch.Generator().manual_seed(len(s.keys))), dim=-1).to(s.dev); s.SL = torch.cat([s.SL, v])
        return s.ki[k]

    def train(s, items, steps, bs=48):
        """items: (sentence, [(start, end, key), ...]) - one name masked per sentence, its slot predicted (InfoNCE over the slots, temperature 0.1)"""
        if not items: return float('nan')
        for (t, sp) in items:
            for (_, _, k) in sp: s._slot(k)
        s.ctx.train(); s.pred.train(); ls = []
        with torch.enable_grad():                                              # the sleep's readouts run without gradients
          for _ in range(steps):
              b = [items[i] for i in s.rs.randint(0, len(items), bs)]; Xs, sp, ix, full = [], [], [], []
              for t, spans in b:
                  a_, b_, k = spans[s.rs.randint(len(spans))]
                  Xs.append((t[:a_] + '\x01' * (b_ - a_) + t[b_:]).encode('latin-1', 'replace')); sp.append((a_, b_)); ix.append(s.ki[k]); full.append(t.encode('latin-1', 'replace'))
              with torch.no_grad(): z = F.normalize(_span_mean(s.tgt(_tens(full, s.dev)), sp), dim=-1)
              p = F.normalize(s.pred(_span_mean(s.ctx(_tens(Xs, s.dev)), sp)), dim=-1); it = torch.tensor(ix, device=s.dev)
              wu = int(getattr(s, 'warm', 0))
              if wu:                                                     # 24 Nov kra51: learning-rate warm-up over the first steps (tools/jepa_store.py used OneCycle)
                  for g_ in s.opt.param_groups: g_['lr'] = s.lr0 * min(1.0, (s.steps + 1) / wu)
              l = F.cross_entropy(p @ s.SL.clone().T / 0.1, it); s.opt.zero_grad(); l.backward(); s.opt.step(); ls.append(float(l.detach()))
              with torch.no_grad():
                  for k_, v in zip(ix, z): s.SL[k_] = F.normalize(0.9 * s.SL[k_] + 0.1 * v, dim=0)
                  for pt, pc in zip(s.tgt.parameters(), s.ctx.parameters()): pt.mul_(0.995).add_(pc.detach(), alpha=0.005)
              s.steps += 1
        return float(np.mean(ls[-50:]))

    @torch.no_grad()
    def hop(s, heads, wording):
        """'<head> <wording> ______.' -> (answers, margins) over the slots"""
        if not s.keys: return [''] * len(heads), [0.0] * len(heads)
        s.ctx.eval(); s.pred.eval(); out, mg = [], []
        for i in range(0, len(heads), 256):
            ch = heads[i:i + 256]; qs, sp = [], []
            for h in ch:
                t = f"{h} {wording} " + '\x01' * 6 + '.'; a_ = len(t) - 7; qs.append(t.encode('latin-1', 'replace')); sp.append((a_, a_ + 6))
            v = F.normalize(s.pred(_span_mean(s.ctx(_tens(qs, s.dev)), sp)), dim=-1); t2 = (v @ s.SL.T).topk(min(2, len(s.keys)), -1)
            out += [s.keys[j] for j in t2.indices[:, 0].tolist()]
            mg += ((t2.values[:, 0] - t2.values[:, 1]) if t2.values.shape[1] > 1 else t2.values[:, 0]).tolist()
        return out, mg


def _hopw(s, heads, wordings):
    """hop with a wording per head (grouped by wording)"""
    out, mg = [''] * len(heads), [0.0] * len(heads); by = {}
    for i, w in enumerate(wordings): by.setdefault(w, []).append(i)
    for w, ix in by.items():
        o, g = s.hop([heads[i] for i in ix], w)
        for i, a, b in zip(ix, o, g): out[i] = a; mg[i] = b
    return out, mg


JMem.hopw = _hopw


def _items_from(sents):
    out = []
    for t, names in sents:
        sp = []
        for n in names:
            i = t.find(n)
            if i >= 0 and i + len(n) <= L_: sp.append((i, i + len(n), n))
        if sp and len(t) <= L_: out.append((t, sp))
    return out


def _jmem_answer(model, cfg, vis, W, seen, rw, J, th, up, pure):
    """24 Nov kra53: THE JEPA MEMORY SUPPLIES HOPS. (1) CROSS-CHECK on the held-back composites: its gated two-hop answer against route J's
    gated composite and the step-by-step writes - agreement, and right when they agree / disagree; COMBINED: route J, else a step write, else the
    JEPA memory. (2) MODEL CHANGE (inf_jmem_write): where route J and step by step both abstain, its gated answer is WRITTEN as a composite
    (taught like their writes); every JEPA write is RE-DERIVED each sleep (kept only if its gated answer is the same) and RETRACTED when route
    J's gated composite contradicts it; contradicted writes raise its gate (self-calibration, as kra49)."""
    RJ9 = model.__dict__.get('_rj_ans') or {}; SSW = model.__dict__.get('_ss_written') or {}
    BADJ = model.__dict__.setdefault('_jw_bad_conf', [])
    if BADJ and int(getattr(cfg, 'inf_rule_steps_selfcal', 0)): th = max(th, max(BADJ) + 1e-3)
    H = [x for x in W.get('held', []) if x[2] in ('grand', 'skip') and (not seen or int(x[0]) in seen)][:400]
    ans = {}
    for k in ('grand', 'skip'):
        Lk = [x for x in H if x[2] == k]
        if not Lk: continue
        r_ = 'parent' if k == 'grand' else 'mgr'; b1, g1 = J.hop([x[3] for x in Lk], rw[r_]); c2, g2 = J.hop(b1, rw[r_])
        for x, b, c, a1, a2 in zip(Lk, b1, c2, g1, g2): ans[(rw[k], x[3])] = (c, b, min(a1, a2), x)
    pr9 = lambda L: f"{np.mean(L):.2f}" if L else 'n/a'
    ag, dj, dr, cb = [], [], [], []
    for key, (c, b, g, x) in ans.items():
        jg = g >= th and bool(c) and c not in (x[3], b); rj = RJ9.get((x[3], rw[x[2]])); rjg = bool(rj and rj[1]); sw = SSW.get(key)
        if jg and rjg: ag.append((c == rj[0], c == x[4], rj[0] == x[4]))
        if rjg: cb.append(('J', rj[0] == x[4]))
        elif sw: cb.append(('S', sw[0] == x[4]))
        elif jg: cb.append(('M', c == x[4]))
        else: cb.append(('-', None))
    agr = [a for a, b_, c_ in ag]
    an = [v for t, v in cb if t != '-']
    print(f"  [JEPA MEMORY ANSWERS, CROSS-CHECK | visit {vis + 1}] held-back composites ({len(ans)}), its gated two hops (t = {th:.3f}{', self-calibrated on ' + str(len(BADJ)) + ' contradicted writes' if BADJ else ''}) vs route J's gated composite: "
          f"BOTH answered {len(ag)}: AGREE {pr9(agr)} | when they agree right {pr9([b_ for a, b_, c_ in ag if a])} | when they disagree: JEPA right {pr9([b_ for a, b_, c_ in ag if not a])}, route J right {pr9([c_ for a, b_, c_ in ag if not a])} || "
          f"COMBINED (route J, else a step write, else the JEPA memory): answered {len(an) / max(1, len(cb)):.2f} (route J {sum(t == 'J' for t, v in cb) / max(1, len(cb)):.2f}, step {sum(t == 'S' for t, v in cb) / max(1, len(cb)):.2f}, "
          f"JEPA {sum(t == 'M' for t, v in cb) / max(1, len(cb)):.2f}), right {pr9(an)} (JEPA's share right {pr9([v for t, v in cb if t == 'M'])})", flush=True)
    if not int(getattr(cfg, 'inf_jmem_write', 0)): return
    RW = model.__dict__.setdefault('_ra_written', {}); BR = model.__dict__.setdefault('_ra_bridge', {}); WRm = model.__dict__.setdefault('_rm_written', {})
    JW = model.__dict__.setdefault('_jw_written', {}); kp, rt, rc, nw, okn = [], [], [], 0, []
    for key in list(JW):
        C0, v0, cf0 = JW[key]; a = ans.get(key); A = key[1]; rj = RJ9.get((A, key[0]))
        drop = None
        if rj and rj[1] and rj[0] != C0: drop = 'contra'; BADJ.append(cf0)
        elif a is None or not (a[2] >= th and a[0] == C0): drop = 'rederive'
        if drop is None: kp.append(C0 == (a[3][4] if a else None)); continue
        RW.pop((key[0], A, C0), None); WRm.pop((key[0], A, C0), None); BR.pop((key[0], A, C0), None); JW.pop(key, None)
        tru = a[3][4] if a else None; (rc if drop == 'contra' else rt).append(C0 == tru)
    for key, (c, b, g, x) in ans.items():
        if key in JW or key in SSW or g < th or not c or c in (x[3], b): continue
        rj = RJ9.get((x[3], rw[x[2]]))
        if rj and rj[1]: continue                                              # route J answers this one itself
        RW[(key[0], key[1], c)] = vis; BR[(key[0], key[1], c)] = b; JW[key] = (c, vis, float(g)); nw += 1; okn.append(c == x[4])
    print(f"  [JEPA MEMORY WRITES | visit {vis + 1}] MODEL CHANGE: its gated two-hop answers written where route J and step by step abstained: NEW {nw} (right {pr9(okn)}) | "
          f"earlier JEPA writes: KEPT {len(kp)} (right {pr9(kp)}), RETRACTED on re-derivation {len(rt)} (were right {pr9(rt)}), RETRACTED on route J's contradiction {len(rc)} (were right {pr9(rc)}) | "
          f"{len(JW)} JEPA writes now", flush=True)


def jmem_sleep(model, cfg, vis, W, seen, rw):
    """train the parallel JEPA memory on everything read so far, then compare its recall and chaining with the trunk's"""
    if not int(getattr(cfg, 'inf_jmem', 0)): return
    t0 = time.time()
    try:
        dev = next(model.parameters()).device
        J = model.__dict__.get('_jmem')
        if J is None: J = JMem(cfg, dev); model.__dict__['_jmem'] = J
        # what has been read: rule-world premises of the seen domains, and the recorded premises of the real-text chains
        rwp = [(d, s_, k, a, b) for (d, s_, k, a, b) in W.get('planted', []) if k in ('parent', 'mgr') and (not seen or int(d) in seen)]
        nd9 = ''
        DG9 = int(getattr(cfg, 'inf_jmem_dream_gate', 0)) and getattr(J, 'th1', None) is not None   # 24 Nov kra53: dream only the recalls that pass its own gate
        if int(getattr(cfg, 'inf_jmem_dream', 0)) and J.steps > 0 and (DG9 or getattr(J, 'last_r1', 0.0) >= float(getattr(cfg, 'inf_jmem_dream_min', 0.9))):   # dreams only once its own recall is reliable             # 24 Nov kra52 DREAM REPLAY: premises of EARLIER domains are replayed from the memory's
            cur9 = int(list((model.__dict__.get('_t2s') or {0: 0}).keys())[-1])  #   OWN recall, not the read text (jmem_sim: D0 0.98 / D1 0.99 / D2 1.00 after 3 sleeps)
            old9 = [x for x in rwp if int(x[0]) != cur9]; new9 = [x for x in rwp if int(x[0]) == cur9]; dr9 = []; ok9 = []
            for k in ('parent', 'mgr'):
                Lk = [x for x in old9 if x[2] == k]
                if Lk:
                    o9, g9 = J.hop([x[3] for x in Lk], rw[k])
                    for x, c, gg in zip(Lk, o9, g9):
                        if DG9 and gg < J.th1: continue                        #   (kra53) a recall below its own one-hop gate is not dreamed
                        if c and c != x[3]: dr9.append((f"{x[3]} {rw[k]} {c}.", (x[3], c))); ok9.append(c == x[4])
            items = _items_from(dr9 + [(f"{a} {rw[k]} {b}.", (a, b)) for (d, s_, k, a, b) in new9])
            nd9 = f", {len(dr9)} of them DREAMED from its own recall of earlier domains{' (gated: ' + str(len(old9)) + ' earlier premises, only recalls above its one-hop gate dreamed)' if DG9 else ''} (right {np.mean(ok9) if ok9 else float('nan'):.2f})"
        else:
            items = _items_from([(f"{a} {rw[k]} {b}.", (a, b)) for (d, s_, k, a, b) in rwp])
        from platypus.inference.realchain import _RC
        trn, tst = _RC.get('train') or [], _RC.get('test') or []
        prem = sorted({(c[0], c[1], c[2]) for c in trn + tst} | {(c[2], c[3], c[4]) for c in trn + tst})
        items += _items_from([(f"{h} {w} {t}.", (h, t)) for h, w, t in prem])
        ls = J.train(items, int(getattr(cfg, 'inf_jmem_steps', 1500)))
        # RULE WORLD: one hop on read premises, two hops on planted / held-back / pure composites, the gate on never-read families
        par, mgr = W.get('parent', {}), W.get('mgr', {}); up = {'grand': (par, 'parent'), 'skip': (mgr, 'mgr')}
        s1 = rwp[:300]
        r1 = []; m1 = []
        for k in ('parent', 'mgr'):
            L = [x for x in s1 if x[2] == k]
            if L: o, g1_ = J.hop([x[3] for x in L], rw[k]); r1 += [y == x[4] for y, x in zip(o, L)]; m1 += [g for y, x, g in zip(o, L, g1_) if y == x[4]]
        J.last_r1 = float(np.mean(r1)) if r1 else 0.0
        J.th1 = float(np.quantile(m1, 0.25)) if len(m1) >= 8 else None          # kra53: its one-hop gate (keeps 75% of its right recalls of what it read)                          # kra52: the memory's own one-hop recall on what it read (a self-test on stored text)
        pure = W.get('pure') or set()
        def two(L):
            res = []
            for k in ('grand', 'skip'):
                Lk = [x for x in L if x[2] == k and (not seen or int(x[0]) in seen)][:120]
                if not Lk: continue
                m_, r_ = up[k]; b1, g1 = J.hop([x[3] for x in Lk], rw[r_]); c2, g2 = J.hop(b1, rw[r_])
                res += [(b == m_.get(x[3]), c == x[4], min(a1, a2)) for b, c, a1, a2, x in zip(b1, c2, g1, g2, Lk)]
            return res
        pl = two([x for x in W.get('planted', []) if x[2] in ('grand', 'skip')]); hb = two([x for x in W.get('held', []) if x[3] not in pure]); hu = two([x for x in W.get('held', []) if x[3] in pure])
        okm = [g for (b, c, g) in pl if c]; th = float(np.quantile(okm, 0.25)) if okm else float('inf')
        nr = [(d, s_, k, a, b) for (d, s_, k, a, b) in W.get('planted', []) if k in ('parent', 'mgr') and seen and int(d) not in seen][:200]
        mn = []
        for k in ('parent', 'mgr'):
            Lk = [x for x in nr if x[2] == k]
            if Lk: _, g = J.hop([x[3] for x in Lk], rw[k]); mn += g
        mr = [g for (b, c, g) in pl if c]
        auc = np.mean([[float(a > b) + 0.5 * (a == b) for b in mn] for a in mr]) if mr and mn else float('nan')
        f = lambda R: (f"{len(R)}: bridge {np.mean([b for b, c, g in R]):.2f}, C {np.mean([c for b, c, g in R]):.2f}, GATED answered {np.mean([g >= th for b, c, g in R]):.2f} "
                       f"right {np.mean([c for b, c, g in R if g >= th]) if any(g >= th for b, c, g in R) else float('nan'):.2f}") if R else "0"
        print(f"  [JEPA MEMORY (parallel) | visit {vis + 1}] keyed-memory JEPA beside the trunk, trained this sleep on {len(items)} premise sentences read so far{nd9} "
              f"({len(J.keys)} slots, {J.steps} steps in all, loss {ls:.3f}; readout only) || RULE WORLD: one hop on read premises {np.mean(r1) if r1 else float('nan'):.2f} (n {len(r1)}) | "
              f"TWO HOPS by its own hops: PLANTED {f(pl)} || HELD BACK {f(hb)} || PURE FAMILIES {f(hu)} | never-read heads vs right planted, margin AUC {auc:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if int(getattr(cfg, 'inf_jmem_answer', 0)) or int(getattr(cfg, 'inf_jmem_write', 0)):
            _jmem_answer(model, cfg, vis, W, seen, rw, J, th, up, pure)
        # REAL TEXT: the frozen TEST chains by its own two hops (premises read, chains never)
        if tst:
            T = tst[:150]
            bb, gb = J.hopw([c[0] for c in T], [c[1] for c in T]); cc, gc = J.hopw(bb, [c[3] for c in T])
            rb = np.mean([b == c[2] for b, c in zip(bb, T)]); rc = np.mean([y == c[4] for y, c in zip(cc, T)])
            PRS = _RC.get('pairs') or set(); vd = np.mean([(c[0], b) in PRS and (b, y) in PRS for b, y, c in zip(bb, cc, T)])
            tt = trn[:80]; o, g = J.hopw([c[0] for c in tt], [c[1] for c in tt]); o2, g2 = J.hopw(o, [c[3] for c in tt])
            tb = [(y == c[4], min(a1, a2)) for y, c, a1, a2 in zip(o2, tt, g, g2)]
            tq = [g for ok, g in tb if ok]; t2 = float(np.quantile(tq, 0.25)) if tq else float('inf'); cf = np.minimum(gb, gc); gt = cf >= t2
            print(f"  [JEPA MEMORY (parallel), REAL TEXT | visit {vis + 1}] the frozen TEST chains ({len(T)}) by its own two hops: hop 1 (bridge) {rb:.2f}, C {rc:.2f}, valid (both links recorded) {vd:.2f} | "
                  f"GATED (calibrated on taught chains) answered {gt.mean():.2f}, right {np.mean([y == c[4] for y, c, g in zip(cc, T, gt) if g]) if gt.any() else float('nan'):.2f} | "
                  f"taught chains C {np.mean([ok for ok, g in tb]) if tb else float('nan'):.2f}  (compare: the trunk's STEP BY STEP on the same chains above)", flush=True)
    except Exception as e:
        print(f"  [JEPA MEMORY (parallel) | visit {vis + 1}] unavailable ({type(e).__name__}: {str(e)[:160]})", flush=True)
