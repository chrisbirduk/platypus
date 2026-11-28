"""24 Nov kra50: THE KEYED-MEMORY JEPA AS A PARALLEL MEMORY (inf_jmem). Local tests (tools/jepa_store.py): a JEPA that predicts, from a sentence with
one name masked, the SLOT of that name (one learned slot per name, the slot an EMA of the target encoder's states of the name) stores knowledge
without a generative objective - one-hop recall 0.98-1.00 and 2-3 hop chaining by its own hops 0.99-1.00 on three synthetic worlds; protected
across domains by replay of its own recall, verified by independent views.
Here it runs beside the trunk, on the facts the model has READ: each sleep it trains on the premise sentences of the domains read so far (rule
world: the planted parent / manager facts; real text: the recorded containment premises of the real-text chains), then answers by its own hops,
for comparison with the trunk. Readouts only: nothing it predicts is written or taught.
Own small byte encoder (context + EMA target), predictor, slot table; kept in model.__dict__ (not a submodule: never in the trunk's optimizer or
checkpoint)."""
import collections, time
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


def _pcal(pairs, target, nmin):
    """PRECISION-TARGETED GATE: the lowest margin t such that answers with margin >= t, on items whose truth it READ, are right at least
    `target` of the time with at least `nmin` of them; inf (closed) when no such t (kra53 local: a quantile gate let a weak memory write
    composites 0.26 right and dream recalls 0.45 right)"""
    P_ = sorted(pairs, key=lambda x: -x[0]); best = float('inf'); ok = 0
    for i, (g, c) in enumerate(P_):
        ok += bool(c)
        if i + 1 >= nmin and ok / (i + 1) >= target: best = g
    return best


def _jmem_answer(model, cfg, vis, W, seen, rw, J, th, up, pure):
    """24 Nov kra53: THE JEPA MEMORY SUPPLIES HOPS. (1) CROSS-CHECK on the held-back composites: its gated two-hop answer against route J's
    gated composite and the step-by-step writes - agreement, and right when they agree / disagree; COMBINED: route J, else a step write, else the
    JEPA memory. (2) MODEL CHANGE (inf_jmem_write): where route J and step by step both abstain, its gated answer is WRITTEN as a composite
    (taught like their writes); every JEPA write is RE-DERIVED each sleep (kept only if its gated answer is the same) and RETRACTED when route
    J's gated composite contradicts it; contradicted writes raise its gate (self-calibration, as kra49)."""
    RJ9 = model.__dict__.get('_rj_ans') or {}; SSW = model.__dict__.get('_ss_written') or {}
    BADJ = model.__dict__.setdefault('_jw_bad_conf', [])
    if BADJ and int(getattr(cfg, 'inf_rule_steps_selfcal', 0)): th = max(th, max(BADJ) + 1e-3)
    H = [x for x in W.get('held', []) if x[2] in ('grand', 'skip') and (not seen or int(x[0]) in seen)][:int(getattr(cfg, 'inf_jmem_heldmax', 400))]
    ans = {}
    for k in ('grand', 'skip'):
        Lk = [x for x in H if x[2] == k]
        if not Lk: continue
        r_ = 'parent' if k == 'grand' else 'mgr'; b1, g1 = J.hop([x[3] for x in Lk], rw[r_]); c2, g2 = J.hop(b1, rw[r_])
        for x, b, c, a1, a2 in zip(Lk, b1, c2, g1, g2): ans[(rw[k], x[3])] = (c, b, min(a1, a2), x)
    pr9 = lambda L: f"{np.mean(L):.2f}" if L else 'n/a'
    thp = float(getattr(J, 'thq', th)) if np.isfinite(getattr(J, 'thq', float('inf'))) else th
    model.__dict__['_jm_ans'] = {key: (c, bool(c) and g >= thp and c not in (x[3], b)) for key, (c, b, g, x) in ans.items()}   # kra54: for route J's protection
    th_rj = float('inf'); n_rj = 0
    if int(getattr(cfg, 'inf_jmem_gate_rj', 0)) and np.isfinite(th):          # 25 Nov kra60: the write / answer gate also calibrated on held-back composites that ROUTE J
        pj = [(g, c == RJ9[(x[3], rw[x[2]])][0]) for key, (c, b, g, x) in ans.items()   #   answers above its own gate (0.96-1.00 right), as a label-free reference: the
              if c and RJ9.get((x[3], rw[x[2]])) and RJ9[(x[3], rw[x[2]])][1]]     #   planted-only gate fell to t = 0.003 by visit 5 and the memory's share of the
        n_rj = len(pj); th_rj = _pcal(pj, float(getattr(cfg, 'inf_jmem_wprec', 0.9)), 20)   #   combined answer to 0.71 right by visit 6 (kra57 v3 GPU)
        if np.isfinite(th_rj): th = max(th, th_rj)
    SSL = model.__dict__.get('_ss_last') or {}
    sa = [(c == SSL[key], c == x[4]) for key, (c, b, g, x) in ans.items() if key in SSL and g >= th and c]
    print(f"  [JEPA MEMORY vs STEP BY STEP | visit {vis + 1}] its gated answers where the trunk's step-by-step (ungated) also answered: {len(sa)} | AGREE {pr9([a for a, r in sa])} | "
          f"right when they agree {pr9([r for a, r in sa if a])} | right when they disagree (JEPA) {pr9([r for a, r in sa if not a])}", flush=True)
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
    print(f"  [JEPA MEMORY ANSWERS, CROSS-CHECK | visit {vis + 1}] held-back composites ({len(ans)}), its gated two hops (t = {th:.3f}{f', route-J-calibrated t = {th_rj:.3f} on {n_rj} composites route J answers (kra60)' if int(getattr(cfg, 'inf_jmem_gate_rj', 0)) else ''}{', self-calibrated on ' + str(len(BADJ)) + ' contradicted writes' if BADJ else ''}) vs route J's gated composite: "
          f"BOTH answered {len(ag)}: AGREE {pr9(agr)} | when they agree right {pr9([b_ for a, b_, c_ in ag if a])} | when they disagree: JEPA right {pr9([b_ for a, b_, c_ in ag if not a])}, route J right {pr9([c_ for a, b_, c_ in ag if not a])} || "
          f"COMBINED (route J, else a step write, else the JEPA memory): answered {len(an) / max(1, len(cb)):.2f} (route J {sum(t == 'J' for t, v in cb) / max(1, len(cb)):.2f}, step {sum(t == 'S' for t, v in cb) / max(1, len(cb)):.2f}, "
          f"JEPA {sum(t == 'M' for t, v in cb) / max(1, len(cb)):.2f}), right {pr9(an)} (JEPA's share right {pr9([v for t, v in cb if t == 'M'])})", flush=True)
    if not int(getattr(cfg, 'inf_jmem_write', 0)): return
    RW = model.__dict__.setdefault('_ra_written', {}); BR = model.__dict__.setdefault('_ra_bridge', {}); WRm = model.__dict__.setdefault('_rm_written', {})
    JW = model.__dict__.setdefault('_jw_written', {}); kp, rt, rc, nw, okn = [], [], [], 0, []
    EO = int(getattr(cfg, 'inf_jmem_edit_override', 0)); EDJ = (model.__dict__.get('_jm_edited') or {}) if EO else {}   # 25 Nov kra57 EDIT OVERRIDE
    OVR = model.__dict__.setdefault('_jw_override', {}); eo_new, eo_rj, eo_ss = [], [], []
    MVt = W.get('moves') or {}; tr7 = lambda x: (MVt[x[3]]['C_new'] if x[3] in MVt and MVt[x[3]]['d2'] in seen else x[4])   # truth after an edit read (readout only)
    for key in list(JW):
        C0, v0, cf0 = JW[key]; a = ans.get(key); A = key[1]; rj = RJ9.get((A, key[0]))
        drop = None
        if rj and rj[1] and rj[0] != C0 and key not in EDJ: drop = 'contra'; BADJ.append(cf0)   # kra57: on an edited premise route J is the stale one
        elif a is None or not (a[2] >= (min(th, thp) if int(getattr(cfg, 'inf_jmem_write_hyst', 0)) else th) and a[0] == C0): drop = 'rederive'   # kra54: re-derived at the looser quantile gate (kra53 v5: the precision gate closed and all 19 right JEPA writes were dropped)
        if drop == 'rederive' and int(getattr(cfg, 'inf_jmem_write_hyst', 0)):   # kra54 HYSTERESIS: dropped only after failing at two successive sleeps (kra53: right writes churned)
            PJ = model.__dict__.setdefault('_jw_pend', {})
            if PJ.get(key) != vis - 1: PJ[key] = vis; hj = model.__dict__.setdefault('_jw_held', [0]); hj[0] += 1; drop = None
            else: PJ.pop(key, None)
        if drop is None: kp.append(C0 == (tr7(a[3]) if a else None)); continue   # 25 Nov: judged against the truth after an edit read (kra57 v3 GPU visit 5: KEPT right 0.81 = the 3 right overrides judged against the OLD truth)
        RW.pop((key[0], A, C0), None); WRm.pop((key[0], A, C0), None); BR.pop((key[0], A, C0), None); JW.pop(key, None); OVR.pop(key, None)
        tru = tr7(a[3]) if a else None; (rc if drop == 'contra' else rt).append(C0 == tru)
    RD = model.__dict__.get('_jm_read') or {}; eo_blk = collections.Counter(); eo_ngc = [0]
    for key, (c, b, g, x) in ans.items():                                     # kra57 EDIT OVERRIDE: where the memory REPLACED the subject's first-hop premise (an edit
        if key not in EDJ: continue                                           #   was read; gate = the looser quantile gate thp (at th only 1 of ~10 NEW passed)
        r9 = 'parent' if x[2] == 'grand' else 'mgr'
        if not np.isfinite(th): eo_blk['precision gate closed'] += 1; continue  # 25 Nov GPU v2, local v2: before th opens thp is ~0.01; the untrained memory wrote 1 and 6 overrides, all wrong
        if not c or c in (x[3], b): eo_blk['no answer'] += 1; continue
        if g < thp and not int(getattr(cfg, 'inf_jmem_edit_nogate', 0)): eo_blk['below the gate'] += 1; continue   # kra60: grounded answers need no gate (both hops
        if RD.get((r9, x[3])) != b: eo_blk['first hop is not the edited premise it read'] += 1; continue   # kra57 v3 GROUNDING (25 Nov GPU visit 3: 4 overrides, all wrong;
        if (r9, b) not in RD: eo_blk["new bridge's own premise not read yet"] += 1; continue              #   most edits are read before the new parent's family): both hops
        if RD[(r9, b)] != c: eo_blk["second hop disagrees with the bridge's read premise"] += 1; continue  #   must match premises the memory READ
        if g < thp: eo_ngc[0] += 1                                                                 #   kra58 GPU visit 4: 8 grounded-but-below-gate subjects abstained)
                                                                               #   precision gate th is open (25 Nov GPU v2, local v2: before that thp is ~0.01 and the untrained memory wrote 1 and 6 overrides, all wrong), its gated two-hop answer is written WITHOUT the trunk's agreement (the trunk
        if JW.get(key, (None,))[0] == c: OVR[key] = c; continue               #   still follows the old premise) and RETRACTS contradicting route J / step writes
        for (q_, a_, c_) in [t for t in list(RW) if t[0] == key[0] and t[1] == key[1] and t[2] != c]:   #   (kra56 GPU visit 4: combined NEW 0.00, stale 0.38 from route J;
            RW.pop((q_, a_, c_), None); WRm.pop((q_, a_, c_), None); BR.pop((q_, a_, c_), None)          #   the memory's gated answers NEW 0.19 / 0.60, stale 0.00)
            if SSW.get(key, (None,))[0] == c_: SSW.pop(key, None); eo_ss.append(c_ == tr7(x))
            else: eo_rj.append(c_ == tr7(x))
        SSW.pop(key, None) if SSW.get(key, (c,))[0] != c else None
        if key in JW: RW.pop((key[0], key[1], JW[key][0]), None)
        RW[(key[0], key[1], c)] = vis; BR[(key[0], key[1], c)] = b; JW[key] = (c, vis, float(g)); OVR[key] = c; eo_new.append(c == tr7(x))
    ER = int(getattr(cfg, 'inf_jmem_edit_retract', 0)) and EO; er_rt, er_ab = [], 0
    if ER:                                                                     # 25 Nov kra58 RETRACT ON EDIT: a subject whose first-hop premise the memory REPLACED has no
        tv9 = lambda A: (MVt[A]['C_new'] if A in MVt and MVt[A]['d2'] in seen else None)   #   grounded composite until the override writes one: every written composite of it
        H1R9 = int(getattr(cfg, 'inf_kmc_rulej_hop1_rec', 0)); er_kp = []
        for key in list(EDJ):                                                  #   (route J, step, JEPA) is RETRACTED and the key ABSTAINS (route J's write and re-derive
            if OVR.get(key) is not None: continue                              #   loops skip it); kra57 v3 GPU visit 3: the override blocked its guesses, but route J's
            r9e = 'parent' if key[0] == rw['grand'] else 'mgr'; rdb = RD.get((r9e, key[1]))   #   stale composites stayed (stale 0.50)
            for t in [t for t in list(RW) if t[0] == key[0] and t[1] == key[1]]:
                if H1R9 and rdb is not None and BR.get(t) == rdb and RD.get((r9e, rdb)) == t[2]:   # 25 Nov kra61: a composite through the EDITED premise the memory read, whose second
                    OVR[key] = t[2]; er_kp.append(t[2] == tv9(key[1])); break          #   hop matches the bridge's read premise, is GROUNDED: kept, and the subject answers again
                RW.pop(t, None); WRm.pop(t, None); BR.pop(t, None); er_rt.append(t[2] == tv9(key[1]))
            if OVR.get(key) is not None: continue
            SSW.pop(key, None); JW.pop(key, None)
            if key not in OVR: er_ab += 1
            OVR[key] = None
    GW = int(getattr(cfg, 'inf_jmem_write_grounded', 0)); gw_ok = []
    for key, (c, b, g, x) in ans.items():
        if key in JW or key in SSW or not c or c in (x[3], b): continue
        if ER and key in OVR and OVR[key] is None: continue                   # kra58: abstaining until a grounded override
        r9 = 'parent' if x[2] == 'grand' else 'mgr'                          # 25 Nov kra59 GROUNDED WRITES: an answer whose two hops match premises the memory READ is
        grd = bool(GW) and np.isfinite(th) and RD.get((r9, x[3])) == b and RD.get((r9, b)) == c   #   written at the looser gate thp WITHOUT the trunk's agreement (kra56 GPU: the
        if g < (thp if grd else th): continue                                  #   agreement rule kept JEPA at ~3% of writes while it answers ~25% at 0.82-0.86; kra57 v3 GPU
        if not grd and int(getattr(cfg, 'inf_jmem_write_agree', 0)) and SSL.get(key) != c: continue   #   visit 5: the precision gate fell to 0.003 and its share fell to 0.74 right)
        rj = RJ9.get((x[3], rw[x[2]]))
        if rj and rj[1]: continue                                              # route J answers this one itself
        RW[(key[0], key[1], c)] = vis; BR[(key[0], key[1], c)] = b; JW[key] = (c, vis, float(g)); nw += 1; okn.append(c == tr7(x))
        if grd: gw_ok.append(c == tr7(x))
    print(f"  [JEPA MEMORY WRITES | visit {vis + 1}] MODEL CHANGE: its gated two-hop answers written where route J and step by step abstained: NEW {nw} (right {pr9(okn)})" + (f", of them GROUNDED in premises it read, without the trunk's agreement (kra59) {len(gw_ok)} (right {pr9(gw_ok)})" if GW else "") + " | "
          f"earlier JEPA writes: KEPT {len(kp)} (right {pr9(kp)}), RETRACTED on re-derivation {len(rt)} (were right {pr9(rt)}), RETRACTED on route J's contradiction {len(rc)} (were right {pr9(rc)}) | "
          f"{len(JW)} JEPA writes now" + (f" | held by hysteresis {model.__dict__.get('_jw_held', [0])[0]} in all" if int(getattr(cfg, 'inf_jmem_write_hyst', 0)) else "") + (" | written only where the trunk's step by step agrees" if int(getattr(cfg, 'inf_jmem_write_agree', 0)) else "") + (f" || EDIT OVERRIDE (kra57, gate t = {thp if np.isfinite(th) else float('inf'):.3f}): {len(EDJ)} (composite, subject) with an edited first-hop premise; written without the trunk's agreement NEW {len(eo_new)} (right {pr9(eo_new)}) | "
                                                                  f"contradicting route J writes RETRACTED {len(eo_rj)} (were right {pr9(eo_rj)}) | step writes RETRACTED {len(eo_ss)} (were right {pr9(eo_ss)}) | {sum(v_ is not None for v_ in OVR.values())} override writes now" + (f", {eo_ngc[0]} of the new ones below the gate, grounded (kra60)" if int(getattr(cfg, 'inf_jmem_edit_nogate', 0)) else "")
                                                                  + (" | not written: " + ', '.join(f"{k_} {v_}" for k_, v_ in eo_blk.most_common()) if eo_blk else "")
                                                                  + (f" || RETRACT ON EDIT (kra58): {len(er_rt)} written composites of edited subjects RETRACTED (were right {pr9(er_rt)}), {er_ab} newly abstaining, {sum(v_ is None for v_ in OVR.values())} abstaining now" + (f" | route J composites through the read edit KEPT (kra61) {len(er_kp)} (right {pr9(er_kp)})" if H1R9 else "") if ER else "") if EO else ""), flush=True)


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
        SRC9 = str(getattr(cfg, 'inf_jmem_src', 'planted') or 'planted'); srcl9 = ''
        if SRC9 in ('reader', 'reader_recent'):                                 # 27 Nov kra62d: the memory learns from what the READER RECORDED (the last sleep's record of the
            ph9 = {str(rw['parent']).strip(): 'parent', str(rw['mgr']).strip(): 'mgr'}   #   rule world's parent / manager facts), not the harness's planted table; 'reader_recent'
            R9 = [e for e in (model.__dict__.get('_rec_fs') or []) if str(e.get('r', '')).strip() in ph9 and (not seen or int(e['d']) in seen)]   #   keeps, per subject
            if SRC9 == 'reader_recent':                                         #   and relation, the record from the most recently visited domain (route J's recency rule);
                vo9 = {int(k_): i_ for i_, k_ in enumerate((model.__dict__.get('_t2s') or {}).keys())}; lt9 = {}   #   'reader' keeps every record (old and new values)
                for e in R9:
                    k9 = (e['A'], str(e['r']).strip()); o9 = vo9.get(int(e['d']), -1)
                    if k9 not in lt9 or o9 > lt9[k9][0]: lt9[k9] = (o9, e)
                R9 = [v_[1] for v_ in lt9.values()]
            rwp = [(int(e['d']), '', ph9[str(e['r']).strip()], e['A'], e['ans']) for e in R9]
            if int(getattr(cfg, 'inf_jmem_edit_detect', 0)):                     # 28 Nov kra62f EDIT DETECTION FROM READING: a (subject, relation) the reader recorded with two
                vo8 = {int(k_): i_ for i_, k_ in enumerate((model.__dict__.get('_t2s') or {}).keys())}   #   different answers in different domains is EDITED, the answer from
                byk8 = {}                                                         #   the most recently visited domain current; feeds the edit override and retraction
                if int(getattr(cfg, 'inf_jmem_edit_detect', 0)) >= 2:             # 28 Nov kra62g FIX (kra62f GPU visit 3: 0 of 28 moves detected): the reader's record list
                    byk8 = _store_records(model, cfg, ph9, seen)                  #   keeps ONE object per subject and relation (inf_nm_update_first); read the store's evidence record
                else:
                  for e in [e for e in (model.__dict__.get('_rec_fs') or []) if str(e.get('r', '')).strip() in ph9 and (not seen or int(e['d']) in seen)]:
                    byk8.setdefault((ph9[str(e['r']).strip()], e['A']), []).append((vo8.get(int(e['d']), -1), e['ans']))
                EDJ8 = model.__dict__.setdefault('_jm_edited', {}); new8 = 0; cur8 = {}
                for (k_, a_), L_ in byk8.items():
                    L_.sort(); cur8[(k_, a_)] = L_[-1][1]
                    if len({x[1] for x in L_}) > 1 and (rw['grand' if k_ == 'parent' else 'skip'], a_) not in EDJ8:
                        EDJ8[(rw['grand' if k_ == 'parent' else 'skip'], a_)] = vis; new8 += 1
                model.__dict__['_jm_cur'] = cur8
                MVt = {l_ for l_, mv_ in (W.get('moves') or {}).items() if mv_['d2'] in seen}
                det8 = {a_ for (c_, a_) in EDJ8}
                srcl9_ed = f" | EDITS DETECTED from the reader's records: {len(EDJ8)} subjects ({new8} new this sleep), of them truly moved {len(det8 & MVt)} (precision {len(det8 & MVt) / max(1, len(det8)):.2f}); truly moved and read {len(MVt)}, detected {len(det8 & MVt) / max(1, len(MVt)):.2f}"
                model.__dict__['_jm_edl'] = srcl9_ed
                if int(getattr(cfg, 'inf_jmem_delta', 0)):                        # 28 Nov kra62g DELTA-RULE EDIT MEMORY (readout; local test 11m on a byte stream: edits
                    try: _delta_edit_readout(model, cfg, vis, W, seen, rw, byk8)  #   new 0.96 / stale 0.03, written once in stream order)
                    except Exception as e8: print(f"  [DELTA-RULE EDIT MEMORY | visit {vis + 1}] unavailable ({type(e8).__name__}: {str(e8)[:160]})", flush=True)
            par9, mgr9 = W.get('parent', {}), W.get('mgr', {}); MV9 = {l_: mv_ for l_, mv_ in (W.get('moves') or {}).items() if mv_['d2'] in seen}
            tru9 = lambda k_, a_: (MV9[a_]['new'] if a_ in MV9 and MV9[a_]['kind'] == k_ else (par9 if k_ == 'parent' else mgr9).get(a_))
            srcl9 = f" | SOURCE: the READER's records ({SRC9}): {len(rwp)} premises, right by the rule world's current truth {np.mean([tru9(k_, a_) == b_ for (_, _, k_, a_, b_) in rwp]) if rwp else float('nan'):.2f}, planted premises of the domains seen {sum(1 for x in W.get('planted', []) if x[2] in ('parent', 'mgr') and (not seen or int(x[0]) in seen))}"
        if int(getattr(cfg, 'inf_jmem_moves', 0)) and SRC9 == 'planted':        # 25 Nov kra56: an EDITED premise (moved leaf) once its domain is read REPLACES the old one (kra62d: never with the reader's records)
            MVj = W.get('moves') or {}                                          #   (kra55 local: the moves live in W['moves'], not W['planted'], so the JEPA memory never
            mvr = {(mv['kind'], l): mv for l, mv in MVj.items() if mv['d2'] in seen}   #   saw an edit - gated answers through edited premises NEW 0.00, stale 0.39-0.57)
            rwp = [x for x in rwp if (x[2], x[3]) not in mvr] + [(mv['d2'], mv['sent'], mv['kind'], l, mv['new']) for (k, l), mv in mvr.items()]
            model.__dict__['_jm_moves_used'] = len(mvr)
            EDJ = model.__dict__.setdefault('_jm_edited', {})                   # 25 Nov kra57: (composite, subject) whose first-hop premise the memory replaced
            for (k_, l_), mv_ in mvr.items(): EDJ.setdefault((rw['grand' if k_ == 'parent' else 'skip'], l_), vis)
        model.__dict__['_jm_read'] = {(k, a): b for (d, s_, k, a, b) in rwp}
        if SRC9 != 'planted' and model.__dict__.get('_jm_cur'): model.__dict__['_jm_read'].update(model.__dict__['_jm_cur'])   # kra62f: the newest value read
        _ = 0  # 25 Nov kra57 v3: the premises the memory READ (after edits), for the override's grounding
        nd9 = ''
        DG9 = int(getattr(cfg, 'inf_jmem_dream_gate', 0)) and getattr(J, 'th1', None) is not None and np.isfinite(J.th1)   # 24 Nov kra53: dream only the recalls that pass its own precision gate
        if int(getattr(cfg, 'inf_jmem_dream', 0)) and J.steps > 0 and (DG9 or (not int(getattr(cfg, 'inf_jmem_dream_gate', 0)) and getattr(J, 'last_r1', 0.0) >= float(getattr(cfg, 'inf_jmem_dream_min', 0.9)))):   # dreams only once its own recall is reliable             # 24 Nov kra52 DREAM REPLAY: premises of EARLIER domains are replayed from the memory's
            cur9 = int(list((model.__dict__.get('_t2s') or {0: 0}).keys())[-1])  #   OWN recall, not the read text (jmem_sim: D0 0.98 / D1 0.99 / D2 1.00 after 3 sleeps)
            old9 = [x for x in rwp if int(x[0]) != cur9]; new9 = [x for x in rwp if int(x[0]) == cur9]; dr9 = []; ok9 = []; kt9 = []
            if SRC9 != 'planted' and int(getattr(cfg, 'inf_jmem_dream_seen', 0)):   # 28 Nov kra62f: dream only premises the memory has ALREADY trained on; a premise first
                TR9 = model.__dict__.setdefault('_jm_trained', set())               #   seen now (the reader's records arrive a sleep late, so an edit's new value comes from a domain
                new9 = new9 + [x for x in old9 if (x[2], x[3], x[4]) not in TR9]    #   no longer current) is trained from its read text, not replaced by a dream of the memory's
                old9 = [x for x in old9 if (x[2], x[3], x[4]) in TR9]              #   OLD recall (kra62d GPU visits 3-5: the memory's gated answers stale 0.30-0.67)
            for k in ('parent', 'mgr'):
                Lk = [x for x in old9 if x[2] == k]
                if Lk:
                    o9, g9 = J.hop([x[3] for x in Lk], rw[k])
                    for x, c, gg in zip(Lk, o9, g9):
                        if DG9 and gg < J.th1: kt9.append((f"{x[3]} {rw[k]} {x[4]}.", (x[3], x[4]))); continue   #   (kra53) below its precision gate: the READ premise, not a dream
                        if c and c != x[3]: dr9.append((f"{x[3]} {rw[k]} {c}.", (x[3], c))); ok9.append(c == x[4])
            items = _items_from(dr9 + kt9 + [(f"{a} {rw[k]} {b}.", (a, b)) for (d, s_, k, a, b) in new9])
            nd9 = f", {len(dr9)} of them DREAMED from its own recall of earlier domains{' (precision gate t = ' + format(J.th1, '.3f') + ': ' + str(len(old9)) + ' earlier premises, ' + str(len(kt9)) + ' below the gate replayed from the read text)' if DG9 else ''} (right {np.mean(ok9) if ok9 else float('nan'):.2f})"
        else:
            items = _items_from([(f"{a} {rw[k]} {b}.", (a, b)) for (d, s_, k, a, b) in rwp])
        from platypus.inference.realchain import _RC
        trn, tst = _RC.get('train') or [], _RC.get('test') or []
        prem = sorted({(c[0], c[1], c[2]) for c in trn + tst} | {(c[2], c[3], c[4]) for c in trn + tst})
        items += _items_from([(f"{h} {w} {t}.", (h, t)) for h, w, t in prem])
        RECW9 = float(getattr(cfg, 'inf_jmem_recency', 0) or 0)                 # 28 Nov kra62e RECENCY-WEIGHTED REPLAY: with the reader's records (old and new values both kept,
        if RECW9 > 0 and SRC9 != 'planted' and rwp:                             #   inf_jmem_src 'reader'), a premise read in a later-visited domain is replayed more often:
            vo9 = {int(k_): i_ for i_, k_ in enumerate((model.__dict__.get('_t2s') or {}).keys())}   #   weight 1 + RECW9 x (visit rank / last rank) - the memory, not the harness,
            om9 = max(1, max(vo9.values()) if vo9 else 1)                       #   has to come to prefer the newer value (the byte-stream edits tests: old records never
            sd9 = {f"{a} {rw[k]} {b}.": vo9.get(int(d), 0) for (d, s_, k, a, b) in rwp}   #   withdrawn, new answer 0.92-0.97, stale 0.02-0.08)
            ext9 = []
            for it in items:
                o9 = sd9.get(it[0])
                if o9 is not None: ext9 += [it] * int(round(RECW9 * o9 / om9))
            items += ext9
        JQ = None
        if int(getattr(cfg, 'inf_mquake', 0)) > 0:                            # 26 Nov MQuAKE PORT: the facts read (edits in place of the old values once read)
            from platypus.inference.mquake import mq_jmem_items
            if int(getattr(cfg, 'inf_mquake_jsep', 0)):                         # 27 Nov kra62c: the MQuAKE facts in their OWN JEPA memory instance (kra62 / kra62b: in the
                JQ = model.__dict__.get('_jmem_mq')                             #   shared memory they diluted the rule world: one hop on read premises 0.92 -> 0.51,
                if JQ is None: JQ = JMem(cfg, dev); model.__dict__['_jmem_mq'] = JQ   #   gated edit coverage 0.76 -> 0.05); the rule-world memory is then kra61's
                if int(getattr(cfg, 'inf_mquake_nl', 0)):                       # 28 Nov kra62f: cloze sentences, old values kept, recency-weighted (no gold replacement)
                    from platypus.inference.mquake import mq_jmem_items_nl
                    voq = {int(k_): i_ for i_, k_ in enumerate((model.__dict__.get('_t2s') or {}).keys())}
                    iq = mq_jmem_items_nl(cfg, seen, voq, float(getattr(cfg, 'inf_jmem_recency', 0) or 0))
                else: iq = _items_from(mq_jmem_items(cfg, seen))
                lq = JQ.train(iq, int(getattr(cfg, 'inf_mquake_jsteps', 0)) or int(getattr(cfg, 'inf_jmem_steps', 1500))) if iq else float('nan')
                print(f"  [MQuAKE PORT, OWN JEPA MEMORY | visit {vis + 1}] {len(iq)} MQuAKE fact sentences read so far, in a separate keyed-memory JEPA ({len(JQ.keys)} slots, {JQ.steps} steps in all, loss {lq:.3f}); the rule-world memory does not see them", flush=True)
            else:
                items += _items_from(mq_jmem_items(cfg, seen))
        if int(getattr(cfg, 'inf_jmem_src_debug', 0)) >= 2:                      # 28 Nov kra62d DEBUG 2: the exact training items and steps, for an offline replay
            import pickle, os as _os
            _dp = _os.environ.get('PLATYPUS_JMEM_DUMP', '/tmp/jmem_items'); _os.makedirs(_dp, exist_ok=True)
            pickle.dump({'items': items, 'steps': int(getattr(cfg, 'inf_jmem_steps', 1500)), 'J_steps_before': J.steps, 'n_keys_before': len(J.keys), 'cfg': {k_: getattr(cfg, k_) for k_ in dir(cfg) if k_.startswith('inf_jmem')}}, open(f"{_dp}/visit{vis + 1}.pkl", 'wb'))
        MIN9 = int(getattr(cfg, 'inf_jmem_min_items', 0)); MINR9 = int(getattr(cfg, 'inf_jmem_min_rule', 0))   # kra62e: also wait for MINR9 RULE-WORLD premises (kra62d GPU
        if len(items) < MIN9 or len(rwp) < MINR9:                             #   visit 2: 589 real-text premises and none from the reader yet). 28 Nov kra62d: no training until enough premises are read (local replay: 3,000 steps on the
                                                                              #   17 real-text premises of visit 2, before the reader's first rule-world records, left the
            ls = float('nan')                                                 #   memory recalling 0.07 of 224 premises at visit 3; the same items fresh: 0.50)
            print(f"  [JEPA MEMORY (parallel) | visit {vis + 1}] {len(items)} premise sentences read so far ({len(rwp)} of the rule world), below inf_jmem_min_items {MIN9} / inf_jmem_min_rule {MINR9}: not trained this sleep", flush=True)
        else:
            ls = J.train(items, int(getattr(cfg, 'inf_jmem_steps', 1500)))
            if SRC9 != 'planted': model.__dict__.setdefault('_jm_trained', set()).update((x[2], x[3], x[4]) for x in rwp)
        # RULE WORLD: one hop on read premises, two hops on planted / held-back / pure composites, the gate on never-read families
        par, mgr = W.get('parent', {}), W.get('mgr', {}); up = {'grand': (par, 'parent'), 'skip': (mgr, 'mgr')}
        s1 = rwp[:300]
        r1 = []; p1 = []
        for k in ('parent', 'mgr'):
            L = [x for x in s1 if x[2] == k]
            if L: o, g1_ = J.hop([x[3] for x in L], rw[k]); r1 += [y == x[4] for y, x in zip(o, L)]; p1 += [(g, y == x[4]) for y, x, g in zip(o, L, g1_)]
        J.last_r1 = float(np.mean(r1)) if r1 else 0.0
        if SRC9 != 'planted' and int(getattr(cfg, 'inf_jmem_src_debug', 0)) and s1:   # 28 Nov kra62d DEBUG (readout only): what the memory trained on and what it answers
            dbg9 = [x for x in s1 if x[2] == 'parent'][:4] + [x for x in s1 if x[2] == 'mgr'][:2]
            oo9 = {k_: J.hop([x[3] for x in dbg9 if x[2] == k_], rw[k_])[0] for k_ in ('parent', 'mgr')}
            it9 = [it for it in items if any(x[3] in it[0] for x in dbg9)][:4]
            print(f"  [JEPA MEMORY SOURCE DEBUG | visit {vis + 1}] items (sentence, masked spans): {[(t_, sp_) for t_, sp_ in it9]!r}"[:900], flush=True)
            print(f"  [JEPA MEMORY SOURCE DEBUG | visit {vis + 1}] premises -> answer: " + '; '.join(f"{x[3]!r} {rw[x[2]]!r} -> {o_!r} (read {x[4]!r})" for k_ in ('parent', 'mgr') for x, o_ in zip([y for y in dbg9 if y[2] == k_], oo9[k_]))[:900], flush=True)
        J.th1 = _pcal(p1, float(getattr(cfg, 'inf_jmem_dprec', 0.95)), 20)       # kra53: its DREAM gate, precision-targeted on its recalls of what it read (inf = no dreams)                          # kra52: the memory's own one-hop recall on what it read (a self-test on stored text)
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
        J.thq = th; thw = _pcal([(g, c) for (b, c, g) in pl], float(getattr(cfg, 'inf_jmem_wprec', 0.9)), 10)   # kra53: the WRITE / ANSWER gate, precision-targeted on the planted composites
        nr = [(d, s_, k, a, b) for (d, s_, k, a, b) in W.get('planted', []) if k in ('parent', 'mgr') and seen and int(d) not in seen][:200]
        mn = []
        for k in ('parent', 'mgr'):
            Lk = [x for x in nr if x[2] == k]
            if Lk: _, g = J.hop([x[3] for x in Lk], rw[k]); mn += g
        mr = [g for (b, c, g) in pl if c]
        auc = np.mean([[float(a > b) + 0.5 * (a == b) for b in mn] for a in mr]) if mr and mn else float('nan')
        f = lambda R: (f"{len(R)}: bridge {np.mean([b for b, c, g in R]):.2f}, C {np.mean([c for b, c, g in R]):.2f}, GATED answered {np.mean([g >= th for b, c, g in R]):.2f} "
                       f"right {np.mean([c for b, c, g in R if g >= th]) if any(g >= th for b, c, g in R) else float('nan'):.2f}") if R else "0"
        print(f"  [JEPA MEMORY (parallel) | visit {vis + 1}] keyed-memory JEPA beside the trunk, trained this sleep on {len(items)} premise sentences read so far{nd9}" + (f" (EDITED premises read and used in place of the old: {model.__dict__.get('_jm_moves_used', 0)})" if int(getattr(cfg, 'inf_jmem_moves', 0)) else "") + f" "
              f"({len(J.keys)} slots, {J.steps} steps in all, loss {ls:.3f}; readout only) || RULE WORLD: one hop on read premises {np.mean(r1) if r1 else float('nan'):.2f} (n {len(r1)}) | "
              f"TWO HOPS by its own hops: PLANTED {f(pl)} || HELD BACK {f(hb)} || PURE FAMILIES {f(hu)} | never-read heads vs right planted, margin AUC {auc:.2f} ({time.time() - t0:.0f}s)" + srcl9 + (model.__dict__.get('_jm_edl', '') if SRC9 != 'planted' else ''), flush=True)
        if int(getattr(cfg, 'inf_jmem_answer', 0)) or int(getattr(cfg, 'inf_jmem_write', 0)):
            _jmem_answer(model, cfg, vis, W, seen, rw, J, thw, up, pure)
        # REAL TEXT: the frozen TEST chains by its own two hops (premises read, chains never)
        t2 = None
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
        if int(getattr(cfg, 'inf_mquake', 0)) > 0:                            # 26 Nov MQuAKE PORT: the memory's own hops on the cases read
            from platypus.inference.mquake import mq_jmem_readout
            if int(getattr(cfg, 'inf_mquake_nl', 0)) and JQ is not None:
                from platypus.inference.mquake import mq_jmem_readout_nl
                mq_jmem_readout_nl(JQ, cfg, vis, seen)                              # 28 Nov kra62f: hops by cloze wording, old values kept
            else:
                mq_jmem_readout(JQ if JQ is not None else J, cfg, vis, seen, {} if JQ is not None else {'taught real-text chains': t2, 'planted composites (write gate)': thw})   # kra62c: its own memory's gates are its own
    except Exception as e:
        print(f"  [JEPA MEMORY (parallel) | visit {vis + 1}] unavailable ({type(e).__name__}: {str(e)[:160]})", flush=True)


def _dkey(s, D=8192):
    """sparse unit key of a string: hashed byte 3- to 5-grams, L2-normalised (strings that share n-grams share features, so writes interfere)"""
    import zlib
    b = (' ' + s + ' ').encode('latin-1', 'replace'); ix = {}
    for n in (3, 4, 5):
        for i in range(len(b) - n + 1): h = zlib.crc32(b[i:i + n]) % D; ix[h] = ix.get(h, 0.0) + 1.0
    h = zlib.crc32(b'\x00' + b) % D; ix[h] = ix.get(h, 0.0) + 2.0 * max(1.0, float(np.sqrt(sum(v * v for v in ix.values()))))   # the whole string: dominant, so
    k = np.array(sorted(ix)); v = np.array([ix[j] for j in k], dtype=np.float32); return k, v / max(1e-9, float(np.linalg.norm(v)))


def _delta_edit_readout(model, cfg, vis, W, seen, rw, byk8):
    """28 Nov kra62g (inf_jmem_delta; READOUT ONLY): a delta-rule associative memory (Widrow-Hoff / DeltaNet: W += beta (v - W k) k^T) written ONCE, in
    reading order, with every (subject, relation) -> object record of the rule world that the store holds (old and new values both; the order is the
    first sighting). A later write to the same key overwrites the earlier answer; similar keys interfere. Read: one hop argmax W k, two hops through
    its own bridge. Scored on the true moves read (edits): NEW vs OLD (stale) composite, and one hop on subjects never moved (control)."""
    t0 = time.time(); D = 8192
    wr = sorted((o, k, A, v) for (k, A), L in byk8.items() for (o, v) in L)
    if not wr: print(f"  [DELTA-RULE EDIT MEMORY | visit {vis + 1}] no rule-world records yet", flush=True); return
    names = sorted({v for _, _, _, v in wr} | {A for _, _, A, _ in wr}); NI = {n: i for i, n in enumerate(names)}
    MM = {k: np.zeros((len(names), D), dtype=np.float32) for k in ('parent', 'mgr')}; KC = {}   # one memory per relation, keyed by the subject's bytes
    def key(A):
        if A not in KC: KC[A] = _dkey(A, D)
        return KC[A]
    for _, k, A, v in wr:
        M = MM[k]; ix, kv = key(A); pred = M[:, ix] @ kv; tgt = np.zeros(len(names), dtype=np.float32); tgt[NI[v]] = 1.0
        M[:, ix] += np.outer(tgt - pred, kv)
    def hop(k, A):
        if A not in NI and A not in KC and not A: return None
        ix, kv = key(A); s_ = MM[k][:, ix] @ kv; j = int(np.argmax(s_)); return names[j] if s_[j] > 0.2 else None
    MV = {l: mv for l, mv in (W.get('moves') or {}).items() if mv['d2'] in seen and l in NI}
    nw1, st1, nw2, st2 = [], [], [], []
    for l, mv in MV.items():
        k = mv['kind']; b = hop(k, l); nw1.append(b == mv['new']); st1.append(b == mv['old'])
        c = hop(k, b) if b else None; nw2.append(c is not None and c == mv.get('C_new')); st2.append(c is not None and c == mv.get('C_old'))
    ctl = []; par, mgr = W.get('parent', {}), W.get('mgr', {})
    for (k, A) in list(byk8)[:2000]:
        if A in (W.get('moves') or {}): continue
        t = (par if k == 'parent' else mgr).get(A)
        if t is not None: ctl.append(hop(k, A) == t)
    f = lambda L: f"{np.mean(L):.2f}" if L else 'n/a'
    print(f"  [DELTA-RULE EDIT MEMORY | visit {vis + 1}] READOUT: a delta-rule memory (W += (v - W k) k^T, one per relation, keys = the subject's hashed byte n-grams + the whole string, {D} features) written once in reading order "
          f"with the {len(wr)} rule-world records the store holds ({len(byk8)} subject-relation keys, old and new values both) || EDITS READ (n {len(MV)}): ONE HOP NEW parent {f(nw1)}, "
          f"OLD (stale) {f(st1)} | TWO HOPS NEW composite {f(nw2)}, OLD (stale) {f(st2)} || NOT MOVED (control) one hop right {f(ctl)} (n {len(ctl)}) ({time.time() - t0:.0f}s)", flush=True)


def _store_records(model, cfg, ph9, seen):
    """28 Nov kra62g: every rule-world (relation, subject) -> [(first sighting, object)] in the store's evidence record (S_['rel']), which keeps every
    object read, where the reader's record list keeps only the newest introduced (inf_nm_update_first). An object counts once seen inf_nm_min_seen
    (2) times; its domain is that of its first sighting; the object introduced most recently is current."""
    from platypus.inference.sentences import _inf_store
    from platypus.inference.namemem import _nm_isname
    S8 = _inf_store(model); it8 = S8.get('it', []); mins8 = max(1, int(getattr(cfg, 'inf_nm_min_seen', 2) or 2)); out = {}
    for (a8, r8), objs8 in S8.get('rel', {}).items():
        r8s = bytes(r8).decode('latin-1', 'replace').strip()
        if r8s not in ph9: continue
        A8 = bytes(a8).decode('latin-1', 'replace')
        for o8, ix8 in objs8.items():
            o8 = bytes(o8); ix8 = sorted(set(x for x in ix8 if x < len(it8)))
            if not _nm_isname(o8) or len(ix8) < mins8: continue
            d8 = int(it8[ix8[0]].get('d', -1))
            if seen and d8 not in seen: continue
            out.setdefault((ph9[r8s], A8), []).append((ix8[0], o8.decode('latin-1', 'replace')))
    return out
