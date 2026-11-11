"""LATENT JUDGE IN SLEEP (north-star rung 2, readout only: inf_latent_judge 1). Every sleep, the pairs the byte-level judge tests (proposals,
random pairs, and the premise pairs of committed conclusions as known-linked controls) are also scored by a LATENT judge: the cosine of
their closed-form slow-feature codes (SFA on this sleep's store: consecutive real stored sentences, 32 slowest directions, refitted each
sleep). Reported: how well each judge separates known-linked controls from random pairs (AUC), how far the latent judge agrees with the
byte judge's verdicts on proposals, a shared-rare-name baseline, and the cost per pair. The byte judge still decides what is dreamed."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import _fail


def _auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    p = pos[:, None]; n = neg[None, :]; return float((p > n).mean() + 0.5 * (p == n).mean())


def _rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or a.std() == 0 or b.std() == 0: return float('nan')
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def latent_judge_readout(model, cfg, S_, G_, T_, thr_, byte_secs):
    """G_: (a, b, source, margin (weaker side, bits), ...) as the byte judge scored them this sleep. Run under a forked random state so
    the readout cannot change training (9 Nov: without it a local run ended 0.006 BPC off its twin)"""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        st0 = np.random.get_state()
        try: _latent_judge(model, cfg, S_, G_, T_, thr_, byte_secs)
        finally: np.random.set_state(st0)


def _sfa_for_sleep(model, cfg, S_):
    """closed-form SFA on consecutive real stored sentences, fitted once per sleep and cached (shared by the readout and the first pass);
    run under a forked random state by its callers. Returns (mu, P, w) or None when the store has too few adjacent pairs"""
    from platypus.inference.selcmp import _codes, _fit_linear
    it = S_['it']; vis = int(model.__dict__.get('_visit', 0)); L = model.__dict__.setdefault('_ljudge', {})
    if L.get('vis') != vis:
        rs = np.random.RandomState(int(cfg.seed) + 5151 + vis)
        ix = [i for i in range(len(it) - 1) if it[i].get('k') == 'real' and it[i + 1].get('k') == 'real' and it[i]['d'] == it[i + 1]['d']
              and 20 <= len(it[i]['s']) <= 300 and 20 <= len(it[i + 1]['s']) <= 300]
        if len(ix) < 400: return None
        ix = [ix[int(k)] for k in rs.choice(len(ix), min(3000, len(ix)), replace=False)]
        _, sfa, _, _, _ = _fit_linear(_codes(model, [it[i]['s'] for i in ix]).cpu(), _codes(model, [it[i + 1]['s'] for i in ix]).cpu(), seed=int(cfg.seed))
        L.update(vis=vis, sfa=sfa)
    return L['sfa']


def _latent_judge(model, cfg, S_, G_, T_, thr_, byte_secs):
    from platypus.inference.selcmp import _codes
    try:
        vis = int(model.__dict__.get('_visit', 0)); t0 = time.time()
        if _sfa_for_sleep(model, cfg, S_) is None: print(f"  [LATENT JUDGE | visit {vis + 1}] skipped: too few adjacent stored pairs", flush=True); return
        mu, P, _ = model.__dict__['_ljudge']['sfa']; tfit = time.time() - t0; t1 = time.time()
        G = [g for g in G_ if g[0] >= 0 and g[1] >= 0]
        if len(G) < 10: return
        tx = sorted({g[0] for g in G} | {g[1] for g in G}); pos = {t: j for j, t in enumerate(tx)}
        Z = F.normalize((_codes(model, [T_(t) for t in tx]).cpu() - mu) @ P, dim=-1)
        sim = [float((Z[pos[g[0]]] * Z[pos[g[1]]]).sum()) for g in G]; tlat = time.time() - t1
        dfm = S_.get('df', {}); dmx = int(getattr(cfg, 'inf_df', 40))
        def rare(t):
            b = T_(t); b = b if isinstance(b, (bytes, bytearray)) else bytes(b)
            return {w for w in __import__('re').findall(rb"[A-Z][a-z]{2,}", bytes(b)) if 0 < dfm.get(w, 0) <= dmx}
        shr = [float(len(rare(g[0]) & rare(g[1])) > 0) for g in G]
        src = [g[2] for g in G]; mg = [float(g[3]) for g in G]
        ctl = [j for j, s in enumerate(src) if s == 'control+']; rnd = [j for j, s in enumerate(src) if s == 'random']; prp = [j for j, s in enumerate(src) if s not in ('random', 'control+')]
        acc = [j for j in prp if mg[j] >= thr_]; rej = [j for j in prp if mg[j] < thr_]
        pick = lambda v, J: [v[j] for j in J]
        nb = max(1, len(G)); per_byte = byte_secs / nb; per_lat = (tlat) / nb
        print(f"  [LATENT JUDGE | visit {vis + 1}] {len(G)} pairs judged by both ({len(ctl)} known-linked controls, {len(rnd)} random, {len(prp)} proposals: "
              f"{len(acc)} accepted by the byte judge) | CONTROLS vs RANDOM, AUC: latent (SFA cosine) {_auc(pick(sim, ctl), pick(sim, rnd)):.2f}, "
              f"byte judge (bits saved) {_auc(pick(mg, ctl), pick(mg, rnd)):.2f}, shared rare name {_auc(pick(shr, ctl), pick(shr, rnd)):.2f} | "
              f"PROPOSALS: latent AUC for the byte judge's accept vs reject {_auc(pick(sim, acc), pick(sim, rej)):.2f}, rank correlation with the byte margin "
              f"{_rho(pick(sim, prp), pick(mg, prp)):+.2f} | COST per pair: byte judge {per_byte * 1000:.1f} ms, latent {per_lat * 1000:.2f} ms "
              f"(+ SFA refit {tfit:.1f} s per sleep)", flush=True)
        try: _truth_and_hard(model, cfg, S_, G, T_, thr_, sim, mg, src, prp, ctl, rnd, rare, mu, P, vis)
        except Exception as e2: _fail('inference.latentjudge:truth'); print(f"  [LATENT JUDGE | HARD CONTROLS] unavailable ({type(e2).__name__}: {e2})"[:240], flush=True)
    except Exception as e:
        _fail('inference.latentjudge'); print(f"  [LATENT JUDGE] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _truth_and_hard(model, cfg, S_, G, T_, thr_, sim, mg, src, prp, ctl, rnd, rare, mu, P, vis):
    """(The TRUTH score was removed 9 Nov: the mixing judge never sees pairs of planted facts - planted links are found by the linking step - so
    it labelled nothing. The truth test for this judge on real text is Re-DocRED, inf_rdr_judge.)
    HARD CONTROLS (latent only; the byte judge is not re-run): the known-linked controls with the shared rare name replaced by 'the person'
    in the second sentence, against random pairs; and the controls against SAME-DOMAIN TOPIC PAIRS (two stored sentences of one domain sharing
    2+ lower-case content words and no shared rare name) - unlinked pairs that look alike."""
    import re
    from platypus.inference.selcmp import _codes
    def tb(t):
        b = T_(t); return bytes(b) if not isinstance(b, (bytes, bytearray)) else bytes(b)
    # (2) hard controls
    rs = np.random.RandomState(int(cfg.seed) + 5353 + vis); it = S_['it']
    def code(txts): return F.normalize((_codes(model, txts).cpu() - mu) @ P, dim=-1)
    msk = []
    for j in ctl:
        a, b = tb(G[j][0]), tb(G[j][1]); sh = rare(G[j][0]) & rare(G[j][1])
        if not sh: continue
        for w in sh: b = b.replace(w, b'the person')
        msk.append((a, b))
    real = [i for i in range(len(it)) if it[i].get('k') == 'real' and not it[i].get('gone') and 20 <= len(it[i]['s']) <= 300]
    bydom = {}
    for i in real: bydom.setdefault(it[i]['d'], []).append(i)
    cw = lambda b: {w for w in re.findall(rb"\b[a-z]{5,}\b", b)}
    topic = []
    for _ in range(4000):
        if len(topic) >= 64 or not bydom: break
        d = list(bydom)[rs.randint(len(bydom))]; L9 = bydom[d]
        if len(L9) < 2: continue
        i1, i2 = L9[rs.randint(len(L9))], L9[rs.randint(len(L9))]
        a, b = bytes(it[i1]['s']), bytes(it[i2]['s'])
        if i1 != i2 and len(cw(a) & cw(b)) >= 2 and not (rare(i1) & rare(i2)): topic.append((a, b))   # no shared rare name (names elsewhere allowed)
    if len(msk) < 2 or not rnd: print(f"  [LATENT JUDGE | HARD CONTROLS | visit {vis + 1}] skipped ({len(msk)} maskable controls)", flush=True); return
    def cs(pairs):
        if not pairs: return []
        A = code([p[0] for p in pairs]); B = code([p[1] for p in pairs]); return (A * B).sum(-1).tolist()
    s_ctl = [sim[j] for j in ctl]; s_rnd = [sim[j] for j in rnd]; s_msk = cs(msk); s_top = cs(topic)
    print(f"  [LATENT JUDGE | HARD CONTROLS | visit {vis + 1}] latent AUC: controls vs random {_auc(s_ctl, s_rnd):.2f} | NAME MASKED controls ({len(s_msk)}, "
          f"the shared name replaced by 'the person') vs random {_auc(s_msk, s_rnd):.2f} | controls vs SAME-DOMAIN TOPIC PAIRS ({len(s_top)}, 2+ shared content "
          f"words, no shared rare name) {_auc(s_ctl, s_top) if s_top else float('nan'):.2f} | masked controls vs topic pairs {_auc(s_msk, s_top) if s_top else float('nan'):.2f} | "
          f"shared rare name after masking: 0 by construction", flush=True)
