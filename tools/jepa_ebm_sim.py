"""25 Nov LECUN-STYLE INFERENCE FOR THE KEYED-MEMORY JEPA (offline, synthetic; the five-level world of tools/jepa_store.py).
Two questions, both about platypus/inference/jmem.py's memory (same encoders, slot table, InfoNCE over slots at t 0.1, EMA slots and target):
 1. LATENT-VARIABLE ENERGY (one-to-many answers). The memory predicts ONE point per query, so a query with several right answers ('<region>
    includes ______.', about 5 cities) is pulled to their average. Here the predictor outputs K points (a latent z in 1..K), trained
    winner-take-all (loss = 0.95 min over z + 0.05 mean over z, so no point dies); the energy of a candidate is its best match over z,
    E(x, y) = min_z -cos(P(x, z), slot_y) / t. K = 1 is the production memory. Readout: recall@m on one-to-many queries (m = number of right
    answers), and one-hop recall on the functional facts (must not fall).
 2. MIN-ENERGY CHAIN SEARCH ('mode 2' inference). Production chains greedily (argmax at each hop). Here each hop's log-probabilities over
    the slots form a transition table T_w (slots x slots); a beam of B bridges per hop keeps the path energies; the answer is the
    minimum-energy path (Viterbi) or the lowest marginal energy (sum over paths). Readouts: 2- and 3-hop accuracy, accuracy of the most
    confident half, and GOAL-DIRECTED VERIFICATION: given a claimed answer C, the minimum path energy from A to C, as a true/false score
    (AUC against a random wrong C and against a NEAR MISS - another region of the same country).
    Also the two together on a one-to-many chain: '<country> includes ______' twice (country -> regions -> cities, about 15 right answers).
Memory strength varies: readouts after 500, 1500 and 3000 training steps (the GPU memory is mid-way: one hop 0.65-0.86).
Usage: LS_SEED=0 python3 tools/jepa_ebm_sim.py   (env EB_STEPS '500,1500,3000', EB_K '1,8', EB_THREADS 2) - EB_WARM 300 (learning-rate warm-up steps, as production inf_jmem_warm; 0 = the 25 Nov runs)"""
import os, sys, types, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
torch.set_num_threads(int(os.environ.get('EB_THREADS', 2))); T0 = time.time(); P = lambda s: print(s, flush=True)
src = open(os.path.join(ROOT, 'tools', 'jepa_store.py')).read(); cut = src.index('PRE1 = sorted(')
g = {'__name__': 'jm'}; exec(compile(src[:cut], 'jepa_store_world', 'exec'), g)
W, FACTS, CIT, REG, CTY, PEO = g['W'], g['FACTS'], g['CIT'], g['REG'], g['CTY'], g['TRP'] + g['HEP']
from platypus.inference.jmem import JMem, _items_from, _tens, _span_mean


class LVJMem(JMem):
    def __init__(s, cfg, dev, K):
        super().__init__(cfg, dev); s.K = K
        if K > 1:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(17); d = s.d
                s.pred = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, d * K)).to(dev)
            s.opt = torch.optim.AdamW(list(s.ctx.parameters()) + list(s.pred.parameters()), lr=s.lr0, weight_decay=0.01)

    def _P(s, h): return F.normalize(s.pred(h).view(h.shape[0], s.K, s.d), dim=-1)

    def train(s, items, steps, bs=48, eps=0.05):
        for (t, sp) in items:
            for (_, _, k) in sp: s._slot(k)
        s.ctx.train(); s.pred.train(); ls = []
        with torch.enable_grad():
            for _ in range(steps):
                b = [items[i] for i in s.rs.randint(0, len(items), bs)]; Xs, sp, ix, full = [], [], [], []
                for t, spans in b:
                    a_, b_, k = spans[s.rs.randint(len(spans))]
                    Xs.append((t[:a_] + '\x01' * (b_ - a_) + t[b_:]).encode('latin-1', 'replace')); sp.append((a_, b_)); ix.append(s.ki[k]); full.append(t.encode('latin-1', 'replace'))
                with torch.no_grad(): z = F.normalize(_span_mean(s.tgt(_tens(full, s.dev)), sp), dim=-1)
                p = s._P(_span_mean(s.ctx(_tens(Xs, s.dev)), sp)); it = torch.tensor(ix, device=s.dev)
                if s.warm:                                                     # 26 Nov: production's warm-up (JMem.train; 25 Nov seeds 0-1 ran without it and never learned)
                    for g_ in s.opt.param_groups: g_['lr'] = s.lr0 * min(1.0, (s.steps + 1) / s.warm)
                lg =torch.einsum('bkd,sd->bks', p, s.SL.clone()) / 0.1
                ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), it.repeat_interleave(s.K), reduction='none').view(-1, s.K)
                l = ((1 - eps) * ce.min(1).values + eps * ce.mean(1)).mean() if s.K > 1 else ce[:, 0].mean()
                s.opt.zero_grad(); l.backward(); s.opt.step(); ls.append(float(l.detach()))
                with torch.no_grad():
                    for k_, v in zip(ix, z): s.SL[k_] = F.normalize(0.9 * s.SL[k_] + 0.1 * v, dim=0)
                    for pt, pc in zip(s.tgt.parameters(), s.ctx.parameters()): pt.mul_(0.995).add_(pc.detach(), alpha=0.005)
                s.steps += 1
        return float(np.mean(ls[-50:]))

    @torch.no_grad()
    def logp(s, heads, wording):
        """(len(heads), slots) log-probabilities; energy of a slot = min over the latent z of -cos / 0.1"""
        s.ctx.eval(); s.pred.eval(); out = []
        for i in range(0, len(heads), 256):
            qs, sp = [], []
            for h in heads[i:i + 256]:
                t = f"{h} {wording} " + '\x01' * 6 + '.'; a_ = len(t) - 7; qs.append(t.encode('latin-1', 'replace')); sp.append((a_, a_ + 6))
            v = s._P(_span_mean(s.ctx(_tens(qs, s.dev)), sp)); out.append(F.log_softmax(torch.einsum('bkd,sd->bks', v, s.SL).max(1).values / 0.1, -1))
        return torch.cat(out)


def auc(pos, neg):
    x = np.r_[pos, neg]; r = x.argsort().argsort().astype(float) + 1
    for v in np.unique(x):                                                     # average ranks for ties
        m = x == v; r[m] = r[m].mean()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / max(1, len(pos) * len(neg))


def chain(T_list, start, B):
    """start: (N,) slot ids; T_list: per-hop (S,S) log-prob tables. Returns greedy (ans, conf), viterbi scores (N,S), marginal scores (N,S)"""
    g_ = start.clone(); conf = torch.full((len(start),), 1e9)
    for T in T_list:
        row = T[g_]; t2 = row.topk(2, -1); conf = torch.minimum(conf, t2.values[:, 0] - t2.values[:, 1]); g_ = t2.indices[:, 0]
    res = {}
    for mode in ('vit', 'mar'):
        v = T_list[0][start]
        for T in T_list[1:]:
            vb, ib = v.topk(B, -1); cand = vb[:, :, None] + T[ib]              # (N, B, S)
            v = cand.max(1).values if mode == 'vit' else torch.logsumexp(cand, 1)
        res[mode] = v
    return (g_, conf), res['vit'], res['mar']


def top_half(conf, ok):
    o = np.argsort(-np.asarray(conf)); return float(np.mean(np.asarray(ok)[o[:max(1, len(o) // 2)]]))


SENT = [(W[r][0].format(a=a, b=b), (a, b)) for r in ('lives', 'city_in', 'region_of', 'country_in') for a, b in FACTS[r]]
SENT += [(f"{b} includes {a}.", (a, b)) for r in ('city_in', 'region_of') for a, b in FACTS[r]]     # one-to-many inverse facts
reg_of = {c: REG[r][0] for c, r in CIT}; cty_of = {r: CTY[k][0] for r, k in REG}
R_C = {}; [R_C.setdefault(REG[r][0], set()).add(c) for c, r in CIT]
K_R = {}; [K_R.setdefault(CTY[k][0], set()).add(r) for r, k in REG]
K_C = {k: set().union(*[R_C.get(r, set()) for r in rs]) for k, rs in K_R.items()}
PE = [(a, CIT[ci][0]) for a, ci in PEO]
STEPS = [int(x) for x in os.environ.get('EB_STEPS', '500,1500,3000').split(',')]; KS = [int(x) for x in os.environ.get('EB_K', '1,8').split(',')]
P(f"JEPA ENERGY-INFERENCE TEST (world seed {os.environ.get('LS_SEED', 0)}): {len(PE)} people, {len(CIT)} cities, {len(REG)} regions, {len(CTY)} countries; "
  f"{len(SENT)} sentences ({len(SENT) - sum(len(FACTS[r]) for r in ('lives', 'city_in', 'region_of', 'country_in'))} one-to-many 'includes'); latent K {KS}; steps {STEPS}")
items = _items_from(SENT); rs9 = np.random.RandomState(3)
for K in KS:
    cfg = types.SimpleNamespace(inf_jmem_dim=192, inf_jmem_layers=3, inf_jmem_lr=1e-3, inf_jmem_warm=int(os.environ.get('EB_WARM', 300)))
    J = LVJMem(cfg, torch.device('cpu'), K); done = 0
    for st in STEPS:
        ls = J.train(items, st - done); done = st; ki = J.ki; keys = J.keys
        T = {w: J.logp(keys, w) for w in ('lives in', 'lies in', 'is part of', 'includes')}
        a1 = lambda w, pairs: float(np.mean([keys[int(T[w][ki[a]].argmax())] == b for a, b in pairs]))
        def rec(Q):                                                            # recall@m on one-to-many queries {head: set of right answers}
            return float(np.mean([len({keys[j] for j in T['includes'][ki[h]].topk(len(S)).indices.tolist()} & S) / len(S) for h, S in Q.items() if len(S) >= 2]))
        L1 = f"one hop: lives {a1('lives in', PE):.2f}, lies in {a1('lies in', FACTS['city_in']):.2f}, part of {a1('is part of', FACTS['region_of']):.2f}"
        L2 = f"ONE-TO-MANY recall@m: region -> cities {rec(R_C):.2f} (m ~{np.mean([len(v) for v in R_C.values()]):.1f}), country -> regions {rec(K_R):.2f} (m ~{np.mean([len(v) for v in K_R.values()]):.1f})"
        st0 = torch.tensor([ki[a] for a, c in PE]); out = []
        for nm, ws, tru in (('2 hops', ('lives in', 'lies in'), [reg_of[c] for a, c in PE]), ('3 hops', ('lives in', 'lies in', 'is part of'), [cty_of[reg_of[c]] for a, c in PE])):
            tr = torch.tensor([ki[x] for x in tru]); Tl = [T[w] for w in ws]
            seg = []
            for B in (8, 32):
                (ga, gc), vit, mar = chain(Tl, st0, B)
                okg = (ga == tr).numpy(); okv = (vit.argmax(1) == tr).numpy(); okm = (mar.argmax(1) == tr).numpy()
                cv = vit.topk(2, -1).values; cv = (cv[:, 0] - cv[:, 1]).numpy(); cm = mar.max(1).values.numpy()
                if B == 8: seg.append(f"GREEDY {okg.mean():.2f} (top half {top_half(gc.numpy(), okg):.2f})")
                seg.append(f"B{B} MIN-ENERGY PATH {okv.mean():.2f} (top half {top_half(cv, okv):.2f}) | B{B} MARGINAL {okm.mean():.2f} (top half {top_half(cm, okm):.2f})")
                if B == 32 and nm == '2 hops':                                 # goal-directed verification: min path energy A -> C
                    rnd = torch.tensor([ki[REG[rs9.randint(len(REG))][0]] for _ in PE])
                    near = []
                    for c in tru:
                        sib = [r for r in K_R[cty_of[c]] if r != c]; near.append(ki[sib[rs9.randint(len(sib))]] if sib else ki[REG[rs9.randint(len(REG))][0]])
                    near = torch.tensor(near); I = torch.arange(len(PE))
                    gs = lambda C: np.where((ga == C).numpy(), gc.numpy(), -1.0)
                    seg.append(f"VERIFY a claimed answer, AUC true vs random wrong / vs NEAR MISS (same country): GREEDY {auc(gs(tr), gs(rnd)):.2f} / {auc(gs(tr), gs(near)):.2f} | "
                               f"MIN PATH ENERGY {auc(vit[I, tr].numpy(), vit[I, rnd].numpy()):.2f} / {auc(vit[I, tr].numpy(), vit[I, near].numpy()):.2f}")
            out.append(f"{nm}: " + ' | '.join(seg))
        Qk = list(K_C); s0 = torch.tensor([ki[k] for k in Qk]); mres = []
        for B in (1, 8):
            v = T['includes'][s0]; vb, ib = v.topk(B, -1); sc = torch.logsumexp(vb[:, :, None] + T['includes'][ib], 1)
            mres.append(f"{'GREEDY (top region)' if B == 1 else 'MARGINAL over B8 regions'} {np.mean([len({keys[j] for j in sc[i].topk(len(K_C[k])).indices.tolist()} & K_C[k]) / len(K_C[k]) for i, k in enumerate(Qk)]):.2f}")
        P(f"  K {K} | {st} steps (loss {ls:.3f}, {time.time() - T0:.0f}s) | {L1} | {L2}\n      " + '\n      '.join(out)
          + f"\n      ONE-TO-MANY CHAIN country -> regions -> cities (m ~{np.mean([len(v) for v in K_C.values()]):.0f}) recall@m: " + ' | '.join(mres))
P(f"DONE ({time.time() - T0:.0f}s)")
