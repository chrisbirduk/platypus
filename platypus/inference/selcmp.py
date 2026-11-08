"""SELECTOR COMPARISON (inf_selcmp 1; readout at the end of the run, the model is unchanged): does the trained JEPA selector do more than its
two-rule linear precursor? Every selector ranks candidate partner sentences for a cue sentence:
  JEPA (trained)   the dream selector (_jd_scores: predicted partner code against target codes), and the JEPA index arms if present
  CCA (linear)     closed-form canonical correlation between trunk codes of CONSECUTIVE stored sentences (a linear temporal JEPA), top 32
  SFA (linear)     slow feature analysis: in the whitened trunk-code space, the 32 directions that change least between consecutive sentences
  TRUNK            raw cosine of trunk codes (mean of the trunk's hidden states)
  TF-IDF NAMES     shared rare capitalised words
  RANDOM
Scored three ways:
  (1) RETRIEVAL   planted chains (readout only): from a chain's first fact, the rank of its second fact among those of every chain and
                  400 stored real sentences (top-1, top-5)
  (2) JUDGE       mutual compression of the chosen pair: bits on the partner after the cue minus after a random stored sentence (mean
                  bits/byte saved, share of picks that pass, > 0), each selector's top pick from 400 candidates, 64 cues
  (3) RELATIONS   Re-DocRED paragraphs: from each sentence, the selector's top pick among the other sentences of the same paragraph; share
                  of picks that hold a GOLD RELATION across the two sentences, against a random sentence of the paragraph (the floor)"""
import re
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import _fail


def _codes(model, txts):
    from platypus.inference.jepa import _jepa_emb
    with torch.no_grad(): return _jepa_emb(model, [bytes(t) for t in txts]).float()


def _whiten(Z, k=64, eps=1e-3):
    mu = Z.mean(0); X = Z - mu; U, S, V = torch.linalg.svd(X, full_matrices=False); k = min(k, V.shape[0])
    W = V[:k].T / (S[:k] / max(1.0, (len(Z) - 1) ** 0.5) + eps); return mu, W


def _fit_linear(ZA, ZB, r=32):
    """CCA and SFA projections from consecutive pairs (ZA[i], ZB[i]) of trunk codes"""
    Z = torch.cat([ZA, ZB]); mu, W = _whiten(Z)
    A = (ZA - mu) @ W; B = (ZB - mu) @ W
    C = A.T @ B / len(A); U, S, V = torch.linalg.svd(C); r = min(r, U.shape[1])
    cca = (mu, W @ U[:, :r], W @ V.T[:, :r])                                      # query projection, candidate projection
    D = A - B; ev, evec = torch.linalg.eigh(D.T @ D / len(D))                     # slowest = smallest change between neighbours
    P = W @ evec[:, :r]; sfa = (mu, P, P)
    return cca, sfa, float(S[:r].mean())


def _score_lin(proj, Zq, Zc):
    mu, Pq, Pc = proj
    return F.normalize((Zq - mu) @ Pq, dim=-1) @ F.normalize((Zc - mu) @ Pc, dim=-1).T


def selector_compare(model, cfg, S_, slot_of, tag):
    from platypus.inference.jepa import _jd_scores, _tfidf_sim
    from platypus.inference.probes import inf_probes, _bench
    from platypus.inference.residual import _bits_rows
    try:
        rs = np.random.RandomState(int(cfg.seed) + 8801); it = S_.get('it', [])
        real = [x for x in it if x.get('k') in ('real', 'ver') and not x.get('gone') and 20 <= len(x['s']) <= 300]
        if len(real) < 600: print(f"  [SELECTOR COMPARISON | {tag}] skipped: {len(real)} stored sentences", flush=True); return
        # linear precursors fitted on consecutive stored sentences of the same domain and visit (the stream supplies the pairs)
        ix = [i for i in range(len(it) - 1) if it[i].get('k') == 'real' and it[i + 1].get('k') == 'real' and it[i]['d'] == it[i + 1]['d']
              and it[i].get('v') == it[i + 1].get('v') and 20 <= len(it[i]['s']) <= 300 and 20 <= len(it[i + 1]['s']) <= 300]
        ix = [ix[int(k)] for k in rs.choice(len(ix), min(3000, len(ix)), replace=False)]
        cca, sfa, rho = _fit_linear(_codes(model, [it[i]['s'] for i in ix]), _codes(model, [it[i + 1]['s'] for i in ix]))
        sels = ['JEPA (trained)', 'CCA (linear)', 'SFA (linear)', 'TRUNK cosine', 'TF-IDF names', 'RANDOM']
        def score(name, qt, ct):
            M = score0(name, qt, ct)
            return None if M is None else M + 1e-6 * torch.rand(M.shape)               # ties broken at random (TF-IDF rows are often all zero)
        def score0(name, qt, ct):
            if name == 'JEPA (trained)':
                m = _jd_scores(model, cfg, [bytes(t) for t in qt], [bytes(t) for t in ct]); return None if m is None else m.float().cpu()
            if name in ('CCA (linear)', 'SFA (linear)', 'TRUNK cosine'):
                Zq, Zc = _codes(model, qt), _codes(model, ct)
                if name == 'TRUNK cosine': return (F.normalize(Zq, dim=-1) @ F.normalize(Zc, dim=-1).T).cpu()
                return _score_lin(cca if name.startswith('CCA') else sfa, Zq, Zc).cpu()
            if name == 'TF-IDF names': return _tfidf_sim(S_, [bytes(t) for t in qt], [bytes(t) for t in ct], int(cfg.inf_df), True).float()
            return torch.tensor(rs.rand(len(qt), len(ct)), dtype=torch.float32)
        out = {s: {} for s in sels}
        # (1) RETRIEVAL on planted chains
        ch = [c for c in inf_probes(cfg) if len(c.get('facts', [])) >= 2]
        q1 = [c['facts'][0][1].encode('latin-1', 'replace') for c in ch]; t2 = [c['facts'][1][1].encode('latin-1', 'replace') for c in ch]
        pool = t2 + [real[int(k)]['s'] for k in rs.choice(len(real), 400, replace=False)]
        for s in sels:
            M = score(s, q1, pool)
            if M is None: continue
            rk = [(M[i] > M[i, i]).sum().item() + 1 for i in range(len(q1))]
            out[s]['ret'] = (float(np.mean([r <= 1 for r in rk])), float(np.mean([r <= 5 for r in rk])))
        # (2) JUDGE: mutual compression of each selector's top pick
        cues = [real[int(k)]['s'] for k in rs.choice(len(real), 64, replace=False)]; cand = [real[int(k)]['s'] for k in rs.choice(len(real), 400, replace=False)]
        rnd = [real[int(k)]['s'] for k in rs.choice(len(real), 64, replace=False)]; sl = int(slot_of(0)) if callable(slot_of) else 0
        for s in sels:
            M = score(s, cues, cand)
            if M is None: continue
            pick = [cand[int(M[i].argmax())] for i in range(len(cues))]
            rows = [c + b' ' + p for c, p in zip(cues, pick)] + [r + b' ' + p for r, p in zip(rnd, pick)]
            bits = _bits_rows(model, rows, sl, False); n = len(cues); g = []
            for i in range(n):
                a = bits[i][len(cues[i]) + 1:].mean(); b = bits[n + i][len(rnd[i]) + 1:].mean(); g.append(float(b - a))
            out[s]['judge'] = (float(np.mean(g)), float(np.mean(np.array(g) > 0)))
        # (3) RELATIONS within Re-DocRED paragraphs
        class _C: inf_bench = 'embedded'; inf_bench_maxamb = 1
        G = _bench(_C()); t2e = {t.encode('utf-8'): next(iter(es)) for t, es in G['_t2e'].items()}
        REL = {(min(a, b), max(a, b)) for a, b, _, _ in G['rel']}; WR = re.compile(rb"[A-Za-z0-9][A-Za-z0-9'-]*")
        ents = lambda s: {t2e[w] for w in WR.findall(s) if w in t2e}
        def gold(a, b): return any((min(x, y), max(x, y)) in REL for x in ents(a) for y in ents(b) if x != y)
        par = []
        for docs in G['docs'].values():
            for doc in docs:
                ss = [s[:300] for s in re.split(rb'(?<=[.!?])\s+', doc.encode('utf-8', 'replace')) if len(s) >= 20]
                if len(ss) >= 3: par.append(ss)
        floor = []
        for ss in par:
            for i in range(len(ss)):
                if ents(ss[i]): floor += [gold(ss[i], ss[j]) for j in range(len(ss)) if j != i]
        for s in sels:
            hits = []
            for ss in par:
                M = score(s, ss, ss)
                if M is None: hits = None; break
                M = M.clone(); M.fill_diagonal_(-1e9)
                hits += [gold(ss[i], ss[int(M[i].argmax())]) for i in range(len(ss)) if ents(ss[i])]
            if hits: out[s]['rel'] = (float(np.mean(hits)), len(hits))
        f = lambda v, k, j, fmt: (fmt.format(v[k][j]) if k in v else '-')
        print(f"  [SELECTOR COMPARISON | {tag}] trained JEPA selector against its linear precursors (CCA / SFA fitted in closed form on {len(ix)} consecutive "
              f"stored sentence pairs, mean canonical correlation {rho:.2f}) | (1) RETRIEVAL of a planted chain's second fact from its first ({len(q1)} chains, "
              f"{len(pool)} candidates): top-1/top-5 | (2) JUDGE on the top pick of 400: mean bits/byte saved over a random cue, share passing | "
              f"(3) RELATIONS: top pick within a Re-DocRED paragraph holds a gold relation (floor: a random sentence of the paragraph {np.mean(floor):.3f}, n {len(floor)}) || "
              + " || ".join(f"{s}: (1) {f(out[s], 'ret', 0, '{:.2f}')}/{f(out[s], 'ret', 1, '{:.2f}')} (2) {f(out[s], 'judge', 0, '{:+.3f}')}, {f(out[s], 'judge', 1, '{:.2f}')} "
                            f"(3) {f(out[s], 'rel', 0, '{:.3f}')}" for s in sels), flush=True)
    except Exception as e:
        _fail('inference.selcmp'); print(f"  [SELECTOR COMPARISON] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
