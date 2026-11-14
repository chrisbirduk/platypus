"""INFERENCE AS COMPRESSION OF THE FACT TABLE (14 Nov 2026; tools/fact_compress.py) - offline, no effect on the package.
The record compressor (tools/record_mdl.py) codes p(A | class A) and p(B | class B) independently: WHICH names go together - the facts - is
never compressed, so the redundancy an inference is (A-B, B-C => A-C) is invisible to it. Here the store's records are compressed as a
graph of names instead (a low-rank code per name, fitted without labels) and the code is asked about pairs it never saw together.
HELD-OUT DOCUMENTS (TFC_NT of Re-DocRED train + dev, by FC_SEED): every record of those documents is removed from the store, and so is any
other record (Re-DocRED or distant) that pairs two names co-occurring in a held-out document. Candidates: name pairs co-occurring in a held-out
document, both names still in the store; gold = related in that document (evaluation only). Nothing in the store states any candidate pair.
Scores (label-free):
  random | degree product (popularity: preferential attachment) | common neighbours (bridges B) | Adamic-Adar (bridges weighted 1/log deg)
  CODE k     cosine of the names' codes: PPMI of the name x name record graph, truncated SVD (k dimensions) - the compressed fact table
  CODE k rel the same graph weighted by the label-free relational-wording learner (consistency teacher) on Re-DocRED records
AUC and gold share in the top 5 / 10 / 25 %, on all candidates and on those with a bridge (where chaining could act).
Usage: RM_EXTRA=<distant documents> python3 tools/fact_compress.py   env FC_NT (held-out documents, 400), FC_SEED (0), FC_K (32,64,128)"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_rm = open(os.path.join(HERE, 'record_mdl.py')).read()
exec(compile(_rm[:_rm.index('FITS = {}')], 'record_mdl', 'exec'))
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
NT = int(os.environ.get('FC_NT', 400)); FS = int(os.environ.get('FC_SEED', 0)); KS_ = [int(k) for k in os.environ.get('FC_K', '32,64,128').split(',')]
rh = np.random.RandomState(100 + FS); HOLD = set(rh.permutation(len(docs))[:NT].tolist())
cnz = lambda d: [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
CAND = {}; GOLD = set()
for di in HOLD:
    d = docs[di]; cn = cnz(d); ns = sorted(set(cn))
    for i_ in range(len(ns)):
        for j_ in range(i_ + 1, len(ns)): CAND[frozenset((ns[i_], ns[j_]))] = di
    for l in d['labels']:
        if cn[l['h']] != cn[l['t']]: GOLD.add(frozenset((cn[l['h']], cn[l['t']])))
NAMESA = np.array(NAMES, dtype=object)
dvec = np.array([c['di'] for c in C] + [-1] * (len(Af) - N))
pk = [frozenset((NAMES[x], NAMES[z])) for x, z in zip(Af, Bf)]
keep = np.array([(dvec[i] not in HOLD) and (pk[i] not in CAND) and Af[i] != Bf[i] for i in range(len(Af))])
P(f"held-out documents {NT}: candidate pairs {len(CAND)}, gold-related {sum(1 for k in CAND if k in GOLD)} | store records kept {int(keep.sum())} of {len(Af)} "
  f"(removed: the held-out documents' records and any record pairing two names of one held-out document)  ({time.time() - T0:.0f}s)")
# relational weight (label-free learner) for Re-DocRED records; distant records weight 1
rw = np.ones(len(Af)); rw[:N] = 1.0 / (1.0 + np.exp(-(s_all - np.median(s_all))))
def graph(wt):
    A_ = csr_matrix((wt[keep], (Af[keep], Bf[keep])), shape=(NN, NN)); A_ = A_ + A_.T; A_.sum_duplicates(); return A_
G1 = graph(np.ones(len(Af))); G1.data[:] = 1.0                                          # distinct neighbours
deg = np.asarray(G1.sum(1)).ravel()
ok = [k for k in CAND if all(NI.get(n) is not None and deg[NI[n]] > 0 for n in k)]
if int(os.environ.get('FC_SAMEC', 0)):                                                   # the candidate set of the Re-DocRED-only store: both names
    G0 = csr_matrix((np.ones(int((keep & (np.arange(len(Af)) < N)).sum())), (Af[keep & (np.arange(len(Af)) < N)], Bf[keep & (np.arange(len(Af)) < N)])), shape=(NN, NN))
    d0_ = np.asarray((G0 + G0.T).sum(1)).ravel(); ok = [k for k in ok if all(d0_[NI[n]] > 0 for n in k)]   #   keep a Re-DocRED record

XA_ = np.array([NI[sorted(k)[0]] for k in ok]); XB_ = np.array([NI[sorted(k)[1]] for k in ok]); yy = np.array([k in GOLD for k in ok])
cn_cnt = np.asarray(G1[XA_].multiply(G1[XB_]).sum(1)).ravel()
iw = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
aa = np.asarray(G1[XA_].multiply(G1[XB_]).multiply(iw[None, :]).sum(1)).ravel()
SC = {'random': np.random.RandomState(FS).rand(len(ok)), 'degree product (popularity)': deg[XA_] * deg[XB_], 'common neighbours (bridges)': cn_cnt,
      'Adamic-Adar': aa}
def ppmi(A_):
    A_ = A_.tocoo(); rs_ = np.asarray(A_.sum(1)).ravel(); tot = A_.sum(); v = np.log(tot * A_.data / (rs_[A_.row] * rs_[A_.col]) + 1e-12); m = v > 0
    return csr_matrix((v[m].astype(np.float32), (A_.row[m], A_.col[m])), shape=A_.shape)
for tag, wt in ((('', np.ones(len(Af))), (' rel', rw)) if not int(os.environ.get('FC_NOREL', 0)) else (('', np.ones(len(Af))),)):
    Gw = graph(wt); Pm = ppmi(Gw)
    for k in KS_:
        U = normalize(TruncatedSVD(k, random_state=0).fit_transform(Pm).astype(np.float32)); SC[f'CODE {k}{tag} (compressed fact table)'] = np.einsum('ij,ij->i', U[XA_], U[XB_])
    P(f"   codes fitted{tag}  ({time.time() - T0:.0f}s)")
def rep(mask, title):
    yv = yy[mask]; P(f"{title}: {int(mask.sum())} candidates, gold {yv.mean():.3f}")
    for k, s in SC.items():
        sv = s[mask]; o = np.argsort(-sv + 1e-12 * np.random.RandomState(1).rand(len(sv)))
        pk_ = "/".join(f"{yv[o[:max(1, len(o) * q // 100)]].mean():.3f}" for q in (5, 10, 25))
        dp = np.log1p(SC['degree product (popularity)'][mask]); qb = np.quantile(dp, [0.2, 0.4, 0.6, 0.8]); bins = np.searchsorted(qb, dp)
        sa = [(auc(sv[(bins == q) & yv], sv[(bins == q) & ~yv]), int(((bins == q) & yv).sum())) for q in range(5) if ((bins == q) & yv).sum() >= 5 and ((bins == q) & ~yv).sum() >= 5]
        pm = sum(a * n for a, n in sa) / max(1, sum(n for a, n in sa)) if sa else float('nan')
        P(f"   {k:44s} AUC {auc(sv[yv], sv[~yv]):.3f} | gold in top 5/10/25%: {pk_} | POPULARITY-MATCHED AUC (within degree-product quintiles) {pm:.3f}")
if int(os.environ.get('FC_SELECT', 0)):                                                  # 14 Nov: which SELECTION RULE writes the most right inferences?
    rk_ = lambda x: np.argsort(np.argsort(x)) / max(1, len(x) - 1); ck = max(k for k in SC if k.startswith('CODE') and 'rel' not in k)
    cd_ = SC[ck]; SEL = {'code': cd_, 'code, bridged pairs only': np.where(cn_cnt > 0, cd_, -9), 'code + Adamic-Adar (rank sum)': rk_(cd_) + rk_(SC['Adamic-Adar']),
           'code + Adamic-Adar, bridged only': np.where(cn_cnt > 0, rk_(cd_) + rk_(SC['Adamic-Adar']), -9), 'code, 2+ bridges only': np.where(cn_cnt >= 2, cd_, -9),
           'Adamic-Adar': SC['Adamic-Adar']}
    P(f"SELECTION RULES ({ck}), gold among the top N written (all {len(ok)} candidates, base {yy.mean():.3f}):")
    for k_, sv in SEL.items():
        o = np.argsort(-sv + 1e-12 * np.random.RandomState(2).rand(len(sv))); P(f"   {k_:40s} " + " | ".join(f"top {n} {yy[o[:n]].mean():.3f}" for n in (50, 100, 200, 400)))
rep(np.ones(len(ok), bool), "ALL CANDIDATE PAIRS (never stated together in the store)")
rep(cn_cnt > 0, "WITH A BRIDGE (a chain A-B-C exists in the store)")
rep(cn_cnt == 0, "NO BRIDGE (no chain: only a code can relate them)")
P(f"DONE ({time.time() - T0:.0f}s)")
