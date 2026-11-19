"""SEVERAL INDEPENDENT TEACHERS, COMBINED WITHOUT LABELS (19 Nov 2026; tools/teachers_reader.py) - offline, distant set, LABEL-FREE (own-document
labels for evaluation only). Can feedback from several weak, independent label-free signals let the reader keep improving as it reads?
Teachers (each votes +1, -1 or abstains on a record, from what has been read so far):
  USEFUL   the pair recurs in other documents and predicts its names well (top 20% +1, bottom half -1; never recurs: abstain)
  CONFIRM  another document states the pair in a RELATIONAL wording (+1); it recurs only in lists (-1); otherwise abstain (unconfirmed is NOT false)
  WORDING  the words between are a FUNCTIONAL wording (most subjects have one object with it, 5+ subjects) (+1); a list or coordination (-1)
  FAR      more than 8 words between the names (-1); otherwise abstain
A LABEL MODEL (Dawid-Skene EM; no labels) estimates each teacher's reliability from their agreements and gives each record a probability of
being related; the STUDENT (logistic regression on the words between) learns from those probabilities. Documents arrive in blocks (TR_BLOCK);
after each block the teachers and the student are rebuilt from everything read and the student is scored on the NEXT, unread block.
Arms: COMBINED (label model), USEFUL ONLY (the current reader's teacher), MAJORITY VOTE (no label model).
Usage: python3 tools/teachers_reader.py <distant.json.gz>   env TR_BLOCK (5000), TR_ITERS (30)"""
import os, sys, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=1))
R, ND = pm._PM['R'], pm._PM['ND']; BL = int(os.environ.get('TR_BLOCK', 5000)); NB = ND // BL
y = np.array([bool(r[6]) for r in R]); doc = np.array([r[0] for r in R]); blk = doc // BL; key = [frozenset((r[1], r[2])) for r in R]
lst = np.array([pm._rret_listlike(r) for r in R]); dist = np.array([r[5] for r in R]); relw = (dist >= 1) & (dist <= 8) & ~lst
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3}" for r in R])
TN = ['USEFUL', 'CONFIRM', 'WORDING', 'FAR']
P(f"{ND} documents in {NB} blocks of {BL}, {len(R)} records (related {y.mean():.3f}; list-like {lst.mean():.3f})  ({time.time() - T0:.0f}s)")
def teachers(upto):
    I = np.where(blk < upto)[0]; pd, nd = collections.defaultdict(set), collections.defaultdict(set); relD = collections.defaultdict(set); lstD = collections.defaultdict(set)
    txt = collections.defaultdict(set); fsub = collections.defaultdict(lambda: collections.defaultdict(set))
    for i in I:
        r = R[i]; k = key[i]; pd[k].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0]); txt[k].add(r[7]); fsub[r[3]][r[1]].add(r[2])
        (relD if relw[i] else lstD if lst[i] else collections.defaultdict(set))[k].add(r[0])
    Nd = upto * BL; FUN = {f: np.mean([len(v) == 1 for v in d.values()]) for f, d in fsub.items() if len(d) >= 5}
    def votes(i):
        r = R[i]; k = key[i]; own = r[0]; v = [0, 0, 0, 0]
        a_ = len(pd[k] - {own}); nA, nB = len(nd[r[1]] - {own}), len(nd[r[2]] - {own})
        if a_:
            hA, hB = a_ / (nA + 2), a_ / (nB + 2); u = max(hA * np.log(Nd / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(Nd / (nA + 2)) - (1 - hB) * 0.5); v[0] = u
        else: v[0] = np.nan
        if len(relD[k] - {own}) > 0: v[1] = 1
        elif len(lstD[k] - {own}) > 0 and not relD[k] - {own}: v[1] = -1
        if lst[i]: v[2] = -1
        elif FUN.get(r[3], 0) >= 0.8: v[2] = 1
        if dist[i] > 8: v[3] = -1
        return v
    return votes
def useful_to_vote(V):
    u = V[:, 0]; ok = ~np.isnan(u); out = np.zeros(len(u))
    if ok.sum() > 10: hi, lo = np.quantile(u[ok], 0.8), np.quantile(u[ok], 0.5); out[ok & (u >= hi)] = 1; out[ok & (u <= lo)] = -1
    return out
def label_model(L, iters=int(os.environ.get('TR_ITERS', 30))):
    """Dawid-Skene EM, abstains NOT modelled as evidence (as in data programming: whether a teacher votes is assumed independent of the class;
    modelling it let EM call 'recurs in other documents' the related class): p(y=1) and each teacher's P(vote = +1 | y) among its votes"""
    n, m = L.shape; post = np.clip(0.2 + 0.3 * np.tanh(L[:, 0] + 0.5 * L[:, 1]), 0.02, 0.9)
    for _ in range(iters):
        pi = float(np.clip(post.mean(), 0.02, 0.5)); a = np.zeros((m, 2))      # a[j, c] = P(vote +1 | y = c, teacher j votes)
        for j in range(m):
            v = L[:, j] != 0
            for c, w in ((1, post), (0, 1 - post)): a[j, c] = ((w * (L[:, j] == 1))[v].sum() + 1) / (w[v].sum() + 2)
        lp1 = np.full(n, np.log(pi)); lp0 = np.full(n, np.log(1 - pi))
        for j in range(m):
            lp1 += np.where(L[:, j] == 1, np.log(a[j, 1]), np.where(L[:, j] == -1, np.log(1 - a[j, 1]), 0.0))
            lp0 += np.where(L[:, j] == 1, np.log(a[j, 0]), np.where(L[:, j] == -1, np.log(1 - a[j, 0]), 0.0))
        post = 1 / (1 + np.exp(np.clip(lp0 - lp1, -50, 50)))
    if a[0, 1] < a[0, 0]: post = 1 - post; pi = 1 - pi                       # identity: USEFUL +1 likelier for the related class
    return post, a, pi
def fit_soft(ix, p):
    c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced')
    c.fit(X[np.r_[ix, ix]], np.r_[np.ones(len(ix)), np.zeros(len(ix))], sample_weight=np.r_[p, 1 - p]); return c
def auc5(s, I):
    yy = y[I]; o = np.argsort(-s); return f"AUC {roc_auc_score(yy, s):.3f} top5% {yy[o[:max(1, len(o) // 20)]].mean():.2f}"
curves = collections.defaultdict(list)
for t in range(1, NB):
    vt = teachers(t); I = np.where(blk < t)[0]; I = I[np.random.RandomState(t).permutation(len(I))[:300000]]
    V = np.array([vt(i) for i in I], float); L = np.c_[useful_to_vote(V), V[:, 1:]].astype(int); act = (L != 0).any(1)
    post, th, pi = label_model(L[act]); ia = I[act]
    acc = []
    for j, nm in enumerate(TN):
        for val, lab in ((1, '+'), (-1, '-')):
            m_ = L[act][:, j] == val
            if m_.sum() >= 20: acc.append(f"{nm}{lab} {m_.sum()} est {np.mean(post[m_] if val == 1 else 1 - post[m_]):.2f} true {np.mean(y[ia][m_] if val == 1 else ~y[ia][m_]):.2f}")
    sure = post >= 0.8
    cC = fit_soft(ia, post)
    uv = L[act][:, 0]; cU = fit_soft(ia[uv != 0], (uv[uv != 0] > 0).astype(float))
    mv = np.sign(L[act].sum(1)); cM = fit_soft(ia[mv != 0], (mv[mv != 0] > 0).astype(float))
    La = L[act]                                                                # RULES (label-free union): related = useful +1, or confirmed AND functional wording;
    rp = (La[:, 0] == 1) | ((La[:, 1] == 1) & (La[:, 2] == 1)); rn = (La[:, 0] == -1) | (La[:, 3] == -1) | ((La[:, 2] == -1) & (La[:, 1] != 1))   # unrelated = useful -1,
    rr_ = rp ^ rn; cR = fit_soft(ia[rr_], rp[rr_].astype(float))               #   far, or a list never confirmed
    uu = La[:, 0] != 0; cF = fit_soft(ia[uu], post[uu])                        # REFINE: the useful-voted records only, labelled by the combined posterior
    Inext = np.where(blk == t)[0]
    for nm, c in (('RULES', cR), ('REFINE', cF)): curves[nm].append(roc_auc_score(y[Inext], c.decision_function(X[Inext])))
    for nm, c in (('COMBINED', cC), ('USEFUL ONLY', cU), ('MAJORITY VOTE', cM)):
        s = c.decision_function(X[Inext]); curves[nm].append(roc_auc_score(y[Inext], s))
    P(f"   read {t * BL:5d} docs: {act.sum()} of {len(I)} sampled records with a vote | label model: prior {pi:.3f}, posterior >= 0.8 for {sure.sum()} (right {y[ia][sure].mean() if sure.any() else float('nan'):.2f}) "
      f"| teachers (estimated vs true accuracy): {'; '.join(acc)} || next block: " + " | ".join(f"{nm} {auc5(c.decision_function(X[Inext]), Inext)}" for nm, c in (('COMBINED', cC), ('USEFUL ONLY', cU), ('MAJORITY VOTE', cM), ('RULES', cR), ('REFINE', cF)))
      + f"  ({time.time() - T0:.0f}s)")
P("CURVES (AUC on each unread block): " + " || ".join(f"{nm}: " + " ".join(f"{a:.3f}" for a in v) for nm, v in curves.items()))
P(f"DONE ({time.time() - T0:.0f}s)")
