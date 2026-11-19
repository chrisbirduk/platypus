"""PRETRAINED-BASE READER TEST (19 Nov 2026; tools/pretrained_reader.py) - Colab GPU, distant set, LABEL-FREE (own-document labels for evaluation).
Would a pretrained base lift the reader? The self-trained reader of tools/state_reader.py (usefulness pseudo-labels -> a logistic regression on the
model's states: after A, mean over the words between, just before B) on small PRETRAINED causal language models, against the word-count reader
(our best label-free reader: offline AUC 0.646-0.654, GPU benchmark 0.653) and each model's supervised ceiling (true labels, evaluation only).
Caveat: these models were pretrained on the web, Wikipedia included, so part of any gain can be KNOWING the fact rather than READING it; the
pairs-seen-once rows and the ceiling help to read the numbers, but a clean test of reading needs invented names (a later step).
Usage: python tools/pretrained_reader.py <train_distant.json.gz>   env PR_DOCS (20000), PR_MODELS (comma list), PR_LAYERS (mid,last),
       PR_TRAIN (20000 a class at most), PR_TEST (30000), PR_BS (128)"""
import os, sys, gzip, json, time, collections
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
T0 = time.time(); P = lambda s: print(s, flush=True)
DOCS = json.load(gzip.open(sys.argv[1], 'rt')); ND = min(len(DOCS), int(os.environ.get('PR_DOCS', 20000))); DOCS = DOCS[:ND]
MODELS = os.environ.get('PR_MODELS', 'gpt2,HuggingFaceTB/SmolLM2-360M,Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B').split(',')
NOSP_B = {',', '.', ';', ':', ')', "'s", "n't", '%', '!', '?', "'", "''"}; NOSP_A = {'(', '$', '``'}
def detok(ws):
    out, prev = '', None
    for w in ws:
        if out and not (w in NOSP_B or prev in NOSP_A): out += ' '
        out += w; prev = w
    return out
pd, nd = collections.defaultdict(set), collections.defaultdict(set); recs = []
for di, d in enumerate(DOCS):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    lab = {(l['h'], l['t']) for l in d['labels']}; bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms)
        for x in range(len(ms)):
            for y_ in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y_]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb] or b0 - a1 > 12: continue
                pd[frozenset((cn[ea], cn[eb]))].add(di); nd[cn[ea]].add(di); nd[cn[eb]].add(di)
                recs.append((di, si, a0, a1, b0, b1, cn[ea], cn[eb], (ea, eb) in lab or (eb, ea) in lab))
y = np.array([r[8] for r in recs], float); doc = np.array([r[0] for r in recs]); te = doc >= int(ND * 0.8); tr = ~te
def use_(r):
    a_ = len(pd[frozenset((r[6], r[7]))]) - 1; nA, nB = len(nd[r[6]]) - 1, len(nd[r[7]]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); sA, sB = np.log(ND / (nA + 2)), np.log(ND / (nB + 2))
    return max(hA * sB - (1 - hA) * 0.5, hB * sA - (1 - hB) * 0.5)
U = np.array([use_(r) for r in recs]); rs = np.random.RandomState(0)
sn = np.where(tr & ~np.isnan(U))[0]; hi, lo = np.quantile(U[sn], 0.8), np.quantile(U[sn], 0.5)
NT = int(os.environ.get('PR_TRAIN', 20000)); ps = sn[U[sn] >= hi]; ng = sn[U[sn] <= lo]
ps = rs.choice(ps, min(NT, len(ps)), replace=False); ng = rs.choice(ng, min(NT, len(ng)), replace=False)
trS = rs.choice(np.where(tr)[0], min(2 * NT, tr.sum()), replace=False)                       # a training sample for the supervised ceiling
teS = rs.choice(np.where(te)[0], min(int(os.environ.get('PR_TEST', 30000)), te.sum()), replace=False)
once = np.isnan(U[teS])
P(f"{ND} documents, {len(recs)} records; pseudo-labels {len(ps)} / {len(ng)} (true related {y[ps].mean():.3f} / {y[ng].mean():.3f}); held-out sample {len(teS)} "
  f"(base {y[teS].mean():.3f}; pairs seen once {once.mean():.2f})  ({time.time() - T0:.0f}s)")
def rep(nm, s):
    out = []
    for lab_, m in (('all', np.ones(len(teS), bool)), ('seen once', once)):
        yy, ss = y[teS][m], s[m]; o = np.argsort(-ss)
        out.append(f"{lab_}: AUC {roc_auc_score(yy, ss):.3f} top 1% {yy[o[:max(1, len(o) // 100)]].mean():.2f} 5% {yy[o[:max(1, len(o) // 20)]].mean():.2f}")
    P(f"   {nm:62s} " + " | ".join(out))
def lr(Xtr, ytr, Xte, C=0.1):
    c = LogisticRegression(C=C, max_iter=1000, class_weight='balanced'); c.fit(Xtr, ytr); return c.decision_function(Xte)
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
wtx = lambda r: ' '.join(DOCS[r[0]]['sents'][r[1]][r[3]:r[4]]).lower() + f" DIST_{min(r[4] - r[3], 12) // 3}"
Xw = lambda ix: hv.transform([wtx(recs[i]) for i in ix])
P("WORD-COUNT READER (our current best, label-free):")
rep('words between the names, self-trained from usefulness', lr(Xw(np.r_[ps, ng]), np.r_[np.ones(len(ps)), np.zeros(len(ng))], Xw(teS), 1.0))
rep('reference: words, TRUE labels (ceiling)', lr(Xw(trS), y[trS], Xw(teS), 1.0))
def text_and_spans(r):
    ws = DOCS[r[0]]['sents'][r[1]]; pa = len(detok(ws[:r[3]])); pre = detok(ws[:r[4]]); pb = len(pre) + (1 if r[4] > 0 else 0)
    return detok(ws[:r[5]]), pa, pb
from transformers import AutoTokenizer, AutoModel
for name in MODELS:
    try:
        DV = 'cuda' if torch.cuda.is_available() else 'cpu'
        tk = AutoTokenizer.from_pretrained(name); md = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if DV == 'cuda' else torch.float32).to(DV).eval()
        if tk.pad_token is None: tk.pad_token = tk.eos_token
        tk.padding_side = 'right'; NLY = md.config.num_hidden_layers; LY = {'mid': NLY // 2, 'last': NLY}
        LS = [l for l in os.environ.get('PR_LAYERS', 'mid,last').split(',') if l in LY]
        def feats(ix):
            F = {l: [] for l in LS}; BS = int(os.environ.get('PR_BS', 128))
            for i0 in range(0, len(ix), BS):
                ch = [text_and_spans(recs[i]) for i in ix[i0:i0 + BS]]
                enc = tk([c[0] for c in ch], return_tensors='pt', padding=True, truncation=True, max_length=192, return_offsets_mapping=True)
                off = enc.pop('offset_mapping').numpy()
                with torch.no_grad(): out = md(**{k: v.to(DV) for k, v in enc.items()}, output_hidden_states=True)
                for j, (t, pa, pb) in enumerate(ch):
                    o = off[j]; valid = np.where(o[:, 1] > o[:, 0])[0]
                    if not len(valid): valid = np.array([0])
                    ia = valid[o[valid, 0] < pa]; ia = int(ia[-1]) if len(ia) else int(valid[0])          # last token of A
                    ib = valid[o[valid, 1] <= pb]; ib = int(ib[-1]) if len(ib) else ia                   # last token before B
                    ib = max(ib, ia)
                    for l in LS:
                        H = out.hidden_states[LY[l]][j].float()
                        F[l].append(torch.cat([H[ia], H[ia:ib + 1].mean(0), H[ib]]).cpu().numpy())
            return {l: np.stack(v) for l, v in F.items()}
        t1 = time.time(); Ftr = feats(np.r_[ps, ng]); Fsu = feats(trS); Fte = feats(teS)
        P(f"{name} ({sum(p.numel() for p in md.parameters()) / 1e6:.0f}M parameters, {NLY} layers; states in {time.time() - t1:.0f}s):")
        for l in LS:
            sc = StandardScaler().fit(Ftr[l])
            rep(f'STATES layer {LY[l]} ({l}), self-trained from usefulness', lr(sc.transform(Ftr[l]), np.r_[np.ones(len(ps)), np.zeros(len(ng))], sc.transform(Fte[l])))
            sc2 = StandardScaler().fit(Fsu[l]); rep(f'reference: STATES layer {LY[l]} ({l}), TRUE labels (ceiling)', lr(sc2.transform(Fsu[l]), y[trS], sc2.transform(Fte[l])))
        del md; torch.cuda.empty_cache()
    except Exception as e: P(f"{name}: unavailable ({type(e).__name__}: {str(e)[:200]})")
P(f"DONE ({time.time() - T0:.0f}s)")
