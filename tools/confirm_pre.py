"""PRETRAINED READER IN THE PIPELINE (20 Nov 2026; tools/confirm_pre.py) - Colab GPU, distant set, the package's own writes and confirmation.
tools/pretrained_reader.py: with TRUE labels a small pretrained model's states rank relation records better than word counts (AUC 0.77-0.80
vs 0.72 on pairs seen once). Does that carry into what the package does with the reader? As tools/confirm_min.py, with two seeded readers
on the same 20,000 labelled records from the documents read so far (CP_FRAC): WORDS (the package's: words between + 3 context words each
side) and STATES (CP_MODEL's hidden states at CP_LAYER: last token of A, mean from A to B, last token before B). For each reader: the
package's code writes (premises._rcode_step, kra14's rule, the reader's top half), how many are right, then confirmation on the documents
read after (any relational record = kra14; one record the reader rates in its top half = kra17; two rated records), right and right kept.
Usage: python tools/confirm_pre.py <train_distant.json.gz>   env CP_DOCS (50000), CP_FRAC (0.4), CP_W (2400), CP_MODEL (gpt2), CP_LAYER (mid|last|N),
       CP_BS (256)"""
import os, sys, gzip, json, types, time, collections
import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time()
NDOC = int(os.environ.get('CP_DOCS', 50000)); FR_ = float(os.environ.get('CP_FRAC', 0.4)); NW = int(os.environ.get('CP_W', 2400))
MODEL = os.environ.get('CP_MODEL', 'gpt2'); LAYER = os.environ.get('CP_LAYER', 'mid'); BS = int(os.environ.get('CP_BS', 256))
D = json.load(gzip.open(sys.argv[1], 'rt'))[:NDOC]; sub = '/tmp/confirm_pre_sub.json.gz'
with gzip.open(sub, 'wt') as f: json.dump(D, f)
cfg = types.SimpleNamespace(inf_premises_path=sub, inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa', inf_rcode_asym=1.0,
                            inf_rcode_selreport=0, inf_rti_max=NW, inf_rret=0, inf_rti_ctl=0.0, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rcode_reader=0.5, inf_premises_seed_ctx=1)
pm._load_distant(cfg); ND = pm._PM['ND']; CX = pm._PM.get('CTX') or {}; R = pm._PM['R']; n0 = int(round(ND * FR_))
G = []                                                                        # each record's sentence and token spans, in the loader's order
for di, d in enumerate(D):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]; bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms); toks = d['sents'][si]
        for x in range(len(ms)):
            for y_ in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y_]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb] or b0 - a1 > 12: continue
                if len(' '.join(toks[a0:b1])) > 60: continue
                G.append((di, si, a0, a1, b0, b1))
assert len(G) == len(R) and all(R[i][7] == ' '.join(D[G[i][0]]['sents'][G[i][1]][G[i][2]:G[i][5]]) for i in range(0, len(R), 997)), 'record order does not match the loader'
y = np.array([bool(r[6]) for r in R]); iRD = np.array([i for i, r in enumerate(R) if r[0] < n0]); RD = [R[i] for i in iRD]
sd = iRD[np.random.RandomState(4 + 707).permutation(len(iRD))[:20000]]
later = np.array([i for i, r in enumerate(R) if r[0] >= n0]); teS = later[np.random.RandomState(5).permutation(len(later))[:30000]]
P(f"{ND} documents, {len(R)} records; {len(RD)} read ({n0} documents); seed 20,000 (related {y[sd].mean():.3f}); held-out sample 30,000 later (related {y[teS].mean():.3f})  ({time.time() - T0:.0f}s)")
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
wtx = lambda r: f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(int(r[5]), 12) // 3} " + ' '.join('C_' + w for w in CX.get(id(r), '').split())
cw = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); cw.fit(hv.transform([wtx(R[i]) for i in sd]), y[sd])
SW = np.concatenate([cw.decision_function(hv.transform([wtx(R[i]) for i in range(j, min(len(R), j + 200000))])) for j in range(0, len(R), 200000)])
P(f"WORDS reader: held-out AUC {roc_auc_score(y[teS], SW[teS]):.3f}  ({time.time() - T0:.0f}s)")
NOSP_B = {',', '.', ';', ':', ')', "'s", "n't", '%', '!', '?', "'", "''"}; NOSP_A = {'(', '$', '``'}
def detok(ws):
    out, prev = '', None
    for w in ws:
        if out and not (w in NOSP_B or prev in NOSP_A): out += ' '
        out += w; prev = w
    return out
def tsp(i):
    di, si, a0, a1, b0, b1 = G[i]; ws = D[di]['sents'][si]; pa = len(detok(ws[:a1])); pre = detok(ws[:b0]); pb = len(pre) + (1 if b0 > 0 else 0)
    return detok(ws[:b1]), pa, pb
from transformers import AutoTokenizer, AutoModel
DV = 'cuda' if torch.cuda.is_available() else 'cpu'
tk = AutoTokenizer.from_pretrained(MODEL); md = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16 if DV == 'cuda' else torch.float32).to(DV).eval()
if tk.pad_token is None: tk.pad_token = tk.eos_token
tk.padding_side = 'right'; NLY = md.config.num_hidden_layers; LY = {'mid': NLY // 2, 'last': NLY}.get(LAYER) or int(LAYER)
FC = {}
def feats(ix):
    need = [int(i) for i in ix if int(i) not in FC]
    for i0 in range(0, len(need), BS):
        ch = [tsp(i) for i in need[i0:i0 + BS]]
        enc = tk([c[0] for c in ch], return_tensors='pt', padding=True, truncation=True, max_length=192, return_offsets_mapping=True); off = enc.pop('offset_mapping').numpy()
        with torch.no_grad(): H_ = md(**{k: v.to(DV) for k, v in enc.items()}, output_hidden_states=True).hidden_states[LY].float()
        for j, (t, pa, pb) in enumerate(ch):
            o = off[j]; valid = np.where(o[:, 1] > o[:, 0])[0]
            if not len(valid): valid = np.array([0])
            ia = valid[o[valid, 0] < pa]; ia = int(ia[-1]) if len(ia) else int(valid[0]); ib = valid[o[valid, 1] <= pb]; ib = max(int(ib[-1]) if len(ib) else ia, ia)
            v = torch.cat([H_[j, ia], H_[j, ia:ib + 1].mean(0), H_[j, ib]]).cpu().numpy()
            FC[need[i0 + j]] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return np.stack([FC[int(i)] for i in ix])
t1 = time.time(); Xs = feats(sd); sc = StandardScaler().fit(Xs)
cs = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced'); cs.fit(sc.transform(Xs), y[sd])
score_s = lambda ix: cs.decision_function(sc.transform(feats(ix)))
SS = np.full(len(R), np.nan); SS[iRD] = np.concatenate([score_s(iRD[j:j + 50000]) for j in range(0, len(iRD), 50000)]); SS[teS] = score_s(teS)
P(f"STATES reader ({MODEL}, layer {LY} of {NLY}): held-out AUC {roc_auc_score(y[teS], SS[teS]):.3f}  (states for {len(FC)} records in {time.time() - t1:.0f}s)")
for nm, S_ in (('WORDS', SW), (f'STATES {MODEL}', SS)):
    pm._PM['rdscore'] = S_[iRD]; pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], RD, 3)
    Wr = {frozenset(k): bool(w['truth']) for k, w in pm._PM['rti'].items()}; thr = float(np.median(S_[iRD])); nR = sum(Wr.values())
    P(f"{nm} reader: {len(Wr)} written, right {np.mean(list(Wr.values())):.3f} ({nR} right)")
    ev_i = [i for i in later if frozenset((R[i][1], R[i][2])) in Wr and 1 <= R[i][5] <= 8 and not pm._rret_listlike(R[i])]
    sc_ = S_ if nm == 'WORDS' else None
    if sc_ is None:
        miss = [i for i in ev_i if np.isnan(SS[i])]
        if miss: SS[miss] = score_s(np.array(miss))
        sc_ = SS
    ev = collections.defaultdict(list)
    for i in ev_i: ev[frozenset((R[i][1], R[i][2]))].append((R[i][0], float(sc_[i])))
    RULES = {'>= 1 record (kra14)': lambda L: len(L) >= 1, '1 rated record (kra17)': lambda L: any(s >= thr for _, s in L),
             '2 rated records': lambda L: sum(s >= thr for _, s in L) >= 2}
    for k in (1, 3):
        cut = n0 + (ND - n0) * k // 3
        for rn, f in RULES.items():
            cf = [Wr[p] for p in Wr if f([e for e in ev.get(p, []) if e[0] < cut])]
            P(f"   read to {cut / ND:.0%} | {rn:24s} confirmed {len(cf):5d} | right {np.mean(cf) if cf else float('nan'):.3f} | right ones kept {sum(cf) / max(nR, 1):.2f}")
P(f"DONE ({time.time() - T0:.0f}s)")
