"""REAL-TEXT NAME FINDING (12 Nov 2026; tools/realtext_names.py) - the toy test of tools/toy_jname.py on real text: can the gap between
BYTE surprise (a causal byte language model) and JEPA surprise (a text JEPA predicting the latents of a masked word) pick out the words
of names, including the ones the fast channel misses (82% of Re-DocRED gold relations involve a name it never isolates)?
Both models trained here on local Wikipedia + the benchmark text (128-byte windows; the toy's architectures). Scored on the benchmark
documents, per word: GOLD = the word is part of a benchmark entity's name (Re-DocRED annotation); AUCs of byte surprise (high = name),
JEPA surprise (low = name), the GAP z(byte) - z(JEPA), and CAPITALISED-NOT-FIRST (the fast channel's cue) - on all words, and on the
gold-entity words the fast channel cannot see (lower-case, or sentence-initial) against ordinary words.
Usage: python3 tools/realtext_names.py [lm steps] [jepa steps]   (env RN_KIND ema | lejepa, default lejepa)"""
import os
import re
import sys
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
LMS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; JS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
KIND = os.environ.get('RN_KIND', 'lejepa')
_t = open(os.path.join(HERE, 'toy_jname.py')).read()
sys_argv = sys.argv; sys.argv = [sys.argv[0], '0', '10']
exec(compile(_t[:_t.index('# ---------------------------------------------------------------- readouts')], 'toy_jname', 'exec'))   # models, mask_words, sigreg
sys.argv = sys_argv
_c = open(os.path.join(HERE, 'chunk_reader.py')).read()
exec(_c[:_c.index('\nbsents = ')])                                          # raw (local Wikipedia), docs (benchmark), G, TOK, SPL
P = lambda *a: print(*a, flush=True); T0 = time.time()


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


corpus = (raw.replace('\n', ' ') + ' ' + ' '.join(docs)).encode('latin-1', 'replace')
rsx = np.random.RandomState(0)
def rbatch(n):
    ix = rsx.randint(0, len(corpus) - L - 1, n); return torch.tensor([list(corpus[i:i + L]) for i in ix])
batch = lambda n, *a, **k: rbatch(n)                                         # the toy's trainers draw from real text now
lm = train_lm(LMS); P(f"   ({time.time() - T0:.0f}s)")
jp = train_jepa(KIND, JS); P(f"   ({time.time() - T0:.0f}s)")
# ---- per-word scores on the benchmark documents
names = set()
for e, nm in G['ent_name'].items():
    for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", nm): names.add(w)
STOP = {'the', 'of', 'and', 'in', 'a', 'an', 'to', 'for', 'on', 'at', 'by', 'de', 'la', 'le', 'The'}
rows = []                                                                     # (byte surprise, jepa surprise, capitalised-not-first, gold, visible)
with torch.no_grad():
    for d in docs[:int(os.environ.get('RN_DOCS', 262))]:
        b = d.encode('latin-1', 'replace')
        for st in range(0, max(1, len(b) - 16), 112):
            w = b[st:st + L]; t = w.decode('latin-1')
            if len(w) < 32: continue
            X = torch.tensor([list(w) + [PAD] * (L - len(w))]); Xi = torch.cat([torch.full((1, 1), PAD), X[:, :-1]], 1)
            lp = F.log_softmax(lm(Xi), -1)[0]; bs = -lp[torch.arange(L), X[0]] / math.log(2)
            tg = jp.target(X)[0]
            for a, e_, wd in spans(t):
                if a < 8 or e_ > len(w) - 4: continue                        # whole word with some context on both sides
                Xm = X.clone(); Xm[0, a:e_] = MASK; pr = jp.predict(Xm)[0]
                js = float(F.smooth_l1_loss(pr[a:e_], tg[a:e_]))
                first = a < 2 or t[max(0, a - 2):a].strip() in ('.', '!', '?', '"')
                gold = wd in names and wd not in STOP
                vis = wd[:1].isupper() and not first
                rows.append((float(bs[a:e_].mean()), js, float(vis), gold, vis))
R = np.array([r[:3] for r in rows]); g = np.array([r[3] for r in rows]); v = np.array([r[4] for r in rows])
z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
gap = z(R[:, 0]) - z(R[:, 1])
P(f"WORDS scored: {len(rows)} in the benchmark documents, gold-name words {int(g.sum())} ({g.mean():.2f}); of those NOT visible to the fast channel "
  f"(lower-case or sentence-initial) {int((g & ~v).sum())}  ({time.time() - T0:.0f}s)")
for tag, sel_pos, sel_neg in (('ALL WORDS', g, ~g), ('INVISIBLE gold-name words vs invisible ordinary words', g & ~v, ~g & ~v)):
    P(f"  {tag}: AUC byte surprise {auc(R[sel_pos, 0], R[sel_neg, 0]):.3f} | JEPA surprise (low = name) {auc(-R[sel_pos, 1], -R[sel_neg, 1]):.3f} | "
      f"GAP {auc(gap[sel_pos], gap[sel_neg]):.3f} | capitalised-not-first " + (f"{auc(R[sel_pos, 2], R[sel_neg, 2]):.3f}" if tag == "ALL WORDS" else "n/a (none capitalised by definition)"))
P("DONE")
