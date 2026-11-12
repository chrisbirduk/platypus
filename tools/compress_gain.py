"""COMPRESSION GAIN: DO PATTERNS THE STORE COULD PROPOSE COMPRESS TEXT FOR THE TRUNK? (12 Nov 2026; tools/compress_gain.py) - offline readout.
The minimum-description-length version of 'the trunk guides the hippocampus' (Rissanen; de Marcken 1996; Schmidhuber's compression progress;
Spens & Burgess 2024): a pattern is real when the trunk needs fewer bits for the text once the pattern is in place. Same documents, stand-in
trunk and untrained control as tools/trunk_guides.py (which runs first, unchanged; set TG_CKPT to reuse its trained model). Gold: the Re-DocRED
dev mention annotations. Every gain is in bits: cost of the target WITHOUT the pattern minus cost WITH it ('Thing' stands in for a removed name).
  (a) UNITS - BINDING GAIN at each join inside a run of capitalised words: bits the trunk saves on the second word when the real first word
      precedes it instead of 'Thing' ('New' -> 'York'). Same mention vs mention boundary: AUC, against the plain surprise of trunk_guides
      (0.565); exact-span F1 when runs are split where the gain is below a threshold (fitted on documents 1-250, scored on 251-500).
  (b) LINKS - MUTUAL COMPRESSION GAIN of an earlier mention for a later one: bits saved on the later mention's bytes when a snippet around the
      earlier mention precedes its sentence, against the same snippet with the earlier mention replaced by 'Thing'. Candidates: the 12 nearest
      earlier mentions. Same entity vs not: AUC on all, HARD (no shared word) and VERY HARD (+ few shared letter triples) pairs; antecedent right;
      logistic combinations with the string features (fitted on documents 1-250). Control: the same gains under the untrained network.
Usage: TG_CKPT=<model file> python3 tools/compress_gain.py [lm steps] [documents]"""
import os
import re
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'trunk_guides.py')).read()
exec(compile(_src, 'trunk_guides', 'exec'))                                  # DOCS, lm, lm0, auc, runs_of, cap, toks_of, c3, initials, half, P, T0
P(f"\n================ COMPRESSION GAIN  ({time.time() - T0:.0f}s)")
TH = b'Thing'


@torch.no_grad()
def bits(m, seqs, spans, bs=256):
    """total bits the model spends on bytes [a, b) of each sequence (read from its start; PAD as the start symbol)"""
    out = np.zeros(len(seqs))
    for i in range(0, len(seqs), bs):
        ch = [s_[-L:] for s_ in seqs[i:i + bs]]; sh = [len(s_) - len(c_) for s_, c_ in zip(seqs[i:i + bs], ch)]
        X = torch.full((len(ch), L), PAD, dtype=torch.long)
        for j, c_ in enumerate(ch): X[j, :len(c_)] = torch.tensor(list(c_))
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(m(Xi), -1)
        nl = -lp.gather(-1, X.clamp(max=V - 1)[..., None])[..., 0] / math.log(2)
        for j, ((a, b), s0) in enumerate(zip(spans[i:i + bs], sh)):
            a, b = max(0, a - s0), max(0, b - s0); out[i + j] = float(nl[j, a:b].sum())
    return out


def sent_bounds(D):
    sb = {}
    for si, ti, a, b, w in D['toks']: sb.setdefault(si, [a, b]); sb[si][1] = b
    return sb
# ---------------------------------------------------------------- (a) units: binding gain
JW, JO, JY, JD = [], [], [], []                                              # with / without sequences, label, doc
for di, D in enumerate(DOCS):
    tk = D['toks']; b_ = D['b']; mk = {}; starts = set(m['k0'] for m in D['ments']); sb = sent_bounds(D)
    for i, m in enumerate(D['ments']):
        for k in range(m['k0'], m['k1'] + 1): mk.setdefault(k, i)
    for a0, b0 in runs_of(D):
        for k in range(a0 + 1, b0 + 1):
            if not cap(tk[k][4]): continue
            si, _, wa, wb, w = tk[k]; pa, pb = tk[k - 1][2], tk[k - 1][3]; c0 = max(sb[si][0], pa - 60)
            pre = b_[c0:pa]; mid = b_[pb:wa]; tgt = b_[wa:wb]
            JW.append((pre + b_[pa:pb] + mid + tgt, len(pre) + (pb - pa) + len(mid))); JO.append((pre + TH + mid + tgt, len(pre) + len(TH) + len(mid)))
            JY.append(k in mk and (k - 1) in mk and mk[k] == mk[k - 1] and k not in starts); JD.append(di)
JY = np.array(JY); JD = np.array(JD)
def gain_a(m):
    bw = bits(m, [s_ for s_, p_ in JW], [(p_, len(s_)) for s_, p_ in JW]); bo = bits(m, [s_ for s_, p_ in JO], [(p_, len(s_)) for s_, p_ in JO]); return bo - bw, bw
ga, bw_a = gain_a(lm); ga0, _ = gain_a(lm0)
P(f"(a) UNITS: {len(JY)} joins inside capitalised runs, {int(JY.sum())} inside one mention | AUC same mention vs boundary: BINDING GAIN {auc(ga[JY], ga[~JY]):.3f} "
  f"(mean {ga[JY].mean():.2f} vs {ga[~JY].mean():.2f} bits) | plain surprise of the second word {auc(-bw_a[JY], -bw_a[~JY]):.3f} | untrained network's gain {auc(ga0[JY], ga0[~JY]):.3f}")
jix = 0; GJ = {}
for di, D in enumerate(DOCS):                                                # join gains by (doc, token) in the same order as built
    tk = D['toks']
    for a0, b0 in runs_of(D):
        for k in range(a0 + 1, b0 + 1):
            if not cap(tk[k][4]): continue
            GJ[(di, k)] = ga[jix]; jix += 1
def spans_split(D, di, th):
    out = []
    for a0, b0 in runs_of(D):
        s0 = a0
        for k in range(a0 + 1, b0 + 1):
            if (di, k) in GJ and GJ[(di, k)] < th: out.append((s0, k - 1)); s0 = k
        out.append((s0, b0))
    return out
def prf2(idx, th):
    tp = npred = ngold = 0
    for di in idx:
        D = DOCS[di]; g = {(m['k0'], m['k1']) for m in D['ments']}; p = set(spans_split(D, di, th)); tp += len(g & p); npred += len(p); ngold += len(g)
    pr, rc = tp / max(1, npred), tp / max(1, ngold); return pr, rc, 2 * pr * rc / max(1e-9, pr + rc)
best = max((prf2(range(half), th)[2], th) for th in [-99.0] + list(np.arange(-4.0, 8.01, 0.5)))
p0, r0, f0 = prf2(range(half, len(DOCS)), -99.0); p1, r1, f1 = prf2(range(half, len(DOCS)), best[1])
P(f"    exact mention spans (documents 251-500): capitalised runs P {p0:.3f} R {r0:.3f} F1 {f0:.3f} | split where the binding gain < {best[1]:.1f} bits (fitted on 1-250) "
  f"P {p1:.3f} R {r1:.3f} F1 {f1:.3f}")
# ---------------------------------------------------------------- (b) links: mutual compression gain
PW, PO, PY, PF, PK = [], [], [], [], []
for di, D in enumerate(DOCS):
    M = D['ments']; b_ = D['b']; sb = sent_bounds(D)
    for j, mj in enumerate(M):
        cands = [i for i in range(j) if M[i]['b'] <= mj['a']][-12:]
        for i in cands:
            mi = M[i]; s_i, e_i = max(sb[mi['sid']][0], mi['a'] - 24), min(sb[mi['sid']][1], mi['b'] + 24); s_j = max(sb[mj['sid']][0], mj['a'] - 48)
            tgt = b_[mj['a']:mj['b']][:40]
            if mi['sid'] == mj['sid'] or e_i > s_j:                          # same sentence / overlapping snippets: one span of text
                s0 = min(s_i, s_j); w_ = b_[s0:mj['a']]; o_ = b_[s0:mi['a']] + TH + b_[mi['b']:mj['a']]
            else:
                w_ = b_[s_i:e_i] + b' ' + b_[s_j:mj['a']]; o_ = b_[s_i:mi['a']] + TH + b_[mi['b']:e_i] + b' ' + b_[s_j:mj['a']]
            PW.append((w_ + tgt, len(w_))); PO.append((o_ + tgt, len(o_))); PY.append(mi['e'] == mj['e']); PK.append((di, j))
            ta, tb = set(toks_of(mi['name'])), set(toks_of(mj['name'])); ca_, cb_ = c3(mi['name']), c3(mj['name'])
            ia, ib = initials(mi['name']), initials(mj['name'])
            PF.append([len(ta & tb) / max(1, len(ta | tb)), float(mi['name'].lower() == mj['name'].lower()),
                       float(bool(ta) and bool(tb) and toks_of(mi['name'])[-1:] == toks_of(mj['name'])[-1:]),
                       float(len(ia) > 1 and (ia == re.sub(r'[^a-z]', '', mj['name'].lower()) or ib == re.sub(r'[^a-z]', '', mi['name'].lower()))),
                       len(ca_ & cb_) / max(1, len(ca_ | cb_)), float(len(ta & tb) == 0)])
PY = np.array(PY); PF = np.array(PF); PD = np.array([k[0] for k in PK]); te2 = PD >= half; hard2 = PF[:, 5] == 1; vh2 = hard2 & (PF[:, 4] < 0.25)
P(f"(b) LINKS: {len(PY)} candidate pairs (12 nearest earlier mentions), {int(PY.sum())} the same entity, {int((PY & hard2).sum())} of those share no word, "
  f"{int((PY & vh2).sum())} very hard  ({time.time() - T0:.0f}s)")
def gain_b(m):
    bw = bits(m, [s_ for s_, p_ in PW], [(p_, len(s_)) for s_, p_ in PW]); bo = bits(m, [s_ for s_, p_ in PO], [(p_, len(s_)) for s_, p_ in PO]); return bo - bw
gb = gain_b(lm); gb0 = gain_b(lm0)
P(f"    raw gain (no fitting), documents 251-500 | trained: all {auc(gb[te2 & PY], gb[te2 & ~PY]):.3f}, HARD {auc(gb[te2 & PY & hard2], gb[te2 & ~PY & hard2]):.3f}, "
  f"VERY HARD {auc(gb[te2 & PY & vh2], gb[te2 & ~PY & vh2]):.3f} (mean gain same {gb[PY].mean():.2f} vs different {gb[~PY].mean():.2f} bits) | untrained: all "
  f"{auc(gb0[te2 & PY], gb0[te2 & ~PY]):.3f}, HARD {auc(gb0[te2 & PY & hard2], gb0[te2 & ~PY & hard2]):.3f}, VERY HARD {auc(gb0[te2 & PY & vh2], gb0[te2 & ~PY & vh2]):.3f}")
def fitpred(Xc):
    X = torch.tensor(Xc[~te2], dtype=torch.float32); Y = torch.tensor(PY[~te2], dtype=torch.float32); mu_, sd_ = X.mean(0), X.std(0) + 1e-6
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True); op = torch.optim.LBFGS([w, b], max_iter=200); pw = float((1 - Y.mean()) / max(1e-6, Y.mean()))
    def cl():
        op.zero_grad(); l = F.binary_cross_entropy_with_logits(((X - mu_) / sd_) @ w + b, Y, pos_weight=torch.tensor(pw)); l.backward(); return l
    op.step(cl); return ((torch.tensor(Xc, dtype=torch.float32) - mu_) / sd_ @ w + b).detach().numpy()
def antec(sc):
    best = {}
    for k in np.where(te2)[0]:
        if PK[k] not in best or sc[k] > best[PK[k]][0]: best[PK[k]] = (sc[k], bool(PY[k]))
    has = {PK[k] for k in np.where(te2 & PY)[0]}; return (np.mean([best[k][1] for k in has]) if has else float('nan')), len(has)
for nm_, Xc in (('string', PF[:, :5]), ('compression gain', np.stack([gb], 1)), ('untrained gain', np.stack([gb0], 1)),
                ('string + gain', np.concatenate([PF[:, :5], gb[:, None]], 1)), ('string + untrained gain', np.concatenate([PF[:, :5], gb0[:, None]], 1))):
    s_ = fitpred(Xc); acc, n_ = antec(s_)
    P(f"    {nm_:24s} AUC {auc(s_[te2 & PY], s_[te2 & ~PY]):.3f} | HARD {auc(s_[te2 & PY & hard2], s_[te2 & ~PY & hard2]):.3f} | VERY HARD "
      f"{auc(s_[te2 & PY & vh2], s_[te2 & ~PY & vh2]):.3f} | antecedent right {acc:.3f} ({n_})")
top = np.where(PY & vh2)[0]; top = top[np.argsort(-gb[top])][:6]
P("    largest gains on VERY HARD same-entity pairs: " + ", ".join(f"{PW[k][0][max(0, PW[k][1] - 30):PW[k][1]].decode('utf-8', 'replace')!r} -> "
                                                               f"{PW[k][0][PW[k][1]:].decode('utf-8', 'replace')!r} {gb[k]:+.1f}" for k in top))
P(f"DONE ({time.time() - T0:.0f}s)")
