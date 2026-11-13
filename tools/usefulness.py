"""USEFULNESS AS THE TEACHER (13 Nov 2026; tools/usefulness.py) - offline, no effect on the package.
Mattar & Daw 2018 (replay priority = gain x need) and RIME (Xu et al. 2026: a memory writer trained on the improvement it brings to LATER
tasks): judge what is extracted by its later benefit, policy by policy, not by each record's compression. Held-out Re-DocRED documents
(tools/consistency_teacher.py split); a record's LATER BENEFIT = bits it saves the stand-in trunk on the second name where the two names meet
again in another document (as tools/extract_gain.py), raw and beyond what the two names alone save. Each extraction POLICY keeps a quarter of
the held-out candidates; reported: later benefit per record kept (mean bits saved), related share, and benefit of related vs unrelated kept.
Policies: random, closeness, consistency-teacher learner, trunk-feedback learner (tools/extract_combo.py's), and the two combined. Then:
a learner trained to predict LATER BENEFIT itself (label-free) - does optimising usefulness keep real relations?
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/usefulness.py"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_co = open(os.path.join(HERE, 'extract_combo.py')).read()
exec(compile(_co[:_co.index('base_t = rk(a2)')], 'extract_combo', 'exec'))
# later benefit of held-out records (their names meet again in another document, any split)
occ2 = collections.defaultdict(list)
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms)
        for x in range(len(ms)):
            for y_ in range(x + 1, len(ms)):
                if ms[x][2] != ms[y_][2] and ms[x][1] <= ms[y_][0] and cn[ms[x][2]] != cn[ms[y_][2]]:
                    occ2[tuple(sorted((cn[ms[x][2]], cn[ms[y_][2]])))].append((di, si, ms[y_][0], ms[y_][1]))
rq = np.random.RandomState(3); HB = []; SQ = {k: [] for k in ('w', 'o', 'pw', 'po')}
for k_, i in enumerate(TE):
    c = C[i]; oth = [o for o in occ2[tuple(sorted((c['cA'], c['cB'])))] if o[0] != c['di']]
    if not oth: continue
    di2, si2, y0, y1 = oth[rq.randint(len(oth))]; t2 = docs[di2]['sents'][si2]; pre = txt(t2[max(0, y0 - 6):y0])[-36:] + b' '; tg = txt(t2[y0:y1])[:24]
    A, B, M = txt(c['A']), txt(c['B']), txt(c['mid']); HB.append(k_)
    for k, rec in (('w', A + b' ' + M + b' ' + B), ('o', TH + b' ' + M + b' ' + B), ('pw', A + b' ' + B), ('po', TH + b' ' + B)):
        q = rec.replace(b'  ', b' ') + b' . ' + pre; SQ[k].append((q + tg, (len(q), len(q) + len(tg))))
t1 = time.time(); BH = {k: bits(v) for k, v in SQ.items()}; gr = BH['o'] - BH['w']; gb = gr - (BH['po'] - BH['pw'])
HB = np.array(HB); yh = yt[HB]
P(f"held-out records with a later meeting of their names: {len(HB)} of {len(TE)} (related {yh.mean():.3f}) ({time.time() - t1:.0f}s) | later benefit, "
  f"related vs not: raw {gr[yh].mean():+.3f} vs {gr[~yh].mean():+.3f} bits, beyond the names {gb[yh].mean():+.3f} vs {gb[~yh].mean():+.3f}")
lab_c = rk(a2) + rk(a3) + 2 * a1 + 2 * a4
s_cons = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, lab_c >= np.percentile(lab_c, 75)).decision_function(Xte)
sel = np.array(FBi); s_trk = Ridge(alpha=10.0).fit(Xtr[sel], np.clip(fb, np.percentile(fb, 1), np.percentile(fb, 99))).predict(Xte)
TRR = np.full(len(TR), np.nan); TRR[FBi] = Bt['o'] - Bt['w']                    # raw later benefit on training records (label-free)
s_use = Ridge(alpha=10.0).fit(Xtr[sel], np.clip(TRR[FBi], np.percentile(TRR[FBi], 1), np.percentile(TRR[FBi], 99))).predict(Xte)
close = -np.array([len(C[i]['mid']) for i in TE], float)
def pol(tag, sc):
    s = sc[HB]; o = np.argsort(-s + 1e-9 * rq.rand(len(s))); keep = o[:len(s) // 4]
    P(f"   {tag:48s} kept: related {yh[keep].mean():.3f} | LATER BENEFIT per record kept: raw {gr[keep].mean():+.3f}, beyond the names {gb[keep].mean():+.3f} bits")
P("POLICIES (each keeps a quarter):")
pol('random', rq.rand(len(TE))); pol('closeness', close); pol('consistency-teacher learner', s_cons); pol('trunk-feedback learner', s_trk)
pol('consistency + trunk (rank sum)', rk(s_cons) + rk(s_trk)); pol('LEARNED FROM LATER BENEFIT (raw, label-free)', s_use)
P(f"   all held-out records: related {yh.mean():.3f}, later benefit raw {gr.mean():+.3f}, beyond the names {gb.mean():+.3f}")
P(f"DONE ({time.time() - T0:.0f}s)")
