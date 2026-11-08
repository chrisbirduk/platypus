"""OFFLINE RE-DOCRED READER TEST (no model): how well do the system's readers find entities, relation pairs and relations in real text?

The 262 Re-DocRED Wikipedia paragraphs embedded in the build are planted in ordinary text of their domain (local stand-in Pile files),
streamed as windows the way the store reads them (df = windows a word occurs in; rare = df <= inf_df), and stored as sentences with at
least two rare words. Readers compared, on the Re-DocRED sentences only (the only text with gold):
  BASE      the rule-free reader: a clause's two rarest words are the arguments, the 1-6 words between the relation
  ENT-MDL   entities by compression (inf_ent_mdl): a word is an argument only if treating it as a node saves bits on its records
  READER v2 / v3   the MDL reader (inf_reader 2 / 3): proposal schemes kept when their records pay for themselves (v3: core + modifiers)
Scores:
  ARGUMENT PRECISION  share of record arguments that are a token of some gold entity
  PAIR PRECISION      among records whose two arguments name two different gold entities: share that are a GOLD RELATION pair (and share
                      merely co-mentioned in the document); BASELINE = every pair of gold-entity tokens in the same clause
  PAIR RECALL         gold relation pairs whose entities share a sentence: share read by at least one record
  RELATION PURITY     records on gold relation pairs: for each relation phrase used >= 2 times, the share of its records carrying its most
                      common Wikidata property; and the normalised mutual information between phrase and property
usage: python3 redocred_reader_test.py PKG_DIR [filler_kb_per_domain ...]"""
import sys, os, math, collections, re
pkg = sys.argv[1]; sys.path.insert(0, pkg)
FILL = [int(x) for x in sys.argv[2:]] or [400, 2000]
sys.path.insert(0, '/tmp/claude-0/sweep'); os.environ.setdefault('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile')
import threshold_sweep_local as SL
import platypus
from platypus.inference.probes import _bench
from platypus.inference import units as U


class C:  # the options the readers read (GPU defaults)
    inf_bench = 'embedded'; inf_bench_maxamb = 1; inf_df = 40; inf_reader_n = 20000; inf_reader_dmin = 4; inf_reader_cost = 16.0
    inf_reader_recur = 0; inf_reader_newrel = 4.0; inf_reader_kmod = 3; inf_ent_n = 20000; inf_ent_cost = 16.0; inf_ent_bpr = 0.0; inf_reader = 0


G = _bench(C()); DOMS = SL.PILE_DOMAINS if hasattr(SL, 'PILE_DOMAINS') else platypus.config.PILE_DOMAINS
tok2e = collections.defaultdict(set)                                            # every token of every gold entity (argument precision)
for e, ts in G['ent_tok'].items():
    for t in ts: tok2e[t.encode('utf-8')].add(int(e))
t2e = {t.encode('utf-8'): es for t, es in G['_t2e'].items()}                       # distinctive tokens: one entity (pair scoring)
REL = {}
for a, b, _, ps in G['rel']: REL.setdefault((min(a, b), max(a, b)), set()).update(ps)
COM = G['_com']
WIN = 1025                                                                        # GPU seq_len + 1


def run(fill_kb):
    U._UNITS.clear(); U._UNITS.update(lex=set(), all=True, df={}, mx=C.inf_df, rdem=False, slotc={})
    U._RDR.clear(); U._RDR.update(on=False, acc=set(), gain={}, drop=set())
    df = U._UNITS['df']; items = []; nwin = [0]
    for d, docs in sorted(G['docs'].items()):
        fill = SL.minipile_bytes(DOMS[int(d)], 1000 + int(d), fill_kb * 1000)[:fill_kb * 1000]
        db = [t.encode('utf-8') for t in docs]
        cuts = sorted([m.end() for m in re.finditer(rb'[.!?] (?=[A-Z])', fill)][::max(1, len(fill) // 4000 // max(1, len(db)))][:len(db)])
        stream = bytearray(); j = 0; marks = []
        for c, x in zip(cuts, db): stream += fill[j:c]; marks.append((len(stream), len(stream) + len(x))); stream += x + b' '; j = c
        stream += fill[j:]
        wins = [(bytes(stream[i:i + WIN]), i) for i in range(0, len(stream) - 64, WIN)]; nwin[0] += len(wins)
        for w, _ in wins:
            for n in set(U._UWORD.findall(w)): df[n] = df.get(n, 0) + 1
        dcnt = collections.Counter(t.lower() for t in U._UWORD.findall(bytes(stream)))   # the visit: case-insensitive counts in the domain's stream
        for w, off in wins:                                                       # store: sentences with >= 2 rare words (cut pieces dropped)
            ss = U._INF_SENT.split(w); pos = 0; wcnt = collections.Counter(t.lower() for t in U._UWORD.findall(w)); wcnt['__visit__'] = dcnt
            for k, s in enumerate(ss):
                p0 = w.find(s, pos); pos = p0 + len(s)
                if k == 0 or k == len(ss) - 1: continue
                if len(s) > 300: continue
                ns = [m.group(0) for m in U._UWORD.finditer(s) if U._u_rare(m.group(0))]
                if len(set(ns)) < 2: continue
                g = any(a <= off + p0 < b for a, b in marks)
                items.append({'s': s, 'd': int(d), 'k': 'real', 'g': g, 'wc': wcnt})
    S = {'it': items, 'facts': [], 'conc': []}
    rows = {}
    rows['BASE'] = [(it, list(U._UnitTrip()._base(it['s']))) for it in items if it['g']]
    rare0 = U._u_rare; CUR = {}                                                  # ENTITY FILTERS on top of rarity (the base reader with the argument test replaced)
    filt = {'+RECUR (window)': lambda w: CUR['it']['wc'][w.lower()] >= 2,
            '+RECUR (visit)': lambda w: CUR['it']['wc']['__visit__'][w.lower()] >= 2,
            '+CAPITAL': lambda w: w[:1].isupper(),
            '+RECUR+CAP': lambda w: w[:1].isupper() and CUR['it']['wc'][w.lower()] >= 2}
    for fn, f in filt.items():
        U._u_rare = (lambda f_: (lambda w: rare0(w) and f_(w)))(f); out = []
        for it in items:
            if it['g']: CUR['it'] = it; out.append((it, list(U._UnitTrip()._base(it['s']))))
        rows['BASE ' + fn] = out
    U._u_rare = rare0
    U._ent_update(C, S)
    rows['ENT-MDL'] = [(it, list(U._UnitTrip()._base(it['s']))) for it in items if it['g']]
    U._UNITS['ent_on'] = False
    for v in (2, 3):
        C.inf_reader = v; U._RDR.clear(); U._RDR.update(on=True, acc=set(), gain={}, drop=set(), v2=v >= 2)
        info = U._rdr_update(C, S)
        rows[f'READER v{v}'] = [(it, list(U._UnitTrip().finditer(it['s']))) for it in items if it['g']]
    U._RDR['on'] = False
    ng = sum(1 for it in items if it['g'])
    print(f"\n=== filler {fill_kb} KB per domain: {len(items)} stored sentences, {ng} from Re-DocRED paragraphs | rare = in <= {C.inf_df} of "
          f"{nwin[0]} windows; distinct words {len(df)}, rare {sum(1 for v in df.values() if v <= C.inf_df)}")
    # baseline: all gold-entity token pairs within a clause
    bp = []; cooc = set()
    for it in items:
        if not it['g']: continue
        for cl in re.finditer(rb'[^.,;:!?()"]+', it['s']):
            es = sorted(set(next(iter(t2e[w])) for w in U._UWORD.findall(cl.group(0)) if w in t2e))
            for i in range(len(es)):
                for j in range(i + 1, len(es)): bp.append((es[i], es[j]))
        es_s = sorted(set(next(iter(t2e[w])) for w in U._UWORD.findall(it['s']) if w in t2e))
        cooc |= {(es_s[i], es_s[j]) for i in range(len(es_s)) for j in range(i + 1, len(es_s)) if (es_s[i], es_s[j]) in REL}
    print(f"  BASELINE (every pair of gold-entity tokens in one clause): {len(bp)} pairs, gold relation {sum(p in REL for p in bp) / max(1, len(bp)):.3f}, "
          f"co-mentioned {sum(p in COM for p in bp) / max(1, len(bp)):.3f} | gold relation pairs sharing a stored sentence: {len(cooc)}")
    PN = {'P17': 'country', 'P131': 'located in', 'P27': 'citizenship', 'P150': 'contains', 'P577': 'publication date', 'P569': 'birth date', 'P570': 'death date',
          'P108': 'employer', 'P175': 'performer', 'P527': 'has part', 'P361': 'part of', 'P571': 'inception', 'P495': 'country of origin', 'P19': 'birthplace',
          'P161': 'cast member', 'P102': 'party', 'P54': 'team', 'P463': 'member of', 'P264': 'record label', 'P39': 'position', 'P50': 'author', 'P57': 'director'}
    byp = collections.defaultdict(set)
    for pr_ in cooc:
        for q in REL[pr_]: byp[q].add(pr_)
    top = sorted(byp, key=lambda q: -len(byp[q]))[:8]
    for nm, R in rows.items():
        na = ng_ = 0; pairs = []; read = set(); phr = []
        for it, ms in R:
            for m in ms:
                a, r, c = m.group(1), m.group(2), m.group(3)
                na += 2; ng_ += (a in tok2e) + (c in tok2e)
                if a in t2e and c in t2e:
                    ea, ec = next(iter(t2e[a])), next(iter(t2e[c]))
                    if ea != ec:
                        p = (min(ea, ec), max(ea, ec)); pairs.append(p)
                        if p in REL: read.add(p); phr.append((r, sorted(REL[p])[0]))
        cnt = collections.Counter(r for r, _ in phr); pur = []
        for r, n in cnt.items():
            if n >= 2: pur.append((n, collections.Counter(q for r2, q in phr if r2 == r).most_common(1)[0][1]))
        nmi = float('nan')
        if phr:
            N = len(phr); pr = collections.Counter(r for r, _ in phr); pq = collections.Counter(q for _, q in phr); pj = collections.Counter(phr)
            I = sum(c / N * math.log(c * N / (pr[r] * pq[q])) for (r, q), c in pj.items()); Hr = -sum(c / N * math.log(c / N) for c in pr.values()); Hq = -sum(c / N * math.log(c / N) for c in pq.values())
            nmi = I / max(1e-9, math.sqrt(Hr * Hq))
        print(f"  {nm:10s} records {sum(len(ms) for _, ms in R):5d} | ARGUMENT PRECISION {ng_ / max(1, na):.3f} | PAIR PRECISION gold relation "
              f"{sum(p in REL for p in pairs) / max(1, len(pairs)):.3f}, co-mentioned {sum(p in COM for p in pairs) / max(1, len(pairs)):.3f} (n {len(pairs)}) | "
              f"PAIR RECALL {len(read & cooc) / max(1, len(cooc)):.3f} | RELATION PURITY {sum(b for _, b in pur) / max(1, sum(a for a, _ in pur)):.3f} "
              f"over {len(pur)} phrases used >= 2 times, NMI {nmi:.3f}")
        print("      recall by relation: " + " | ".join(f"{PN.get(q, q)} {len(read & byp[q])}/{len(byp[q])}" for q in top))
        if nm == 'BASE':
            ex = [(m.group(1).decode('latin-1', 'replace'), m.group(2).decode('latin-1', 'replace'), m.group(3).decode('latin-1', 'replace')) for _, ms in R[:40] for m in ms][:8]
            print("    e.g. " + " | ".join(f"{a} [{r}] {c}" for a, r, c in ex))


for f in FILL: run(f)
