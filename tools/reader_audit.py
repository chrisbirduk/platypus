"""READER AUDIT against the embedded real-text benchmark (Re-DocRED Wikipedia paragraphs, gold relations; 11 Nov 2026): runs the reader's
rule-free extraction (two rarest words of a clause, the words between them the relation) on the benchmark sentences and scores the records
against the gold relation pairs, stage by stage. Window frequencies are estimated from English word frequencies (needs: pip install wordfreq).
Usage: python3 tools/reader_audit.py [windows read, default 12800]"""
import sys, re, collections
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from wordfreq import word_frequency
from platypus.inference import probes, units
from platypus.inference.namemem import _NM_REL, _nm_isname
import platypus.config as C
cfg = C.Config(); cfg.inf_bench = 'embedded'; G = probes._bench(cfg)
NWIN = int(sys.argv[1]) if len(sys.argv) > 1 else 12800
# window frequency estimate: a 1024-byte window holds ~170 words
docs = [d for k in sorted(G['docs']) for d in G['docs'][k]]
dfdoc = collections.Counter(w for d in docs for w in set(units._UWORD.findall(d.encode())))
df = {}
def dfw(w):
    if w not in df: df[w] = NWIN * 170 * word_frequency(w.decode('latin-1').lower(), 'en') + dfdoc.get(w, 0)
    return df[w]
class DF(dict):
    def get(self, k, d=0): return dfw(k)
units._UNITS.update(all=True, df=DF(), mx=40, rdem=False, slotc={}); units._RDR['on'] = False
U = units._UnitTrip(); N = units._UnitName()
t2e = G['_t2e']; gold = {}; 
for h, t, fl, ps in G['rel']: gold.setdefault((min(h, t), max(h, t)), ps)
ents_of = lambda w: t2e.get(w.decode('latin-1'), set())
st = collections.Counter(); recs_all = []; recs_first = []
for d in docs:
    for s in units._INF_SENT.split(d.encode('utf-8', 'replace')):
        st['sentences'] += 1
        ns = N.findall(s)
        if len(set(ns)) < 2 or len(s) > 300: st['not stored (<2 units or >300 bytes)'] += 1; continue
        st['stored'] += 1
        ms = list(U.finditer(s)); st['clauses giving a triple'] += len(ms)
        recs_all += [(m.group(1), m.group(2), m.group(3)) for m in ms]
        m = U.search(s)
        if m: recs_first.append((m.group(1), m.group(2), m.group(3)))
def score(recs, tag):
    pairs = set(); inb = 0; hit = 0; samples = []
    for a, r, b in recs:
        ea, eb = ents_of(a), ents_of(b)
        if ea and eb:
            inb += 1; ok = any((min(x, y), max(x, y)) in gold for x in ea for y in eb if x != y)
            hit += ok
            for x in ea:
                for y in eb:
                    if (min(x, y), max(x, y)) in gold: pairs.add((min(x, y), max(x, y)))
            if len(samples) < 6: samples.append((a.decode(), r.decode('latin-1'), b.decode(), ok))
    print(f"{tag}: {len(recs)} records | both arguments benchmark entities {inb} | of those a GOLD relation pair {hit} ({hit / max(1, inb):.2f}) | gold pairs covered {len(pairs)} of {len(gold)} ({len(pairs) / len(gold):.3f})")
    return samples
print('stage counts', dict(st))
score(recs_all, 'A every clause (finditer)')
smp = score(recs_first, 'B first triple per sentence (what the evidence record keeps)')
filt = [(a, r, b) for a, r, b in recs_first if _nm_isname(a) and _nm_isname(b) and _NM_REL.match(r.decode('latin-1', 'replace').strip())]
score(filt, 'C + the name-memory / distillation filters (capitalised one-word names, 1-4 lower-case relation words)')
print('C would still need the fact SEEN TWICE (benchmark documents are planted once per run)')
print('examples of B:', smp)
import random; random.seed(1)
print('random B records:', [(a.decode('latin-1'), r.decode('latin-1'), b.decode('latin-1')) for a, r, b in random.sample(recs_first, 12)])
# what gold looks like: rarity of gold entity names
rare = sum(1 for e, nm in G['ent_name'].items() if all(dfw(w.encode()) > 40 for w in nm.split() if w[:1].isupper()))
print(f"benchmark entities whose every capitalised word is too FREQUENT to be an argument (df > 40): {rare} of {len(G['ent_name'])}")
multi = sum(1 for nm in G['ent_name'].values() if len(nm.split()) > 1); print(f"multi-word entity names: {multi} of {len(G['ent_name'])}")
# ceiling: gold pairs whose two entities are named in ONE sentence (by distinctive tokens)
co = set()
for d in docs:
    for s in units._INF_SENT.split(d.encode('utf-8', 'replace')):
        es = set()
        for w in units._UWORD.findall(s): es |= ents_of(w)
        for x in es:
            for y in es:
                if x < y and (x, y) in gold: co.add((x, y))
print(f"CEILING: gold pairs with both entities named in one sentence {len(co)} of {len(gold)} ({len(co) / len(gold):.2f}) - the rest need coreference or cross-sentence reading")
import collections as cc
P = cc.Counter(p for ps in gold.values() for p in ps); print('commonest gold relation types (Wikidata):', P.most_common(10))
