"""REFERENCE CEILING (12 Nov 2026; tools/reference_ceiling.py) - how far can reference (who 'he' / 'it' / 'the company' is) lift the ~7%
one-sentence ceiling for views on real text? On the Re-DocRED benchmark documents, gold relations whose two names are found (fast-channel
fillers) in ONE sentence, when a sentence is also taken to mention:
  none           only the names it contains (the ceiling so far)
  title/pronoun  + the document's title (its first filler) when it has a third-person pronoun or opens with frame material
  title/all      + the title in every sentence (the document is about it)
  last subject   + the previous sentence's subject filler when it has a pronoun (centering: the latest centre)
  both           title/pronoun + last subject
and the upper bound: both names anywhere in the same document. Also: the share of gold relations with the title as one argument.
Usage: python3 tools/reference_ceiling.py"""
import os
import collections
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, 'chunk_reader.py')).read()
exec(_src[:_src.index('\nbsents = ')])
P = lambda *a: print(*a, flush=True)
PRON = {'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their', 'him', 'them', 'He', 'She', 'It', 'They', 'His', 'Her', 'Its', 'Their'}
bdocs = [[TOK.findall(s) for s in SPL.split(d)] for d in docs]
modes = ('none', 'title/pronoun', 'title/all', 'last subject', 'both')
cov = {m: set() for m in modes}; doc_cov = set(); title_rel = set()
for d in bdocs:
    fill = [[tuple(t[a:b]) for a, b in segment(t, 300, 'B')] for t in d]
    title = next((f[0] for f in fill if f), None)
    T = ents_of(title) if title else set(); allE = set()
    for f in fill:
        for x in f: allE |= ents_of(x)
    last = set()
    for t, f in zip(d, fill):
        E = set()
        for x in f: E |= ents_of(x)
        pron = any(w in PRON for w in t) or (bool(f) and t and tuple(t[:len(f[0])]) != f[0])
        sets = {'none': E, 'title/pronoun': E | (T if pron else set()), 'title/all': E | T, 'last subject': E | (last if pron else set()),
                'both': E | (T if pron else set()) | (last if pron else set())}
        for m, S in sets.items():
            for x in S:
                for y in S:
                    if x < y and (x, y) in gold: cov[m].add((x, y))
        if f and t and tuple(t[:len(f[0])]) == f[0]: last = ents_of(f[0])
    for x in allE:
        for y in allE:
            if x < y and (x, y) in gold:
                doc_cov.add((x, y))
                if x in T or y in T: title_rel.add((x, y))
for m in modes: P(f"  one sentence, reference = {m:14s}: {len(cov[m])} of {len(gold)} gold relations ({len(cov[m]) / len(gold):.3f})")
P(f"  UPPER BOUND, both names anywhere in the document: {len(doc_cov)} ({len(doc_cov) / len(gold):.3f}) | of those with the TITLE as one argument "
  f"{len(title_rel)} ({len(title_rel) / max(1, len(doc_cov)):.2f})")
P("DONE")
