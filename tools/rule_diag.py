"""Why no real-text rule passes: every candidate of the relational module on the fast channel's real-text records (untyped frames, adjacent + centre records, no two-apart records) with its functional check outcome. Usage: python3 tools/rule_diag.py"""
import os, sys, collections, numpy as np
sys.argv=['x']; os.environ['RT_DIAG']='1'
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reader_tests.py')).read()
pre=src[:src.index('# ---------------- T1 REDUNDANCY')]
exec(compile(pre, 'rt', 'exec'))
allr=[(a,f,b) for a,f,b,pv,k in BR+WR if k in (0,1)]
fc=collections.Counter(f for _,f,_ in allr); top=set(f for f,_ in fc.most_common(60)); deg=collections.Counter()
for a,f,b in allr:
    if f in top: deg[a]+=1; deg[b]+=1
keep=set(n for n,_ in deg.most_common(4000)); pairs={}; nf=0
for a,f,b in allr:
    if f in top and a in keep and b in keep and a!=b and nf<20000: pairs.setdefault(f,set()).add((' '.join(a),' '.join(b))); nf+=1
R=_rm_mine(pairs,800,923)
D=R.get('cdiag',[])
print(f"untyped frames, adjacent + centre records only (no two-apart records): {nf} facts, {R['N']} fillers, {len(R['rels'])} frames | candidates {R['cand']} | rules {len(R['rules'])}", flush=True)
g=sum(1 for x in D if x[6]==-1); z=sum(1 for x in D if x[6]==0); lt3=sum(1 for x in D if 0<x[6]<3); ok=[x for x in D if x[6]>=3]
print(f"WHY CANDIDATES FAIL: chain-existence guard {g} | no subject with both an r1 and an r fact {z} | fewer than 3 testable subjects {lt3} | testable (3+) {len(ok)}: functional accuracy median {np.median([x[5] for x in ok]) if ok else float('nan'):.2f}, >= 0.5 {sum(1 for x in ok if x[5]>=0.5)}", flush=True)
for x in sorted(ok, key=lambda x:-x[5])[:12]: print(f"   '{x[0]}' o '{x[1]}' -> '{x[2]}' dist {x[3]:.2f} null {x[4]:.2f} | predicts {x[5]:.2f} of {x[6]}", flush=True)
print("RULES:", [r[:3]+(round(r[6],2), r[7]) for r in R['rules']][:10])
