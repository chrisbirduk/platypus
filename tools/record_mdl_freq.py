"""Purity of name classes (tools/record_mdl.py RM_OUT files) by how many store records a name has (1-2 / 3-9 / 10+). 13 Nov 2026.
Usage: python3 tools/record_mdl_freq.py <record_mdl json> ..."""
import json, collections, numpy as np, sys
from sklearn.metrics import normalized_mutual_info_score
S='/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
docs=json.load(open(S+'/bench_dl/train_revised.json'))+json.load(open(S+'/bench_dl/dev_revised.json'))
gtc=collections.defaultdict(collections.Counter); occ=collections.Counter()
for d in docs:
    cn=[collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    bys=collections.defaultdict(list)
    for ei,vs in enumerate(d['vertexSet']):
        for m in vs: gtc[cn[ei]][m.get('type','X')]+=1; bys[m['sent_id']].append((m['pos'][0],m['pos'][1],ei))
    for si,ms in bys.items():
        ms=sorted(ms); toks=d['sents'][si]
        for x in range(len(ms)):
            for y in range(x+1,len(ms)):
                (a0,a1),(b0,b1)=ms[x][:2],ms[y][:2]; ea,eb=ms[x][2],ms[y][2]
                if ea==eb or a1>b0 or cn[ea]==cn[eb]: continue
                if b0-a1<=12 and len(' '.join(toks[a0:b1]))<=60: occ[cn[ea]]+=1; occ[cn[eb]]+=1
for f in sys.argv[1:]:
    J=json.load(open(f)); ks=J['k']; T=J['types']; names=[n for n in T if n in occ]
    o=np.array([occ[n] for n in names]); gt=np.array([gtc[n].most_common(1)[0][0] for n in names])
    print(f.split('/')[-1], 'names by records: 1-2', (o<=2).sum(), '| 3-9', ((o>=3)&(o<10)).sum(), '| 10+', (o>=10).sum(), '| 10+ hold', round(o[o>=10].sum()/o.sum(),2), 'of records')
    for ci,k in enumerate(ks):
        g=np.array([T[n][ci] for n in names]); row=[]
        for lo,hi in ((1,2),(3,9),(10,10**9)):
            m=(o>=lo)&(o<=hi); pur=sum(max((m&(g==c)&(gt==t)).sum() for t in set(gt[m&(g==c)])) for c in set(g[m]))/m.sum()
            row.append(f"{lo}-{hi if hi<1e9 else '+'}: purity {pur:.2f} NMI {normalized_mutual_info_score(gt[m],g[m]):.2f}")
        print('  ',k,' | '.join(row))
