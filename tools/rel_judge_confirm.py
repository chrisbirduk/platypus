"""CONFIRM THE RELATIONAL-RECORDS JUDGE ON TWO BATCHES THROUGH THE GPU CODE PATH (15 Nov 2026; tools/rel_judge_confirm.py) - offline.
tools/readiness.py part 1: relational records of the pair judge the visit-2 writes at AUC 0.697 (documents since 0.639). Here the package's
own code (premises._rcode_step writes at visits 2 and 4, as on the GPU; premises._rret_judge with inf_rret_sig 'rel' judges each batch two
visits later) on BOTH batches; the judge line prints every signal's AUC (documents since, relational records, relational since the write).
Usage: python3 tools/rel_judge_confirm.py <train_distant.json.gz>"""
import sys, types, time
sys.path.insert(0, '.')
import platypus.inference.premises as pm
T0 = time.time()
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=200, inf_rret=1, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5,
                            inf_rret_sig='rel')
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; pm._PM['rti'] = {}
h = lambda v: int(round(ND * min(1.0, (v + 1) / 10)))
for wv in (1, 3):                                                              # writes at visits 2 and 4
    pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(wv)], wv)
    print(f"written at visit {wv + 1}: {sum(1 for w in pm._PM['rti'].values() if w['vis'] == wv)}  ({time.time() - T0:.0f}s)", flush=True)
for jv in (3, 5):                                                              # judged at visits 4 and 6
    pm._rret_judge(cfg, [r for r in R if r[0] < h(jv)], h(jv), jv)
print(f"DONE ({time.time() - T0:.0f}s)", flush=True)
