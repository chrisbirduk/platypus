"""python -m platypus: one GPU run. Options: LAB_OVR (JSON overrides) and the PLATYPUS_* switches (see README)."""
import os
import numpy as np
import platypus
from platypus.config import Config, _CONFIG_DEFAULTS, PILE_DOMAINS
from platypus.runner import run
from platypus.presets import ALLPILE_CONFIG, BASE_NAME, BUILD_NAME, DEFAULT_LAB_OVR

NAME = BASE_NAME
for k,v in _CONFIG_DEFAULTS.items(): setattr(Config,k,v)
cfg=Config(); cfg.seed=0
for k,v in ALLPILE_CONFIG.items(): setattr(cfg,k,v)
import json as _json                                             # OVERRIDES: LAB_OVR='{"key": value, ...}' (e.g. the replay check below)
_OVR=os.environ.get('LAB_OVR',DEFAULT_LAB_OVR)
for k,v in _json.loads(_OVR).items(): setattr(cfg,k,tuple(v) if isinstance(v,list) else v)
if os.environ.get('PLATYPUS_RULEFREE','')=='1': cfg.inf_rulefree=1
cfg.infer_drift_fix=0 if os.environ.get('PLATYPUS_ROUTEFIX','')=='0' else 1   # cg20x routing fix on by default; PLATYPUS_ROUTEFIX=0 restores the old rule
if os.environ.get('PLATYPUS_RULEFREE','')=='0': cfg.inf_rulefree=0        # cg20w: rule-free is the default; PLATYPUS_RULEFREE=0 runs the rules setup
if int(getattr(cfg,'inf_rulefree',0)):                                   # cg20t RULE-FREE INFERENCE: no relation list, no name pattern
    _rf_={"inf_units": 1, "inf_units_mode": "all", "inf_name_strict": 0, "inf_rem_stop": 0, "inf_units_ctx": 8, "inf_units_every": 1, "inf_units_rows": 64,
          "inf_units_warm": 300, "inf_pool": 1000, "inf_pool_rel": 0.5, "inf_link_slots": 1, "inf_relword_demote": 1, "inf_mem_fxsep": 16, "inf_bank_first": 1,
          "inf_bank_span": 10, "inf_rem_minrep": 2, "inf_conc_minsup": 2, "inf_conc_reenter": 1, "inf_evict_fade": 1, "inf_store_roll": 1, "inf_rem_cumul": 1, "inf_bridge_facts": 1, "inf_drop_cut": 1, "inf_pool_room": 1, "inf_gap_debug": 1,
          "inf_graph": 3, "inf_graph_theta": 1.585, "inf_replay_gumbel": 1, "inf_conc_refresh": 1, "inf_stage": 2}
    _ok_=_json.loads(_OVR)
    for k,v in _rf_.items():
        if k not in _ok_ or k in ('inf_name_strict','inf_rem_stop','inf_rem_minrep'): setattr(cfg,k,v)
    print("  [cg20x] RULE-FREE inference preset on: "+", ".join(f"{k}={getattr(cfg,k)}" for k in _rf_),flush=True)
if os.environ.get('PLATYPUS_CF','1')!='0':                                # cg26m: closed-form edits with centred keys ON by default (PLATYPUS_CF=0 turns them off)
    for k_,v_ in {'inf_cf_apply': 1, 'inf_cf_center': 1, 'inf_cf_scale': 0.5, 'inf_cf_lam': 1.0, 'inf_cf_brake': 0.015, 'inf_cf_budget': 0.05, 'inf_cf_n': 32, 'inf_cf_layer': -1, 'inf_cf_maxgrow': 1.5, 'inf_cf_min_others': 16, 'inf_cf_refresh': 8, 'inf_cf_refresh_mode': 'old', 'inf_cf_diag': 1, 'inf_cf_resolve': 1, 'inf_release_strict': 1, 'inf_cf_layerdrift': 1}.items(): setattr(cfg,k_,v_)
    print('  [cg26p] closed-form edits with centred keys on (x0.5, brake 0.015, budget 0.05, growth cap 1.5, at least 16 other conclusions to centre on, every edited conclusion re-solved with its current key each sleep, edit decay and layer drift readouts) and stricter release | PLATYPUS_CF=0 turns them off',flush=True)
if os.environ.get('PLATYPUS_SLEEP','adaptive').lower()!='fixed':          # cg26n: adaptive sleep ON by default (PLATYPUS_SLEEP=fixed restores the fixed schedule)
    cfg.inf_sleep_adapt=1; cfg.inf_sleep_cycles_max=6
    print('  [cg26n] adaptive sleep on: NREM chunks end at write convergence, REM cycles end at closure or falling yield (at most 6) | PLATYPUS_SLEEP=fixed restores the fixed schedule',flush=True)
_dm_=os.environ.get('PLATYPUS_DREAMS','jepa').lower()   # cg26: JEPA dreams by default                     # cg21 DREAMS: 'nrem' (store splices), 'rem' (store cue x the trunk's own traces), 'both', 'random' (control), '0' = off
if _dm_ not in ('0','off','none',''):
    _ok2_=_json.loads(_OVR)
    _dr_={"inf_cons": "both", "inf_cons_wlr": 0.0033, "inf_cons_wsteps": 60, "inf_cons_brake": 0.03, "inf_cons_mix": 0.25, "inf_cons_retry": 2, "inf_cons_retry_div": 3.0, "inf_cons_minpool": 16, "inf_jepa_probe": 1, "inf_jepa_cca": 1, "inf_mix_hybrid": 1, "inf_mix_cuerare": 0.75, "inf_mix_latent_score": 1, "inf_mix_nodream_src": "latent-linear,cortex-latent,cortex-latent-fam",
          "inf_jepa_views": 1, "inf_jepa_geom": 3, "inf_jepa_next": 1, "inf_jepa_shadow": 0, "inf_jepa_consol": 0, "inf_bench": "embedded", "inf_ceiling": 1, "inf_mix_null_q": 0.97, "inf_mix_tfidf_names": 1, "inf_jepa_sigreg": 1.0, "inf_cons_protect": 1, "inf_cons_adapt": 1, "inf_ceiling2": 1, "inf_ceiling2_steps": 2000, "inf_jepa_pca": 64, "inf_cons_wlr_max": 0.0033, "inf_mix_minlen": 60, "inf_jepa_drop": 3, "inf_mix_sharename": 1, "inf_mix_cxlatent": 0, "inf_mix_cxshow": 1, "inf_jepa_every": 2, "inf_mix_nameidx": 1, "inf_jd": 1, "inf_jd_steps": 300, "inf_jd_pool": "mean", "inf_jd_drop": 3, "inf_jd_sigreg": 1.0, "inf_mix_benchcue": 0.33, "inf_mix_jdname": 1, "inf_mix_cxname": 1, "inf_mix_revisit": 1, "inf_jd_hn": 4, "inf_jd_hnw": 0.5, "inf_jd_tau": 0.1, "inf_tgrad": 1, "inf_tgrad_n": 24, "inf_release": 1, "inf_release_margin": 1.0, "inf_rem_conflict": 1, "inf_mix_neardup": 0.5, "inf_cons_need": 1, "inf_gist": 1, "inf_schema_fast": 1, "inf_mix_seq": 1, "brake_eval_skip": 15, "inf_mix_seq_cap": 0.25, "inf_tgrad_track": 64, "inf_ledger": 1, "inf_mem_guard": 0.85, "inf_sleep_budget": 600, "inf_cons_spaced": 1, "inf_cons_cap_sleep": 4, "inf_cons_anchor_target": 1, "inf_hl": 1, "inf_delete": 1, "inf_delete_grace": 2, "inf_ghost_n": 32, "inf_release_hl": 1, "inf_release_horizon": 10, "inf_graph_window": 12, "inf_jd_sigreg_mode": "lejepa", "inf_jd_lambda": 0.05, "inf_jd_statm": 0.8, "inf_jd_es": 1, "inf_jd_wd": 0.05, "inf_jd_judge": 1, "inf_mix_jdwide": 1, "inf_mix_jdwide_bonus": 0.3, "inf_mix_jdwide_k": 1, "inf_jd_es_tol": 0.02, "inf_jd_judge_link": 1, "inf_mix_jdfinal": 1, "inf_mix_jdfinal_k": 1, "inf_mix_jdfinal_fair": 1, "inf_jd_outcome": 1, "inf_jd_out_n": 48, "inf_jd_out_pos": 1.0, "inf_jd_out_rel": 1, "inf_sleep_inferred": 1, "inf_cons_rest": 0.2, "inf_cons_hlsched": 2, "inf_evict_clean": 1, "inf_schema_speed": 1, "inf_release_pnext_q": 0.25, "inf_release_pnext_min": 0.2, "inf_ghost_reinstate": 2, "inf_cue_repair": 1, "inf_cue_repair_d": 2, "inf_jd_entpairs": 1000, "inf_jd_plantw": 1, "inf_upd_revisited": 1, "inf_mix_jdfinal_skip": "jepa,sequence,cortex-mmr,cortex-jepa,cortex-mmr-fam,cortex-jepa-fam", "inf_jd_negrand": 2, "inf_jd_obj": "lejepa", "inf_jd_real": 1,
          "inf_splicemix": 1, "inf_mix_spectral": 1, "inf_mix_rarecopy": 1, "inf_mix_null": 1, "inf_mix_bits": 2.0, "inf_mix_q": 32,
          "inf_mix_kr": 4, "inf_mix_kn": 4, "inf_mix_nbase": 3, "inf_mix_dream": 0.5}
    if _dm_ in ('rem','both','jepa','noreplay','random'): _dr_.update({"inf_mix_cortex": 4, "inf_mix_cxsel": "mmr", "inf_mix_cxsrc": "familiar"})   # cg26g: the controls propose and log the same REM traces
    if _dm_=='rem': _dr_["inf_mix_dream_src"]="cortex-mmr,cortex-mmr-fam"
    if _dm_=='nrem': _dr_["inf_mix_dream_src"]="hybrid,spectral"
    if _dm_=='random': _dr_["inf_mix_dream_rand"]=1
    if _dm_=='noreplay': _dr_.update({"inf_cons": "mem", "inf_mix_dream": 0.0, "inf_mix_dream_src": "jepa-name,jepa-wide,cortex-jepa-name-fam,sequence"})   # cg26g CONTROL: everything else identical (dreams proposed, tested and logged, CLS readouts), but no trunk replay
    if _dm_=='jepa': _dr_["inf_mix_dream_src"]="jepa-name,jepa-wide,cortex-jepa-name-fam,sequence"   # cg26g: + sequence replay; cg26c: NREM and REM dreams both chosen by the JEPA within the name index (whole-store jepa was no better than random in cg26b)   # cg26: dream ONLY what the trained JEPA selector chose (NREM + REM)                         #   CONTROL: dream random real-text pairs (locally as good as selected pairs on BPC / forgetting)
    for k,v in _dr_.items():
        if k not in _ok2_ or k in ('inf_cons',): setattr(cfg,k,v)   # cg21b FIX: the default run options carry inf_cons=mem, which kept every dream out of the weights in cg21
    if os.environ.get('PLATYPUS_PASSES','').strip().isdigit(): cfg.n_passes=int(os.environ['PLATYPUS_PASSES']); print(f"  [cg26g] PLATYPUS_PASSES: {cfg.n_passes} pass(es)",flush=True)
    print(f"  [cg21] DREAMS ({_dm_}): "+", ".join(f"{k}={getattr(cfg,k)}" for k in _dr_)+" | PLATYPUS_DREAMS=0 turns them off",flush=True)
if os.environ.get('PLATYPUS_CLEAN','1')!='0':                            # cg26q: ON by default; PLATYPUS_CLEAN=0 restores the cg26p evaluation and oracle options                            # CLEAN EVALUATION (review fixes): held-out wordings, bridges as candidates, no
    for k_,v_ in {'inf_clean':1,'inf_rem_probe_frac':0.0,'inf_mix_benchcue':0.0,'inf_jd_plantw':0,'inf_jd_real':0,'inf_fm_debug':1}.items(): setattr(cfg,k_,v_)   #   slot given; oracle options off
    print('  [cg26q] CLEAN EVALUATION on: held-out probe wordings, bridges as candidates, no slot given; REM probe subjects, benchmark cues and planted-pair weighting off | PLATYPUS_CLEAN=0 restores cg26p',flush=True)
if os.environ.get('PLATYPUS_HINTS','1')=='0':
    cfg.cd_hint_frac=0.0; cfg.cd_hint_adapt=0.0; print('  [clean] PLATYPUS_HINTS=0: no true bytes given to the conduit generator in the closed-loop passes',flush=True)
if os.environ.get('PLATYPUS_SLOTS','').strip().isdigit():
    cfg.infer_slots=int(os.environ['PLATYPUS_SLOTS']); print(f'  [clean] PLATYPUS_SLOTS: the router may open up to {cfg.infer_slots} slots',flush=True)
NAME=BUILD_NAME+NAME+' | overrides '+_OVR
print(f"\n################ {NAME} | seed {cfg.seed} ################")
r=run(cfg)
print("\n==================== SUMMARY ====================")
print(f"final BPC {r['bpc']:.3f} | forgetting {r['forget']:+.3f} | post-train {np.mean(r['post']):.3f}")
for k,p in enumerate(r.get('passes',[])):
    pre=[x for x in p['pre'] if x is not None]
    print(f"pass {k+1}: end {p['bpc']:.3f} | forgetting {p['forget']:+.4f} | post-train {np.mean(p['post']):.3f}"
          +(f" | before relearning {np.mean(pre):.3f}" if pre else "")+" | label-free "+" ".join(f"{m} {v:.3f}" for m,v in p['alt'].items()))
dn=r.get('domains',PILE_DOMAINS)
for d in range(len(dn)):
    print(f"   {dn[d][:18]:18} end of each pass: "+" -> ".join(f"{p['final'][d]:.3f}" for p in r.get('passes',[])))
tl=r.get('timing',{})
if tl: print(f"timing: {np.mean([v['ms'] for v in tl.values()]):.1f} ms per regular step | training {sum(v['wall'] for v in tl.values())/60:.1f} min | run {r.get('wall_run',float('nan'))/60:.1f} min")
