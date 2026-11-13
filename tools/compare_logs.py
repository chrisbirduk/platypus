"""COMPARE GPU LOGS VISIT BY VISIT (12 Nov 2026; tools/compare_logs.py) - the readouts that decide the current runs, side by side.
Usage: python3 tools/compare_logs.py NAME=log.txt [NAME=log.txt ...] [--groups rt,diag,fm,...]   (a bare path is named by its file name)
Groups (a group is printed only if some log has it):
  bpc   post-train BPC of each visit (safety: the same seed's runs should agree within ~0.01)
  rt    [REAL-TEXT RECALL]: replayed and control documents, recall vs the fresh-subject prior, and the GAP (recall - prior)
  diag  [REAL-TEXT RECALL, DIAGNOSTICS]: storage / own-frame / template / prior, replayed vs control, named sentences
  fm    [FALSE MEMORY PROBE]: (a) fresh name -> stored name + confidence, (b) known subject, missing relation, (c) known facts + AUC
  rpl   replay losses (last 20 steps): source sentences, real-text sentences
  ctl   [EVER TAUGHT] never-asked CONTROL facts from the trunk (learning from reading)
  ab    [ABSTENTION] explicit refusals, false answers with no threshold and at the threshold keeping 75% of known-right; refusal-row loss
  gate  [GATED ANSWERS] the trunk answers only above last sleep's calibrated confidence: known answered / right, never-read false vs ungated
  sc    [REPLAY BY STORAGE CHECK] the trunk's loss on replayed sentences, replay share to the hardest quarter, last hardest quarter's change
  cg    [COMPRESSION GAIN] units (binding gain vs surprise) and links (mutual compression gain vs letter triples)
  preq  [PREQUENTIAL BITS] held-out text of the domains read and of the next domain: all bytes, known names, unknown names
  link  [STORE LINKING] letter and compression links, their precision on benchmark entities, records added
  gates [GATED ANSWERS SMOOTHED] the gate with a threshold averaged across sleeps
  xg    [EXTRACTION GAIN] candidate records from real text: what they save the trunk where their names meet again, related vs not
  s4    [RELATIONAL MODULE WRITES] step 4, edits subtracted: taught vs held back
Missing values print as '-'. Reads plain text only."""
import os
import re
import sys
import collections

NUM = r'(nan|-?\d+\.\d+)'


def parse(path):
    V = collections.defaultdict(dict); vis = 0
    for ln in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'=== Phase (\d+):', ln)
        if m: vis = int(m.group(1)); continue
        m = re.search(r'\| visit (\d+)\]', ln); v = int(m.group(1)) if m else vis
        d = V[v]
        m = re.match(r'\s*post-train BPC: ' + NUM, ln)
        if m and vis: V[vis]['bpc'] = m.group(1); continue
        if '[REAL-TEXT RECALL | visit' in ln:
            pat = r'the true object among up to 10 same-type names ' + NUM + r' \| with a FRESH subject \(relation prior alone\) ' + NUM
            for tag, seg in (('rep', ln.split('REPLAYED documents:')[1].split('||')[0] if 'REPLAYED documents:' in ln else None),
                             ('ctl', ln.split('CONTROL documents')[1] if 'CONTROL documents' in ln else None),
                             ('all', ln if 'REPLAYED documents:' not in ln else None)):
                if seg is None: continue
                m = re.search(pat, seg)
                if m:
                    d[f'rt_{tag}'], d[f'rt_{tag}_pri'] = m.group(1), m.group(2)
                    try: d[f'rt_{tag}_gap'] = f"{float(m.group(1)) - float(m.group(2)):+.2f}"
                    except ValueError: pass
        elif '[REAL-TEXT RECALL, DIAGNOSTICS | visit' in ln:
            for grp, kd, body in re.findall(r'(REPLAYED|CONTROL|ALL), (named|pronoun-led): ([^|]*\|[^|]*\|[^|]*\|[^|(]*\(\d+\))', ln):
                m = re.search(r'storage ' + NUM + r' \| own-frame ' + NUM + r' \| template ' + NUM + r' \| prior ' + NUM + r' \((\d+)\)', body)
                if m:
                    k = f"dg_{grp[:3].lower()}_{'nm' if kd == 'named' else 'pr'}"
                    d[k + '_st'], d[k + '_own'], d[k + '_tpl'], d[k + '_pri'], d[k + '_n'] = m.groups()
        elif '[DISTILLATION, FALSE MEMORY PROBE | visit' in ln:
            m = re.search(r'\(a\).*?STORED name ' + NUM + r', confidence ' + NUM, ln)
            if m: d['fm_a'], d['fm_a_c'] = m.groups()
            m = re.search(r'\(b\).*?STORED name ' + NUM + r', its object for ANOTHER relation ' + NUM + r', confidence ' + NUM, ln)
            if m: d['fm_b'], d['fm_b_oth'], d['fm_b_c'] = m.groups()
            m = re.search(r'\(c\).*?answered right ' + NUM + r', confidence when RIGHT ' + NUM + r', when wrong ' + NUM + r'.*?AUC ' + NUM, ln)
            if m: d['fm_k'], d['fm_k_cr'], d['fm_k_cw'], d['fm_auc'] = m.groups()
        elif '[DISTILLATION, SOURCE REPLAY | visit' in ln:
            m = re.search(r'-> ' + NUM + r' \(last 20\)', ln)
            if m: d['rp_src'] = m.group(1)
        elif '[DISTILLATION, REAL-TEXT REPLAY | visit' in ln:
            m = re.search(r'-> ' + NUM + r' \(last 20\)', ln)
            if m: d['rp_rtx'] = m.group(1)
        elif '[DISTILLATION, EVER TAUGHT | visit' in ln:
            m = re.search(r'CONTROL, sample of \d+ ever held back: ALL (?:conclusions [^|]*\| )?facts ' + NUM + r' \((\d+)\)', ln)
            if m: d['ctl'], d['ctl_n'] = m.groups()
            m = re.search(r'TAUGHT FOR AT LEAST ONE VISIT \(\d+\) (?:conclusions [^|]*\| )?facts ' + NUM, ln)
            if m: d['taught'] = m.group(1)
        elif '[DISTILLATION, ABSTENTION | visit' in ln:
            m = re.search(r'on known facts ' + NUM + r', on \(a\) fresh names ' + NUM + r', on \(b\) missing relations ' + NUM, ln)
            if m: d['ab_rk'], d['ab_ra'], d['ab_rb'] = m.groups()
            m = re.search(r'WITHOUT ABSTENTION: never-read given a stored name ' + NUM, ln)
            if m: d['ab_leak0'] = m.group(1)
            m = re.search(r'keep 75% of known-right \(t [0-9.]+\): known answered ' + NUM + r', accuracy when answered ' + NUM + r', NEVER-READ still given a stored name ' + NUM, ln)
            if m: d['ab_k75'], d['ab_acc75'], d['ab_leak75'] = m.groups()
        elif '[DISTILLATION, REFUSAL ROWS | visit' in ln:
            m = re.search(r'-> ' + NUM + r' \(last 20\)', ln)
            if m: d['rf_loss'] = m.group(1)
        elif '[DISTILLATION, REPLAY BY STORAGE CHECK | visit' in ln:
            m = re.search(r'median ' + NUM + r'.*?HARDEST quarter ' + NUM + r'.*?loss change ([-+]?[0-9.]+|[-+]?nan)', ln)
            if m: d['sc_med'], d['sc_hard'], d['sc_mv'] = m.groups()
        elif '[DISTILLATION, GATED ANSWERS | visit' in ln:
            m = re.search(r'answered ' + NUM + r', right when answered ' + NUM + r', right overall ' + NUM + r' \(ungated ' + NUM + r'\)', ln)
            if m: d['g_ans'], d['g_acc'], d['g_right'], d['g_right0'] = m.groups()
            m = re.search(r'given a stored name ' + NUM + r' \(ungated ' + NUM + r'\), abstained ' + NUM, ln)
            if m: d['g_false'], d['g_false0'], d['g_abst'] = m.groups()
        elif '[PREQUENTIAL BITS | visit' in ln:
            m = re.search(r'domains READ \(\d+\): all bytes ' + NUM + r' \| KNOWN names ' + NUM + r' \((\d+) bytes\) \| unknown names ' + NUM, ln)
            if m: d['pq_all'], d['pq_known'], d['pq_kn'], d['pq_unk'] = m.groups()
            m = re.search(r'NEXT domain \d+ \(not yet read\): all bytes ' + NUM + r' \| KNOWN names ' + NUM + r' \((\d+) bytes\) \| unknown names ' + NUM, ln)
            if m: d['pq_nall'], d['pq_nknown'], d['pq_nkn'], d['pq_nunk'] = m.groups()
        elif '[STORE LINKING | visit' in ln:
            m = re.search(r'LETTER links (\d+) \(on benchmark entities (\d+), same entity ' + NUM + r'\) \| COMPRESSION links (\d+) of (\d+) candidate pairs.*?on benchmark entities (\d+), same entity ' + NUM + r'\) \| records ADDED with linked names (\d+)', ln)
            if m: d['lk_l'], d['lk_ln'], d['lk_lp'], d['lk_c'], d['lk_cc'], d['lk_cn'], d['lk_cp'], d['lk_add'] = m.groups()
        elif '[DISTILLATION, GATED ANSWERS SMOOTHED | visit' in ln:
            m = re.search(r'right overall ' + NUM + r' \| NEVER READ given a stored name ' + NUM, ln)
            if m: d['gs_right'], d['gs_false'] = m.groups()
        elif '[COMPRESSION GAIN | visit' in ln:
            m = re.search(r'RELATIONS by substitution.*?AUC same gold type vs not ' + NUM + r' \| best candidate of the same type ' + NUM + r' \(chance ' + NUM, ln)
            if m: d['cg_rel'], d['cg_reltop'], d['cg_relch'] = m.groups()
            m = re.search(r'PMI AUC same mention vs boundary ' + NUM, ln)
            if m: d['cg_pmi'] = m.group(1)
            m = re.search(r'BINDING GAIN ' + NUM, ln)
            if m: d['cg_unit'] = m.group(1)
            m = re.search(r'plain surprise of the second word ' + NUM, ln)
            if m: d['cg_surp'] = m.group(1)
            m = re.search(r'COMPRESSION GAIN all ' + NUM + r', HARD ' + NUM + r', VERY HARD ' + NUM, ln)
            if m: d['cg_all'], d['cg_hard'], d['cg_vh'] = m.groups()
            m = re.search(r'letter triples all ' + NUM + r', HARD ' + NUM, ln)
            if m: d['cg_str'], d['cg_strh'] = m.groups()
        elif '[EXTRACTION GAIN | visit' in ln:
            m = re.search(r'(\d+) candidate records.*?related ' + NUM + r' \|\| AUC related vs not: WORDS BETWEEN beyond the names ' + NUM + r', whole record ' + NUM + r', names only ' + NUM + r', closeness ' + NUM + r' \| top quarter kept, related: words between ' + NUM, ln)
            if m: d['xg_n'], d['xg_base'], d['xg_wb'], d['xg_all'], d['xg_pair'], d['xg_cl'], d['xg_top'] = m.groups()
        elif '[RELATIONAL MODULE WRITES | visit' in ln:
            m = re.search(r'edits SUBTRACTED: TAUGHT ' + NUM + r' \((\d+)\) vs HELD BACK ' + NUM + r' \((\d+)\)', ln)
            if m: d['s4_t'], d['s4_tn'], d['s4_h'], d['s4_hn'] = m.groups()
    return V


GROUPS = {
    'bpc': ('Post-train BPC', [('bpc', 'BPC')]),
    'rt': ('Real-text recall (gap = recall - fresh-subject prior)', [('rt_rep', 'replayed'), ('rt_rep_pri', 'prior'), ('rt_rep_gap', 'GAP'),
                                                                     ('rt_ctl', 'control'), ('rt_ctl_gap', 'ctl gap'), ('rt_all', 'all'), ('rt_all_gap', 'all gap')]),
    'diag': ('Real-text diagnostics, named sentences', [('dg_rep_nm_st', 'rep storage'), ('dg_rep_nm_own', 'rep own-frame'), ('dg_rep_nm_pri', 'rep prior'),
                                                        ('dg_con_nm_st', 'ctl storage'), ('dg_con_nm_own', 'ctl own-frame'), ('dg_all_nm_st', 'all storage')]),
    'fm': ('False-memory probe', [('fm_a', '(a) fresh->stored'), ('fm_a_c', '(a) conf'), ('fm_b', '(b) stored'), ('fm_b_c', '(b) conf'),
                                  ('fm_k', '(c) known right'), ('fm_k_cr', '(c) conf right'), ('fm_auc', 'AUC')]),
    'rpl': ('Replay loss, last 20 steps', [('rp_src', 'source'), ('rp_rtx', 'real text')]),
    'ctl': ('Learning from reading (ledger, from the trunk)', [('taught', 'taught'), ('ctl', 'never asked')]),
    'ab': ('Abstention (from the trunk)', [('ab_rk', 'refuse known'), ('ab_ra', 'refuse (a) fresh'), ('ab_rb', 'refuse (b) missing rel'), ('ab_leak0', 'false, no threshold'),
                                           ('ab_k75', 'known answered @75%'), ('ab_acc75', 'accuracy @75%'), ('ab_leak75', 'false @75%'), ('rf_loss', 'refusal-row loss')]),
    'gate': ('Gated answers (threshold from the previous sleep)', [('g_ans', 'known answered'), ('g_acc', 'right when answered'), ('g_right', 'right overall'),
                                                                  ('g_right0', 'right ungated'), ('g_false', 'never-read: false'), ('g_false0', 'false ungated'), ('g_abst', 'abstained')]),
    'preq': ('Prequential bits (held-out, weights only)', [('pq_all', 'read: all'), ('pq_known', 'read: known names'), ('pq_unk', 'read: unknown names'),
                                                           ('pq_nall', 'next: all'), ('pq_nknown', 'next: known names'), ('pq_nkn', 'next: known bytes'), ('pq_nunk', 'next: unknown names')]),
    'link': ('Store linking', [('lk_l', 'letter links'), ('lk_lp', 'letter: same entity'), ('lk_c', 'compression links'), ('lk_cc', 'candidates'),
                               ('lk_cp', 'compression: same entity'), ('lk_add', 'records added')]),
    'gates': ('Gated answers, smoothed threshold', [('gs_right', 'right overall'), ('gs_false', 'never-read: false')]),
    'sc': ('Replay by storage check', [('sc_med', 'median loss'), ('sc_hard', 'share to hardest quarter'), ('sc_mv', 'hardest quarter change')]),
    'cg': ('Compression gain (AUC)', [('cg_rel', 'relations: substitution AUC'), ('cg_reltop', 'relations: best'), ('cg_relch', 'relations: chance'), ('cg_pmi', 'units: PMI (counts)'), ('cg_unit', 'units: binding gain'), ('cg_surp', 'units: surprise'), ('cg_all', 'links: gain'), ('cg_hard', 'links: gain HARD'),
                                       ('cg_vh', 'links: gain VERY HARD'), ('cg_str', 'links: letter triples'), ('cg_strh', 'letter triples HARD')]),
    'xg': ('Extraction gain (AUC related vs not; top quarter kept by words-between gain)', [('xg_n', 'records'), ('xg_base', 'related (base)'), ('xg_wb', 'words between'), ('xg_all', 'whole record'),
                                                                                         ('xg_pair', 'names only'), ('xg_cl', 'closeness'), ('xg_top', 'top quarter related')]),
    's4': ('Step 4, edits subtracted', [('s4_t', 'taught'), ('s4_tn', 'n'), ('s4_h', 'held back'), ('s4_hn', 'n')]),
}


def main(argv):
    want = None; runs = []
    for a in argv:
        if a.startswith('--groups'): want = a.split('=', 1)[1].split(',') if '=' in a else None; continue
        nm, p = a.split('=', 1) if '=' in a and not os.path.exists(a) else (os.path.basename(a).rsplit('.', 1)[0], a)
        runs.append((nm, parse(p)))
    if not runs: print(__doc__); return
    for g, (title, cols) in GROUPS.items():
        if want and g not in want: continue
        cols_ = [(k, lab) for k, lab in cols if any(k in d for _, V in runs for d in V.values())]
        if not cols_: continue
        heads = [(nm, k, lab) for nm, V in runs for k, lab in cols_ if any(k in d for d in V.values())]
        vis = sorted({v for _, V in runs for v, d in V.items() if v and any(k in d for k, _ in cols_)})
        print(f"\n### {title}\n")
        print("| visit | " + " | ".join(f"{nm} {lab}" if len(runs) > 1 else lab for nm, k, lab in heads) + " |")
        print("|---|" + "---|" * len(heads))
        RV = dict(runs)
        for v in vis: print(f"| {v} | " + " | ".join(RV[nm].get(v, {}).get(k, '-') for nm, k, lab in heads) + " |")


if __name__ == '__main__':
    main(sys.argv[1:])
