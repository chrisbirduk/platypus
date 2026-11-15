"""Smoke test of SPARSE MEMORY CONSOLIDATION IN THE TRUNK (platypus.inference.smf; inf_smf 1 + inf_distill_wake_params 'smem') on a small
platypus Model on the CPU, through the package's own code path (distill_wake_step with a fabricated question pool, as distill_wake_sleep
builds it). 1. visit 1: the model learns wiki text, the memory layer with it (its text phase); bits on held-out wiki with the layer on and
off. 2. the layer is frozen for text (smf_visit); then visit 2: text training continues and every step adds the question rows of 60 facts
(seven self-written wordings, never the test wording) - written to the layer only ('smem') or, as the GPU builds so far, to every block's
MLP output matrix ('mlp_out'), or not at all (control). Readouts: recall in the test wording (rank among 10 of the same kind, open-ended
exact), held-out wiki bits (forgetting against the control), slots written.
Usage: python3 tools/smf_smoke.py   env SS_PRE (600), SS_STEPS (600), SS_D (96), SS_N (64), SS_T (256), SS_LR (0.01)"""
import os, sys, json, time, copy, math
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus
from platypus.config import Config, _CONFIG_DEFAULTS
from platypus.presets import ALLPILE_CONFIG
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2)
for k, v in _CONFIG_DEFAULTS.items(): setattr(Config, k, v)
cfg = Config(); cfg.seed = 0
for k, v in ALLPILE_CONFIG.items(): setattr(cfg, k, v)
cfg.d_model = int(os.environ.get('SS_D', 96)); cfg.d_state = cfg.d_model; cfg.n_layers = 3; cfg.seq_len = 128
cfg.inf_smf = 1; cfg.inf_smf_n = int(os.environ.get('SS_N', 64)); cfg.inf_smf_k = 8; cfg.inf_smf_t = int(os.environ.get('SS_T', 256)); cfg.inf_smf_lr = float(os.environ.get('SS_LR', 0.01))
cfg.inf_distill_wake = 1.0; cfg.inf_distill_wake_rows = 8; cfg.inf_kmi = 0; cfg.inf_kmc = 0; cfg.inf_rti_rows = 0; cfg.inf_rret = 0; cfg.lr = 1e-3
from platypus.model import Model
from platypus.inference.closed_form import _ws_batch, _ws_run
from platypus.inference.distill import distill_wake_step
from platypus.inference.smf import smf_text_step, smf_visit, _blk
torch.manual_seed(0); model = Model(cfg); rs = np.random.RandomState(0)
wiki = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin', 'rb').read().replace(b'\n', b' '); WK = np.frombuffer(wiki, np.uint8); HO = len(WK) - 100000
L = 128
def wb(r, n=16): return torch.tensor(np.stack([WK[i:i + L + 1] for i in r.randint(0, HO - L - 2, n)]).astype(np.int64))
EV = torch.tensor(np.stack([WK[HO + k * (L + 1):HO + (k + 1) * (L + 1)] for k in range(48)]).astype(np.int64))
@torch.no_grad()
def bits(m, off=False):
    b = _blk(m); b._smf_off = off; m.eval()
    try: lg = m(EV[:, :-1], 0); return float(F.cross_entropy(lg.reshape(-1, 256).float(), EV[:, 1:].reshape(-1))) / math.log(2)
    finally: b._smf_off = False; m.train()
def text_step(m, op, r):
    X = wb(r); lg = m(X[:, :-1], 0); loss = F.cross_entropy(lg.reshape(-1, 256).float(), X[:, 1:].reshape(-1)); op.zero_grad(); loss.backward(); return loss
# 1. visit 1: text, the layer learns with the trunk
CACHE = os.environ.get('SS_CACHE', '')
if CACHE and os.path.exists(CACHE):
    ck = torch.load(CACHE, weights_only=False); model.load_state_dict(ck['sd']); model.__dict__['_smf_st'] = ck['st']; smf_visit(model, cfg, 0); P(f"visit 1 loaded from {CACHE}")
else:
    op = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr); r1 = np.random.RandomState(1)
    for st in range(int(os.environ.get('SS_PRE', 600))):
        l = text_step(model, op, r1); smf_text_step(model, cfg); op.step()
        if (st + 1) % 200 == 0: P(f"   visit 1 step {st + 1}: train {float(l.detach()) / math.log(2):.3f} bits per byte  ({time.time() - T0:.0f}s)")
    smf_visit(model, cfg, 0)
    if CACHE: model.__dict__['_smf_st']['kopt'] = None; torch.save(dict(sd=model.state_dict(), st=model.__dict__['_smf_st']), CACHE)
b0, b0off = bits(model), bits(model, True); st0 = model.__dict__['_smf_st']
P(f"after visit 1: held-out wiki {b0:.3f} bits per byte, layer OFF {b0off:.3f} ({b0off - b0:+.3f}: the trunk uses it) | slots used by text {int((st0['bg'] > 0).sum())} of {cfg.inf_smf_n ** 2}")
# 2. the facts and their question rows
syl = ['ka', 'lo', 'mi', 'ner', 'tas', 'vol', 'ru', 'den', 'sa', 'por', 'el', 'ith']
nm = lambda: (''.join(rs.choice(syl, 2)) + ' ' + ''.join(rs.choice(syl, 3))).title()
rels = ['works with', 'lives near', 'reports to', 'is a partner of', 'trained under', 'knows']
people = list(dict.fromkeys(nm() for _ in range(300)))
FACTS = [(people[i], rels[i % 6], people[150 + i]) for i in range(60)]
FORMS = ("In short, {A} {r} ", "Notes on {A}: {A} {r} ", "{A}, as noted before, {r} ", "As for {A}, people say {A} {r} ", "Q: {A} {r} whom? A: ", "From the notes: {A} {r} ", "{A}, whose name came up earlier, {r} ")
rows = [(f.format(A=A, r=r).encode(), (C + '.').encode(), 0, ('fact', A, r, C), None, None) for A, r, C in FACTS for f in FORMS]
@torch.no_grad()
def recall(m):
    rk, ex = [], []; rr = np.random.RandomState(5)
    for A, r, C in FACTS:
        pb = f"It is on record that {A} {r} ".encode(); alts = [p for p in people[150:] if p != C]; cands = [C] + [alts[j] for j in rr.choice(len(alts), 9, replace=False)]
        X, _ = _ws_batch(m, [pb + c.encode() + b'.' for c in cands]); lg = _ws_run(m, X, 0)
        sc = [float(F.cross_entropy(lg[k, len(pb) - 1:len(pb) - 1 + len(c) + 1].float(), X[k, len(pb):len(pb) + len(c) + 1])) for k, c in enumerate(cands)]
        rk.append(int(np.argmin(sc) == 0)); tb = (C + '.').encode()
        ex.append(int(bool((lg[0, len(pb) - 1:len(pb) - 1 + len(tb) - 1].argmax(-1) == X[0, len(pb):len(pb) + len(tb) - 1]).all())))
    return np.mean(rk), np.mean(ex)
r0 = recall(model); P(f"before teaching: rank {r0[0]:.2f}, open {r0[1]:.2f} | {len(rows)} question rows (60 facts x 7 wordings, the test wording never used)")
BASE = copy.deepcopy(model)
def run(mode):
    m = copy.deepcopy(BASE); m.__dict__['_smf_st'] = copy.deepcopy(BASE.__dict__['_smf_st'])
    cfg.inf_distill_wake_params = mode if mode != 'none' else 'mlp_out'
    if mode != 'none': m.__dict__['_dtw'] = dict(bysl={0: rows}, rs=np.random.RandomState(7), n=0, losses=[], akl=[], st=dict(proj=0, conf=0, cut=[]), pdst=dict(n=0, b=0, t=0, s=0, nos=0))
    op = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=cfg.lr); r2 = np.random.RandomState(2); t0 = time.time()
    for st in range(int(os.environ.get('SS_STEPS', 600))):
        text_step(m, op, r2); smf_text_step(m, cfg)
        if mode != 'none': distill_wake_step(m, cfg)
        op.step()
    rk, ex = recall(m); b1 = bits(m); s = m.__dict__['_smf_st']; D = m.__dict__.get('_dtw') or {}
    P(f"   {mode:8s}: test wording rank {rk:.2f}, open {ex:.2f} | held-out wiki {b1:.3f} bits per byte | question loss {np.mean(D.get('losses', [0])[:20]):.3f} -> {np.mean(D.get('losses', [0])[-20:]):.3f} | "
      f"slots written {int(s['wrote'].sum())} ({s['steps_q']} steps) | {time.time() - t0:.0f}s")
    return b1
P("visit 2: text training continues, + the question rows each step (8 a step), written to:")
GRID = [tuple(x.split(':')) for x in os.environ.get('SS_GRID', '').split(',') if x]   # e.g. 0.03:256,0.1:1024 (lr:slots) - smem runs only
if GRID:
    bn = run('none')
    for lr_, t_ in GRID:
        cfg.inf_smf_lr = float(lr_); cfg.inf_smf_t = int(t_); P(f"   smem lr {lr_} slots {t_}:"); bs = run('smem'); P(f"      forgetting against the control {bs - bn:+.4f}")
else:
    bn = run('none'); bs = run('smem'); bm = run('mlp_out')
    P(f"FORGETTING against the control (bits added on held-out wiki by the question rows): smem {bs - bn:+.4f}, mlp_out {bm - bn:+.4f}")
P(f"DONE ({time.time() - T0:.0f}s)")
