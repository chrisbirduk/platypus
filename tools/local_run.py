"""LOCAL PACKAGE RUN on the five-domain mini-Pile (16 Nov 2026; tools/local_run.py): the GPU preset's overrides at local scale, through
python -m platypus, with the Pile replaced by the local mini-Pile (MINIPILE_DIR: prose, wiki, arxiv, pubmed, law; train = the start of each
file, validation = the last 500 KB) and the distant set by LR_DISTANT. For package-level checks of a build's change (forgetting, final bits,
the build's own readouts) before a GPU run.
Usage: LR_PRESET=configs/presets/cg27a_kty.overrides.json LR_OVR='{"key": value}' LR_DISTANT=<distant_50k.json.gz> python3 tools/local_run.py
       env LR_STEPS (steps per task, 250), LR_SEED (4); prints the package's own log."""
import os, sys, json, runpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile')
FILES = {"Gutenberg (PG-19)": "prose", "Wikipedia (en)": "wiki", "ArXiv": "arxiv", "PubMed Abstracts": "pubmed", "FreeLaw": "law"}
TRAIN_BYTES = 2_600_000; VAL_BYTES = 500_000
def minipile_bytes(domain, seed, n):
    b = open(os.path.join(DATA, FILES[domain] + '.bin'), 'rb').read()
    return b[-VAL_BYTES:] if seed % 10000 >= 7777 else b[:min(int(n), TRAIN_BYTES)]
import platypus.data as D
D._stream_bytes = minipile_bytes; D._cache_path = lambda key: None; D._BYTES_CACHE.clear()
from platypus.presets import DEFAULT_LAB_OVR                     # as scripts/merge_overrides.py: LAB_OVR REPLACES the defaults, so merge onto them
ovr = json.loads(DEFAULT_LAB_OVR)                               #   (16 Nov: without this the first local run had inf_on off - no inference at all)
if os.environ.get('LR_PRESET'): ovr.update({k: v for k, v in json.load(open(os.environ['LR_PRESET'])).items() if not k.startswith('_')})
ovr.update(dict(d_model=64, d_state=64, n_layers=2, seq_len=64, batch_size=16, steps_per_task=int(os.environ.get('LR_STEPS', 250)),
                d_state_max=512, widen_chunk=16, min_fresh=32, widen_per_difficulty=128, log_locality=False, log_every=250, sens_every=125,
                extra_domains=(), amp='', adam_fused=False))   # the five local domains only; CPU: no bf16 autocast, no fused Adam
if os.environ.get('LR_DISTANT'): ovr['inf_premises_path'] = os.environ['LR_DISTANT']
ovr.update(json.loads(os.environ.get('LR_OVR', '{}')))
ovr['seed'] = int(json.loads(os.environ.get('LR_OVR', '{}')).get('seed', os.environ.get('LR_SEED', '4')))   # as run_gpu.sh does; before 25 Nov local runs ignored LR_SEED and ran on seed 1
os.environ['LAB_OVR'] = json.dumps(ovr)
for k, v in dict(PLATYPUS_CLEAN='1', PLATYPUS_HINTS='1', PLATYPUS_CF='1', PLATYPUS_SLEEP='adaptive', PLATYPUS_DREAMS='jepa', PLATYPUS_RULEFREE='1',
                 PLATYPUS_ROUTEFIX='1', PLATYPUS_PASSES='1', PLATYPUS_FNTIME='0').items(): os.environ.setdefault(k, v)
os.environ.setdefault('PRESET_SEED', os.environ.get('LR_SEED', '4'))
runpy.run_module('platypus', run_name='__main__')
