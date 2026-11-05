"""Identity harness: run the GPU build (single file or package) locally, fixed seed, local stand-in data, with the options
of the full-build local safety run (local_base_overrides.json + local_safety_overrides.json); "short" = 2 passes x 24 steps.
usage: python3 harness.py SRC LOG {short|safe}
  SRC: a .py file (the single-file build) or a directory holding the platypus package
Writes the full stdout to LOG (unfiltered)."""
import sys, os, json, time, contextlib, textwrap, importlib, importlib.util as iu
src, log, mode = sys.argv[1], sys.argv[2], sys.argv[3]
import torch; torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
# the local five-domain data and its helper (threshold_sweep_local.py) live outside the repository: LOCAL_HARNESS names their folder
sys.path.insert(0, os.environ.get('LOCAL_HARNESS', '/tmp/claude-0/sweep'))
os.environ.setdefault('MINIPILE_DIR', os.path.join(os.environ.get('LOCAL_HARNESS', '/tmp/claude-0/sweep'), 'minipile'))
import threshold_sweep_local as S

if src.endswith('.py'):
    spec = iu.spec_from_file_location('platypus_build', src); L = iu.module_from_spec(spec); sys.modules['platypus_build'] = L
    spec.loader.exec_module(L)
    if hasattr(L, 'ALLPILE_CONFIG'): ALL = L.ALLPILE_CONFIG
    else:
        s_ = open(src).read(); i_ = s_.index('    ALLPILE_CONFIG=dict('); j_ = s_.index('\n    )\n', i_) + 6
        ns_ = {}; exec(textwrap.dedent(s_[i_:j_]), vars(L), ns_); ALL = ns_['ALLPILE_CONFIG']
    set_stream = lambda f: setattr(L, '_stream_bytes', f)
    Config, DEF, run = L.Config, L._CONFIG_DEFAULTS, L.run
else:
    sys.path.insert(0, src); P = importlib.import_module('platypus')
    ALL = P.presets.ALLPILE_CONFIG
    set_stream = P.data.set_stream_source
    Config, DEF, run = P.config.Config, P.config._CONFIG_DEFAULTS, P.run

set_stream(S.minipile_bytes)
ovr = json.load(open(os.path.join(HERE, 'local_base_overrides.json')))
ovr.update(json.load(open(os.path.join(HERE, 'local_safety_overrides.json'))))
if mode == 'short': ovr.update(n_passes=2, steps_per_task=24)
for k, v in DEF.items(): setattr(Config, k, v)
cfg = Config(); cfg.seed = 0
for k, v in ALL.items(): setattr(cfg, k, v)
for k, v in {**S.LOCAL, **S.CONFIGS.get('Q both, 10x3', {}), **ovr}.items(): setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
t0 = time.time()
with open(log, 'w') as f, contextlib.redirect_stdout(f):
    r = run(cfg)
    print(f"MEAN FORGETTING: {r['forget']:+.3f} | MEAN FINAL BPC: {r['bpc']:.3f}")
print(f"WALL {time.time()-t0:.0f}s", file=sys.stderr)
