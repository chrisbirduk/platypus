"""Checkpoint equivalence test: a run stopped after visit K and resumed must print what the uninterrupted run prints.

usage: python tools/ckpt/equiv.py OUT_DIR [K ...]
  Runs tools/ckpt/tiny_run.py (CPU, local text) with the preset in PRESET (default configs/presets/cg26q.env) and
  PLATYPUS_PASSES (default 2): once straight through, and once per K stopped after visit K (exit 75) and resumed.
  Each resumed log, from its "[checkpoint] resumed after visit K" line on, is compared with the uninterrupted log
  from its "[checkpoint] saved after visit K" line on, with times and checkpoint sizes masked.
Exit 0 when every comparison matches.

One run at a time (each fits a 2-hour job limit), for the REPEATABILITY check (26 Nov):
  python tools/ckpt/equiv.py run OUT_DIR NAME [K]     one run; with K it stops after visit K (exit 75)
  python tools/ckpt/equiv.py resume OUT_DIR NAME      continue NAME from its checkpoint (NAME.ckpt) to the end
  python tools/ckpt/equiv.py same OUT_DIR A B         compare the whole logs of A and B (each A.log, plus Ar.log after a resume)
Every run sets PYTHONHASHSEED (EQ_HASHSEED, default 0). Two runs with different EQ_HASHSEED must print the same lines:
the training must not depend on Python's per-process set order.
"""
import os, re, subprocess, sys, shutil
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
SUB = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('run', 'resume', 'same') else None
ARGS = sys.argv[2:] if SUB else sys.argv[1:]
OUT = os.path.abspath(ARGS[0]); KS = ([int(k) for k in ARGS[1:]] or [3]) if not SUB else []
os.makedirs(OUT, exist_ok=True)


def env_for(ckpt, stop=None):
    e = dict(os.environ)
    for line in open(os.path.join(REPO, os.environ.get('PRESET', 'configs/presets/cg26q.env'))):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1); e.setdefault(k, v)
    e['PYTHONHASHSEED'] = os.environ.get('EQ_HASHSEED', '0'); e['PLATYPUS_PASSES'] = os.environ.get('PLATYPUS_PASSES', '2'); e['PLATYPUS_CKPT'] = ckpt; e['PLATYPUS_CKPT_STRICT'] = '1'
    e.pop('PLATYPUS_CKPT_STOP_AFTER', None)
    if stop is not None: e['PLATYPUS_CKPT_STOP_AFTER'] = str(stop)
    return e


def run(name, stop=None, ckpt=None, fresh=True):
    ck = ckpt or os.path.join(OUT, name + '.ckpt')
    if fresh: shutil.rmtree(ck, ignore_errors=True)
    with open(os.path.join(OUT, name + '.log'), 'w') as f:
        p = subprocess.run([sys.executable, os.path.join(HERE, 'tiny_run.py')], env=env_for(ck, stop), stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    return p.returncode


# the masks of tools/refactor/compare.py (times, host memory, timing lines)
PATS = [(re.compile(r'\[elapsed [0-9.]+ min\]'), '[elapsed T min]'),
        (re.compile(r'\b[0-9]+(\.[0-9]+)?(e[-+]?[0-9]+)? ?(ms|s|min)\b'), 'T'),
        (re.compile(r'\b[0-9]+(\.[0-9]+)? steps/s'), 'R steps/s'),
        (re.compile(r'host process [0-9.]+ GB( \([-+][0-9.]+\))?'), 'host process M GB'),
        (re.compile(r'(\[timing\][^\n]*)'), lambda m: re.sub(r'[0-9]+(\.[0-9]+)?', 'N', m.group(1))),
        (re.compile(r'(\[delta solve\][^\n]*)'), lambda m: re.sub(r'[0-9]+(\.[0-9]+)?', 'N', m.group(1))),
        (re.compile(r'(SLEEP TIMING[^\n]*|TIMING\][^\n]*)'), lambda m: re.sub(r'[0-9]+(\.[0-9]+)?', 'N', m.group(1)))]


def norm(l):
    for p, r in PATS: l = p.sub(r, l)
    return l


CK = re.compile(r'(\[checkpoint\] saved after visit [0-9]+ of [0-9]+): .*')


def tail(path, marker):
    L = open(path, encoding='utf-8', errors='replace').read().split('\n')
    i = next((j for j, l in enumerate(L) if l.startswith(marker)), None)
    if i is None: raise SystemExit(f'{path}: no line starting {marker!r}')
    return [CK.sub(r'\1', l) for l in L[i + 1:]]


def main():
    jobs = [('straight', None)] + [(f'stop{k}', k) for k in KS]
    with ThreadPoolExecutor(len(jobs)) as ex:
        rc = dict(zip([j[0] for j in jobs], ex.map(lambda j: run(*j), jobs)))
    if rc['straight'] != 0: raise SystemExit(f"straight run failed (exit {rc['straight']}): see {OUT}/straight.log")
    bad = 0
    with ThreadPoolExecutor(len(KS)) as ex:
        rr = dict(zip(KS, ex.map(lambda k: (rc[f'stop{k}'], run(f'resume{k}', ckpt=os.path.join(OUT, f'stop{k}.ckpt'), fresh=False)), KS)))
    for k in KS:
        r1, r2 = rr[k]
        if r1 != 75 or r2 != 0: print(f'K={k}: stop exit {r1} (want 75), resume exit {r2} (want 0): see {OUT}/stop{k}.log, resume{k}.log'); bad += 1; continue
        a = tail(os.path.join(OUT, 'straight.log'), f'  [checkpoint] saved after visit {k} ')
        b = tail(os.path.join(OUT, f'resume{k}.log'), f'  [checkpoint] resumed after visit {k} ')
        n = max(len(a), len(b)); d = [i for i in range(n) if (norm(a[i]) if i < len(a) else None) != (norm(b[i]) if i < len(b) else None)]
        print(f'K={k}: {n} lines after the resume point, {len(d)} differ')
        for i in d[:5]:
            print(f'  +{i+1}\n    straight: {a[i][:220] if i < len(a) else "<missing>"}\n    resumed:  {b[i][:220] if i < len(b) else "<missing>"}')
        bad += bool(d)
    sys.exit(1 if bad else 0)


def whole(name):
    """the log of run NAME; after a resume, the stopped part up to its last save, then the resumed part after the resume line"""
    a = open(os.path.join(OUT, name + '.log'), encoding='utf-8', errors='replace').read().split('\n')
    rp = os.path.join(OUT, name + 'r.log')
    if os.path.exists(rp):
        b = open(rp, encoding='utf-8', errors='replace').read().split('\n')
        j = next((i for i, l in enumerate(b) if '[checkpoint] resumed after visit' in l), None)
        i = max((i for i, l in enumerate(a) if '[checkpoint] saved after visit' in l), default=None)
        if i is None or j is None: raise SystemExit(f'{name}: no save line in {name}.log or no resume line in {name}r.log')
        a = a[:i + 1] + b[j + 1:]
    return [norm(CK.sub(r'\1', l)) for l in a]


def sub_main():
    if SUB == 'run':
        k = int(ARGS[2]) if len(ARGS) > 2 else None; rc = run(ARGS[1], stop=k)
        print(f'{ARGS[1]}: exit {rc} (want {75 if k is not None else 0})'); sys.exit(0 if rc == (75 if k is not None else 0) else 1)
    if SUB == 'resume':
        rc = run(ARGS[1] + 'r', ckpt=os.path.join(OUT, ARGS[1] + '.ckpt'), fresh=False); print(f'{ARGS[1]} resumed: exit {rc}'); sys.exit(0 if rc == 0 else 1)
    a, b = whole(ARGS[1]), whole(ARGS[2]); n = max(len(a), len(b))
    d = [i for i in range(n) if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)]
    print(f'{ARGS[1]} vs {ARGS[2]}: {n} lines, {len(d)} differ')
    for i in d[:8]:
        print(f'  line {i+1}\n    {ARGS[1]}: {a[i][:220] if i < len(a) else "<missing>"}\n    {ARGS[2]}: {b[i][:220] if i < len(b) else "<missing>"}')
    sys.exit(1 if d else 0)


if __name__ == '__main__':
    sub_main() if SUB else main()
