"""Compare two run logs with wall-clock readings masked. python3 compare.py A.log B.log [N lines]
Exit 0 if identical after masking. Masked: times (s, ms, min, steps/s), host memory, the [timing] lines' figures."""
import re, sys
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
a = open(sys.argv[1]).read().split('\n'); b = open(sys.argv[2]).read().split('\n')
n = int(sys.argv[3]) if len(sys.argv) > 3 else max(len(a), len(b))
bad = 0
for i in range(min(n, max(len(a), len(b)))):
    x = norm(a[i]) if i < len(a) else '<missing>'; y = norm(b[i]) if i < len(b) else '<missing>'
    if x != y:
        bad += 1
        if bad <= 8:
            k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
            print(f'line {i+1}:\n  A ...{x[max(0,k-80):k+120]}\n  B ...{y[max(0,k-80):k+120]}')
print(f'{min(n, max(len(a), len(b)))} lines compared, {bad} differ after masking times')
sys.exit(1 if bad else 0)
