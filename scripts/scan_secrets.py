#!/usr/bin/env python3
"""Scan files and folders for keys and tokens before anything is shared.

Exit status 0 = clean, 1 = something found (the matches are printed with
the secret masked), 2 = usage error. Binary files are skipped; zip archives
are opened and their members scanned.

Usage:
    scan_secrets.py PATH [PATH ...]
"""
import os
import re
import sys
import zipfile

PATTERNS = [
    ("Anthropic API key", r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    ("OpenAI-style key", r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}"),
    ("Hugging Face token", r"\bhf_[A-Za-z0-9]{30,}"),
    ("GitHub token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,}"),
    ("AWS access key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("Google API key", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    ("Slack token", r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    ("Private key block", r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ("Key assigned in code", r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"][A-Za-z0-9_\-\.]{24,}['\"]"),
]
RX = [(n, re.compile(p)) for n, p in PATTERNS]
SKIP_DIRS = {".git", "__pycache__", "pile_cache", ".ipynb_checkpoints"}
MAX_BYTES = 50 * 2**20


def mask(s):
    return s[:6] + "..." + s[-2:] if len(s) > 10 else "***"


def scan_text(label, text, hits):
    for i, line in enumerate(text.splitlines(), 1):
        for name, rx in RX:
            for m in rx.finditer(line):
                hits.append(f"{label}:{i}: {name}: {mask(m.group(0))}")


def scan_file(path, hits):
    if os.path.getsize(path) > MAX_BYTES:
        print(f"  skipped (over 50 MB): {path}")
        return
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or info.file_size > MAX_BYTES:
                    continue
                data = z.read(info)
                if b"\0" not in data[:4096]:
                    scan_text(f"{path}!{info.filename}", data.decode("utf-8", "replace"), hits)
        return
    data = open(path, "rb").read()
    if b"\0" in data[:4096]:
        return
    scan_text(path, data.decode("utf-8", "replace"), hits)


def main(paths):
    if not paths:
        print(__doc__)
        return 2
    hits, n = [], 0
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fn in files:
                    scan_file(os.path.join(root, fn), hits)
                    n += 1
        elif os.path.isfile(p):
            scan_file(p, hits)
            n += 1
        else:
            print(f"  not found: {p}")
    if hits:
        print(f"SECRET SCAN: {len(hits)} possible secret(s) in {n} file(s):")
        print("\n".join("  " + h for h in hits))
        return 1
    print(f"secret scan: clean ({n} file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
