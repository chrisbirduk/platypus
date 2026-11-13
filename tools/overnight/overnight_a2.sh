# stage A (resumable): stand-in trunk 4000 steps (loads the saved model if present), compression gain, relation identity, pronoun reference
cd /home/claude/platypus-jt
S=/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad
export TG_CKPT=$S/tg_lm4000.pt
sed -n "/= COMPRESSION GAIN/,\$p" $S/cg_full.txt 2>/dev/null | grep -q "^DONE" || python3 tools/compress_gain.py 4000 500 > $S/cg_full.txt 2>&1
grep -q "^DONE" $S/relsub_full.txt 2>/dev/null || python3 tools/relsub.py 4000 500 > $S/relsub_full.txt 2>&1
grep -q "^DONE" $S/pronoun_full.txt 2>/dev/null || python3 tools/pronoun_ref.py 4000 500 > $S/pronoun_full.txt 2>&1
echo "stage A done $(date)"
