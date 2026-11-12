# stage D (resumable): relation identity and pronoun reference on the larger trunk (needs its saved model)
cd /home/claude/platypus-jt
S=/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad
export TG_D=192 TG_CKPT=$S/tg_lm192_6000.pt
[ -f $TG_CKPT ] || { echo "no 192 model yet"; exit 1; }
grep -q "^DONE" $S/relsub192_full.txt 2>/dev/null || python3 tools/relsub.py 6000 500 > $S/relsub192_full.txt 2>&1
grep -q "^DONE" $S/pronoun192_full.txt 2>/dev/null || python3 tools/pronoun_ref.py 6000 500 > $S/pronoun192_full.txt 2>&1
echo "stage D done $(date)"
