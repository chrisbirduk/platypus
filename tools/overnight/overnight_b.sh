# stage B (resumable): the refusal shortcut toy
cd /home/claude/platypus-jt
S=/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad
grep -q "^DONE" $S/toy_refuse_full.txt 2>/dev/null || python3 tools/toy_refuse.py 3000 0 > $S/toy_refuse_full.txt 2>&1
echo "stage B done $(date)"
