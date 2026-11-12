# one line per overnight job: DONE / RUNNING / PENDING, plus heavy-job count
S=/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad
st() { f=$S/$2; if case $2 in cg*) sed -n "/= COMPRESSION GAIN/,\$p" $f 2>/dev/null | grep -q "^DONE";; *) grep -q "^DONE" $f 2>/dev/null;; esac; then echo "DONE   "; elif pgrep -f "$3" >/dev/null; then echo "RUNNING"; else echo "PENDING"; fi; }
echo "A1 compression gain (128)   $(st A cg_full.txt 'compress_gain.py 4000')  $(grep -o 'step [0-9]*: [0-9.]* bits' $S/cg_full.txt 2>/dev/null | tail -1)"
echo "A2 relation identity (128)  $(st A relsub_full.txt 'relsub.py 4000')"
echo "A3 pronoun reference (128)  $(st A pronoun_full.txt 'pronoun_ref.py 4000')"
echo "B  refusal toy              $(st B toy_refuse_full.txt toy_refuse.py)  $(grep -c '^  ' $S/toy_refuse_full.txt 2>/dev/null) of 4 conditions"
echo "C  compression gain (192)   $(st C cg192_full.txt 'compress_gain.py 6000')  $(grep -o 'step [0-9]*: [0-9.]* bits' $S/cg192_full.txt 2>/dev/null | tail -1)  model saved: $([ -f $S/tg_lm192_6000.pt ] && echo yes || echo no)"
echo "D1 relation identity (192)  $(st D relsub192_full.txt 'relsub.py 6000')"
echo "D2 pronoun reference (192)  $(st D pronoun192_full.txt 'pronoun_ref.py 6000')"
L=$S/cl/logs/GATEL.log; echo "GATEL harness check          $(grep -q 'MEAN FINAL' $L && echo DONE || (pgrep -f 'harness_srv.py.*GATEL' >/dev/null && echo RUNNING || echo STOPPED)) visit $(grep -c '=== Phase' $L), errors $(grep -c -E 'Traceback|unavailable|a step failed' $L)"
echo "heavy jobs now: $(pgrep -fc '^python3 (harness_srv|tools/)')"
