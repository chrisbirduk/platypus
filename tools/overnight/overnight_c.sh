# stage C (resumable): larger stand-in trunk (192 wide, 6000 steps; saved right after training, reloaded on a restart)
cd /home/claude/platypus-jt
S=/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad
sed -n "/= COMPRESSION GAIN/,\$p" $S/cg192_full.txt 2>/dev/null | grep -q "^DONE" || TG_D=192 TG_CKPT=$S/tg_lm192_6000.pt python3 tools/compress_gain.py 6000 500 > $S/cg192_full.txt 2>&1
echo "stage C done $(date)"
