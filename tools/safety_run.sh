#!/bin/bash
# LOCAL SAFETY RUN of a cg27a_kra* build, exactly as kra13l-kra15l were run (20 Nov 2026): 150 steps a domain, the 10k distant set, the small
# route-J sizes (jx 100 steps, 3 layers, 128 dims) and the oracle-write / parse settings - so HEADLINE compares with -0.041 / 2.615 (kra14l, kra15l).
# Usage: tools/safety_run.sh <build, e.g. kra17> <scratch dir>   -> <scratch>/lr_<build>l.log
B=$1; S=$2; cd "$(dirname "$0")/.."
LR_STEPS=150 LR_PRESET=configs/presets/cg27a_$B.overrides.json LR_DISTANT=$S/distant/distant_10k.json.gz \
LR_OVR='{"inf_kmc_rulecomp_oracle_write": 1, "inf_kmc_rulej_mlp_steps": 800, "inf_premises_parse": 0, "inf_jx_steps": 100, "inf_jx_layers": 3, "inf_jx_dim": 128}' \
python3 tools/local_run.py > $S/lr_${B}l.log 2>&1
grep -E "HEADLINE|Traceback" $S/lr_${B}l.log | cut -c1-200
