#!/bin/bash
# Reprioritized Gate 4, second ordering (user request 2026-09-23): after PR-2n/16 (already running) finishes,
# the post-hoc layers-0-1-EXACT variant, then OPT/32 if time remains. rho = 25%, ctx 2048, 48-chunk subset.
cd /home/jl_fs/rope_equiv; source /home/jl_fs/venv/bin/activate; export HF_HOME=/home/jl_fs/hf
run() { echo "=== $(date +%T) $*"; python -u -m kernel_fact.eval_e2e "$@" 2>&1 | grep -E '^\{|Traceback|Error'; }
while pgrep -f "[e]val_e2e ppl --arm PR-2n --p 16" >/dev/null; do sleep 5; done
run ppl --arm OPT --p 16 --rho 0.25 --ctx 2048 --max_chunks 48 --exact_layers 0,1   # post-hoc variant
run ppl --arm OPT --p 32 --rho 0.25 --ctx 2048 --max_chunks 48
echo "=== REPRIO DONE $(date +%T)"
