#!/bin/bash
# Reprioritized Gate 4 (user request 2026-09-23): WikiText-2 PPL only, rho = 25%, 60 GPU-minute time box.
cd /home/jl_fs/rope_equiv; source /home/jl_fs/venv/bin/activate; export HF_HOME=/home/jl_fs/hf
run() { echo "=== $(date +%T) $*"; python -u -m kernel_fact.eval_e2e "$@" 2>&1 | grep -E '^\{|Traceback|Error'; }
run ppl --arm OPT   --p 16 --rho 0.25 --ctx 2048 --max_chunks 48
run ppl --arm PR-2n --p 16 --rho 0.25 --ctx 2048 --max_chunks 48
run ppl --arm OPT   --p 32 --rho 0.25 --ctx 2048 --max_chunks 48
run ppl --arm OPT   --p 16 --rho 0.25 --ctx 2048 --max_chunks 48 --exact_layers 0,1   # post-hoc variant
run ppl --arm EXACT        --rho 0.25 --ctx 4096 --max_chunks 24
run ppl --arm OPT   --p 16 --rho 0.25 --ctx 4096 --max_chunks 24
run ppl --arm PR-2n --p 16 --rho 0.25 --ctx 4096 --max_chunks 24
run ppl --arm OPT   --p 16 --rho 0.25 --ctx 4096 --max_chunks 24 --exact_layers 0,1   # post-hoc variant
echo "=== REPRIO DONE $(date +%T)"
