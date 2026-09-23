#!/bin/bash
# Gate 4 queue. Usage: bash kernel_fact/run_gate4.sh <phase>   (phases run sequentially inside)
# PPL subset: first 48 x 2048 and 24 x 4096 WikiText-2 test chunks (~98k tokens each), identical for all arms.
cd /home/jl_fs/rope_equiv
source /home/jl_fs/venv/bin/activate
export HF_HOME=/home/jl_fs/hf
run() { echo "=== $(date +%T) $*"; python -u -m kernel_fact.eval_e2e "$@" 2>&1 | grep -E '^\{|Traceback|Error|  gsm8k .*(/1319|00/)|  [a-z_]+: ' ; }

phase=$1
case $phase in
  baseline)   # uncompressed, full test set + the subset used for the arms
    for ctx in 2048 4096; do
      run ppl --arm off --ctx $ctx
      run ppl --arm off --ctx $ctx --max_chunks $((ctx == 2048 ? 48 : 24))
    done ;;
  ppl25|ppl125)
    rho=$([ $phase == ppl25 ] && echo 0.25 || echo 0.125)
    for ctx in 2048 4096; do
      mc=$((ctx == 2048 ? 48 : 24))
      run ppl --arm EXACT --rho $rho --ctx $ctx --max_chunks $mc
      run ppl --arm NOPE --rho $rho --ctx $ctx --max_chunks $mc
      for p in 8 16 32; do
        for arm in OPT PR-hi PR-en FOLD-mean; do
          run ppl --arm $arm --p $p --rho $rho --ctx $ctx --max_chunks $mc
        done
      done
      run ppl --arm PR-2n --p 16 --rho $rho --ctx $ctx --max_chunks $mc
    done ;;
  gsm8k)
    run gsm8k --arm off
    run gsm8k --arm EXACT
    run gsm8k --arm OPT --p 16
    for arm in PR-en PR-2n FOLD-mean PR-hi; do run gsm8k --arm $arm --p 16; done
    run gsm8k --arm OPT --p 8
    run gsm8k --arm OPT --p 32
    run gsm8k --arm NOPE ;;
  longbench)  # best special case at p=16 is passed as $2 (from Gate 3)
    run longbench --arm off
    run longbench --arm EXACT
    run longbench --arm OPT --p 16
    run longbench --arm $2 --p 16 ;;
esac
