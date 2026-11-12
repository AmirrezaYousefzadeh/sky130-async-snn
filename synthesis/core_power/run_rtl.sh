#!/usr/bin/env bash
# RTL simulation of every scenario against the reference model of gen_core.py (light: run directly).
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
mkdir -p runs/rtl
{ iverilog -V 2>&1 || true; } | head -1
declare -A DEF=([spike]="-DMODE_EVENT=1" [spike_ref]="-DMODE_EVENT=1 -DUSE_REF=1" [add]="-DMODE_EVENT=0" [leak]="" [thr]="" [spike_rand]="" [idle]="")
for sc in ${SCEN:-spike spike_ref add leak thr spike_rand idle}; do
  iverilog -g2012 -o runs/rtl/$sc.vvp -I rtl -DSTIM="\"stim/$sc.hex\"" -DOUT_FILE="\"runs/rtl/$sc.out\"" ${DEF[$sc]} rtl/snn_core_meas.v rtl/core_rom.v tb/tb_core.v 2> runs/rtl/$sc.iverilog.log || { cat runs/rtl/$sc.iverilog.log; exit 1; }
  vvp -n runs/rtl/$sc.vvp > runs/rtl/$sc.log 2>&1
  if cmp -s runs/rtl/$sc.out stim/$sc.expect && grep -q "^PASS" runs/rtl/$sc.log; then r=MATCH; else r=MISMATCH; fi
  echo "$sc: $r ($(wc -l < runs/rtl/$sc.out) spikes, expected $(wc -l < stim/$sc.expect)); $(grep 'TB: window [0-9]' runs/rtl/$sc.log | sed 's/TB: //')"
done
