#!/usr/bin/env bash
# The power of one scenario again with a liberty wire-load model instead of no wire capacitance (synthesis netlist
# only; a bracket until the hardened layout's SPEF is available). Usage: power_wl.sh <gls dir> <netlist> <Small|Large>
set -euo pipefail
cd "$(dirname "$0")"
G="$(readlink -f "$1")"; NL="$(readlink -f "$2")"; WL="$3"
source ./env.sh pdk librelane
mkdir -p "$G/sta_wl_$WL"
env TOP=${TOP:-snn_core_meas} NETLIST="$NL" WIRE_LOAD="$WL" LIB_SC="$LIB" PERIOD_NS=14.925 OUT="$G/sta_wl_$WL" ACT_TCL="$G/activity.tcl" \
  nix --extra-experimental-features "nix-command flakes" develop --accept-flake-config "$LIBRELANE" -c \
  sta -no_splash -exit tools/power_sta.tcl > "$G/sta_wl_$WL/sta.log" 2>&1 || { tail -n 20 "$G/sta_wl_$WL/sta.log"; exit 1; }
grep -E "^Total" "$G/sta_wl_$WL/power_vcd.rpt"; head -4 "$G/sta_wl_$WL/net_example.rpt"
