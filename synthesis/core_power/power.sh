#!/usr/bin/env bash
# OpenSTA power of one scenario from its toggle counts: typical liberty, one propagated clock at the
# simulated period, unannotated inputs at 0, measured per-pin activities, annotation check. The core's root clock runs
# throughout every window, so no root-clock correction applies.
# Usage: power.sh <gls out dir> <netlist> [spef]
set -euo pipefail
cd "$(dirname "$0")"
G="$(readlink -f "$1")"; NL="$(readlink -f "$2")"; SPEF="${3:+$(readlink -f "$3")}"
source ./env.sh pdk librelane
python3 tools/toggles_to_activity.py "$G/toggles.tsv" --scope tb/u_dut --period-s 14.925e-9 -o "$G/activity.tcl" --summary "$G/activity_summary.json"
mkdir -p "$G/sta"
env TOP=snn_core_meas NETLIST="$NL" SPEF="$SPEF" LIB_SC="$LIB" PERIOD_NS=14.925 OUT="$G/sta" ACT_TCL="$G/activity.tcl" \
  nix --extra-experimental-features "nix-command flakes" develop --accept-flake-config "$LIBRELANE" -c \
  sta -no_splash -exit tools/power_sta.tcl > "$G/sta/sta.log" 2>&1 || { tail -n 20 "$G/sta/sta.log"; exit 1; }
UNM=$(grep -c "spef.* not found" "$G/sta/sta.log" || true); BOX=$(grep "Creating black box" "$G/sta/sta.log" | grep -vE "tapvpwrvgnd|__fill_|__decap_" || true)
[[ "$UNM" -eq 0 ]] || { echo "$UNM unmatched SPEF entries"; exit 1; }
[[ -z "$BOX" ]] || { echo "black boxes: $BOX"; exit 1; }
python3 tools/check_annotation.py "$G/sta" "$NL" "$LIB" > "$G/sta/annotation.log" || { cat "$G/sta/annotation.log"; exit 1; }
tail -n 3 "$G/sta/annotation.log"
grep -E "^(Group|Sequential|Combinational|Clock|Macro|Pad|Total)" "$G/sta/power_vcd.rpt"
