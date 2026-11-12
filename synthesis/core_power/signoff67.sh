#!/usr/bin/env bash
# Own sign-off of the hardened layout at the typical corner with the 67 MHz clock (14.925 ns), on the renamed netlist
# and nominal SPEF (tools/signoff_sta.tcl). Usage: signoff67.sh <p2 dir>
set -euo pipefail
cd "$(dirname "$0")"
P="$(readlink -f "$1")"
source ./env.sh pdk librelane
env LIB_SC="$LIB" NETLIST="$P/design.nl.v" SPEF="$P/design.spef" TOP=snn_core_meas SDC="$PWD/core.sdc" CLOCK_PERIOD=14.925 OUT="$P/signoff67" \
  nix --extra-experimental-features "nix-command flakes" develop --accept-flake-config "$LIBRELANE" -c \
  sta -no_splash -exit tools/signoff_sta.tcl > "$P/signoff67.log" 2>&1 || { tail "$P/signoff67.log"; exit 1; }
grep -E "SIGNOFF_NS|not found" "$P/signoff67.log" | head
