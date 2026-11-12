#!/usr/bin/env bash
# Hardening of the measurement core with tools/harden_ladder.sh,
# LibreLane 3.0.16, utilisations dense to sparse, the clock gates fixed in place before CTS (320 gates on the
# core's gated clock).
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh pdk librelane
export LIBRELANE PDK_ROOT PRE_CTS_TCL="$PWD/tools/firm_clock_gates.tcl"
tools/harden_ladder.sh config.yaml harden "${1:-30 20}"
