#!/usr/bin/env bash
# Hardening at 25 ns (40 MHz) so that setup closes at every corner, including ss_100C_1v60; the power is measured at
# 67 MHz at the typical corner, after an own sign-off at tt with 14.925 ns (signoff67.sh).
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh pdk librelane
export LIBRELANE PDK_ROOT PRE_CTS_TCL="$PWD/tools/firm_clock_gates.tcl"
tools/harden_ladder.sh h25/config.yaml h25/work "${1:-30 20}"
