#!/usr/bin/env bash
# Phase 2 on the accepted layout: renamed copies, own sign-off at 67 MHz tt, then the SDF-annotated and zero-delay
# simulations of the short scenarios with SPEF-based power (one thread each, up to an hour per scenario; run them in
# parallel if the machine allows). Usage: phase2.sh <run dir>
set -euo pipefail
cd "$(dirname "$0")"
./phase2_prep.sh "$1" runs/p2
./signoff67.sh runs/p2
mkdir -p logs
for sc in spike_s leak_s thr_s add_s idle_s idle_on_s; do
  ./measure.sh $sc runs/p2/design.nl.v runs/gls_sdf "$PWD/runs/p2/design.sdf" "$PWD/runs/p2/design.spef" > logs/p2_$sc.log 2>&1
done
for sc in spike_s leak_s idle_on_s; do
  ./measure.sh $sc runs/p2/design.nl.v runs/gls_p2zd '' "$PWD/runs/p2/design.spef" > logs/p2zd_$sc.log 2>&1
done
