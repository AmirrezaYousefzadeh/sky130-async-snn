#!/usr/bin/env bash
# One scenario end to end: gate-level simulation (gls.sh) then OpenSTA power (power.sh).
# Usage: measure.sh <scenario> <netlist> <runs dir> [<sdf> <spef>]
set -euo pipefail
cd "$(dirname "$0")"
SC="$1"; NL="$2"; RUNS="$3"; SDF="${4:-}"; SPEF="${5:-}"
if [[ -n "$SDF" ]]; then ./gls.sh "$SC" "$NL" "$RUNS/$SC" --sdf "$SDF"; else ./gls.sh "$SC" "$NL" "$RUNS/$SC"; fi
./power.sh "$RUNS/$SC" "$NL" "$SPEF"
