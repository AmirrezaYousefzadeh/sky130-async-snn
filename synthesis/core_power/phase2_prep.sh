#!/usr/bin/env bash
# Plain-name copies of the accepted layout's netlist, typical-corner SDF and nominal SPEF, for the
# annotated simulation and the power analysis. Usage: phase2_prep.sh <run dir> <out dir>
set -euo pipefail
cd "$(dirname "$0")"
RUN="$(readlink -f "$1")"; OUT="$2"; mkdir -p "$OUT"
NL="$RUN/final/nl/snn_core_meas.nl.v"
SDF="$(ls "$RUN"/final/sdf/nom_tt_025C_1v80/*.sdf | head -n 1)"
SPEF="$(ls "$RUN"/final/spef/nom/*.spef | head -n 1)"
python3 tools/legal_names.py "$NL" "$SDF" "$SPEF" "$OUT"
echo "escaped names left: $(grep -c '\\\\[^ ]*[./]' "$OUT/design.nl.v" || true)"
grep -c "sky130_fd_sc_hd__" "$OUT/design.nl.v"
