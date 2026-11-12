#!/usr/bin/env bash
# Flatten the synthesised netlist (kept ROM modules) and give every escaped identifier a plain name, so that the
# simulation, the toggle names and OpenSTA all use flat plain instance names (check_annotation.py
# reads flat netlists). Same cells and nets as the synthesis output.
set -euo pipefail
cd "$(dirname "$0")"
NL="$1"; OUT="$2"; mkdir -p "$OUT"
source ./env.sh pdk
yosys -q -p "read_liberty -lib $LIB; read_verilog $NL; hierarchy -top snn_core_meas; flatten; opt_clean -purge; write_verilog -noattr $OUT/flat.nl.v; stat -liberty $LIB" > "$OUT/flatten.log" 2>&1 || { tail "$OUT/flatten.log"; exit 1; }
: > "$OUT/empty.sdf"; : > "$OUT/empty.spef"
python3 tools/legal_names.py "$OUT/flat.nl.v" "$OUT/empty.sdf" "$OUT/empty.spef" "$OUT"
grep -c "sky130_fd_sc_hd__" "$OUT/design.nl.v"
