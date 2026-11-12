#!/usr/bin/env bash
# Gate-level simulation of the measurement core, one scenario, the waveform of the window streamed into the toggle
# counter. Zero-delay functional models by default; --sdf <file> annotates typical-corner cell delays
# (the SDF is sanitised for Icarus and the run fails on any annotation error).
# Usage: gls.sh <scenario> <netlist> <out dir> [--sdf <sdf>]
# Scenarios: ospike oleak oidle oidle_on
set -euo pipefail
cd "$(dirname "$0")"
SC="$1"; NL="$(readlink -f "$2")"; OUT="$3"; SDF="${5:-}"
source ../env.sh pdk
mkdir -p "$OUT"; OUT="$(readlink -f "$OUT")"
STIM="$SC"; DEFS=()
case "$SC" in
  ospike|oleak) ;;
  oidle) DEFS+=(-DCG_OFF_WINDOW=0) ;;
  oidle_on) STIM=oidle; DEFS+=(-DCG_OFF_WINDOW=1) ;;
  *) echo "unknown scenario $SC"; exit 2 ;;
esac
DEFS+=(-DSTIM="\"$PWD/stim/$STIM.hex\"" -DOUT_FILE="\"$OUT/out.txt\"" -DDUMP_PATH="\"$OUT/wave.vcd\"" -DCLK_PERIOD_NS=14.925)
FLAGS=(-g2012); SRCS=("$PDKV/primitives.v")
if [[ -n "$SDF" ]]; then
  python3 tools/sdf_sanitize_for_icarus.py "$SDF" -o "$OUT/design.icarus.sdf" > "$OUT/sdf_sanitize.log"
  DEFS+=(-DSDF_ANNOTATE="\"$OUT/design.icarus.sdf\"" -DUNIT_DELAY='#1'); FLAGS+=(-gspecify -Ttyp); SRCS+=(tools/sky130_timing_icarus_fixes.v)
else
  DEFS+=(-DFUNCTIONAL -DUNIT_DELAY='#1')
fi
python3 tools/gen_phys_stubs.py "$NL" "$OUT/phys_stubs.v" "$PDKV/primitives.v" "$PDKV/sky130_fd_sc_hd.v" > /dev/null
SRCS+=("$PDKV/sky130_fd_sc_hd.v" "$OUT/phys_stubs.v" "$NL" tb/tb_out.v)
echo "==> $SC: compiling ($([[ -n "$SDF" ]] && echo sdf || echo zero-delay)) $(date +%H:%M:%S)"
iverilog "${FLAGS[@]}" -o "$OUT/gls.vvp" -I rtl "${DEFS[@]}" "${SRCS[@]}" 2> "$OUT/iverilog.log" || { tail -n 20 "$OUT/iverilog.log"; exit 1; }
rm -f "$OUT/wave.vcd"; mkfifo "$OUT/wave.vcd"
tools/vcd_toggles -o "$OUT/toggles.tsv" "$OUT/wave.vcd" 2> "$OUT/toggles.log" &
TPID=$!
echo "==> $SC: running $(date +%H:%M:%S)"
( time vvp -n "$OUT/gls.vvp" ) > "$OUT/sim.log" 2>&1 || true
if kill -0 $TPID 2> /dev/null && ! grep -q "window start" "$OUT/sim.log"; then timeout 10 bash -c ": > '$OUT/wave.vcd'" || kill $TPID 2> /dev/null; fi
wait $TPID || true
rm -f "$OUT/wave.vcd"
SDF_ERRORS=$(grep -c -E "SDF ERROR|Unable to match ModPath" "$OUT/sim.log" || true)
if cmp -s "$OUT/out.txt" "stim/$STIM.expect" && grep -q "^PASS" "$OUT/sim.log"; then M=MATCH; else M=MISMATCH; fi
echo "==> $SC: $M against the reference ($(wc -l < "$OUT/out.txt") spikes); SDF errors $SDF_ERRORS; $(grep 'TB: window [0-9]' "$OUT/sim.log" | sed 's/TB: //'); $(grep real "$OUT/sim.log" | tail -1)"
cat "$OUT/toggles.log"
[[ "$M" == MATCH && "$SDF_ERRORS" -eq 0 ]]
