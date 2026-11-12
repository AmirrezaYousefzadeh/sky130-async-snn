# Per-pin activity power of a routed netlist with OpenSTA (run inside the LibreLane environment).
# Environment: TOP NETLIST SPEF LIB_SC (one or more liberties, space-separated) PERIOD_NS OUT ACT_TCL; for SRAM22 designs
# LIB_SRAM and MACRO_INST; optionally EXTRA_ACT_TCL (activities that override the measured ones: the macro state
# correction), CLK_PORT (default clk) and ZERO_DATA=1 (every data activity zero, the clock at its defined rate, no
# activity file: run B of root_clock_correction_multipdk.py).
# PERIOD_NS is in the time unit of the liberties (ns on sky130, ps on ASAP7) and must be the period the simulation used:
# OpenSTA converts per-cycle activities with it.
# Writes OUT/power_vcd.rpt (the group table) and OUT/power_vcd_by_instance.rpt (one line per instance).
set OUT $::env(OUT)
foreach l $::env(LIB_SC) { read_liberty $l }
if {[info exists ::env(MACRO_INST)] && $::env(MACRO_INST) ne ""} { read_liberty $::env(LIB_SRAM) }
read_verilog $::env(NETLIST)
link_design $::env(TOP)
if {[info exists ::env(SPEF)] && $::env(SPEF) ne ""} { read_spef $::env(SPEF) } elseif {[info exists ::env(WIRE_LOAD)] && $::env(WIRE_LOAD) ne ""} {
  set_wire_load_mode top; set_wire_load_model -name $::env(WIRE_LOAD); puts "NO_SPEF: liberty wire-load model $::env(WIRE_LOAD)"
} else { puts "NO_SPEF: pin capacitances only (no wire capacitance)" }

set period $::env(PERIOD_NS)
set clkport [expr {[info exists ::env(CLK_PORT)] ? $::env(CLK_PORT) : "clk"}]
create_clock -name clk -period $period -waveform [list 0.0 [expr {$period / 2.0}]] [get_ports $clkport]
set_propagated_clock [all_clocks]

file mkdir $OUT
# input ports without an annotation get activity 0 instead of OpenSTA's default (0.1 transitions per cycle); every port and
# pin the simulation saw is annotated by the activity file
set_power_activity -input -activity 0.0 -duty 0.0
if {[info exists ::env(ZERO_DATA)] && $::env(ZERO_DATA)} {
  set_power_activity -global -activity 0.0 -duty 0.0
  puts "ZERO_DATA: every data activity zero, clocks at their defined rate"
} else {
  puts "sourcing activities $::env(ACT_TCL)"
  source $::env(ACT_TCL)
  if {[info exists ::env(EXTRA_ACT_TCL)]} { source $::env(EXTRA_ACT_TCL); puts "EXTRA_ACT_TCL: $::env(EXTRA_ACT_TCL)" }
}
report_power -digits 6 > $OUT/power_vcd.rpt
report_power -instances [get_cells *] -digits 4 > $OUT/power_vcd_by_instance.rpt
puts "WROTE $OUT/power_vcd.rpt"
# measurement-core addition: the capacitance of one net (does a wire load apply without SPEF?)
catch { report_net -digits 4 [get_full_name [lindex [get_nets -quiet *] 0]] > $OUT/net_example.rpt }
