# Sign-off timing of a routed run at one corner (the liberties, SPEF and SDC given), with the parasitics of every net.
# LibreLane's own sign-off reads the SPEF of the original netlist, whose escaped hierarchical names it cannot match
# (flow/tools/legal_names.py): the nets inside sub-modules would be timed without their wires. This script reads the
# renamed netlist and SPEF instead, so that every net has its parasitics.
# Env: LIB_SC (liberty files), LIB_MACROS (optional), NETLIST, SPEF, TOP, SDC (and the variables it reads), OUT
# Output: "SIGNOFF <key> <value>" in seconds (OpenSTA's Tcl slack commands return SI seconds whatever the liberty's
# time unit) and "SIGNOFF_NS <key> <value>" in ns; a script that reads the first as ns or ps is off by 1e9 or 1e12.
foreach l $::env(LIB_SC) { read_liberty $l }
if {[info exists ::env(LIB_MACROS)]} { foreach l $::env(LIB_MACROS) { read_liberty $l } }
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_spef $::env(SPEF)
read_sdc $::env(SDC)
# as LibreLane does: an SDC that says nothing about clock propagation gets propagated clocks
set sdc_text [read [set fh [open $::env(SDC)]]]; close $fh
if { ![string match "*set_propagated_clock*" $sdc_text] } {
  puts "signoff_sta: the SDC does not propagate the clocks: propagating all clocks"
  set_propagated_clock [all_clocks]
}
file mkdir $::env(OUT)
report_checks -path_delay max -group_count 20 -fields {slew cap input_pins} -digits 4 > $::env(OUT)/max.rpt
report_checks -path_delay min -group_count 20 -fields {slew cap input_pins} -digits 4 > $::env(OUT)/min.rpt
report_check_types -max_slew -max_capacitance -violators > $::env(OUT)/slew_cap_violators.rpt
puts "SIGNOFF setup_ws [sta::worst_slack_cmd max]"
puts "SIGNOFF hold_ws [sta::worst_slack_cmd min]"
puts "SIGNOFF setup_tns [sta::total_negative_slack_cmd max]"
puts "SIGNOFF hold_tns [sta::total_negative_slack_cmd min]"
puts "SIGNOFF_NS setup_ws [expr {[sta::worst_slack_cmd max] * 1e9}]"
puts "SIGNOFF_NS hold_ws [expr {[sta::worst_slack_cmd min] * 1e9}]"
puts "SIGNOFF_NS setup_tns [expr {[sta::total_negative_slack_cmd max] * 1e9}]"
puts "SIGNOFF_NS hold_tns [expr {[sta::total_negative_slack_cmd min] * 1e9}]"
exit
