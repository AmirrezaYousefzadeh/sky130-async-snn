# Measurement core: one clock at 67 MHz, inputs and outputs at 20 % of the period, reset the only false path.
# Derating 5 % (LibreLane's base.sdc divides by 100.0; written out here).
set period $::env(CLOCK_PERIOD)
create_clock -name clk -period $period [get_ports clk]
set_clock_uncertainty 0.25 [all_clocks]
set_clock_transition 0.15 [all_clocks]
set io_delay [expr {0.2 * $period}]
set ins [lsearch -inline -all -not -exact [all_inputs] [get_ports clk]]
set_input_delay $io_delay -clock clk $ins
set_output_delay $io_delay -clock clk [all_outputs]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin Y $ins
set_load 0.0334 [all_outputs]
set_false_path -from [get_ports rst]
set_timing_derate -early 0.95
set_timing_derate -late 1.05
