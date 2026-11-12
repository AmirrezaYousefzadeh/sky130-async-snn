// Icarus-friendly stubs for sky130 timing models used in SDF GLS.
// The PDK lpflow_bleeder timing cell references VPWR without declaring it
// when USE_POWER_PINS is off; skip the broken PDK body and provide a no-op.

`ifndef SKY130_FD_SC_HD__LPFLOW_BLEEDER_1_TIMING_V
`define SKY130_FD_SC_HD__LPFLOW_BLEEDER_1_TIMING_V
`endif
`ifndef SKY130_FD_SC_HD__LPFLOW_BLEEDER_1_TIMING_PP_V
`define SKY130_FD_SC_HD__LPFLOW_BLEEDER_1_TIMING_PP_V
`endif

`celldefine
module sky130_fd_sc_hd__lpflow_bleeder_1 (
    SHORT
`ifdef USE_POWER_PINS
  , VPWR
  , VGND
  , VPB
  , VNB
`endif
);
    input SHORT;
`ifdef USE_POWER_PINS
    input VPWR, VGND, VPB, VNB;
`endif
endmodule
`endcelldefine
