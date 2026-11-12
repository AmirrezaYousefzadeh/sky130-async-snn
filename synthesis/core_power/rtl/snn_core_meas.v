// Measurement core: one core of docs/hardware.md, written for synthesis and power measurement in sky130
// (synthesis/core_power). 256 axons x 256 neurons, hard-wired int8 weight ROM (core_rom.v, one module per neuron),
// 256 words of {refractory bit, 16-bit membrane}, one neuron unit time-multiplexed over the 256 neurons, a 64-entry
// input FIFO of axon numbers. Clock gating with the library clock gate: one gate for the whole core (closed when the
// core is idle, unless cg_off); the FIFO behind a write gate, 8 group gates and one gate per entry; the membrane words
// behind 16 group gates and one gate per word (only the word being written, and its group, are clocked).
//
// Operations (one per 256 cycles, back to back; requests are served when the running operation ends):
//   ADDCMP (event-driven spike): v += w with saturation, threshold check after the add, hard reset, refractory bit
//   ADD    (stepped spike)     : v += w with saturation, no check (the stepped core checks at the tick)
//   LEAK   (tick)              : v -= v >>> K, refractory bits cleared
//   THR    (stepped tick)      : threshold scan, v >= THETA fires and resets
// Priority at every start: leak request, threshold request, then the FIFO head (as the tick wire of the spec).
// Pipeline: an operation registered at edge E reads neuron j (membrane word and ROM weight) in cycle E + 1 + j
// (stage A), computes and writes it back at the end of the next cycle (stage B); its spike is on out_valid one cycle
// after the write.
`timescale 1ns/1ps
module snn_core_meas (
  input  wire clk,
  input  wire rst,                 // synchronous, held for a few cycles at start
  input  wire cg_off,              // 1: the core clock gate stays open (idle core with a running clock)
  input  wire mode_event,          // 1: event-driven adds (ADDCMP); 0: stepped adds (ADD)
  input  wire use_ref,             // event-driven: one spike per neuron between leak ticks
  input  wire in_we,
  input  wire [7:0] in_axon,
  output wire in_ready,            // FIFO not full (back-pressure)
  input  wire tick,                // leak request (one-cycle pulse)
  input  wire thr_req,             // threshold-scan request (one-cycle pulse)
  output reg  out_valid,
  output reg  [7:0] out_neuron,
  output reg  op_start,            // one-cycle pulse: an operation was started at the last edge
  output wire idle
);
`include "core_params.vh"
  localparam [2:0] OP_NONE = 3'd0, OP_ADDCMP = 3'd1, OP_ADD = 3'd2, OP_LEAK = 3'd3, OP_THR = 3'd4;

  // ---------------------------------------------------------------- state clocked by the core clock
  reg        busy;
  reg  [7:0] j;
  reg  [2:0] op;
  reg  [7:0] axon_r;
  reg  [6:0] wr_ptr, rd_ptr;
  reg        leak_pend, thr_pend;
  reg        b_valid;
  reg  [2:0] b_op;
  reg  [7:0] b_j;
  reg  signed [15:0] b_v;
  reg        b_r;
  reg  signed [7:0]  b_w;

  wire fifo_empty = (wr_ptr == rd_ptr);
  wire fifo_full  = (wr_ptr[5:0] == rd_ptr[5:0]) && (wr_ptr[6] != rd_ptr[6]);
  assign in_ready = ~fifo_full;
  wire fifo_we = in_we & ~fifo_full;
  assign idle = ~busy & ~b_valid & ~out_valid & fifo_empty & ~leak_pend & ~thr_pend;

  // ---------------------------------------------------------------- core clock gate
  wire core_en = cg_off | rst | in_we | tick | thr_req | busy | b_valid | out_valid | op_start | leak_pend | thr_pend | ~fifo_empty;
  wire gclk;
  icg u_core_icg (.clk(clk), .en(core_en), .gclk(gclk));

  // ---------------------------------------------------------------- input FIFO: gated per write, per group of 8, per entry
  wire [8*64-1:0] fifo_flat;
  wire fifo_gck;
  icg u_fifo_icg (.clk(gclk), .en(fifo_we), .gclk(fifo_gck));
  genvar e, eg;
  generate for (eg = 0; eg < 8; eg = eg + 1) begin : g_fgrp
    wire grp_gck;
    icg u_icg (.clk(fifo_gck), .en(wr_ptr[5:3] == eg), .gclk(grp_gck));
    for (e = 0; e < 8; e = e + 1) begin : g_fifo
      wire gck;
      reg [7:0] q;
      icg u_icg (.clk(grp_gck), .en(wr_ptr[2:0] == e), .gclk(gck));
      always @(posedge gck) q <= in_axon;
      assign fifo_flat[8*(8*eg + e) +: 8] = q;
    end
  end endgenerate
  wire [7:0] head = fifo_flat[8*rd_ptr[5:0] +: 8];

  // ---------------------------------------------------------------- weight ROM and membrane words
  wire [2047:0] row;
  core_rom u_rom (.a(axon_r), .row(row));
  wire signed [7:0] w_rd = row[8*j +: 8];

  wire [16:0] wdata;                                   // {refractory, membrane} written back in stage B
  wire [17*256-1:0] mem_flat;
  genvar k, kg;
  generate for (kg = 0; kg < 16; kg = kg + 1) begin : g_wgrp   // 16 groups of 16 words: two levels of clock gates
    wire grp_gck;
    icg u_icg (.clk(gclk), .en(rst | (b_valid && (b_j[7:4] == kg))), .gclk(grp_gck));
    for (k = 0; k < 16; k = k + 1) begin : g_word
      wire gck;
      reg [16:0] q;
      icg u_icg (.clk(grp_gck), .en(rst | (b_j[3:0] == k)), .gclk(gck));
      always @(posedge gck) q <= wdata;
      assign mem_flat[17*(16*kg + k) +: 17] = q;
    end
  end endgenerate
  wire [16:0] rd_word = mem_flat[17*j +: 17];

  // ---------------------------------------------------------------- next operation
  wire last = busy && (j == 8'd255);
  wire can_start = ~busy | last;
  reg [2:0] nop;
  always @* begin
    if (leak_pend | tick) nop = OP_LEAK;
    else if (thr_pend | thr_req) nop = OP_THR;
    else if (~fifo_empty) nop = mode_event ? OP_ADDCMP : OP_ADD;
    else nop = OP_NONE;
  end
  wire start = can_start && (nop != OP_NONE);

  // ---------------------------------------------------------------- stage B: compute
  wire signed [17:0] s = {{2{b_v[15]}}, b_v} + {{10{b_w[7]}}, b_w};
  wire signed [15:0] sat = (s > 18'sd32767) ? 16'sd32767 : (s < -18'sd32768) ? -16'sd32768 : s[15:0];
  wire signed [15:0] lk = b_v - (b_v >>> KLEAK);
  reg  signed [15:0] nv; reg nr; reg fire;
  always @* begin
    nv = b_v; nr = b_r; fire = 1'b0;
    case (b_op)
      OP_ADDCMP: begin
        nv = sat;
        if ($signed({sat[15], sat}) >= THETA && !(use_ref && b_r)) begin fire = 1'b1; nv = 16'sd0; nr = b_r | use_ref; end
      end
      OP_ADD:  nv = sat;
      OP_LEAK: begin nv = lk; nr = 1'b0; end
      OP_THR:  if ($signed({b_v[15], b_v}) >= THETA) begin fire = 1'b1; nv = 16'sd0; end
      default: ;
    endcase
  end
  assign wdata = rst ? 17'd0 : {nr, nv};

  // ---------------------------------------------------------------- sequential part (core clock)
  always @(posedge gclk) begin
    if (rst) begin
      busy <= 1'b0; j <= 8'd0; op <= OP_NONE; axon_r <= 8'd0; wr_ptr <= 7'd0; rd_ptr <= 7'd0;
      leak_pend <= 1'b0; thr_pend <= 1'b0; b_valid <= 1'b0; b_op <= OP_NONE; b_j <= 8'd0; b_v <= 16'sd0; b_r <= 1'b0;
      b_w <= 8'sd0; out_valid <= 1'b0; out_neuron <= 8'd0; op_start <= 1'b0;
    end else begin
      if (fifo_we) wr_ptr <= wr_ptr + 7'd1;
      leak_pend <= (leak_pend | tick) & ~(start && nop == OP_LEAK);
      thr_pend  <= (thr_pend | thr_req) & ~(start && nop == OP_THR);
      op_start <= start;
      if (start) begin
        busy <= 1'b1; j <= 8'd0; op <= nop;
        if (nop == OP_ADDCMP || nop == OP_ADD) begin axon_r <= head; rd_ptr <= rd_ptr + 7'd1; end
      end else if (last) busy <= 1'b0;
      else if (busy) j <= j + 8'd1;
      // stage A: read
      b_valid <= busy;
      if (busy) begin b_op <= op; b_j <= j; b_v <= rd_word[15:0]; b_r <= rd_word[16]; b_w <= w_rd; end
      // stage B: the spike of the word written at this edge
      out_valid <= b_valid & fire;
      if (b_valid & fire) out_neuron <= b_j;
    end
  end
endmodule

// Library clock gate for synthesis (CORE_SYNTH), a latch-based model for RTL simulation.
module icg (input wire clk, input wire en, output wire gclk);
`ifdef CORE_SYNTH
  sky130_fd_sc_hd__dlclkp_1 u_icg (.CLK(clk), .GATE(en), .GCLK(gclk));
`else
  reg en_l;
  always @(clk or en) if (!clk) en_l <= en;
  assign gclk = clk & en_l;
`endif
endmodule
