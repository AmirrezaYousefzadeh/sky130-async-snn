// Measurement output unit (docs/hardware.md, spiking output): a hard-wired 4096 x NC int8 weight ROM (out_rom.v),
// NC spiking class neurons (16-bit integrator, threshold, hard reset) with a 12-bit spike counter each, one unit
// time-multiplexed over the classes, a 64-entry FIFO of last-layer neuron numbers. A spike costs NC cycles (one per
// class), a leak tick NC cycles. Clock gating as in the core: core gate, FIFO write/group/entry gates, one per class word.
`timescale 1ns/1ps
module snn_out_meas (
  input  wire clk, input wire rst, input wire cg_off,
  input  wire in_we, input wire [11:0] in_src, output wire in_ready,
  input  wire tick,
  output reg  out_valid, output reg [3:0] out_class, output reg op_start, output wire idle
);
`include "out_params.vh"
  localparam [1:0] OP_NONE = 2'd0, OP_ADD = 2'd1, OP_LEAK = 2'd2;
  reg busy; reg [3:0] c; reg [1:0] op; reg [11:0] src_r; reg [6:0] wr_ptr, rd_ptr; reg leak_pend;
  reg b_valid; reg [1:0] b_op; reg [3:0] b_c; reg signed [15:0] b_v; reg [11:0] b_n; reg signed [7:0] b_w;
  wire fifo_empty = (wr_ptr == rd_ptr);
  wire fifo_full = (wr_ptr[5:0] == rd_ptr[5:0]) && (wr_ptr[6] != rd_ptr[6]);
  assign in_ready = ~fifo_full;
  wire fifo_we = in_we & ~fifo_full;
  assign idle = ~busy & ~b_valid & ~out_valid & fifo_empty & ~leak_pend;
  wire core_en = cg_off | rst | in_we | tick | busy | b_valid | out_valid | op_start | leak_pend | ~fifo_empty;
  wire gclk;
  icg u_core_icg (.clk(clk), .en(core_en), .gclk(gclk));
  // FIFO: write gate, 8 groups of 8 entries
  wire [12*64-1:0] fifo_flat; wire fifo_gck;
  icg u_fifo_icg (.clk(gclk), .en(fifo_we), .gclk(fifo_gck));
  genvar e, eg;
  generate for (eg = 0; eg < 8; eg = eg + 1) begin : g_fgrp
    wire grp_gck;
    icg u_icg (.clk(fifo_gck), .en(wr_ptr[5:3] == eg), .gclk(grp_gck));
    for (e = 0; e < 8; e = e + 1) begin : g_fifo
      wire gck; reg [11:0] q;
      icg u_icg (.clk(grp_gck), .en(wr_ptr[2:0] == e), .gclk(gck));
      always @(posedge gck) q <= in_src;
      assign fifo_flat[12*(8*eg + e) +: 12] = q;
    end
  end endgenerate
  wire [11:0] head = fifo_flat[12*rd_ptr[5:0] +: 12];
  // ROM row of the source neuron and the class words {counter, integrator}
  wire [8*NC-1:0] row;
  out_rom u_rom (.a(src_r), .row(row));
  wire signed [7:0] w_rd = row[8*c +: 8];
  wire [27:0] wdata; wire [28*NC-1:0] mem_flat;
  genvar k;
  generate for (k = 0; k < NC; k = k + 1) begin : g_word
    wire gck; reg [27:0] q;
    icg u_icg (.clk(gclk), .en(rst | (b_valid && (b_c == k))), .gclk(gck));
    always @(posedge gck) q <= wdata;
    assign mem_flat[28*k +: 28] = q;
  end endgenerate
  wire [27:0] rd_word = mem_flat[28*c +: 28];
  wire last = busy && (c == NC - 1);
  wire can_start = ~busy | last;
  wire [1:0] nop = (leak_pend | tick) ? OP_LEAK : (~fifo_empty ? OP_ADD : OP_NONE);
  wire start = can_start && (nop != OP_NONE);
  wire signed [17:0] s = {{2{b_v[15]}}, b_v} + {{10{b_w[7]}}, b_w};
  wire signed [15:0] sat = (s > 18'sd32767) ? 16'sd32767 : (s < -18'sd32768) ? -16'sd32768 : s[15:0];
  wire signed [15:0] lk = b_v - (b_v >>> KOUT);
  wire fire = (b_op == OP_ADD) && ($signed({sat[15], sat}) >= THETA_OUT);
  wire signed [15:0] nv = (b_op == OP_ADD) ? (fire ? 16'sd0 : sat) : (b_op == OP_LEAK ? lk : b_v);
  wire [11:0] nn = b_n + {11'd0, fire};
  assign wdata = rst ? 28'd0 : {nn, nv};
  always @(posedge gclk) begin
    if (rst) begin
      busy <= 1'b0; c <= 4'd0; op <= OP_NONE; src_r <= 12'd0; wr_ptr <= 7'd0; rd_ptr <= 7'd0; leak_pend <= 1'b0;
      b_valid <= 1'b0; b_op <= OP_NONE; b_c <= 4'd0; b_v <= 16'sd0; b_n <= 12'd0; b_w <= 8'sd0;
      out_valid <= 1'b0; out_class <= 4'd0; op_start <= 1'b0;
    end else begin
      if (fifo_we) wr_ptr <= wr_ptr + 7'd1;
      leak_pend <= (leak_pend | tick) & ~(start && nop == OP_LEAK);
      op_start <= start;
      if (start) begin
        busy <= 1'b1; c <= 4'd0; op <= nop;
        if (nop == OP_ADD) begin src_r <= head; rd_ptr <= rd_ptr + 7'd1; end
      end else if (last) busy <= 1'b0;
      else if (busy) c <= c + 4'd1;
      b_valid <= busy;
      if (busy) begin b_op <= op; b_c <= c; b_v <= rd_word[15:0]; b_n <= rd_word[27:16]; b_w <= w_rd; end
      out_valid <= b_valid & fire;
      if (b_valid & fire) out_class <= b_c;
    end
  end
endmodule

module icg (input wire clk, input wire en, output wire gclk);
`ifdef CORE_SYNTH
  sky130_fd_sc_hd__dlclkp_1 u_icg (.CLK(clk), .GATE(en), .GCLK(gclk));
`else
  reg en_l;
  always @(clk or en) if (!clk) en_l <= en;
  assign gclk = clk & en_l;
`endif
endmodule
