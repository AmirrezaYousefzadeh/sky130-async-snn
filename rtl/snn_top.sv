// The multicore system: input distribution, L layers of G cores, routers, output unit (docs/hardware.md).
// An input stream item (event or tick marker) driven during cycle c is visible in its first-layer cores at c + 2.
module snn_top #(
  parameter int L = 5, parameter int G = 16, parameter int N = 256, parameter int NIN = 2048, parameter string DIR = "."
)(
  input  logic clk, input logic rst, input logic mode_event, input logic tick,
  input  logic s_valid, input logic s_marker, input logic [10:0] s_ch, input logic [10:0] s_tag,
  output logic spk_v [L][G], output logic [7:0] spk_n [L][G], output logic opst [L][G], output logic mis [L][G],
  output logic rec_valid, output logic signed [15:0] rec_o [10]
);
  // ---------------------------------------------------------------- input distribution
  // route_in[ch]: {v1, 3'b0, core1[3:0], axon1[7:0], v0, 3'b0, core0[3:0], axon0[7:0]}
  logic [31:0] rin [0:NIN-1];
  initial $readmemh($sformatf("%s/route_in.hex", DIR), rin);
  logic r_valid, r_marker; logic [10:0] r_ch; logic [10:0] r_tag;
  always_ff @(posedge clk) begin
    r_valid <= s_valid; r_marker <= s_marker; r_ch <= s_ch; r_tag <= s_tag;
    if (rst) r_valid <= 1'b0;
  end
  logic in_we [L][G]; logic [7:0] in_ax [L][G]; logic in_mk [L][G]; logic [10:0] in_tg [L][G];
  always_comb begin
    logic [31:0] e; e = rin[r_ch];
    for (int c = 0; c < G; c++) begin
      logic m0, m1;
      m0 = e[15] && (e[11:8] == 4'(c)); m1 = e[31] && (e[27:24] == 4'(c));
      in_we[0][c] = r_valid && (r_marker || m0 || m1);
      in_ax[0][c] = m0 ? e[7:0] : e[23:16];
      in_mk[0][c] = r_marker; in_tg[0][c] = r_tag;
    end
  end

  // ---------------------------------------------------------------- layers and routers
  logic [G-1:0] sv [L]; logic [7:0] sn [L][G]; logic [10:0] stg [L][G];
  genvar l, c;
  generate
    for (l = 0; l < L; l++) begin : lay
      for (c = 0; c < G; c++) begin : core
        snn_core #(.N(N), .NA(256), .FIRST(l == 0 ? 1 : 0), .LAYER(l), .CORE(c), .DIR(DIR)) u (
          .clk(clk), .rst(rst), .mode_event(mode_event), .tick(tick),
          .in_we(in_we[l][c]), .in_axon(in_ax[l][c]), .in_marker(in_mk[l][c]), .in_tag(in_tg[l][c]),
          .out_valid(sv[l][c]), .out_neuron(sn[l][c]), .out_tag(stg[l][c]), .op_start(opst[l][c]), .miss(mis[l][c]));
        assign spk_v[l][c] = sv[l][c]; assign spk_n[l][c] = sn[l][c];
      end
      if (l < L - 1) begin : rt
        logic [G-1:0] twe; logic [7:0] tax [G]; logic [10:0] ttg [G];
        snn_router #(.NS(G), .NT(G), .AXW(8), .LAYER(l), .DIR(DIR)) r (
          .clk(clk), .rst(rst), .s_valid(sv[l]), .s_neuron(sn[l]), .s_tag(stg[l]), .t_we(twe), .t_axon(tax), .t_tag(ttg));
        for (c = 0; c < G; c++) begin : w
          assign in_we[l + 1][c] = twe[c]; assign in_ax[l + 1][c] = tax[c]; assign in_mk[l + 1][c] = 1'b0; assign in_tg[l + 1][c] = ttg[c];
        end
      end
    end
  endgenerate

  // ---------------------------------------------------------------- output
  logic [0:0] owe; logic [11:0] oax [1]; logic [10:0] otg [1];
  snn_router #(.NS(G), .NT(1), .AXW(12), .LAYER(L - 1), .DIR(DIR)) ro (
    .clk(clk), .rst(rst), .s_valid(sv[L - 1]), .s_neuron(sn[L - 1]), .s_tag(stg[L - 1]), .t_we(owe), .t_axon(oax), .t_tag(otg));
  snn_out #(.NI(G * N), .NC(10), .DIR(DIR)) uo (
    .clk(clk), .rst(rst), .mode_event(mode_event), .tick(tick), .in_we(owe[0]), .in_src(oax[0]), .in_tag(otg[0]),
    .rec_valid(rec_valid), .rec_o(rec_o));
endmodule
