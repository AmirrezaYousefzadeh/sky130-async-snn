// Router between two layers: NS source cores -> NT targets. A source emits at most one spike per cycle (neuron index);
// a hardwired table gives its target and axon. Spikes wait in per-source, per-target queues; every cycle each target
// takes the head of the lowest-numbered non-empty queue. A spike emitted at cycle e enters its queue at the end of e,
// is selected (combinationally, t_we) at e + 1 at the earliest and written into the target at the end of that cycle,
// so it is visible at the target at e + 2 (H = 2).
module snn_router #(
  parameter int NS = 16, parameter int NT = 16, parameter int AXW = 8, parameter int DQ = 64, parameter int TB = 11,
  parameter int LAYER = 0, parameter string DIR = "."
)(
  input  logic clk, input logic rst,
  input  logic [NS-1:0] s_valid, input logic [7:0] s_neuron [NS], input logic [TB-1:0] s_tag [NS],
  output logic [NT-1:0] t_we, output logic [AXW-1:0] t_axon [NT], output logic [TB-1:0] t_tag [NT]
);
  // routing table: entry for (source core s, neuron n) at s*256 + n: bit 31 valid, bits [AXW+7:AXW] target, [AXW-1:0] axon
  logic [31:0] route [0:NS*256-1];
  initial $readmemh($sformatf("%s/route_l%0d.hex", DIR, LAYER), route);

  localparam int QB = $clog2(DQ);
  logic [AXW+TB-1:0] q [NS][NT][DQ];                     // {tag, axon}
  logic [QB:0] qw [NS][NT], qr [NS][NT];
  logic [NS-1:0] sel [NT];                              // one-hot source chosen by each target this cycle

  always_comb begin
    for (int t = 0; t < NT; t++) begin
      logic found; found = 1'b0; sel[t] = '0; t_we[t] = 1'b0; t_axon[t] = '0; t_tag[t] = '0;
      for (int s = 0; s < NS; s++) begin
        if (!found && qw[s][t] != qr[s][t]) begin
          found = 1'b1; sel[t][s] = 1'b1; t_we[t] = 1'b1;
          {t_tag[t], t_axon[t]} = q[s][t][qr[s][t][QB-1:0]];
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    for (int s = 0; s < NS; s++) begin
      if (s_valid[s]) begin
        logic [31:0] ent; int t;
        ent = route[s * 256 + 32'(s_neuron[s])];
        if (ent[31]) begin
          t = int'(ent[AXW + 7:AXW]);
          q[s][t][qw[s][t][QB-1:0]] <= {s_tag[s], ent[AXW-1:0]};
          qw[s][t] <= qw[s][t] + 1;
        end
      end
      for (int t = 0; t < NT; t++) if (sel[t][s]) qr[s][t] <= qr[s][t] + 1;
    end
    if (rst) for (int s = 0; s < NS; s++) for (int t = 0; t < NT; t++) begin qw[s][t] <= '0; qr[s][t] <= '0; end
  end
endmodule
