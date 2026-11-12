// Output unit: NC class integrators fed by the NI neurons of the last hidden layer (weight ROM NI x NC). Class c of an
// operation starting at S is read at S + c and written at S + c + 1; a record (rec_valid with the integrators) is
// visible at F + 2, where F is the first free cycle after the recorded operation(s).
//   step : spikes visible at or before a tick are integrated at that tick: leak (NC cycles), one row per spike (NC
//          cycles each, lowest neuron first), one record.
//   event: lazy (time-step) leak: a spike of a newer step first triggers one leak (NC cycles) per elapsed step; every
//          spike then adds its row (NC cycles) and is recorded.
module snn_out #(
  parameter int NI = 4096, parameter int NC = 10, parameter int WB = 8, parameter int OB = 16, parameter int FIFO_D = 4096, parameter int TB = 11,
  parameter string DIR = "."
)(
  input  logic clk, input logic rst, input logic mode_event, input logic tick,
  input  logic in_we, input logic [11:0] in_src, input logic [TB-1:0] in_tag,
  output logic rec_valid, output logic signed [OB-1:0] rec_o [NC]
);
  logic signed [WB-1:0] rom [0:NI*NC-1];
  logic signed [31:0] cfg [0:0];                        // k_out
  initial begin $readmemh($sformatf("%s/out_rom.hex", DIR), rom); $readmemh($sformatf("%s/out_cfg.hex", DIR), cfg); end
  wire [4:0] kl = cfg[0][4:0];
  localparam logic signed [OB-1:0] OMAX = (1 <<< (OB - 1)) - 1;
  localparam logic signed [OB-1:0] OMIN = -(1 <<< (OB - 1));
  logic signed [OB-1:0] o [NC], snap [NC];
  initial for (int c = 0; c < NC; c++) begin o[c] = '0; snap[c] = '0; end
  assign rec_o = snap;                                  // integrators captured right after the recorded operation

  // event FIFO: {tag, source}
  logic [TB+11:0] fifo [0:FIFO_D-1];
  logic [$clog2(FIFO_D):0] wr, rd;
  wire fifo_empty = (wr == rd);
  // step pending vectors
  logic [NI-1:0] pend [0:1]; logic fill; logic [NI-1:0] cons;
  initial begin pend[0] = '0; pend[1] = '0; fill = 1'b0; cons = '0; end
  logic [TB-6:0] cur;                                   // event: step of the last leak
  wire [TB+11:0] head = fifo[rd[$clog2(FIFO_D)-1:0]];

  function automatic logic [12:0] lowest(input logic [NI-1:0] x);
    lowest = 13'h1000;
    for (int i = NI - 1; i >= 0; i--) if (x[i]) lowest = 13'(i);
  endfunction

  // operation: 0 none, 1 leak, 2 add
  logic busy; logic [3:0] c; logic [1:0] op; logic [11:0] src; logic last;    // last: record after this operation
  logic stepping;                                       // step: inside a tick's sequence
  logic b_valid; logic [1:0] b_op; logic [3:0] b_c; logic signed [OB-1:0] b_o; logic signed [WB-1:0] b_w;
  logic r1, r2;

  logic [1:0] nop; logic [11:0] nsrc; logic nlast; logic take_fifo, lazy_leak, start_tick; logic [12:0] nxt;
  always_comb begin
    nop = 2'd0; nsrc = '0; nlast = 1'b0; take_fifo = 1'b0; lazy_leak = 1'b0; start_tick = 1'b0; nxt = lowest(cons);
    if (!mode_event) begin
      if (!stepping) begin
        if (tick) begin nop = 2'd1; start_tick = 1'b1; nlast = (lowest(pend[fill]) == 13'h1000); end
      end else begin
        if (nxt != 13'h1000) begin
          nop = 2'd2; nsrc = nxt[11:0];
          nlast = (lowest(cons & ~(NI'(1) << nxt[11:0])) == 13'h1000);
        end
      end
    end else begin
      if (!fifo_empty) begin
        if (head[TB+11:17] > cur) begin nop = 2'd1; lazy_leak = 1'b1; nlast = 1'b0; end     // leak per step, no record
        else begin nop = 2'd2; nsrc = head[11:0]; take_fifo = 1'b1; nlast = 1'b1; end
      end
    end
  end
  wire starting = !busy && nop != 2'd0;
  logic [1:0] a_op; logic [3:0] a_c; logic [11:0] a_src; logic a_valid;
  assign a_op = busy ? op : nop; assign a_c = busy ? c : 4'd0; assign a_src = busy ? src : nsrc; assign a_valid = busy || starting;
  wire a_last_class = a_valid && a_c == 4'(NC - 1) && (busy ? last : nlast);

  always_ff @(posedge clk) begin
    if (in_we) begin
      if (!mode_event) pend[(tick && !stepping) ? ~fill : fill][in_src] <= 1'b1;
      else begin fifo[wr[$clog2(FIFO_D)-1:0]] <= {in_tag, in_src}; wr <= wr + 1; end
    end
    if (starting) begin
      busy <= 1'b1; c <= 4'd1; op <= nop; src <= nsrc; last <= nlast;
      if (take_fifo) rd <= rd + 1;
      if (lazy_leak) cur <= cur + 1;
      if (start_tick) begin cons <= pend[fill]; pend[fill] <= '0; fill <= ~fill; stepping <= !nlast; end
      if (!mode_event && stepping && nop == 2'd2) begin cons[nsrc] <= 1'b0; if (nlast) stepping <= 1'b0; end
    end else if (busy) begin
      c <= c + 1;
      if (c == 4'(NC - 1)) busy <= 1'b0;
    end
    b_valid <= a_valid; b_op <= a_op; b_c <= a_c; b_o <= o[a_c]; b_w <= rom[32'(a_src) * NC + 32'(a_c)];
    if (b_valid) begin
      logic signed [OB+1:0] s;
      if (b_op == 2'd1) o[b_c] <= b_o - (b_o >>> kl);
      else begin s = b_o + b_w; o[b_c] <= (s > OMAX) ? OMAX : (s < OMIN) ? OMIN : s[OB-1:0]; end
    end
    r1 <= a_last_class; r2 <= r1; rec_valid <= r2;
    if (r2) snap <= o;                                  // o during F + 1 holds the last write of the operation
    if (rst) begin
      busy <= 1'b0; c <= '0; op <= '0; b_valid <= 1'b0; wr <= '0; rd <= '0; cur <= '0; stepping <= 1'b0;
      r1 <= 1'b0; r2 <= 1'b0; rec_valid <= 1'b0;
    end
  end
endmodule
