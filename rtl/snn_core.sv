// One core of the multicore SNN (docs/hardware.md): NA axons x N neurons, one neuron unit time-multiplexed over
// the N neurons, hardwired weight ROM (loaded from a hex file in simulation), two schedules.
//
// Timing (must match snn/system.py): an operation that starts at cycle S touches neuron j at cycle S + j (stage A:
// read membrane and weight) and writes it at S + j + 1 (stage B); a spike of neuron j is visible on out_valid at
// S + j + 2. The next operation may start at S + N. Entries written by the router at the end of cycle c are visible
// at c + 1.
//
// mode_event = 0, time steps with one step of delay per hop:
//   FIRST = 1: FIFO entries are input events (add the row) and in-band tick markers (fused leak and threshold scan).
//   FIRST = 0: router writes set bits of the filling pending vector; at a tick (cycle K) the vector filled with the
//              entries visible at or before K is consumed: leak scan, one row per set bit (lowest axon first),
//              threshold scan.
// mode_event = 1, event-driven with lazy (time-step) leak: every FIFO entry carries the step of the input it descends
//   from; before an entry of a newer step the core runs one leak scan per elapsed step (clearing the refractory bits),
//   then adds the row and compares each neuron with the threshold (one spike per neuron per step). Spikes inherit the
//   step tag. The tag is {step[5:0], refractory window within the step[4:0]}: a newer step leaks (one scan per step),
//   a newer window within the step flash-clears the refractory bits. An idle core does no work.
module snn_core #(
  parameter int N = 256, parameter int NA = 256, parameter int WB = 8, parameter int VB = 16,
  parameter int FIRST = 0, parameter int FIFO_D = 1024, parameter int TB = 11,
  parameter int LAYER = 0, parameter int CORE = 0, parameter string DIR = "."
)(
  input  logic clk, input logic rst,
  input  logic mode_event,
  input  logic tick,                                   // timer tick (deeper cores): global step tick or local timer
  input  logic in_we, input logic [7:0] in_axon, input logic in_marker, input logic [TB-1:0] in_tag,
  output logic out_valid, output logic [7:0] out_neuron, output logic [TB-1:0] out_tag,
  output logic op_start,                               // pulse at the first cycle of every operation (busy accounting)
  output logic miss                                    // pulse when a step tick is served late
);
  // ---------------------------------------------------------------- parameters loaded per core
  logic signed [WB-1:0] rom [0:NA*N-1];
  logic signed [31:0] cfg [0:2];                       // theta, k_leak, n_axons
  initial begin
    $readmemh($sformatf("%s/l%0d_c%0d.hex", DIR, LAYER, CORE), rom);
    $readmemh($sformatf("%s/l%0d_cfg.hex", DIR, LAYER), cfg);
  end
  wire signed [VB:0] theta = cfg[0][VB:0];
  wire [4:0] kl = cfg[1][4:0];
  localparam logic signed [VB-1:0] VMAX = (1 <<< (VB - 1)) - 1;
  localparam logic signed [VB-1:0] VMIN = -(1 <<< (VB - 1));

  logic signed [VB-1:0] v [0:N-1];
  logic fired [0:N-1];
  initial for (int i = 0; i < N; i++) begin v[i] = '0; fired[i] = 1'b0; end

  // ---------------------------------------------------------------- input FIFO (entries: marker bit, step tag, axon)
  logic [TB+8:0] fifo [0:FIFO_D-1];
  logic [$clog2(FIFO_D):0] wr, rd;
  wire fifo_empty = (wr == rd);
  wire [TB+8:0] head = fifo[rd[$clog2(FIFO_D)-1:0]];
  wire head_mk = head[TB+8]; wire [TB-1:0] head_tag = head[TB+7:8]; wire [7:0] head_ax = head[7:0];
  logic [TB-6:0] cur;                                  // event mode: step of the last leak
  logic [TB-1:0] cref;                                 // event mode: current refractory window (step, window)

  // ---------------------------------------------------------------- pending vectors (step, deeper cores)
  logic [NA-1:0] pend [0:1];
  logic fill;                                          // index of the filling vector
  logic [NA-1:0] cons;                                 // vector being consumed
  initial begin pend[0] = '0; pend[1] = '0; fill = 1'b0; cons = '0; end

  // ---------------------------------------------------------------- operation state
  typedef enum logic [2:0] {OP_NONE, OP_ADD, OP_ADDCMP, OP_LEAK, OP_LEAKCLR, OP_FUSED, OP_THR} op_t;
  logic busy; logic [8:0] j; op_t op; logic [7:0] axon;
  logic [1:0] phase;                                   // step deeper tick: 0 idle, 1 leak, 2 adds, 3 threshold
  logic step_pend;                                     // step deeper: a tick to serve
  logic [TB-1:0] optag;                                // tag of the running operation
  // stage B pipeline registers
  logic b_valid; op_t b_op; logic [7:0] b_j; logic signed [VB-1:0] b_v; logic signed [WB-1:0] b_w; logic [TB-1:0] b_tag;

  function automatic logic [8:0] lowest(input logic [NA-1:0] x);
    lowest = 9'h100;
    for (int i = NA - 1; i >= 0; i--) if (x[i]) lowest = 9'(i);
  endfunction

  // decision of the next operation when the unit is free (combinational)
  op_t nop; logic [7:0] nax; logic take_fifo; logic lazy_leak; logic start_step; logic [8:0] nxt;
  always_comb begin
    nop = OP_NONE; nax = '0; take_fifo = 1'b0; lazy_leak = 1'b0; start_step = 1'b0;
    nxt = lowest(cons);
    if (!mode_event && !FIRST) begin
      if (phase == 2'd0) begin
        if (step_pend || tick) begin nop = OP_LEAK; start_step = 1'b1; end
      end else if (phase == 2'd1 || phase == 2'd2) begin
        if (nxt != 9'h100) begin nop = OP_ADD; nax = nxt[7:0]; end
        else nop = OP_THR;
      end
    end else if (!mode_event && FIRST) begin
      if (!fifo_empty) begin take_fifo = 1'b1; nop = head_mk ? OP_FUSED : OP_ADD; nax = head_ax; end
    end else begin
      if (!fifo_empty) begin
        if (head_tag[TB-1:5] > cur) begin lazy_leak = 1'b1; nop = OP_LEAKCLR; end   // one leak scan per elapsed step
        else begin take_fifo = 1'b1; nop = OP_ADDCMP; nax = head_ax; end
      end
    end
  end

  wire starting = !busy && (nop != OP_NONE);
  op_t a_op; assign a_op = busy ? op : nop;
  wire [7:0] a_j = busy ? j[7:0] : 8'd0;
  wire [7:0] a_ax = busy ? axon : nax;
  wire a_valid = busy || starting;
  assign op_start = starting;

  // ---------------------------------------------------------------- sequential part
  always_ff @(posedge clk) begin
    miss <= 1'b0;
    // router writes
    if (in_we) begin
      if (!mode_event && !FIRST) pend[(tick && phase == 2'd0 && !step_pend) ? ~fill : fill][in_axon] <= 1'b1;
      else begin fifo[wr[$clog2(FIFO_D)-1:0]] <= {in_marker, in_tag, in_axon}; wr <= wr + 1; end
    end
    // ticks
    if (!mode_event && !FIRST && tick) begin
      if (phase == 2'd0 && !step_pend) begin cons <= pend[fill]; pend[fill] <= '0; fill <= ~fill; end
      else begin step_pend <= 1'b1; miss <= 1'b1; end      // late: previous step still running
    end
    // a timer tick comes after the entries visible at or before its cycle (wr counts them) and before later ones
    // stage A: issue
    if (starting) begin
      busy <= (N > 1); j <= 9'd1; op <= nop; axon <= nax; optag <= head_tag;
      if (take_fifo) rd <= rd + 1;
      if (lazy_leak) begin cur <= cur + 1; cref <= {cur + 1'b1, 5'b0}; end
    end
    if (starting) begin
      if (start_step) begin
        phase <= 2'd1;
        if (step_pend) begin cons <= pend[fill]; pend[fill] <= '0; fill <= ~fill; step_pend <= 1'b0; end
      end
      if (nop == OP_ADD && !FIRST && !mode_event) begin cons[nax] <= 1'b0; phase <= 2'd2; end
      if (nop == OP_THR) phase <= 2'd3;
    end else if (busy) begin
      j <= j + 1;
      if (j == 9'(N - 1)) begin
        busy <= 1'b0;
        if (op == OP_THR) phase <= 2'd0;
      end
    end
    b_valid <= a_valid; b_op <= a_op; b_j <= a_j; b_v <= v[a_j]; b_w <= rom[a_ax * N + a_j]; b_tag <= busy ? optag : head_tag;
    // stage B: compute and write back
    out_valid <= 1'b0;
    if (b_valid) begin
      logic signed [VB+1:0] s; logic signed [VB-1:0] r; logic fire;
      fire = 1'b0; r = b_v;
      case (b_op)
        OP_ADD, OP_ADDCMP: begin
          s = b_v + b_w;
          r = (s > VMAX) ? VMAX : (s < VMIN) ? VMIN : s[VB-1:0];
          if (b_op == OP_ADDCMP && $signed({r[VB-1], r}) >= theta && !fired[b_j]) begin fire = 1'b1; fired[b_j] <= 1'b1; end
        end
        OP_LEAK, OP_LEAKCLR: begin
          r = b_v - (b_v >>> kl);
          if (b_op == OP_LEAKCLR) fired[b_j] <= 1'b0;
        end
        OP_FUSED: begin
          r = b_v - (b_v >>> kl);
          if ($signed({r[VB-1], r}) >= theta) fire = 1'b1;
        end
        OP_THR: if ($signed({b_v[VB-1], b_v}) >= theta) fire = 1'b1;
        default: ;
      endcase
      v[b_j] <= fire ? '0 : r;
      if (fire) begin out_valid <= 1'b1; out_neuron <= b_j; out_tag <= b_tag; end
    end
    // a spike of a newer refractory window within the same step: flash-clear the refractory bits before its first add
    if (starting && mode_event && nop == OP_ADDCMP && head_tag > cref) begin
      for (int i = 0; i < N; i++) fired[i] <= 1'b0;
      cref <= head_tag;
    end
    if (rst) begin
      busy <= 1'b0; j <= '0; op <= OP_NONE; phase <= 2'd0; b_valid <= 1'b0; out_valid <= 1'b0;
      wr <= '0; rd <= '0; step_pend <= 1'b0; miss <= 1'b0; cur <= '0; cref <= '0; out_tag <= '0;
    end
  end
endmodule
