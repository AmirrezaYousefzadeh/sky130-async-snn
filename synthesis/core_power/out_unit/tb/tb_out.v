// Testbench of the measurement core: plays a command file (gen_core.py) and opens the measurement window between the
// two markers. Outside the window the commands are deterministic in order (a leak tick or threshold request waits until
// the core is idle), so the output spikes match the reference model exactly; inside a spike window the FIFO is kept
// full (operations back to back), inside a scan window the next request is raised as soon as the previous one started.
// Defines: STIM (command file), OUT_FILE (output spikes), MODE_EVENT, USE_REF, CG_OFF_WINDOW (core clock gate held open
// in the window), IDLE_CYCLES (length of an idle window), DUMP_PATH (VCD of the window, all levels of the DUT),
// CLK_PERIOD_NS, PROGRESS.
`timescale 1ns/1ps
`ifndef CLK_PERIOD_NS
  `define CLK_PERIOD_NS 14.925
`endif
`ifndef MODE_EVENT
  `define MODE_EVENT 1
`endif
`ifndef USE_REF
  `define USE_REF 0
`endif
`ifndef CG_OFF_WINDOW
  `define CG_OFF_WINDOW 0
`endif
`ifndef IDLE_CYCLES
  `define IDLE_CYCLES 20000
`endif
module tb;
  reg clk = 1'b0, rst = 1'b1, cg_off = 1'b0, in_we = 1'b0, tick = 1'b0, thr_req = 1'b0;
  reg [11:0] in_axon = 12'd0;
  wire in_ready, out_valid, op_start, idle;
  wire [3:0] out_neuron;
  snn_out_meas u_dut (.clk(clk), .rst(rst), .cg_off(cg_off),
                       .in_we(in_we), .in_src(in_axon), .in_ready(in_ready), .tick(tick),
                       .out_valid(out_valid), .out_class(out_neuron), .op_start(op_start), .idle(idle));
  always #(`CLK_PERIOD_NS / 2.0) clk = ~clk;

  reg [13:0] cmds [0:65535];
  integer n_cmd, i, fo, cyc, win_start, win_end, in_win, ops_win, spk_win, spk_all, timeout;
  initial cyc = 0;
  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (out_valid) begin $fwrite(fo, "%0d\n", out_neuron); spk_all = spk_all + 1; if (in_win) spk_win = spk_win + 1; end
    if (op_start && in_win) ops_win = ops_win + 1;
  end

  task wait_idle; begin
    timeout = 0;
    @(negedge clk);
    while (!(idle && !in_we)) begin @(negedge clk); timeout = timeout + 1; if (timeout > 2000000) begin $display("FAIL: timeout waiting for idle"); $finish; end end
  end endtask

  initial begin
    for (i = 0; i < 65536; i = i + 1) cmds[i] = 14'h3fff;
    $readmemh(`STIM, cmds);
    n_cmd = 0; while (n_cmd < 65536 && cmds[n_cmd] !== 14'h3fff && cmds[n_cmd] !== 14'hxxxx) n_cmd = n_cmd + 1;
    if (n_cmd == 0) begin $display("FAIL: empty stimulus %s", `STIM); $finish; end
    fo = $fopen(`OUT_FILE, "w");
    in_win = 0; ops_win = 0; spk_win = 0; spk_all = 0;
    repeat (4) @(negedge clk);
    rst = 1'b0;
    for (i = 0; i < n_cmd; i = i + 1) begin
      case (cmds[i][13:12])
        2'd0: begin                                            // spike: queue it when the FIFO has room
          @(negedge clk); timeout = 0;
          while (!in_ready) begin @(negedge clk); timeout = timeout + 1; if (timeout > 100000) begin $display("FAIL: FIFO stuck"); $finish; end end
          in_we = 1'b1; in_axon = cmds[i][11:0];
          @(negedge clk); in_we = 1'b0;
        end
        2'd1, 2'd2: begin                                      // leak tick or threshold request
          if (!in_win) wait_idle;
          else @(negedge clk);
          if (cmds[i][13:12] == 2'd1) tick = 1'b1; else thr_req = 1'b1;
          @(negedge clk); tick = 1'b0; thr_req = 1'b0;
          if (in_win) begin timeout = 0; while (!op_start) begin @(negedge clk); timeout = timeout + 1; if (timeout > 100000) begin $display("FAIL: no start"); $finish; end end end
        end
        2'd3: begin                                            // window marker
          wait_idle;
          if (!in_win) begin
            in_win = 1; win_start = cyc; cg_off = (`CG_OFF_WINDOW == 1);
`ifdef DUMP_PATH
            $dumpfile(`DUMP_PATH); $dumpvars(0, tb.u_dut);
`endif
            $display("TB: window start at cycle %0d", cyc);
            if (i + 1 < n_cmd && cmds[i + 1][13:12] == 2'd3) repeat (`IDLE_CYCLES) @(negedge clk);   // an idle window
          end else begin
            win_end = cyc; in_win = 0;
            $display("TB: window end at cycle %0d", cyc);
            $display("TB: window %0d cycles, %0d operations, %0d output spikes in the window, %0d output spikes in all, clock %0.3f ns",
                     win_end - win_start, ops_win, spk_win, spk_all, `CLK_PERIOD_NS);
            // the window ends here: no $dumpoff (it would write every signal as x), stop the simulation at the window end
            $fclose(fo);
            $display("PASS: %0d commands played", n_cmd);
            $finish;
          end
        end
      endcase
`ifdef PROGRESS
      if (i % 200 == 0) $display("TB: command %0d of %0d at cycle %0d", i, n_cmd, cyc);
`endif
    end
    wait_idle;
    repeat (4) @(negedge clk);
    $fclose(fo);
    $display("PASS: %0d commands played", n_cmd);
    $finish;
  end

endmodule
