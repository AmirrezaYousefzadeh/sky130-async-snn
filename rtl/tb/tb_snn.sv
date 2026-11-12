// Testbench: replays one input stream (cycle, kind, channel, step tag) produced by snn/system.py, issues the step ticks, and logs
// every spike (layer core neuron cycle), every operation start (layer core cycle) and every output record
// (cycle o0..o9). Inputs are driven and outputs sampled at the falling edge of cycle `cyc`.
// Plusargs: +stream=<file> +out=<prefix> +mode=<0 step|1 event> +tcyc=<cycles per step> +nticks=<n> +end=<cycle>
module tb;
  parameter string DIR = ".";
  localparam int L = 5, G = 16;
  logic clk = 1'b0, rst = 1'b1;
  always #1 clk = ~clk;
  logic mode_event = 1'b0, tick = 1'b0, s_valid = 1'b0, s_marker = 1'b0; logic [10:0] s_ch = '0; logic [10:0] s_tag = '0;
  logic spk_v [L][G]; logic [7:0] spk_n [L][G]; logic opst [L][G]; logic mis [L][G];
  logic rec_valid; logic signed [15:0] rec_o [10];
  snn_top #(.L(L), .G(G), .DIR(DIR)) dut (.clk, .rst, .mode_event, .tick, .s_valid, .s_marker, .s_ch, .s_tag,
                                          .spk_v, .spk_n, .opst, .mis, .rec_valid, .rec_o);

  int it_cyc [$], it_kind [$], it_ch [$], it_tag [$];
  int tcyc, nticks, endc, mode, p, cyc, fs, fo, fp, nmiss;
  string fstream, prefix;
  initial begin
    int fd, a, b, cc, tg, r;
    if (!$value$plusargs("stream=%s", fstream)) $fatal(1, "no +stream");
    if (!$value$plusargs("out=%s", prefix)) prefix = "rtl_out";
    if (!$value$plusargs("mode=%d", mode)) mode = 0;
    if (!$value$plusargs("tcyc=%d", tcyc)) $fatal(1, "no +tcyc");
    if (!$value$plusargs("nticks=%d", nticks)) $fatal(1, "no +nticks");
    if (!$value$plusargs("end=%d", endc)) $fatal(1, "no +end");
    fd = $fopen(fstream, "r");
    while (!$feof(fd)) begin
      r = $fscanf(fd, "%d %d %d %d\n", a, b, cc, tg);
      if (r == 4) begin it_cyc.push_back(a); it_kind.push_back(b); it_ch.push_back(cc); it_tag.push_back(tg); end
    end
    $fclose(fd);
    fs = $fopen({prefix, "_spikes.txt"}, "w"); fo = $fopen({prefix, "_out.txt"}, "w"); fp = $fopen({prefix, "_ops.txt"}, "w");
    mode_event = (mode != 0); p = 0; cyc = 0; nmiss = 0;
    repeat (4) @(posedge clk);
    @(negedge clk); rst = 1'b0;
  end

  always @(posedge clk) if (!rst) cyc <= cyc + 1;

  always @(negedge clk) if (!rst) begin
    // drive cycle `cyc`
    s_valid = 1'b0; s_marker = 1'b0; s_ch = '0; s_tag = '0;
    if (p < it_cyc.size() && it_cyc[p] == cyc) begin
      s_valid = 1'b1; s_marker = (it_kind[p] != 0); s_ch = 11'(it_ch[p]); s_tag = 11'(it_tag[p]); p = p + 1;
    end
    tick = (cyc > 0) && (cyc % tcyc == 0) && (cyc / tcyc <= nticks);
    // sample cycle `cyc`
    for (int l = 0; l < L; l++) for (int c = 0; c < G; c++) begin
      if (spk_v[l][c]) $fwrite(fs, "%0d %0d %0d %0d\n", l, c, spk_n[l][c], cyc);
      if (opst[l][c]) $fwrite(fp, "%0d %0d %0d\n", l, c, cyc);
      if (mis[l][c]) nmiss = nmiss + 1;
    end
    if (rec_valid) begin
      $fwrite(fo, "%0d", cyc);
      for (int k = 0; k < 10; k++) $fwrite(fo, " %0d", rec_o[k]);
      $fwrite(fo, "\n");
    end
    if (cyc >= endc) begin
      $fclose(fs); $fclose(fo); $fclose(fp);
      $display("done at cycle %0d, %0d stream items, %0d step misses", cyc, p, nmiss);
      $finish;
    end
  end
endmodule
