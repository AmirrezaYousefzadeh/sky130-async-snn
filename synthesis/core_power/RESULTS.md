# Energy and area of one core (and of the output unit) in sky130

**Use the "layout" column of the first table**: a routed layout (LibreLane, 30 %
utilisation), SDF-annotated gate-level simulation (glitches) and extracted wires (SPEF), typical corner. Phase 1
(synthesised netlist, zero delay, no wires) is kept below for the breakdown and the other scenarios.

## Headline numbers (core: layer 2, core 0 of the N-MNIST 5 ms network; tt, 1.8 V, 25 °C; 67 MHz)

| quantity | layout (SDF + SPEF) | synthesis, zero-delay |
|---|---|---|
| **incoming spike, event-driven** (256 neuron updates, threshold after every add) | **53.1 nJ** (207 pJ per synaptic operation) | 41.8 nJ |
| incoming spike, stepped add (no threshold check) | 50.5 nJ | 42.5 nJ |
| **leak scan** | **49.2 nJ** | 39.0 nJ |
| **threshold scan** (stepped tick) | **49.8 nJ** | 38.2 nJ |
| **idle, core clock gated** | **2.20 µW** | 2.09 µW |
| idle, clock running into the core | 829 µW | 313 µW |
| leakage | 0.34 µW | 0.31 µW |
| cell area | 0.967 mm² (with repair buffers and taps); core 2.85 mm², die 2.91 mm² at 30 % utilisation | 0.850 mm² |
| output unit (synthesis, zero-delay; about ×1.25 for layout and glitches) | | 0.378 nJ per incoming spike, 0.100 nJ per leak tick, 1.85 µW idle, 0.213 mm² |

Short windows for the layout runs (100 warm-up spikes, then 250 spikes or 50 scans; idle 5,000 cycles); every run
bit-exact against the reference, 0 SDF annotation errors, 0 SPEF misses, 0 measured activities lost. From
synthesis to layout: wires add 2 % (spike) to 7 % (scan), glitches another 25 % and 18 % (SDF over zero-delay,
both with wires). The clock tree of the routed layout makes a running-but-idle core 2.6 times costlier, which is why
the core's clock is gated when idle.

Tools: LibreLane 3.0.16 (OpenROAD, and OpenSTA 2.7.0 from its Nix environment), Yosys and Icarus Verilog 14.0 from the
OSS CAD Suite 2026-10-24, open_pdks commit 8afc834 (sky130A, sky130_fd_sc_hd). Paths: `env.sh`.

## The designs

**Core** (`rtl/snn_core_meas.v`), as in `docs/hardware.md`: 256 axons x 256 neurons, the hard-wired int8 weight
ROM of layer 2, core 0 of the N-MNIST 5 ms spike-count network (`tn4096t5so`; threshold 152, leak shift 10),
256 words of {refractory bit, 16-bit membrane} in flip-flops, one neuron unit time-multiplexed over the 256 neurons
(one neuron per cycle, a two-stage pipeline), a 64-entry FIFO of axon numbers. An operation (a spike's row, a leak
scan, a threshold scan) takes 256 cycles. Clock gating with `sky130_fd_sc_hd__dlclkp_1`: one gate for the whole core
(closed when idle), the FIFO behind a write gate, 8 group gates and one gate per entry, the membrane words behind 16
group gates and one gate per word, so only the word being written and its group are clocked. The ROM is one kept
module per neuron (`rtl/core_rom.v`, generated), kept for routability.

**Output unit** (`out_unit/rtl/snn_out_meas.v`): the hard-wired 4096 x 10 int8 ROM of the same network, 10 spiking
class neurons (16-bit integrator, threshold 55, hard reset, leak shift 5) with 12-bit spike counters, one unit
time-multiplexed over the classes (a spike costs 10 cycles), the same 64-entry FIFO and clock gating.

RTL of both is bit-exact against a Python reference model (`gen_core.py`, `out_unit/gen_out.py`) on every scenario,
and every gate-level run is bit-exact against the same reference (output spikes compared over the whole run).

## Phase 1 results: synthesised netlist, zero-delay (typical corner tt, 1.8 V, 25 °C; clock 67 MHz = 14.925 ns)

| quantity | core | output unit |
|---|---|---|
| standard-cell area after synthesis | **0.850 mm²** (85,211 cells) | **0.213 mm²** (22,142 cells) |
| of which weight ROM | 0.679 mm² (79 %) | |
| of which flip-flops (4,947: membranes 4,352, FIFO 512, control 83) | 0.105 mm² (12 %) | 999 flip-flops |
| of which multiplexers, neuron unit, control, clock gates | 0.072 mm² (8 %) | |
| **dynamic energy per incoming spike** (event-driven: add, threshold check after every add) | **42.5 nJ** | **0.378 nJ** (10 classes) |
| per synaptic operation (one neuron updated) | **166 pJ** | 37.8 pJ |
| incoming spike, stepped add (no threshold check) | 42.9 nJ | |
| incoming spike, event-driven with the refractory bit | 43.4 nJ | |
| incoming spike on uniformly random axons (instead of the real input) | 42.9 nJ | |
| **leak scan** (v -= v >> k over all neurons) | **38.4 nJ** | 0.100 nJ |
| **threshold scan** (stepped tick) | **38.5 nJ** | |
| **idle, core clock gate closed** (FIFO empty) | **2.09 µW** | 1.85 µW |
| idle, clock running into the core (gate held open) | 313 µW | 256 µW |
| leakage alone | 0.31 µW | 0.075 µW |

Power while busy: 11.1 mW for back-to-back spikes, 10.1 mW for back-to-back scans. Every energy above includes the
leakage of the window (negligible: 0.31 µW against 10–11 mW).

Where the busy power goes (spike window): multiplexer cells 56 % (the 256:1 read multiplexers of the membrane words,
17 bits, and of the ROM row, 8 bits, whose select changes every cycle), other logic 19 %, flip-flops 14 % (mostly the
shared write-data bus that reaches the D pin of every membrane flip-flop), weight ROM 10 % (its 2,048 outputs change
once per spike), clock gates 2 %. In a scan the ROM is quiet, so a scan costs 4 nJ less than a spike. Idle with the
clock running: the 83 control flip-flops (73 %) and the clock gates (27 %).

## Phase 2: the hardened layout

- Layout `h25/runs/u30` (LibreLane 3.0.16, settings of `config.yaml`, clock gates fixed before CTS with
  `tools/firm_clock_gates.tcl`): 30 % utilisation, core 2.85 mm² (die 2.91 mm²), 128,930 standard cells
  (0.967 mm² cell area including taps and the repair buffers), 4.25 m of wire, routing DRC 0, power grid clean,
  24 hold buffers, 9 slew violations at nominal tt.
- **Timing.** Hardened at 25 ns, because at 14.925 ns the post-CTS repair could not close the slow corner (worst
  setup slack −4 to −5 ns at ss_100C_1v60): the read path from the counter j through the 256:1 multiplexer trees is
  long (its select nets are buffered in chains of up to twenty `buf_12`). Hold is met at all 9 corners (worst
  +0.12 ns at ff). Setup at 25 ns: +9.7 to +10.4 ns at tt, +15 ns at ff, −3.0 to −4.2 ns at ss: the slow corner
  needs about 29 ns (35 MHz). Own sign-off at the typical corner at 67 MHz (14.925 ns, renamed netlist, nominal SPEF,
  `signoff67.sh`): setup −0.019 ns on a single path, with the SDC's 0.25 ns clock uncertainty and 5 % derating
  (positive without these margins), hold +0.31 ns. So the core runs at 67 MHz at the typical corner, at the edge;
  a slow-corner chip would need 35 MHz or a pipelined read path.
- **SDF-annotated simulation**: Icarus annotation of the routed netlist ran for over an hour with its 820,662 fill,
  tap and endcap instances; the simulation netlist leaves them out (`tools_strip_phys.py`; they have no logic and no
  delays) while OpenSTA reads the full netlist, so their leakage stays in. Annotation then took about 25 min; the
  spike window 57 min in all. Results in the headline table.
- **Energy with the wires, zero-delay** (to separate wires from glitches; short windows as above):

| per operation | synthesis netlist, no wires | layout, with wires |
|---|---|---|
| incoming spike (event-driven) | 41.8 nJ | 42.7 nJ (+2 %) |
| leak scan | 39.0 nJ | 41.9 nJ (+7 %) |
| idle, clock running into the core | 313 µW | 829 µW (the clock tree: buffers and wires) |

  The total hardly moves, but its composition does: switching power (wires) rises from 17 to 57 % of the busy power
  while internal power falls (the repair resized the multiplexer and select cells).

## How to extrapolate to the chip (spike counting)

Per core c and sample: E_c = E_spike · n_spikes_in(c) + E_leak · n_leak_scans(c) + E_thr · n_thr_scans(c)
+ P_idle · T_sample, with E_spike = 53.1 nJ (50.5 nJ for stepped adds), E_leak = 49.2 nJ, E_thr = 49.8 nJ,
P_idle = 2.20 µW per core when the core clock is gated while idle (829 µW per core if the clock keeps running).
These are the layout values; the synthesis-level ones (42.5, 38.4, 38.5 nJ, 2.09 µW) are about 20 % lower.
Output unit: 0.378 nJ per incoming spike, 0.100 nJ per leak tick, 1.85 µW idle. The router, the tick/barrier wires
and the input port are not included.
- A stepped core pays a leak scan and a threshold scan at every tick (first-layer cores one fused scan), an
  event-driven core a leak scan per tick: at 66–381 ticks per sample the scans are a large part of a core's energy.
- With gated idle cores, idle energy is negligible (2 µW x 80 cores = 0.17 mW, against 11 mW per busy core): the
  energy follows the number of operations (spikes + scans) almost exactly.

## How it was measured

1. RTL and reference: `gen_core.py` generates the ROM, the stimulus of every scenario and the expected output
   spikes; `run_rtl.sh` simulates the RTL (Icarus 14.0, OSS CAD Suite 2026-10-24) and compares.
   Stimulus: the spikes layer 1 really sends to this core in event-driven, input-paced execution of N-MNIST test
   samples (system model, tick wire), 300 of them with their leak ticks as warm-up (realistic membranes: 291 output
   spikes during warm-up, 883 in the 1,000-spike window), then the measurement window between two markers holding
   one kind of operation only, back to back (FIFO kept full, or the next scan request raised while the previous one
   runs). Idle windows: 20,000 cycles.
2. Synthesis: LibreLane 3.0.16 (`synth.sh`, `--to Yosys.Synthesis`, `config.yaml`), PDK open_pdks 8afc834,
   sky130_fd_sc_hd. Flip-flop and clock-gate counts
   equal the RTL's (4,947 and 346); no latches. ABC prints "The network is combinational" for every ROM module (its
   sequential passes on a combinational module; harmless).
3. The netlist is flattened and every escaped name made plain (`flatten.sh`: Yosys `flatten`, then
   `tools/legal_names.py`), so that simulation, toggle names and OpenSTA agree.
4. Gate-level simulation (`gls.sh`): Icarus with the PDK's functional models (`-DFUNCTIONAL -DUNIT_DELAY=#1`, zero
   delay), the window's waveform streamed through a FIFO into `tools/vcd_toggles`.
5. Power (`power.sh`): `tools/toggles_to_activity.py` (nets after pins), OpenSTA 2.7.0 from LibreLane 3.0.16 with the
   tt_025C_1v80 liberty, one propagated clock at 14.925 ns, unannotated inputs at 0, `tools/check_annotation.py`:
   0 measured activities lost, 358,822 pins annotated (core). The root clock runs during every window, so no
   root-clock correction applies.
6. `report.py` turns the reports into `results_synth_zd.json` (energy = window power x window time / operations).

## What is missing, and how large the error may be

- **No wire capacitance.** Without SPEF, OpenSTA counts pin capacitances only (OpenSTA does not apply the liberty's
  default wire load by itself; with `set_wire_load_model Large` the leak scan rose only 1.8 %, because sky130's wire
  load tables are tiny and say nothing about a 1.7 mm x 1.7 mm block). Switching power is 17 % of the busy power;
  real wires will raise it, most for the long select and data nets of the read multiplexers.
- **No glitches.** Zero-delay simulation; earlier sky130 measurements gave glitch factors of 1.13–1.20 for decoder cores and
  1.13–1.93 for adder-tree cores. Multiplexer trees with a changing select glitch too.
- **No clock tree and no repair buffers.** LibreLane's post-placement repair added energy in earlier sky130 cores (+20 %
  on a 34,000-cell logic core); the clock tree is small here (clock gates are 2 % of the busy power).
- Measured in phase 2: wires +2 to +7 %, glitches +18 to +25 %, together about +20 to +27 % on the busy energies.
- **Design choices that set the numbers:** a flip-flop register file read through 256:1 multiplexers dominates
  (56 %); a one-hot read pointer with an AND-OR tree, or an SRAM macro, would likely cut the busy energy by several
  times. The per-neuron ROM modules (kept for routability) cost area (1.30 µm² per weight bit against 0.42 in the
  earlier unsplit probe) but little energy (10 %).
- Corner: typical only; leakage at 25 °C.

## Run times

Synthesis 8 min (core), 2.5 min (output unit). Gate-level simulation of the core, one thread each: 35–39 min for the
spike windows (364,000 cycles, of which 256,000 dumped), 3.5–4 min for the scan windows, 1.5–2.5 min idle; OpenSTA
power about 1 min each. Output unit: under a minute per scenario.

## Files

- `rtl/` (core RTL, generated ROM and constants), `tb/tb_core.v`, `stim/` (commands, expected spikes, `meta.json`),
  `gen_core.py`, `run_rtl.sh`, `config.yaml`, `core.sdc`, `synth.sh`, `flatten.sh`, `gls.sh`, `power.sh`,
  `power_wl.sh`, `measure.sh`, `report.py`, `harden.sh`, `tools/` (flow helpers: hardening ladder, metric and annotation checks, toggle
  counting, OpenSTA scripts).
- Runs (not tracked): `runs/synth/` (synthesis), `runs/flat/` (flat, renamed netlist), `runs/gls_zd/<scenario>/` (simulation log,
  toggles, activity, `sta/` reports), `logs/`; `results_synth_zd.json`.
- `runs/synth_v1/`, `runs/gls_zd_v1/`: a first version with one clock gate per word and entry directly on the core
  clock (idle with the clock running 971 µW instead of 313 µW); superseded.
- Output unit: `out_unit/` with the same structure.
- Phase 2: `h25/` (configuration at 25 ns, `h25/runs/u30` the routed layout, `h25/work/` the ladder log; a 20 %
  attempt `h25/runs/u20` was cancelled), `runs/p2/` (renamed netlist, SDF, SPEF, `design_sim.nl.v` without physical
  cells, `signoff67*`), `runs/gls_sdf/` (annotated runs), `runs/gls_p2zd/` (zero-delay on the layout),
  `runs/gls_zd_s/` (the short windows on the synthesis netlist), `gen_short.py`, `phase2_prep.sh`, `signoff67.sh`,
  `tools_strip_phys.py`; `results_layout_and_synth.json`. Hardening at 30 %: about 50 min; annotated runs 26–57 min.
