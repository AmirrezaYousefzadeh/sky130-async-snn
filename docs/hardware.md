# Hardware

One multicore system runs the same trained network under three schedules. It is defined by the cycle-accurate model
(`snn/system.py`); the RTL in `rtl/` reproduces that model cycle for cycle where it implements the same features (see
the last section).

## System and mapping

- Five hidden layers of 4,096 neurons, 16 cores of 256 axons and 256 neurons each: 80 cores. An output unit with one
  spiking neuron and one spike counter per class. An input port and a tick controller.
- Connections come from training by pruning to the core graph. Every hidden neuron has exactly one target (core,
  axon) in the next layer; every input channel reaches a fixed number of first-layer cores (2 for N-MNIST and DVS
  Gesture, 5 for SSC, all 16 for N-TIDIGITS18). The targets sit in a hard-wired routing table (4 bits of core and 8
  bits of axon per neuron).
- One clock of 67 MHz for every schedule: the smallest clock at which a core finishes its worst case within a 1 ms
  tick. The worst case is a spike on each of its 256 axons: a leak pass, 256 rows of 256 cycles and a threshold pass
  (66,048 cycles), plus up to 262 cycles to deliver its last spikes.

## Core

| part | content |
|---|---|
| weight ROM | 256 axons x 256 neurons x int8, hard-wired |
| state | 256 membranes of 16 bits; 256 refractory bits |
| neuron unit | one, time-multiplexed: one synaptic operation per cycle (read membrane, add weight with saturation, compare with the threshold, write back, or reset and emit a spike) |
| input FIFO | 64 axon numbers, with back-pressure (step-free schedule) |
| pending vectors | two 256-bit vectors used alternately, one filling during a step and one consumed at the tick (time-stepped schedules) |
| output | spikes leave in neuron order, one per cycle, through the routing table to the router |

Every operation takes 256 cycles: an incoming spike (add row a to all membranes), a leak pass (v -= v >> k), a
threshold pass (fire and reset), or the fused leak-and-threshold pass of a first-layer core in time steps. The
neuron is deliberately simple: instantaneous synapse, shift leak, hard reset to 0.

The output unit has a 4096 x C ROM. A spike into it costs C cycles. Its class neurons have a threshold, a hard reset
and a leak like the hidden neurons, plus a spike counter; the decision is the class with the most output spikes, and
there is none while the two highest counts tie.

## Schedules (names in the code in parentheses)

- **Global tick** (`step:aligned`). A tick at every step boundary, at least 1 ms (67,000 cycles) after the previous
  one. First-layer cores add input events on arrival and run the fused leak-and-threshold pass at the tick. Deeper
  cores mark arriving spikes in the filling pending vector; at the tick they swap the vectors, run a leak pass, add
  one row per marked axon (lowest axon first) and run a threshold pass. A spike crosses one layer per step, as in
  training; the model is bit-exact with the time-stepped torch model.
- **Barrier** (`step:barrier`). The same, but the tick follows 16 cycles after every core is done (FIFO empty,
  neuron unit idle, no spike in flight): one done wire per core into one controller. It computes the same spikes as
  the global tick.
- **Step-free** (`event:wire`). Every input event and every spike is added on arrival, with a threshold check after
  each add. A broadcast tick wire drives only the leak: once the row in progress is done, a core runs one leak pass,
  and spikes still in its FIFO are added after it. Spike packets carry no step.
- **Refractory period** (`--wire_ref`, M1; with `--wire_ref_hidden`, M1h). One bit per neuron, set when it fires and
  cleared by the tick, limits a neuron to one spike per step, as in training. M1h applies it to the hidden neurons
  only, so that the class neurons, whose spike counts are the decision, stay free.

The model also has variants that the experiments do not use: per-core tick phases, a full-speed stepped mode, and a
step-free mode with lazy leak (every spike carries the step of its input, `event:lazy`).

## Router, FIFOs and input port

- Between two layers, every target takes at most one spike per cycle, from the lowest-numbered source that has one
  waiting. A spike emitted at cycle e is visible at its target at e + 2 at the earliest.
- Every core's input FIFO, and the output unit's, holds 64 entries (8-bit axon numbers: 512 bits, as much as the two
  pending vectors). A slot is taken when a spike or input event is sent towards the core and freed when the core
  starts that row. A core holds one outgoing spike in an output register; the router moves it only when the target
  FIFO has room, and a firing neuron that finds the register occupied stalls the neuron unit for a cycle. No spike is
  dropped inside the chip, and the feed-forward network cannot deadlock.
- **Real time** (`--speeds rt`): sensor events arrive at their time stamps in a 64-event port buffer (`--port_buf`)
  and leave in order, at most one per cycle, when their first-layer cores have room; an event that arrives while the
  buffer is full is dropped and counted. A timer raises the tick every T of sensor time. The port also limits every
  first-layer core to 256 events per millisecond, its worst case.
- **Burst mode** (`--speeds ip`, input-paced): recorded events leave the port one per 256 cycles (`--in_period`),
  the rate at which no first-layer core can fall behind, and only when all their first-layer cores have room. When
  the next event passes a step boundary of the recorded time stamps, the port raises the tick (global tick: at least
  1 ms after the previous one; barrier: once all cores are done; step-free: at once, on the wire). A time-stepped
  system takes no event of a step before the tick that opens it.
- **Input memory** (`--in_delays_ms` in training, stored with the model). Every sensor event is repeated after the
  given delays (32 and 96 ms in the experiments) on further copies of the channels, a tapped delay line like the
  axonal delays of TrueNorth. In burst mode the copies are released by a delay buffer inside the chip and take one
  cycle of the port each; in real time they enter the port at their time.

## Verification status

- `rtl/` (`snn_core.sv`, `snn_router.sv`, `snn_out.sv`, `snn_top.sv`, `tb/tb_snn.sv`) reproduces `snn/system.py`
  exactly, with every spike (layer, core, neuron, cycle) and every output record identical, for the global tick and
  for the lazy-leak step-free variant with a refractory window, with an integrating output unit and unlimited FIFOs.
  Checked on N-MNIST test samples (sample 0: 5,190 spikes with the global tick and 11,598 step-free).
- The step-free schedule on the tick wire, the barrier controller, the spiking output unit, the bounded FIFOs with
  back-pressure, the hidden-only refractory period and the input delay buffer are in the model only.
- The measurement core and output unit of the energy flow (`synthesis/core_power/rtl/`) implement the spiking output,
  the 64-entry FIFO and the clock gating; they match their own Python reference on every scenario, at RTL and at
  gate level.
