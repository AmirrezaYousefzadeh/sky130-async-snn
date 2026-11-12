# sky130-async-snn

A cycle-accurate model and RTL of a TrueNorth-like multicore for spiking neural networks (SNNs), with energy measured
on a placed-and-routed core in the open SkyWater 130 nm process (sky130). The same trained networks run on the same
hardware under three schedules:

- **global tick** (TrueNorth): all cores advance one time step per tick, sized for their worst case;
- **barrier** (Loihi-like): the next step starts as soon as every core is done;
- **step-free**: every spike is processed on arrival and no core waits for a step boundary; a broadcast tick only
  drives the leak.

The system has 80 cores (five hidden layers of 16 cores with 256 axons and 256 neurons each, a hard-wired int8 weight
ROM and one time-multiplexed neuron unit per core, clocked at 67 MHz for a worst case of 256 spikes per 1 ms tick),
routers, 64-entry FIFOs with back-pressure, and an output unit with one spiking neuron and spike counter per class.
The model reports accuracy against time, latency to a target accuracy, spikes, core utilization, throughput and
energy on N-MNIST, DVS Gesture, N-TIDIGITS18 and SSC, with the sensor in real time and with recorded data replayed
as a burst. `docs/hardware.md` describes the hardware and the schedules; `docs/runs.md` lists the exact commands for
the data, every network and the runs.

## Layout

| path | contents |
|---|---|
| `snn/data.py` | dataset preparation (Tonic download, compact event arrays), input memory (delayed copies of the channels) |
| `snn/model.py` | integer-exact feed-forward SNN in PyTorch: time-stepped (pipelined, as executed by the time-stepped schedules) and micro-binned step-free dynamics, pruning to the core graph, integer export |
| `snn/train.py` | training (`python -m snn.train --help`) |
| `snn/system.py` | the cycle-accurate model of the multicore: schedules, routers, FIFOs and back-pressure, input port, real time and burst mode |
| `snn/granularity.py` | why networks trained in time steps lose accuracy step-free: threshold checks per step on partial sums |
| `snn/eval.py`, `snn/intref.py`, `snn/workload.py`, `snn/plot.py`, `snn/summarize.py` | an earlier per-layer reference model and its evaluation tools |
| `rtl/` | SystemVerilog of the core, router, output unit and top, a testbench, and `run_rtl.py` (export, Verilator build, spike-by-spike comparison with the model) |
| `synthesis/core_power/` | energy and area of one core and of the output unit: measurement RTL, LibreLane synthesis and hardening, gate-level simulation, OpenSTA power; results and method in `RESULTS.md` |
| `docs/` | `hardware.md` (hardware and schedules), `runs.md` (data, training, system-model, RTL and energy commands) |

Names in the code: the step-free schedule is `event:wire`, the global tick `step:aligned`, the barrier
`step:barrier`; burst mode is `--speeds ip` (input-paced) and real time `--speeds rt`. The refractory period of one
step in the hidden layers (M1h) is `--wire_ref --wire_ref_hidden`, on all neurons (M1) `--wire_ref`.

## Install

Python 3.10 or later; training needs a CUDA GPU, the system model runs on CPUs (numba).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The RTL check needs Verilator 5. The energy flow needs LibreLane 3.0.16 (with Nix), Yosys, Icarus Verilog and the
sky130 PDK from open_pdks; set their paths in the environment or in `synthesis/core_power/env.local.sh` (see
`synthesis/core_power/env.sh`).

## Quick start

```bash
.venv/bin/python -m snn.data prepare --dataset nmnist && .venv/bin/python -m snn.data prepare --dataset nmnist32
# train a network with one of the commands in docs/runs.md, then run it on the system model in burst mode:
.venv/bin/python -m snn.system --model results/models/<name>/model_int.npz --dataset nmnist32 \
  --configs step:aligned,step:barrier,event:wire --wire_ref --wire_ref_hidden --speeds ip --in_period 256 \
  --tick_us 1000 --f_mhz 67 --fifo 64 --port_buf 64 --work_curves --out results/system/<name>_ip.json
```

## Status and limits

- The cycle-accurate model is the reference for all results. The RTL reproduces it spike for spike for the global
  tick and for an earlier step-free variant (lazy leak with step tags), with an integrating output unit and
  unlimited FIFOs. The step-free schedule on the tick wire, the barrier controller, the spiking output unit, the
  bounded FIFOs with back-pressure, the hidden-only refractory period and the input delay buffer are in the model
  only so far (`docs/hardware.md`).
- Energy is extrapolated from operation counts (rows, leak and threshold passes, idle time) with the gate-level
  measurements of one core and of the output unit (`synthesis/core_power/RESULTS.md`). The routers, the tick and
  barrier wires, the input port and the delay buffer are not included.
- Trained models and result files are not part of the repository (about 30 GB).

## License

Apache-2.0 for code and RTL (`LICENSE`); documentation CC-BY-4.0. The datasets are downloaded from their original
sources and are not redistributed.
