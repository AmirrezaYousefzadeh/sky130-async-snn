#!/usr/bin/env python3
"""Energies and areas of the measurement core from the gate-level runs (gls.sh, power.sh) and the netlist.
Usage: report.py <runs dir with one folder per scenario> <netlist> [--label text] [-o result.json]"""
import argparse, json, os, re
from collections import defaultdict
from pathlib import Path
LIB = os.path.join(os.environ.get("PDK_ROOT", ""), "sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib")
T_NS = 14.925

def cell_areas(lib):
    areas, cell = {}, None
    for line in open(lib):
        m = re.match(r'\s*cell\s*\(\s*"?([\w$]+)"?\s*\)', line)
        if m: cell = m.group(1); continue
        m = re.match(r"\s*area\s*:\s*([\d.]+)", line)
        if m and cell and cell not in areas: areas[cell] = float(m.group(1))
    return areas

def netlist_area(nl, areas):
    """Area by part from the synthesised (hierarchical) netlist: kept ROM modules, membrane words, FIFO, the rest."""
    mods, cur = defaultdict(list), None
    for line in open(nl):
        m = re.match(r"\s*module\s+([^\s(]+)", line)
        if m: cur = m.group(1); continue
        m = re.match(r"\s*(sky130_fd_sc_hd__\w+)\s+(\S+)\s*\(", line)
        if m and cur: mods[cur].append((m.group(1), m.group(2)))
    rom = sum(areas.get(c, 0) for mname, insts in mods.items() if mname.startswith("rom_n") for c, _ in insts)
    rom_cells = sum(len(insts) for mname, insts in mods.items() if mname.startswith("rom_n"))
    parts = defaultdict(float); counts = defaultdict(int)
    for c, inst in mods.get("snn_core_meas", []):
        a = areas.get(c, 0)
        if "g_word" in inst: k = "membrane words (flip-flops and clock gates)"
        elif "g_fifo" in inst: k = "FIFO entries (flip-flops and clock gates)"
        elif "dfxtp" in c or "dfrtp" in c or "edfxtp" in c: k = "other flip-flops (control, pipeline, output)"
        elif "dlclkp" in c: k = "core clock gate"
        else: k = "logic: read multiplexers, neuron unit, control"
        parts[k] += a; counts[k] += 1
    parts["weight ROM (256 kept modules)"] = rom; counts["weight ROM (256 kept modules)"] = rom_cells
    return dict(parts), dict(counts)

def power(rpt):
    tot = {}
    for line in open(rpt):
        f = line.split()
        if f and f[0] in ("Sequential", "Combinational", "Clock", "Macro", "Pad", "Total") and len(f) >= 5:
            try: tot[f[0]] = dict(internal=float(f[1]), switching=float(f[2]), leakage=float(f[3]), total=float(f[4]))
            except ValueError: pass
    return tot

def window(simlog):
    m = re.search(r"TB: window (\d+) cycles, (\d+) operations, (\d+) output spikes in the window", Path(simlog).read_text())
    return dict(cycles=int(m.group(1)), ops=int(m.group(2)), out_spikes=int(m.group(3)))

ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("netlist"); ap.add_argument("--label", default="")
ap.add_argument("-o", default=None); a = ap.parse_args()
areas = cell_areas(LIB); parts, counts = netlist_area(a.netlist, areas)
res = dict(label=a.label, clock_ns=T_NS, area_um2=parts, cells=counts, area_total_um2=sum(parts.values()), scenarios={})
for d in sorted(Path(a.runs).iterdir()):
    rpt = d / "sta/power_vcd.rpt"
    if not rpt.exists(): continue
    w = window(d / "sim.log"); p = power(rpt); t = w["cycles"] * T_NS * 1e-9
    P, L = p["Total"]["total"], p["Total"]["leakage"]
    r = dict(window=w, power_W=p, window_s=t, energy_window_J=P * t, dynamic_energy_window_J=(P - L) * t)
    if w["ops"]: r.update(energy_per_op_J=P * t / w["ops"], dynamic_energy_per_op_J=(P - L) * t / w["ops"],
                          dynamic_energy_per_synop_J=(P - L) * t / w["ops"] / 256)
    res["scenarios"][d.name] = r
print(json.dumps(res, indent=1))
if a.o: Path(a.o).write_text(json.dumps(res, indent=1))
