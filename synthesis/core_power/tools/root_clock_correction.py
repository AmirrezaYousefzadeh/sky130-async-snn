#!/usr/bin/env python3
"""Correct OpenSTA's power for a clock that is stopped most of the time (flow/power.sh).

Usage: root_clock_correction.py <netlist.v> <clock port> <duty> <report dir> [-o out.json]
Reads <report dir>/power_vcd_by_instance.rpt and power_vcd.rpt (flow/tools/power_sta.tcl).

OpenSTA gives every pin of the clock network between the clock port and the first clock gate the activity of the clock
definition, two transitions per period, whatever the annotation says; downstream of a clock gate it uses the measured
activity. In the real-time window the testbench runs the clock a fraction d of the time (the duty). The root network is
found by walking from the clock port through buffers, inverters and hold or delay buffers; it ends at clock gates
(dlclkp, sdlclkp) and at any other cell the root clock reaches (flip-flops clocked by the ungated clock, macros,
logic). The correction removes the part of their dynamic power that the stopped clock does not spend:
  * buffers and inverters of the root: (1 - d) x (internal + switching);
  * clock gates and the other cells on the root: (1 - d) x internal (their output switching is measured);
  P = P_report - removed, where P_report is the total of the group table. Leakage is not changed.
Clocks should be gated with the sky130 dlclkp cell: a clock gated with ordinary logic ends the root walk at that gate,
and the buffers behind it are not corrected.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

OUTPUT_PINS = ("X", "Y", "Q", "Q_N", "GCLK")
SUPPLY_PINS = ("VGND", "VNB", "VPB", "VPWR")


def read_netlist(path):
    """Instances: name -> (cell type, {pin: net})."""
    text = Path(path).read_text(errors="ignore")
    cells = {}
    for m in re.finditer(r"^\s*([A-Za-z_][\w$]*)\s+([A-Za-z_][\w$]*)\s*\((.*?)\);", text, flags=re.S | re.M):
        if m.group(1) in ("module", "wire", "input", "output", "inout", "assign", "reg"):
            continue
        cells[m.group(2)] = (m.group(1), dict(re.findall(r"\.(\w+)\(\s*([^()]*?)\s*\)", m.group(3))))
    return cells


def read_report(path):
    """Per-instance power from report_power -instances: name -> (internal, switching, leakage, total) in W."""
    power = {}
    for line in open(path, errors="ignore"):
        f = line.split()
        if len(f) == 5:
            try:
                power[f[4]] = tuple(float(x) for x in f[:4])
            except ValueError:
                pass
    return power


def read_total(path):
    """The Total line of the group table: (internal, switching, leakage, total) in W."""
    for line in open(path):
        if line.startswith("Total"):
            return tuple(float(x) for x in line.split()[1:5])
    raise SystemExit(f"no Total line in {path}")


def root_network(cells, clock_port):
    """The cells of the root clock network: name -> 'buffer' | 'gate' | 'other'."""
    sinks = {}
    for name, (_, pins) in cells.items():
        for pin, net in pins.items():
            if pin not in OUTPUT_PINS and pin not in SUPPLY_PINS:
                sinks.setdefault(net, []).append(name)
    root, todo, seen = {}, [clock_port], set()
    while todo:
        net = todo.pop()
        if net in seen:
            continue
        seen.add(net)
        for name in sinks.get(net, []):
            cell, pins = cells[name]
            kind = re.sub(r"^sky130_fd_sc_[a-z]+__", "", cell)         # any sky130 standard-cell library
            inputs = [p for p in pins if p not in OUTPUT_PINS and p not in SUPPLY_PINS]
            if kind.startswith(("dlclkp", "sdlclkp")):
                root[name] = "gate"
            elif kind.startswith(("clkbuf", "clkinv", "clkdlybuf", "clkdlyinv", "dlygate", "dlymetal", "dlybuf", "buf",
                                  "inv")) and len(inputs) == 1:      # hold and delay buffers continue the walk too
                root[name] = "buffer"
                todo += [pins[p] for p in OUTPUT_PINS if p in pins]
            else:
                root[name] = "other"
    return root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("netlist")
    ap.add_argument("clock_port")
    ap.add_argument("duty", type=float)
    ap.add_argument("report_dir", type=Path)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    cells = read_netlist(a.netlist)
    root = root_network(cells, a.clock_port)
    power = read_report(a.report_dir / "power_vcd_by_instance.rpt")
    missing = sorted(name for name in root if name not in power)
    removed_internal = removed_switching = 0.0
    for name, kind in root.items():
        if name in power:
            internal, switching = power[name][:2]
            removed_internal += (1 - a.duty) * internal
            removed_switching += (1 - a.duty) * switching if kind == "buffer" else 0.0
    report = read_total(a.report_dir / "power_vcd.rpt")
    removed = removed_internal + removed_switching
    keys = ("internal", "switching", "leakage", "total")
    result = {
        "duty": a.duty, "root_cells": dict(Counter(root.values())), "root_cells_without_power": missing,
        "removed_W": removed, "removed_internal_W": removed_internal, "removed_switching_W": removed_switching,
        "report": dict(zip(keys, report)),
        "corrected": dict(zip(keys, (report[0] - removed_internal, report[1] - removed_switching, report[2],
                                     report[3] - removed))),
    }
    if a.out:
        json.dump(result, open(a.out, "w"), indent=1)
    print(f"root network of {a.clock_port}: {result['root_cells']}; duty {a.duty:.6f}; removed {removed * 1e6:.4f} µW; "
          f"{report[3] * 1e6:.4f} -> {result['corrected']['total'] * 1e6:.4f} µW")
    if missing:
        print(f"WARNING: {len(missing)} root cells without a power entry, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
