#!/usr/bin/env python3
"""Fail when an integrated clock gate in a netlist has no clock input.

OpenROAD dcf3613's CTS (LibreLane 3.0.16/3.0.17, and ORFS run with those binaries) can leave a clock gate's clock pin
unconnected (OpenROAD issue #9816). LibreLane's disconnected-pins checker reports it only after
detailed routing and ORFS not at all; run this on the netlist written after CTS, or on the final one.

Usage: check_clock_gates.py <netlist.v> [--cells REGEX] [--clock-pins NAMES]
  --cells       cell names of the clock gates (default: sky130 dlclkp/sdlclkp, ASAP7 ICGx, gf180 icgtp, IHP lgcp,
                NanGate45 CLKGATE)
  --clock-pins  their clock input pins, comma separated (default CLK,CK)
Exit status 0 when every clock gate has a connected clock pin, 1 otherwise.
"""
import argparse
import re
import sys

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("netlist")
ap.add_argument("--cells", default=r"\w*(dlclkp|sdlclkp|ICGx|icgtp|lgcp|CLKGATE)\w*")
ap.add_argument("--clock-pins", default="CLK,CK")
a = ap.parse_args()

text = re.sub(r"//[^\n]*", "", open(a.netlist).read())
pins = a.clock_pins.split(",")
inst = re.compile(rf"\b(?P<cell>(?:{a.cells}))\s+(?P<name>\\\S+|\w+)\s*\((?P<body>.*?)\)\s*;", re.S)
gates, lost = 0, []
for m in inst.finditer(text):
    gates += 1
    conn = dict(re.findall(r"\.(\w+)\s*\(\s*([^()]*?)\s*\)", m["body"]))
    if not any(conn.get(p) for p in pins):
        lost.append(f"{m['name']} ({m['cell']})")
print(f"check_clock_gates: {gates} clock gates, {len(lost)} without a clock input"
      + (": " + ", ".join(lost[:20]) + (" ..." if len(lost) > 20 else "") if lost else ""))
sys.exit(1 if lost else 0)
