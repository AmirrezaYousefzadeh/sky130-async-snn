#!/usr/bin/env python3
"""Drop the physical-only cells (fill, tap, decap/endcap, diode) from a routed netlist, for the gate-level simulation
only: they have no logic and no SDF delays, but 820,000 of them made Icarus' SDF annotation run for hours. The power
analysis keeps the full netlist (their leakage stays in). Usage: tools_strip_phys.py <in.nl.v> <out.nl.v>"""
import re, sys
text = open(sys.argv[1]).read()
pat = re.compile(r"^\s*sky130_fd_sc_hd__(?:fill|tapvpwrvgnd|decap|diode)\w*\s+\S+\s*\(.*?\);\s*$\n?", re.S | re.M)
out, n = pat.subn("", text)
open(sys.argv[2], "w").write(out)
print(f"removed {n} physical-only instances")
