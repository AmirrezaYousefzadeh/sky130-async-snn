#!/usr/bin/env python3
"""Empty Verilog modules for the physical-only cells of a routed netlist (flow/gls.sh).

Usage: gen_phys_stubs.py <netlist.v> <out stubs.v> <simulation model .v files ...>
Tap, fill and antenna cells appear in the routed netlist without pin connections, and the library's simulation models
do not define all of them; each such cell type gets an empty module. Connected cells without a model are reported.
"""
import re
import sys
from pathlib import Path

NAME = r"[A-Za-z0-9_\\\[\]$./]+"


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    netlist = Path(sys.argv[1]).read_text(errors="replace")
    models = " ".join(Path(f).read_text(errors="replace") for f in sys.argv[3:])
    defined = set(re.findall(r"^\s*(?:module|primitive)\s+([A-Za-z0-9_\\$]+)", models + "\n" + netlist, re.M))
    keywords = {"module", "wire", "input", "output", "inout", "assign"}
    unconnected = {c for c in re.findall(rf"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+{NAME}\s*\(\s*\)\s*;", netlist, re.M)}
    connected = {c for c in re.findall(rf"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+{NAME}\s*\(\s*\.", netlist, re.M)}
    stubs = sorted(unconnected - defined - keywords)
    missing = sorted(connected - defined - keywords)
    Path(sys.argv[2]).write_text("// physical-only cells (no pins in the netlist): empty modules for the simulation\n"
                                 + "".join(f"module {c} (); endmodule\n" for c in stubs))
    print(f"{len(stubs)} physical-only cell types stubbed: {', '.join(stubs[:6])}{' ...' if len(stubs) > 6 else ''}")
    if missing:
        print("WARNING: connected cells without a simulation model:", ", ".join(missing[:10]))


if __name__ == "__main__":
    main()
