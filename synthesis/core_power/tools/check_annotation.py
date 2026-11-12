#!/usr/bin/env python3
"""Check that every cell pin the simulation saw received its measured activity in OpenSTA (flow/power.sh).

Usage: check_annotation.py <sta out dir> <netlist> <liberty> [<liberty> ...]
Reads annot_found_pins.txt, annot_notfound_pins.txt and annot_notfound_top.txt (written by the activity file of
toggles_to_activity.py); liberties may be gzipped.
Names that OpenSTA could not find are expected for supply pins and for the internal signals of the cell models
(and0_out_X, notifier, D_delayed, ...). A name <instance>/<pin> whose pin is a signal pin of the instance's liberty cell,
a name whose instance is not in the netlist (unless the pin is a supply pin: fill and tap instances renamed by a
refill have only supply pins in the toggle file), or any top-level name, means a measured activity was lost: the pin
falls back to propagated activity. A name whose instance's cell is in none of the liberties is lost too, unless
the pin is a supply pin. Prints the counts and exits 1 if any activity was lost, or if no pin activity was annotated
at all (a scope prefix that matched nothing).
"""
import gzip
import re
import sys
from pathlib import Path

from root_clock_correction import read_netlist


def signal_pins(liberties):
    """Liberty cell -> its signal pin and bus names (pg_pin supplies excluded); and the set of all supply pin names."""
    pins, supplies, cell = {}, set(), None
    for path in liberties:
        opener = gzip.open if str(path).endswith(".gz") else open
        for line in opener(path, "rt", errors="ignore"):
            m = re.match(r"\s*cell\s*\(\s*\"?([\w$]+)\"?\s*\)", line)
            if m:
                cell = m.group(1)
                pins.setdefault(cell, set())
                continue
            m = re.match(r"\s*pg_pin\s*\(\s*\"?(\w+)\"?\s*\)", line)
            if m:
                supplies.add(m.group(1))
                continue
            m = re.match(r"\s*(pin|bus)\s*\(\s*\"?([\w$\[\]]+)\"?\s*\)", line)
            if m and cell:
                pins[cell].add(m.group(2))
    return pins, supplies


def tokens(path):
    """The names of a coverage file (a Tcl list: names with brackets are braced), read in chunks (millions of names)."""
    rest = ""
    with open(path, errors="ignore") as f:
        while chunk := f.read(1 << 24):
            parts = (rest + chunk).split()
            rest = "" if chunk[-1].isspace() else parts.pop()
            yield from (p.strip("{}") for p in parts)
    if rest:
        yield rest.strip("{}")


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    out, netlist, liberties = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    pins, supplies = signal_pins(liberties)
    cells = {name: cell for name, (cell, _) in read_netlist(netlist).items()}
    found = sum(1 for _ in tokens(out / "annot_found_pins.txt"))
    supplies |= {"VPWR", "VGND", "VPB", "VNB", "VDD", "VSS", "vdd", "vss", "vccd1", "vssd1"}
    expected, absent_supplies, lost, unknown_cells = 0, 0, [], set()
    for name in tokens(out / "annot_notfound_pins.txt"):
        instance, _, pin = name.rpartition("/")
        if instance not in cells:
            if pin in supplies:
                absent_supplies += 1                           # e.g. fill renamed by a refill: no activity to lose
            else:
                lost.append(name)                              # an instance name OpenSTA does not know: a naming mismatch
            continue
        if cells[instance] not in pins:
            if pin not in supplies:
                unknown_cells.add(cells[instance])
                lost.append(name)                              # a cell missing from the liberties given
            continue
        cell_pins = pins[cells[instance]]
        if pin in cell_pins or re.sub(r"\[\d+\]$", "", pin) in cell_pins:
            lost.append(name)
        else:
            expected += 1
    lost += list(tokens(out / "annot_notfound_top.txt"))
    print(f"check_annotation: {found} pin activities annotated, {expected} names not found as expected (supplies, "
          f"model internals), {absent_supplies} supply pins of instances absent from the netlist, "
          f"{len(lost)} measured activities lost")
    if unknown_cells:
        print(f"   cells in none of the liberties: {' '.join(sorted(unknown_cells)[:10])}")
    for name in lost[:20]:
        print(f"   lost: {name}")
    if not found:
        print("   no pin activity was annotated: check the --scope of toggles_to_activity.py")
    sys.exit(1 if lost or not found else 0)


if __name__ == "__main__":
    main()
