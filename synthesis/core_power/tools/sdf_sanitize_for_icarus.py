#!/usr/bin/env python3
"""Reduce an OpenSTA SDF to what Icarus Verilog can annotate (flow/gls.sh).

Usage: sdf_sanitize_for_icarus.py <design.sdf> -o <design.icarus.sdf>
Icarus aborts the whole annotation on constructs it does not parse, so this keeps the cell delays (IOPATH) and drops the
rest:
  * VOLTAGE / TEMPERATURE / PROCESS triplets "a::b" become "a:a:b";
  * INTERCONNECT entries are dropped (wire delays are not modelled; the cell delays include the load of the wires);
  * TIMINGCHECK blocks are dropped (Icarus does not check them);
  * conditional delays (COND ...) (IOPATH ...) become plain IOPATHs, since Icarus does not parse the conditions;
  * IOPATHs on bus bits (dout[0] of a macro) are dropped, since Icarus does not parse them; the macro's own simulation
    model carries its output delay;
  * CELL entries left without any IOPATH are dropped;
  * empty delay triplets "()" become zeros;
  * within a CELL, a repeated IOPATH of an input/output pair already seen is dropped when one of its triplets is all
    zero: the unwrapped conditional paths of clock gates (and xor/mux cells) carry zero-filled rise triplets, and Icarus
    applies the last match, which would set the clock-gate rise delay to 0;
  * a time scale other than 1 ns (OpenSTA writes "(TIMESCALE 1ps)" for ASAP7) is rescaled to 1 ns, because Icarus up to
    v13 ignores the header and the testbench expects ns.
Instance names must be plain identifiers (legal_names.py runs first).
"""
import argparse
import re
import sys
from pathlib import Path


def skip_balanced(lines, start):
    """Index just past the parenthesised block that starts at lines[start]."""
    depth = 0
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i + 1
    return len(lines)


UNIT_NS = {"fs": 1e-6, "ps": 1e-3, "ns": 1.0, "us": 1e3}
TRIPLET = re.compile(r"\(([-+0-9.eE]*)(?::([-+0-9.eE]*):([-+0-9.eE]*))?\)")


def scale_iopath(line, factor):
    """Multiply every delay value of an IOPATH line by factor (the port names are left alone)."""
    m = re.match(r"(\s*\(IOPATH\s+\S+\s+\S+\s+)(.*)$", line, flags=re.S)
    if not m:
        return line
    def one(v):
        return v if v == "" else f"{float(v) * factor:.6f}"
    return m.group(1) + TRIPLET.sub(lambda t: "(" + ":".join(one(v) for v in t.groups() if v is not None) + ")"
                                    if t.group(2) is not None else "(" + one(t.group(1)) + ")", m.group(2))


def dedupe_zero_repeats(cell, stats):
    """Drop later IOPATHs of an already seen input/output pair whose delays are all zero."""
    seen, out = set(), []
    for line in cell.splitlines(keepends=True):
        m = re.match(r"\s*\(IOPATH\s+(\S+)\s+(\S+)\s+(.*)$", line)
        if m:
            key = (m.group(1), m.group(2))
            triplets = [[v for v in t if v != ""] for t in TRIPLET.findall(m.group(3))]
            if key in seen and any(t and all(float(v) == 0.0 for v in t) for t in triplets):
                stats["dropped_zero_repeats"] += 1
                continue
            seen.add(key)
        out.append(line)
    return "".join(out)


def sanitize(text):
    stats = dict.fromkeys(("dropped_interconnect", "dropped_timingcheck", "unwrapped_cond", "dropped_bus_iopath",
                           "dropped_empty_cells", "filled_empty_triplets", "dropped_zero_repeats",
                           "rescaled_to_ns"), 0)
    factor = 1.0
    timescale = re.search(r"\(TIMESCALE\s+([0-9.]*)\s*(fs|ps|ns|us)\s*\)", text)
    if timescale:
        factor = float(timescale.group(1) or 1) * UNIT_NS[timescale.group(2)]
        if factor != 1.0:
            text = text.replace(timescale.group(0), "(TIMESCALE 1ns)", 1)
            stats["rescaled_to_ns"] = 1

    lines, out, i = text.splitlines(keepends=True), [], 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith(("(VOLTAGE", "(TEMPERATURE", "(PROCESS")):
            line = re.sub(r"([-+0-9.eE]+)::([-+0-9.eE]+)", r"\1:\1:\2", line)
        if stripped.startswith("(INTERCONNECT"):
            stats["dropped_interconnect"] += 1
            i += 1
            continue
        if stripped.startswith("(TIMINGCHECK"):
            stats["dropped_timingcheck"] += 1
            i = skip_balanced(lines, i)
            continue
        if stripped.startswith("(COND"):
            # the next line is the IOPATH of the condition; it carries the closing parenthesis of the COND as well
            i += 1
            if i >= len(lines):
                break
            body = lines[i].rstrip()
            if body.endswith(")"):
                body = body[:-1].rstrip()
            i += 1
            if "[" in body:
                stats["dropped_bus_iopath"] += 1
                continue
            out.append(body + "\n")
            stats["unwrapped_cond"] += 1
            continue
        if stripped.startswith("(IOPATH") and "[" in stripped:
            stats["dropped_bus_iopath"] += 1
            i += 1
            continue
        out.append(line)
        i += 1

    # drop the CELL entries that kept no IOPATH; the file's closing parenthesis is taken off first and put back at the end
    body = "".join(out).rstrip()
    if body.endswith(")"):
        body = body[:-1].rstrip()
    first = body.find("(CELL")
    head, cells = (body, []) if first < 0 else (body[:first], re.split(r"(?=\(CELL\b)", body[first:]))
    kept = []
    for cell in cells:
        if "(IOPATH" in cell:
            cell = re.sub(r"\(IOPATH ([^\n]*?) \(\) ", lambda m: f"(IOPATH {m.group(1)} (0.000000:0.000000:0.000000) ", cell)
            kept.append(dedupe_zero_repeats(cell, stats))
        elif cell.strip():
            stats["dropped_empty_cells"] += 1
    stats["filled_empty_triplets"] = sum(len(re.findall(r"\(IOPATH [^\n]*? \(\) ", c)) for c in cells)
    body = head + "".join(kept).rstrip() + "\n)\n"
    if factor != 1.0:
        body = "".join(scale_iopath(l, factor) if l.lstrip().startswith("(IOPATH") else l
                       for l in body.splitlines(keepends=True))
    return body, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    body, stats = sanitize(a.src.read_text(errors="replace"))
    a.out.write_text(body)
    print(f"wrote {a.out}: " + ", ".join(f"{k} {v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
