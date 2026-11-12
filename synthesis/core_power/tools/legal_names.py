#!/usr/bin/env python3
"""Give every escaped identifier of a routed netlist a plain name, consistently in the netlist, its SDF and its SPEF.

Usage: legal_names.py <netlist.nl.v> <delays.sdf> <parasitics.spef> <out dir>
Writes <out dir>/design.nl.v, design.sdf, design.spef and names.tsv (plain name <tab> original name).

Why: the routed netlist of a design with sub-modules or generate loops names its cells and nets by their hierarchical
path, as Verilog escaped identifiers such as `\\u_rom.u_col_0/_307_ ` or `\\g_ws[9].u_icg `. Icarus Verilog cannot
annotate SDF delays onto such instances (it splits the name at the dot), and OpenSTA reads the slash in a SPEF name as a
hierarchy divider, so the parasitics of these nets are lost. The gate-level simulation and the power analysis therefore
both run on renamed copies: every escaped identifier becomes a plain one (each character other than a letter, a digit
or '_' replaced by '_', plus a suffix when that name exists already). The netlist spells such a name `\\a.b/c `, SDF and
SPEF spell it `a\\.b\\/c`, and SPEF also `a\\.b/c` (so a net of a sub-module, `\\u_demux/_000_ ` in the netlist, is
plain `u_demux/_000_` in the SPEF); ports and bus bits keep their names.
"""
import re
import sys
from pathlib import Path

ESCAPED = re.compile(r"\\(\S+)(?=\s)")              # a Verilog escaped identifier and its terminating whitespace
PLAIN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_$]*\b")
TOOL_NAME = re.compile(r"(?:[A-Za-z0-9_$/]|\\.)+")  # an SDF/SPEF name, with backslash escapes and hierarchy slashes


def tool_forms(name):
    """The SDF/SPEF spellings of an escaped identifier: every special character preceded by a backslash, and the same
    with the slash left bare (OpenROAD's SPEF writes the nets inside a module that way); `$` escaped or bare (ORFS
    names Yosys' `\\u_wmem/lat[39]$_DLATCH_P_` instances as `u_wmem/lat\\[39\\]$_DLATCH_P_` in the SPEF)."""
    escaped = re.sub(r"([^A-Za-z0-9_$])", r"\\\1", name)
    forms = (escaped, escaped.replace("\\/", "/"))
    return forms + tuple(f.replace("$", "\\$") for f in forms)


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    netlist, sdf, spef, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    text = netlist.read_text()

    taken = set(PLAIN.findall(ESCAPED.sub(" ", text)))
    plain_of = {}
    for name in dict.fromkeys(ESCAPED.findall(text)):
        base = re.sub(r"[^A-Za-z0-9_]", "_", name)
        base = base if re.match(r"[A-Za-z_]", base) else "n_" + base
        candidate, k = base, 1
        while candidate in taken:
            candidate, k = f"{base}_{k}", k + 1
        taken.add(candidate)
        plain_of[name] = candidate

    (out / "design.nl.v").write_text(ESCAPED.sub(lambda m: plain_of[m.group(1)], text))
    by_tool_form = {form: plain for name, plain in plain_of.items() for form in tool_forms(name)}

    def plain(m):                                       # only escaped or hierarchical names can need a new name
        name = m.group(0)
        return by_tool_form.get(name, name) if "\\" in name or "/" in name else name

    def rename(line):
        return TOOL_NAME.sub(plain, line)

    with open(sdf, errors="replace") as src, open(out / "design.sdf", "w") as dst:
        for line in src:
            dst.write(rename(line) if "(INSTANCE" in line or "(INTERCONNECT" in line else line)
    with open(spef, errors="replace") as src, open(out / "design.spef", "w") as dst:
        body = False                                    # names start with the name map; the header stays as it is
        for line in src:
            body = body or line.startswith("*NAME_MAP")
            dst.write(rename(line) if body and ("\\" in line or "/" in line) else line)
    with open(out / "names.tsv", "w") as f:
        f.writelines(f"{plain}\t{name}\n" for name, plain in plain_of.items())
    print(f"legal_names: {len(plain_of)} escaped identifiers renamed -> {out}/design.nl.v, design.sdf, design.spef")


if __name__ == "__main__":
    main()
