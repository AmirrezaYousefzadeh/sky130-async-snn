#!/usr/bin/env python3
"""Turn the per-signal toggle counts of flow/tools/vcd_toggles into OpenSTA activity commands (flow/power.sh).

Usage: toggles_to_activity.py <toggles.tsv> --scope tb/u_dut --period-s <clock period in s> -o activity.tcl
                              [--summary summary.json]
For every variable of the design's scope, one Tcl call that annotates it with
    activity = transitions x clock period / window duration,   duty = high time / window duration,
the definitions of OpenSTA's read_power_activities -vcd (the initial value does not count, x/z transitions count 0.5).
A variable <instance>/<port> is a cell or macro pin; a variable without '/' is a top-level port or a net, whose activity
goes to the net's driver pin. Net commands come after the pin commands: with annotated delays Icarus sometimes leaves a
cell's output-port variable constant while the net it drives toggles, and the net's measured activity must win.
Names are converted to OpenSTA's form (an escaped identifier loses its backslash; [ ] / and \\ inside it are escaped).
The Tcl also writes coverage lists next to the report: annot_found_pins.txt, annot_found_nets.txt and the names that
were not found (annot_notfound_*.txt; supply pins and the internal signals of the cell models are expected there).
"""
import argparse
import json
import sys

TCL_HELPERS = r"""set __n_pins 0; set __n_ports 0; set __n_nets 0
set __nf_pins {}; set __nf_top {}; set __found_pins {}; set __found_nets {}
proc __net_act {net act duty} {
  set n [get_nets -quiet $net]
  if {![llength $n]} { return 0 }
  set drv {}
  foreach p [get_pins -quiet -of_objects $n] { if {[get_property $p direction] eq "output"} { lappend drv $p } }
  if {[llength $drv]} { set_power_activity -pins $drv -activity $act -duty $duty; lappend ::__found_nets $net; return 1 }
  return 0
}
proc __top_act {nm act duty} {
  set p [get_ports -quiet $nm]
  if {[llength $p] && [get_property $p direction] eq "input"} {
    set_power_activity -input_ports $p -activity $act -duty $duty; incr ::__n_ports; return 1
  }
  if {[__net_act $nm $act $duty]} { incr ::__n_nets; return 1 }
  lappend ::__nf_top $nm; return 0
}
proc __pin_act {nm act duty} {
  if {[llength [get_pins -quiet $nm]]} {
    set_power_activity -pins $nm -activity $act -duty $duty; incr ::__n_pins; lappend ::__found_pins $nm; return 1
  }
  if {[__net_act $nm $act $duty]} { incr ::__n_nets; return 1 }
  lappend ::__nf_pins $nm; return 0
}"""

TCL_COVERAGE = r"""puts "toggles_to_activity: annotated $__n_pins pins, $__n_ports input ports and $__n_nets nets (driver pins)"
if {[info exists ::env(OUT)]} {
  foreach {v f} {__found_pins annot_found_pins.txt __found_nets annot_found_nets.txt __nf_pins annot_notfound_pins.txt __nf_top annot_notfound_top.txt} {
    set fh [open "$::env(OUT)/$f" w]; puts $fh [set ::$v]; close $fh
  }
}"""


def sta_name(component):
    """OpenSTA's verilogToSta: an escaped identifier loses its backslash; [ ] / and backslash inside it are escaped."""
    if component.startswith("\\"):
        return "".join(("\\" + c) if c in "[]/\\" else c for c in component[1:].rstrip())
    return component


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv")
    ap.add_argument("--scope", required=True, help="VCD scope of the design, e.g. tb/u_dut")
    ap.add_argument("--period-s", type=float, required=True, help="clock period in seconds (activity is per clock cycle)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()

    header, rows = {}, []
    with open(a.tsv) as f:
        for line in f:
            if line.startswith("#"):
                key, value = line[1:].rstrip("\n").split("\t")
                header[key] = float(value)
                continue
            fields = line.rstrip("\n").split("\t")
            bit = int(fields[3]) if len(fields) > 3 else None          # the bit index of a bus bit (the name is the bus)
            rows.append((fields[0], float(fields[1]), float(fields[2]), bit))
    timescale = header["timescale_s"]
    duration = (header["time_max"] - header["time_min"]) * timescale
    if duration <= 0:
        sys.exit("toggles_to_activity: zero duration")

    prefix = a.scope.rstrip("/") + "/"
    pins, nets, skipped = [], [], 0
    for name, transitions, high, bit in rows:
        if not name.startswith(prefix):
            skipped += 1
            continue
        parts = name[len(prefix):].split("/")
        last = sta_name(parts[-1]) + (f"[{bit}]" if bit is not None else "")
        activity = transitions * a.period_s / duration
        duty = min(max(high * timescale / duration, 0.0), 1.0)
        if len(parts) == 1:
            nets.append(f"__top_act {{{last}}} {activity:.6g} {duty:.6f}")
        else:
            pins.append(f"__pin_act {{{'/'.join(sta_name(p) for p in parts[:-1])}/{last}}} {activity:.6g} {duty:.6f}")
    lines = [f"# generated by flow/tools/toggles_to_activity.py from {a.tsv}",
             f"# window {duration:.6e} s, clock period {a.period_s:.6e} s; activity = transitions x period / window",
             TCL_HELPERS, *pins, "# nets and ports last: their measured activity overrides a constant output-port variable",
             *nets, TCL_COVERAGE]
    open(a.out, "w").write("\n".join(lines) + "\n")
    summary = {"tsv": a.tsv, "duration_s": duration, "period_s": a.period_s, "cycles": duration / a.period_s,
               "pin_variables": len(pins), "top_variables": len(nets), "skipped": skipped,
               "time_min": header["time_min"], "time_max": header["time_max"], "timescale_s": timescale}
    if a.summary:
        json.dump(summary, open(a.summary, "w"), indent=1)
    print(f"toggles_to_activity: {len(pins)} pin variables, {len(nets)} top-level variables, {skipped} skipped; "
          f"window {duration * 1e6:.3f} µs = {duration / a.period_s:.1f} cycles")
    if not pins:
        examples = [name for name, *_ in rows[:3]]
        sys.exit(f"toggles_to_activity: no variable under the scope {prefix!r}; the file has names like {examples}")


if __name__ == "__main__":
    main()
