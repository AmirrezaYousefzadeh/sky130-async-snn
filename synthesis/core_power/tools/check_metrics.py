#!/usr/bin/env python3
"""Decide from the metrics of a LibreLane run whether the hardened design is accepted: routing DRC
clean, setup and hold met at one corner (--corner; run it once per sign-off corner), no power-grid violation, no
unmatched SPEF entry in the own sign-off, and optionally at most --cell-cap standard cells. Tap cells are not counted:
their number grows with the core area, not with the design.

The flow's slacks are post-route only if the run reached OpenROAD.STAPostPNR (skipped with RUN_MCSTA: false or an
earlier --to): otherwise the per-corner metrics keep STAPrePNR's values (unplaced netlist, ideal clocks) for every
corner STAMidPNR did not analyse (LibreLane issue #1040). A <run dir>/final/metrics.json without a STAPostPNR step
directory beside it is rejected unless --signoff gives the timing.

The timing comes from --signoff, the result of flow/tools/signoff_sta.tcl on the renamed netlist and SPEF that the
simulation and the power analysis also read (with OpenLane 2, the flow's own sign-off lost the parasitics of the nets
inside sub-modules); the flow's own slacks are kept for comparison. Without --signoff the flow's slacks are used.

With --synth-stat (the run's Yosys reports/stat.json), cells left unmapped by synthesis reject the run. LibreLane
3.0.16's own check, which the configuration therefore disables, counts every cell type starting with `$`, including
the kept sub-modules of parametrised modules (named `$paramod...`); those are modules of the design, not unmapped cells.

Usage: check_metrics.py <run dir>/final/metrics.json [--signoff signoff.json] [--synth-stat stat.json]
                        [--corner nom_tt_025C_1v80] [--cell-cap N] [-o summary.json]
       check_metrics.py --synth-stat stat.json      (right after synthesis: unmapped cells only)
Prints a one-line summary and PASS or FAIL; exit status 0 on PASS.
"""
import argparse
import glob
import json
import os


def unmapped(stat_json):
    """Cells of a `$` type that is not a module of the design (kept `$paramod...` sub-modules are modules)."""
    stat = json.load(open(stat_json))
    return sum(n for cell, n in stat["design"]["num_cells_by_type"].items()
               if cell.startswith("$") and cell not in stat["modules"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="?")
    ap.add_argument("--corner", default="nom_tt_025C_1v80")
    ap.add_argument("--cell-cap", type=int, default=None)
    ap.add_argument("--signoff", default=None, help="signoff.json of flow/tools/signoff_sta.tcl")
    ap.add_argument("--synth-stat", default=None, help="reports/stat.json of the run's Yosys synthesis")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    if a.metrics is None:
        if not a.synth_stat:
            ap.error("give the run's final/metrics.json, or --synth-stat alone")
        n = unmapped(a.synth_stat)
        print(f"{n} cells left unmapped by synthesis")
        print("PASS" if n == 0 else "FAIL")
        raise SystemExit(0 if n == 0 else 1)
    m = json.load(open(a.metrics))
    taps = m.get("design__instance__count__class:tap_cell", 0)
    all_cells = m.get("design__instance__count__stdcell")
    summary = {
        "cells": None if all_cells is None else all_cells - taps,
        "tap_cells": taps,
        "cell_area_um2": m.get("design__instance__area__stdcell"),
        "die_area_um2": m.get("design__die__area"),
        "core_area_um2": m.get("design__core__area"),
        "utilization": m.get("design__instance__utilization"),
        "setup_ws_ns": m.get(f"timing__setup__ws__corner:{a.corner}"),
        "hold_ws_ns": m.get(f"timing__hold__ws__corner:{a.corner}"),
        "drc_errors": m.get("route__drc_errors"),
        "power_grid_violations": m.get("design__power_grid_violation__count", 0),
        "antenna_violating_nets": m.get("antenna__violating__nets"),   # reported only: no antenna repair in the flow
        "corner": a.corner,
        "cell_cap": a.cell_cap,
    }
    final_dir = os.path.dirname(os.path.abspath(a.metrics))
    if os.path.basename(final_dir) == "final":
        summary["librelane_timing_post_route"] = bool(
            glob.glob(os.path.join(os.path.dirname(final_dir), "*-openroad-stapostpnr*")))
    if a.signoff:
        signoff = json.load(open(a.signoff))
        summary["librelane_setup_ws_ns"], summary["librelane_hold_ws_ns"] = summary["setup_ws_ns"], summary["hold_ws_ns"]
        summary["setup_ws_ns"], summary["hold_ws_ns"] = signoff["setup_ws_ns"], signoff["hold_ws_ns"]
        summary["timing"] = "flow/tools/signoff_sta.tcl (renamed netlist and SPEF)"
        summary["spef_unmatched"] = signoff["spef_unmatched"]
    if a.synth_stat:
        summary["unmapped_cells"] = unmapped(a.synth_stat)
    reasons = []
    if summary.get("unmapped_cells"):
        reasons.append(f"{summary['unmapped_cells']} cells left unmapped by synthesis")
    if summary["drc_errors"] != 0:
        reasons.append(f"routing DRC errors: {summary['drc_errors']}")
    if not a.signoff and summary.get("librelane_timing_post_route") is False:
        reasons.append("no STAPostPNR step in the run (RUN_MCSTA off or the run stopped earlier): its slacks are "
                       "pre-placement values")
    if summary["setup_ws_ns"] is None or summary["setup_ws_ns"] < 0:
        reasons.append(f"setup slack {summary['setup_ws_ns']} ns at {a.corner}")
    if summary["hold_ws_ns"] is None or summary["hold_ws_ns"] < 0:
        reasons.append(f"hold slack {summary['hold_ws_ns']} ns at {a.corner}")
    if summary.get("spef_unmatched"):
        reasons.append(f"{summary['spef_unmatched']} SPEF entries without a match in the sign-off timing")
    if summary["power_grid_violations"]:
        reasons.append(f"{summary['power_grid_violations']} power-grid violations (unconnected supply pins)")
    if summary["cells"] is None or (a.cell_cap is not None and summary["cells"] > a.cell_cap):
        reasons.append(f"{summary['cells']} standard cells, cap {a.cell_cap}")
    summary["pass"] = not reasons
    summary["reasons"] = reasons

    print(f"cells {summary['cells']} (+ {taps} taps)  cell area {summary['cell_area_um2']} µm²  "
          f"die {summary['die_area_um2']} µm²  "
          f"setup {summary['setup_ws_ns']} ns  hold {summary['hold_ws_ns']} ns  drc {summary['drc_errors']}  "
          f"pdn {summary['power_grid_violations']}")
    print("PASS" if summary["pass"] else "FAIL: " + "; ".join(reasons))
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)
    raise SystemExit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
