#!/usr/bin/env python3
"""Summary table from evaluation JSON files: per execution config the final accuracy, anytime accuracy at a few
times, the time at which the anytime curve first reaches given accuracy levels, spikes per sample, and the synaptic
operations per sample (energy proxy: every spike costs a weight-row read plus one add per neuron of the next layer;
every input event one row of layer 1).

  python -m snn.summarize results/eval/*.json --levels 0.90,0.95,0.97
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .model import load_int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--levels", default="0.90,0.95,0.97")
    ap.add_argument("--at_ms", default="25,50,100")
    a = ap.parse_args()
    levels = [float(x) for x in a.levels.split(",")]; at_ms = [float(x) for x in a.at_ms.split(",")]
    hdr = ["model", "exec", "final", *[f"acc@{int(t)}ms" for t in at_ms], *[f"t({l:.0%})" for l in levels], "spikes/sample", "SynOps/sample"]
    rows = []
    for f in a.files:
        d = json.load(open(f))
        model = load_int(d["model"]); sizes = model["hidden"] + [model["n_out"]]
        name = Path(d["model"]).parent.name.replace("nmnist_", "").replace("_dt1000_T100_h512-384-256-192-128_w8_th256_s0", " ")
        for key, c in d["configs"].items():
            g = np.array(c["accuracy_vs_time"]["grid_ms"]); acc = np.array(c["accuracy_vs_time"]["accuracy"])
            spikes = np.array(c["spikes_per_sample_per_layer"])
            synops = c["events_per_sample"] * sizes[0] + sum(spikes[l] * sizes[l + 1] for l in range(len(spikes)))
            at = [acc[np.searchsorted(g, t)] if t <= g[-1] else float("nan") for t in at_ms]
            tl = []
            for l in levels:
                hit = np.flatnonzero(acc >= l); tl.append(f"{g[hit[0]]:.0f}" if len(hit) else "-")
            rows.append([name, key, f"{c['accuracy_final']:.4f}", *[f"{v:.3f}" for v in at], *tl, f"{spikes.sum():.0f}", f"{synops/1e6:.2f} M"])
    w = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    print(" | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    print("-+-".join("-" * x for x in w))
    for r in rows: print(" | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)))


if __name__ == "__main__":
    main()
