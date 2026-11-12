#!/usr/bin/env python3
"""Per-core workload of time-stepped execution under a spatial mapping, against hard-real-time sizing.

Spatial mapping (SpiNNaker / Loihi / TrueNorth style): every hidden layer is split into `--cores` cores of contiguous
neurons (for a grouped layer the groups coincide with the cores when --cores equals the number of groups); the output
layer is one core. A core updates only its own neurons, one synaptic operation per cycle per neuron unit.

Work of core c in step t, in synaptic operations:
    ops(c, t) = pre(c, t) * n_c + n_c
pre(c, t) = spikes in step t from the presynaptic neurons connected to core c (input events for the first layer),
n_c = neurons in the core; the second term is the per-step scan of the core's neurons (leak and threshold).

Hard-real-time sizing: every connected presynaptic neuron fires once in the step (the refractory window is the step)
and every input channel emits one event, so cap(c) = n_pre(c) * n_c + n_c must fit in one step. A core sized this
way is busy u = ops / cap of each step and idle for the rest.

Reported per layer: cap, utilisation u (mean, p50, p99, max over samples x steps x cores), the spread of the
per-sample mean utilisation (sparse versus busy data points), the core-time a per-layer barrier leaves idle while
cores wait for the slowest core of their layer, and the slack 1 - max_c u per step: the part of the step a
self-timed schedule (barrier or event-driven) does not have to wait for, per layer hop.

Spikes come from the torch stepped forward, which matches the stepped reference (snn/intref.py) bit-exactly except
for rare 16-bit saturation-order differences in audio bursts.

  python -m snn.workload --model results/models/<run>/model_int.npz --dataset nmnist --cores 8
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from .data import EventData, Batcher, ROOT, input_groups
from .model import QSNN, load_int


def build(d: dict, dev, dataset: str):
    cfg = d.get("config", {})
    le = int(d.get("leak_every", 1))
    m = QSNN(d["n_in"], d["hidden"], d["n_out"], d["wbits"], d["theta"], d["k_leak"], d["k_out"], d["vbits"], d["obits"],
             leak_every=le, wscale=int(d.get("wscale", 1))).to(dev)
    if int(cfg.get("core_graph", 0) or 0):
        cg = int(cfg["core_graph"]); m.set_core_graph(cg, int(cfg.get("core_k", 2)), input_groups(dataset, d["hidden"][0] // cg, cg))
    G = int(cfg.get("groups", 1) or 1)
    if G > 1:
        m.set_groups(G, [int(x) for x in str(cfg.get("group_layers", "")).split(",") if x != ""])
    with torch.no_grad():
        for l, W in enumerate(d["W"]): m.W[l].copy_(torch.from_numpy(W.astype(np.float32)))
    m.snapped = True
    return m, cfg, G


def core_partition(sizes, n_cores):
    """Core index of every neuron of every layer after the input: hidden layers in n_cores blocks, output one core."""
    parts = []
    for l, n in enumerate(sizes[1:]):
        C = n_cores if l < len(sizes) - 2 else 1
        assert n % C == 0, f"layer of {n} neurons not divisible into {C} cores"
        parts.append(np.arange(n) // (n // C))
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--cores", type=int, default=8, help="cores per hidden layer")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_int(a.model)
    model, cfg, G = build(d, dev, a.dataset)
    dt_us, T, n_eval = int(cfg["dt_us"]), int(cfg["T"]), int(cfg.get("n_eval", 1))
    mode = cfg.get("mode") if cfg.get("mode") in ("layered", "pipelined") else "layered"
    cg = int(cfg.get("core_graph", 0) or 0) or int(cfg.get("core_prune", 0) or 0)
    if int(cfg.get("core_prune", 0) or 0):              # pruned core graph: the connections are the non-zero weight blocks
        with torch.no_grad():
            for l, W in enumerate(d["W"][:-1]):
                blk = (W != 0).reshape(W.shape[0], -1, cg).any(2)
                model.masks()[l].copy_(torch.from_numpy(np.repeat(blk, cg, axis=1).astype(np.float32)).to(dev))
    sizes = [d["n_in"]] + list(d["hidden"]) + [d["n_out"]]
    L = len(d["hidden"])
    n_cores = sizes[1] // cg if cg else a.cores
    parts = core_partition(sizes, n_cores)
    # connectivity of every core: conn[l] (n_pre, C_l) = 1 if any neuron of the core receives from presynaptic i
    conn, n_c, n_pre = [], [], []
    if cg:                                   # core graph: one class-integrator core per last-layer group
        parts[L] = np.zeros(d["n_out"], dtype=np.int64)
    for l in range(L + 1):
        mask = model.masks()[l].detach().cpu().numpy() > 0                      # (sizes[l], sizes[l+1])
        C = parts[l].max() + 1
        cm = np.stack([mask[:, parts[l] == c].any(1) for c in range(C)], 1).astype(np.float32)
        if cg and l == L:
            Gl = sizes[L] // cg
            cm = (np.arange(sizes[L])[:, None] // cg == np.arange(Gl)[None, :]).astype(np.float32)
            conn.append(torch.from_numpy(cm).to(dev)); n_c.append(np.full(Gl, float(d["n_out"]))); n_pre.append(cm.sum(0).astype(np.float64))
            continue
        conn.append(torch.from_numpy(cm).to(dev))
        n_c.append(np.bincount(parts[l]).astype(np.float64)); n_pre.append(cm.sum(0).astype(np.float64))
    cap = [n_pre[l] * n_c[l] + n_c[l] for l in range(L + 1)]

    data = EventData(a.dataset, a.split)
    if cfg.get("stim_ms") is not None: data.stim_us = int(cfg["stim_ms"] * 1000)
    data.limit_us = cfg.get("in_limit_us"); data.frame_events = cfg.get("frame_events")
    if a.n_samples: data = data.subset(np.arange(min(a.n_samples, len(data))))
    bat = Batcher(data, dt_us, T, a.batch, False, dev)
    U = [[] for _ in range(L + 1)]                                           # utilisation (samples, steps, cores)
    t0 = time.time()
    with torch.no_grad():
        for x, y, ids, t_last in bat:
            _, _, rec = model(x, mode, n_eval, record=True)
            steps = [t for t in range(T) if t % n_eval == n_eval - 1]
            for l in range(L + 1):
                P = x if l == 0 else torch.stack([rec[t][l - 1] for t in range(T)], 1)   # (B, T, n_pre)
                if mode == "pipelined" and l > 0:     # a step integrates the spikes of the previous evaluation
                    P = torch.cat([torch.zeros_like(P[:, :n_eval]), P[:, :-n_eval]], 1)
                P = P[:, steps]
                pre = P @ conn[l]                                                # (B, steps, C)
                ops = pre * torch.from_numpy(n_c[l]).float().to(dev) + torch.from_numpy(n_c[l]).float().to(dev)
                U[l].append((ops / torch.from_numpy(cap[l]).float().to(dev)).cpu().numpy().astype(np.float32))
    U = [np.concatenate(u, 0) for u in U]
    names = [f"hidden {l}" for l in range(L)] + ["output"]
    grouped_layers = [int(x) for x in str(cfg.get("group_layers", "")).split(",") if x != ""] if G > 1 else []
    res = {"model": str(a.model), "dataset": a.dataset, "split": a.split, "n_samples": int(U[0].shape[0]), "cores_per_hidden_layer": n_cores,
           "groups": G, "grouped_weight_layers": grouped_layers, "step_us": dt_us * n_eval, "steps": int(U[0].shape[1]), "layers": []}
    print(f"{a.dataset} {a.model.parent.name}: {U[0].shape[0]} samples, {U[0].shape[1]} steps of {dt_us * n_eval / 1000:g} ms, "
          f"{n_cores} cores per hidden layer, groups {G} in weight layers {grouped_layers}, core graph {cg}, mode {mode}", flush=True)
    print(f"{'layer':9s} {'cores':>5s} {'n/core':>6s} {'pre/core':>8s} {'cap ops':>9s} {'u mean':>7s} {'u p50':>6s} {'u p99':>6s} {'u max':>6s} "
          f"{'sample p10/p50/p90':>19s} {'barrier idle':>12s} {'slack':>6s}")
    tot_ops = tot_cap = 0.0
    for l in range(L + 1):
        u = U[l]                                                             # (S, steps, C)
        per_sample = u.mean((1, 2))
        mx = u.max(2)                                                        # slowest core per step
        busy = u.sum(2); idle = 1.0 - busy.sum() / max(1e-12, (u.shape[2] * mx).sum())
        slack = float((1.0 - mx).mean())
        layer = {"name": names[l], "cores": int(u.shape[2]), "neurons_per_core": n_c[l].tolist(), "pre_per_core": n_pre[l].tolist(),
                 "cap_ops_per_step": cap[l].tolist(), "u_mean": float(u.mean()), "u_p50": float(np.percentile(u, 50)),
                 "u_p99": float(np.percentile(u, 99)), "u_max": float(u.max()),
                 "u_per_sample_p10_p50_p90": [float(np.percentile(per_sample, q)) for q in (10, 50, 90)],
                 "barrier_idle_fraction": float(idle), "slack_mean": slack,
                 "u_mean_per_step": u.mean((0, 2)).tolist(), "u_mean_per_core": u.mean((0, 1)).tolist(),
                 "core_imbalance_cv": float((u.std(2) / np.maximum(u.mean(2), 1e-12)).mean())}
        res["layers"].append(layer)
        tot_ops += float((u * cap[l][None, None, :]).sum()); tot_cap += float(cap[l].sum() * u.shape[0] * u.shape[1])
        print(f"{names[l]:9s} {u.shape[2]:5d} {n_c[l].mean():6.0f} {n_pre[l].mean():8.0f} {cap[l].mean():9.0f} {u.mean():7.3f} "
              f"{np.percentile(u, 50):6.3f} {np.percentile(u, 99):6.3f} {u.max():6.3f} "
              f"{'%.3f/%.3f/%.3f' % tuple(layer['u_per_sample_p10_p50_p90']):>19s} {idle:12.3f} {slack:6.3f}")
    res["chip_utilisation"] = tot_ops / tot_cap
    print(f"chip utilisation (all ops / worst-case capacity): {res['chip_utilisation']:.4f}  ({time.time() - t0:.0f} s)")
    out = a.out or (ROOT / "results" / "workload" / f"{a.model.parent.name}_{a.split}_c{n_cores}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
