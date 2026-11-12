#!/usr/bin/env python3
"""Evaluate an exported integer network with the bit-exact, cycle-counting reference (snn.intref) in both execution
modes on real event streams: accuracy versus time since stimulus onset, decision latency at output margins, spikes and
events per decision, queue depths. Also checks the torch layered forward against the reference (bit-exactness) and
measures the gap between the torch async micro-bin approximation and per-event execution.

  python -m snn.eval --model results/models/<name>/model_int.npz --dataset nmnist --n_samples 2000 \
      --modes async,layered --n_eval 1,10,20 --margins 0,32,64,128 --out results/eval/<name>.json
"""
from __future__ import annotations
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from .data import EventData, ROOT, bin_indices
from .model import load_int, QSNN
from .intref import IntNet, sample_on_grid, margin_decisions

_G = {}


def _init(model_path, dataset, split, cfg):
    _G["net"] = IntNet(load_int(model_path)); _G["data"] = EventData(dataset, split); _G["cfg"] = cfg
    _G["data"].stim_us = cfg.get("stim_us"); _G["data"].limit_us = cfg.get("limit_us"); _G["data"].frame_events = cfg.get("frame_events")
    _G["data"].set_delays(cfg.get("delays_us"))


def _one(i):
    net, data, cfg = _G["net"], _G["data"], _G["cfg"]
    t, ch, y = data.get(i)
    grid = cfg["grid_us"]
    res = {}
    for key, (mode, n_eval) in cfg["configs"].items():
        r = net.run(t, ch, mode, cfg["dt_leak_us"], cfg["t_end_us"], n_eval, cfg["tclk_ns"], cfg["pipe"], cfg["dt_ref_us"])
        og = sample_on_grid(r["time_us"], r["o"], grid)
        correct_t = (og.argmax(1) == y) & (og.max(1) > og.min(1))      # all-zero output counts as undecided
        md = margin_decisions(r["time_us"], r["o"], cfg["margins"])
        # spikes (hidden) and input events consumed up to the decision record; spikes emitted up to each grid time
        spk = r["spk_cum"]
        gi = np.searchsorted(r["time_us"], grid, side="right") - 1
        spk_t = np.where(gi >= 0, spk[np.maximum(gi, 0)], 0) if len(spk) else np.zeros(len(grid), np.int64)
        res[key] = dict(correct_t=correct_t.astype(np.uint8),
                        margin=[(tm, p == y, p >= 0, int(spk[ix]) if ix >= 0 else 0, int(np.searchsorted(t, tm, side="right")) if ix >= 0 else 0)
                                for tm, p, ix in md],
                        spikes_t=spk_t.astype(np.int64), spikes=r["spikes"].tolist(), qmax=r["qmax"].tolist(), n_events=r["n_events"],
                        n_rec=int(len(r["o"])), overflow=r["overflow"], final_o=r["o"][-1].tolist() if len(r["o"]) else None)
    return i, y, res


def check_exact(model_path, dataset, split, n, dt_us, t_end_us, n_eval, tclk_ns, pipe, stepped="layered"):
    """Torch stepped forward (layered or pipelined) versus the reference stepped run (must match); torch async versus
    reference async (gap)."""
    import torch
    d = load_int(model_path); data = EventData(dataset, split); net = IntNet(d)
    sm = d.get("config", {}).get("stim_ms"); data.stim_us = int(sm * 1000) if sm is not None else None
    data.limit_us = d.get("config", {}).get("in_limit_us"); data.frame_events = d.get("config", {}).get("frame_events")
    data.set_delays(d.get("config", {}).get("in_delays_us"))
    le = int(d.get("leak_every", 1))
    m = QSNN(d["n_in"], d["hidden"], d["n_out"], d["wbits"], d["theta"], d["k_leak"], d["k_out"], d["vbits"], d["obits"], leak_every=le,
             wscale=int(d.get("wscale", 1)))
    with torch.no_grad():
        for l, W in enumerate(d["W"]): m.W[l].copy_(torch.from_numpy(W.astype(np.float32)))
    T = int(t_end_us // dt_us)                 # dt_us = micro-bin; the reference leak period is dt_us * le
    mism_layered = 0; agree_async = 0; absdiff = []
    for i in range(n):
        t, ch, y = data.get(i)
        b, c = bin_indices(t, ch, dt_us, T)
        x = torch.zeros(1, T, d["n_in"]); x.index_put_((torch.zeros(len(b), dtype=torch.long), torch.from_numpy(b), torch.from_numpy(c)), torch.ones(len(b)), accumulate=True)
        with torch.no_grad():
            o_l, _ = m(x, stepped, n_eval); o_a, _ = m(x, "async", 1)
        r_l = net.run(t, ch, stepped, dt_us * le, t_end_us, max(1, n_eval // le), tclk_ns, pipe)
        # reference records one entry per leak tick (tick k -> torch step k*le-1)
        ref = r_l["o"][:T // le]
        tor = o_l[0].numpy().astype(np.int64)[le - 1::le][:T // le]
        if ref.shape[0] != T // le or not np.array_equal(ref, tor): mism_layered += 1
        r_a = net.run(t, ch, "async", dt_us * le, t_end_us, 1, tclk_ns, pipe, dt_us * int(d.get("refractory_every", le)))
        fa = r_a["o"][-1] if len(r_a["o"]) else np.zeros(d["n_out"], np.int64)
        ta = o_a[0, -1].numpy().astype(np.int64)
        agree_async += int(fa.argmax() == ta.argmax()); absdiff.append(np.abs(fa - ta).mean())
    return {"n": n, "layered_mismatching_samples": mism_layered, "async_argmax_agreement": agree_async / n,
            "async_mean_abs_diff": float(np.mean(absdiff))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dataset", default="nmnist")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--dt_leak_us", type=int, default=None, help="default: the training micro-bin x leak_every")
    ap.add_argument("--dt_ref_us", type=int, default=None, help="async refractory period; default from the model (refractory_every x dt_us), 0 = none")
    ap.add_argument("--t_end_ms", type=float, default=None)
    ap.add_argument("--stim_ms", type=float, default=None, help="drop input events at or after this time (default: the model's)")
    ap.add_argument("--modes", default="async,layered", help="async, layered, pipelined (comma separated)")
    ap.add_argument("--n_eval", default="1,10")
    ap.add_argument("--margins", default="0,32,64,128,256")
    ap.add_argument("--grid_ms", type=float, default=1.0)
    ap.add_argument("--tclk_ns", type=int, default=20)
    ap.add_argument("--pipe", type=int, default=2)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--check_exact", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    d = load_int(a.model); cfg0 = d.get("config", {})
    dt = a.dt_leak_us or int(cfg0.get("dt_us", 1000)) * int(d.get("leak_every", 1))      # leak period of the reference
    t_end_us = int((a.t_end_ms or cfg0.get("t_end_ms", 100)) * 1000)
    grid = np.arange(a.grid_ms * 1000, t_end_us + 1, a.grid_ms * 1000).astype(np.int64)
    configs = {}
    for mode in a.modes.split(","):
        if mode == "async": configs["async"] = ("async", 1)
        else:
            for n in a.n_eval.split(","): configs[f"{mode}{n}"] = (mode, int(n))
    margins = [int(m) for m in a.margins.split(",")]
    dt_ref = a.dt_ref_us if a.dt_ref_us is not None else int(cfg0.get("dt_us", 1000)) * int(d.get("refractory_every", d.get("leak_every", 1)))
    sm = a.stim_ms if a.stim_ms is not None else cfg0.get("stim_ms")
    cfg = dict(dt_leak_us=dt, t_end_us=t_end_us, grid_us=grid, configs=configs, margins=margins, tclk_ns=a.tclk_ns, pipe=a.pipe, dt_ref_us=dt_ref,
               stim_us=int(sm * 1000) if sm is not None else None, limit_us=cfg0.get("in_limit_us"), frame_events=cfg0.get("frame_events"),
               delays_us=cfg0.get("in_delays_us"))
    data = EventData(a.dataset, a.split)
    n = a.n_samples or len(data)
    ids = list(range(n))
    t0 = time.time()
    print(f"evaluating {n} {a.split} samples of {a.dataset}, configs {list(configs)}, dt_leak {dt} us, window {t_end_us/1000:.0f} ms", flush=True)
    stepped = cfg0.get("mode") if cfg0.get("mode") in ("layered", "pipelined") else "layered"
    exact = check_exact(a.model, a.dataset, a.split, a.check_exact, int(cfg0.get("dt_us", 1000)), t_end_us, int(a.n_eval.split(",")[-1]) * int(d.get("leak_every", 1)), a.tclk_ns, a.pipe, stepped) if a.check_exact else None
    if exact: print("exactness:", exact, flush=True)
    with Pool(a.workers, initializer=_init, initargs=(a.model, a.dataset, a.split, cfg)) as pool:
        results = pool.map(_one, ids, chunksize=8)
    summary = {"model": str(a.model), "dataset": a.dataset, "split": a.split, "n_samples": n, "dt_leak_us": dt, "dt_ref_us": dt_ref, "t_end_us": t_end_us,
               "tclk_ns": a.tclk_ns, "pipe": a.pipe, "margins": margins, "grid_ms": a.grid_ms, "exactness": exact, "configs": {}}
    for key in configs:
        ct = np.stack([r[2][key]["correct_t"] for r in results]).astype(np.float64)
        spikes = np.array([r[2][key]["spikes"] for r in results], np.float64)
        qmax = np.array([r[2][key]["qmax"] for r in results]).max(0)
        n_events = np.mean([r[2][key]["n_events"] for r in results])
        marg = {}
        for mi, m in enumerate(margins):
            rows = [r[2][key]["margin"][mi] for r in results]
            decided = np.array([x[2] for x in rows]); correct = np.array([x[1] for x in rows]); tm = np.array([x[0] for x in rows])
            s2d = np.array([x[3] for x in rows], np.float64); e2d = np.array([x[4] for x in rows], np.float64)
            marg[str(m)] = {"accuracy": float(correct.mean()), "decided": float(decided.mean()),
                            "spikes_to_decision_mean": float(s2d[decided].mean()) if decided.any() else None,
                            "events_to_decision_mean": float(e2d[decided].mean()) if decided.any() else None,
                            "latency_ms_mean": float(np.mean(tm[decided]) / 1000) if decided.any() else None,
                            "latency_ms_median": float(np.median(tm[decided]) / 1000) if decided.any() else None,
                            "latency_ms_p90": float(np.percentile(tm[decided], 90) / 1000) if decided.any() else None}
        summary["configs"][key] = {"mode": configs[key][0], "n_eval": configs[key][1], "accuracy_final": float(ct[:, -1].mean()),
                                   "accuracy_vs_time": {"grid_ms": (grid / 1000).tolist(), "accuracy": ct.mean(0).tolist()},
                                   "spikes_per_sample_per_layer": spikes.mean(0).tolist(), "events_per_sample": float(n_events),
                                   "spikes_vs_time": np.stack([r[2][key]["spikes_t"] for r in results]).mean(0).tolist(),
                                   "queue_max_depth": qmax.tolist(), "margin_decisions": marg,
                                   "overflow_samples": int(sum(r[2][key]["overflow"] for r in results))}
        print(f"{key:10s} final acc {ct[:, -1].mean():.4f}  acc@25/50/75% {ct[:, len(grid)//4-1].mean():.3f}/{ct[:, len(grid)//2-1].mean():.3f}/{ct[:, 3*len(grid)//4-1].mean():.3f}  "
              f"spikes/sample {spikes.mean(0).round(0).tolist()} qmax {qmax.tolist()}", flush=True)
        for m in margins:
            s = marg[str(m)]
            print(f"    margin {m:4d}: acc {s['accuracy']:.4f} decided {s['decided']:.3f} latency median {s['latency_ms_median']} ms mean {s['latency_ms_mean']} p90 {s['latency_ms_p90']}"
                  f"  spikes to decision {s['spikes_to_decision_mean']} events {s['events_to_decision_mean']}")
    out = a.out or (ROOT / "results" / "eval" / (a.model.parent.name + f"_{a.split}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
