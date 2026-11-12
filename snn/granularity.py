#!/usr/bin/env python3
"""Why a network trained in time steps loses accuracy when executed event-driven: evaluate it with
the torch model at finer threshold-check granularity, from the trained schedule to per-event execution.

  pipelined   the trained schedule: one threshold check per step T, one step of delay per layer hop
  layered     one check per step, but every layer passes its spikes on within the step (no hop delay)
  async k     micro-bins of T/k, every layer checks its thresholds after every micro-bin and passes its spikes on at
              once, the leak stays at every step: k checks per step on partial sums, as event-driven execution does
              per event (k -> infinity); with a refractory of one step (one spike per neuron per step, as trained and
              as method M1) or none (the event-driven hardware without M1, a neuron fires once per micro-bin at most)
Accuracy at the end of the window (a tie between the top two classes is no decision) and spikes per layer per sample.
Per-event numbers come from snn.system (event:wire with and without --wire_ref).
Overdrive (trained schedule): per layer, how far a firing neuron's membrane exceeds the threshold at its step (the
hard reset discards the excess; event-driven execution turns it into further spikes), and how many spikes the step's
positive input alone could fire, floor(positive input / threshold), against the one spike a step allows.

  python -m snn.granularity --model results/models/<run>/model_int.npz --dataset ssc --k 1,2,4,8,16 --n_samples 2000
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from .data import EventData, Batcher, ROOT
from .model import QSNN, load_int


def build(d, leak_every, refractory_every, dev):
    m = QSNN(d["n_in"], d["hidden"], d["n_out"], d["wbits"], d["theta"], d["k_leak"], d["k_out"], d["vbits"], d["obits"],
             leak_every=leak_every, refractory_every=refractory_every, wscale=int(d.get("wscale", 1)),
             spiking_out=bool(d.get("spiking_out")), theta_out=float(d.get("theta_out") or 256)).to(dev)
    with torch.no_grad():
        for l, W in enumerate(d["W"]): m.W[l].copy_(torch.from_numpy(W.astype(np.float32)))
    m.snapped = True; m.eval()
    return m


@torch.no_grad()
def run(m, data, dt_us, n_bins, mode, batch, dev):
    correct = 0; n = 0; spk = np.zeros(len(m.hidden))
    for x, y, _, _ in Batcher(data, dt_us, n_bins, batch, False, dev):
        o, s = m(x, mode, 1)
        top = o[:, -1].topk(2, dim=1).values
        correct += ((o[:, -1].argmax(1) == y) & (top[:, 0] > top[:, 1])).sum().item()
        spk += s.cpu().numpy(); n += len(y)
    return correct / n, (spk / n).tolist()


@torch.no_grad()
def overdrive(m, data, dt_us, n_bins, batch, dev):
    """The pipelined forward of QSNN (hidden layers only, leak every step) with statistics at every threshold check."""
    Wq = [w.detach() for w in m.qweights()]; L = len(m.hidden)
    th = [float(m.theta_l(l)) for l in range(L)]; ks = [m.k_l(l) for l in range(L)]
    Wp = [w.clamp(min=0) for w in Wq]
    fired = np.zeros(L); excess = np.zeros(L); cap = np.zeros(L); n = 0
    from .model import leak
    for x, _, _, _ in Batcher(data, dt_us, n_bins, batch, False, dev):
        B = x.shape[0]; n += B
        v = [x.new_zeros(B, h) for h in m.hidden]; prev = [x.new_zeros(B, h) for h in m.hidden]
        for t in range(n_bins):
            v[0] = (v[0] + x[:, t] @ Wq[0]).clamp(-m.vmax - 1, m.vmax)
            for l in range(L): v[l] = leak(v[l], ks[l])
            new = []
            for l in range(L):
                if l > 0: v[l] = (v[l] + prev[l - 1] @ Wq[l]).clamp(-m.vmax - 1, m.vmax)
                pos = (x[:, t] @ Wp[0]) if l == 0 else (prev[l - 1] @ Wp[l])
                s = (v[l] >= th[l]).float()
                fired[l] += s.sum().item(); excess[l] += (s * v[l] / th[l]).sum().item()
                cap[l] += torch.floor(pos / th[l]).clamp(min=0).sum().item()
                v[l] = v[l] * (1 - s); new.append(s)
            prev = new
    return [dict(spikes=fired[l] / n, v_over_theta_when_firing=excess[l] / max(1, fired[l]), positive_input_spikes=cap[l] / n)
            for l in range(L)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", default="1,2,4,8,16", help="micro-bins per step (T must divide into whole microseconds)")
    ap.add_argument("--n_samples", type=int, default=2000, help="test samples, a fixed random subset (0 = all)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--overdrive_only", action="store_true", help="only the overdrive statistics (output *_overdrive.json)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_int(a.model); cfg = d.get("config", {})
    T_us = int(cfg["dt_us"]) * int(d.get("leak_every", 1)); t_end_us = int(cfg["t_end_ms"] * 1000)
    data = EventData(a.dataset, "test")
    if cfg.get("stim_ms") is not None: data.stim_us = int(cfg["stim_ms"] * 1000)
    data.limit_us = cfg.get("in_limit_us"); data.frame_events = cfg.get("frame_events"); data.set_delays(cfg.get("in_delays_us"))
    if a.n_samples and a.n_samples < len(data):      # a fixed random subset (test sets can be ordered by class)
        data = data.subset(np.sort(np.random.default_rng(0).permutation(len(data))[:a.n_samples]))
    print(f"{a.model.parent.name}: step {T_us} us, window {t_end_us / 1000:.0f} ms, {len(data)} test samples", flush=True)
    rows = []; t0 = time.time()
    steps = t_end_us // T_us
    od = overdrive(build(d, 1, 1, dev), data, T_us, steps, a.batch, dev)
    for l, r in enumerate(od):
        print(f"overdrive L{l + 1}: spikes {r['spikes']:8.0f}  v/theta when firing {r['v_over_theta_when_firing']:5.2f}  "
              f"spikes the positive input alone could fire {r['positive_input_spikes']:9.0f}", flush=True)
    if a.overdrive_only:
        out = a.out or (ROOT / "results" / "granularity" / f"{a.model.parent.name}_overdrive.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(model=str(a.model), dataset=a.dataset, step_us=T_us, n_samples=len(data), overdrive=od), open(out, "w"), indent=1)
        print("saved", out); return
    for mode in ("pipelined", "layered"):
        acc, spk = run(build(d, 1, 1, dev), data, T_us, steps, mode, a.batch, dev)
        rows.append(dict(schedule=mode, k=1, refractory="step", acc=acc, spikes=spk))
        print(f"{mode:10s} k  1            acc {acc:.4f}  spikes/layer {[round(s) for s in spk]}  ({time.time() - t0:.0f} s)", flush=True)
    for k in [int(x) for x in a.k.split(",")]:
        if T_us % k: print(f"skip k={k}: {T_us} us does not split"); continue
        for ref, r_every in (("step", k), ("none", 0)):
            acc, spk = run(build(d, k, r_every, dev), data, T_us // k, steps * k, "async", a.batch, dev)
            rows.append(dict(schedule="async", k=k, refractory=ref, acc=acc, spikes=spk))
            print(f"async      k {k:2d} ref {ref:4s}  acc {acc:.4f}  spikes/layer {[round(s) for s in spk]}  ({time.time() - t0:.0f} s)", flush=True)
    out = a.out or (ROOT / "results" / "granularity" / f"{a.model.parent.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(model=str(a.model), dataset=a.dataset, step_us=T_us, n_samples=len(data), overdrive=od, rows=rows), open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
