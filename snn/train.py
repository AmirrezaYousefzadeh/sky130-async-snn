#!/usr/bin/env python3
"""Train the quantization-exact deep SNN in one execution mode (async micro-bin approximation or layered at a bin size).

Example:
  python -m snn.train --dataset nmnist --hidden 512,384,256,192,128 --mode async --dt_us 1000 --t_end_ms 100 \
      --epochs 30 --batch 128 --jitter_us 500 --drop 0.1 --tag j500
  python -m snn.train --dataset nmnist --mode layered --n_eval 10 --dt_us 1000 --t_end_ms 100 --tag bin10ms
Outputs results/models/<name>/{model_int.npz, train.json, best.pt}.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from .data import EventData, Augment, Batcher, train_val_split, ROOT, input_groups
from .model import QSNN, save_int

DEFAULT_T_END_MS = {"nmnist": 100, "ntidigits": 1500}


def hits(o, y):
    """Correct decisions: the true class has the strictly largest score (a tie between the top two is no decision)."""
    top = o.topk(2, dim=1).values
    return (o.argmax(1) == y) & (top[:, 0] > top[:, 1])


@torch.no_grad()
def calibrate_theta_out(model, x, y, t_last, mode, n_eval, verbose=True):
    """Spiking output: pick the output threshold from quantiles of the positive integrator values, by the accuracy of the
    spike-count decision at the end of the stimulus plus its mean over time (one batch of training samples)."""
    model.spiking_out = False
    o, _ = model(x, mode, n_eval)
    pos = o[o > 0].float()
    cands = sorted({max(1, int(torch.quantile(pos[torch.randperm(len(pos), device=pos.device)[:1_000_000]], q).item()))
                    for q in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)}) if len(pos) else [256]
    model.spiking_out = True; best = (-1.0, cands[0])
    te = t_last.clamp(min=n_eval - 1)
    if mode == "pipelined": te = te + len(model.hidden) * n_eval
    te = te.clamp(max=o.shape[1] - 1)
    for th in cands:
        model.theta_out_p.fill_(float(th))
        c, _ = model(x, mode, n_eval)
        end = hits(c[torch.arange(len(y), device=c.device), te], y).float().mean().item()
        over = torch.stack([hits(c[:, t], y) for t in range(n_eval - 1, c.shape[1], n_eval)]).float().mean().item()
        if verbose: print(f"  theta_out {th}: end {end:.3f}, mean over time {over:.3f}, output spikes per sample {c[:, -1].sum(1).mean().item():.1f}", flush=True)
        if end + over > best[0]: best = (end + over, th)
    model.theta_out_p.fill_(float(best[1]))
    return best[1]


@torch.no_grad()
def evaluate(model, batcher, mode, n_eval, fractions=(0.25, 0.5, 0.75, 1.0)):
    model.eval()
    correct = None; n = 0; spk = None; steps = None; correct_end = 0
    for x, y, _, t_last in batcher:
        o_hist, s = model(x, mode, n_eval)
        T = o_hist.shape[1]
        if correct is None:
            steps = [min(T - 1, int(round(f * T)) - 1) for f in fractions]
            correct = np.zeros(len(steps)); spk = np.zeros(len(s))
        for k, t in enumerate(steps):
            correct[k] += hits(o_hist[:, t], y).sum().item()
        # accuracy at the end of each sample's stimulus (last evaluation step at or before its last input)
        te = t_last.clamp(min=n_eval - 1)
        if mode == "pipelined": te = te + len(model.hidden) * n_eval      # evidence reaches the output L steps later
        if mode in ("layered", "pipelined"): te = (te // n_eval) * n_eval + n_eval - 1
        te = te.clamp(max=T - 1)
        o_end = o_hist[torch.arange(len(y), device=o_hist.device), te]
        correct_end += hits(o_end, y).sum().item()
        spk += s.cpu().numpy(); n += len(y)
    n_evals = T if mode == "async" else T // n_eval
    rates = spk / (n * np.array(model.hidden) * max(1, n_evals))
    acc = {str(f): correct[k] / n for k, f in enumerate(fractions)}; acc["end"] = correct_end / n
    return {"acc": acc, "spike_rate_per_eval": rates.tolist(), "spikes_per_sample": (spk / n).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nmnist")
    ap.add_argument("--hidden", default="512,384,256,192,128")
    ap.add_argument("--mode", default="async", choices=["async", "layered", "pipelined"],
                    help="pipelined = time steps with one step of delay per layer hop (neuromorphic hardware)")
    ap.add_argument("--n_eval", type=int, default=1, help="layered mode: bin = n_eval micro-bins")
    ap.add_argument("--dt_us", type=int, default=1000, help="micro-bin = leak period")
    ap.add_argument("--t_end_ms", type=float, default=None)
    ap.add_argument("--stim_ms", type=float, default=None, help="drop input events at or after this time (e.g. 100 = first N-MNIST saccade)")
    ap.add_argument("--in_limit_ms", type=float, default=None, help="input preprocessing: at most one event per channel per window of this length")
    ap.add_argument("--frame_events", type=int, default=None, help="count-based time: every this many input events form one step (use --dt_us 1000)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--wbits", type=int, default=8)
    ap.add_argument("--wbits_first", type=int, default=None, help="weight bits of the first layer (2 = ternary)")
    ap.add_argument("--leak_every", type=int, default=1, help="leak tick = leak_every micro-bins (10 for a 10 ms tick at 1 ms bins)")
    ap.add_argument("--train_theta", action="store_true"); ap.add_argument("--train_leak", action="store_true")
    ap.add_argument("--refractory_every", type=int, default=None, help="async: one spike per neuron per this many micro-bins (default leak_every, 0 = none)")
    ap.add_argument("--snap_at", type=float, default=0.7, help="fraction of the epochs after which thresholds/leaks are frozen at integers")
    ap.add_argument("--theta", type=int, default=256)
    ap.add_argument("--theta_min", type=float, default=1.0, help="floor of the hidden thresholds (128: no single int8 weight fires a neuron)")
    ap.add_argument("--k_leak", type=int, default=4)
    ap.add_argument("--k_out", type=int, default=4)
    ap.add_argument("--vbits", type=int, default=16)
    ap.add_argument("--obits", type=int, default=16)
    ap.add_argument("--sg_scale", type=float, default=5.0)
    ap.add_argument("--weighting", default="ramp", choices=["uniform", "ramp", "late", "early", "early2"])
    ap.add_argument("--spike_penalty", type=float, default=0.0)
    ap.add_argument("--jitter_us", type=float, default=0.0)
    ap.add_argument("--spike_jitter", type=float, default=0.0, help="pipelined training: probability that a hidden spike reaches its targets one step off")
    ap.add_argument("--jitter_mode", default="defer", choices=["defer", "advance", "both"], help="spike jitter: one step late, early (same step) or either")
    ap.add_argument("--warp", type=float, default=0.0)
    ap.add_argument("--shift_us", type=float, default=0.0)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--calib_rate", type=float, default=0.05)
    ap.add_argument("--prune", type=float, default=1.0, help="final weight density of the layers in --prune_layers")
    ap.add_argument("--prune_layers", default="0")
    ap.add_argument("--prune_start", type=float, default=0.2)
    ap.add_argument("--prune_end", type=float, default=0.6)
    ap.add_argument("--core_graph", type=int, default=0, help="TrueNorth-like core graph with cores of this many neurons and axons (0 = off)")
    ap.add_argument("--core_prune", type=int, default=0, help="start fully connected and prune to a core graph with cores of this size (0 = off)")
    ap.add_argument("--prune_epochs", default="3,5,7,9", help="core pruning: epochs at which every neuron's target cores are halved; the last one leaves one")
    ap.add_argument("--in_fanout", type=int, default=2, help="core pruning: cores each input channel keeps")
    ap.add_argument("--in_delays_ms", default="", help="input memory: every input event is repeated after each of these delays (comma list, ms) "
                    "on its own copy of the channels, as an input port with delay lines would do")
    ap.add_argument("--in_fanout_delayed", type=int, default=None, help="core pruning: cores each delayed input channel keeps (default --in_fanout)")
    ap.add_argument("--wscale", type=int, default=1, help="ternary layers (wbits 2): the neuron adds +-wscale")
    ap.add_argument("--lr_ternary", type=float, default=0.01, help="learning rate of the latent weights of ternary layers")
    ap.add_argument("--core_k", type=int, default=2, help="core graph: groups of the previous layer that feed one core")
    ap.add_argument("--groups", type=int, default=1, help="grouped (block-diagonal) connectivity with this many groups")
    ap.add_argument("--group_layers", default="1,2,3", help="weight layers that are grouped (W0 = input -> first hidden layer)")
    ap.add_argument("--spiking_out", action="store_true", help="spiking output neurons with a spike count per class; decision = most spikes")
    ap.add_argument("--theta_out", type=float, default=0.0, help="spiking output threshold (0 = calibrate on a training batch)")
    ap.add_argument("--train_theta_out", action="store_true")
    ap.add_argument("--temp0", type=float, default=None, help="initial loss temperature (default 64 for integrators, 1 for spike counts)")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_from", type=Path, default=None)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_end_ms = a.t_end_ms or DEFAULT_T_END_MS[a.dataset]
    T = int(round(t_end_ms * 1000 / a.dt_us))
    hidden = [int(h) for h in a.hidden.split(",")]
    name = f"{a.dataset}_{a.mode}{'' if a.mode == 'async' else a.n_eval}_dt{a.dt_us}_T{T}_h{'-'.join(map(str, hidden))}_w{a.wbits}_th{a.theta}_s{a.seed}{a.tag}"
    out = a.out or (ROOT / "results" / "models" / name); out.mkdir(parents=True, exist_ok=True)
    print("run", name, "->", out, flush=True)

    full = EventData(a.dataset, "train"); test = EventData(a.dataset, "test")
    delays_us = [int(float(x) * 1000) for x in a.in_delays_ms.split(",") if x]
    if delays_us: full.set_delays(delays_us); test.set_delays(delays_us)
    if a.stim_ms is not None: full.stim_us = test.stim_us = int(a.stim_ms * 1000)
    if a.in_limit_ms: full.limit_us = test.limit_us = int(a.in_limit_ms * 1000)
    if a.frame_events: full.frame_events = test.frame_events = a.frame_events
    train, val = train_val_split(full, a.val_frac, a.seed)
    aug = Augment(a.jitter_us, a.warp, a.shift_us, a.drop, seed=a.seed) if (a.jitter_us or a.warp or a.shift_us or a.drop) else None
    tr_b = Batcher(train, a.dt_us, T, a.batch, True, dev, aug, seed=a.seed, drop_last=True)
    va_b = Batcher(val, a.dt_us, T, 256, False, dev)
    te_b = Batcher(test, a.dt_us, T, 256, False, dev)
    print(f"train {len(train)} val {len(val)} test {len(test)} samples; T={T} micro-bins of {a.dt_us} us; mode {a.mode} n_eval {a.n_eval}", flush=True)

    wbits = [a.wbits_first or a.wbits] + [a.wbits] * len(hidden)
    model = QSNN(full.n_in, hidden, full.n_classes, wbits, a.theta, a.k_leak, a.k_out, a.vbits, a.obits, a.sg_scale,
                 leak_every=a.leak_every, train_theta=a.train_theta, train_leak=a.train_leak, refractory_every=a.refractory_every,
                 wscale=a.wscale, spiking_out=a.spiking_out, theta_out=a.theta_out or 256.0, train_theta_out=a.train_theta_out,
                 spike_jitter=a.spike_jitter, jitter_mode=a.jitter_mode, theta_min=a.theta_min).to(dev)
    if a.core_graph:
        model.set_core_graph(a.core_graph, a.core_k, input_groups(a.dataset, hidden[0] // a.core_graph, a.core_graph))
        print(f"core graph: cores of {a.core_graph}, {hidden[0] // a.core_graph} cores per layer, k = {a.core_k}, offsets {model.core_offsets}, "
              f"density {[round(d, 4) for d in model.density()]}", flush=True)
    if a.groups > 1:
        model.set_groups(a.groups, [int(l) for l in a.group_layers.split(",")])
        print(f"grouped connectivity: {a.groups} groups in weight layers {a.group_layers}, density {[round(d, 3) for d in model.density()]}", flush=True)
    if a.init_from is not None:
        sd = torch.load(a.init_from, map_location=dev)
        from_spiking = "theta_out_p" in sd and a.spiking_out    # a spike-count network keeps its output threshold and loss temperature
        if a.spiking_out and "theta_out_p" not in sd: sd["theta_out_p"] = model.theta_out_p.detach().clone()   # from an integrator network
        model.load_state_dict(sd); print("initialised from", a.init_from)
        if a.theta_min > 1.0:
            with torch.no_grad(): model.theta_p.clamp_(min=a.theta_min)      # start at the floor, so that the gradient reaches it
            print("thresholds raised to the floor:", model.theta, flush=True)
        if not (a.train_theta or a.train_leak): model.snapped = True
        if a.spiking_out and not a.theta_out and not from_spiking:
            x0, y0, _, tl0 = next(iter(Batcher(train, a.dt_us, T, 256, True, dev, seed=a.seed + 1)))
            print(f"calibrated theta_out {calibrate_theta_out(model, x0, y0, tl0, a.mode, a.n_eval)}", flush=True)
    else:
        x0, _, _, _ = next(iter(Batcher(train, a.dt_us, T, 32, True, dev, seed=a.seed + 1)))
        print("calibrating initial weights ...", flush=True)
        model.calibrate(x0, a.mode, a.n_eval, a.calib_rate)
        if a.spiking_out and not a.theta_out:          # spike-count output from the start: threshold from the calibrated integrators
            x0, y0, _, tl0 = next(iter(Batcher(train, a.dt_us, T, 256, True, dev, seed=a.seed + 1)))
            print(f"calibrated theta_out {calibrate_theta_out(model, x0, y0, tl0, a.mode, a.n_eval)}", flush=True)
    if a.temp0 or (a.spiking_out and not (a.init_from is not None and from_spiking)):
        with torch.no_grad(): model.log_temp.fill_(math.log(a.temp0 or 1.0))
    tern = [w for w, b in zip(model.W, wbits) if b == 2]; wide = [w for w, b in zip(model.W, wbits) if b != 2]
    groups = [{"params": [model.log_temp], "lr": 1e-2}]
    if wide: groups.append({"params": wide, "lr": a.lr})
    if tern: groups.append({"params": tern, "lr": a.lr_ternary})
    if a.train_theta: groups.append({"params": [model.theta_p], "lr": a.lr})
    if a.train_leak: groups.append({"params": [model.k_p], "lr": 0.02})
    if a.spiking_out and a.train_theta_out: groups.append({"params": [model.theta_out_p], "lr": a.lr})
    opt = torch.optim.Adam(groups)
    steps_total = a.epochs * len(tr_b)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(st, steps_total) / steps_total)))
    prune_eps = [int(e) for e in a.prune_epochs.split(",")] if a.core_prune else []
    prune_layers = [int(l) for l in a.prune_layers.split(",")] if a.prune < 1.0 else []
    n_neurons, hist, best, step = sum(hidden), [], (-1.0, None), 0
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    cfg.update(name=name, T=T, t_end_ms=t_end_ms, n_in=full.n_in, n_classes=full.n_classes,
               in_limit_us=int(a.in_limit_ms * 1000) if a.in_limit_ms else None, in_delays_us=delays_us)
    t0 = time.time()
    for ep in range(a.epochs):
        if ep in prune_eps:
            i = prune_eps.index(ep); G = hidden[0] // a.core_prune
            last = i == len(prune_eps) - 1
            m_h = 1 if last else max(1, G >> (i + 1))
            fo = np.full(full.n_in, a.in_fanout); fo[full.n_in0:] = a.in_fanout if a.in_fanout_delayed is None else a.in_fanout_delayed
            m_i = fo if last else np.maximum(fo, G >> (i + 1))
            tp = time.time(); model.prune_to_cores(a.core_prune, m_h, m_i)
            print(f"core pruning at epoch {ep}: every neuron keeps {m_h} of {G} cores, every input channel {sorted(set(m_i.tolist()))}; "
                  f"density {[round(dd, 4) for dd in model.density()]} ({time.time() - tp:.0f} s)", flush=True)
            if last:     # every core must end with at most core_prune axons
                fan = [int((mk.reshape(mk.shape[0], G, a.core_prune) > 0).any(2).sum(0).max()) for mk in model.masks()[:-1]]
                assert max(fan) <= a.core_prune, f"cores would need {fan} axons; widen the layers or lower --in_fanout"
                print(f"axons used per core, per layer: {fan}", flush=True)
        if not model.snapped and ep >= int(a.snap_at * a.epochs):
            th, kk = model.snap(); print(f"snapped thresholds {th} and leak shifts {kk} at epoch {ep}", flush=True)
        model.train(); run_loss = 0.0; nb = 0
        for x, y, _, t_last in tr_b:
            o_hist, spk = model(x, a.mode, a.n_eval)
            loss = model.loss(o_hist, y, a.mode, a.n_eval, a.weighting, spk, a.spike_penalty, n_neurons, T, t_last)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.W, 1e3)
            opt.step(); sched.step(); step += 1
            run_loss += loss.item(); nb += 1
            if prune_layers:
                s0, s1 = int(a.prune_start * steps_total), int(a.prune_end * steps_total)
                if s0 <= step <= s1 and step % 50 == 0:
                    frac = (step - s0) / max(1, s1 - s0)
                    model.prune_to(1.0 - (1.0 - a.prune) * (1.0 - (1.0 - frac) ** 3), prune_layers)
            if nb % a.log_every == 0:
                rates = (spk / (x.shape[0] * torch.tensor(hidden, device=dev, dtype=torch.float) * (T if a.mode == "async" else T // a.n_eval))).tolist()
                print(f"  ep {ep} batch {nb}/{len(tr_b)} loss {run_loss/nb:.4f} rates {['%.3f' % r for r in rates]} temp {model.log_temp.exp().item():.1f} ({time.time()-t0:.0f} s)", flush=True)
        va = evaluate(model, va_b, a.mode, a.n_eval)
        rec = {"epoch": ep + 1, "loss": run_loss / max(1, nb), "val": va, "density": model.density(), "seconds": time.time() - t0,
               "theta": [round(float(t), 2) for t in model.theta_p.tolist()], "k_leak": [round(float(k), 3) for k in model.k_p.tolist()]}
        hist.append(rec)
        print(f"epoch {ep+1}: loss {rec['loss']:.4f} val acc {va['acc']} rates {['%.3f' % r for r in va['spike_rate_per_eval']]} ({rec['seconds']:.0f} s)", flush=True)
        eligible = (not prune_layers or step > int(a.prune_end * steps_total)) and model.snapped and (not prune_eps or ep >= prune_eps[-1])
        if eligible and va["acc"]["end"] >= best[0]:
            best = (va["acc"]["end"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            torch.save(best[1], out / "best.pt")
            save_int(out / "model_int.npz", model.export_int(), extra={"config": cfg})   # usable even if the job is cut short
    if not model.snapped: model.snap()
    model.load_state_dict(best[1])
    te = evaluate(model, te_b, a.mode, a.n_eval); va = evaluate(model, va_b, a.mode, a.n_eval)
    print(f"BEST val acc(end) {va['acc']['end']:.4f} | TEST acc {te['acc']} spikes/sample {['%.0f' % s for s in te['spikes_per_sample']]}", flush=True)
    d = model.export_int()
    save_int(out / "model_int.npz", d, extra={"config": cfg})
    json.dump({"config": cfg, "history": hist, "best_val": va, "test": te, "density": model.density()}, open(out / "train.json", "w"), indent=1)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
