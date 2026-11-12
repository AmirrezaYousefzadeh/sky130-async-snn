#!/usr/bin/env python3
"""Quantization-exact deep feed-forward spiking network with the execution modes of the time-multiplexed core.

Integer semantics (all state integer-valued; weights int(wbits_l); membranes saturating int(vbits); outputs int(obits)):
  hidden layer l (size N_l), presynaptic spike/event i:   v_l[j] += W_l[i][j]            (saturating)
  threshold:   s_j = (v_l[j] >= THETA_l) ; reset to 0 on spike ; in async mode at most one spike per neuron per leak tick
  leak tick (every leak_every micro-bins = the step, 10 ms):  v_l -= v_l >> K_l ;  o -= o >> K_OUT ; refractory cleared
  output layer (non-spiking integrators):                   o[c] += W_out[j][c] for every spike j of the last hidden layer
  decision: argmax(o) (with a margin, evaluated by snn.intref / snn.eval)
  spiking output (spiking_out): the same integrators get a threshold THETA_OUT and a hard reset, like the hidden
  neurons, and a spike counter per class; the decision is the class with the most spikes (none while the top two tie),
  and o_hist holds the running counts

Execution modes, one micro-bin (dt) per loop step t with x[:, t] = event counts per input channel:
  async   : add inputs, evaluate thresholds and propagate through all layers immediately (depth first); a neuron that
            fired stays silent until the next leak tick (the core-wide refractory of the tick). This is the micro-bin
            approximation of per-event hardware execution (the bit-exact per-event model is snn.intref).
  layered : add inputs to layer 1 every micro-bin; at every n_eval-th micro-bin (the step) leak, then evaluate the
            thresholds layer by layer (the cascade), as a stepped core does at its step boundary.
  pipelined: as layered, but every deeper layer integrates the spikes of the previous evaluation (one step per hop).
            Spike jitter (training only, spike_jitter > 0): every hidden spike reaches its targets one evaluation late
            (jitter_mode 'defer'), in the same evaluation ('advance') or either ('both', half each) with that
            probability, so that the network does not depend on which step a spike falls in.
Thresholds THETA_l and leak shifts K_l may be trained (float during training, snapped to integers by snap()); the
exported integer network is what the RTL hardwires.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, g): return g


class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.floor(x)
    @staticmethod
    def backward(ctx, g): return g


class SpikeFn(torch.autograd.Function):
    """Heaviside with a fast-sigmoid surrogate on x = (v - theta) / theta."""
    scale = 5.0
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (SpikeFn.scale * x.abs() + 1.0) ** 2


def leak(x, k):
    """x - floor(x / 2^k): exactly x - (x >>> k) for integer x and integer k (arithmetic shift floors).
    k may be a float tensor during training (continuous leak); the gradient reaches k through 2^k."""
    return x - FloorSTE.apply(x / torch.pow(torch.tensor(2.0, device=x.device), k) if torch.is_tensor(k) else x / float(2 ** k))


class QSNN(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], n_out: int, wbits=8, theta=256, k_leak=4,
                 k_out: int = 4, vbits: int = 16, obits: int = 16, sg_scale: float = 5.0, temp0: float = 64.0,
                 w_init: float = 1.0, leak_every: int = 1, train_theta: bool = False, train_leak: bool = False,
                 refractory_every: int | None = None, wscale: int = 1, spiking_out: bool = False, theta_out: float = 256.0,
                 train_theta_out: bool = False, spike_jitter: float = 0.0, jitter_mode: str = "defer", theta_min: float = 1.0):
        super().__init__()
        self.theta_min = float(theta_min)          # floor of the hidden thresholds (e.g. 128: no single int8 weight fires a neuron)
        assert jitter_mode in ("defer", "advance", "both")
        self.spike_jitter, self.jitter_mode = float(spike_jitter), jitter_mode
        self.n_in, self.hidden, self.n_out = n_in, list(hidden), n_out
        L = len(hidden)
        self.wbits = [int(wbits)] * (L + 1) if isinstance(wbits, int) else [int(b) for b in wbits]   # per weight layer (L + 1)
        assert len(self.wbits) == L + 1
        # ternary layers (wbits 2) store -1 / 0 / +1; the neuron adds +-wscale, so that the shift leak keeps resolution
        self.wscale_value = int(wscale)
        self.wscale = [int(wscale) if b == 2 else 1 for b in self.wbits]
        self.k_out, self.vbits, self.obits, self.leak_every = k_out, vbits, obits, int(leak_every)
        # async mode: a neuron fires at most once per refractory period (micro-bins); None = the leak tick, 0 = no refractory
        self.refractory_every = int(leak_every) if refractory_every is None else int(refractory_every)
        theta = [float(theta)] * L if isinstance(theta, (int, float)) else [float(t) for t in theta]
        k_leak = [float(k_leak)] * L if isinstance(k_leak, (int, float)) else [float(k) for k in k_leak]
        self.theta_p = nn.Parameter(torch.tensor(theta), requires_grad=train_theta)
        self.k_p = nn.Parameter(torch.tensor(k_leak), requires_grad=train_leak)
        self.snapped = not (train_theta or train_leak)
        SpikeFn.scale = sg_scale
        self.vmax, self.omax = 2 ** (vbits - 1) - 1, 2 ** (obits - 1) - 1
        sizes = [n_in] + self.hidden + [n_out]
        self.W = nn.ParameterList([nn.Parameter(torch.randn(sizes[l], sizes[l + 1]) * w_init) for l in range(L + 1)])
        for l in range(L + 1):
            self.register_buffer(f"mask{l}", torch.ones(sizes[l], sizes[l + 1]))
        self.log_temp = nn.Parameter(torch.tensor(math.log(temp0)))   # loss temperature only (argmax is scale free)
        self.spiking_out = bool(spiking_out)
        self.theta_out_p = nn.Parameter(torch.tensor(float(theta_out)), requires_grad=bool(spiking_out and train_theta_out))

    # ---------------------------------------------------------------------------------------- parameters
    def theta_l(self, l):
        """Threshold of layer l: rounded (STE) to an integer >= 1 so that training sees the integer the hardware uses."""
        return RoundSTE.apply(self.theta_p[l].clamp(min=self.theta_min))

    def theta_o(self):
        """Threshold of the spiking output neurons, an integer >= 1 (STE)."""
        return RoundSTE.apply(self.theta_out_p.clamp(min=1.0))

    def k_l(self, l):
        """Leak shift of layer l: continuous while training the leak, integer after snap()."""
        return self.k_p[l].clamp(1.0, 10.0) if not self.snapped else int(round(float(self.k_p[l])))

    @torch.no_grad()
    def snap(self):
        """Freeze thresholds and leak shifts at their integers (end of the continuous phase)."""
        self.theta_p.copy_(self.theta_p.clamp(min=self.theta_min).round()); self.k_p.copy_(self.k_p.clamp(1.0, 10.0).round())
        self.theta_p.requires_grad_(False); self.k_p.requires_grad_(False); self.snapped = True
        return [int(t) for t in self.theta_p.tolist()], [int(k) for k in self.k_p.tolist()]

    @property
    def theta(self): return [int(round(float(t))) for t in self.theta_p.detach().clamp(min=self.theta_min)]

    @property
    def k_leak(self): return [int(round(float(k))) for k in self.k_p.detach().clamp(1.0, 10.0)]

    # ---------------------------------------------------------------------------------------- weights
    def masks(self): return [getattr(self, f"mask{l}") for l in range(len(self.W))]

    @torch.no_grad()
    def set_core_graph(self, core: int = 256, k: int = 2, in_groups=None):
        """TrueNorth-like core graph: hidden layers of equal width made of cores of `core` neurons with `core` axons each.
        Hidden -> hidden: every neuron feeds exactly one core of the next layer; a core's axons come from k groups (cores)
        of the previous layer, core / k neurons from each, with offsets that change from layer to layer so that groups
        mix (k = 2 and 8 groups: offsets {0,1}, {0,2}, {0,4}, ...). Input -> first hidden layer: in_groups lists the input
        channels of each input group (patch or band, at most `core` channels); input group p feeds cores p, p + P, ...
        Last hidden layer -> output: dense; in hardware one class-integrator core per group, partial scores added."""
        L = len(self.hidden); n = self.hidden[0]
        assert all(h == n for h in self.hidden) and n % core == 0, "core graph needs equal hidden widths, multiples of core"
        G = n // core
        assert core % k == 0 and 1 <= k <= G
        dev = self.mask0.device
        groups = in_groups if in_groups is not None else [np.arange(self.n_in)]
        m = torch.zeros(self.n_in, n, device=dev)
        for c in range(G):
            ch = torch.as_tensor(np.asarray(groups[c % len(groups)]), device=dev, dtype=torch.long)
            assert len(ch) <= core, f"input group of {len(ch)} channels exceeds {core} axons"
            m[ch, c * core:(c + 1) * core] = 1.0
        self.mask0.copy_(m)
        part = core // k
        n_mix = max(1, math.ceil(math.log(G, k) - 1e-9)) if G > 1 else 1
        self.core_offsets = []
        for l in range(1, L):
            st = (k ** ((l - 1) % n_mix)) % G or 1
            offs = [(j * st) % G for j in range(k)]
            if len(set(offs)) < k:
                st = max(1, G // k); offs = [(j * st) % G for j in range(k)]
            m = torch.zeros(n, n, device=dev)
            for g in range(G):
                for j in range(k):
                    tgt = (g + offs[j]) % G
                    m[g * core + j * part: g * core + (j + 1) * part, tgt * core:(tgt + 1) * core] = 1.0
            getattr(self, f"mask{l}").copy_(m)
            self.core_offsets.append(offs)
        self.core, self.core_k = int(core), int(k)

    @torch.no_grad()
    def set_groups(self, G: int, layers):
        """Grouped (block-diagonal) connectivity, like a grouped convolution: in weight layer l (layer l-1 -> layer l,
        W[0] = input -> first hidden layer) the neurons of layer l are split into G contiguous groups and group g only
        receives from group g of layer l-1. Each group can then be one core whose work depends on its own input group."""
        sizes = [self.n_in] + self.hidden + [self.n_out]
        for l in layers:
            a, b = sizes[l], sizes[l + 1]
            assert a % G == 0 and b % G == 0, f"weight layer {l} ({a} x {b}) not divisible into {G} groups"
            m = getattr(self, f"mask{l}")
            gi = torch.arange(a, device=m.device) // (a // G); go = torch.arange(b, device=m.device) // (b // G)
            m.copy_((gi[:, None] == go[None, :]).to(m.dtype))
        self.groups, self.group_layers = int(G), [int(l) for l in layers]

    def qweights(self):
        return [RoundSTE.apply(w.clamp(-(2 ** (b - 1) - 1), 2 ** (b - 1) - 1)) * (sc * m) for w, b, m, sc in zip(self.W, self.wbits, self.masks(), self.wscale)]

    @torch.no_grad()
    def prune_to_cores(self, core: int, m_hidden: int, m_in: int):
        """Structured pruning to a core graph. The hidden layers (equal widths, multiples of `core`) are split into G cores
        of `core` neurons. In every weight layer feeding a hidden layer, each presynaptic unit keeps its connections to
        m of the G cores (m = m_in for input channels, m_hidden for neurons) and loses all others; every core receives the
        same number of presynaptic units (core * m_hidden for hidden layers, n_in * m_in / G for the input; at most
        ceil(n_in / G) * m_in when the channels do not split evenly, e.g. SSC's 700 over 16 cores). Units keep
        their strongest cores (score = summed |weight| into the core), chosen among the cores still connected, by m
        successive optimal assignments. With m_hidden = 1 every neuron feeds exactly one core, which has `core` axons.
        Input layers of at most `core` channels stay dense (every core can take all channels). m_in may be an array with
        one fan-out per input channel (e.g. fewer cores for the delayed copies of the input); the channels of each
        fan-out value are then assigned as a group, balanced over the cores."""
        from scipy.optimize import linear_sum_assignment
        L = len(self.hidden); n = self.hidden[0]
        assert all(h == n for h in self.hidden) and n % core == 0
        G = n // core
        for l in range(L):
            n_pre = self.W[l].shape[0]
            if l == 0 and n_pre <= core:
                continue
            m = m_in if l == 0 else m_hidden
            mv = np.full(n_pre, int(m)) if np.isscalar(m) else np.asarray(m, np.int64)     # fan-out per presynaptic unit
            assert mv.shape == (n_pre,)
            mask = self.masks()[l]
            Wa = (self.W[l].detach().abs() * mask).cpu().numpy().reshape(n_pre, G, core)
            S = Wa.sum(2).astype(np.float64)
            avail = (mask.cpu().numpy().reshape(n_pre, G, core) > 0).any(2)
            if np.all(avail.sum(1) <= mv):   # nothing left to prune (e.g. SSC's input stays at 5 cores from the second stage on);
                continue                     # the earlier stage already balanced the cores, and m rounds over exactly m cores can fail
            keep = np.zeros((n_pre, G), bool)
            for v in np.unique(mv):                                   # every fan-out group balanced over the cores on its own,
                act = np.flatnonzero(mv == v)                         # so that each round stays feasible for the next
                if np.all(avail[act].sum(1) <= v):                    # this group is already at its fan-out
                    keep[act] = avail[act]; continue
                cap = -(-len(act) // G)                               # units of the group per core in each round (rounded up)
                for r in range(int(v)):
                    cost = np.where(avail[act] & ~keep[act], -S[act], 1e12)
                    rows, cols = linear_sum_assignment(np.repeat(cost, cap, axis=1))
                    tgt = cols // cap
                    assert np.all((avail & ~keep)[act[rows], tgt]), f"layer {l}: no balanced assignment within the remaining connections"
                    keep[act[rows], tgt] = True
            new = torch.from_numpy(np.repeat(keep, core, axis=1).astype(np.float32)).to(mask.device)
            mask.copy_(new)
        self.core, self.core_m = int(core), (int(m_in) if np.isscalar(m_in) else [int(x) for x in np.asarray(m_in)], int(m_hidden))

    @torch.no_grad()
    def prune_to(self, density: float, layers=None):
        """Keep the `density` fraction of largest-magnitude weights per layer (global within the layer)."""
        layers = range(len(self.W)) if layers is None else layers
        for l in layers:
            w, m = self.W[l], self.masks()[l]
            a = (w.abs() * m).flatten()
            k = int(round(density * a.numel()))
            thr = torch.topk(a, k).values.min() if k > 0 else a.max() + 1
            m.copy_(((w.abs() >= thr) & (m > 0)).float()); w.mul_(m)

    def density(self):
        return [float(((self.qweights()[l] != 0).float()).mean()) for l in range(len(self.W))]

    # ---------------------------------------------------------------------------------------- forward
    def forward(self, x, mode: str = "async", n_eval: int = 1, record: bool = False):
        """x: (B, T, n_in) event counts per micro-bin. Returns o_hist (B, T, n_out) integer-valued, spike counts per layer (L,).
        Leak ticks every self.leak_every micro-bins; in layered mode n_eval must be a multiple of leak_every (the step).
        record=True also returns rec: per micro-bin a list of the L spike tensors (B, N_l) (zeros where no evaluation)."""
        Wq = self.qweights()
        B, T, _ = x.shape
        L = len(self.hidden)
        v = [x.new_zeros(B, h) for h in self.hidden]
        fired = [x.new_zeros(B, h) for h in self.hidden]
        o = x.new_zeros(B, self.n_out)
        spk = [x.new_zeros(()) for _ in range(L)]
        o_hist = []
        thetas = [self.theta_l(l) for l in range(L)]
        ks = [self.k_l(l) for l in range(L)]
        rec = [] if record else None
        prev = [x.new_zeros(B, h) for h in self.hidden]          # pipelined mode: spikes of the previous evaluation
        late = [x.new_zeros(B, h) for h in self.hidden]          # spike jitter: spikes due one evaluation after prev
        p_jit = self.spike_jitter if (self.training and mode == "pipelined") else 0.0
        p_late = p_jit if self.jitter_mode == "defer" else (0.0 if self.jitter_mode == "advance" else 0.5 * p_jit)
        p_adv = p_jit - p_late
        cnt = x.new_zeros(B, self.n_out); tho = self.theta_o() if self.spiking_out else None

        def fire_out(o, cnt):                                    # spiking output: threshold, hard reset, count
            if not self.spiking_out: return o, cnt
            so = SpikeFn.apply((o - tho) / tho)
            return o * (1 - so), cnt + so
        for t in range(T):
            inp = x[:, t]
            st = [x.new_zeros(B, h) for h in self.hidden] if record else None
            tick = (t % self.leak_every == self.leak_every - 1)
            rtick = self.refractory_every > 0 and (t % self.refractory_every == self.refractory_every - 1)
            if mode == "async":
                for l in range(L):
                    v[l] = (v[l] + inp @ Wq[l]).clamp(-self.vmax - 1, self.vmax)
                    s = SpikeFn.apply((v[l] - thetas[l]) / thetas[l])
                    if self.refractory_every > 0: s = s * (1 - fired[l]); fired[l] = (fired[l] + s).clamp(max=1.0)
                    v[l] = v[l] * (1 - s)
                    spk[l] = spk[l] + s.sum()
                    if record: st[l] = s
                    inp = s
                o = (o + inp @ Wq[L]).clamp(-self.omax - 1, self.omax)
                o, cnt = fire_out(o, cnt)
                if rtick:
                    for l in range(L): fired[l] = torch.zeros_like(fired[l])
                if tick:
                    for l in range(L): v[l] = leak(v[l], ks[l])
                    o = leak(o, self.k_out)
            elif mode == "layered":
                v[0] = (v[0] + inp @ Wq[0]).clamp(-self.vmax - 1, self.vmax)
                if tick:
                    for l in range(L): v[l] = leak(v[l], ks[l])
                    o = leak(o, self.k_out)
                if t % n_eval == n_eval - 1:
                    for l in range(L):
                        if l > 0: v[l] = (v[l] + inp @ Wq[l]).clamp(-self.vmax - 1, self.vmax)
                        s = SpikeFn.apply((v[l] - thetas[l]) / thetas[l])
                        v[l] = v[l] * (1 - s)
                        spk[l] = spk[l] + s.sum()
                        if record: st[l] = s
                        inp = s
                    o = (o + inp @ Wq[L]).clamp(-self.omax - 1, self.omax)
                    o, cnt = fire_out(o, cnt)
            elif mode == "pipelined":
                # neuromorphic time steps: one step of delay per layer hop. Layer 0 integrates the input events of the
                # step; every deeper layer and the output integrate the spikes their source layer emitted at the
                # previous evaluation (after the leak), then all thresholds are evaluated.
                v[0] = (v[0] + inp @ Wq[0]).clamp(-self.vmax - 1, self.vmax)
                if tick:
                    for l in range(L): v[l] = leak(v[l], ks[l])
                    o = leak(o, self.k_out)
                if t % n_eval == n_eval - 1:
                    new, new_late, adv = [], [], None
                    for l in range(L):
                        if l > 0:
                            src = prev[l - 1] if adv is None else prev[l - 1] + adv
                            v[l] = (v[l] + src @ Wq[l]).clamp(-self.vmax - 1, self.vmax)
                        s = SpikeFn.apply((v[l] - thetas[l]) / thetas[l])
                        v[l] = v[l] * (1 - s)
                        spk[l] = spk[l] + s.sum()
                        if record: st[l] = s
                        if p_jit:                   # split the spikes: this evaluation / the next (on time) / the one after
                            u = torch.rand_like(s)
                            m_late = (u < p_late).to(s.dtype); m_adv = ((u >= p_late) & (u < p_late + p_adv)).to(s.dtype)
                            adv = s * m_adv if p_adv else None
                            new.append(s * (1 - m_late - m_adv)); new_late.append(s * m_late)
                        else:
                            new.append(s)
                    src = prev[L - 1] if adv is None else prev[L - 1] + adv
                    o = (o + src @ Wq[L]).clamp(-self.omax - 1, self.omax)
                    o, cnt = fire_out(o, cnt)
                    if p_jit:
                        prev = [n_ + d_ for n_, d_ in zip(new, late)]; late = new_late
                    else:
                        prev = new
            else:
                raise ValueError(mode)
            o_hist.append(cnt if self.spiking_out else o)
            if record: rec.append(st)
        if record:
            return torch.stack(o_hist, 1), torch.stack(spk), rec
        return torch.stack(o_hist, 1), torch.stack(spk)

    # ---------------------------------------------------------------------------------------- loss
    def loss(self, o_hist, y, mode: str, n_eval: int, weighting: str = "ramp", spike_counts=None, spike_penalty: float = 0.0,
             n_neurons: int = 1, n_steps: int = 1, t_last=None):
        """Anytime cross-entropy over the micro-steps at which the output can change, weighted in time and restricted
        to the stimulus (steps <= t_last[b]). weighting: uniform | ramp | late | early | early2."""
        B, T, C = o_hist.shape
        steps = torch.arange(T, device=o_hist.device)
        valid = torch.ones(T, device=o_hist.device) if mode == "async" else ((steps % n_eval) == n_eval - 1).float()
        if t_last is None:
            t_last = torch.full((B,), T - 1, device=o_hist.device)
        if mode == "pipelined":       # the evidence of the last input step reaches the output L evaluations later
            t_last = (t_last + len(self.hidden) * n_eval).clamp(max=T - 1)
        tl = t_last.clamp(min=n_eval - 1)[:, None].float()
        on = (steps[None].float() <= tl).float() * valid[None]                           # (B, T)
        if weighting == "uniform": w = on
        elif weighting == "ramp": w = on * (steps[None] + 1).float()
        elif weighting == "late": w = on * (steps[None].float() >= 0.75 * tl).float()
        elif weighting == "early": w = on * (tl - steps[None].float() + 1).clamp(min=1.0)
        elif weighting == "early2": w = on * ((tl - steps[None].float() + 1).clamp(min=1.0) ** 2)
        else: raise ValueError(weighting)
        w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
        logits = o_hist / self.log_temp.exp()
        ce = F.cross_entropy(logits.reshape(B * T, C), y.repeat_interleave(T), reduction="none").reshape(B, T)
        loss = (ce * w).sum(1).mean()
        if spike_penalty > 0 and spike_counts is not None:
            loss = loss + spike_penalty * spike_counts.sum() / (B * n_neurons * n_steps)
        return loss

    # ---------------------------------------------------------------------------------------- calibration
    @torch.no_grad()
    def calibrate(self, x, mode: str, n_eval: int, target_rate: float = 0.05, iters: int = 8, verbose=True):
        """Scale the weights of each layer in turn so that its mean spike rate per neuron per evaluation is about
        target_rate (log-space bisection). Keeps deep networks from going silent or saturating at initialization."""
        L = len(self.hidden)
        n_evals = x.shape[1] if mode == "async" else x.shape[1] // n_eval
        for l in range(L):
            lo, hi = math.log(1e-3), math.log(1e3)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                w_backup = self.W[l].clone()
                self.W[l].mul_(math.exp(mid))
                _, spk = self.forward(x, mode, n_eval)
                rate = float(spk[l]) / (x.shape[0] * self.hidden[l] * max(1, n_evals))
                self.W[l].copy_(w_backup)
                if rate < target_rate: lo = mid
                else: hi = mid
            self.W[l].mul_(math.exp(0.5 * (lo + hi)))
            if verbose:
                _, spk = self.forward(x, mode, n_eval)
                print(f"  calibrated layer {l}: scale {math.exp(0.5*(lo+hi)):.3g}, rate {float(spk[l])/(x.shape[0]*self.hidden[l]*max(1,n_evals)):.4f}", flush=True)

    # ---------------------------------------------------------------------------------------- export
    @torch.no_grad()
    def export_int(self):
        Wq = [w.detach().cpu().numpy().astype(np.int32) for w in self.qweights()]
        return dict(W=Wq, n_in=self.n_in, hidden=self.hidden, n_out=self.n_out, wbits=self.wbits, theta=self.theta,
                    k_leak=self.k_leak, k_out=self.k_out, vbits=self.vbits, obits=self.obits, leak_every=self.leak_every,
                    refractory_every=self.refractory_every, wscale=self.wscale_value, spiking_out=self.spiking_out,
                    theta_out=int(round(float(self.theta_out_p.detach().clamp(min=1.0)))) if self.spiking_out else 0)


def save_int(path, d, extra=None):
    arrs = {f"W{l}": w.astype(np.int8 if max(d["wbits"]) <= 8 else np.int16) for l, w in enumerate(d["W"])}
    meta = {k: v for k, v in d.items() if k != "W"}
    if extra: meta.update(extra)
    import json
    np.savez(path, meta=json.dumps(meta), **arrs)


def load_int(path):
    import json
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    L = len(meta["hidden"]) + 1
    meta["W"] = [z[f"W{l}"].astype(np.int32) for l in range(L)]
    if isinstance(meta.get("k_leak"), int): meta["k_leak"] = [meta["k_leak"]] * (L - 1)        # older exports
    if isinstance(meta.get("wbits"), int): meta["wbits"] = [meta["wbits"]] * L
    return meta
