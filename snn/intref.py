#!/usr/bin/env python3
"""Bit-exact, cycle-counting reference of the hardwired core (the behaviour the RTL must reproduce).

Two execution modes on the same integer network (see snn.model for the arithmetic):

run_async   per-event execution without layer barriers. Every input event is added to layer 0 on arrival and the
            thresholds of that layer are evaluated immediately; new spikes are serialized in index order into the next
            layer's FIFO, and so on down to the output integrators. A neuron fires at most once per refractory window
            (= the leak period, `fired` bits cleared at every leak tick). Leak ticks every dt_leak_us.
run_layered layer-synchronized execution: inputs are integrated on arrival, leak ticks every dt_leak_us, and only at every
            n_eval-th tick (the bin) are the thresholds evaluated layer by layer (leak first, then the cascade).
pipelined   neuromorphic time steps (SpiNNaker, TrueNorth, Loihi): as layered, but a spike emitted at one evaluation is
            integrated by the next layer at the next evaluation, one step of delay per layer hop.

Cycle model (what the RTL implements): clock period tclk_ns; an input event occupies the input port for one cycle (events
closer than one cycle are serialized); a layer services one incoming spike per cycle, evaluates its thresholds in the
same cycle, and serializes its k new spikes one per cycle into the next layer, which receives them `pipe` cycles later;
a layer accepts the next input only after its spike register is empty (busy = service cycle + max(1, k)). The output
integrators add one spike per cycle. Every change of the output vector is recorded with its cycle number; the physical
time of a record is cycle * tclk_ns.

Weights are passed packed (see pack_weights) because numba handles flat arrays best.
"""
from __future__ import annotations
import numpy as np
from numba import njit


def pack_weights(W_list):
    """W_list: [W0 (n_in x N0), W1 (N0 x N1), ..., WL (N_{L-1} x n_out)] int arrays -> flat int32 + offsets."""
    offs = np.zeros(len(W_list) + 1, np.int64)
    rows = np.zeros(len(W_list), np.int64); cols = np.zeros(len(W_list), np.int64)
    for l, W in enumerate(W_list):
        rows[l], cols[l] = W.shape
        offs[l + 1] = offs[l] + W.size
    flat = np.concatenate([W.astype(np.int32).ravel() for W in W_list])
    return flat, offs, rows, cols


@njit(cache=True)
def _ceil_div(a, b):
    return -((-a) // b)


@njit(cache=True)
def _add_row(vflat, voff, n, wflat, woff, ncol, i, vmax):
    """v[layer] += W[layer][i, :] with saturation."""
    base = woff + i * ncol
    for j in range(n):
        s = vflat[voff + j] + wflat[base + j]
        if s > vmax: s = vmax
        elif s < -vmax - 1: s = -vmax - 1
        vflat[voff + j] = s


@njit(cache=True)
def _leak(vflat, voff, n, k):
    for j in range(n):
        vflat[voff + j] = vflat[voff + j] - (vflat[voff + j] >> k)


@njit(cache=True)
def _fire(vflat, voff, n, fired, foff, theta, use_refractory, qidx, qarr, qoff, qtail, arrival):
    """Threshold evaluation of one layer; spikes enqueued in index order with arrivals arrival, arrival+1, ...
    Returns the number of spikes."""
    k = 0
    for j in range(n):
        if vflat[voff + j] >= theta and (not use_refractory or fired[foff + j] == 0):
            vflat[voff + j] = 0
            if use_refractory: fired[foff + j] = 1
            qidx[qoff + qtail + k] = j
            qarr[qoff + qtail + k] = arrival + k
            k += 1
    return k


@njit(cache=True)
def _run(t_us, ch, wflat, woff, wrows, wcols, theta, k_leak, k_out, vmax, omax, dt_leak_us, t_end_us, tclk_ns, pipe,
         mode_id, n_eval, rec_cycle, rec_o, rec_spk, spikes, qmax, dt_ref_us):
    # mode_id: 0 = per event (async), 1 = layered (cascade inside the step), 2 = pipelined (one step of delay per hop)
    L = wrows.shape[0] - 1                      # hidden layers
    n_out = wcols[L]
    # membranes, fired bits and queues packed
    voff = np.zeros(L + 1, np.int64)
    for l in range(L): voff[l + 1] = voff[l] + wcols[l]
    vflat = np.zeros(voff[L], np.int64)
    fired = np.zeros(voff[L], np.uint8)
    # queue l (l = 1..L) holds spikes from layer l-1, capacity wcols[l-1]
    qoff = np.zeros(L + 2, np.int64)
    for l in range(1, L + 1): qoff[l + 1] = qoff[l] + wcols[l - 1]
    qidx = np.zeros(qoff[L + 1], np.int64); qarr = np.zeros(qoff[L + 1], np.int64)
    qhead = np.zeros(L + 1, np.int64); qtail = np.zeros(L + 1, np.int64)
    o = np.zeros(n_out, np.int64)
    busy = np.zeros(L + 1, np.int64)
    nrec = 0
    nrec_max = rec_cycle.shape[0]
    last_in = -1
    next_tick = dt_leak_us
    tick_no = 0
    use_ref = dt_ref_us > 0
    next_ref = dt_ref_us if use_ref else t_end_us + 1
    n_ev = t_us.shape[0]
    e = 0
    done = False
    while not done:
        # ---------------- events before the next tick
        while e < n_ev and t_us[e] < next_tick and t_us[e] < t_end_us:
            while use_ref and t_us[e] >= next_ref:
                for j in range(voff[L]): fired[j] = 0
                next_ref += dt_ref_us
            c_in = _ceil_div(t_us[e] * 1000, tclk_ns)
            if c_in <= last_in: c_in = last_in + 1
            last_in = c_in
            s = c_in if c_in > busy[0] else busy[0]
            _add_row(vflat, voff[0], wcols[0], wflat, woff[0], wcols[0], ch[e], vmax)
            if mode_id == 0:
                k = _fire(vflat, voff[0], wcols[0], fired, voff[0], theta[0], use_ref, qidx, qarr, qoff[1], qtail[1], s + pipe)
                qtail[1] += k; spikes[0] += k
                busy[0] = s + (k if k > 1 else 1)
                # cascade through the queues
                for l in range(1, L + 1):
                    if qtail[l] - qhead[l] > qmax[l]: qmax[l] = qtail[l] - qhead[l]
                    while qhead[l] < qtail[l]:
                        arr = qarr[qoff[l] + qhead[l]]; i = qidx[qoff[l] + qhead[l]]; qhead[l] += 1
                        sl = arr if arr > busy[l] else busy[l]
                        if l < L:
                            _add_row(vflat, voff[l], wcols[l], wflat, woff[l], wcols[l], i, vmax)
                            kk = _fire(vflat, voff[l], wcols[l], fired, voff[l], theta[l], use_ref, qidx, qarr, qoff[l + 1], qtail[l + 1], sl + pipe)
                            qtail[l + 1] += kk; spikes[l] += kk
                            busy[l] = sl + (kk if kk > 1 else 1)
                        else:
                            base = woff[L] + i * n_out
                            for c in range(n_out):
                                so = o[c] + wflat[base + c]
                                if so > omax: so = omax
                                elif so < -omax - 1: so = -omax - 1
                                o[c] = so
                            busy[L] = sl + 1
                            if nrec < nrec_max:
                                rec_cycle[nrec] = sl + pipe
                                for c in range(n_out): rec_o[nrec, c] = o[c]
                                tot = 0
                                for ll in range(L): tot += spikes[ll]
                                rec_spk[nrec] = tot
                                nrec += 1
                    qhead[l] = 0; qtail[l] = 0
            else:
                busy[0] = s + 1
            e += 1
        if next_tick > t_end_us:
            done = True
            break
        # ---------------- leak tick
        c_tick = _ceil_div(next_tick * 1000, tclk_ns)
        for l in range(L): _leak(vflat, voff[l], wcols[l], k_leak[l])
        for c in range(n_out): o[c] = o[c] - (o[c] >> k_out)
        tick_no += 1
        if mode_id == 0:
            while use_ref and next_ref <= next_tick:
                for j in range(voff[L]): fired[j] = 0
                next_ref += dt_ref_us
            rc = c_tick
        elif mode_id == 2:
            rc = c_tick
            if tick_no % n_eval == 0:
                # 1) every deeper layer and the output integrate the spikes their source layer emitted at the previous
                #    evaluation (queue l holds them), after the leak
                for l in range(1, L + 1):
                    while qhead[l] < qtail[l]:
                        i = qidx[qoff[l] + qhead[l]]; qhead[l] += 1
                        if l < L:
                            _add_row(vflat, voff[l], wcols[l], wflat, woff[l], wcols[l], i, vmax)
                        else:
                            base = woff[L] + i * n_out
                            for c in range(n_out):
                                so = o[c] + wflat[base + c]
                                if so > omax: so = omax
                                elif so < -omax - 1: so = -omax - 1
                                o[c] = so
                    qhead[l] = 0; qtail[l] = 0
                # 2) all thresholds; the new spikes wait in the queues for the next evaluation
                for l in range(L):
                    k = _fire(vflat, voff[l], wcols[l], fired, voff[l], theta[l], False, qidx, qarr, qoff[l + 1], qtail[l + 1], c_tick + 1)
                    qtail[l + 1] += k; spikes[l] += k
                    if qtail[l + 1] > qmax[l + 1]: qmax[l + 1] = qtail[l + 1]
                rc = c_tick + 1
        else:
            rc = c_tick
            if tick_no % n_eval == 0:
                # cascade: layer 0 compare one cycle after the tick (after any pending input add)
                s = c_tick if c_tick > busy[0] else busy[0]
                s = s + 1
                k = _fire(vflat, voff[0], wcols[0], fired, voff[0], theta[0], False, qidx, qarr, qoff[1], qtail[1], s + pipe)
                qtail[1] += k; spikes[0] += k
                busy[0] = s + (k if k > 1 else 1)
                last_cycle = s + pipe
                for l in range(1, L + 1):
                    if qtail[l] - qhead[l] > qmax[l]: qmax[l] = qtail[l] - qhead[l]
                    had_input = qtail[l] > qhead[l]
                    # layered semantics: add ALL spikes of the previous layer first, then evaluate the thresholds once
                    while qhead[l] < qtail[l]:
                        arr = qarr[qoff[l] + qhead[l]]; i = qidx[qoff[l] + qhead[l]]; qhead[l] += 1
                        sl = arr if arr > busy[l] else busy[l]
                        if l < L:
                            _add_row(vflat, voff[l], wcols[l], wflat, woff[l], wcols[l], i, vmax)
                        else:
                            base = woff[L] + i * n_out
                            for c in range(n_out):
                                so = o[c] + wflat[base + c]
                                if so > omax: so = omax
                                elif so < -omax - 1: so = -omax - 1
                                o[c] = so
                        busy[l] = sl + 1
                        if sl + pipe > last_cycle: last_cycle = sl + pipe
                    qhead[l] = 0; qtail[l] = 0
                    if l < L and had_input:          # without input no neuron can have crossed the threshold
                        sc = busy[l]                  # compare one cycle after the last add
                        kk = _fire(vflat, voff[l], wcols[l], fired, voff[l], theta[l], False, qidx, qarr, qoff[l + 1], qtail[l + 1], sc + pipe)
                        qtail[l + 1] += kk; spikes[l] += kk
                        busy[l] = sc + (kk if kk > 1 else 1)
                        if sc + pipe > last_cycle: last_cycle = sc + pipe
                rc = last_cycle
        if nrec < nrec_max:
            rec_cycle[nrec] = rc
            for c in range(n_out): rec_o[nrec, c] = o[c]
            tot = 0
            for ll in range(L): tot += spikes[ll]
            rec_spk[nrec] = tot
            nrec += 1
        next_tick += dt_leak_us
    return nrec, e


class IntNet:
    """Integer network from snn.model.export_int() / load_int() dictionaries."""
    def __init__(self, d: dict):
        self.d = d
        self.wflat, self.woff, self.wrows, self.wcols = pack_weights(d["W"])
        self.n_out = int(d["n_out"]); self.L = len(d["hidden"])
        self.theta = np.array(d["theta"], np.int64)
        kl = d["k_leak"]; self.k_leak = np.array([kl] * self.L if isinstance(kl, int) else kl, np.int64); self.k_out = int(d["k_out"])
        self.vmax, self.omax = 2 ** (int(d["vbits"]) - 1) - 1, 2 ** (int(d["obits"]) - 1) - 1
        pass

    def run(self, t_us, ch, mode: str, dt_leak_us: int, t_end_us: int, n_eval: int = 1, tclk_ns: int = 20, pipe: int = 2, dt_ref_us=None):
        """dt_ref_us: async refractory period (fired bits cleared every dt_ref_us); None = the leak period; 0 = no refractory."""
        if dt_ref_us is None: dt_ref_us = dt_leak_us
        """Returns dict: cycles (n_rec,), o (n_rec, n_out), time_us (n_rec,), spikes (L,), qmax (L+1,), n_events,
        spk_cum (n_rec,) = hidden spikes emitted up to each record (spikes to decision = spk_cum at the decision record)."""
        t_us = np.ascontiguousarray(t_us, dtype=np.int64); ch = np.ascontiguousarray(ch, dtype=np.int64)
        n_ticks = int(t_end_us // dt_leak_us) + 2
        nrec_max = 4 * len(t_us) + n_ticks + 64
        rec_cycle = np.zeros(nrec_max, np.int64); rec_o = np.zeros((nrec_max, self.n_out), np.int64)
        rec_spk = np.zeros(nrec_max, np.int64)
        spikes = np.zeros(self.L, np.int64); qmax = np.zeros(self.L + 1, np.int64)
        nrec, n_ev = _run(t_us, ch, self.wflat, self.woff, self.wrows, self.wcols, self.theta, self.k_leak, self.k_out,
                          self.vmax, self.omax, int(dt_leak_us), int(t_end_us), int(tclk_ns), int(pipe),
                          {"async": 0, "layered": 1, "pipelined": 2}[mode], int(n_eval), rec_cycle, rec_o, rec_spk, spikes, qmax, int(dt_ref_us))
        cycles = rec_cycle[:nrec].copy(); o = rec_o[:nrec].copy()
        return dict(cycles=cycles, o=o, time_us=cycles * tclk_ns / 1000.0, spikes=spikes, qmax=qmax, n_events=int(n_ev),
                    spk_cum=rec_spk[:nrec].copy(), overflow=bool(nrec >= nrec_max))


# ----------------------------------------------------------------------------------------------- metrics
def sample_on_grid(time_us, o, grid_us):
    """Output vector at each grid time (last record at or before it; zeros before the first record)."""
    idx = np.searchsorted(time_us, grid_us, side="right") - 1
    out = np.zeros((len(grid_us), o.shape[1]), np.int64)
    ok = idx >= 0
    out[ok] = o[idx[ok]]
    return out


def margin_decisions(time_us, o, margins):
    """For each margin: (time_us of the first record whose top1 - top2 >= margin, predicted class, record index)
    or (inf, -1, -1)."""
    if len(o) == 0:
        return [(np.inf, -1, -1) for _ in margins]
    part = np.sort(o, axis=1)
    gap = part[:, -1] - part[:, -2] if o.shape[1] > 1 else part[:, -1]
    pred = o.argmax(1)
    res = []
    for m in margins:
        hit = np.flatnonzero(gap >= m)
        res.append((float(time_us[hit[0]]), int(pred[hit[0]]), int(hit[0])) if len(hit) else (np.inf, -1, -1))
    return res
