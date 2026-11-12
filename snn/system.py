#!/usr/bin/env python3
"""Cycle-accurate model of the multicore system of docs/hardware.md, the behaviour the RTL must reproduce.

Mapping: every hidden layer of 4096 neurons is 16 cores of 256 neurons; a core's axons are the presynaptic units whose
weight block into the core is non-zero (the pruned connectivity; at most 256 per core), numbered in increasing order of
the presynaptic index. The output unit holds the 10 class integrators and receives every neuron of the last hidden
layer. Costs: a spike into a core takes ceil(256 / U) cycles (U neuron units), a leak or threshold scan the same, a
spike into the output unit 10 cycles; a spike emitted by neuron j of an operation starting at cycle S leaves at
S + j // U + P and reaches its target FIFO H cycles later, at most one spike per target per cycle (ties to the lower
source core, then the lower neuron). Input events enter at most one per cycle at ceil(t * f) and go to their target
cores H cycles later.

Schedules:
  step   pipelined time steps of T with a global tick at the end of every step (as trained, bit-exact with the torch
         'pipelined' mode). First-layer cores add input events on arrival and run a fused leak-and-threshold scan at the
         tick; deeper cores collect the spikes that arrive during the step and, at the tick, run a leak scan, add one
         row per collected axon, and run a threshold scan; the output unit leaks and adds at the tick.
  event  no global tick: every spike is processed on arrival with a threshold check after each add (one spike per
         neuron per refractory window); every core has a local timer of period T that leaks and clears the refractory
         bits. The output unit adds every spike on arrival.
Input-paced (--speeds ip): one input event every --in_period cycles and no wall clock (the delayed copies of a network with
input memory are released by a delay buffer inside the chip and take one cycle at the port). The input asks for the tick that
closes a step once the step's events are in; stepped systems take no event before the tick that opens its step. Three
schedules: step:aligned = global tick at that request but at least --tick_us after the previous tick (TrueNorth: the
worst case of a core); step:barrier = tick once every core has finished and every spike is delivered (global done wires
and a controller, --barrier_cyc); event:wire = the tick goes out on a broadcast wire at that request without waiting
for anything, and the cores use it only for the leak (one leak scan once the row in progress is done; spikes still in a
FIFO are added after it). event:lazy is a variant without the wire: the step travels with the spikes as a tag.
Real time (--speeds rt): input events at their recorded times; event:wire gets its tick from a timer every T of wall time.

Bounded FIFOs (--fifo, default 64): event:wire runs with back-pressure. A spike leaves the router
only when its target FIFO has room, a core that cannot hand a spike to the router stalls, the input-paced port waits
until all first-layer cores of the next event have room, and the real-time port holds --port_buf events and drops
(counts) what arrives while it is full. That couples every layer to the next one, so this path simulates all cores in
one event loop (_cosim_wire); with --fifo 0 it equals the layer-by-layer model below. The stepped schedules need no
queue beyond the first layer (deeper cores mark pending axons in bit vectors); their fifo_max is the first layer's.

Otherwise the model runs layer by layer (the network is feed-forward and the FIFOs are sized never to fill), so a
sample costs about one numba operation per synaptic operation.

  python -m snn.system --model results/models/<run>/model_int.npz --dataset nmnist32 --f_mhz 4 --units 12,1,1,1,1
"""
from __future__ import annotations
import argparse, json, math, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from numba import njit
from .data import EventData, ROOT
from .model import load_int
from .intref import sample_on_grid, margin_decisions


# ----------------------------------------------------------------------------------------------- mapping
class Mapping:
    """Cores, axons and routes derived from the exported integer weights (non-zero blocks = connections)."""
    def __init__(self, d: dict, core: int = 256):
        self.W = [np.asarray(w, np.int64) for w in d["W"]]
        self.hidden = list(d["hidden"]); self.L = len(self.hidden); self.n_out = int(d["n_out"]); self.core = core
        n = self.hidden[0]
        assert all(h == n for h in self.hidden) and n % core == 0
        self.G = n // core
        self.theta = [int(t) for t in d["theta"]]
        kl = d["k_leak"]; self.k = [int(x) for x in (kl if isinstance(kl, list) else [kl] * self.L)]
        self.k_out = int(d["k_out"]); self.vmax = 2 ** (int(d["vbits"]) - 1) - 1; self.omax = 2 ** (int(d["obits"]) - 1) - 1
        # per weight layer l < L: for every core c of layer l, its axon list (presynaptic indices) and its weight rows
        self.axons, self.rows = [], []
        self.route_core, self.route_axon = [], []           # per presynaptic unit: up to F targets (core, axon), -1 = none
        for l in range(self.L):
            W = self.W[l]; n_pre = W.shape[0]
            blk = (W.reshape(n_pre, self.G, core) != 0).any(2)          # (n_pre, G)
            axl, rws = [], []
            fan = int(blk.sum(1).max()) if n_pre else 0
            rc = -np.ones((n_pre, max(1, fan)), np.int64); ra = -np.ones((n_pre, max(1, fan)), np.int64); used = np.zeros(n_pre, np.int64)
            for c in range(self.G):
                pre = np.flatnonzero(blk[:, c])
                assert len(pre) <= core, f"layer {l} core {c} needs {len(pre)} axons"
                axl.append(pre); rws.append(np.ascontiguousarray(W[pre][:, c * core:(c + 1) * core]))
                for a, i in enumerate(pre):
                    rc[i, used[i]] = c; ra[i, used[i]] = a; used[i] += 1
            self.axons.append(axl); self.rows.append(rws); self.route_core.append(rc); self.route_axon.append(ra)
        self.Wout = np.ascontiguousarray(self.W[self.L])                 # (n_last, n_out)
        self.spiking_out = bool(d.get("spiking_out", False))              # spiking output: threshold, hard reset, a counter per class
        self.theta_out = int(d.get("theta_out", 0)) if self.spiking_out else 0


# ----------------------------------------------------------------------------------------------- kernels
@njit(cache=True)
def _sat(x, vmax):
    if x > vmax: return vmax
    if x < -vmax - 1: return -vmax - 1
    return x


@njit(cache=True)
def _core_event(Wc, theta, k, vmax, ent_cyc, ent_axon, U, P, T_cyc, busy, out_cyc, out_neu, stats, D=1):
    """Event-driven core. Entries in processing order (a tick marker, axon < 0, after the entries visible at or before
    its cycle). busy[cycle // T_cyc] collects the work; stats: [FIFO maximum, synaptic operations, total busy cycles]."""
    n = Wc.shape[1]
    per = ((n + U - 1) // U) * D
    v = np.zeros(n, np.int64); fired = np.zeros(n, np.uint8)
    F = 0; n_out = 0; p = 0; N = ent_cyc.shape[0]; occ_max = 0
    nb = busy.shape[0]
    for e in range(N):
        A = ent_cyc[e]
        S = A if A > F else F
        while p < N and ent_cyc[p] <= S: p += 1
        if p - e > occ_max: occ_max = p - e
        a = ent_axon[e]
        if a < 0:
            for j in range(n):
                v[j] = v[j] - (v[j] >> k); fired[j] = 0
        else:
            for j in range(n):
                x = _sat(v[j] + Wc[a, j], vmax)
                if x >= theta and fired[j] == 0:
                    x = 0; fired[j] = 1
                    out_cyc[n_out] = S + (j // U) * D + P; out_neu[n_out] = j; n_out += 1
                v[j] = x
            stats[1] += n
        F = S + per; stats[2] += per
        b = S // T_cyc
        if b < nb: busy[b] += per
    stats[0] = occ_max
    return n_out


@njit(cache=True)
def _core_step_first(Wc, theta, k, vmax, ent_cyc, ent_axon, ent_tick, U, P, busy, done, out_cyc, out_neu, stats, D=1, out_tag=None):
    """First-layer core, time-stepped: input events are added on arrival; tick marker t (axon < 0, t = 1, 2, ...,
    nominal tick in ent_tick) runs the fused leak-and-threshold scan. busy[t-1] / done[t-1]: work and completion of
    hardware step t-1 (the scan of marker t-1 and the adds of the events before marker t). stats[0] = FIFO maximum."""
    n = Wc.shape[1]
    per = ((n + U - 1) // U) * D
    v = np.zeros(n, np.int64)
    F = 0; n_out = 0; t = 0; p = 0; N = ent_cyc.shape[0]; occ_max = 0
    for e in range(N):
        A = ent_cyc[e]; S = A if A > F else F
        while p < N and ent_cyc[p] <= S: p += 1
        if p - e > occ_max: occ_max = p - e
        a = ent_axon[e]
        if a >= 0:
            for j in range(n): v[j] = _sat(v[j] + Wc[a, j], vmax)
            stats[1] += n
            if t < busy.shape[0]: busy[t] += per
        else:
            if t < done.shape[0]: done[t] = F                        # hardware step t complete (previous scan and adds)
            t += 1
            for j in range(n):
                x = v[j] - (v[j] >> k)
                if x >= theta:
                    x = 0; out_cyc[n_out] = S + (j // U) * D + P; out_neu[n_out] = j
                    if out_tag is not None: out_tag[n_out] = t
                    n_out += 1
                v[j] = x
            if t < busy.shape[0]: busy[t] += per
        F = S + per
    if t < done.shape[0]: done[t] = F
    stats[0] = occ_max
    return n_out


@njit(cache=True)
def _core_step(Wc, theta, k, vmax, ent_cyc, ent_axon, ticks, U, P, busy, done, out_cyc, out_neu, stats, D=1, out_tag=None):
    """Deeper core, time-stepped with one step of delay per hop. At tick t (cycle ticks[t-1]) the spikes visible at or
    before the tick are integrated: leak scan, one row per collected axon, threshold scan. busy[t] / done[t]: work and
    completion of tick t's processing. A tick served after the next tick counts as a miss."""
    n = Wc.shape[1]; n_ax = Wc.shape[0]
    per = ((n + U - 1) // U) * D
    v = np.zeros(n, np.int64); pend = np.zeros(n_ax, np.uint8)
    F = 0; n_out = 0; p = 0; N = ent_cyc.shape[0]; misses = 0
    for t in range(1, ticks.shape[0] + 1):
        K = ticks[t - 1]
        cnt = 0
        while p < N and ent_cyc[p] <= K:
            if pend[ent_axon[p]] == 0: cnt += 1
            pend[ent_axon[p]] = 1; p += 1
        S = K if K > F else F
        for j in range(n): v[j] = v[j] - (v[j] >> k)
        for a in range(n_ax):
            if pend[a]:
                for j in range(n): v[j] = _sat(v[j] + Wc[a, j], vmax)
                pend[a] = 0
        stats[1] += n * cnt
        S2 = S + per * (1 + cnt)
        for j in range(n):
            if v[j] >= theta:
                v[j] = 0; out_cyc[n_out] = S2 + (j // U) * D + P; out_neu[n_out] = j
                if out_tag is not None: out_tag[n_out] = t
                n_out += 1
        F = S2 + per
        busy[t] += per * (2 + cnt); done[t] = F
        if t < ticks.shape[0] and F > ticks[t]: misses += 1
    stats[0] = misses
    return n_out


@njit(cache=True)
def _core_lazy(Wc, theta, k, vmax, ent_cyc, ent_axon, ent_tag, U, P, out_cyc, out_neu, out_tag, stats, T_cyc=1, busy=None, D=1, use_ref=1):
    """Event-driven core with lazy leak: every entry carries the step of the input it descends from; before adding an
    entry of a newer step the core runs one leak scan per elapsed step (clearing the refractory bits). An idle core does
    no work at all. Spikes inherit the tag. stats: [FIFO maximum, synaptic operations, busy cycles]."""
    n = Wc.shape[1]
    per = ((n + U - 1) // U) * D
    v = np.zeros(n, np.int64); fired = np.zeros(n, np.uint8)
    F = 0; n_out = 0; p = 0; N = ent_cyc.shape[0]; occ_max = 0; cur = 0; cref = 0
    for e in range(N):
        A = ent_cyc[e]
        S = A if A > F else F
        while p < N and ent_cyc[p] <= S: p += 1
        if p - e > occ_max: occ_max = p - e
        tg = ent_tag[e]                                          # tag = step * 32 + refractory window within the step
        while cur < (tg >> 5):                                   # one leak scan per elapsed step
            for j in range(n):
                v[j] = v[j] - (v[j] >> k); fired[j] = 0
            if busy is not None and S // T_cyc < busy.shape[0]: busy[S // T_cyc] += per
            cur += 1; S += per; stats[2] += per; cref = cur << 5
        if tg > cref:                                            # newer refractory window: flash-clear the refractory bits
            for j in range(n): fired[j] = 0
            cref = tg
        a = ent_axon[e]
        for j in range(n):
            x = _sat(v[j] + Wc[a, j], vmax)
            if x >= theta and (use_ref == 0 or fired[j] == 0):
                x = 0; fired[j] = use_ref
                out_cyc[n_out] = S + (j // U) * D + P; out_neu[n_out] = j; out_tag[n_out] = tg; n_out += 1
            v[j] = x
        stats[1] += n; stats[2] += per
        if busy is not None and S // T_cyc < busy.shape[0]: busy[S // T_cyc] += per
        F = S + per
    stats[0] = occ_max
    return n_out


@njit(cache=True)
def _core_wire(Wc, theta, k, vmax, ent_cyc, ent_axon, wire, U, P, out_cyc, out_neu, stats, T_cyc=1, busy=None, D=1, use_ref=0):
    """Event-driven core on the broadcast tick wire: every spike is added on arrival with a threshold check; a tick
    (wire: arrival cycles, increasing) only drives the leak: one leak scan as soon as the row in progress is finished,
    before any entry not yet started, so spikes still in the FIFO are added after the leak. use_ref = 1: one spike per
    neuron between two ticks. stats: [FIFO maximum, synaptic operations, busy cycles]."""
    n = Wc.shape[1]
    per = ((n + U - 1) // U) * D
    v = np.zeros(n, np.int64); fired = np.zeros(n, np.uint8)
    F = 0; n_out = 0; p = 0; N = ent_cyc.shape[0]; occ_max = 0; q = 0; nq = wire.shape[0]
    for e in range(N + 1):
        if e < N: A = ent_cyc[e]
        elif nq: A = wire[nq - 1]                                # after the last entry: the remaining ticks
        else: A = 0
        while q < nq and wire[q] <= (A if A > F else F):         # ticks that come before this entry starts
            S = wire[q] if wire[q] > F else F
            for j in range(n):
                v[j] = v[j] - (v[j] >> k); fired[j] = 0
            if busy is not None and S // T_cyc < busy.shape[0]: busy[S // T_cyc] += per
            F = S + per; stats[2] += per; q += 1
        if e == N: break
        S = A if A > F else F
        while p < N and ent_cyc[p] <= S: p += 1
        if p - e > occ_max: occ_max = p - e
        a = ent_axon[e]
        for j in range(n):
            x = _sat(v[j] + Wc[a, j], vmax)
            if x >= theta and (use_ref == 0 or fired[j] == 0):
                x = 0; fired[j] = use_ref
                out_cyc[n_out] = S + (j // U) * D + P; out_neu[n_out] = j; n_out += 1
            v[j] = x
        stats[1] += n; stats[2] += per
        if busy is not None and S // T_cyc < busy.shape[0]: busy[S // T_cyc] += per
        F = S + per
    stats[0] = occ_max
    return n_out


@njit(cache=True)
def _out_wire(Wout, omax, k_out, ent_cyc, ent_src, wire, P, rec_cyc, rec_o, stats, D=1, theta_out=0, use_ref=0):
    """Output unit on the tick wire (see _core_wire): every spike adds its row, a tick leaks once the add in progress is
    done; the integrators are recorded after every add and every leak. theta_out > 0: spiking output, a threshold check
    after every add (hard reset, at most one spike per class between ticks with use_ref) and the spike counts are
    recorded instead of the integrators."""
    n_out = Wout.shape[1]
    o = np.zeros(n_out, np.int64); F = 0; nrec = 0; N = ent_cyc.shape[0]; q = 0; nq = wire.shape[0]
    cnt = np.zeros(n_out, np.int64); fired = np.zeros(n_out, np.uint8); spk = theta_out > 0
    for e in range(N + 1):
        if e < N: A = ent_cyc[e]
        elif nq: A = wire[nq - 1]
        else: A = 0
        while q < nq and wire[q] <= (A if A > F else F):
            S = wire[q] if wire[q] > F else F
            for c in range(n_out):
                o[c] = o[c] - (o[c] >> k_out); fired[c] = 0
            F = S + n_out * D; stats[2] += n_out * D; q += 1
            rec_cyc[nrec] = F + P
            for c in range(n_out): rec_o[nrec, c] = cnt[c] if spk else o[c]
            nrec += 1
        if e == N: break
        S = A if A > F else F
        i = ent_src[e]
        for c in range(n_out):
            x = _sat(o[c] + Wout[i, c], omax)
            if spk and x >= theta_out and (use_ref == 0 or fired[c] == 0):
                x = 0; fired[c] = use_ref; cnt[c] += 1
            o[c] = x
        stats[1] += n_out; stats[2] += n_out * D
        F = S + n_out * D
        rec_cyc[nrec] = F + P
        for c in range(n_out): rec_o[nrec, c] = cnt[c] if spk else o[c]
        nrec += 1
    return nrec


@njit(cache=True)
def _out_lazy(Wout, omax, k_out, ent_cyc, ent_src, ent_tag, P, rec_cyc, rec_o, stats, D=1):
    """Output unit with lazy leak (see _core_lazy): every spike adds its row and is recorded."""
    n_out = Wout.shape[1]
    o = np.zeros(n_out, np.int64); F = 0; nrec = 0; N = ent_cyc.shape[0]; cur = 0
    for e in range(N):
        A = ent_cyc[e]; S = A if A > F else F
        tg = ent_tag[e] >> 5                                     # the output unit only leaks per step
        while cur < tg:
            for c in range(n_out): o[c] = o[c] - (o[c] >> k_out)
            cur += 1; S += n_out * D; stats[2] += n_out * D
        i = ent_src[e]
        for c in range(n_out): o[c] = _sat(o[c] + Wout[i, c], omax)
        stats[1] += n_out; stats[2] += n_out * D
        F = S + n_out * D
        if nrec < rec_cyc.shape[0]:
            rec_cyc[nrec] = F + P
            for c in range(n_out): rec_o[nrec, c] = o[c]
            nrec += 1
    return nrec


@njit(cache=True)
def _out_event(Wout, omax, k_out, ent_cyc, ent_src, P, rec_cyc, rec_o, stats, D=1):
    """Output unit, event-driven: every spike adds its row (10 cycles); tick markers (src < 0) leak. stats[2] = busy."""
    n_out = Wout.shape[1]
    o = np.zeros(n_out, np.int64); F = 0; nrec = 0; N = ent_cyc.shape[0]
    for e in range(N):
        A = ent_cyc[e]; S = A if A > F else F
        i = ent_src[e]
        if i < 0:
            for c in range(n_out): o[c] = o[c] - (o[c] >> k_out)
        else:
            for c in range(n_out): o[c] = _sat(o[c] + Wout[i, c], omax)
            stats[1] += n_out
        F = S + n_out * D; stats[2] += n_out * D
        if nrec < rec_cyc.shape[0]:
            rec_cyc[nrec] = F + P
            for c in range(n_out): rec_o[nrec, c] = o[c]
            nrec += 1
    return nrec


@njit(cache=True)
def _out_step(Wout, omax, k_out, ent_cyc, ent_src, P, ticks, rec_cyc, rec_o, busy, done, stats, D=1, theta_out=0):
    """Output unit, time-stepped: at every tick leak, then add the spikes visible at or before the tick. theta_out > 0:
    spiking output, then a threshold scan (hard reset, one spike per class and step) and the counts are recorded."""
    n_out = Wout.shape[1]; n_src = Wout.shape[0]
    o = np.zeros(n_out, np.int64); pend = np.zeros(n_src, np.uint8); F = 0; p = 0; N = ent_cyc.shape[0]; nrec = 0
    cnt_o = np.zeros(n_out, np.int64); spk = theta_out > 0
    for t in range(1, ticks.shape[0] + 1):
        K = ticks[t - 1]
        cnt = 0
        while p < N and ent_cyc[p] <= K:
            if pend[ent_src[p]] == 0: cnt += 1
            pend[ent_src[p]] = 1; p += 1
        S = K if K > F else F
        for c in range(n_out): o[c] = o[c] - (o[c] >> k_out)
        for i in range(n_src):
            if pend[i]:
                for c in range(n_out): o[c] = _sat(o[c] + Wout[i, c], omax)
                pend[i] = 0
        stats[1] += n_out * cnt
        scans = 1
        if spk:
            scans = 2
            for c in range(n_out):
                if o[c] >= theta_out:
                    o[c] = 0; cnt_o[c] += 1
        F = S + n_out * (scans + cnt) * D
        busy[t] += n_out * (scans + cnt) * D; done[t] = F
        rec_cyc[nrec] = F + P
        for c in range(n_out): rec_o[nrec, c] = cnt_o[c] if spk else o[c]
        nrec += 1
    return nrec


# ----------------------------------------------------------------------------------------------- router
@njit(cache=True)
def _arbitrate(ready, src, n_src):
    """One target. Spikes sorted by (ready, src); per source the ready cycles increase. Every cycle the target takes
    the ready spike of the lowest-numbered source (per-source queues, fixed priority). Returns arrival cycles."""
    n = ready.shape[0]
    arr = np.empty(n, np.int64)
    # per-source index lists in order
    cnt = np.zeros(n_src, np.int64)
    for i in range(n): cnt[src[i]] += 1
    start = np.zeros(n_src + 1, np.int64)
    for s in range(n_src): start[s + 1] = start[s] + cnt[s]
    idx = np.empty(n, np.int64); fill = start[:n_src].copy()
    for i in range(n):
        idx[fill[src[i]]] = i; fill[src[i]] += 1
    ptr = start[:n_src].copy()
    done = 0; t = ready[0] if n else 0
    while done < n:
        pick = -1; nxt = -1
        for s in range(n_src):
            if ptr[s] < start[s + 1]:
                r = ready[idx[ptr[s]]]
                if r <= t:
                    pick = s; break
                if nxt < 0 or r < nxt: nxt = r
        if pick < 0:
            t = nxt; continue
        arr[idx[ptr[pick]]] = t; ptr[pick] += 1; done += 1; t += 1
    return arr


def _route(emit_cyc, src_core, src_unit, route_core, route_axon, H, n_targets, n_src, tag=None):
    """Spikes (nominal emit cycle, source core, source unit[, step tag]) -> per target core: arrival cycles, axons, tags
    and emission cycles, in arrival order. A source core sends at most one spike per cycle (in nominal order, ties to
    the lower unit); a spike is ready at its target H cycles after it leaves; every target takes one spike per cycle
    from per-source queues, the lowest source first."""
    z = np.zeros(0, np.int64)
    if len(emit_cyc) == 0:
        return [(z, z, z, z) for _ in range(n_targets)]
    if tag is None: tag = np.zeros(len(emit_cyc), np.int64)
    o = np.lexsort((src_unit, emit_cyc, src_core))
    emit_cyc, src_core, src_unit, tag = emit_cyc[o], src_core[o], src_unit[o], tag[o]
    e = emit_cyc.copy()
    for i in range(1, len(e)):                                                   # one spike per cycle per source
        if src_core[i] == src_core[i - 1] and e[i] <= e[i - 1]: e[i] = e[i - 1] + 1
    tc = route_core[src_unit]; ta = route_axon[src_unit]                        # (n, F)
    nF = tc.shape[1]
    r = np.repeat(e, nF) + H; sc = np.repeat(src_core, nF); tg = np.repeat(tag, nF); em = np.repeat(e, nF)
    tc = tc.ravel(); ta = ta.ravel()
    keep = tc >= 0
    r, sc, tc, ta, tg, em = r[keep], sc[keep], tc[keep], ta[keep], tg[keep], em[keep]
    order = np.lexsort((sc, r, tc))
    r, sc, tc, ta, tg, em = r[order], sc[order], tc[order], ta[order], tg[order], em[order]
    out = []
    bounds = np.searchsorted(tc, np.arange(n_targets + 1))
    for c in range(n_targets):
        lo, hi = bounds[c], bounds[c + 1]
        if hi == lo:
            out.append((z, z, z, z)); continue
        arr = _arbitrate(r[lo:hi], sc[lo:hi], n_src)
        oo = np.argsort(arr, kind="stable")
        out.append((arr[oo], ta[lo:hi][oo], tg[lo:hi][oo], em[lo:hi][oo]))
    return out


def _input_stream(t_us, ch, ev_nom, mk_pos_us, mk_nom):
    """Items sent by the input interface, one per cycle, in this order: the sensor events by time, with the tick markers
    in between. Marker t of first-layer core c sits at sensor time mk_pos_us[c, t-1] (after every earlier event, before
    every later one) and its nominal cycle is mk_nom[c, t-1]; an event's nominal cycle is ev_nom. Returns item cycles,
    channels (-1 = marker), marker target cores and marker nominal ticks."""
    G, n_ticks = mk_pos_us.shape
    key = np.concatenate([t_us.astype(np.float64), mk_pos_us.ravel() - 0.5])
    nom = np.concatenate([ev_nom.astype(np.int64), mk_nom.ravel().astype(np.int64)])
    chs = np.concatenate([ch.astype(np.int64), -np.ones(G * n_ticks, np.int64)])
    tc = np.concatenate([-np.ones(len(t_us), np.int64), np.repeat(np.arange(G), n_ticks)])
    tk = np.concatenate([np.zeros(len(t_us), np.int64), mk_nom.ravel().astype(np.int64)])
    o = np.argsort(key, kind="stable")
    nom, chs, tc, tk = nom[o], chs[o], tc[o], tk[o]
    cyc = nom.copy()
    for i in range(1, len(cyc)):
        if cyc[i] <= cyc[i - 1]: cyc[i] = cyc[i - 1] + 1
    return cyc, chs, tc, tk


def _first_layer_entries(cyc, chs, tc, tk, tag, mp, H):
    """Per first-layer core: the items of the input stream that reach it (its events and its markers), H cycles later,
    with the marker ticks and the step tag of every event."""
    rc, ra = mp.route_core[0], mp.route_axon[0]
    out = []
    is_m = chs < 0; ch0 = np.where(is_m, 0, chs)
    for c in range(mp.G):
        hit = rc[ch0] == c                                       # (n_items, F)
        take = (is_m & ((tc == c) | (tc == -2))) | (~is_m & hit.any(1))
        ax = np.where(is_m, -1, (ra[ch0] * hit).sum(1))
        out.append((cyc[take] + H, ax[take].astype(np.int64), np.where(is_m, tk, 0)[take], tag[take]))
    return out


# ----------------------------------------------------------------------------------------------- co-simulation with bounded FIFOs
@njit(cache=True)
def _cosim_wire(Wall, theta, kk, vmax, U, D, P, H, tgt_core, tgt_axon, in_core, in_axon, Wout, omax, k_out, theta_out,
                D_out, ev_ch, ev_cyc, paced, ev_gap, idx_tick, wire_rt, n_ticks, use_ref, Q, B, R, CAP, PCAP,
                rec_cyc, rec_o, tr_on, tr, spikes, cstat):
    """Event-driven execution on the tick wire with back-pressure: all cores, the output unit, the routers and the input
    port in one event loop. Every core and the output unit have an input FIFO of Q entries (0 = unlimited); a FIFO entry
    is reserved when the router or the input port sends a spike and freed when the core starts its row. Per cycle every
    target takes at most one spike, from the lowest-numbered source whose spike is waiting. A source core holds at most R
    spikes waiting for their targets (0 = unlimited, per-target queues as in _route); a firing neuron that finds them
    full stalls the core's pipeline for a cycle, so the rest of its operations come later. Input-paced (paced > 0):
    event i leaves the port at least ev_gap[i - 1] cycles after event i - 1 (`paced` for a sensor event, 1 for a
    delayed copy released inside the chip), once all its first-layer cores have room, and the tick of every step
    boundary the input passes leaves with it. Real time (paced = 0): event i reaches the port at
    ev_cyc[i]; the port holds at most B events, drops an arriving event when full, and forwards one per cycle, in
    order, once all its cores have room; the tick is a timer (wire_rt). With Q = R = 0 the result equals _core_wire,
    _out_wire and _route layer by layer. cstat per unit: busy cycles, largest FIFO (entries arrived when a row starts,
    as in _core_wire), stall cycles, synaptic operations, time it goes idle.
    Returns (records, trace rows, dropped events, error: 0 ok, 1 records full, 2 trace full, 3 FIFO ring full,
    4 router pool full, 5 emission ring full)."""
    NCH, n = Wall.shape[0], Wall.shape[2]
    L = theta.shape[0]; G = NCH // L; OUT = NCH; NC = NCH + 1
    n_out = Wout.shape[1]; N = ev_ch.shape[0]; F0 = in_core.shape[1]
    INF = np.int64(1) << 62
    MASK = CAP - 1; ECAP = 2048; EMASK = ECAP - 1
    spk_out = theta_out > 0
    # unit state
    F = np.zeros(NC, np.int64); qn = np.zeros(NC, np.int64)
    v = np.zeros((NCH, n), np.int64); fired = np.zeros((NCH, n), np.uint8)
    o = np.zeros(n_out, np.int64); cnt = np.zeros(n_out, np.int64); fo = np.zeros(n_out, np.uint8)
    fax = np.empty((NC, CAP), np.int32); farr = np.empty((NC, CAP), np.int64)
    fh = np.zeros(NC, np.int64); ft = np.zeros(NC, np.int64)
    # emissions scheduled by the rows (stored as time - sigma, so a stall delays all of a core's pending work at once)
    em_t = np.empty((NCH, ECAP), np.int64); em_j = np.empty((NCH, ECAP), np.int32)
    eh = np.zeros(NCH, np.int64); et = np.zeros(NCH, np.int64); sigma = np.zeros(NCH, np.int64)
    last_el = np.full(NCH, -INF, np.int64); src_q = np.zeros(NCH, np.int64)
    # router: per (target, source) FIFO queues as linked lists in one pool
    qh = np.full((NC, G), -1, np.int64); qt = np.full((NC, G), -1, np.int64); mask = np.zeros(NC, np.int64)
    p_next = np.empty(PCAP, np.int64); p_ax = np.empty(PCAP, np.int64); p_el = np.empty(PCAP, np.int64); p_src = np.empty(PCAP, np.int64)
    for i in range(PCAP - 1): p_next[i] = i + 1
    p_next[PCAP - 1] = -1; free = 0
    # input port and the tick wire
    wt = np.full(n_ticks, INF, np.int64)
    if paced == 0:
        for k in range(n_ticks): wt[k] = wire_rt[k]
    i_next = 0; s_next = 0; k_next = 0
    if paced > 0 and N == 0:
        for k in range(n_ticks): wt[k] = H
        k_next = n_ticks
    pb = np.empty(max(B, 1), np.int64); pbh = 0; pbt = 0; i_arr = 0; drops = 0
    nrec = 0; ntr = 0; err = 0
    t = 0
    while True:
        # ---- 1. emissions due now enter the router (one per cycle per source); a full source stalls its core
        #         (first, so that a stalled core does not start an operation in this cycle)
        for c in range(NCH):
            while eh[c] < et[c]:
                pos = eh[c] & EMASK
                if em_t[c, pos] + sigma[c] > t: break
                l = c // G; g = c - l * G; j = em_j[c, pos]; u = g * n + j
                X = tgt_core[l, u]
                if X >= 0 and R > 0 and src_q[c] >= R:
                    sigma[c] += 1; F[c] = (F[c] if F[c] > t else t) + 1; cstat[c, 2] += 1
                    break
                el = t if t > last_el[c] else last_el[c] + 1
                last_el[c] = el; eh[c] += 1
                if tr_on:
                    if ntr >= tr.shape[0]: return nrec, ntr, drops, 2
                    tr[ntr, 0] = l; tr[ntr, 1] = g; tr[ntr, 2] = j; tr[ntr, 3] = t; ntr += 1
                if X >= 0:
                    if free < 0: return nrec, ntr, drops, 4
                    pi = free; free = p_next[pi]
                    p_ax[pi] = tgt_axon[l, u]; p_el[pi] = el; p_src[pi] = c; p_next[pi] = -1
                    if qt[X, g] < 0: qh[X, g] = pi
                    else: p_next[qt[X, g]] = pi
                    qt[X, g] = pi; mask[X] |= np.int64(1) << g; src_q[c] += 1
        # ---- 2. free units start their next operation: a tick that has arrived first, then the oldest arrived entry
        for c in range(NC):
            if F[c] > t: continue
            qq = qn[c]
            if qq < n_ticks and wt[qq] <= t:
                if c < NCH:
                    l = c // G; kl = kk[l]
                    for j in range(n):
                        v[c, j] = v[c, j] - (v[c, j] >> kl); fired[c, j] = 0
                    per = ((n + U[l] - 1) // U[l]) * D[l]
                else:
                    for cc in range(n_out):
                        o[cc] = o[cc] - (o[cc] >> k_out); fo[cc] = 0
                    per = n_out * D_out
                    if nrec >= rec_cyc.shape[0]: return nrec, ntr, drops, 1
                    rec_cyc[nrec] = t + per + P
                    for cc in range(n_out): rec_o[nrec, cc] = cnt[cc] if spk_out else o[cc]
                    nrec += 1
                F[c] = t + per; cstat[c, 0] += per; qn[c] = qq + 1
                continue
            if ft[c] == fh[c] or farr[c, fh[c] & MASK] > t: continue
            lo = fh[c]; hi = ft[c]                                     # entries that have arrived (for the FIFO maximum)
            while lo < hi:
                mid = (lo + hi) >> 1
                if farr[c, mid & MASK] <= t: lo = mid + 1
                else: hi = mid
            if lo - fh[c] > cstat[c, 1]: cstat[c, 1] = lo - fh[c]
            a = fax[c, fh[c] & MASK]; fh[c] += 1
            if c < NCH:
                l = c // G; th = theta[l]; Ul = U[l]; Dl = D[l]
                per = ((n + Ul - 1) // Ul) * Dl
                base = t - sigma[c]
                for j in range(n):
                    x = _sat(v[c, j] + Wall[c, a, j], vmax)
                    if x >= th and (use_ref == 0 or fired[c, j] == 0):
                        x = 0; fired[c, j] = use_ref
                        if et[c] - eh[c] >= ECAP: return nrec, ntr, drops, 5
                        pos = et[c] & EMASK; em_t[c, pos] = base + (j // Ul) * Dl + P; em_j[c, pos] = j; et[c] += 1
                        spikes[l] += 1
                    v[c, j] = x
                cstat[c, 3] += n
            else:
                for cc in range(n_out):
                    x = _sat(o[cc] + Wout[a, cc], omax)
                    if spk_out and x >= theta_out and (use_ref != 1 or fo[cc] == 0):
                        x = 0; fo[cc] = 1 if use_ref == 1 else 0; cnt[cc] += 1
                    o[cc] = x
                per = n_out * D_out
                cstat[c, 3] += n_out
                if nrec >= rec_cyc.shape[0]: return nrec, ntr, drops, 1
                rec_cyc[nrec] = t + per + P
                for cc in range(n_out): rec_o[nrec, cc] = cnt[cc] if spk_out else o[cc]
                nrec += 1
            F[c] = t + per; cstat[c, 0] += per
        # ---- 3. every target with room takes one waiting spike, from the lowest-numbered source
        for X in range(G, NC):
            m = mask[X]
            if m == 0: continue
            if Q > 0 and ft[X] - fh[X] >= Q: continue
            pick = -1
            for g in range(G):
                if (m >> g) & 1 and p_el[qh[X, g]] <= t:
                    pick = g; break
            if pick < 0: continue
            pi = qh[X, pick]; nx = p_next[pi]; qh[X, pick] = nx
            if nx < 0:
                qt[X, pick] = -1; mask[X] &= ~(np.int64(1) << pick)
            src_q[p_src[pi]] -= 1
            if ft[X] - fh[X] >= CAP: return nrec, ntr, drops, 3
            fax[X, ft[X] & MASK] = p_ax[pi]; farr[X, ft[X] & MASK] = t + H; ft[X] += 1
            p_next[pi] = free; free = pi
        # ---- 4. input port
        if paced > 0:
            if i_next < N and t >= s_next:
                chn = ev_ch[i_next]; ok = True
                if Q > 0:
                    for f in range(F0):
                        X = in_core[chn, f]
                        if X >= 0 and ft[X] - fh[X] >= Q: ok = False
                if ok:
                    for f in range(F0):
                        X = in_core[chn, f]
                        if X >= 0:
                            if ft[X] - fh[X] >= CAP: return nrec, ntr, drops, 3
                            fax[X, ft[X] & MASK] = in_axon[chn, f]; farr[X, ft[X] & MASK] = t + H; ft[X] += 1
                    s_next = t + ev_gap[i_next]
                    while k_next < n_ticks and idx_tick[k_next] <= i_next:
                        wt[k_next] = t + H; k_next += 1
                    i_next += 1
                    if i_next == N:
                        while k_next < n_ticks:
                            wt[k_next] = s_next + H; k_next += 1
        else:
            sent = False
            for rnd in range(2):                                       # forward the oldest event, accept arrivals, forward
                if rnd == 1:
                    while i_arr < N and ev_cyc[i_arr] <= t:
                        if pbt - pbh < B: pb[pbt % B] = i_arr; pbt += 1
                        else: drops += 1
                        i_arr += 1
                if sent or pbt == pbh: continue
                chn = ev_ch[pb[pbh % B]]; ok = True
                if Q > 0:
                    for f in range(F0):
                        X = in_core[chn, f]
                        if X >= 0 and ft[X] - fh[X] >= Q: ok = False
                if ok:
                    for f in range(F0):
                        X = in_core[chn, f]
                        if X >= 0:
                            if ft[X] - fh[X] >= CAP: return nrec, ntr, drops, 3
                            fax[X, ft[X] & MASK] = in_axon[chn, f]; farr[X, ft[X] & MASK] = t + H; ft[X] += 1
                    pbh += 1; sent = True
        # ---- next cycle at which something can happen
        nt = INF
        for c in range(NC):
            if F[c] > t:
                if F[c] < nt: nt = F[c]
                continue
            qq = qn[c]
            if qq < n_ticks and wt[qq] < nt: nt = wt[qq]
            if ft[c] > fh[c]:
                A = farr[c, fh[c] & MASK]
                if A < nt: nt = A
        for c in range(NCH):
            if eh[c] < et[c]:
                e = em_t[c, eh[c] & EMASK] + sigma[c]
                if e < nt: nt = e
        for X in range(G, NC):
            if mask[X] != 0 and (Q == 0 or ft[X] - fh[X] < Q) and t + 1 < nt: nt = t + 1
        if paced > 0:
            if i_next < N:
                nxt = s_next
                if nxt <= t: nxt = t + 1
                if nxt < nt: nt = nxt
        else:
            if i_arr < N and ev_cyc[i_arr] < nt: nt = ev_cyc[i_arr]
            if pbt > pbh and t + 1 < nt: nt = t + 1
        if nt >= INF: break
        if nt <= t: nt = t + 1
        t = nt
    for c in range(NC): cstat[c, 4] = F[c]
    return nrec, ntr, drops, 0


def _cosim_tables(mp):
    """Packed weights and routes of a Mapping for _cosim_wire (built once per mapping)."""
    if getattr(mp, "_cosim", None) is not None:
        return mp._cosim
    L, G, n = mp.L, mp.G, mp.core
    Wall = np.zeros((L * G, n, n), np.int32)
    for l in range(L):
        for g in range(G):
            r = mp.rows[l][g]; Wall[l * G + g, :r.shape[0]] = r
    nh = mp.hidden[0]
    tgt_core = -np.ones((L, nh), np.int64); tgt_axon = -np.ones((L, nh), np.int64)
    for l in range(L - 1):
        rc, ra = mp.route_core[l + 1], mp.route_axon[l + 1]
        if rc.shape[1] != 1 and (rc[:, 1:] >= 0).any():
            raise NotImplementedError("co-simulation: every hidden neuron must feed at most one core")
        tgt_core[l] = np.where(rc[:, 0] >= 0, (l + 1) * G + rc[:, 0], -1); tgt_axon[l] = ra[:, 0]
    tgt_core[L - 1] = L * G; tgt_axon[L - 1] = np.arange(nh)
    in_core = np.where(mp.route_core[0] >= 0, mp.route_core[0], -1).astype(np.int64); in_axon = mp.route_axon[0].astype(np.int64)
    mp._cosim = (Wall, tgt_core, tgt_axon, in_core, in_axon)
    return mp._cosim


def _port_gap(ch, paced, gap=None, n_in0=None):
    """Cycles the input-paced port is held after every event (events in time order): `paced` for a sensor event, 1 for a
    delayed copy of the input memory (channel >= n_in0), which a delay buffer inside the chip releases at its time."""
    if gap is not None: return np.asarray(gap, np.int64)
    g = np.full(len(ch), int(paced), np.int64)
    if n_in0 is not None and paced: g[np.asarray(ch) >= n_in0] = 1
    return g


def _simulate_cosim(mp, t_us, ch, f_mhz, T_us, n_ticks, units, H, P, div, ref_us, paced, fifo, port_buf, out_reg, trace, ref_out=True, gap=None):
    """Event-driven execution on the tick wire with bounded FIFOs and back-pressure (see _cosim_wire); returns the
    fields of _simulate plus dropped input events, stall cycles, the output unit's largest FIFO and the time the last
    unit goes idle."""
    L, G = mp.L, mp.G
    Wall, tgt_core, tgt_axon, in_core, in_axon = _cosim_tables(mp)
    tt = np.arange(1, n_ticks + 1, dtype=np.int64)
    gap = _port_gap(ch, paced, gap)
    ev_nom = np.cumsum(gap) - gap if paced else np.ceil(t_us * f_mhz).astype(np.int64)
    z = np.zeros((G, 0))
    cyc, chs, tc, tk = _input_stream(t_us, ch, ev_nom, z, z.astype(np.int64))
    ts = np.sort(t_us)
    idx_tick = np.searchsorted(ts, tt * T_us, side="left").astype(np.int64)
    wire_rt = np.ceil(tt * T_us * f_mhz).astype(np.int64) + H
    use_ref = 0 if (ref_us is not None and int(ref_us) == 0) else (1 if ref_out else 2)    # 2: hidden layers only
    Q = int(fifo); R = int(out_reg) if out_reg is not None and out_reg >= 0 else (0 if Q == 0 else 1)
    CAP = 1 << 16 if Q == 0 else 1 << max(6, int(np.ceil(np.log2(max(Q, 2)))))
    U = np.asarray(units[:L], np.int64); D = np.asarray(div[:L], np.int64)
    theta = np.asarray(mp.theta, np.int64); kk = np.asarray(mp.k, np.int64)
    n_rec = 1 << 17; n_tr = (1 << 18) if trace else 1
    while True:
        rec_cyc = np.zeros(n_rec, np.int64); rec_o = np.zeros((n_rec, mp.n_out), np.int64)
        tr = np.zeros((n_tr, 4), np.int64); spikes = np.zeros(L, np.int64); cstat = np.zeros((L * G + 1, 5), np.int64)
        nrec, ntr, drops, err = _cosim_wire(Wall, theta, kk, mp.vmax, U, D, P, H, tgt_core, tgt_axon, in_core, in_axon,
                                            mp.Wout, mp.omax, mp.k_out, mp.theta_out, int(div[L]), chs, cyc, int(paced), gap,
                                            idx_tick, wire_rt, n_ticks, use_ref, Q, int(port_buf), R, CAP, 1 << 20,
                                            rec_cyc, rec_o, bool(trace), tr, spikes, cstat)
        if err == 1: n_rec *= 4; continue
        if err == 2: n_tr *= 4; continue
        if err: raise RuntimeError(f"co-simulation buffer full (error {err})")
        break
    busy = cstat[:L * G, 0].reshape(L, G); occ = cstat[:L * G, 1].reshape(L, G)
    z3 = np.zeros((L, G, n_ticks + 2), np.int64); z1 = np.zeros(n_ticks + 2, np.int64)
    return dict(rec_cyc=rec_cyc[:nrec], o=rec_o[:nrec], spikes=spikes, ops=int(cstat[:, 3].sum()), work=z3, done=z3, out_work=z1,
                out_done=z1, fin_arr=z1, late=0, busy_core=busy, out_busy=int(cstat[L * G, 0]), fifo_max=occ,
                trace=tr[:ntr] if trace else None, stream=(cyc, chs, tc, tk, np.zeros(len(chs), np.int64)),
                drops=int(drops), stall_cycles=int(cstat[:, 2].sum()), fifo_out_max=int(cstat[L * G, 1]),
                last_idle=int(cstat[:, 4].max()) if len(cstat) else 0)


# ----------------------------------------------------------------------------------------------- one sample
def _simulate(mp, t_us, ch, schedule, f_mhz, T_us, n_ticks, units, H, P, ticks, out_ticks, fullspeed, trace, sync="aligned",
              ph0=None, period=None, div=None, ref_us=None, paced=0, gap=None):
    """One pass over the network for given tick times: ticks (L, G, n_ticks) per core, out_ticks (n_ticks,).
    paced > 0: input-paced (full speed without a wall clock): event i enters once the port has spent gap[j] cycles on
    every earlier event j (`paced` for a sensor event, 1 for a delayed copy; see _port_gap); a stepped system does not
    take an event before the tick that opens its step, so its input pauses while a tick is late."""
    T_cyc = int(round(T_us * f_mhz)); L, G = mp.L, mp.G
    div = [1] * (L + 1) if div is None else [int(x) for x in div]
    spikes = np.zeros(L, np.int64); ops = 0; tr = []
    work = np.zeros((L, G, n_ticks + 2), np.int64); done = np.zeros((L, G, n_ticks + 2), np.int64)
    busy_core = np.zeros((L, G), np.int64); occ = np.zeros((L, G), np.int64)
    tt = np.arange(1, n_ticks + 1, dtype=np.int64)
    if period is None: period = T_cyc
    ph = np.zeros(G, np.int64) if ph0 is None else np.asarray(ph0, np.int64)    # phase of every first-layer core (cycles)
    gap = _port_gap(ch, paced, gap); cum = np.concatenate([[0], np.cumsum(gap)]).astype(np.int64)   # port cycle before event i
    if not fullspeed:                                    # real time: sensor time is wall time
        ev_nom = np.ceil(t_us * f_mhz).astype(np.int64)
        mk_pos = tt[None, :] * float(T_us) + ph[:, None] / f_mhz; mk_nom = ticks[0]
    elif schedule == "step" and paced:                   # input-paced, stepped: one event per `paced` cycles, never before its step opens
        start_s = np.concatenate([[0], ticks[0, 0]]).astype(np.int64)
        idx = cum[:-1]
        ev_nom = idx + np.maximum.accumulate(start_s[np.minimum(t_us // T_us, n_ticks)] - idx)
        mk_pos = tt[None, :] * float(T_us) + (ph[:, None] / period) * T_us; mk_nom = ticks[0]
    elif schedule == "step":                             # full speed, stepped: step s's events from the start of hardware step s
        start_s = np.concatenate([[0], ticks[0, 0]]).astype(np.int64)
        ev_nom = start_s[np.minimum(t_us // T_us, n_ticks)]
        mk_pos = tt[None, :] * float(T_us) + (ph[:, None] / period) * T_us; mk_nom = ticks[0]
    else:                                                # full speed, event-driven: everything as fast as possible
        ev_nom = cum[:-1] if paced else np.zeros(len(t_us), np.int64)
        mk_pos = tt[None, :] * float(T_us) + (ph[:, None] / period) * T_us; mk_nom = np.zeros((G, n_ticks), np.int64)
    lazy = schedule == "event" and sync == "lazy"
    wire = schedule == "event" and sync == "wire"
    if mp.spiking_out and schedule == "event" and not wire: raise ValueError("spiking output: event-driven execution on the tick wire only")
    if lazy or wire:                                     # no markers: every event carries its step (lazy), or the tick has its own wire
        cyc, chs, tc, tk = _input_stream(t_us, ch, ev_nom, mk_pos[:, :0], mk_nom[:, :0])
    elif np.all(ph == ph[0]):                            # aligned: one marker per tick, broadcast to every first-layer core
        cyc, chs, tc, tk = _input_stream(t_us, ch, ev_nom, mk_pos[:1], mk_nom[:1]); tc = np.where(chs < 0, -2, tc)
    else:
        cyc, chs, tc, tk = _input_stream(t_us, ch, ev_nom, mk_pos, mk_nom)
    ts = np.sort(t_us); R_us = int(ref_us) if ref_us else T_us
    use_ref = 0 if (ref_us is not None and int(ref_us) == 0) else 1   # refractory window 0 = no refractory
    ev_tag = np.zeros(len(chs), np.int64)                 # events stay in time order; tag = step * 32 + refractory window
    ev_tag[chs >= 0] = ((ts // T_us) << 5) + ((ts % T_us) // R_us)
    first = _first_layer_entries(cyc, chs, tc, tk, ev_tag, mp, H)
    if wire and paced:      # the input raises the tick when it passes a step boundary (in the slot of the next event); every core sees it H later
        wire_cyc = cum[np.searchsorted(ts, tt * T_us, side="left")] + H
    elif wire:              # real time: a timer raises the tick every T of wall time
        wire_cyc = np.ceil(tt * T_us * f_mhz).astype(np.int64) + H
    # full-speed event-driven: deeper cores leak when the input stream passes the step boundary of their column
    if fullspeed and schedule == "event" and not (lazy or wire):
        mk_stream = np.zeros((G, n_ticks), np.int64)
        for c in range(G):
            m = (chs < 0) & ((tc == c) | (tc == -2)); mk_stream[c] = cyc[m][:n_ticks] + H
    ents = None
    fin_arr = np.zeros(n_ticks + 2, np.int64)              # latest delivery of the spikes emitted in every hardware step
    late = 0; aligned_ticks = bool(np.all(ticks == ticks[0, 0][None, None, :]))
    for l in range(L):
        em_c, em_core, em_unit, em_tag = [], [], [], []
        for c in range(G):
            Wc = mp.rows[l][c]; st = np.zeros(3, np.int64)
            out_tag = None
            if l == 0 and lazy:
                ec, ea, et, eg = first[c]
                out_cyc = np.zeros(len(ec) * Wc.shape[1] + 1, np.int64); out_neu = np.zeros_like(out_cyc); out_tag = np.zeros_like(out_cyc)
                n = _core_lazy(Wc, mp.theta[l], mp.k[l], mp.vmax, ec, ea, eg, units[l], P, out_cyc, out_neu, out_tag, st, T_cyc, work[l, c], div[l], use_ref)
                occ[l, c] = st[0]; busy_core[l, c] = st[2]
            elif l > 0 and lazy:
                arr, ax, tg, _ = ents[c]
                out_cyc = np.zeros(len(arr) * Wc.shape[1] + 1, np.int64); out_neu = np.zeros_like(out_cyc); out_tag = np.zeros_like(out_cyc)
                n = _core_lazy(Wc, mp.theta[l], mp.k[l], mp.vmax, arr, ax, tg, units[l], P, out_cyc, out_neu, out_tag, st, T_cyc, work[l, c], div[l], use_ref)
                occ[l, c] = st[0]; busy_core[l, c] = st[2]
            elif wire:
                ec, ea = (first[c] if l == 0 else ents[c])[:2]
                out_cyc = np.zeros(len(ec) * Wc.shape[1] + 1, np.int64); out_neu = np.zeros_like(out_cyc)
                n = _core_wire(Wc, mp.theta[l], mp.k[l], mp.vmax, ec, ea, wire_cyc, units[l], P, out_cyc, out_neu, st, T_cyc, work[l, c], div[l], use_ref)
                occ[l, c] = st[0]; busy_core[l, c] = st[2]
            elif l == 0:
                ec, ea, et, eg = first[c]
                out_cyc = np.zeros(len(ec) * Wc.shape[1] + 1, np.int64); out_neu = np.zeros_like(out_cyc)
                if schedule == "event":
                    n = _core_event(Wc, mp.theta[l], mp.k[l], mp.vmax, ec, ea, units[l], P, T_cyc, work[l, c], out_cyc, out_neu, st, div[l])
                    occ[l, c] = st[0]; busy_core[l, c] = st[2]
                else:
                    out_tag = np.zeros_like(out_cyc)
                    n = _core_step_first(Wc, mp.theta[l], mp.k[l], mp.vmax, ec, ea, et, units[l], P, work[l, c], done[l, c], out_cyc, out_neu, st, div[l], out_tag)
                    busy_core[l, c] = work[l, c].sum(); occ[l, c] = st[0]
            else:
                arr, ax, _, _ = ents[c]
                out_cyc = np.zeros(len(arr) * Wc.shape[1] + (n_ticks + 1) * Wc.shape[1] + 1, np.int64); out_neu = np.zeros_like(out_cyc)
                if schedule == "event":
                    tk_c = mk_stream[c] if fullspeed else ticks[l, c]
                    ec = np.concatenate([tk_c, arr]); ea = np.concatenate([-np.ones(n_ticks, np.int64), ax])
                    o = np.lexsort((ea < 0, ec)); ec, ea = ec[o], ea[o]                  # entries visible at the tick cycle first
                    n = _core_event(Wc, mp.theta[l], mp.k[l], mp.vmax, ec, ea, units[l], P, T_cyc, work[l, c], out_cyc, out_neu, st, div[l])
                    occ[l, c] = st[0]; busy_core[l, c] = st[2]
                else:
                    out_tag = np.zeros_like(out_cyc)
                    n = _core_step(Wc, mp.theta[l], mp.k[l], mp.vmax, arr, ax, ticks[l, c], units[l], P, work[l, c], done[l, c], out_cyc, out_neu, st, div[l], out_tag)
                    busy_core[l, c] = work[l, c].sum()
            ops += int(st[1]); spikes[l] += n
            em_c.append(out_cyc[:n]); em_core.append(np.full(n, c, np.int64)); em_unit.append(c * mp.core + out_neu[:n])
            em_tag.append(out_tag[:n] if out_tag is not None else np.zeros(n, np.int64))
        em_c = np.concatenate(em_c); em_core = np.concatenate(em_core); em_unit = np.concatenate(em_unit); em_tag = np.concatenate(em_tag)
        if trace: tr.append(np.stack([np.full(len(em_c), l), em_core, em_unit % mp.core, em_c], 1))
        if l < L - 1:
            ents = _route(em_c, em_core, em_unit, mp.route_core[l + 1], mp.route_axon[l + 1], H, G, G, em_tag)
            routed = ents
        else:
            rc = np.zeros((mp.hidden[-1], 1), np.int64); ra = np.arange(mp.hidden[-1], dtype=np.int64)[:, None]
            (arr, src, tgo, _), = routed = _route(em_c, em_core, em_unit, rc, ra, H, 1, G, em_tag)
        if schedule == "step":                            # deliveries per hardware step (barrier) and late spikes
            for tci, (aa, _, stp, ee) in enumerate(routed):
                if len(aa):                                                    # stp: the tick whose processing emitted the spike
                    np.maximum.at(fin_arr, np.minimum(stp, n_ticks + 1), aa)
                    if aligned_ticks:                     # a spike of tick t must be visible at its target by tick t + 1
                        tgt = ticks[l + 1, tci] if l < L - 1 else out_ticks
                        lim = np.where(stp < n_ticks, tgt[np.minimum(stp, n_ticks - 1)], np.iinfo(np.int64).max)
                        late += int((aa > lim).sum())
    st = np.zeros(3, np.int64); out_work = np.zeros(n_ticks + 2, np.int64); out_done = np.zeros(n_ticks + 2, np.int64)
    if lazy:
        rec_cyc = np.zeros(len(arr) + 1, np.int64); rec_o = np.zeros((len(arr) + 1, mp.n_out), np.int64)
        nrec = _out_lazy(mp.Wout, mp.omax, mp.k_out, arr, src, tgo, P, rec_cyc, rec_o, st, div[L]); out_busy = int(st[2])
    elif wire:
        rec_cyc = np.zeros(len(arr) + len(wire_cyc) + 1, np.int64); rec_o = np.zeros((len(rec_cyc), mp.n_out), np.int64)
        nrec = _out_wire(mp.Wout, mp.omax, mp.k_out, arr, src, wire_cyc, P, rec_cyc, rec_o, st, div[L], mp.theta_out, use_ref if ref_out else 0); out_busy = int(st[2])
    elif schedule == "event":
        tk_o = (mk_stream[0] if fullspeed else out_ticks)
        ec = np.concatenate([tk_o, arr]); es = np.concatenate([-np.ones(n_ticks, np.int64), src])
        o = np.lexsort((es < 0, ec)); ec, es = ec[o], es[o]
        rec_cyc = np.zeros(len(ec) + 1, np.int64); rec_o = np.zeros((len(ec) + 1, mp.n_out), np.int64)
        nrec = _out_event(mp.Wout, mp.omax, mp.k_out, ec, es, P, rec_cyc, rec_o, st, div[L]); out_busy = int(st[2])
    else:
        rec_cyc = np.zeros(n_ticks + 1, np.int64); rec_o = np.zeros((n_ticks + 1, mp.n_out), np.int64)
        nrec = _out_step(mp.Wout, mp.omax, mp.k_out, arr, src, P, out_ticks, rec_cyc, rec_o, out_work, out_done, st, div[L], mp.theta_out)
        out_busy = int(out_work.sum())
    ops += int(st[1])
    rec_cyc, rec_o = rec_cyc[:nrec], rec_o[:nrec]
    return dict(rec_cyc=rec_cyc, o=rec_o, spikes=spikes, ops=ops, work=work, done=done, out_work=out_work, out_done=out_done, fin_arr=fin_arr, late=late,
                busy_core=busy_core, out_busy=out_busy, fifo_max=occ, trace=np.concatenate(tr) if trace and tr else None,
                stream=(cyc, chs, tc, tk, ev_tag))


def run_sample(mp: Mapping, t_us, ch, schedule: str, f_mhz: float, T_us: int, t_end_us: int, units, H: int = 2, P: int = 2,
               trace: bool = False, sync: str = "aligned", fullspeed: bool = False, phases=None, P_wc: int | None = None,
               barrier_cyc: int = 16, max_iter: int = 6, div=None, ref_us=None, paced: int = 0, tick_cyc: int | None = None,
               fifo: int = 0, port_buf: int = 64, out_reg: int | None = None, cosim: bool | None = None, ref_out: bool = True,
               n_in0: int | None = None):
    """One sample on the system. paced > 0: input-paced mode (one input event every `paced` cycles, no wall clock;
    implies full speed; with input memory, n_in0 = number of sensor channels, a delayed copy (channel >= n_in0) is
    released by a delay buffer inside the chip at its time and takes one cycle at the port). The input asks for the tick that closes a step once the step's events are in, and a stepped
    system takes no event before the tick that opens its step. Stepped ticks: sync 'aligned' = global tick, issued at
    that request but at least tick_cyc after the previous tick (the worst case of a core, TrueNorth's 1 ms); 'barrier'
    = issued once every core has finished the step and its spikes are delivered, plus barrier_cyc. The event schedule
    never waits: sync 'wire' = the input's tick reaches every core on a broadcast wire and only drives the leak; 'lazy'
    = the step travels with the spikes as a tag.
    schedule 'step' (pipelined time steps) with sync 'aligned' (global tick), 'phases' (local ticks, phase per core) or
    'barrier' (the next tick is issued when every core has finished the step, plus barrier_cyc); schedule 'event' with
    local leak timers, sync 'aligned' or 'phases'. Real time: input events at their timestamps, ticks every T.
    Full speed: the recorded input as fast as possible; a fixed step lasts P_wc cycles (the worst case of a core), a
    barrier step as long as its slowest core; event-driven leaks follow the step boundaries in the input stream.
    phases: (L + 1, G) phase offsets in cycles (row L, column 0: output unit), used with sync 'phases'.
    Event-driven on the tick wire with fifo > 0 (or cosim=True): every FIFO holds `fifo` spikes with back-pressure,
    simulated for all cores at once (_cosim_wire); real time drops input events once the port's `port_buf` events
    wait; out_reg = spikes a core may hold for the router (default 1 with fifo > 0, unlimited with fifo 0)."""
    T_cyc = int(round(T_us * f_mhz)); n_ticks = int(t_end_us // T_us); L, G = mp.L, mp.G
    if paced:
        fullspeed = True
        if schedule == "step" and sync not in ("aligned", "barrier"): raise ValueError("input-paced stepped execution: global tick or barrier")
        if schedule == "step" and sync == "aligned" and not tick_cyc: raise ValueError("input-paced global tick needs tick_cyc")
    elif schedule == "event" and sync == "wire" and fullspeed: raise ValueError("the tick wire needs input-paced or real-time input")
    keep = t_us < t_end_us
    t_us, ch = t_us[keep], ch[keep]
    o = np.argsort(t_us, kind="stable"); t_us, ch = t_us[o], ch[o]                # events in time order (port gaps follow it)
    div = [1] * (L + 1) if div is None else [int(x) for x in div]                 # clock divider of every layer (GALS)
    if P_wc is None: P_wc = (mp.core * (mp.core + 2) + 64) * max(div[:L])       # worst case of a core: 256 rows + 2 scans
    period = P_wc if fullspeed else T_cyc
    if sync == "phases" and fullspeed: raise ValueError("per-core phases are a real-time option")
    ph = np.zeros((L + 1, G), np.int64) if (phases is None or sync != "phases") else np.asarray(phases, np.int64)
    tt = np.arange(1, n_ticks + 1, dtype=np.int64)
    ticks = tt[None, None, :] * period + ph[:L, :, None]                          # (L, G, n_ticks)
    out_ticks = tt * period + ph[L, 0]
    n_s = np.bincount(t_us // T_us, minlength=n_ticks)[:n_ticks]                   # input events per step
    port_gap = _port_gap(ch, paced, None, n_in0)                                   # port cycles of every event
    if schedule == "step" and sync == "aligned" and paced:
        g_s = np.bincount(t_us // T_us, weights=port_gap, minlength=n_ticks)[:n_ticks]   # port cycles per step
        new = np.zeros(n_ticks, np.int64); prev = 0; c = 0                         # prev: last tick; c: next free input slot
        for s in range(n_ticks):
            if n_s[s]: c = max(c, prev) + int(g_s[s])                              # the step's events enter from its opening tick
            prev = max(c, prev + int(tick_cyc)); new[s] = prev
        ticks = np.broadcast_to(new[None, None, :], (L, G, n_ticks)).copy(); out_ticks = new
    elif schedule == "step" and sync == "barrier" and fullspeed:
        # a barrier step starts with every earlier step finished, so how long it takes does not depend on when it starts:
        # one pass with the ticks far apart measures every step, and the barrier ticks are the running sum
        gap = n_s * (max(int(paced), mp.core * max(div[:L])) + 64) + 4 * P_wc + 4096
        wide = np.cumsum(gap).astype(np.int64)
        r = _simulate(mp, t_us, ch, "step", f_mhz, T_us, n_ticks, units, H, P, np.broadcast_to(wide[None, None, :], (L, G, n_ticks)).copy(),
                      wide, True, False, "barrier", ph[0], period, div, paced=paced, gap=port_gap)
        fin = np.maximum(np.maximum(r["done"][:, :, :n_ticks].max((0, 1)), r["out_done"][:n_ticks]), r["fin_arr"][:n_ticks])
        start = np.concatenate([[0], wide[:-1]])
        new = np.zeros(n_ticks, np.int64); prev = 0
        for s in range(n_ticks):
            prev += max(int(fin[s] - start[s]) + barrier_cyc, 1); new[s] = prev
        ticks = np.broadcast_to(new[None, None, :], (L, G, n_ticks)).copy(); out_ticks = new
    elif schedule == "step" and sync == "barrier":                               # real time: the input does not wait, iterate
        for it in range(max_iter):
            r = _simulate(mp, t_us, ch, "step", f_mhz, T_us, n_ticks, units, H, P, ticks, out_ticks, fullspeed, False, "barrier", ph[0], period, div,
                          paced=paced, gap=port_gap)
            # hardware step s ends when every core has finished its work of step s (first layer: scan and the adds of its
            # events) and every spike emitted in it has been delivered
            fin = np.maximum(np.maximum(r["done"][:, :, :n_ticks + 1].max((0, 1)), r["out_done"][:n_ticks + 1]), r["fin_arr"][:n_ticks + 1])
            new = np.zeros(n_ticks, np.int64); prev = 0
            for t in range(1, n_ticks + 1):
                nt = max(fin[t - 1] + barrier_cyc, prev + 1)
                if not fullspeed: nt = max(nt, t * T_cyc)                         # real time: not before the input exists
                new[t - 1] = nt; prev = nt
            if np.array_equal(new, out_ticks): break
            ticks = np.broadcast_to(new[None, None, :], (L, G, n_ticks)).copy(); out_ticks = new
    if ref_us and (T_us % int(ref_us) or T_us // int(ref_us) > 32): raise ValueError("refractory window must divide the step, at most 32 per step")
    bp = schedule == "event" and sync == "wire" and (cosim if cosim is not None else fifo > 0)
    if bp:
        r = _simulate_cosim(mp, t_us, ch, f_mhz, T_us, n_ticks, units, H, P, div, ref_us, paced, fifo, port_buf, out_reg, trace, ref_out, gap=port_gap)
    else:
        r = _simulate(mp, t_us, ch, schedule, f_mhz, T_us, n_ticks, units, H, P, ticks, out_ticks, fullspeed, trace, sync, ph[0], period, div,
                      ref_us, paced, gap=port_gap)
    misses = int(r["late"]) if schedule == "step" else 0                          # spikes that missed the tick that should integrate them
    rec_cyc = r["rec_cyc"]
    makespan = int(rec_cyc.max()) if len(rec_cyc) else 1
    if bp: makespan = max(makespan, r["last_idle"])                               # until the last unit goes idle
    res = dict(time_us=rec_cyc / f_mhz, o=r["o"], spikes=r["spikes"], ops=r["ops"], busy_core=r["busy_core"], out_busy=r["out_busy"],
               work=r["work"], out_work=r["out_work"], ticks=out_ticks, misses=misses, fifo_max=r["fifo_max"], makespan=makespan,
               n_events=int(len(t_us)), T_cyc=T_cyc, period=period, port_cum=np.concatenate([[0], np.cumsum(port_gap)]).astype(np.int64))
    if bp: res.update(drops=r["drops"], stall_cycles=r["stall_cycles"], fifo_out_max=r["fifo_out_max"], last_idle=r["last_idle"])
    if trace:
        cyc, chs, tc, tk, tg = r["stream"]
        res.update(trace=r["trace"] if r["trace"] is not None else np.zeros((0, 4), np.int64), stream=(cyc, chs, tk, tg), rec_cyc=rec_cyc)
    return res


# ----------------------------------------------------------------------------------------------- evaluation
_G = {}


def _init(model_path, dataset, split, cfg):
    d = load_int(model_path); _G["mp"] = Mapping(d); _G["data"] = EventData(dataset, split); _G["cfg"] = cfg
    sm = d.get("config", {}).get("stim_ms"); _G["data"].stim_us = int(sm * 1000) if sm is not None else None
    lim = cfg.get("in_limit_us"); _G["data"].limit_us = lim; _G["data"].frame_events = cfg.get("frame_events")
    _G["data"].set_delays(d.get("config", {}).get("in_delays_us"))


def _work_curves(mp, r, t_us, sch, sync, paced, f_mhz, T_us, grid_us):
    """Cumulative work of one sample on the time grid, for accuracy against operations and energy: rows into hidden
    cores (input events x their first-layer cores, plus hidden spikes that feed the next layer; each row is core-size
    synaptic operations and core-size busy cycles), spikes into the output unit (n_out operations each), and core scans
    (leak or threshold passes, core-size cycles each: stepped 1 per first-layer core and 2 per deeper core per tick,
    event-driven 1 per core per tick of the wire). Counted when the causing spike leaves or the tick comes, so
    queueing delays are ignored."""
    g = np.floor(np.asarray(grid_us, np.float64) * f_mhz).astype(np.int64)
    def upto(times, weights=None):
        o = np.argsort(times, kind="stable"); tt = np.asarray(times)[o]
        c = np.cumsum(np.ones(len(tt)) if weights is None else np.asarray(weights, np.float64)[o])
        k = np.searchsorted(tt, g, side="right")
        return np.where(k > 0, c[np.maximum(k - 1, 0)] if len(c) else 0.0, 0.0)
    cyc, chs = r["stream"][0], r["stream"][1]
    ev = chs >= 0; fan = (mp.route_core[0] >= 0).sum(1)
    rows = upto(cyc[ev], fan[chs[ev]])
    tr = r["trace"]
    rows = rows + upto(tr[tr[:, 0] < mp.L - 1, 3])
    out_sp = upto(tr[tr[:, 0] == mp.L - 1, 3])
    n_ticks = len(r["ticks"]); tt = np.arange(1, n_ticks + 1, dtype=np.int64)
    if sch == "step":
        tick_c = np.asarray(r["ticks"], np.int64); per_tick = mp.G * (1 + 2 * (mp.L - 1))
    elif paced:
        tick_c = r["port_cum"][np.searchsorted(np.sort(t_us), tt * T_us, side="left")]; per_tick = mp.G * mp.L
    else:
        tick_c = np.ceil(tt * T_us * f_mhz).astype(np.int64); per_tick = mp.G * mp.L
    scans = upto(tick_c) * per_tick
    return rows, out_sp, scans


def _one(i):
    mp, data, cfg = _G["mp"], _G["data"], _G["cfg"]
    t, ch, y = data.get(i)
    res = {}
    for key, spec in cfg["configs"].items():
        sch, sync, fs = spec[:3]; paced = spec[3] if len(spec) > 3 else 0
        r = run_sample(mp, t, ch, sch, cfg["f_mhz"], cfg["T_us"], cfg["t_end_us"], cfg["units"], cfg["H"], cfg["P"],
                       sync=sync, fullspeed=fs, phases=cfg["phases"], barrier_cyc=cfg["barrier_cyc"], div=cfg["div"].get(key),
                       ref_us=cfg["ref"].get(key), paced=paced, tick_cyc=cfg.get("tick_cyc"), fifo=cfg.get("fifo", 0),
                       port_buf=cfg.get("port_buf", 64), out_reg=cfg.get("out_reg"), trace=bool(cfg.get("work_curves")),
                       ref_out=cfg.get("ref_out", {}).get(key, True), n_in0=data.n_in0 if data.delays_us else None)
        og = sample_on_grid(r["time_us"], r["o"], cfg["grid_us"])
        wc = _work_curves(mp, r, t, sch, sync, paced, cfg["f_mhz"], cfg["T_us"], cfg["grid_us"]) if cfg.get("work_curves") else None
        top = np.sort(og, axis=1)
        correct_t = (og.argmax(1) == y) & (top[:, -1] > top[:, -2])                 # a tie between the top two is no decision
        md = margin_decisions(r["time_us"], r["o"], cfg["margins"])
        L, G = r["busy_core"].shape
        span = max(1, r["makespan"])
        step = {}
        if sch == "step":
            tk = r["ticks"]; lens = np.diff(np.concatenate([[0], tk])).astype(np.float64)      # length of every hardware step
            w = r["work"][:, :, 1:len(tk) + 1].astype(np.float64)                                # work at every tick
            step = dict(u_step=(w / np.maximum(lens[None, None, :], 1)).mean((1, 2)), tick_gap=float(lens.mean()),
                        idle_num=(w.max(1) * G - w.sum(1)).sum(1), idle_den=(w.max(1) * G).sum(1),
                        lens=lens.astype(np.int64), wsum=w.sum((0, 1)).astype(np.int64), wmax=w.max((0, 1)).astype(np.int64))
            o_t = np.asarray(r["o"])[:len(tk)]; srt = np.sort(o_t, axis=1)                       # the decision after every tick
            step["ok"] = ((o_t.argmax(1) == y) & (srt[:, -1] > srt[:, -2])).astype(np.uint8)
        res[key] = dict(correct_t=correct_t.astype(np.uint8), margin=[(tm, p == y, p >= 0) for tm, p, ix in md],
                        spikes=r["spikes"].tolist(), ops=r["ops"], misses=r["misses"], fifo_max=int(r["fifo_max"].max()),
                        u_layer=(r["busy_core"].sum(1) / (G * span)).tolist(),
                        u_chip=float((r["busy_core"].sum() + r["out_busy"]) / ((L * G + 1) * span)), makespan_us=span / cfg["f_mhz"],
                        n_events=r["n_events"], drops=r.get("drops", 0), stall_cycles=r.get("stall_cycles", 0),
                        fifo_out_max=r.get("fifo_out_max", 0), u_core_mean=float(r["busy_core"].mean() / span),
                        u_core_max=float(r["busy_core"].max() / span), out_busy=int(r["out_busy"]), busy_total=int(r["busy_core"].sum()),
                        work=None if wc is None else np.stack(wc).astype(np.float32), **step)
    return i, y, res


def check_step_exact(model_path, dataset, n, f_mhz, units, H, P):
    """System model, step schedule, against the torch pipelined forward: the output at every tick must match."""
    import torch
    from .model import QSNN
    from .data import bin_indices
    d = load_int(model_path); cfg0 = d.get("config", {}); mp = Mapping(d)
    m = QSNN(d["n_in"], d["hidden"], d["n_out"], d["wbits"], d["theta"], d["k_leak"], d["k_out"], d["vbits"], d["obits"],
             leak_every=int(d.get("leak_every", 1)), wscale=int(d.get("wscale", 1)), spiking_out=mp.spiking_out, theta_out=max(1, mp.theta_out))
    with torch.no_grad():
        for l, W in enumerate(d["W"]): m.W[l].copy_(torch.from_numpy(W.astype(np.float32)))
    m.snapped = True
    data = EventData(dataset, "test"); sm = cfg0.get("stim_ms"); data.stim_us = int(sm * 1000) if sm is not None else None
    data.limit_us = cfg0.get("in_limit_us"); data.frame_events = cfg0.get("frame_events"); data.set_delays(cfg0.get("in_delays_us"))
    dt = int(cfg0["dt_us"]); T = int(cfg0["T"]); bad = 0
    for i in range(n):
        t, ch, y = data.get(i)
        b, c = bin_indices(t, ch, dt, T)
        x = torch.zeros(1, T, d["n_in"]); x.index_put_((torch.zeros(len(b), dtype=torch.long), torch.from_numpy(b), torch.from_numpy(c)), torch.ones(len(b)), accumulate=True)
        with torch.no_grad(): o, _ = m(x, "pipelined", 1)
        r = run_sample(mp, t, ch, "step", f_mhz, dt, T * dt, units, H, P)
        if not np.array_equal(r["o"][:T], o[0].numpy().astype(np.int64)): bad += 1
    return {"n": n, "step_mismatching_samples": bad}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dataset", default="nmnist32")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_samples", type=int, default=0)
    ap.add_argument("--random_subset", action="store_true", help="with --n_samples: a fixed random subset (seed 0) instead of the first ones "
                    "(test sets can be ordered by class)")
    ap.add_argument("--configs", default="step:aligned,step:phases,step:barrier,event:aligned,event:phases,event:lazy",
                    help="schedule:sync pairs; step with aligned|phases|barrier, event with aligned|phases (local timers), lazy (step tags) or wire (broadcast tick: from the input in ip, from a timer in rt)")
    ap.add_argument("--speeds", default="rt", help="comma list of rt (real time), fs (full speed), ip (input-paced: one event per --in_period cycles)")
    ap.add_argument("--in_period", type=int, default=256, help="ip: cycles between input events (256 = a first-layer core can never fall behind)")
    ap.add_argument("--tick_us", type=float, default=1000.0, help="ip: least time between two global ticks (step:aligned), in us at --f_mhz: "
                    "TrueNorth's 1 ms, which covers the worst case of a core (66,310 cycles with delivery) at 67 MHz")
    ap.add_argument("--f_mhz", type=float, default=3.4, help="the smallest round clock that meets a core's worst case (66,310 cycles per 20 ms step with delivery)")
    ap.add_argument("--units", default="1,1,1,1,1", help="neuron units per core, per hidden layer")
    ap.add_argument("--H", type=int, default=2); ap.add_argument("--P", type=int, default=2)
    ap.add_argument("--barrier_cyc", type=int, default=16, help="cycles a global barrier adds to every step")
    ap.add_argument("--phase_seed", type=int, default=0)
    ap.add_argument("--sizing", type=float, default=0.0, help="also run step:aligned and event:lazy (real time) on hardware sized for "
                    "this percentile of the per-step load (e.g. 99): one clock for all cores, set by the busiest layer, measured on training samples")
    ap.add_argument("--sizing_samples", type=int, default=1000)
    ap.add_argument("--lazy_ref_ms", default="", help="extra event:lazy runs with these refractory windows in ms (e.g. 5,1)")
    ap.add_argument("--fifo", type=int, default=64, help="event:wire: entries of every core's and the output unit's input FIFO, with "
                    "back-pressure (default 64); 0 = unlimited, the layer-by-layer model")
    ap.add_argument("--port_buf", type=int, default=64, help="event:wire, real time: events the input port holds; an event arriving "
                    "while it is full is dropped and counted")
    ap.add_argument("--out_reg", type=int, default=-1, help="event:wire: spikes a core may hold for the router (-1: 1 with --fifo, "
                    "unlimited without; 0 = unlimited)")
    ap.add_argument("--wire_ref", action="store_true", help="ip and rt: extra event:wire run with a refractory of one step (one spike per neuron between ticks)")
    ap.add_argument("--wire_ref_hidden", action="store_true", help="ip and rt: extra event:wire run with that refractory in the hidden layers only, "
                    "not in the output unit (config *_rsteph)")
    ap.add_argument("--margins", default=None, help="decision margins (default 256,...,8192 for integrators, 1,2,3,4,6,8 spikes for a spiking output)")
    ap.add_argument("--grid_ms", type=float, default=1.0)
    ap.add_argument("--work_curves", action="store_true", help="also record the cumulative rows, output spikes and scans on the time grid "
                    "(for accuracy against operations and energy; runs every sample with a spike trace)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check_exact", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump_steps", type=Path, default=None, help="npz with every sample's step lengths, busy cycles and decision "
                    "per step (stepped configs) and makespan and busy cycles (all configs), for the scaling study")
    a = ap.parse_args()
    d = load_int(a.model); cfg0 = d.get("config", {}); mp = Mapping(d)
    T_us = int(cfg0["dt_us"]) * int(d.get("leak_every", 1)); t_end_us = int(cfg0.get("t_end_ms", 100) * 1000)
    units = [int(u) for u in a.units.split(",")]
    n_ticks = t_end_us // T_us; T_cyc = int(round(T_us * a.f_mhz)); P_wc = mp.core * (mp.core + 2) + 64
    rng = np.random.default_rng(a.phase_seed)
    phases = rng.integers(0, T_cyc, size=(mp.L + 1, mp.G))                        # local tick phase of every core (cycles)
    configs = {}
    fs_ok = {("step", "aligned"), ("step", "barrier"), ("event", "lazy")}          # meaningful without a wall clock
    ip_ok = {("step", "aligned"), ("step", "barrier"), ("event", "wire"), ("event", "lazy")}   # input-paced: global tick, barrier, event-driven
    tick_cyc = int(round(a.tick_us * a.f_mhz))
    refs = {}
    for sp in a.speeds.split(","):
        for cs in a.configs.split(","):
            sch, sync = cs.split(":")
            if sp == "fs" and (sch, sync) not in fs_ok: continue
            if sp == "ip" and (sch, sync) not in ip_ok: continue
            key = f"{sch}_{sync}_{sp}"
            configs[key] = (sch, sync, sp != "rt", a.in_period if sp == "ip" else 0)
            if (sp == "ip" and sch == "event") or (sch, sync) == ("event", "wire"): refs[key] = 0   # event-driven baseline: no refractory
    for rm in [x for x in a.lazy_ref_ms.split(",") if x]:
        for sp in a.speeds.split(","):
            key = f"event_lazy_{sp}_r{rm}"; configs[key] = ("event", "lazy", sp != "rt", a.in_period if sp == "ip" else 0); refs[key] = int(float(rm) * 1000)
    for sp in (a.speeds.split(",") if a.wire_ref else []):
        if sp == "fs": continue
        key = f"event_wire_{sp}_rstep"; configs[key] = ("event", "wire", sp != "rt", a.in_period if sp == "ip" else 0); refs[key] = T_us
    ref_out = {}
    for sp in (a.speeds.split(",") if a.wire_ref_hidden else []):
        if sp == "fs": continue
        key = f"event_wire_{sp}_rsteph"; configs[key] = ("event", "wire", sp != "rt", a.in_period if sp == "ip" else 0); refs[key] = T_us; ref_out[key] = False
    divs = {}
    if a.sizing:
        tr_data = EventData(a.dataset, "train"); tr_data.limit_us = cfg0.get("in_limit_us"); tr_data.frame_events = cfg0.get("frame_events")
        tr_data.set_delays(cfg0.get("in_delays_us"))
        idx = np.random.default_rng(1).choice(len(tr_data), min(a.sizing_samples, len(tr_data)), replace=False)
        for sch, sync in (("step", "aligned"), ("event", "lazy")):
            W = []
            for i in idx:
                t, ch, _ = tr_data.get(int(i))
                r = run_sample(mp, t, ch, sch, a.f_mhz, T_us, t_end_us, units, a.H, a.P, sync=sync)
                W.append(r["work"][:, :, :n_ticks + 1])
            W = np.stack(W)                                                        # (samples, L, G, steps) base cycles
            q = [float(np.percentile(W[:, l], a.sizing)) for l in range(mp.L)]
            d = [max(1, int(T_cyc // max(x, 1.0))) for x in q]
            g = min(d)                                                             # one clock for all cores: the busiest layer decides
            d = [g] * (mp.L + 1)
            key = f"{sch}_{sync}_rt_typ"; configs[key] = (sch, sync, False); divs[key] = d
            print(f"sizing {sch}:{sync} on {len(idx)} training samples: p{a.sizing:g} step load per layer {[round(x) for x in q]} cycles "
                  f"of {T_cyc}; one clock for all cores: {a.f_mhz / d[0]:g} MHz (divider {d[0]})", flush=True)
    end_us = max(t_end_us, (n_ticks + 2) * P_wc / a.f_mhz if "fs" in a.speeds else 0)
    if "ip" in a.speeds:                                                           # the longest sample's input plus the deepest tick lag
        ev_max = int(np.diff(EventData(a.dataset, a.split).offsets).max()) * (1 + len(cfg0.get("in_delays_us") or []))
        end_us = max(end_us, (ev_max * a.in_period + (n_ticks + 2) * max(P_wc, tick_cyc)) / a.f_mhz)
    grid = np.arange(a.grid_ms * 1000, end_us + 1, a.grid_ms * 1000)
    cfg = dict(configs=configs, f_mhz=a.f_mhz, T_us=T_us, t_end_us=t_end_us, units=units, H=a.H, P=a.P, grid_us=grid,
               margins=[int(m) for m in (a.margins or ("1,2,3,4,6,8" if mp.spiking_out else "256,512,1024,2048,4096,8192")).split(",")], phases=phases, barrier_cyc=a.barrier_cyc, in_limit_us=cfg0.get("in_limit_us"),
               frame_events=cfg0.get("frame_events"), div=divs, ref=refs, ref_out=ref_out, tick_cyc=tick_cyc, fifo=a.fifo, port_buf=a.port_buf, work_curves=a.work_curves,
               out_reg=a.out_reg if a.out_reg >= 0 else None)
    t0 = time.time()
    exact = check_step_exact(a.model, a.dataset, a.check_exact, a.f_mhz, units, a.H, a.P) if a.check_exact else None
    print("step schedule against torch pipelined:", exact, flush=True)
    data = EventData(a.dataset, a.split); n = a.n_samples or len(data)
    ids = np.sort(np.random.default_rng(0).choice(len(data), n, replace=False)) if (a.random_subset and n < len(data)) else np.arange(n)
    with Pool(a.workers, initializer=_init, initargs=(a.model, a.dataset, a.split, cfg)) as pool:
        results = pool.map(_one, [int(i) for i in ids], chunksize=8)
    out = {"model": str(a.model), "dataset": a.dataset, "n_samples": n, "random_subset": bool(a.random_subset), "f_mhz": a.f_mhz, "units": units, "H": a.H, "P": a.P, "dividers": divs,
           "T_us": T_us, "t_end_us": t_end_us, "P_wc": P_wc, "barrier_cyc": a.barrier_cyc, "phase_seed": a.phase_seed,
           "in_period": a.in_period if "ip" in a.speeds else None, "tick_cyc": tick_cyc if "ip" in a.speeds else None,
           "exactness": exact, "fifo": a.fifo, "port_buf": a.port_buf, "out_reg": a.out_reg, "configs": {}}
    for key in configs:
        R = [r[2][key] for r in results]
        ct = np.stack([x["correct_t"] for x in R]).astype(np.float64)
        marg = {}
        for mi, m in enumerate(cfg["margins"]):
            rows = [x["margin"][mi] for x in R]
            dec = np.array([r[2] for r in rows]); cor = np.array([r[1] for r in rows]); tm = np.array([r[0] for r in rows])
            marg[str(m)] = {"accuracy": float(cor.mean()), "decided": float(dec.mean()),
                            "latency_ms_mean": float(tm[dec].mean() / 1000) if dec.any() else None,
                            "latency_ms_median": float(np.median(tm[dec]) / 1000) if dec.any() else None}
        rec = {"accuracy_final": float(ct[:, -1].mean()),
               "accuracy_vs_time": {"grid_ms": (grid / 1000).tolist(), "accuracy": ct.mean(0).tolist()},
               "margin_decisions": marg,
               "spikes_per_sample_per_layer": np.mean([x["spikes"] for x in R], 0).tolist(),
               "ops_per_sample": float(np.mean([x["ops"] for x in R])),
               "utilisation_per_layer": np.mean([x["u_layer"] for x in R], 0).tolist(),
               "chip_utilisation": float(np.mean([x["u_chip"] for x in R])),
               "makespan_ms_mean": float(np.mean([x["makespan_us"] for x in R]) / 1000),
               "deadline_misses": int(sum(x["misses"] for x in R)), "fifo_max": int(max(x["fifo_max"] for x in R)),
               "fifo_out_max": int(max(x["fifo_out_max"] for x in R)), "dropped_events_total": int(sum(x["drops"] for x in R)),
               "dropped_events_per_sample": float(np.mean([x["drops"] for x in R])),
               "stall_cycles_per_sample": float(np.mean([x["stall_cycles"] for x in R])),
               "events_per_sample": float(np.mean([x["n_events"] for x in R])),
               "core_utilisation_mean": float(np.mean([x["u_core_mean"] for x in R])),
               "core_utilisation_max": float(np.mean([x["u_core_max"] for x in R])),
               "busy_core_cycles_per_sample": float(np.mean([x["busy_total"] for x in R])),
               "output_busy_cycles_per_sample": float(np.mean([x["out_busy"] for x in R]))}
        if a.work_curves:
            W = np.mean([x["work"] for x in R], 0)
            rec["work_vs_time"] = {"rows": W[0].tolist(), "output_spikes": W[1].tolist(), "core_scans": W[2].tolist(),
                                   "core_size": mp.core, "n_out": mp.n_out}
        if configs[key][0] == "step":
            rec["utilisation_per_layer_per_step"] = np.mean([x["u_step"] for x in R], 0).tolist()
            rec["tick_gap_cyc_mean"] = float(np.mean([x["tick_gap"] for x in R]))
            num = np.sum([x["idle_num"] for x in R], 0); den = np.sum([x["idle_den"] for x in R], 0)
            rec["barrier_idle_per_layer"] = (num / np.maximum(den, 1)).tolist()
        out["configs"][key] = rec
        best = max(marg.items(), key=lambda kv: kv[1]["accuracy"])
        print(f"{key:20s} final {rec['accuracy_final']:.4f}  best decision {best[1]['accuracy']:.4f} at {best[1]['latency_ms_mean']} ms (margin {best[0]})  "
              f"chip utilisation {rec['chip_utilisation']:.4f}  makespan {rec['makespan_ms_mean']:.1f} ms  misses {rec['deadline_misses']}  "
              f"fifo {rec['fifo_max']}/{rec['fifo_out_max']}  drops {rec['dropped_events_total']}  stalls {rec['stall_cycles_per_sample']:.0f}  "
              f"spikes {round(sum(rec['spikes_per_sample_per_layer']))}"
              + (f"  idle behind per-layer barrier {[round(x, 3) for x in rec['barrier_idle_per_layer']]}" if 'barrier_idle_per_layer' in rec else ''), flush=True)
    if a.dump_steps:
        dump = {"f_mhz": a.f_mhz, "n_cores": mp.L * mp.G, "labels": np.array([r[1] for r in results])}
        for key in configs:
            R = [r[2][key] for r in results]
            dump[f"{key}_makespan_cyc"] = np.array([x["makespan_us"] * a.f_mhz for x in R]); dump[f"{key}_busy"] = np.array([x["busy_total"] for x in R])
            if configs[key][0] == "step":
                for f in ("lens", "wsum", "wmax", "ok"): dump[f"{key}_{f}"] = np.stack([x[f] for x in R])
        a.dump_steps.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(a.dump_steps, **dump)
    path = a.out or (ROOT / "results" / "system" / f"{a.model.parent.name}_{a.split}_{'_'.join(a.speeds.split(','))}.json")
    path.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(path, "w"), indent=1)
    print(f"saved {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
