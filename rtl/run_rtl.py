#!/usr/bin/env python3
"""Export a trained network to the RTL's hex files, build the Verilator simulation, run samples, and compare every
spike (layer, core, neuron, cycle) and every output record with the cycle-accurate model (snn/system.py).

  python rtl/run_rtl.py export --model <model_int.npz> --dir rtl/build/<name>
  python rtl/run_rtl.py build  --dir rtl/build/<name>
  python rtl/run_rtl.py run    --model <model_int.npz> --dir rtl/build/<name> --samples 0,1,2 --schedule step
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snn.model import load_int
from snn.system import Mapping, run_sample, _input_stream
from snn.data import EventData

VERILATOR = os.environ.get("VERILATOR", "verilator")     # Verilator 5 (OSS CAD Suite 2026-10-24 was used)
RTL = Path(__file__).resolve().parent


def _hex8(x): return f"{int(x) & 0xFF:02x}"
def _hex32(x): return f"{int(x) & 0xFFFFFFFF:08x}"


def export(model: Path, out: Path):
    d = load_int(model); mp = Mapping(d); out.mkdir(parents=True, exist_ok=True)
    N, G, L = mp.core, mp.G, mp.L
    for l in range(L):
        for c in range(G):
            rows = np.zeros((256, N), np.int64); r = mp.rows[l][c]; rows[:r.shape[0]] = r
            (out / f"l{l}_c{c}.hex").write_text("\n".join(_hex8(w) for w in rows.ravel()) + "\n")
        n_ax = max(len(a) for a in mp.axons[l])
        (out / f"l{l}_cfg.hex").write_text("\n".join(_hex32(x) for x in (mp.theta[l], mp.k[l], n_ax)) + "\n")
    # input routing: up to two (core, axon) per channel
    rc, ra = mp.route_core[0], mp.route_axon[0]
    lines = []
    for ch in range(rc.shape[0]):
        w = 0
        for k in range(min(2, rc.shape[1])):
            if rc[ch, k] >= 0: w |= ((1 << 15) | (int(rc[ch, k]) << 8) | int(ra[ch, k])) << (16 * k)
        lines.append(_hex32(w))
    (out / "route_in.hex").write_text("\n".join(lines) + "\n")
    # routers between layers: entry (source core s, neuron n) -> target core and axon in the next layer
    for l in range(L - 1):
        rc, ra = mp.route_core[l + 1], mp.route_axon[l + 1]
        assert rc.shape[1] == 1, "hidden neurons must have one target"
        lines = [_hex32(((1 << 31) | (int(rc[i, 0]) << 8) | int(ra[i, 0])) if rc[i, 0] >= 0 else 0) for i in range(G * N)]
        (out / f"route_l{l}.hex").write_text("\n".join(lines) + "\n")
    lines = [_hex32((1 << 31) | i) for i in range(G * N)]                       # last layer -> output unit, axon = neuron
    (out / f"route_l{L - 1}.hex").write_text("\n".join(lines) + "\n")
    (out / "out_rom.hex").write_text("\n".join(_hex8(w) for w in mp.Wout.ravel()) + "\n")
    (out / "out_cfg.hex").write_text(_hex32(mp.k_out) + "\n")
    print(f"exported {model} to {out}: {L} layers x {G} cores, theta {mp.theta}, k {mp.k}, k_out {mp.k_out}")


def build(out: Path, threads: int = 4):
    obj = out / "obj"
    cmd = [VERILATOR, "--binary", "--timing", "-O3", "-Wno-fatal", "-Wno-lint", "-Wno-style", "--top-module", "tb",
           f"-GDIR=\"{out.resolve()}\"", "-j", str(threads), "--Mdir", str(obj), "-o", "Vtb",
           str(RTL / "snn_core.sv"), str(RTL / "snn_router.sv"), str(RTL / "snn_out.sv"), str(RTL / "snn_top.sv"), str(RTL / "tb" / "tb_snn.sv")]
    t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True)
    (out / "build.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-6000:]); raise SystemExit("verilator build failed")
    print(f"built {obj / 'Vtb'} ({time.time() - t0:.0f} s)")


def run(model: Path, out: Path, dataset: str, samples, schedule: str, f_mhz: float, units, H=2, P=2, ref_us=None):
    d = load_int(model); cfg0 = d.get("config", {}); mp = Mapping(d)
    T_us = int(cfg0["dt_us"]) * int(d.get("leak_every", 1)); t_end_us = int(cfg0.get("t_end_ms", 100) * 1000)
    n_ticks = int(t_end_us // T_us); T_cyc = int(round(T_us * f_mhz))
    data = EventData(dataset, "test")
    sm = cfg0.get("stim_ms"); data.stim_us = int(sm * 1000) if sm is not None else None
    data.limit_us = cfg0.get("in_limit_us"); data.frame_events = cfg0.get("frame_events")
    ok = True
    for i in samples:
        t, ch, y = data.get(i)
        keep = t < t_end_us; t, ch = t[keep], ch[keep]
        ref = run_sample(mp, t, ch, schedule, f_mhz, T_us, t_end_us, units, H, P, trace=True,
                         sync="lazy" if schedule == "event" else "aligned", ref_us=ref_us)
        cyc, chs, tk, tg = ref["stream"]
        sf = out / f"stream_{i}.txt"
        sf.write_text("".join(f"{int(c)} {int(k < 0)} {max(0, int(k))} {int(g)}\n" for c, k, g in zip(cyc, chs, tg)))
        end = int(max(n_ticks * T_cyc, ref["trace"][:, 3].max() if len(ref["trace"]) else 0, ref["rec_cyc"].max() if len(ref["rec_cyc"]) else 0)) + 64
        pref = out / f"rtl_{schedule}_{i}" if not ref_us else out / f"rtl_{schedule}_r{ref_us}_{i}"
        t0 = time.time()
        r = subprocess.run([str(out / "obj" / "Vtb"), f"+stream={sf}", f"+out={pref}", f"+mode={int(schedule == 'event')}",
                            f"+tcyc={T_cyc}", f"+nticks={n_ticks}", f"+end={end}"], capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode != 0: print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit("simulation failed")
        sp = np.loadtxt(f"{pref}_spikes.txt", dtype=np.int64, ndmin=2).reshape(-1, 4)
        rec = np.loadtxt(f"{pref}_out.txt", dtype=np.int64, ndmin=2).reshape(-1, 11)
        a = sp[np.lexsort((sp[:, 2], sp[:, 1], sp[:, 0], sp[:, 3]))]
        tr = ref["trace"]; b = tr[np.lexsort((tr[:, 2], tr[:, 1], tr[:, 0], tr[:, 3]))]
        same_sp = a.shape == b.shape and np.array_equal(a, b)
        ro = np.concatenate([ref["rec_cyc"][:, None], ref["o"]], 1)
        same_o = rec.shape == ro.shape and np.array_equal(rec, ro)
        ok &= same_sp and same_o
        print(f"sample {i} ({schedule}): {len(a)} RTL spikes vs {len(b)} model, identical {same_sp}; {len(rec)} records vs {len(ro)}, "
              f"identical {same_o}; {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''} ({dt:.1f} s)", flush=True)
        if not same_sp and len(a) and len(b):
            n = min(len(a), len(b)); k = np.flatnonzero((a[:n] != b[:n]).any(1))
            if len(k): print("   first difference: RTL", a[k[0]].tolist(), "model", b[k[0]].tolist())
        if not same_o and len(rec) and len(ro):
            n = min(len(rec), len(ro)); k = np.flatnonzero((rec[:n] != ro[:n]).any(1))
            if len(k): print("   first record difference: RTL", rec[k[0]].tolist(), "model", ro[k[0]].tolist())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "build", "run"])
    ap.add_argument("--model", type=Path); ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--dataset", default="nmnist32"); ap.add_argument("--samples", default="0")
    ap.add_argument("--schedule", default="step"); ap.add_argument("--f_mhz", type=float, default=4.0)
    ap.add_argument("--units", default="1,1,1,1,1"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ref_ms", type=float, default=None, help="event: refractory window (default: the step)")
    a = ap.parse_args()
    if a.cmd == "export": export(a.model, a.dir)
    elif a.cmd == "build": build(a.dir, a.threads)
    else:
        ok = run(a.model, a.dir, a.dataset, [int(s) for s in a.samples.split(",")], a.schedule, a.f_mhz, [int(u) for u in a.units.split(",")],
                 ref_us=int(a.ref_ms * 1000) if a.ref_ms else None)
        print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND"); raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
