#!/usr/bin/env python3
"""Event datasets in one compact format, plus micro-binning and timing-domain augmentation.

Prepared format (data/prepared/<dataset>_<split>.npz):
    t        int32  concatenated event times in microseconds (per sample, ascending)
    ch       int16  concatenated input channel index (N-MNIST: p*34*34 + y*34 + x; N-TIDIGITS18: cochlea channel)
    offsets  int64  sample i owns events offsets[i]:offsets[i+1]
    labels   int16
    n_in, n_classes, sensor (json string)

Datasets:
    nmnist     N-MNIST, 34x34x2 = 2312 inputs, 10 classes, 60k/10k, three 100 ms saccades per sample (Tonic download)
    ntidigits  N-TIDIGITS18 single digits, 64 cochlea channels, 11 classes, 2464/2486 (Tonic download)
    shd, ssc   Spiking Heidelberg Digits / Spiking Speech Commands, 700 cochlea-model channels, 20 / 35 classes (Tonic)
    dvsgesture DVS128 Gesture downsampled to 32x32x2 = 2048 inputs, 11 classes, 1.5 s slices of each recording (Tonic)
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "nmnist":    {"n_in": 34 * 34 * 2, "n_classes": 10, "sensor": [34, 34, 2]},
    "ntidigits": {"n_in": 64,          "n_classes": 11, "sensor": [64, 1, 1]},
    "shd":       {"n_in": 700,         "n_classes": 20, "sensor": [700, 1, 1]},   # Spiking Heidelberg Digits, 8156 / 2264
    "ssc":       {"n_in": 700,         "n_classes": 35, "sensor": [700, 1, 1]},   # Spiking Speech Commands, 75466 / 9981 valid / 20382
    "dvsgesture": {"n_in": 32 * 32 * 2, "n_classes": 11, "sensor": [32, 32, 2]},  # DVS128 Gesture 1077 / 264 recordings, see below
    "nmnist32":  {"n_in": 32 * 32 * 2, "n_classes": 10, "sensor": [32, 32, 2]},  # N-MNIST cropped to the central 32x32 pixels
}
# DVS128 Gesture: the 128x128 sensor is downsampled to 32x32 (x // 4, y // 4, events kept, none merged) and every recording
# (~6 s, one gesture repeated) is cut into non-overlapping DVS_SLICE_US slices from its first event; every full slice is
# one sample (an independent inference from reset). The slice and recording index are stored as `slice` and `rec`.
DVS_SLICE_US = 1_500_000
DVS_DOWN = 4


# ----------------------------------------------------------------------------------------------- preparation
def _iter_tonic(dataset: str, root: Path, train: bool):
    import tonic
    if dataset == "nmnist":
        ds = tonic.datasets.NMNIST(save_to=str(root), train=train)
        for ev, y in ds:
            t = ev["t"].astype(np.int64)
            ch = (ev["p"].astype(np.int64) * 34 * 34 + ev["y"].astype(np.int64) * 34 + ev["x"].astype(np.int64))
            yield t, ch, int(y)
    elif dataset == "ntidigits":
        ds = tonic.datasets.NTIDIGITS18(save_to=str(root), train=train, single_digits=True)
        for ev, y in ds:
            t = ev["t"].astype(np.int64)            # already in microseconds (Tonic multiplies by 1e6)
            ch = ev["x"].astype(np.int64)
            yield t, ch, int(y)
    elif dataset in ("shd", "ssc"):                 # Heidelberg cochlea-model spikes, times in seconds -> microseconds (Tonic)
        ds = tonic.datasets.SHD(save_to=str(root), train=train) if dataset == "shd" else \
             tonic.datasets.SSC(save_to=str(root), split="train" if train else "test")
        for ev, y in ds:
            t = ev["t"].astype(np.int64)
            ch = ev["x"].astype(np.int64)
            yield t, ch, int(y)
    elif dataset == "dvsgesture":
        ds = tonic.datasets.DVSGesture(save_to=str(root), train=train)
        side = 128 // DVS_DOWN
        for rec, (ev, y) in enumerate(ds):
            if len(ev) == 0: continue
            order = np.argsort(ev["t"], kind="stable"); ev = ev[order]
            t = ev["t"].astype(np.int64); t = t - t[0]
            ch = (ev["p"].astype(np.int64) * side * side + (ev["y"].astype(np.int64) // DVS_DOWN) * side
                  + ev["x"].astype(np.int64) // DVS_DOWN)
            for k in range(int(t[-1] // DVS_SLICE_US)):
                a, b = np.searchsorted(t, [k * DVS_SLICE_US, (k + 1) * DVS_SLICE_US])
                yield t[a:b] - k * DVS_SLICE_US, ch[a:b], int(y), rec, k
    else:
        raise ValueError(dataset)


def prepare(dataset: str, root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for split, train in (("train", True), ("test", False)):
        t0 = time.time()
        ts, chs, offs, labels, extra = [], [], [0], [], []
        n = 0
        for item in _iter_tonic(dataset, root, train):
            t, ch, y = item[:3]
            if len(item) > 3: extra.append(item[3:])
            order = np.argsort(t, kind="stable")
            t = t[order] - 0 if len(t) else t                           # keep absolute times (samples start near 0)
            ch = ch[order]
            keep = (t >= 0) & (ch >= 0) & (ch < DATASETS[dataset]["n_in"])
            t, ch = t[keep], ch[keep]
            ts.append(t.astype(np.int32)); chs.append(ch.astype(np.int16)); n += len(t); offs.append(n); labels.append(y)
            if len(labels) % 5000 == 0:
                print(f"  {split}: {len(labels)} samples, {n/1e6:.1f} M events, {time.time()-t0:.0f} s", flush=True)
        t_all = np.concatenate(ts); ch_all = np.concatenate(chs); offsets = np.array(offs, dtype=np.int64)
        labels = np.array(labels, dtype=np.int16)
        dur = (t_all[offsets[1:] - 1] - t_all[offsets[:-1]]).astype(np.float64)
        meta = dict(DATASETS[dataset]); meta.update(dataset=dataset, split=split, n_samples=int(len(labels)),
                    n_events=int(n), events_per_sample=float(n / len(labels)), duration_us_mean=float(dur.mean()),
                    duration_us_max=float(dur.max()), duration_us_p99=float(np.percentile(dur, 99)))
        more = {}
        if extra:
            ex = np.array(extra, dtype=np.int32); more = dict(rec=ex[:, 0], slice=ex[:, 1])
            meta.update(n_recordings=int(len(np.unique(ex[:, 0]))), slice_us=DVS_SLICE_US)
        np.savez(out / f"{dataset}_{split}.npz", t=t_all, ch=ch_all, offsets=offsets, labels=labels, meta=json.dumps(meta), **more)
        print(f"{dataset} {split}: {meta['n_samples']} samples, {n/1e6:.1f} M events, mean {meta['events_per_sample']:.0f} ev/sample, "
              f"duration mean {dur.mean()/1e3:.1f} ms max {dur.max()/1e3:.1f} ms ({time.time()-t0:.0f} s)", flush=True)


def derive_nmnist32(prepared: Path):
    """N-MNIST cropped to the central 32 x 32 pixels (the outermost row and column on each side are dropped), so that the
    input is 2048 = 8 x 256 channels: ch = p*1024 + (y-1)*32 + (x-1) for 1 <= x, y <= 32."""
    for split in ("train", "test"):
        t0 = time.time()
        d = np.load(prepared / f"nmnist_{split}.npz")
        t, offs, labels = d["t"], d["offsets"], d["labels"]
        ch = d["ch"].astype(np.int32)
        p = ch // (34 * 34); rem = ch - p * (34 * 34); y = rem // 34; x = rem - y * 34
        keep = (x >= 1) & (x <= 32) & (y >= 1) & (y <= 32)
        ch2 = (p * 1024 + (y - 1) * 32 + (x - 1))[keep].astype(np.int16)
        del p, rem, y, x, ch
        kept = np.concatenate([[0], np.cumsum(keep, dtype=np.int64)])
        offs2 = kept[offs]
        t2 = t[keep]
        n = len(labels); dur = (t2[np.maximum(offs2[1:] - 1, 0)] - t2[offs2[:-1]]).astype(np.float64)
        meta = json.loads(str(d["meta"])); meta.update(DATASETS["nmnist32"])
        meta.update(dataset="nmnist32", split=split, n_events=int(len(t2)), events_per_sample=float(len(t2) / n),
                    duration_us_mean=float(dur.mean()), duration_us_max=float(dur.max()), cropped_from="nmnist 34x34x2",
                    events_kept_fraction=float(keep.mean()))
        np.savez(prepared / f"nmnist32_{split}.npz", t=t2, ch=ch2, offsets=offs2, labels=labels, meta=json.dumps(meta))
        print(f"nmnist32 {split}: {n} samples, {len(t2)/1e6:.1f} M events ({100*keep.mean():.1f} % kept) ({time.time()-t0:.0f} s)", flush=True)


def input_groups(dataset: str, n_cores: int, core: int = 256):
    """Input channels of each first-layer core (core graph). The 32 x 32 x 2 vision sets are read in 8 patches of 16 (x)
    by 8 (y) pixels with both polarities, 256 channels each; inputs of at most `core` channels go whole to every core;
    wider inputs (SSC, SHD: 700 channels) are read in n_cores frequency bands of `core` channels with an equal stride."""
    if dataset in ("nmnist32", "dvsgesture"):
        return [np.array([p * 1024 + y * 32 + x for p in range(2) for y in range(py * 8, py * 8 + 8) for x in range(px * 16, px * 16 + 16)])
                for py in range(4) for px in range(2)]
    n_in = DATASETS[dataset]["n_in"]
    if n_in <= core:
        return [np.arange(n_in)]
    stride = (n_in - core) / max(1, n_cores - 1)
    return [np.arange(int(round(i * stride)), int(round(i * stride)) + core) for i in range(n_cores)]


# ----------------------------------------------------------------------------------------------- loading
class EventData:
    """One split in memory. get(i) -> (t_us int64, ch int64, label int)."""
    def __init__(self, dataset: str, split: str, prepared: Path = ROOT / "data" / "prepared"):
        d = np.load(prepared / f"{dataset}_{split}.npz")
        self.t, self.ch, self.offsets, self.labels = d["t"], d["ch"], d["offsets"], d["labels"].astype(np.int64)
        self.meta = json.loads(str(d["meta"]))
        self.n_in, self.n_classes = self.meta["n_in"], self.meta["n_classes"]
        self.stim_us = None          # if set, events at or after this time are dropped (e.g. the first N-MNIST saccade)
        self.limit_us = None         # if set, at most one event per channel per window of this length (the first is kept)
        self.frame_events = None     # if set, count-based time: every frame_events events (in time order) form one 1 ms step
        self.delays_us = []          # input memory: every event is repeated after each of these delays on its own copy of
        self.n_in0 = self.n_in       # the channels (channel + k * n_in0 for the k-th delay), as an input port with delay
                                     # lines would do; set with set_delays() so that n_in follows

    def set_delays(self, delays_us):
        """Delayed copies of the input (see delays_us): n_in becomes n_in0 x (1 + number of delays)."""
        self.delays_us = [int(d) for d in (delays_us or [])]
        self.n_in = self.n_in0 * (1 + len(self.delays_us))

    def __len__(self): return len(self.labels)

    def get(self, i: int):
        a, b = self.offsets[i], self.offsets[i + 1]
        if self.stim_us is not None:
            b = a + int(np.searchsorted(self.t[a:b], self.stim_us, side="left"))
        t, ch = self.t[a:b].astype(np.int64), self.ch[a:b].astype(np.int64)
        if self.limit_us:            # input preprocessing: the first event of every (window, channel)
            _, first = np.unique((t // int(self.limit_us)) * self.n_in + ch, return_index=True)
            first.sort(); t, ch = t[first], ch[first]
        if self.frame_events:        # recorded time is replaced by event count: frame k = events k*F .. k*F+F-1 -> [k ms, k+1 ms)
            F = int(self.frame_events); j = np.arange(len(t), dtype=np.int64)
            t = (j // F) * 1000 + ((j % F) * 1000) // F
        if self.delays_us:           # delayed copies: the same events again after every delay, on their own channels
            t = np.concatenate([t] + [t + d for d in self.delays_us]); ch = np.concatenate([ch + k * self.n_in0 for k in range(1 + len(self.delays_us))])
            order = np.argsort(t, kind="stable"); t, ch = t[order], ch[order]
        return t, ch, int(self.labels[i])

    def subset(self, indices):
        return Subset(self, np.asarray(indices, dtype=np.int64))


class Subset:
    """A view of selected samples of an EventData (same interface)."""
    def __init__(self, base: EventData, indices):
        self.base, self.indices = base, indices
        self.labels = base.labels[indices]
        self.meta, self.n_in, self.n_classes = base.meta, base.n_in, base.n_classes

    def __len__(self): return len(self.indices)

    def get(self, i: int): return self.base.get(int(self.indices[i]))


def train_val_split(data: EventData, val_frac: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data))
    n_val = int(round(val_frac * len(data)))
    return data.subset(np.sort(perm[n_val:])), data.subset(np.sort(perm[:n_val]))


# ----------------------------------------------------------------------------------------------- augmentation
class Augment:
    """Timing-domain augmentation on raw events (times in microseconds).
    jitter_us: Gaussian noise on every event time; warp: random speed factor in [1-warp, 1+warp];
    shift_us: random delay of the whole sample in [0, shift_us]; drop: probability of dropping an event."""
    def __init__(self, jitter_us=0.0, warp=0.0, shift_us=0.0, drop=0.0, seed=0):
        self.jitter_us, self.warp, self.shift_us, self.drop = jitter_us, warp, shift_us, drop
        self.rng = np.random.default_rng(seed)

    def __call__(self, t, ch):
        t = t.astype(np.float64)
        if self.warp > 0: t = t * self.rng.uniform(1 - self.warp, 1 + self.warp)
        if self.jitter_us > 0: t = t + self.rng.normal(0.0, self.jitter_us, size=t.shape)
        if self.shift_us > 0: t = t + self.rng.uniform(0.0, self.shift_us)
        if self.drop > 0:
            keep = self.rng.random(t.shape) >= self.drop
            t, ch = t[keep], ch[keep]
        t = np.maximum(t, 0.0)
        order = np.argsort(t, kind="stable")
        return t[order].astype(np.int64), ch[order]


# ----------------------------------------------------------------------------------------------- micro-binning
def bin_indices(t_us, ch, dt_us: int, T: int):
    """Events -> (bin, channel) pairs for events inside the window [0, T*dt_us)."""
    b = t_us // dt_us
    keep = b < T
    return b[keep], ch[keep]


class Batcher:
    """Yields dense micro-binned batches x (B, T, n_in) as float32 event counts on `device`, and labels."""
    def __init__(self, data: EventData, dt_us: int, T: int, batch_size: int, shuffle: bool, device,
                 augment: Augment | None = None, seed: int = 0, drop_last: bool = False):
        self.data, self.dt_us, self.T, self.bs, self.shuffle, self.device = data, dt_us, T, batch_size, shuffle, device
        self.augment, self.rng, self.drop_last = augment, np.random.default_rng(seed), drop_last

    def __len__(self):
        n = len(self.data)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def __iter__(self):
        import torch
        n = len(self.data)
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for s in range(0, n, self.bs):
            ids = idx[s:s + self.bs]
            if self.drop_last and len(ids) < self.bs: break
            bb, tt, cc, ys, tl = [], [], [], [], []
            for k, i in enumerate(ids):
                t, ch, y = self.data.get(i)
                if self.augment is not None: t, ch = self.augment(t, ch)
                b, c = bin_indices(t, ch, self.dt_us, self.T)
                bb.append(np.full(len(b), k, dtype=np.int64)); tt.append(b); cc.append(c); ys.append(y)
                tl.append(int(b.max()) if len(b) else 0)                 # last micro-bin with input = end of the stimulus
            bb, tt, cc = (torch.from_numpy(np.concatenate(a)).to(self.device) for a in (bb, tt, cc))
            x = torch.zeros(len(ids), self.T, self.data.n_in, device=self.device)
            x.index_put_((bb, tt, cc), torch.ones(len(bb), device=self.device), accumulate=True)
            yield x, torch.tensor(ys, device=self.device), ids, torch.tensor(tl, device=self.device)


# ----------------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--root", type=Path, default=ROOT / "data" / "raw"); p.add_argument("--out", type=Path, default=ROOT / "data" / "prepared")
    s = sub.add_parser("stats"); s.add_argument("--dataset", required=True, choices=list(DATASETS))
    s.add_argument("--prepared", type=Path, default=ROOT / "data" / "prepared")
    a = ap.parse_args()
    if a.cmd == "prepare":
        if a.dataset == "nmnist32": derive_nmnist32(a.out)
        else: prepare(a.dataset, a.root, a.out)
    else:
        for split in ("train", "test"):
            d = EventData(a.dataset, split, a.prepared)
            print(json.dumps(d.meta, indent=1))
            rate = d.meta["n_events"] / (d.meta["n_samples"] * d.meta["duration_us_mean"]) * 1e3
            print(f"  mean event rate {rate:.2f} events/ms; classes {np.bincount(d.labels).tolist()}")


if __name__ == "__main__":
    main()
