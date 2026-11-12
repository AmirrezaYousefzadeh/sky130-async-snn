#!/usr/bin/env python3
"""Short scenarios for the SDF-annotated runs on the hardened layout (phase 2): the first 100 warm-up spikes (with
their ticks) of each long scenario, then a shorter window (250 spikes, 50 scans, 5,000 idle cycles via IDLE_CYCLES).
Expected outputs from the same reference model."""
import sys
from pathlib import Path
sys.argv = [sys.argv[0], "unused"]
import importlib.util
spec = importlib.util.spec_from_file_location("gc", Path(__file__).with_name("gen_core.py")); gc = importlib.util.module_from_spec(spec); spec.loader.exec_module(gc)
def read(name):
    return [((int(l, 16) >> 8) & 3, int(l, 16) & 0xff) for l in Path(gc.HERE / f"stim/{name}.hex").read_text().split()]
for name, n_win, kw in (("spike", 250, {}), ("add", 250, dict(mode_event=0)), ("leak", 50, {}), ("thr", 50, {}), ("idle", 0, {})):
    cm = read(name); m1 = cm.index((3, 0))
    warm, win = cm[:m1], cm[m1 + 1:-1]
    k = next(i for i in range(len(warm)) if sum(1 for c in warm[:i] if c[0] == 0) >= 100)
    short = warm[:k] + [(3, 0)] + win[:n_win] + [(3, 0)]
    print(name + "_s", gc.write_stim(name + "_s", short, **kw))
