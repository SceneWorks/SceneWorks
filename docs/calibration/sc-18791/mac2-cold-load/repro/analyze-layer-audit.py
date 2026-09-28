#!/usr/bin/env python3
"""Split render 1 vs render 2 in layer-audit.log and classify divergences.

Each line: `layer <i> <key> bytes=<n> cpu_fnv=<hex|err> cpu_sum=<f|err> gpu_sum=<f|err>`.
The TE pass restarts at `layer 0`, so a second `layer 0` block begins render 2.
"""
import re
import sys
from collections import OrderedDict

path = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Data/calibration/sc-18791/diagnostic/repro/layer-audit.log"
pat = re.compile(r"^layer (\d+) (\S+) bytes=(\d+) cpu_fnv=(\S+) cpu_sum=(\S+) gpu_sum=(\S+)$")

renders = []
current = OrderedDict()
last_layer = -1
for line in open(path):
    m = pat.match(line.strip())
    if not m:
        continue
    layer, key, nbytes, fnv, csum, gsum = m.groups()
    layer = int(layer)
    if layer < last_layer:
        renders.append(current)
        current = OrderedDict()
    last_layer = layer
    current[(layer, key)] = (int(nbytes), fnv, csum, gsum)
if current:
    renders.append(current)
print(f"renders found: {len(renders)}; tensors per render: {[len(r) for r in renders]}")


def close(a, b, rel=1e-3):
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    return abs(fa - fb) <= rel * max(1.0, abs(fa), abs(fb))


# Within-render CPU vs GPU disagreement (visibility signature).
for idx, r in enumerate(renders, 1):
    bad = [(k, v) for k, v in r.items() if not close(v[2], v[3])]
    print(f"render {idx}: {len(bad)} tensors where GPU sum != CPU sum")
    for (layer, key), (nbytes, fnv, csum, gsum) in bad[:10]:
        print(f"   layer {layer} {key} bytes={nbytes} cpu_sum={csum} gpu_sum={gsum}")

# Across renders: bytes-in-memory differ (CPU hash) vs GPU-only differ.
if len(renders) >= 2:
    r1, r2 = renders[0], renders[1]
    cpu_diff = [k for k in r1 if k in r2 and r1[k][1] != r2[k][1]]
    gpu_diff = [k for k in r1 if k in r2 and not close(r1[k][3], r2[k][3])]
    print(f"render1 vs render2: cpu_fnv differs on {len(cpu_diff)} tensors; gpu_sum differs on {len(gpu_diff)}")
    for k in (cpu_diff or gpu_diff)[:15]:
        print(f"   layer {k[0]} {k[1]}: r1 cpu_fnv={r1[k][1]} gpu_sum={r1[k][3]} | r2 cpu_fnv={r2[k][1]} gpu_sum={r2[k][3]}")
    if not cpu_diff and not gpu_diff:
        print("   weights identical across renders at both CPU and GPU views -> corruption is NOT in the weights")

# Activation lines: `layer <i> IN|OUT shape=[...] cpu_fnv=... cpu_sum=... gpu_sum=...`
apat = re.compile(r"^layer (\d+) (IN|OUT) shape=(\[[^\]]*\]) cpu_fnv=(\S+) cpu_sum=(\S+) gpu_sum=(\S+)$")
acts, cur, last = [], [], -1
for line in open(path):
    m = apat.match(line.strip())
    if not m:
        continue
    layer = int(m.group(1))
    if layer < last or (layer == last == 0 and m.group(2) == "IN" and cur):
        acts.append(cur); cur = []
    last = layer
    cur.append((layer, m.group(2), m.group(4), m.group(5), m.group(6)))
if cur:
    acts.append(cur)
print(f"activation passes: {len(acts)} sizes {[len(a) for a in acts]}")
# Two stack passes per render (positive then negative prompt): compare pass 1 with pass 3.
pairs = [(0, 2), (1, 3)] if len(acts) >= 4 else ([(0, 1)] if len(acts) >= 2 else [])
for p1, p2 in pairs:
    a1, a2 = acts[p1], acts[p2]
    print(f"comparing pass {p1+1} (render 1) with pass {p2+1} (render 2)")
    for x, y in zip(a1, a2):
        if x[2] != y[2]:
            print(f"FIRST ACTIVATION DIVERGENCE: layer {x[0]} {x[1]}  r1 fnv={x[2]} sum={x[3]}/{x[4]}  r2 fnv={y[2]} sum={y[3]}/{y[4]}")
            break
    else:
        print("   activations identical between these passes")
if acts:
    for idx, a in enumerate(acts, 1):
        bad = [t for t in a if not close(t[3], t[4])]
        print(f"pass {idx}: {len(bad)} activations where GPU sum != CPU sum")
