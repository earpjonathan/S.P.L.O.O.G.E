#!/usr/bin/env python
"""Scale-shape statistics straight from a Brush PLY.

Flatness is rotation invariant, so this needs no levelling, no camera frame and
no viewer conversion -- which makes it the right tool for asking whether
--flatten-strength did anything at all.
"""
import sys
import numpy as np

def read_ply_scales(path):
    with open(path, 'rb') as f:
        hdr, line = [], b''
        while True:
            line = f.readline()
            hdr.append(line.decode('ascii', 'replace').strip())
            if line.strip() == b'end_header':
                break
        n = 0; props = []
        for h in hdr:
            if h.startswith('element vertex'):
                n = int(h.split()[-1])
            elif h.startswith('property float'):
                props.append(h.split()[-1])
        if 'scale_0' not in props:
            raise SystemExit(f"{path}: no scale properties ({len(props)} float props)")
        data = np.frombuffer(f.read(n * len(props) * 4), dtype='<f4').reshape(n, len(props))
    idx = [props.index(f'scale_{i}') for i in range(3)]
    sc = np.exp(data[:, idx].astype(np.float64))
    # Brush leaves a handful of non-finite splats (~0.01%); np.median
    # propagates NaN, so one bad row poisons the whole statistic.
    keep = np.isfinite(sc).all(1)
    return sc[keep], n, len(sc) - keep.sum()

print(f"{'file':<34} {'splats':>9} {'flatness med':>13} {'p10':>7} {'p90':>7} "
      f"{'s_min med':>11} {'s_max med':>11}")
print('-'*100)
base = None
for p in sys.argv[1:]:
    sc, n, dropped = read_ply_scales(p)
    s = np.sort(sc, axis=1)
    flat = s[:, 0] / np.maximum(s[:, 2], 1e-12)
    med = np.median(flat)
    tag = p.split('/')[-2] + '/' + p.split('/')[-1]
    print(f"{tag:<34} {n:>9,} {med:>13.4f} {np.percentile(flat,10):>7.4f} "
          f"{np.percentile(flat,90):>7.4f} {np.median(s[:,0]):>11.5f} {np.median(s[:,2]):>11.5f}"
          + (f"   ({dropped} non-finite dropped)" if dropped else ""))
    if base is None:
        base = (med, np.median(s[:,0]))
    else:
        print(f"{'':<34} {'':>9} {'vs first:':>13} flatness x{med/base[0]:.3f}   "
              f"s_min x{np.median(s[:,0])/base[1]:.3f}")
