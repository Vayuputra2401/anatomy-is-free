"""
Mechanism analysis for the four shift routings (paper Sec. 5).

Answers *why* anatomical routing beats a hop-based one, using only the
precomputed integer buffers — no data, no checkpoints, no training.

For each shift_type we measure, over all (channel, joint) pairs:

  reach      mean graph-geodesic distance d_G(v, sigma_c(v)) actually bridged
  d>=2       fraction of routes that bridge more than one hop, i.e. that reach
             beyond what a single K=3 graph convolution already sees
  bilateral  fraction of routes landing exactly on the joint's mirror partner
  moved      fraction of routes that are not the identity
  RF1        mean effective 1-layer receptive field: |{sigma_c(v)} u N_1(v)|,
             the joints a single GCN layer can see after the shift

The ordering these produce is the same as the measured accuracy ordering
(Table: none 82.8, random 82.6, full_body 83.4, anatomical 84.6), which is the
point: the gain tracks *where* features are routed, not how far.

Usage:
    python scripts/analyze_shift_mechanism.py [--channels 64] [--latex]
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')

from src.models.blocks.body_region_shift import compute_shift_indices  # noqa: E402
from src.models.blocks.katp import (  # noqa: E402
    NUM_NODES, SKELETON_EDGES, partition_for,
)

SHIFT_TYPES = ['none', 'random', 'full_body', 'anatomical']
PRETTY = {
    'none': 'No shift',
    'random': 'Random per-channel',
    'full_body': 'Full-body (Shift-GCN style)',
    'anatomical': 'Anatomical (BRASP, ours)',
}
# Accuracy from the shift-mechanism ablation (Nano, NTU-60 X-Sub, TLA/KD off).
ACC = {'none': 82.8, 'random': 82.6, 'full_body': 83.4, 'anatomical': 84.6}


def adjacency(layout):
    V = NUM_NODES[layout]
    A = np.zeros((V, V), dtype=np.float32)
    for a, b in SKELETON_EDGES[layout]:
        A[a, b] = A[b, a] = 1.0
    return A


def geodesics(A):
    """All-pairs hop distance by BFS."""
    V = A.shape[0]
    nbr = [np.where(A[v] > 0)[0] for v in range(V)]
    D = np.full((V, V), np.inf)
    for s in range(V):
        D[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in nbr[u]:
                if np.isinf(D[s, w]):
                    D[s, w] = D[s, u] + 1
                    q.append(w)
    return D


def mirror_map(layout):
    """joint -> its bilateral partner, from the KATP partition."""
    part = partition_for(layout)
    m = {}
    for name in (k for k in part if k.startswith('left_')):
        for a, b in zip(part[name], part['right_' + name[5:]]):
            m[a] = b
            m[b] = a
    return m


def region_map(layout):
    """joint -> merged anatomical region id (arms bilateral-merged, as in BRASP)."""
    part = partition_for(layout)
    out = {}
    for name, joints in part.items():
        base = name.split('_', 1)[1] if name.startswith(('left_', 'right_')) else name
        for j in joints:
            out[j] = base
    return out


def analyse(layout, channels):
    A = adjacency(layout)
    V = NUM_NODES[layout]
    D = geodesics(A)
    mirror = mirror_map(layout)
    region = region_map(layout)
    nbr1 = [set(np.where(A[v] > 0)[0]) | {v} for v in range(V)]

    rows = []
    for st in SHIFT_TYPES:
        idx = compute_shift_indices(torch.from_numpy(A), channels,
                                    shift_type=st, seed=0).numpy()
        dists, bilat, moved, same_reg, rf = [], 0, 0, 0, []
        for c in range(channels):
            for v in range(V):
                t = int(idx[c, v])
                d = D[v, t]
                dists.append(0.0 if np.isinf(d) else float(d))
                bilat += (mirror.get(v, -1) == t)
                moved += (t != v)
                if t != v:                       # identity is not a "routing"
                    same_reg += (region.get(v) == region.get(t))
                rf.append(len(nbr1[v] | {t}))
        n = channels * V
        n_moved = max(moved, 1)
        rows.append({
            'shift': st,
            'reach': float(np.mean(dists)),
            'ge2': 100.0 * float(np.mean(np.array(dists) >= 2)),
            'bilateral': 100.0 * bilat / n,
            'moved': 100.0 * moved / n,
            # share of *actual* routes that stay inside one anatomical region
            'same_region': 100.0 * same_reg / n_moved,
            'rf1': float(np.mean(rf)),
            'acc': ACC[st],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layout', default='ntu-rgb+d', choices=sorted(SKELETON_EDGES))
    ap.add_argument('--channels', type=int, default=64)
    ap.add_argument('--latex', action='store_true', help='emit a LaTeX tabular')
    args = ap.parse_args()

    rows = analyse(args.layout, args.channels)

    if args.latex:
        print(r'\begin{tabular}{lrrrrr}')
        print(r'\toprule')
        print(r'Routing rule & Reach & Moved & In-region & RF$_1$ & Acc. \\')
        print(r' & (hops) & (\%) & (\%) & (joints) & (\%) \\')
        print(r'\midrule')
        for r in rows:
            name = PRETTY[r['shift']]
            if r['shift'] == 'anatomical':
                name = r'\textbf{' + name + '}'
            reg = '--' if r['shift'] == 'none' else f"{r['same_region']:.1f}"
            print(f"{name} & {r['reach']:.2f} & {r['moved']:.0f} & {reg} & "
                  f"{r['rf1']:.2f} & {r['acc']:.1f} \\\\")
        print(r'\bottomrule')
        print(r'\end{tabular}')
        return

    print(f"layout={args.layout}  V={NUM_NODES[args.layout]}  C={args.channels}\n")
    hdr = (f"{'routing':<30}{'reach':>7}{'d>=2%':>7}{'moved%':>8}{'sameReg%':>9}"
           f"{'bilat%':>8}{'RF1':>7}{'acc':>7}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{PRETTY[r['shift']]:<30}{r['reach']:>7.2f}{r['ge2']:>7.1f}"
              f"{r['moved']:>8.1f}{r['same_region']:>9.1f}{r['bilateral']:>8.1f}"
              f"{r['rf1']:>7.2f}{r['acc']:>7.1f}")


if __name__ == '__main__':
    main()
