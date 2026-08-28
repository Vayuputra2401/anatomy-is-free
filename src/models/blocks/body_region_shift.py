"""
BodyRegionShift — Body-Region-Aware Spatial Shift (Idea F / BRASP).

Anatomically-partitioned channel shift for skeleton action recognition.
Channels are divided into four groups; each group's channels are shifted
only within the joints that belong to that anatomical region.

Cost: 0 learnable parameters, 0 FLOPs beyond a single torch.gather call.
The shift indices are precomputed at __init__ and stored as a buffer.

Reference: Experiment-LAST-Lite.md — Sections 2, 3.
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Body region definitions.
#
# These are *derived* from the NTU-25 bone hierarchy by KATP rather than written
# out by hand — see katp.py. The derivation reproduces the previously
# hand-listed indices exactly (asserted in tests/test_katp.py), so this is a
# provenance change, not a behavioural one:
#
#   left_arm  [4, 5, 6, 7, 21, 22]    shoulder → elbow → wrist → hand tip, thumb
#   right_arm [8, 9, 10, 11, 23, 24]  shoulder → elbow → wrist → hand tip, thumb
#   left_leg  [12, 13, 14, 15]        hip → knee → ankle → foot
#   right_leg [16, 17, 18, 19]        hip → knee → ankle → foot
#   torso     [0, 1, 2, 3, 20]        spine base → mid → shoulder → neck → head
#
# Call body_regions_for('nw-ucla') etc. to obtain the same structure for another
# skeleton layout without hand-specifying anything.
# ---------------------------------------------------------------------------
from .katp import partition_for as _katp_partition_for

BODY_REGIONS = dict(_katp_partition_for('ntu-rgb+d'))


def body_regions_for(layout: str) -> dict:
    """Derive the body-region partition for any layout known to KATP."""
    return dict(_katp_partition_for(layout))


def get_channel_groups(C: int) -> dict:
    """
    Return channel slice for each anatomical group.

    Allocation (proportional to group importance for action recognition):
      Arms   (left + right): C//4   = 25%  — fine manipulation
      Legs   (left + right): C//4   = 25%  — locomotion
      Torso:                 C//8   = 12.5% — posture
      Cross-body:            rest   = 37.5% — inter-limb coordination
    """
    arm_end   = C // 4          # 0  : arm_end   → arms
    leg_end   = C // 2          # arm_end : leg_end → legs
    torso_end = 5 * C // 8      # leg_end : torso_end → torso
    # torso_end : C → cross-body (37.5%)
    return {
        'arm':   slice(0, arm_end),
        'leg':   slice(arm_end, leg_end),
        'torso': slice(leg_end, torso_end),
        'cross': slice(torso_end, C),
    }


def compute_shift_indices(A: torch.Tensor, C: int,
                          shift_type: str = 'anatomical',
                          seed: int = 0) -> torch.Tensor:
    """
    Precompute per-channel joint-permutation indices for BodyRegionShift.

    For each channel c and joint v, shift_indices[c, v] holds the source
    joint that channel c at position v should read from.

    shift_type selects the routing rule:
      'anatomical' (default) — arms/legs/torso channels permuted within their
                               body region; cross-body via graph-neighbour
                               cycling (the BRASP prior).
      'full_body'            — Shift-GCN-style: every channel cycles graph
                               neighbours across the whole skeleton, ignoring
                               body-region grouping (hop-based, anatomy-free).
      'random'               — random per-channel joint permutation (seeded);
                               a structure-free control.
      'none' / 'identity'    — no shift (identity permutation).

    The last three are ablation controls used to isolate the anatomical prior
    (BMVC'26 rebuttal, Table 12: random 82.6, full-body 83.4, anatomical 84.6).

    Args:
        A:          (V, V) float adjacency matrix.  A[v, w] > 0 ↔ v,w connected.
        C:          Number of channels.
        shift_type: Routing rule (see above).
        seed:       RNG seed for the 'random' control (reproducibility).

    Returns:
        shift_indices: LongTensor (C, V)
    """
    V = A.shape[0]

    # ── Ablation controls (isolate the anatomical prior) ─────────────────────
    if shift_type in ('none', 'identity'):
        return torch.arange(V, dtype=torch.long).unsqueeze(0).expand(C, V).clone()
    if shift_type == 'random':
        g = torch.Generator().manual_seed(seed)
        return torch.stack([torch.randperm(V, generator=g) for _ in range(C)], dim=0)

    # Precompute graph neighbours (used by anatomical cross-body + full_body)
    A_np = A.numpy() if isinstance(A, torch.Tensor) else np.array(A)
    neighbors = [list(np.where(A_np[v] > 0)[0]) for v in range(V)]

    if shift_type == 'full_body':
        # Shift-GCN-style: every channel cycles graph neighbours across the whole
        # skeleton — hop-based, no body-region grouping.
        shift_indices = torch.zeros(C, V, dtype=torch.long)
        for c in range(C):
            for v in range(V):
                nbrs = neighbors[v]
                shift_indices[c, v] = v if len(nbrs) == 0 else nbrs[c % len(nbrs)]
        return shift_indices

    if shift_type != 'anatomical':
        raise ValueError(
            f"Unknown shift_type '{shift_type}' "
            "(expected 'anatomical' | 'full_body' | 'random' | 'none')")

    shift_indices = torch.zeros(C, V, dtype=torch.long)
    channel_groups = get_channel_groups(C)

    for group_name, ch_slice in channel_groups.items():
        ch_list = list(range(ch_slice.start, ch_slice.stop))

        if group_name == 'cross':
            # Shift across all joints: each channel cycles through graph neighbours
            for i, c in enumerate(ch_list):
                for v in range(V):
                    nbrs = neighbors[v]
                    if len(nbrs) == 0:
                        shift_indices[c, v] = v         # isolated joint → identity
                    else:
                        shift_indices[c, v] = nbrs[i % len(nbrs)]

        else:
            # Build the joint set for this group (arms = left + right combined)
            if group_name == 'arm':
                region_joints = BODY_REGIONS['left_arm'] + BODY_REGIONS['right_arm']
            elif group_name == 'leg':
                region_joints = BODY_REGIONS['left_leg'] + BODY_REGIONS['right_leg']
            else:  # torso
                region_joints = BODY_REGIONS['torso']

            region_set = set(region_joints)

            for i, c in enumerate(ch_list):
                for v in range(V):
                    if v in region_set:
                        # Shift within the region: cycle by offset i+1
                        idx = region_joints.index(v)
                        target = region_joints[(idx + i + 1) % len(region_joints)]
                        shift_indices[c, v] = target
                    else:
                        # Joint not in this region → identity (read from self)
                        shift_indices[c, v] = v

    return shift_indices  # (C, V)


# ---------------------------------------------------------------------------
# BodyRegionShift module
# ---------------------------------------------------------------------------
class BodyRegionShift(nn.Module):
    """
    Zero-parameter spatial channel shift structured by body anatomy.

    Splits channels into four groups (arm, leg, torso, cross-body) and
    permutes each group's channels only within the joints of that region.
    This injects structural skeleton bias at zero cost.

    Args:
        channels:     Number of input/output channels (must equal C).
        A:            (V, V) adjacency tensor used for cross-body neighbour lookup.
        shift_type:   'anatomical' (BRASP) | 'full_body' | 'random' | 'none'.
                      The last three are ablation controls (see
                      compute_shift_indices).
        seed:         RNG seed for the 'random' control.
    """

    def __init__(self, channels: int, A: torch.Tensor,
                 shift_type: str = 'anatomical', seed: int = 0):
        super().__init__()
        self.shift_type = shift_type
        shift_indices = compute_shift_indices(A, channels, shift_type=shift_type, seed=seed)
        self.register_buffer('shift_indices', shift_indices)  # (C, V), no grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T, V) — same shape, channels spatially permuted by region
        """
        B, C, T, V = x.shape
        # Expand (C, V) → (B, C, T, V) for torch.gather
        idx = self.shift_indices.unsqueeze(0).unsqueeze(2).expand(B, C, T, V)
        return torch.gather(x, 3, idx)
