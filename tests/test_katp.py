"""
Tests for KATP (Kinematic-Tree Anatomical Partition).

The first test is the paper's central claim: the hand-designed NTU-25 partition
is exactly what the derivation emits from the bone list, so it is not a tuned
artefact and every reported number is also a derived-partition number.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.blocks.katp import (  # noqa: E402
    SKELETON_EDGES,
    NUM_NODES,
    channel_group_fractions,
    derive_partition,
    partition_for,
)
from src.models.blocks.body_region_shift import BODY_REGIONS  # noqa: E402


# ---------------------------------------------------------------------------
# The load-bearing claim
# ---------------------------------------------------------------------------
def test_ntu25_derivation_reproduces_hand_designed_partition():
    """KATP(NTU bone list) == the hand-written BODY_REGIONS literal, exactly."""
    derived = partition_for('ntu-rgb+d')

    assert set(derived) == set(BODY_REGIONS), (
        f"region names differ: derived={sorted(derived)} "
        f"vs hand-designed={sorted(BODY_REGIONS)}")

    for region, joints in BODY_REGIONS.items():
        assert derived[region] == sorted(joints), (
            f"{region}: derived={derived[region]} vs hand-designed={sorted(joints)}")


def test_ntu25_partition_is_a_complete_disjoint_cover():
    derived = partition_for('ntu-rgb+d')
    seen = []
    for joints in derived.values():
        seen.extend(joints)
    assert sorted(seen) == list(range(25)), "partition must cover all 25 joints"
    assert len(seen) == len(set(seen)), "regions must be disjoint"


# ---------------------------------------------------------------------------
# Generality of the construction (structural only — no accuracy is implied)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('layout', sorted(SKELETON_EDGES))
def test_every_layout_yields_a_complete_disjoint_cover(layout):
    derived = partition_for(layout)
    seen = []
    for joints in derived.values():
        seen.extend(joints)
    assert sorted(seen) == list(range(NUM_NODES[layout]))
    assert len(seen) == len(set(seen))


@pytest.mark.parametrize('layout', sorted(SKELETON_EDGES))
def test_every_layout_finds_bilateral_pairs_and_a_torso(layout):
    derived = partition_for(layout)
    assert 'torso' in derived and derived['torso'], f"{layout}: no torso found"
    lefts = {k for k in derived if k.startswith('left_')}
    rights = {k for k in derived if k.startswith('right_')}
    assert lefts, f"{layout}: no bilateral limbs found"
    assert {k[5:] for k in lefts} == {k[6:] for k in rights}, (
        f"{layout}: unmatched bilateral pair")


def test_bilateral_pairs_have_equal_size():
    for layout in SKELETON_EDGES:
        derived = partition_for(layout)
        for key in (k for k in derived if k.startswith('left_')):
            mirror = 'right_' + key[5:]
            assert len(derived[key]) == len(derived[mirror]), (
                f"{layout}/{key}: {len(derived[key])} vs {len(derived[mirror])}")


def test_nw_ucla_torso_is_the_axial_chain():
    """20-joint Kinect v1: hip_centre, spine, shoulder_centre, head."""
    assert partition_for('nw-ucla')['torso'] == [0, 1, 2, 3]


def test_coco_non_tree_layout_recovers_anatomical_groups():
    """COCO-17 is not a tree; the BFS-tree reduction must still recover limbs.

    Expected semantics: arms = (elbow, wrist), legs = (hip, knee, ankle),
    torso = face + shoulders. COCO has no pelvis keypoint, so the shoulders are
    the branch nodes and the hips sit in the leg chains, as they do on NTU.
    """
    p = partition_for('coco')
    assert p['left_arm'] == [7, 9] and p['right_arm'] == [8, 10]
    assert p['left_leg'] == [11, 13, 15] and p['right_leg'] == [12, 14, 16]
    assert p['torso'] == [0, 1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Properties of the derivation
# ---------------------------------------------------------------------------
def test_derivation_is_deterministic():
    for layout in SKELETON_EDGES:
        assert partition_for(layout) == partition_for(layout)


def test_edge_order_does_not_change_the_partition():
    """The partition depends on the graph, not on how the bone list is written."""
    edges = SKELETON_EDGES['ntu-rgb+d']
    shuffled = [(b, a) for a, b in reversed(edges)]
    assert derive_partition(shuffled, 25) == derive_partition(edges, 25)


def test_root_choice_changes_labels_only_not_the_grouping():
    a = partition_for('ntu-rgb+d', root=0)
    b = partition_for('ntu-rgb+d', root=3)
    assert sorted(map(tuple, a.values())) == sorted(map(tuple, b.values()))


def test_channel_fractions_are_a_normalised_distribution():
    fr = channel_group_fractions(partition_for('ntu-rgb+d'))
    assert abs(sum(fr.values()) - 1.0) < 1e-9
    assert all(v > 0 for v in fr.values())
    assert fr['arm'] > fr['leg'] > fr['torso']   # 12 > 8 > 5 joints on NTU-25


def test_unknown_layout_raises():
    with pytest.raises(ValueError, match='Unknown layout'):
        partition_for('kinect-v9')
