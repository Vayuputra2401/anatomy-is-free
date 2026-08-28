"""
KATP — Kinematic-Tree Anatomical Partition.

Derives the body-region partition used by BRASP and SGPShift *from the skeleton's
bone hierarchy alone*, rather than hand-listing joint indices per dataset.

The partition emitted for NTU-25 is identical to the hand-designed
``BODY_REGIONS`` literal in ``body_region_shift.py`` (see ``tests/test_katp.py``),
so every result reported with the hand-designed groups is also a result for the
derived groups. Nothing is re-trained.

Algorithm (deterministic, zero parameters, no data, no training):

  1. Take the kinematic tree. Bone graphs that already are trees (Kinect
     layouts) are used as-is; graphs carrying redundant lateral links (COCO's
     shoulder-shoulder and hip-hip edges) are reduced to their BFS spanning tree
     from the root, which is the standard hierarchy for such formats.
  2. Branch nodes := nodes of degree >= 3 in that tree.
  3. Delete the branch nodes. The tree falls into chains.
  4. A chain adjacent to >= 2 branch nodes is axial -> torso. A chain adjacent to
     exactly one is a limb, anchored at that branch node.
  5. Pair limbs sharing (chain length, canonical rooted-tree shape, anchor depth).
     A matched pair is bilateral; an unmatched chain is median (neck/head) ->
     torso. Depth rather than anchor identity is used because bilateral limbs
     hang off a single hub in Kinect layouts but off mirrored hubs in COCO.
  6. Branch nodes themselves -> torso.

Region *names* (arm/leg) are cosmetic: they are assigned by anchor depth, and
BRASP consumes only the groups, never the labels.
"""

from collections import defaultdict, deque, OrderedDict


# ---------------------------------------------------------------------------
# Bone hierarchies (0-based, undirected). Public dataset constants — no data
# files are needed to derive a partition for any of these.
# ---------------------------------------------------------------------------
SKELETON_EDGES = {
    # NTU RGB+D, 25 joints (Kinect v2). Mirrors graph.py:36-41. Already a tree.
    'ntu-rgb+d': [
        (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
        (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
        (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7),
        (23, 24), (24, 11),
    ],
    # NW-UCLA / N-UCLA, 20 joints (Kinect v1). Already a tree.
    'nw-ucla': [
        (0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (2, 8),
        (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19),
    ],
    # COCO, 17 keypoints (2-D pose estimators). Not a tree: shoulder-shoulder,
    # hip-hip and facial links create cycles, so step 1 takes the BFS tree.
    'coco': [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6),
    ],
}

NUM_NODES = {'ntu-rgb+d': 25, 'nw-ucla': 20, 'coco': 17}

# Root joint per layout. Kinect layouts root at the pelvis; COCO has no pelvis
# keypoint, so its canonical root is the nose.
DEFAULT_ROOT = {'ntu-rgb+d': 0, 'nw-ucla': 0, 'coco': 0}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _adjacency(edges, num_nodes):
    adj = {i: set() for i in range(num_nodes)}
    for a, b in edges:
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _is_tree(adj, num_nodes):
    n_edges = sum(len(v) for v in adj.values()) // 2
    if n_edges != num_nodes - 1:
        return False
    seen, q = {0}, deque([0])
    while q:
        for w in adj[q.popleft()]:
            if w not in seen:
                seen.add(w)
                q.append(w)
    return len(seen) == num_nodes


def _bfs_tree(adj, num_nodes, root):
    """BFS spanning tree. Identity (up to edge order) when `adj` is a tree."""
    tree = {i: set() for i in range(num_nodes)}
    seen, q = {root}, deque([root])
    while q:
        u = q.popleft()
        for w in sorted(adj[u]):
            if w not in seen:
                seen.add(w)
                tree[u].add(w)
                tree[w].add(u)
                q.append(w)
    return tree


def _components_excluding(adj, blocked):
    seen, out = set(), []
    for s in sorted(adj):
        if s in seen or s in blocked:
            continue
        comp, q = set(), deque([s])
        seen.add(s)
        while q:
            u = q.popleft()
            comp.add(u)
            for w in adj[u]:
                if w not in seen and w not in blocked:
                    seen.add(w)
                    q.append(w)
        out.append(comp)
    return out


def _shape_signature(adj, comp, anchor):
    """Canonical (AHU) encoding of a chain entered from `anchor`.

    Mirrored limbs with identical structure produce identical strings, which is
    what lets step 5 pair them without reference to coordinates or joint names.
    """
    entry = min(w for w in comp if anchor in adj[w])

    def enc(u, parent):
        kids = sorted(enc(w, u) for w in adj[u] if w != parent and w in comp)
        return '(' + ''.join(kids) + ')'

    return enc(entry, anchor)


def _depths(adj, root):
    dist = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def derive_partition(edges, num_nodes, root=0):
    """Derive anatomical body regions from a bone hierarchy.

    Args:
        edges:     undirected (i, j) bone list, 0-based.
        num_nodes: number of joints V.
        root:      index of the root joint. For a tree skeleton this affects
                   region *labels* only. For a non-tree bone graph it also
                   selects the spanning tree, so it should be the format's
                   canonical root.

    Returns:
        OrderedDict mapping region name -> sorted joint index list. Bilateral
        regions appear as ``left_*`` / ``right_*`` pairs; everything axial is
        merged into ``torso``.
    """
    adj = _adjacency(edges, num_nodes)
    tree = adj if _is_tree(adj, num_nodes) else _bfs_tree(adj, num_nodes, root)

    branch = {v for v in tree if len(tree[v]) >= 3}
    if not branch:                                  # degenerate: a bare chain
        return OrderedDict(torso=list(range(num_nodes)))

    depth = _depths(tree, root)
    torso = set(branch)
    limbs = []

    for chain in _components_excluding(tree, branch):
        touching = {b for b in branch if any(b in tree[u] for u in chain)}
        if len(touching) == 1:
            limbs.append((chain, next(iter(touching))))
        else:                                       # axial, or detached
            torso |= chain

    buckets = defaultdict(list)
    for chain, anchor in limbs:
        key = (len(chain), _shape_signature(tree, chain, anchor),
               depth.get(anchor, -1))
        buckets[key].append((chain, anchor))

    pairs, unpaired = [], []
    for key in sorted(buckets):
        group = buckets[key]
        if len(group) == 2:
            group.sort(key=lambda ca: min(ca[0]))   # lower index = "left"
            pairs.append((key[2], group[0][0], group[1][0]))
        else:
            unpaired.extend(group)

    for chain, _ in unpaired:                       # median chains -> axial
        torso |= chain

    # Label bilateral pairs by anchor depth, breaking ties toward the longer
    # kinematic chain (the lower limb): nearest the root is "leg", the next
    # "arm", then generic limb2, limb3... The tie-break matters only for
    # layouts whose limbs hang off mirrored hubs at equal depth, such as COCO.
    # Labels are exposition only; BRASP consumes the groups, never the names.
    ordered = sorted(pairs, key=lambda p: (p[0], -len(p[1]), min(p[1])))
    names = ['leg', 'arm'] + [f'limb{i}' for i in range(2, len(ordered) + 2)]

    out = OrderedDict()
    for (_, left, right), name in zip(ordered, names):
        out[f'left_{name}'] = sorted(left)
        out[f'right_{name}'] = sorted(right)
    out['torso'] = sorted(torso)
    return out


def partition_for(layout, root=None):
    """Derive the partition for a named layout in ``SKELETON_EDGES``."""
    if layout not in SKELETON_EDGES:
        raise ValueError(
            f"Unknown layout '{layout}' (known: {sorted(SKELETON_EDGES)})")
    if root is None:
        root = DEFAULT_ROOT[layout]
    return derive_partition(SKELETON_EDGES[layout], NUM_NODES[layout], root=root)


def channel_group_fractions(partition):
    """Channel budget per BRASP group, proportional to joint count.

    Bilateral pairs are merged (left arm + right arm -> 'arm'), matching BRASP's
    combined-group design.
    """
    merged = defaultdict(list)
    for name, joints in partition.items():
        base = name.split('_', 1)[1] if name.startswith(('left_', 'right_')) else name
        merged[base].extend(joints)

    total = sum(len(v) for v in merged.values())
    return {k: len(v) / total for k, v in sorted(merged.items())}
