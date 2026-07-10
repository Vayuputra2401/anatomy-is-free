"""
ShiftFuse-Zero — Five-model family for NTU-60/120 skeleton action recognition.

Models:
  nano_tiny_efficient    — single backbone, 3-stream early fusion,  ~94K params
  small_late_efficient_bb  — 2-backbone late fusion (joint+vel / bone), ~284K
  medium_late_efficient_bb — 2-backbone late fusion, B2-scale channels, ~594K
  large_b4_efficient     — B4-style mid-network fusion, 3 streams, ~1.18M
  x_efficient            — scaled B4-style, 3-stream mid-fusion, ~2.0M

EfficientZero Block pipeline (Nano / Small / Medium):
    BRASP → SGPShift → JE(in_ch) → STCAttention(in_ch)
    → GCN: (1/K) Σ_k W_k(normalize(Ã_k + A_learned[k]) @ x)
    → DepthwiseSepTCN → DropPath → residual
    → [TLA at last block of last stage when use_tla=True]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.stream_fusion_concat import StreamFusionConcat
from .blocks.cross_stream_fusion import CrossStreamFusion
from .blocks.body_region_shift import BodyRegionShift
from .blocks.sgp_shift import SGPShift
from .blocks.joint_embedding import JointEmbedding
from .blocks.temporal_landmark_attn import TemporalLandmarkAttention
from .blocks.drop_path import DropPath
from .blocks.stc_attention import STCAttention
from .blocks.dw_sep_tcn import DepthwiseSepTCN
from .blocks.part_attention import PartAttention
from .graph import Graph, normalize_symdigraph_full, normalize_digraph


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------
ZERO_VARIANTS = {
    # nano: single backbone, 3-stream early fusion. ~94K params.
    'nano_tiny_efficient': {
        'stem_channels':       24,
        'channels':            [32, 64, 128],
        'num_blocks':          [1, 1, 1],
        'strides':             [1, 2, 2],
        'drop_path_rate':      0.05,
        'dropout':             0.10,
        'tla_landmarks':       12,
        'tla_reduce_ratio':    8,
        'use_tla':             True,
        'num_streams':         3,
        'stream_names':        ['joint', 'bone', 'velocity'],
        'share_a_learned':     True,   # 1 global A_learned shared across all blocks
        'stc_channel_se':      False,
        'tcn_depth':           [1, 1, 1],
        'use_part_att':        [False, False, False],
    },
    # small: 2-backbone late fusion (joint+vel / bone). ~284K params.
    'small_late_efficient_bb': {
        'stem_channels':       24,
        'channels':            [32, 64, 128],
        'num_blocks':          [1, 2, 1],
        'strides':             [1, 2, 2],
        'drop_path_rate':      0.05,
        'dropout':             0.10,
        'tla_landmarks':       8,
        'tla_reduce_ratio':    8,
        'use_tla':             True,
        'stc_channel_se':      False,
        'share_a_learned_stage': [False, True, False],  # share A_learned within stage 1 (same resolution)
        'tcn_depth':           [1, 2, 1],
        'use_part_att':        [False, False, True],   # Part_Att on last stage only
        'part_att_reduce_ratio': 16,
    },
    # medium: 2-backbone late fusion, B2-scale channels. ~594K params.
    'medium_late_efficient_bb': {
        'stem_channels':       32,
        'channels':            [40, 80, 160],
        'num_blocks':          [1, 2, 1],
        'strides':             [1, 2, 2],
        'drop_path_rate':      0.10,
        'dropout':             0.15,
        'tla_landmarks':       10,
        'tla_reduce_ratio':    8,
        'use_tla':             True,
        'stc_channel_se':      True,
        'tcn_depth':           [1, 2, 1],
        'use_part_att':        [False, False, True],
        'part_att_reduce_ratio': 4,
    },
}


# ---------------------------------------------------------------------------
# EfficientZeroBlock
# ---------------------------------------------------------------------------
class EfficientZeroBlock(nn.Module):
    """ShiftFuse-Zero block with EfficientGCN-exact graph + DS-TCN + STC-Attention.

    Pipeline:
        BRASP → SGPShift → JE(in_ch) → STCAttention(in_ch)
        → GCN: (1/K) Σ_k W_k(normalize(Ã_k + A_learned[k]) @ x)
        → DepthwiseSepTCN → DropPath → residual → Hardswish

    Ordering rationale:
      1. JE(in_ch) before STC-Attn: spatial attention (softmax over V joints) runs on
         identity-enriched features → attention map knows which joint is which.
      2. STC-Attn before GCN: gates the GCN input so aggregation only propagates
         relevant (attended) signal across edges.
      3. Each block owns its own JE sized to in_channels (not shared) to handle
         in_ch ≠ out_ch at stage boundaries without a channel mismatch.

    Key differences from earlier ZeroGCNBlock:
      - A_k = normalize(Ã_k_fixed + A_learned[k]) — EfficientGCN-exact learnable graph
      - STC-Attention (spatial+temporal+channel) with residual gate replaces ChannelSE
      - DepthwiseSepTCN (~8× cheaper than MultiScaleTCN) enables K=3 W_k in budget
      - JE per-block on in_channels (not shared stage JE on out_channels)

    Args:
        in_channels:       Input feature channels.
        out_channels:      Output feature channels.
        stride:            Temporal stride (1 or 2).
        A_flat:            (V,V) flat adjacency for BRASP.
        A_intra:           (V,V) intra-part adjacency for SGPShift.
        A_inter:           (V,V) inter-part adjacency for SGPShift.
        A_gcn_partitions:  List of K (V,V) normalized fixed structural tensors.
        drop_path_rate:    Stochastic depth probability.
        dropout:           DepthwiseSepTCN dropout rate.
        num_joints:        V (default 25).
        stc_reduce_ratio:  Channel reduction ratio for SE part of STC-Attention.
        stc_channel_se:    Include channel SE branch in STCAttention (default True).
                           Set False for nano/small to save params.
        tcn_depth:         Number of DS-TCN layers (default 1). Stride on first only.
        a_learned_shared:  Pre-created nn.ParameterList to share across blocks.
                           If None, block creates its own (default behaviour).
        use_part_att:      Append PartAttention after GCN (default False).
                           Used on last stage of small/medium/X.
        tla:               Optional TemporalLandmarkAttention (last block of last stage).
    """

    def __init__(
        self,
        in_channels:       int,
        out_channels:      int,
        stride:            int,
        A_flat:            torch.Tensor,
        A_intra:           torch.Tensor,
        A_inter:           torch.Tensor,
        A_gcn_partitions:  list,
        drop_path_rate:    float              = 0.0,
        dropout:           float              = 0.1,
        num_joints:        int                = 25,
        stc_reduce_ratio:  int                = 4,
        stc_channel_se:    bool               = True,
        tcn_depth:         int                = 1,
        a_learned_shared:  nn.ParameterList   = None,
        use_part_att:      bool               = False,
        part_att_reduce:   int                = 4,
        tla:               nn.Module          = None,
        shift_type:        str                = 'anatomical',
    ):
        super().__init__()
        self.K_gcn = len(A_gcn_partitions)
        # Per-block (or shared) learnable adjacency residuals (EfficientGCN-exact)
        if a_learned_shared is not None:
            self.A_learned = a_learned_shared   # shared reference — no new params
        else:
            self.A_learned = nn.ParameterList([
                nn.Parameter(torch.zeros(num_joints, num_joints))
                for _ in range(self.K_gcn)
            ])

        # ── 0-param spatial routing ──────────────────────────────────────────
        # shift_type='anatomical' (default) runs the full BRASP + SGPShift prior.
        # The ablation controls (BMVC'26 rebuttal, Table 12) replace the whole
        # mechanism with a single shift: 'random' / 'full_body' route BRASP
        # accordingly and disable SGPShift; 'none' disables both (82.8 baseline).
        self.shift_type = shift_type
        if shift_type == 'anatomical':
            self.brasp     = BodyRegionShift(channels=in_channels, A=A_flat)
            self.sgp_shift = SGPShift(
                channels=in_channels, A_intra=A_intra, A_inter=A_inter,
                num_joints=num_joints,
            )
        elif shift_type == 'none':
            self.brasp     = nn.Identity()
            self.sgp_shift = nn.Identity()
        else:  # 'random' | 'full_body' — single control shift, SGPShift off
            self.brasp     = BodyRegionShift(channels=in_channels, A=A_flat,
                                             shift_type=shift_type)
            self.sgp_shift = nn.Identity()

        # ── Joint identity embedding (in_channels — before STC and GCN) ─────
        self.je = JointEmbedding(in_channels, num_joints)

        # ── Fixed structural adjacency (K=3 anatomical partitions) ───────────
        for k, A_k in enumerate(A_gcn_partitions):
            self.register_buffer(f'_A_fixed_{k}', A_k)

        # ── Per-partition GCN convolutions (K separate W_k) ──────────────────
        self.gcn_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(self.K_gcn)
        ])
        self.gcn_bn  = nn.BatchNorm2d(out_channels)
        self.gcn_act = nn.Hardswish(inplace=True)

        # ── STC-Attention on in_channels (after JE, before GCN) ─────────────
        # Disable spatial branch when PartAttention is active: both operate on the
        # joint axis (V) around the same GCN and would produce competing signals.
        # Temporal gate (18 params) still runs to weight frames before aggregation.
        self.stc_attn = STCAttention(in_channels, num_joints,
                                     stc_reduce_ratio,
                                     use_channel_se=stc_channel_se,
                                     use_spatial=(not use_part_att))

        # ── Optional PartAttention after GCN (before TCN) ────────────────────
        self.part_att = PartAttention(out_channels, reduce_ratio=part_att_reduce, num_joints=num_joints) if use_part_att else None

        # ── Depthwise-separable multi-scale TCN (stacked tcn_depth times) ───
        # First layer carries the temporal stride; subsequent layers stride=1.
        tcns = [DepthwiseSepTCN(out_channels, out_channels,
                                stride=stride, dropout=dropout)]
        for _ in range(tcn_depth - 1):
            tcns.append(DepthwiseSepTCN(out_channels, out_channels,
                                        stride=1, dropout=dropout))
        self.tcn       = nn.Sequential(*tcns)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # ── Residual ─────────────────────────────────────────────────────────
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.out_act = nn.Hardswish(inplace=True)

        # ── Optional TLA (last block of last stage) ───────────────────────────
        self.tla = tla

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        # ── 0-param spatial routing ──────────────────────────────────────────
        x = self.brasp(x)
        x = self.sgp_shift(x)

        # ── JE: inject joint identity before attention ────────────────────────
        x = self.je(x)

        # ── STC-Attention gates the GCN input ────────────────────────────────
        x = self.stc_attn(x)

        # ── GCN: y = (1/K) Σ_k W_k(normalize(Ã_k + A_learned[k]) @ x) ────
        B, C, T, V = x.shape
        x_flat = x.reshape(B, C * T, V)
        y = None
        for k in range(self.K_gcn):
            A_fixed  = getattr(self, f'_A_fixed_{k}')
            A_learnt = self.A_learned[k]
            A_comb = (A_fixed + A_learnt).abs()
            d      = A_comb.sum(dim=1).clamp(min=1e-6).pow(-0.5)
            A_norm = d.unsqueeze(1) * A_comb * d.unsqueeze(0)
            x_k    = torch.matmul(x_flat, A_norm).reshape(B, C, T, V)
            out_k  = self.gcn_convs[k](x_k)
            y = out_k if y is None else y + out_k
        y = y / self.K_gcn
        x = self.gcn_act(self.gcn_bn(y))

        # ── PartAttention (optional, gates joint features after GCN) ────────
        if self.part_att is not None:
            x = self.part_att(x)

        # ── DS-TCN stack + DropPath ──────────────────────────────────────────
        x = self.drop_path(self.tcn(x))

        # ── Residual ─────────────────────────────────────────────────────────
        x = self.out_act(x + self.residual(res))

        # ── TLA ──────────────────────────────────────────────────────────────
        if self.tla is not None:
            x = self.tla(x)

        return x


# ---------------------------------------------------------------------------
# ShiftFuseZero  (backbone — used by all three models)
# ---------------------------------------------------------------------------
class ShiftFuseZero(nn.Module):
    """ShiftFuse-Zero backbone. Supports early fusion (4 streams) and sub-backbone
    roles inside ShiftFuseZeroLate / ShiftFuseZeroLate3 (1–2 streams).

    Args:
        num_classes:    Output classes.
        variant:        One of the keys in ZERO_VARIANTS.
        in_channels:    Channels per input stream (default 3).
        graph_layout:   Skeleton layout (default 'ntu-rgb+d').
        num_joints:     V (default 25).
        dropout:        Override variant default dropout.
        num_streams:    Number of input streams (4 for standalone, 1-2 for sub-backbone).
        stream_names:   Which stream keys to read from the input dict.
    """

    def __init__(
        self,
        num_classes:  int   = 60,
        variant:      str   = 'nano_tiny_efficient',
        in_channels:  int   = 3,
        graph_layout: str   = 'ntu-rgb+d',
        num_joints:   int   = 25,
        dropout:      float = None,
        num_streams:  int   = 4,
        stream_names: list  = None,
        use_tla:      bool  = None,
        shift_type:   str   = 'anatomical',
    ):
        super().__init__()

        self.shift_type = shift_type
        if variant not in ZERO_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(ZERO_VARIANTS.keys())}")

        cfg            = ZERO_VARIANTS[variant]
        stem_ch        = cfg['stem_channels']
        channels       = cfg['channels']
        num_blocks     = cfg['num_blocks']
        strides        = cfg['strides']
        drop_path_rate = cfg['drop_path_rate']
        tla_landmarks  = cfg['tla_landmarks']
        tla_reduce     = cfg['tla_reduce_ratio']
        _dropout       = dropout if dropout is not None else cfg['dropout']
        use_tla        = use_tla if use_tla is not None else cfg.get('use_tla', False)
        _stc_ch_se          = cfg.get('stc_channel_se', True)
        _share_a_learn      = cfg.get('share_a_learned', False)
        _share_a_learn_stg  = cfg.get('share_a_learned_stage', None)  # list[bool] per stage
        _tcn_depth          = cfg.get('tcn_depth', [1] * len(channels))
        _use_part_att       = cfg.get('use_part_att', [False] * len(channels))
        _part_att_reduce    = cfg.get('part_att_reduce_ratio', 4)
        # Variant can specify stream layout; constructor args override if provided
        num_streams    = num_streams if num_streams != 4 else cfg.get('num_streams', num_streams)

        self.variant     = variant
        self.num_classes = num_classes
        _default_names   = ['joint', 'velocity', 'bone', 'bone_velocity']
        _var_names       = cfg.get('stream_names', None)
        self.stream_names = (
            stream_names if stream_names is not None
            else _var_names if _var_names is not None
            else _default_names[:num_streams]
        )

        # ── 1. Semantic body-part graph ───────────────────────────────────
        graph = Graph(
            layout=graph_layout,
            strategy='semantic_bodypart',
            max_hop=2,
            raw_partitions=True,
        )
        A_raw = graph.A
        A_sym = normalize_symdigraph_full(A_raw)

        A_intra = torch.tensor(A_sym[0], dtype=torch.float32)
        A_inter = torch.tensor(A_sym[1], dtype=torch.float32)
        A_flat  = torch.tensor((A_raw.sum(0) > 0).astype('float32'))

        self.register_buffer('A_intra', A_intra)
        self.register_buffer('A_inter', A_inter)
        self.register_buffer('A_flat',  A_flat)

        # Pass RAW (unnormalized) partitions to each block.
        # EfficientZeroBlock.forward() normalizes (A_raw_fixed + A_learned) once inline.
        # Passing pre-normalized A_sym here would cause double normalization, shrinking
        # the effective adjacency spectrum at every block across 9+ blocks.
        A_gcn_partitions = [
            torch.tensor(A_raw[k], dtype=torch.float32)
            for k in range(A_raw.shape[0])
        ]

        # ── 2. Early-fusion stem ──────────────────────────────────────────
        self.fusion = StreamFusionConcat(
            in_channels=in_channels,
            out_channels=stem_ch,
            num_streams=num_streams,
        )

        # ── 3. Build shared A_learned pools ──────────────────────────────────
        # Global share (nano): one set across ALL blocks
        if _share_a_learn:
            shared_a_learned_global = nn.ParameterList([
                nn.Parameter(torch.zeros(num_joints, num_joints))
                for _ in range(len(A_gcn_partitions))
            ])
            self.register_module('shared_a_learned', shared_a_learned_global)
        else:
            shared_a_learned_global = None

        # Per-stage share (small stage1): one set within a stage (blocks of same resolution)
        # _share_a_learn_stg: list[bool] per stage, e.g. [True, False] → share in stage0 only
        n_stages = len(channels)
        stage_shared_a = []
        if _share_a_learn_stg is not None:
            for si in range(n_stages):
                do_share = (_share_a_learn_stg[si] if si < len(_share_a_learn_stg) else False)
                if do_share and shared_a_learned_global is None:
                    stg_a = nn.ParameterList([
                        nn.Parameter(torch.zeros(num_joints, num_joints))
                        for _ in range(len(A_gcn_partitions))
                    ])
                    self.register_module(f'shared_a_learned_stage{si}', stg_a)
                    stage_shared_a.append(stg_a)
                else:
                    stage_shared_a.append(None)
        else:
            stage_shared_a = [None] * n_stages

        # ── 4. Build stages ───────────────────────────────────────────────
        total_blocks     = sum(num_blocks)
        block_idx_global = 0
        self.stages      = nn.ModuleList()
        prev_ch          = stem_ch

        for stage_idx in range(len(channels)):
            stage_ch      = channels[stage_idx]
            stage_stride  = strides[stage_idx]
            n_blocks      = num_blocks[stage_idx]
            is_last_stage = (stage_idx == len(channels) - 1)
            stage_tcn_d   = _tcn_depth[stage_idx] if isinstance(_tcn_depth, list) else _tcn_depth
            stage_part    = _use_part_att[stage_idx] if isinstance(_use_part_att, list) else _use_part_att

            # Resolve which A_learned set this stage's blocks share
            a_shared = shared_a_learned_global or stage_shared_a[stage_idx]

            stage_tla = (
                TemporalLandmarkAttention(stage_ch, tla_landmarks, tla_reduce)
                if (is_last_stage and use_tla) else None
            )

            stage_blocks = nn.ModuleList()
            for block_idx in range(n_blocks):
                dp_rate  = drop_path_rate * block_idx_global / max(total_blocks - 1, 1)
                stride   = stage_stride if (block_idx == 0 and stage_idx > 0) else 1
                c_in     = prev_ch if block_idx == 0 else stage_ch
                is_last  = (block_idx == n_blocks - 1)

                block = EfficientZeroBlock(
                    in_channels      = c_in,
                    out_channels     = stage_ch,
                    stride           = stride,
                    A_flat           = A_flat,
                    A_intra          = A_intra,
                    A_inter          = A_inter,
                    A_gcn_partitions = A_gcn_partitions,
                    drop_path_rate   = dp_rate,
                    dropout          = _dropout,
                    num_joints       = num_joints,
                    stc_channel_se   = _stc_ch_se,
                    tcn_depth        = stage_tcn_d,
                    a_learned_shared = a_shared,
                    use_part_att     = (stage_part and is_last),
                    part_att_reduce  = _part_att_reduce,
                    tla              = stage_tla if is_last else None,
                    shift_type       = shift_type,
                )
                stage_blocks.append(block)
                block_idx_global += 1

            self.stages.append(stage_blocks)
            prev_ch = stage_ch

        # ── 4. Classifier head ────────────────────────────────────────────
        final_ch = channels[-1]
        self.pool_gate  = nn.Parameter(torch.full((1,), 4.0))  # sigmoid(4)≈0.98 → near-pure GAP (EfficientGCN approach)
        self.classifier = nn.Sequential(
            nn.Dropout(p=_dropout),
            nn.Linear(final_ch, num_classes),
        )

    def extract_features(self, stream_dict: dict):
        """Run stream fusion + all backbone stages.

        Returns:
            feat:       (2B or B, C, T', V) pre-pooling feature map.
            B:          Original batch size.
            multi_body: Whether M=2 body doubling was applied.
        """
        streams    = []
        B          = None
        multi_body = False
        for name in self.stream_names:
            s = stream_dict[name]
            if s.dim() == 5:
                if B is None:
                    B = s.shape[0]
                multi_body = True
                s = torch.cat([s[..., 0], s[..., 1]], dim=0)
            else:
                if B is None:
                    B = s.shape[0]
            streams.append(s)

        x = self.fusion(streams)
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x = block(x)
        return x, B, multi_body

    def classify(self, x: torch.Tensor, B: int, multi_body: bool) -> torch.Tensor:
        """Gated GAP+GMP pool → classifier → body average."""
        x   = x.mean(dim=2)
        gap = x.mean(dim=-1)
        gmp = x.max(dim=-1).values
        g   = torch.sigmoid(self.pool_gate)
        x   = g * gap + (1 - g) * gmp
        out = self.classifier(x)
        if multi_body:
            out = (out[:B] + out[B:]) / 2
        return out

    def forward(self, stream_dict: dict) -> torch.Tensor:
        x, B, multi_body = self.extract_features(stream_dict)
        return self.classify(x, B, multi_body)


# ---------------------------------------------------------------------------
# ShiftFuseZeroLate  (2-backbone late fusion — small_late_efficient)
# ---------------------------------------------------------------------------
class ShiftFuseZeroLate(nn.Module):
    """2-backbone late-fusion ShiftFuse-Zero with Shared TLA & Early Fusion.

    3-stream split (joint / velocity / bone — no bone_velocity):
    Backbone A: joint + velocity  (2-stream — position + temporal dynamics)
    Backbone B: bone              (1-stream — structural pose only)

    Orthogonal split: backbone A captures trajectory (position + its derivative),
    backbone B captures skeleton shape (bone vectors). After CrossStreamFusion,
    backbone B gains access to velocity context through backbone A's features.

    Architecture:
        Stage 1 → Stage 2 → [Early CrossStreamFusion] →
        Stage 3 → [Late CrossStreamFusion] → [Shared TLA] → classify

    The single shared TLA module is placed after the final cross-stream
    fusion, before the classifier. This saves ~4K params vs per-backbone
    TLA while giving both streams access to temporal landmark reasoning.

    Supports variants: small_late_efficient_bb, medium_late_efficient_bb.
    """

    def __init__(
        self,
        num_classes:  int   = 60,
        variant:      str   = 'small_late_efficient_bb',
        in_channels:  int   = 3,
        graph_layout: str   = 'ntu-rgb+d',
        num_joints:   int   = 25,
        dropout:      float = None,
        cross_stream: bool  = True,
    ):
        super().__init__()
        cfg = ZERO_VARIANTS[variant]

        # ── Backbone A: joint + velocity (2-stream — position + motion) ──
        self.backbone_a = ShiftFuseZero(
            num_classes=num_classes, variant=variant,
            in_channels=in_channels, graph_layout=graph_layout,
            num_joints=num_joints, dropout=dropout,
            num_streams=2, stream_names=['joint', 'velocity'],
            use_tla=False,
        )

        # ── Backbone B: bone (1-stream — structural pose only) ───────────
        self.backbone_b = ShiftFuseZero(
            num_classes=num_classes, variant=variant,
            in_channels=in_channels, graph_layout=graph_layout,
            num_joints=num_joints, dropout=dropout,
            num_streams=1, stream_names=['bone'],
            use_tla=False,
        )

        # ── Cross-stream fusion gates ────────────────────────────────────
        ch_s2 = cfg['channels'][1]   # after stage 2
        ch_s3 = cfg['channels'][-1]  # after stage 3
        self.cross_fusion_early = CrossStreamFusion(ch_s2)
        self.cross_fusion_late  = CrossStreamFusion(ch_s3)

        # ── Shared TLA (single instance, applied to both streams) ────────
        if cfg.get('use_tla', False):
            self.tla_shared = TemporalLandmarkAttention(
                ch_s3,
                num_landmarks=cfg.get('tla_landmarks', 8),
                reduce_ratio=cfg.get('tla_reduce_ratio', 8),
            )
        else:
            self.tla_shared = None

    def _run_stem(self, bb, stream_dict):
        """Run a backbone's stem (stream BN + fusion concat)."""
        streams = []
        B, multi_body = None, False
        for s_name in bb.stream_names:
            s = stream_dict[s_name]
            if s.dim() == 5:
                B = s.shape[0]
                s = torch.cat([s[..., 0], s[..., 1]], dim=0)
                multi_body = True
            else:
                if B is None:
                    B = s.shape[0]
            streams.append(s)
        return bb.fusion(streams), B, multi_body

    def forward(self, stream_dict: dict) -> torch.Tensor:
        # ── Stem ─────────────────────────────────────────────────────────
        x_a, B, mb = self._run_stem(self.backbone_a, stream_dict)
        x_b, _, _  = self._run_stem(self.backbone_b, stream_dict)

        # ── Stage 1 ─────────────────────────────────────────────────────
        for block in self.backbone_a.stages[0]:
            x_a = block(x_a)
        for block in self.backbone_b.stages[0]:
            x_b = block(x_b)

        # ── Stage 2 ─────────────────────────────────────────────────────
        for block in self.backbone_a.stages[1]:
            x_a = block(x_a)
        for block in self.backbone_b.stages[1]:
            x_b = block(x_b)

        # ── Early cross-stream fusion (after Stage 2) ────────────────────
        x_a, x_b = self.cross_fusion_early(x_a, x_b)

        # ── Stage 3 ─────────────────────────────────────────────────────
        for block in self.backbone_a.stages[2]:
            x_a = block(x_a)
        for block in self.backbone_b.stages[2]:
            x_b = block(x_b)

        # ── Late cross-stream fusion (after Stage 3) ─────────────────────
        x_a, x_b = self.cross_fusion_late(x_a, x_b)

        # ── Shared TLA ──────────────────────────────────────────────────
        if self.tla_shared is not None:
            x_a = self.tla_shared(x_a)
            x_b = self.tla_shared(x_b)

        # ── Classify each backbone and average ───────────────────────────
        logits_a = self.backbone_a.classify(x_a, B, mb)
        logits_b = self.backbone_b.classify(x_b, B, mb)
        return (logits_a + logits_b) / 2


# ---------------------------------------------------------------------------
# ShiftFuseZeroMidFusion  (B4-style mid-network fusion — large_late_efficient)
# ---------------------------------------------------------------------------
class ShiftFuseZeroMidFusion(nn.Module):
    """B4-matched mid-network fusion: 3 streams processed independently in
    stages 1–2, then concatenated and processed by a single shared stage 3.

    Fusion topology (mirrors EfficientGCN-B4):
        joint    → [stem] → [stage1] → [stage2] ─┐
        bone     → [stem] → [stage1] → [stage2] ──┤ concat → FusionConv → [stage3] → TLA → cls
        velocity → [stem] → [stage1] → [stage2] ─┘

    Architecture (tuned to ~1.09M ≈ B4's 1.1M):
        Per-stream: stem=32, channels=[48,96], blocks=[1,2], strides=[1,2]
        Fusion:     Conv2d(288→192) + BN + Hardswish
        Shared:     channels=192, blocks=4, stride=2 at first block
        TLA:        K=14 learnable anchors, reduce_ratio=8 (last shared block)

    Novel additions vs B4:
        BRASP (0-param anatomical shift), SGPShift (0-param semantic grouping),
        STC-Attention, per-block A_learned residuals, TLA learnable anchors.
    """

    STREAM_NAMES   = ['joint', 'bone', 'velocity']
    STREAM_STEM    = 32
    STREAM_CH      = [48, 96]
    STREAM_BLOCKS  = [1, 2]
    SHARED_CH      = 192
    SHARED_BLOCKS  = 4
    DROP_PATH_RATE = 0.20
    TLA_K          = 14
    TLA_REDUCE     = 8

    def __init__(
        self,
        num_classes:  int   = 60,
        graph_layout: str   = 'ntu-rgb+d',
        num_joints:   int   = 25,
        in_channels:  int   = 3,
        dropout:      float = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        _dropout = dropout if dropout is not None else 0.10

        # ── Graph ─────────────────────────────────────────────────────────
        graph    = Graph(layout=graph_layout, strategy='semantic_bodypart',
                         max_hop=2, raw_partitions=True)
        A_raw    = graph.A
        A_sym    = normalize_symdigraph_full(A_raw)
        A_intra  = torch.tensor(A_sym[0], dtype=torch.float32)
        A_inter  = torch.tensor(A_sym[1], dtype=torch.float32)
        A_flat   = torch.tensor((A_raw.sum(0) > 0).astype('float32'))
        self.register_buffer('A_intra', A_intra)
        self.register_buffer('A_inter', A_inter)
        self.register_buffer('A_flat',  A_flat)
        A_gcn_parts = [torch.tensor(A_raw[k], dtype=torch.float32)
                       for k in range(A_raw.shape[0])]

        # ── Drop-path schedule ─────────────────────────────────────────────
        n_stream_blk = sum(self.STREAM_BLOCKS)
        total_blocks = 3 * n_stream_blk + self.SHARED_BLOCKS

        # ── Per-stream stems (one per stream, 1-stream input) ──────────────
        self.stream_stems = nn.ModuleList([
            StreamFusionConcat(in_channels=in_channels,
                               out_channels=self.STREAM_STEM, num_streams=1)
            for _ in range(3)
        ])

        # ── Per-stream backbones (stages 1–2, independent weights) ─────────
        # Layout: stream_backbone[stream_idx][stage_idx] = nn.ModuleList of blocks
        self.stream_backbone = nn.ModuleList()
        blk_global = 0
        for _ in range(3):
            prev_ch    = self.STREAM_STEM
            stream_mods = nn.ModuleList()
            for si, (ch, nb) in enumerate(zip(self.STREAM_CH, self.STREAM_BLOCKS)):
                stage_mods = nn.ModuleList()
                for b in range(nb):
                    dp     = self.DROP_PATH_RATE * blk_global / max(total_blocks - 1, 1)
                    stride = 2 if (b == 0 and si > 0) else 1
                    c_in   = prev_ch if b == 0 else ch
                    stage_mods.append(EfficientZeroBlock(
                        in_channels=c_in, out_channels=ch, stride=stride,
                        A_flat=A_flat, A_intra=A_intra, A_inter=A_inter,
                        A_gcn_partitions=A_gcn_parts,
                        drop_path_rate=dp, dropout=_dropout,
                        num_joints=num_joints,
                    ))
                    blk_global += 1
                stream_mods.append(stage_mods)
                prev_ch = ch
            self.stream_backbone.append(stream_mods)

        # ── Fusion: concat(3×96=288) → 192 ────────────────────────────────
        fusion_in = 3 * self.STREAM_CH[-1]
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in, self.SHARED_CH, 1, bias=False),
            nn.BatchNorm2d(self.SHARED_CH),
            nn.Hardswish(inplace=True),
        )

        # ── Shared stage 3 ─────────────────────────────────────────────────
        tla = TemporalLandmarkAttention(self.SHARED_CH, self.TLA_K, self.TLA_REDUCE)
        self.shared_stage = nn.ModuleList()
        for b in range(self.SHARED_BLOCKS):
            dp     = self.DROP_PATH_RATE * blk_global / max(total_blocks - 1, 1)
            stride = 2 if b == 0 else 1
            self.shared_stage.append(EfficientZeroBlock(
                in_channels=self.SHARED_CH, out_channels=self.SHARED_CH,
                stride=stride,
                A_flat=A_flat, A_intra=A_intra, A_inter=A_inter,
                A_gcn_partitions=A_gcn_parts,
                drop_path_rate=dp, dropout=_dropout,
                num_joints=num_joints,
                tla=tla if (b == self.SHARED_BLOCKS - 1) else None,
            ))
            blk_global += 1

        # ── Classifier ─────────────────────────────────────────────────────
        self.pool_gate  = nn.Parameter(torch.full((1,), 4.0))  # sigmoid(4)≈0.98 → near-pure GAP (EfficientGCN approach)
        self.classifier = nn.Sequential(
            nn.Dropout(p=_dropout),
            nn.Linear(self.SHARED_CH, num_classes),
        )

    def forward(self, stream_dict: dict) -> torch.Tensor:
        # Collect stream inputs, handle M=2 bodies
        raw = [stream_dict[n] for n in self.STREAM_NAMES]
        B          = raw[0].shape[0]
        multi_body = raw[0].dim() == 5
        if multi_body:
            raw = [torch.cat([s[..., 0], s[..., 1]], dim=0) for s in raw]

        # Per-stream: stem → stage1 → stage2
        feats = []
        for s, stem, stages in zip(raw, self.stream_stems, self.stream_backbone):
            x = stem([s])
            for stage_blocks in stages:
                for block in stage_blocks:
                    x = block(x)
            feats.append(x)

        # Fusion: concat along channel dim → projection
        x = self.fusion_conv(torch.cat(feats, dim=1))

        # Shared stage 3
        for block in self.shared_stage:
            x = block(x)

        # Gated GAP+GMP → classifier
        x   = x.mean(dim=2)
        g   = torch.sigmoid(self.pool_gate)
        x   = g * x.mean(dim=-1) + (1 - g) * x.max(dim=-1).values
        out = self.classifier(x)
        if multi_body:
            out = (out[:B] + out[B:]) / 2
        return out


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
def build_shiftfuse_zero(
    variant: str = 'nano_tiny_efficient', num_classes: int = 60, **kwargs
) -> ShiftFuseZero:
    """Build a ShiftFuseZero model. Stream layout is read from variant config."""
    return ShiftFuseZero(num_classes=num_classes, variant=variant, **kwargs)


def build_shiftfuse_zero_late(
    variant: str = 'small_late_efficient_bb', num_classes: int = 60, **kwargs
) -> ShiftFuseZeroLate:
    """Build 2-backbone late fusion. Supports small and medium variants."""
    return ShiftFuseZeroLate(num_classes=num_classes, variant=variant, **kwargs)


def build_shiftfuse_zero_midfusion(num_classes: int = 60, **kwargs) -> ShiftFuseZeroMidFusion:
    """Build large_b4_efficient (B4-style mid-network fusion, ~1.12M)."""
    return ShiftFuseZeroMidFusion(num_classes=num_classes, **kwargs)


# ---------------------------------------------------------------------------
# TemporalSGLayer  — EfficientGCN-B4-exact Separable-Group temporal layer
# ---------------------------------------------------------------------------
class TemporalSGLayer(nn.Module):
    """EfficientGCN-exact Temporal_SG_Layer (Separable-Group temporal conv).

    4-op pipeline (reduct_ratio=2, kernel=5):
        DW(C,k=5,s=1) → BN → SiLU → PW(C→C//2) → BN →
        PW(C//2→C) → BN → SiLU → DW(C,k=5,stride) → BN
    Residual: Identity if stride=1, Conv(C,C,1,stride)+BN otherwise.
    Output: depth_conv2_out + residual  (no trailing act — handled by next layer).
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        inner = channels // 2
        pad   = 2  # (k=5 - 1) // 2

        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, (5, 1), 1, (pad, 0), groups=channels, bias=True),
            nn.BatchNorm2d(channels),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=True),
            nn.BatchNorm2d(inner),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=True),
            nn.BatchNorm2d(channels),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (5, 1), (stride, 1), (pad, 0),
                      groups=channels, bias=True),
            nn.BatchNorm2d(channels),
        )
        if stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride, 1), bias=True),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = F.silu(self.depth_conv1(x), inplace=True)
        x = self.point_conv1(x)
        x = F.silu(self.point_conv2(x), inplace=True)
        x = self.depth_conv2(x)
        return x + res


# ---------------------------------------------------------------------------
# STJointAtt  — EfficientGCN-exact ST_Joint_Att + Attention_Layer combined
# ---------------------------------------------------------------------------
class STJointAtt(nn.Module):
    """EfficientGCN-exact ST joint attention with internal residual BN+act.

    Computes S+T attention jointly (no channel SE — B4-exact):
        pool_T → pool_V → cat → FCN(C→C//2, Hardswish) → split →
        conv_t(C//2→C) sigmoid → conv_v(C//2→C) sigmoid
    Output: SiLU(BN(x * A_t * A_v) + x)   — residual inside.
    reduct_ratio=2  (B4-exact).
    """

    def __init__(self, channels: int):
        super().__init__()
        inner = channels // 2

        self.fcn = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=True),
            nn.BatchNorm2d(inner),
            nn.Hardswish(inplace=True),
        )
        self.conv_t = nn.Conv2d(inner, channels, 1, bias=True)
        self.conv_v = nn.Conv2d(inner, channels, 1, bias=True)
        self.bn     = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        x_t   = x.mean(3, keepdim=True)                        # (N, C, T, 1)
        x_v   = x.mean(2, keepdim=True).transpose(2, 3)        # (N, C, V, 1)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))         # (N, C//2, T+V, 1)
        x_t_att, x_v_att = torch.split(x_att, [T, V], dim=2)
        A_t   = self.conv_t(x_t_att).sigmoid()                 # (N, C, T, 1)
        A_v   = self.conv_v(x_v_att.transpose(2, 3)).sigmoid() # (N, C, 1, V)
        return F.silu(self.bn(x * A_t * A_v) + x, inplace=True)


# ---------------------------------------------------------------------------
# B4StemGCN  — B4-exact stem spatial GCN (no BRASP/SGPShift)
# ---------------------------------------------------------------------------
class B4StemGCN(nn.Module):
    """B4-exact stem GCN: K-partition with multiplicative A_edge (init=1).

    A_edge is a ParameterList caught by 'A_edge' in trainer no_decay.
    Forward: SiLU(BN(Σ_k gcn_k(x @ A_fixed_k*A_edge_k)) + residual(x))
    """

    def __init__(self, c_in: int, c_out: int, A_partitions: list, num_joints: int = 25):
        super().__init__()
        self.K = len(A_partitions)
        for k, A_k in enumerate(A_partitions):
            self.register_buffer(f'_A_fixed_{k}', A_k)
        # Multiplicative edge weights — init=1 → pure fixed adjacency at epoch 0
        self.A_edge = nn.ParameterList([
            nn.Parameter(torch.ones(num_joints, num_joints))
            for _ in range(self.K)
        ])
        self.gcn_convs = nn.ModuleList([
            nn.Conv2d(c_in, c_out, 1, bias=True) for _ in range(self.K)
        ])
        self.bn = nn.BatchNorm2d(c_out)
        self.residual = (nn.Identity() if c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, bias=True),
            nn.BatchNorm2d(c_out),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res        = self.residual(x)
        B, C, T, V = x.shape
        x_flat     = x.reshape(B, C * T, V)
        y          = None
        for k in range(self.K):
            A_eff = getattr(self, f'_A_fixed_{k}') * self.A_edge[k]
            x_k   = torch.matmul(x_flat, A_eff).reshape(B, C, T, V)
            out_k = self.gcn_convs[k](x_k)
            y     = out_k if y is None else y + out_k
        return F.silu(self.bn(y / self.K) + res, inplace=True)


# ---------------------------------------------------------------------------
# B4Block  — B4-exact block: 1 GCN + depth×TemporalSGLayer + 1 STJointAtt
# Novel additions vs EfficientGCN: BRASP + SGPShift (0-param) before GCN.
# A_edge: multiplicative learnable edge weights (init=1, no_decay in trainer).
# ---------------------------------------------------------------------------
class B4Block(nn.Module):
    """B4-exact block + JointEmbedding (after GCN) + optional TLA (after att).

    Structure:
        BRASP → SGPShift                              (0-param routing)
        → GCN(K=3, A_fixed*A_edge) → BN → SiLU       (B4-exact spatial)
        → JE(c_out)                                   (per-joint identity bias)
        → depth × TemporalSGLayer                     (B4-exact temporal)
        → STJointAtt                                  (B4-exact S+T attention)
        → [TLA(c_out) if use_tla]                     (global temporal, last block only)
    """

    def __init__(
        self,
        c_in:             int,
        c_out:            int,
        depth:            int,          # number of TemporalSGLayer layers
        stride:           int,          # applied to first TCN layer
        A_flat:           torch.Tensor,
        A_intra:          torch.Tensor,
        A_inter:          torch.Tensor,
        A_gcn_partitions: list,         # list of K pre-normalized (V,V) tensors
        num_joints:       int  = 25,
        use_tla:          bool = False,
        tla_landmarks:    int  = 14,
        tla_reduce_ratio: int  = 8,
        use_part_att:     bool = False,
        part_att_reduce:  int  = 4,
    ):
        super().__init__()
        self.K = len(A_gcn_partitions)

        # ── 0-param novel: BRASP + SGPShift ──────────────────────────────────
        self.brasp     = BodyRegionShift(channels=c_in, A=A_flat)
        self.sgp_shift = SGPShift(channels=c_in, A_intra=A_intra,
                                  A_inter=A_inter, num_joints=num_joints)

        # ── GCN: K-partition, multiplicative A_edge (init=1, no_decay) ───────
        for k, A_k in enumerate(A_gcn_partitions):
            self.register_buffer(f'_A_fixed_{k}', A_k)
        self.A_edge = nn.ParameterList([
            nn.Parameter(torch.ones(num_joints, num_joints))
            for _ in range(self.K)
        ])
        self.gcn_convs = nn.ModuleList([
            nn.Conv2d(c_in, c_out, 1, bias=True) for _ in range(self.K)
        ])
        self.gcn_bn  = nn.BatchNorm2d(c_out)
        self.gcn_res = (nn.Identity() if c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, bias=True),
            nn.BatchNorm2d(c_out),
        ))

        # ── JointEmbedding: per-joint identity bias after GCN (zero-init) ────
        # Caught by 'je.embed' in trainer no_decay.
        self.je = JointEmbedding(c_out, num_joints)

        # ── TCN: depth × TemporalSGLayer, first carries stride ───────────────
        self.tcns = nn.ModuleList([
            TemporalSGLayer(c_out, stride=(stride if j == 0 else 1))
            for j in range(depth)
        ])

        # ── Attention: STJointAtt (residual BN+SiLU inside) ──────────────────
        self.att = STJointAtt(c_out)

        # ── TLA: global temporal attention (last shared block only) ──────────
        # Gate init -4.0 → near-zero at epoch 0; caught by '.gate' no_decay.
        # anchor_logits caught by 'anchor_logits' no_decay.
        self.tla = (TemporalLandmarkAttention(
                        channels=c_out,
                        num_landmarks=tla_landmarks,
                        reduce_ratio=tla_reduce_ratio,
                        learnable_anchors=True,
                    ) if use_tla else None)

        # ── PartAttention: body-part gating after att (optional) ─────────────
        self.part_att = PartAttention(c_out, reduce_ratio=part_att_reduce, num_joints=num_joints) if use_part_att else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 0-param routing (novel additions)
        x = self.brasp(x)
        x = self.sgp_shift(x)

        # GCN (B4-exact)
        gcn_res    = self.gcn_res(x)
        B, C, T, V = x.shape
        x_flat     = x.reshape(B, C * T, V)
        y          = None
        for k in range(self.K):
            A_eff = getattr(self, f'_A_fixed_{k}') * self.A_edge[k]
            x_k   = torch.matmul(x_flat, A_eff).reshape(B, C, T, V)
            out_k = self.gcn_convs[k](x_k)
            y     = out_k if y is None else y + out_k
        x = F.silu(self.gcn_bn(y / self.K) + gcn_res, inplace=True)

        # Joint identity bias (novel — zero-init, no effect at epoch 0)
        x = self.je(x)

        # depth × TemporalSGLayer (B4-exact)
        for tcn in self.tcns:
            x = tcn(x)

        # S+T attention (B4-exact, residual inside)
        x = self.att(x)

        # Body-part gating (novel, last block only in X)
        if self.part_att is not None:
            x = self.part_att(x)

        # Global temporal attention (last block only, gated residual)
        if self.tla is not None:
            x = self.tla(x)

        return x


# ---------------------------------------------------------------------------
# ShiftFuseZeroB4  (EfficientGCN-B4-exact + BRASP + SGPShift)
# ---------------------------------------------------------------------------
class ShiftFuseZeroB4(nn.Module):
    """EfficientGCN-B4-exact architecture + BRASP + SGPShift (0-param only).

    Exact B4 block_args after compound scaling (α=1.2^4, β=1.35^4):
        [[96, stride=1, depth=2], [48, stride=1, depth=2],
         [128, stride=2, depth=3], [272, stride=2, depth=3]]

    Per-stream (joint / velocity / bone — separate weights):
        Stem: BN(3) → B4StemGCN(3→64) → BasicTCN(64, k=5)
        Block-0: B4Block(64→96,  depth=2, stride=1)
        Block-1: B4Block(96→48,  depth=2, stride=1)
    Fusion: concat(3×48 = 144)
    Shared:
        Block-2: B4Block(144→128, depth=3, stride=2)
        Block-3: B4Block(128→272, depth=3, stride=2)
    Classifier: GAP → Dropout(0.25) → Linear(272→num_classes)

    GCN:  spatial K=3 (D^{-1}A normalized), multiplicative A_edge (init=1).
    TCN:  Temporal_SG_Layer (DW→PW↓→PW↑→DW, reduct_ratio=2, k=5).
    Att:  ST_Joint_Att (S+T only, reduct_ratio=2, no channel SE).
    Act:  SiLU/Swish throughout.
    Novel: BRASP + SGPShift (0-param) in each B4Block before GCN.
    DROP_PATH=0.0, DROPOUT=0.25  — B4-exact.
    """

    STREAM_NAMES  = ['joint', 'velocity', 'bone']
    STREAM_STEM   = 64
    # (c_in, c_out, depth=n_TCN_layers, stride_on_first_TCN)
    STREAM_BLOCKS = [(64, 96, 2, 1), (96, 48, 2, 1)]
    SHARED_BLOCKS = [(144, 128, 3, 2), (128, 272, 3, 2)]
    DROPOUT       = 0.25

    def __init__(
        self,
        num_classes:  int   = 60,
        graph_layout: str   = 'ntu-rgb+d',
        num_joints:   int   = 25,
        in_channels:  int   = 3,
        dropout:      float = None,
    ):
        super().__init__()
        _dropout = dropout if dropout is not None else self.DROPOUT

        # ── Semantic body-part graph — BRASP + SGPShift ───────────────────────
        sgp_graph = Graph(layout=graph_layout, strategy='semantic_bodypart',
                          max_hop=2, raw_partitions=True)
        sgp_raw   = sgp_graph.A
        A_sym     = normalize_symdigraph_full(sgp_raw)
        A_intra   = torch.tensor(A_sym[0], dtype=torch.float32)
        A_inter   = torch.tensor(A_sym[1], dtype=torch.float32)
        A_flat    = torch.tensor((sgp_raw.sum(0) > 0).astype('float32'))
        self.register_buffer('A_intra', A_intra)
        self.register_buffer('A_inter', A_inter)
        self.register_buffer('A_flat',  A_flat)

        # ── Spatial K=3 graph pre-normalized D^{-1}A — GCN ───────────────────
        # Normalize per-partition with normalize_digraph (same as EfficientGCN).
        # A_edge (init=1) is applied multiplicatively on top at forward time.
        import numpy as np
        spatial_graph = Graph(layout=graph_layout, strategy='spatial',
                              max_hop=1, raw_partitions=True)
        A_gcn_parts   = [
            torch.tensor(normalize_digraph(spatial_graph.A[k]), dtype=torch.float32)
            for k in range(spatial_graph.A.shape[0])
        ]

        def _blk(c_in, c_out, depth, stride):
            return B4Block(c_in, c_out, depth, stride,
                           A_flat=A_flat, A_intra=A_intra, A_inter=A_inter,
                           A_gcn_partitions=A_gcn_parts, num_joints=num_joints)

        # ── Per-stream stems (B4-exact: BN → GCN(3→64) → BasicTCN(64)) ──────
        def _stem():
            stem_gcn = B4StemGCN(in_channels, self.STREAM_STEM, A_gcn_parts, num_joints)
            stem_tcn = _B4BasicTCN(self.STREAM_STEM)
            return _B4Stem(in_channels, stem_gcn, stem_tcn)

        self.stream_stems  = nn.ModuleList([_stem() for _ in range(3)])

        # ── Per-stream B4Blocks (separate weights per stream) ─────────────────
        self.stream_blocks = nn.ModuleList([
            nn.ModuleList([_blk(c_in, c_out, depth, stride)
                           for (c_in, c_out, depth, stride) in self.STREAM_BLOCKS])
            for _ in range(3)
        ])

        # ── Shared B4Blocks ───────────────────────────────────────────────────
        n_shared = len(self.SHARED_BLOCKS)
        self.shared_blocks = nn.ModuleList([
            B4Block(c_in, c_out, depth, stride,
                    A_flat=A_flat, A_intra=A_intra, A_inter=A_inter,
                    A_gcn_partitions=A_gcn_parts, num_joints=num_joints,
                    use_tla=(i == n_shared - 1))   # TLA on last shared block only
            for i, (c_in, c_out, depth, stride) in enumerate(self.SHARED_BLOCKS)
        ])

        # ── Classifier: GAP → Dropout → Linear (B4-exact) ────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=_dropout),
            nn.Linear(self.SHARED_BLOCKS[-1][1], num_classes),
        )

    def forward(self, stream_dict: dict) -> torch.Tensor:
        raw        = [stream_dict[n] for n in self.STREAM_NAMES]
        B          = raw[0].shape[0]
        multi_body = raw[0].dim() == 5
        if multi_body:
            raw = [torch.cat([s[..., 0], s[..., 1]], dim=0) for s in raw]

        feats = []
        for s, stem, blocks in zip(raw, self.stream_stems, self.stream_blocks):
            x = stem(s)
            for blk in blocks:
                x = blk(x)
            feats.append(x)

        x = torch.cat(feats, dim=1)
        for blk in self.shared_blocks:
            x = blk(x)

        x   = x.mean(dim=2).mean(dim=-1)   # GAP: T then V
        out = self.classifier(x)
        if multi_body:
            out = (out[:B] + out[B:]) / 2
        return out


# ---------------------------------------------------------------------------
# ShiftFuseZeroX  (B4-style scaled up to ~2M — 3-stream mid-fusion)
# ---------------------------------------------------------------------------
class ShiftFuseZeroX(nn.Module):
    """B4-style 3-stream mid-fusion scaled to ~2M params.

    Same structural pattern as ShiftFuseZeroB4 (per-stream backbones →
    mid-fusion concat → shared backbone), but with wider channels and
    deeper shared blocks. Novel additions over large (B4): Part_Att on
    last per-stream block and last shared block.

    Per-stream (×3: joint / velocity / bone):
        Stem:        BN(3) → B4StemGCN(3→96) → B4BasicTCN(96, k=5)
        stream_blk0: B4Block(96→128, depth=3, stride=1)   BRASP+SGPShift+JE+STJointAtt
        stream_blk1: B4Block(128→64, depth=3, stride=1)   BRASP+SGPShift+JE+STJointAtt+Part_Att

    Mid-fusion: concat 3×64 = 192

    Shared:
        shared_blk0: B4Block(192→256, depth=4, stride=2)  BRASP+SGPShift+JE+STJointAtt
        shared_blk1: B4Block(256→384, depth=4, stride=2)  BRASP+SGPShift+JE+STJointAtt+Part_Att+TLA

    Head: GAP → Dropout(0.20) → Linear(384, num_classes)

    Param estimate: ~2.0M
    Target: 93–94% NTU-60 xsub.
    """

    STREAM_NAMES    = ['joint', 'velocity', 'bone']
    STREAM_STEM     = 64   # same as large (B4-exact per-stream)
    # (c_in, c_out, depth, stride) — same as large per-stream, but last block gets Part_Att
    STREAM_BLOCKS   = [(64, 96, 2, 1), (96, 48, 2, 1)]
    # Wider shared backbone to reach ~2M total (vs large's [(144,128,3,2),(128,272,3,2)])
    SHARED_BLOCKS   = [(144, 192, 4, 2), (192, 320, 4, 2)]
    DROPOUT         = 0.20
    TLA_K           = 14
    TLA_REDUCE      = 8
    PART_ATT_REDUCE = 8   # inner = c_out//8 for Part_Att in X

    def __init__(
        self,
        num_classes:  int   = 60,
        graph_layout: str   = 'ntu-rgb+d',
        num_joints:   int   = 25,
        in_channels:  int   = 3,
        dropout:      float = None,
    ):
        super().__init__()
        _dropout = dropout if dropout is not None else self.DROPOUT

        # ── Semantic body-part graph — BRASP + SGPShift ───────────────────────
        sgp_graph = Graph(layout=graph_layout, strategy='semantic_bodypart',
                          max_hop=2, raw_partitions=True)
        sgp_raw   = sgp_graph.A
        A_sym     = normalize_symdigraph_full(sgp_raw)
        A_intra   = torch.tensor(A_sym[0], dtype=torch.float32)
        A_inter   = torch.tensor(A_sym[1], dtype=torch.float32)
        A_flat    = torch.tensor((sgp_raw.sum(0) > 0).astype('float32'))
        self.register_buffer('A_intra', A_intra)
        self.register_buffer('A_inter', A_inter)
        self.register_buffer('A_flat',  A_flat)

        # ── Spatial K=3 graph pre-normalized D^{-1}A — GCN ───────────────────
        import numpy as np
        spatial_graph = Graph(layout=graph_layout, strategy='spatial',
                              max_hop=1, raw_partitions=True)
        A_gcn_parts   = [
            torch.tensor(normalize_digraph(spatial_graph.A[k]), dtype=torch.float32)
            for k in range(spatial_graph.A.shape[0])
        ]

        def _blk(c_in, c_out, depth, stride, use_tla=False, use_part_att=False):
            return B4Block(c_in, c_out, depth, stride,
                           A_flat=A_flat, A_intra=A_intra, A_inter=A_inter,
                           A_gcn_partitions=A_gcn_parts, num_joints=num_joints,
                           use_tla=use_tla,
                           tla_landmarks=self.TLA_K,
                           tla_reduce_ratio=self.TLA_REDUCE,
                           use_part_att=use_part_att,
                           part_att_reduce=self.PART_ATT_REDUCE)

        # ── Per-stream stems ──────────────────────────────────────────────────
        def _stem():
            stem_gcn = B4StemGCN(in_channels, self.STREAM_STEM, A_gcn_parts, num_joints)
            stem_tcn = _B4BasicTCN(self.STREAM_STEM)
            return _B4Stem(in_channels, stem_gcn, stem_tcn)

        self.stream_stems = nn.ModuleList([_stem() for _ in range(3)])

        # ── Per-stream blocks (last block per stream gets Part_Att) ──────────
        n_stream_blks = len(self.STREAM_BLOCKS)
        self.stream_blocks = nn.ModuleList([
            nn.ModuleList([
                _blk(c_in, c_out, depth, stride,
                     use_part_att=(bi == n_stream_blks - 1))
                for bi, (c_in, c_out, depth, stride) in enumerate(self.STREAM_BLOCKS)
            ])
            for _ in range(3)
        ])

        # ── Shared blocks (last shared block gets TLA + Part_Att) ────────────
        n_shared = len(self.SHARED_BLOCKS)
        self.shared_blocks = nn.ModuleList([
            _blk(c_in, c_out, depth, stride,
                 use_tla=(i == n_shared - 1),
                 use_part_att=(i == n_shared - 1))
            for i, (c_in, c_out, depth, stride) in enumerate(self.SHARED_BLOCKS)
        ])

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=_dropout),
            nn.Linear(self.SHARED_BLOCKS[-1][1], num_classes),
        )

    def forward(self, stream_dict: dict) -> torch.Tensor:
        raw        = [stream_dict[n] for n in self.STREAM_NAMES]
        B          = raw[0].shape[0]
        multi_body = raw[0].dim() == 5
        if multi_body:
            raw = [torch.cat([s[..., 0], s[..., 1]], dim=0) for s in raw]

        feats = []
        for s, stem, blocks in zip(raw, self.stream_stems, self.stream_blocks):
            x = stem(s)
            for blk in blocks:
                x = blk(x)
            feats.append(x)

        x = torch.cat(feats, dim=1)
        for blk in self.shared_blocks:
            x = blk(x)

        x   = x.mean(dim=2).mean(dim=-1)   # GAP: T then V
        out = self.classifier(x)
        if multi_body:
            out = (out[:B] + out[B:]) / 2
        return out


# Minimal stem helpers (used only by ShiftFuseZeroB4)
class _B4BasicTCN(nn.Module):
    """B4-exact stem temporal conv: full Conv(C,C,k=5) + BN + Identity residual + SiLU."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, (5, 1), 1, (2, 0), bias=True)
        self.bn   = nn.BatchNorm2d(channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(x)) + x, inplace=True)


class _B4Stem(nn.Module):
    """Wrapper: BN(input) → B4StemGCN → _B4BasicTCN."""
    def __init__(self, in_channels: int, gcn: nn.Module, tcn: nn.Module):
        super().__init__()
        self.bn  = nn.BatchNorm2d(in_channels)
        self.gcn = gcn
        self.tcn = tcn
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(self.gcn(self.bn(x)))


def build_shiftfuse_zero_b4(num_classes: int = 60, **kwargs) -> ShiftFuseZeroB4:
    """Build large_b4_efficient (B4-exact + BRASP + SGPShift, target ~1.1M)."""
    return ShiftFuseZeroB4(num_classes=num_classes, **kwargs)


def build_shiftfuse_zero_x(num_classes: int = 60, **kwargs) -> ShiftFuseZeroX:
    """Build x_b4_efficient (B4-style scaled to ~2M + Part_Att, target 93-94%)."""
    return ShiftFuseZeroX(num_classes=num_classes, **kwargs)
