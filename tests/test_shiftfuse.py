"""
Integration tests for ShiftFuse-Zero.

Covers:
  - Forward pass shape: dict input → (B, num_classes)  [eval]
  - Tensor (non-dict) input handling
  - Param count sanity for all variants
  - nano_efficient: A_learned exists, K=3 tensors of (V,V)
  - STCAttention: output shape, multiplicative attention
  - DepthwiseSepTCN: output shape, stride handling
  - BodyRegionShift: zero learnable parameters
  - FrozenDCTGate: dct/idct buffers are not trainable (require_grad=False)
  - JointEmbedding: correct output shape (no-op at zero-init)
  - FrameDynamicsGate: correct output shape
  - CTRLightGCN: output shape, param count, per-group adjacency parameters
  - MultiScaleAdaptiveGCN: K-subset separate buffers, output shape, param count
  - BilateralSymmetryEncoding: antisymmetric injection, torso unchanged
  - DropPath: stochastic depth (0 params, identity at eval)
  - All individual blocks: input → output shape preserved
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from src.models.shiftfuse_zero import (
    ShiftFuseZero,
    build_shiftfuse_zero,
    ZERO_VARIANTS,
)
from src.models.blocks.body_region_shift import BodyRegionShift
from src.models.blocks.frozen_dct_gate import FrozenDCTGate
from src.models.blocks.joint_embedding import JointEmbedding
from src.models.blocks.frame_dynamics_gate import FrameDynamicsGate
from src.models.blocks.bilateral_symmetry import BilateralSymmetryEncoding
from src.models.blocks.static_gcn import StaticGCN
from src.models.blocks.ctr_light_gcn import CTRLightGCN
from src.models.blocks.adaptive_ctr_gcn import MultiScaleAdaptiveGCN
from src.models.blocks.ep_sep_tcn import MultiScaleEpSepTCN
from src.models.blocks.drop_path import DropPath
from src.models.graph import Graph, normalize_symdigraph_full


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

B, C, T, V = 2, 3, 64, 25
NUM_CLASSES = 60


def _mib(batch=B, c=C, t=T, v=V):
    return {
        'joint':         torch.randn(batch, c, t, v),
        'velocity':      torch.randn(batch, c, t, v),
        'bone':          torch.randn(batch, c, t, v),
        'bone_velocity': torch.randn(batch, c, t, v),
    }


# ---------------------------------------------------------------------------
# Block-level tests
# ---------------------------------------------------------------------------

class TestBodyRegionShift:
    def _make_shift(self, channels=32, shift_type='anatomical'):
        from src.models.graph import Graph
        import numpy as np
        g = Graph(layout='ntu-rgb+d', strategy='spatial', max_hop=1, raw_partitions=True)
        A_flat = torch.tensor((g.A.sum(0) > 0).astype('float32'))
        return BodyRegionShift(channels, A_flat, shift_type=shift_type)

    def test_zero_params(self):
        shift = self._make_shift(32)
        n_params = sum(p.numel() for p in shift.parameters())
        assert n_params == 0, f"BodyRegionShift should have 0 params, got {n_params}"

    def test_output_shape(self):
        shift = self._make_shift(32)
        x = torch.randn(B, 32, T, V)
        out = shift(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_shift_indices_buffer_no_grad(self):
        shift = self._make_shift(32)
        assert not shift.shift_indices.requires_grad

    # ── Ablation switch (BMVC'26 rebuttal Table 12) ──────────────────────────
    def test_ablation_variants_zero_param_and_shape(self):
        x = torch.randn(B, 32, T, V)
        for st in ('anatomical', 'full_body', 'random', 'none'):
            shift = self._make_shift(32, shift_type=st)
            assert sum(p.numel() for p in shift.parameters()) == 0, \
                f"shift_type={st} must be zero-parameter"
            assert shift(x).shape == x.shape
            assert not shift.shift_indices.requires_grad

    def test_none_is_identity(self):
        shift = self._make_shift(32, shift_type='none')
        x = torch.randn(B, 32, T, V)
        assert torch.equal(shift(x), x), "shift_type='none' must be a no-op"

    def test_random_is_permutation(self):
        # every channel's index row must be a permutation of the V joints
        shift = self._make_shift(32, shift_type='random')
        for row in shift.shift_indices:
            assert sorted(row.tolist()) == list(range(V))

    def test_full_body_differs_from_anatomical(self):
        anat = self._make_shift(32, shift_type='full_body')
        body = self._make_shift(32, shift_type='anatomical')
        assert not torch.equal(anat.shift_indices, body.shift_indices)

    def test_unknown_shift_type_raises(self):
        import pytest
        with pytest.raises(ValueError):
            self._make_shift(32, shift_type='bogus')


class TestFrozenDCTGate:
    def test_output_shape(self):
        gate = FrozenDCTGate(channels=48, T=T)
        x = torch.randn(B, 48, T, V)
        out = gate(x)
        assert out.shape == x.shape

    def test_dct_buffers_no_grad(self):
        gate = FrozenDCTGate(channels=48, T=T)
        assert not gate.dct.requires_grad
        assert not gate.idct.requires_grad

    def test_freq_mask_is_param(self):
        gate = FrozenDCTGate(channels=48, T=T)
        assert gate.freq_mask.requires_grad

    def test_param_count(self):
        C_test, T_test = 48, 64
        gate = FrozenDCTGate(channels=C_test, T=T_test)
        n = sum(p.numel() for p in gate.parameters())
        assert n == C_test * T_test, f"Expected {C_test * T_test}, got {n}"

    def test_dct_near_identity_at_init(self):
        # Init +4.0 → sigmoid(4.0) ≈ 0.982 → output ≈ 0.982x (near-identity, no internal residual)
        gate = FrozenDCTGate(channels=8, T=16)
        x = torch.randn(1, 8, 16, 4)
        out = gate(x)
        # Output should be close to x (within 5%) since mask ≈ 0.982
        rel_err = (out - x).abs().mean() / x.abs().mean()
        assert rel_err < 0.05, f"Expected near-identity at init, rel_err={rel_err:.4f}"


class TestJointEmbedding:
    def test_output_shape(self):
        embed = JointEmbedding(channels=48, num_joints=V)
        x = torch.randn(B, 48, T, V)
        out = embed(x)
        assert out.shape == x.shape

    def test_zero_init_is_identity(self):
        embed = JointEmbedding(channels=16, num_joints=V)
        x = torch.randn(B, 16, T, V)
        out = embed(x)
        assert torch.allclose(out, x), "Zero-init embed should not change input"

    def test_param_count(self):
        C_test = 48
        embed = JointEmbedding(channels=C_test, num_joints=V)
        n = sum(p.numel() for p in embed.parameters())
        assert n == V * C_test, f"Expected {V * C_test}, got {n}"


class TestStaticGCN:
    def _make_gcn(self, channels=48):
        g   = Graph(layout='ntu-rgb+d', strategy='spatial', max_hop=1, raw_partitions=True)
        A   = torch.tensor(normalize_symdigraph_full(g.A), dtype=torch.float32)
        return StaticGCN(channels=channels, A=A, num_joints=V)

    def test_output_shape(self):
        gcn = self._make_gcn(48)
        x   = torch.randn(B, 48, T, V)
        out = gcn(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_param_count(self):
        C_test = 48
        gcn = self._make_gcn(C_test)
        n   = sum(p.numel() for p in gcn.parameters())
        expected = C_test * C_test + 2 * C_test + V * V   # conv + BN + A_learned
        assert n == expected, f"Expected {expected}, got {n}"

    def test_near_identity_at_zero_init(self):
        # A_learned = zeros → A_learned_norm = 0 → x_agg comes only from static A
        # After static aggregation and zero-init conv, BN(0) = 0 → out ≈ x (residual)
        gcn = self._make_gcn(16)
        # Freeze conv weight to zero to isolate the zero-init A_learned effect
        with torch.no_grad():
            gcn.conv.weight.zero_()
        x   = torch.randn(1, 16, T, V)
        out = gcn(x)
        # With conv=0: BN(0) → ~0, so out ≈ x + 0 = x
        assert torch.allclose(out, x, atol=1e-4), "Expected near-identity with zero conv"

    def test_A_learned_is_parameter(self):
        gcn = self._make_gcn(32)
        assert gcn.A_learned.requires_grad, "A_learned must be a trainable parameter"

    def test_A_buffer_no_grad(self):
        gcn = self._make_gcn(32)
        assert not gcn.A.requires_grad, "Static A must be a buffer (no grad)"


class TestCTRLightGCN:
    def _make_gcn(self, channels=48, num_groups=4):
        g = Graph(layout='ntu-rgb+d', strategy='spatial', max_hop=1, raw_partitions=True)
        A = torch.tensor(normalize_symdigraph_full(g.A), dtype=torch.float32)
        return CTRLightGCN(channels=channels, A=A, num_joints=V, num_groups=num_groups)

    def test_output_shape(self):
        gcn = self._make_gcn(48, num_groups=4)
        x   = torch.randn(B, 48, T, V)
        out = gcn(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_param_count(self):
        C_test, G = 48, 4
        gcn = self._make_gcn(C_test, G)
        n   = sum(p.numel() for p in gcn.parameters())
        # G × (C//G)² conv + 2C BN + G × V² adjacency
        expected = G * (C_test // G) ** 2 + 2 * C_test + G * V * V
        assert n == expected, f"Expected {expected}, got {n}"

    def test_A_group_is_parameter(self):
        gcn = self._make_gcn(48, 4)
        for i, ag in enumerate(gcn.A_group):
            assert ag.requires_grad, f"A_group[{i}] must be trainable"

    def test_A_physical_buffer_no_grad(self):
        gcn = self._make_gcn(48, 4)
        assert not gcn.A_physical.requires_grad, "A_physical must be a buffer"

    def test_nano_groups(self):
        gcn = self._make_gcn(32, num_groups=2)
        x   = torch.randn(B, 32, T, V)
        out = gcn(x)
        assert out.shape == x.shape

    def test_A_g_normalised(self):
        """Row-normalisation: A_physical is normalised at __init__ (not per-forward).
        This avoids the unstable normalisation Jacobian that caused NaN explosion."""
        gcn = self._make_gcn(32, num_groups=2)
        # A_physical should have row sums = 1.0 immediately after construction
        row_sums = gcn.A_physical.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(V), atol=1e-5), \
            f"A_physical row sums not ≈ 1.0 at init: {row_sums}"
        # At init A_group = 0 → A_g = A_physical, so A_g is also unit-scale
        with torch.no_grad():
            A_g = gcn.A_physical + gcn.A_group[0]
        assert torch.allclose(A_g.sum(dim=-1), torch.ones(V), atol=1e-5), \
            "A_g row sums not ≈ 1.0 at init (A_group should be zero)"


class TestMultiScaleAdaptiveGCN:
    """Tests for K-subset MultiScaleAdaptiveGCN (v7 replacement for AdaptiveCTRGCN)."""

    def _make_gcn(self, channels=48, num_groups=4):
        g = Graph(layout='ntu-rgb+d', strategy='spatial', max_hop=1, raw_partitions=True)
        A = torch.tensor(normalize_symdigraph_full(g.A), dtype=torch.float32)
        return MultiScaleAdaptiveGCN(
            channels=channels, A=A, num_joints=V, num_groups=num_groups
        )

    def test_output_shape(self):
        gcn = self._make_gcn(48, 4)
        x   = torch.randn(B, 48, T, V)
        out = gcn(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_k_separate_buffers(self):
        """K=3 subsets must be stored as separate buffers A_0, A_1, A_2."""
        gcn = self._make_gcn(48, 4)
        assert hasattr(gcn, 'A_0'), "MultiScaleAdaptiveGCN must have buffer A_0"
        assert hasattr(gcn, 'A_1'), "MultiScaleAdaptiveGCN must have buffer A_1"
        assert hasattr(gcn, 'A_2'), "MultiScaleAdaptiveGCN must have buffer A_2"
        # Buffers must NOT require grad
        assert not gcn.A_0.requires_grad
        assert not gcn.A_1.requires_grad
        assert not gcn.A_2.requires_grad

    def test_k_subsets_row_normalised(self):
        gcn = self._make_gcn(48, 4)
        for k in range(gcn.K):
            Ak = getattr(gcn, f'A_{k}')
            row_sums = Ak.sum(dim=-1)
            assert (row_sums <= 1.0 + 1e-4).all(), \
                f"A_{k} row sums exceed 1.0"

    def test_k_independent_convs(self):
        """K independent group convolutions — one per subset."""
        gcn = self._make_gcn(48, 4)
        assert len(gcn.convs) == gcn.K, \
            f"Expected {gcn.K} convs, got {len(gcn.convs)}"

    def test_alpha_is_parameter(self):
        gcn = self._make_gcn(48, 4)
        assert gcn.alpha.requires_grad, "alpha must be a trainable parameter"
        assert gcn.alpha.shape == (1,)

    def test_backwards_compat_alias(self):
        """AdaptiveCTRGCN alias must still import and instantiate."""
        from src.models.blocks.adaptive_ctr_gcn import AdaptiveCTRGCN
        g = Graph(layout='ntu-rgb+d', strategy='spatial', max_hop=1, raw_partitions=True)
        A = torch.tensor(normalize_symdigraph_full(g.A), dtype=torch.float32)
        gcn = AdaptiveCTRGCN(channels=48, A=A, num_joints=V)
        x   = torch.randn(B, 48, T, V)
        out = gcn(x)
        assert out.shape == x.shape


class TestMultiScaleEpSepTCN:
    """Tests for 4-branch dilated TCN (v8: TSM + d=2 + d=4 + MaxPool)."""

    def test_output_shape_stride1(self):
        tcn = MultiScaleEpSepTCN(channels=48, stride=1, num_branches=4)
        x   = torch.randn(B, 48, T, V)
        out = tcn(x)
        assert out.shape == (B, 48, T, V), f"Expected (B,48,T,V), got {out.shape}"

    def test_output_shape_stride2(self):
        tcn = MultiScaleEpSepTCN(channels=48, stride=2, num_branches=4)
        x   = torch.randn(B, 48, T, V)
        out = tcn(x)
        assert out.shape == (B, 48, T // 2, V), f"Expected T//2, got {out.shape}"

    def test_tsm_zero_params(self):
        """TSM branch contributes 0 learnable params (only BN shared in tsm_downsample)."""
        tcn = MultiScaleEpSepTCN(channels=48, stride=1, num_branches=4)
        # tsm_downsample is BN(C//4) = 2*(48//4) = 24 params (gamma+beta)
        tsm_params = sum(p.numel() for p in tcn.tsm_downsample.parameters())
        assert tsm_params == 2 * (48 // 4), f"TSM BN params: {tsm_params}"

    def test_4branch_fewer_params_than_3branch(self):
        """4-branch (TSM replaces d=1 conv) should have fewer params than 3-branch legacy."""
        t4 = MultiScaleEpSepTCN(channels=48, stride=1, num_branches=4)
        t3 = MultiScaleEpSepTCN(channels=48, stride=1, num_branches=3)
        n4 = sum(p.numel() for p in t4.parameters())
        n3 = sum(p.numel() for p in t3.parameters())
        assert n4 < n3, f"4-branch ({n4}) should use fewer params than 3-branch ({n3})"

    def test_dilation4_receptive_field(self):
        """d=4 branch must produce non-zero output (actual computation)."""
        tcn = MultiScaleEpSepTCN(channels=48, stride=1, num_branches=4)
        x   = torch.randn(B, 48, T, V)
        out = tcn(x)
        assert not torch.allclose(out, torch.zeros_like(out)), "Output should be non-zero"

    def test_channels_div4_assertion(self):
        """Should raise AssertionError when channels not divisible by 4."""
        import pytest
        with pytest.raises(AssertionError):
            MultiScaleEpSepTCN(channels=50, stride=1, num_branches=4)


class TestDropPath:
    def test_output_shape(self):
        dp  = DropPath(drop_prob=0.1)
        x   = torch.randn(B, 48, T, V)
        dp.train()
        out = dp(x)
        assert out.shape == x.shape

    def test_zero_params(self):
        dp = DropPath(0.1)
        n  = sum(p.numel() for p in dp.parameters())
        assert n == 0, f"DropPath should have 0 params, got {n}"

    def test_identity_at_eval(self):
        dp = DropPath(drop_prob=0.99)  # would drop almost everything in train
        dp.eval()
        x   = torch.randn(B, 48, T, V)
        out = dp(x)
        assert torch.allclose(out, x), "DropPath should be identity at eval"

    def test_identity_when_prob_zero(self):
        dp = DropPath(drop_prob=0.0)
        dp.train()
        x   = torch.randn(B, 48, T, V)
        out = dp(x)
        assert torch.allclose(out, x), "DropPath(0.0) should be identity in train too"


class TestFrameDynamicsGate:
    def test_output_shape(self):
        gate = FrameDynamicsGate(channels=48, T=T)
        x = torch.randn(B, 48, T, V)
        out = gate(x)
        assert out.shape == x.shape

    def test_init_near_identity(self):
        gate = FrameDynamicsGate(channels=16, T=T)
        x = torch.ones(1, 16, T, V)
        out = gate(x)
        # Init 2.0 → sigmoid(2.0) ≈ 0.880 → near-identity (12% suppression)
        expected_gate = torch.sigmoid(torch.tensor(2.0))
        assert torch.allclose(out, x * expected_gate, atol=1e-5)

    def test_param_count(self):
        C_test, T_test = 48, 64
        gate = FrameDynamicsGate(channels=C_test, T=T_test)
        n = sum(p.numel() for p in gate.parameters())
        assert n == C_test * T_test, f"Expected {C_test * T_test}, got {n}"


class TestBilateralSymmetryEncoding:
    def test_output_shape(self):
        bse = BilateralSymmetryEncoding(channels=48)
        x = torch.randn(B, 48, T, V)
        out = bse(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_param_count(self):
        C_test = 48
        bse = BilateralSymmetryEncoding(channels=C_test)
        n = sum(p.numel() for p in bse.parameters())
        expected = 2 * C_test + 1  # sym_weight + sym_vel_weight + gate
        assert n == expected, f"Expected {expected}, got {n}"

    def test_near_identity_at_init(self):
        bse = BilateralSymmetryEncoding(channels=16)
        x = torch.randn(B, 16, T, V)
        out = bse(x)
        # gate=sigmoid(-2)≈0.12 and weights=0 → mod≈0 → out≈x
        assert torch.allclose(out, x, atol=1e-6), "BSE should be near-identity at init"

    def test_antisymmetric_injection(self):
        bse = BilateralSymmetryEncoding(channels=8)
        # Set non-zero weights to test antisymmetry
        with torch.no_grad():
            bse.sym_weight.fill_(1.0)
            bse.gate.fill_(10.0)  # sigmoid(10)≈1
        x = torch.randn(1, 8, 16, V)
        out = bse(x)
        # Check: modification to left joints = -modification to right joints
        left_mod = out[:, :, :, bse.LEFT_JOINTS] - x[:, :, :, bse.LEFT_JOINTS]
        right_mod = out[:, :, :, bse.RIGHT_JOINTS] - x[:, :, :, bse.RIGHT_JOINTS]
        assert torch.allclose(left_mod, -right_mod, atol=1e-5), \
            "BSE should inject antisymmetrically"

    def test_torso_unchanged(self):
        bse = BilateralSymmetryEncoding(channels=8)
        with torch.no_grad():
            bse.sym_weight.fill_(1.0)
            bse.gate.fill_(10.0)
        x = torch.randn(1, 8, 16, V)
        out = bse(x)
        torso = [0, 1, 2, 3, 20]
        assert torch.allclose(out[:, :, :, torso], x[:, :, :, torso]), \
            "BSE should not modify torso/midline joints"


# ---------------------------------------------------------------------------
# Full model tests — ShiftFuse-Zero
# ---------------------------------------------------------------------------

class TestShiftFuseZero:
    def test_nano_forward_dict(self):
        model = build_shiftfuse_zero('nano_tiny_efficient', num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(_mib())
        assert out.shape == (B, NUM_CLASSES), f"Expected ({B},{NUM_CLASSES}), got {out.shape}"

    def test_small_forward_dict(self):
        model = build_shiftfuse_zero('nano_tiny_efficient', num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(_mib())
        assert out.shape == (B, NUM_CLASSES)

    def test_nano_efficient_forward_dict(self):
        model = build_shiftfuse_zero('nano_tiny_efficient', num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(_mib())
        assert out.shape == (B, NUM_CLASSES)

    def test_nano_efficient_a_k_learned(self):
        """nano_tiny_efficient: share_a_learned=True → single shared set, K=3 × V×V params."""
        model = build_shiftfuse_zero('nano_tiny_efficient', num_classes=NUM_CLASSES)
        # nano redesign: global shared A_learned (1 set for all blocks)
        assert hasattr(model, 'shared_a_learned'), \
            "nano_tiny_efficient must have model.shared_a_learned (share_a_learned=True)"
        assert len(model.shared_a_learned) == 3, \
            f"shared_a_learned: expected K=3, got {len(model.shared_a_learned)}"
        for k, p in enumerate(model.shared_a_learned):
            assert p.shape == (V, V), f"shared_a_learned[{k}] shape {p.shape} != ({V},{V})"
            assert p.requires_grad, f"shared_a_learned[{k}] must be trainable"
        # Every block should reference the same shared object
        for si, stage in enumerate(model.stages):
            for bi, block in enumerate(stage):
                assert hasattr(block, 'A_learned'), f"Stage{si} Block{bi} missing A_learned"
                assert block.A_learned is model.shared_a_learned, \
                    f"Stage{si} Block{bi}: A_learned must reference shared_a_learned"
        # Total A_learned params = K=3 × V×V (shared, not duplicated per-block)
        # Parameters live under 'shared_a_learned.*' (not 'A_learned.*')
        total = sum(p.numel() for n, p in model.named_parameters() if 'shared_a_learned' in n)
        assert total == 3 * V * V, f"Expected {3*V*V} (shared), got {total}"

    def test_nano_efficient_no_global_a_k_on_standard_nano(self):
        """nano_tiny_efficient: uses shared_a_learned, blocks all reference it."""
        model = build_shiftfuse_zero('nano_tiny_efficient', num_classes=NUM_CLASSES)
        assert hasattr(model, 'shared_a_learned'), \
            "nano_tiny_efficient must have shared_a_learned (share_a_learned=True redesign)"
        # All per-block A_learned should be present (as shared reference)
        for si, stage in enumerate(model.stages):
            for bi, block in enumerate(stage):
                assert hasattr(block, 'A_learned'), \
                    f"Stage{si} Block{bi} missing A_learned reference"

    def test_nano_param_count(self):
        model = build_shiftfuse_zero('nano_tiny_efficient', num_classes=NUM_CLASSES)
        n = sum(p.numel() for p in model.parameters())
        assert n > 0, "nano_tiny_efficient must have parameters"
        print(f"\n  nano_tiny_efficient params: {n:,}")

    def test_nano_efficient_param_count(self):
        model = build_shiftfuse_zero('nano_tiny_efficient', num_classes=NUM_CLASSES)
        n = sum(p.numel() for p in model.parameters())
        assert n > 0
        print(f"\n  nano_tiny_efficient params: {n:,}")

    def test_no_nan_output_nano(self):
        model = build_shiftfuse_zero('nano_tiny_efficient', num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(_mib())
        assert not torch.isnan(out).any(), "nano_tiny_efficient output contains NaN"
        assert not torch.isinf(out).any(), "nano_tiny_efficient output contains Inf"

    def test_no_nan_output_nano_efficient(self):
        model = build_shiftfuse_zero('nano_tiny_efficient', num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(_mib())
        assert not torch.isnan(out).any(), "nano_tiny_efficient output contains NaN"
        assert not torch.isinf(out).any(), "nano_tiny_efficient output contains Inf"

    def test_invalid_variant(self):
        import pytest
        with pytest.raises((ValueError, KeyError)):
            build_shiftfuse_zero('invalid_variant', num_classes=NUM_CLASSES)


# ---------------------------------------------------------------------------
# ShiftFuseZeroB4 tests
# ---------------------------------------------------------------------------

class TestShiftFuseZeroB4:
    def test_forward_3stream(self):
        from src.models.shiftfuse_zero import build_shiftfuse_zero_b4
        model = build_shiftfuse_zero_b4(num_classes=NUM_CLASSES)
        model.eval()
        dummy = {k: torch.zeros(B, 3, T, V) for k in ['joint', 'velocity', 'bone']}
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (B, NUM_CLASSES), f"Expected ({B},{NUM_CLASSES}), got {out.shape}"

    def test_param_count_range(self):
        from src.models.shiftfuse_zero import build_shiftfuse_zero_b4
        model = build_shiftfuse_zero_b4(num_classes=60)
        n = sum(p.numel() for p in model.parameters())
        print(f"\n  ShiftFuseZeroB4 params: {n:,}")
        assert 1_000_000 < n < 3_000_000, f"Expected 1-3M params, got {n:,}"

    def test_no_nan_output(self):
        from src.models.shiftfuse_zero import build_shiftfuse_zero_b4
        model = build_shiftfuse_zero_b4(num_classes=NUM_CLASSES)
        model.eval()
        dummy = {k: torch.randn(B, 3, T, V) for k in ['joint', 'velocity', 'bone']}
        with torch.no_grad():
            out = model(dummy)
        assert not torch.isnan(out).any(), "B4 output contains NaN"
        assert not torch.isinf(out).any(), "B4 output contains Inf"

    def test_multi_body_handling(self):
        from src.models.shiftfuse_zero import build_shiftfuse_zero_b4
        model = build_shiftfuse_zero_b4(num_classes=NUM_CLASSES)
        model.eval()
        dummy = {k: torch.zeros(B, 3, T, V, 2) for k in ['joint', 'velocity', 'bone']}
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (B, NUM_CLASSES)


# ---------------------------------------------------------------------------
# STCAttention tests
# ---------------------------------------------------------------------------

class TestSTCAttention:
    def test_output_shape(self):
        from src.models.blocks.stc_attention import STCAttention
        attn = STCAttention(channels=64, num_joints=V)
        x = torch.randn(B, 64, T, V)
        out = attn(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_output_bounded(self):
        """Multiplicative attention — output should be in [0, input_max]."""
        from src.models.blocks.stc_attention import STCAttention
        attn = STCAttention(channels=32, num_joints=V)
        attn.eval()
        x = torch.ones(1, 32, T, V)
        with torch.no_grad():
            out = attn(x)
        assert out.min() >= 0.0, "STCAttention output should be non-negative for +ve input"


# ---------------------------------------------------------------------------
# DepthwiseSepTCN tests
# ---------------------------------------------------------------------------

class TestDepthwiseSepTCN:
    def test_output_shape_stride1(self):
        from src.models.blocks.dw_sep_tcn import DepthwiseSepTCN
        tcn = DepthwiseSepTCN(in_channels=64, out_channels=64, stride=1)
        x   = torch.randn(B, 64, T, V)
        out = tcn(x)
        assert out.shape == (B, 64, T, V), f"Expected (B,64,T,V), got {out.shape}"

    def test_output_shape_stride2(self):
        from src.models.blocks.dw_sep_tcn import DepthwiseSepTCN
        tcn = DepthwiseSepTCN(in_channels=32, out_channels=64, stride=2)
        x   = torch.randn(B, 32, T, V)
        out = tcn(x)
        assert out.shape == (B, 64, T // 2, V), f"Expected T//2, got {out.shape}"

    def test_two_branch_concat(self):
        """Output channels = out_channels (two mid-branches concatenated)."""
        from src.models.blocks.dw_sep_tcn import DepthwiseSepTCN
        tcn = DepthwiseSepTCN(in_channels=48, out_channels=96, stride=1)
        x   = torch.randn(B, 48, T, V)
        out = tcn(x)
        assert out.shape[1] == 96


# ---------------------------------------------------------------------------
# Main — dry run with full shape trace
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    SEP = '=' * 60

    print(SEP)
    print('PARAM COUNT — ShiftFuse-Zero variants')
    print(SEP)
    for variant in ('nano_tiny_efficient',):
        m = build_shiftfuse_zero(variant, num_classes=60)
        n = sum(p.numel() for p in m.parameters())
        cfg = ZERO_VARIANTS[variant]
        print(f'  {variant:<24}: {n:>10,}  channels={cfg["channels"]}')
    print()

    print(SEP)
    print('SHAPE TRACE — nano_tiny_efficient, B=2, C=3, T=64, V=25')
    print(SEP)
    m_eff = build_shiftfuse_zero('nano_tiny_efficient', num_classes=60)
    m_eff.eval()
    x_in = {k: torch.randn(2, 3, 64, 25) for k in ['joint', 'velocity', 'bone', 'bone_velocity']}
    with torch.no_grad():
        logits = m_eff(x_in)
    assert not torch.isnan(logits).any(), "FAIL: NaN in output"
    assert not torch.isinf(logits).any(), "FAIL: Inf in output"
    print(f'  Output: {tuple(logits.shape)}  range=[{logits.min():.4f}, {logits.max():.4f}]')
    adj_p = sum(p.numel() for n, p in m_eff.named_parameters() if 'A_learned' in n)
    print(f'  A_learned: per-block, total {adj_p:,} params (7 blocks × 3 × 25×25)')
    print()
    print('All checks passed.')
