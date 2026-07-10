"""
ShiftFuse-Zero Training Script

Usage:
    python scripts/train.py --model shiftfuse_zero_nano_tiny_efficient --dataset ntu60 --amp
    python scripts/train.py --model shiftfuse_zero_small_late_efficient --dataset ntu60 --amp
    python scripts/train.py --model shiftfuse_zero_large_b4_efficient --dataset ntu60 --amp
    python scripts/train.py --model shiftfuse_zero_nano_tiny_efficient --dataset ntu60 --amp \\
        --teacher_checkpoint /path/to/large_best.pth --kd_weight 0.5
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import random
import time
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from src.data.dataset import SkeletonDataset
from src.data.transforms import get_train_transform, get_val_transform
from src.training.trainer import Trainer
from src.utils.config import load_config, set_nested_key


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def count_flops(model, input_shape, device):
    """
    Estimate FLOPs using torch.profiler with_flops=True.

    Args:
        model:        ShiftFuse-Zero model
        input_shape:  tuple (C, T, V) — single stream, no batch or M dim
        device:       torch device

    Returns:
        dict with flops, params, memory info

    Notes:
        PyTorch profiler only counts FLOPs for a subset of ops (conv2d, mm, bmm).
        Raw torch.matmul calls inside the GCN loop are NOT captured, so the
        reported GFLOPs will be a lower bound — treat as a relative ranking tool
        rather than an absolute measure.  Consider replacing with fvcore or
        thop if accurate absolute FLOPs are needed.
    """
    model = model.to(device)
    model.eval()

    # Build a single-sample (B=1, C, T, V) stream dict — no M dimension
    C, T, V = input_shape[:3]
    x_single = torch.randn(1, C, T, V, device=device)
    x = {
        'joint':    x_single,
        'velocity': x_single,
        'bone':     x_single,
    }

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb    = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    buffer_size_mb   = sum(b.nelement() * b.element_size() for b in model.buffers()) / (1024 * 1024)

    flops = 0
    try:
        from torch.profiler import profile, ProfilerActivity
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
                model(x)
        for evt in prof.key_averages():
            if evt.flops:
                flops += evt.flops
    except Exception as e:
        print(f"  [Profiler Error] FLOPs unavailable: {e}")
        flops = 0

    gpu_memory_mb = 0
    if device.type == 'cuda':
        try:
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(x)
            gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass

    return {
        'total_params':        total_params,
        'trainable_params':    trainable_params,
        'param_size_mb':       round(param_size_mb, 2),
        'buffer_size_mb':      round(buffer_size_mb, 2),
        'total_model_size_mb': round(param_size_mb + buffer_size_mb, 2),
        'flops':               flops,
        'gflops':              round(flops / 1e9, 3),
        'gpu_memory_mb':       round(gpu_memory_mb, 2),
    }


def _run_checkpoint_averaging(model, val_loader, run_dir, n_checkpoints, device):
    """
    Average the last N epoch checkpoints and evaluate on val_loader.
    """
    import glob as _glob
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    ckpt_paths = sorted(
        _glob.glob(os.path.join(checkpoint_dir, 'checkpoint_ep*.pth')),
        key=lambda p: int(os.path.basename(p).replace('checkpoint_ep', '').replace('.pth', ''))
    )
    if len(ckpt_paths) == 0:
        print('\n[avg_checkpoints] No epoch checkpoints found — skipping.')
        return
    selected = ckpt_paths[-n_checkpoints:]
    print(f'\n[avg_checkpoints] Averaging {len(selected)} checkpoints:')
    for p in selected:
        print(f'  {os.path.basename(p)}')

    avg_state = None
    for path in selected:
        ckpt = torch.load(path, map_location=device)
        sd   = ckpt['model_state_dict']
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k in avg_state:
                avg_state[k] += sd[k].float()
    for k in avg_state:
        avg_state[k] /= len(selected)
        avg_state[k] = avg_state[k].to(
            next(iter(torch.load(selected[0], map_location=device)['model_state_dict'].values())).dtype
        )

    model.load_state_dict(avg_state)
    model.eval()

    correct1 = correct5 = total = 0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch
            elif isinstance(batch, dict):
                labels = batch.pop('label')
                inputs = batch
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            out = model(inputs)
            _, pred_top5 = out.topk(5, dim=1)
            correct1 += pred_top5[:, 0].eq(labels).sum().item()
            correct5 += pred_top5.eq(labels.unsqueeze(1).expand_as(pred_top5)).any(dim=1).sum().item()
            total    += labels.size(0)

    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    print(f'[avg_checkpoints] Averaged model — Val Top-1: {top1:.2f}%  Top-5: {top5:.2f}%')

    avg_path = os.path.join(checkpoint_dir, f'averaged_last{len(selected)}.pth')
    torch.save({'model_state_dict': avg_state, 'val_top1': top1}, avg_path)
    print(f'[avg_checkpoints] Saved to {avg_path}')


def main():
    parser = argparse.ArgumentParser(description='Train ShiftFuse-Zero model')
    parser.add_argument('--model', type=str, default='shiftfuse_zero_nano_tiny_efficient',
                        choices=[
                            'shiftfuse_zero_nano_tiny_efficient',
                            'shiftfuse_zero_small_late_efficient',
                            'shiftfuse_zero_medium_late_efficient',
                            'shiftfuse_zero_large_b4_efficient',
                            'shiftfuse_zero_x_efficient',
                        ],
                        help='Model variant')
    parser.add_argument('--dataset', type=str, default='ntu60', choices=['ntu60', 'ntu120'],
                        help='Dataset (default: ntu60)')
    parser.add_argument('--shift_type', type=str, default='anatomical',
                        choices=['anatomical', 'full_body', 'random', 'none'],
                        help='Spatial shift ablation (single-backbone variants). '
                             'anatomical=BRASP+SGPShift (default); full_body=Shift-GCN-style; '
                             'random=structure-free control; none=no shift. See BMVC rebuttal Table 12.')
    parser.add_argument('--split_type', type=str, default='xsub', choices=['xsub', 'xview', 'xset'],
                        help='Split type (default: xsub)')
    parser.add_argument('--epochs',     type=int,   default=None, help='Override epochs')
    parser.add_argument('--batch_size', type=int,   default=None, help='Override batch size')
    parser.add_argument('--lr',         type=float, default=None, help='Override learning rate')
    parser.add_argument('--resume',     type=str,   default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed',       type=int,   default=None, help='Random seed')
    parser.add_argument('--amp',        action='store_true',      help='Enable mixed precision')
    parser.add_argument('--workers',    type=int,   default=None, help='DataLoader workers')
    parser.add_argument('--env', type=str, default=None,
                        choices=['local', 'local_server', 'kaggle', 'gcp', 'lambda', 'a100'],
                        help='Environment (default: auto-detect)')
    parser.add_argument('--weight_decay', type=float, default=None, help='Override training.weight_decay')
    parser.add_argument('--dropout',      type=float, default=None, help='Override model.dropout')
    parser.add_argument('--scheduler',    type=str,   default=None,
                        help='Override training.scheduler (cosine_warmup | multistep_warmup | multistep)')
    parser.add_argument('--min_lr',       type=float, default=None, help='Override training.min_lr')
    parser.add_argument('--milestones',   type=str,   default=None,
                        help='Override training.milestones as comma-separated: --milestones 50,65')
    parser.add_argument('--data_root',    type=str,   default=None,
                        help='Override environment.paths.data_base')
    parser.add_argument('--set', nargs='*', metavar='KEY=VALUE', default=None,
                        help='Override any config key via dot-notation: '
                             '--set training.lr=0.1 training.epochs=400 model.dropout=0.1')
    parser.add_argument('--avg_checkpoints', type=int, default=0, metavar='N',
                        help='After training, average the last N epoch checkpoints and evaluate (0 = disabled)')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to teacher checkpoint for knowledge distillation')
    parser.add_argument('--kd_weight', type=float, default=None,
                        help='Override training.kd_weight (KD loss blend 0-1)')
    parser.add_argument('--kd_temp',   type=float, default=None,
                        help='Override training.kd_temperature')
    args = parser.parse_args()

    # ── 1. Load & merge config ──────────────────────────────────────────
    _training_cfg_map = {
        'shiftfuse_zero_nano_tiny_efficient':   'shiftfuse_zero_nano_efficient',
        'shiftfuse_zero_small_late_efficient':  'shiftfuse_zero_small_efficient',
        'shiftfuse_zero_medium_late_efficient': 'shiftfuse_zero_medium_efficient',
        'shiftfuse_zero_large_b4_efficient':    'shiftfuse_zero_large_efficient',
        'shiftfuse_zero_x_efficient':           'shiftfuse_zero_x_efficient',
    }
    _training_cfg_name = _training_cfg_map.get(args.model, 'shiftfuse_zero_nano_efficient')

    training_config_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'training', f'{_training_cfg_name}.yaml'
    )
    with open(training_config_path, 'r') as f:
        default_cfg = yaml.safe_load(f)

    specific_config = load_config(env=args.env, dataset=args.dataset, model=args.model)

    config = default_cfg
    config.update(specific_config)

    _hw = config.get('environment', {}).get('hardware', {})
    if _hw.get('num_workers') is not None and args.workers is None:
        config['training']['num_workers'] = _hw['num_workers']
    if 'pin_memory' in _hw:
        config['training']['pin_memory'] = _hw['pin_memory']

    for _k, _v in config.get('environment', {}).get('training_overrides', {}).items():
        config['training'][_k] = _v

    # CLI overrides
    if args.epochs     is not None: config['training']['epochs']     = args.epochs
    if args.batch_size is not None: config['training']['batch_size'] = args.batch_size
    if args.lr         is not None: config['training']['lr']         = args.lr
    if args.seed       is not None: config['training']['seed']       = args.seed
    if args.amp:                    config['training']['use_amp']    = True
    if args.workers    is not None: config['training']['num_workers'] = args.workers
    if args.weight_decay is not None: config['training']['weight_decay'] = args.weight_decay
    if args.dropout      is not None: config['model']['dropout']        = args.dropout
    if args.scheduler    is not None: config['training']['scheduler']   = args.scheduler
    if args.min_lr       is not None: config['training']['min_lr']      = args.min_lr
    if args.milestones   is not None:
        config['training']['milestones'] = [int(x) for x in args.milestones.split(',')]
    if args.data_root is not None:
        config['environment']['paths']['data_base'] = args.data_root
    if args.teacher_checkpoint is not None:
        config['training']['use_kd'] = True
        config['training']['teacher_checkpoint'] = args.teacher_checkpoint
    if args.kd_weight is not None: config['training']['kd_weight']     = args.kd_weight
    if args.kd_temp   is not None: config['training']['kd_temperature'] = args.kd_temp
    if args.set:
        for kv in args.set:
            key_path, raw_val = kv.split('=', 1)
            set_nested_key(config, key_path, raw_val)

    # ── 2. Setup ────────────────────────────────────────────────────────
    seed = config['training']['seed']
    set_seed(seed)

    if 'environment' in config and 'paths' in config['environment']:
        runs_root = config['environment']['paths'].get('output_root')
    else:
        runs_root = config.get('output', {}).get('runs_root', './LAST-runs')

    run_name = f"run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir  = os.path.join(runs_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)

    run_config = {
        'model_variant': args.model,
        'dataset':       args.dataset,
        'split_type':    args.split_type,
        'training':      config['training'],
        'seed':          seed,
        'start_time':    datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    print("=" * 70)
    print("  ShiftFuse-Zero — Training Pipeline")
    print("=" * 70)
    print(f"  Model:     {args.model}")
    print(f"  Dataset:   {args.dataset.upper()} ({args.split_type})")
    print(f"  Epochs:    {config['training']['epochs']}")
    print(f"  Batch:     {config['training']['batch_size']}")
    print(f"  LR:        {config['training']['lr']}")
    print(f"  Scheduler: {config['training']['scheduler']}")
    print(f"  Seed:      {seed}")
    print(f"  Run Dir:   {run_dir}")
    print("=" * 70)

    # ── 3. Model — dispatched from config['model']['variant'] ───────────
    num_classes = config['data']['dataset'].get('num_classes', 60 if args.dataset == 'ntu60' else 120)
    num_joints  = config['data']['dataset']['num_joints']

    from src.models.shiftfuse_zero import (
        build_shiftfuse_zero, build_shiftfuse_zero_late,
        build_shiftfuse_zero_b4, build_shiftfuse_zero_x,
    )

    variant = config['model']['variant']
    common  = dict(num_classes=num_classes, num_joints=num_joints,
                   dropout=config['model'].get('dropout'))

    _LATE_VARIANTS = {'small_late_efficient_bb', 'medium_late_efficient_bb'}

    print(f"\n  Building variant '{variant}'...")
    if variant in _LATE_VARIANTS:
        model = build_shiftfuse_zero_late(variant=variant, **common)
    elif variant == 'large_b4_efficient':
        model = build_shiftfuse_zero_b4(**common)
    elif variant == 'x_efficient':
        model = build_shiftfuse_zero_x(**common)
    else:
        # nano_tiny_efficient and any future single-backbone variants.
        # shift_type enables the BRASP/SGPShift ablation (BMVC rebuttal Table 12).
        if args.shift_type != 'anatomical':
            print(f"  Shift ablation: shift_type='{args.shift_type}' (BRASP/SGPShift replaced)")
        model = build_shiftfuse_zero(variant=variant, shift_type=args.shift_type, **common)

    print(f"  Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── 4. Measure FLOPs & memory ───────────────────────────────────────
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_frames = config['training']['input_frames']
    input_shape  = (3, input_frames, num_joints)   # (C, T, V) — single stream, B=1

    print(f"  Measuring FLOPs on input {input_shape}...")
    model_stats = count_flops(model, input_shape, device)

    print(f"  Parameters:  {model_stats['total_params']:,}")
    print(f"  Model Size:  {model_stats['total_model_size_mb']} MB")
    print(f"  FLOPs:       {model_stats['gflops']} GFLOPs")
    if model_stats['gpu_memory_mb'] > 0:
        print(f"  GPU Memory:  {model_stats['gpu_memory_mb']} MB (inference)")

    with open(os.path.join(run_dir, 'model_stats.json'), 'w') as f:
        json.dump(model_stats, f, indent=2)

    # ── 5. Data ─────────────────────────────────────────────────────────
    data_base   = config['environment']['paths']['data_base']
    _env_folder = config.get('environment', {}).get('paths', {}).get('data_folder')
    folder_name = _env_folder if _env_folder else ("LAST-60" if args.dataset == 'ntu60' else "LAST-120")
    processed_data_path = os.path.join(data_base, folder_name, "data", "processed")

    data_type = config['data']['dataset'].get('data_type', 'npy')
    print(f"\n  Loading data from: {processed_data_path}")

    merged_transform_config = {
        'dataset':  config['data']['dataset'],
        'training': config['training'],
    }
    merged_transform_config['dataset']['preprocessing']['normalize'] = False

    train_transform = get_train_transform(merged_transform_config)
    val_transform   = get_val_transform(merged_transform_config)
    print(f"  Train transforms: {train_transform}")
    print(f"  Val transforms:   {val_transform}")

    split_config = config['data']['dataset'].get('splits', None)

    train_dataset = SkeletonDataset(
        data_path=processed_data_path,
        data_type=data_type,
        max_frames=config['data']['dataset']['max_frames'],
        num_joints=num_joints,
        transform=train_transform,
        split='train',
        split_type=args.split_type,
        split_config=split_config,
    )
    val_dataset = SkeletonDataset(
        data_path=processed_data_path,
        data_type=data_type,
        max_frames=config['data']['dataset']['max_frames'],
        num_joints=num_joints,
        transform=val_transform,
        split='val',
        split_type=args.split_type,
        split_config=split_config,
    )

    batch_size   = config['training']['batch_size']
    num_workers  = config['training'].get('num_workers', 4)
    pin_memory   = config['training'].get('pin_memory', True)
    _pf          = config.get('environment', {}).get('hardware', {}).get('prefetch_factor', 2)
    prefetch_factor = _pf if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=num_workers > 0, prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0, prefetch_factor=prefetch_factor,
    )

    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # ── 6. Pre-training diagnostics ─────────────────────────────────────
    trainer = Trainer(model, config, run_dir)

    print(f"\n  --- Pre-training diagnostics ---")
    model.eval()

    train_batch_data, train_batch_labels = next(iter(train_loader))
    print(f"  Train batch keys={list(train_batch_data.keys())}, labels={train_batch_labels.shape}")
    for k, v in train_batch_data.items():
        print(f"    {k}: {v.shape}, mean={v.mean():.4f}, std={v.std():.4f}")
    print(f"  Train labels: {train_batch_labels[:10].tolist()}")

    val_batch_data, val_batch_labels = next(iter(val_loader))
    val_input = {k: v.to(device) for k, v in val_batch_data.items()}
    print(f"  Val batch keys={list(val_batch_data.keys())}, labels={val_batch_labels.shape}")
    print(f"  Val labels: {val_batch_labels[:10].tolist()}")

    with torch.no_grad():
        val_out   = model(val_input)
        _, val_preds = val_out.max(1)
        unique_preds = val_preds.unique()
        print(f"  Val predictions (1 batch): {val_preds[:10].tolist()}")
        print(f"  Val unique predictions: {unique_preds.tolist()} ({len(unique_preds)} classes)")
        val_correct = val_preds.eq(val_batch_labels.to(device)).sum().item()
        print(f"  Val batch top-1: {val_correct}/{len(val_batch_labels)} "
              f"= {100*val_correct/len(val_batch_labels):.1f}%")
        print(f"  (Random chance: {100.0/num_classes:.2f}%)")

    model.train()
    print(f"  --- End diagnostics ---\n")

    # ── 7. Train ────────────────────────────────────────────────────────
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.train(train_loader, val_loader)

    if args.avg_checkpoints > 0:
        _run_checkpoint_averaging(model, val_loader, run_dir, args.avg_checkpoints, device)


if __name__ == '__main__':
    main()
