"""Where does a forward pass actually spend its time?

Wraps the coarse stages of ViP3D.forward_test with CUDA-synchronised timers and
reports a per-stage breakdown. Monkeypatches rather than editing the model, so
nothing here changes what the model computes.

    PYTHONPATH=. python tools/profile_stages.py \
        --config work_dirs/.../vip3d_resnet50_6cam.py \
        --checkpoint work_dirs/.../epoch_16.pth

Every timer calls torch.cuda.synchronize() on both sides, so the numbers are
real wall time per stage and they sum to roughly the total. That also means the
total here is slightly higher than an unprofiled run -- the syncs serialise work
the GPU would otherwise overlap. Read the proportions, not the absolutes.
"""
import argparse
import os
import time
from collections import defaultdict

import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

TIMES = defaultdict(float)
CALLS = defaultdict(int)


def timed(name, fn):
    """Wrap fn so each call's GPU-inclusive wall time lands in TIMES[name]."""
    def wrapper(*a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        torch.cuda.synchronize()
        TIMES[name] += time.perf_counter() - t0
        CALLS[name] += 1
        return out
    return wrapper


def import_plugin(cfg, config_path):
    import importlib
    if not cfg.get('plugin', False):
        return
    d = cfg.get('plugin_dir', os.path.dirname(config_path))
    importlib.import_module(d.rstrip('/').replace('/', '.'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--num-warmup', type=int, default=20)
    ap.add_argument('--num-timed', type=int, default=50)
    ap.add_argument('--cfg-options', nargs='+', action=DictAction, default=None)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    import_plugin(cfg, args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.model.debug = False

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1,
                              workers_per_gpu=cfg.data.workers_per_gpu,
                              dist=False, shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    m = model.module
    # Coarse stages, outermost first. Nested ones (the head inside the single
    # frame pass) are reported separately, so they double-count into their
    # parent -- the indentation in the output shows which.
    m.extract_feat = timed('extract_feat (backbones)', m.extract_feat)
    m.pts_bbox_head.forward = timed('  pts_bbox_head (decoder)', m.pts_bbox_head.forward)
    m.track_base.update = timed('  track_base.update (300-query loop)', m.track_base.update)
    m._active_instances2results = timed('_active_instances2results', m._active_instances2results)
    # These two are nn.Module children, so __setattr__ refuses a plain function.
    # Wrapping .forward works because Module.__call__ looks up self.forward.
    m.query_interact.forward = timed('query_interact (QIM)', m.query_interact.forward)
    if getattr(m, 'memory_bank', None) is not None:
        m.memory_bank.forward = timed('memory_bank', m.memory_bank.forward)
    if getattr(m, 'use_lidar', False):
        m._compute_heatmap = timed('  _compute_heatmap', m._compute_heatmap)
        m._fill_empty_slots = timed('  _fill_empty_slots (topk)', m._fill_empty_slots)

    total = 0.0
    n = 0
    for i, data in enumerate(loader):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i == args.num_warmup - 1:          # discard warmup accounting
            TIMES.clear(); CALLS.clear()
        if i >= args.num_warmup:
            total += dt
            n += 1
        if i + 1 >= args.num_warmup + args.num_timed:
            break

    print(f'\n=== per-frame breakdown over {n} frames ===')
    print(f'{"stage":<40}{"ms/frame":>10}{"% total":>9}{"calls/frame":>13}')
    acc = 0.0
    for k in sorted(TIMES, key=lambda x: -TIMES[x]):
        ms = TIMES[k] / n * 1000
        if not k.startswith('  '):
            acc += ms
        print(f'{k:<40}{ms:>10.1f}{100 * ms / (total / n * 1000):>8.1f}%'
              f'{CALLS[k] / n:>13.1f}')
    tot_ms = total / n * 1000
    print(f'{"-" * 72}')
    print(f'{"measured total":<40}{tot_ms:>10.1f}{100.0:>8.1f}%')
    print(f'{"sum of top-level stages":<40}{acc:>10.1f}'
          f'{100 * acc / tot_ms:>8.1f}%')
    print(f'{"unattributed":<40}{tot_ms - acc:>10.1f}'
          f'{100 * (tot_ms - acc) / tot_ms:>8.1f}%')
    print('\nIndented rows are nested inside a row above and are already '
          'counted in it.')


if __name__ == '__main__':
    main()
