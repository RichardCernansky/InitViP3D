"""Check that training with samples_per_gpu > 1 is the mean of B=1 runs.

ViP3D.forward_train runs the sensor backbones once per frame over the whole
batch and then tracks/predicts each clip on its own, averaging the loss
terms. This script checks that on two training samples x1, x2:

  (a) one forward/backward with B=2 on (x1, x2)
  (b) forward/backward with B=1 on x1 and on x2, losses and grads averaged

give the same loss terms and the same gradient for every parameter, to
--atol. It prints the largest |a - b| per loss term and per parameter group
(top-level module) and exits non-zero on any mismatch.

Everything random is switched off so the two runs are comparable:
  - dropout: p=0 in every nn.Dropout*, nn.MultiheadAttention and DropPath
  - GridMask, and QIM's random track drop / FP-track insertion
  - BatchNorm goes to eval mode. With batch statistics a B=2 backbone pass
    is *meant* to differ from two B=1 passes -- that is what a larger batch
    does, not something the per-clip loop should hide.
  - x1 and x2 are loaded from the dataset once and the same tensors feed
    (a) and (b), so the pipeline's own augmentation cannot differ between
    them. Both runs start from the same seed.

It also records the backbone outputs of both runs and prints how far the
batched B=2 features are from the B=1 ones. Near-ties in the heatmap top-k
and in Hungarian matching can turn float-level differences from batching
(e.g. a different cuDNN algorithm for N=2) into different tracks; if that
happens, --unbatched-backbone runs the backbones one sample at a time inside
(a), which makes its features bit-identical to (b) and isolates the per-clip
loop itself. --noise-floor repeats (b) to show the run-to-run noise of
non-deterministic CUDA kernels (atomics in backward).

Needs a GPU and the dataset. From the repo root:
    python tools/check_batch_equivalence.py CONFIG \\
        [--checkpoint ckpt_init/livip3d_init.pth] [--indices 0 1] [--seed 0]
"""
import argparse
import copy
import importlib
import os.path as osp
import sys
from collections import defaultdict

import torch
import torch.nn as nn

_REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from mmcv import Config, DictAction  # noqa: E402
from mmcv.parallel import collate, scatter  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet.apis import set_random_seed  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare a B=2 training step against two B=1 steps')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', default=None,
                        help='weights to load (e.g. the config\'s load_from); '
                             'random init if omitted')
    parser.add_argument('--indices', type=int, nargs=2, default=[0, 1],
                        metavar=('I1', 'I2'),
                        help='dataset indices of x1 and x2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--unbatched-backbone', action='store_true',
                        help='in (a), run the backbones one sample at a time')
    parser.add_argument('--noise-floor', action='store_true',
                        help='run (b) twice and report its run-to-run noise')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override settings in the config, key=value')
    return parser.parse_args()


def import_plugin(cfg, config_path):
    """Register the plugin's modules, the same way tools/train.py does."""
    if not cfg.get('plugin', False):
        return
    if cfg.get('plugin_dir', None):
        module_dir = osp.dirname(cfg.plugin_dir)
    else:
        module_dir = osp.dirname(config_path)
    importlib.import_module('.'.join(module_dir.split('/')))


def disable_randomness(model):
    model.use_grid_mask = False
    model.query_interact.random_drop = 0.0
    model.query_interact.fp_ratio = 0.0
    for m in model.modules():
        if isinstance(m, nn.modules.dropout._DropoutNd):
            m.p = 0.0
        elif isinstance(m, nn.MultiheadAttention):
            m.dropout = 0.0
        elif hasattr(m, 'drop_prob'):   # mmcv DropPath
            m.drop_prob = 0.0
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def disable_visualiser(model):
    import plugin.vip3d.models.bev_vis as bev_vis
    bev_vis.ENABLED = False
    bev_vis.DEBUG_PRINTS = False
    model.bev_vis = False


def _to_cpu(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(e) for e in x)
    return x.detach().float().cpu().clone()


def _cat(outs):
    first = outs[0]
    if first is None:
        return None
    if isinstance(first, (list, tuple)):
        return type(first)(_cat([o[k] for o in outs]) for k in range(len(first)))
    return torch.cat(outs, dim=0)


def _max_abs_diff(x, y):
    if x is None and y is None:
        return 0.0
    if isinstance(x, (list, tuple)):
        return max((_max_abs_diff(a, b) for a, b in zip(x, y)), default=0.0)
    if x.shape != y.shape:
        return float('inf')
    return (x - y).abs().max().item() if x.numel() else 0.0


class BackboneProbe:
    """Stands in for model.extract_feat: keeps a CPU copy of every call's
    output, and with `unbatched` runs a batch one sample at a time."""

    def __init__(self, model):
        self.extract_feat = model.extract_feat
        self.unbatched = False
        self.calls = []
        model.extract_feat = self

    def __call__(self, points, img=None, radar=None, img_metas=None):
        bs = len(img_metas)
        if self.unbatched and bs > 1:
            out = _cat([self.extract_feat(
                points[b:b + 1] if points is not None else None,
                img=img[b:b + 1] if img is not None else None,
                radar=radar[b:b + 1] if radar is not None else None,
                img_metas=img_metas[b:b + 1]) for b in range(bs)])
        else:
            out = self.extract_feat(points, img=img, radar=radar, img_metas=img_metas)
        self.calls.append(_to_cpu(out))
        return out

    def take(self):
        calls, self.calls = self.calls, []
        return calls


def seed_all(seed):
    # deterministic=True: cudnn.deterministic on, cudnn.benchmark off
    set_random_seed(seed, deterministic=True)


def run_batch(model, samples, gpu):
    """Training forward on `samples` as one batch; returns (total, loss terms)."""
    data = collate(copy.deepcopy(samples), samples_per_gpu=len(samples))
    data = scatter(data, [gpu])[0]
    losses, _ = model(return_loss=True, **data)
    loss, _ = model._parse_losses(losses)
    return loss, losses


def grads_of(model):
    return {name: (p.grad.detach().clone() if p.grad is not None else None)
            for name, p in model.named_parameters() if p.requires_grad}


def run_reference(model, x1, x2, gpu, seed):
    """(b): B=1 on x1 and on x2, averaged. Same seed as (a), not reseeded in
    between, so any RNG draws come in the same order as (a)'s clip loop."""
    seed_all(seed)
    model.zero_grad(set_to_none=True)
    per_sample = []
    for x in (x1, x2):
        loss, losses = run_batch(model, [x], gpu)
        (loss / 2).backward()
        per_sample.append({k: v.detach() for k, v in losses.items()})
    keys = per_sample[0].keys()
    assert per_sample[1].keys() == keys, 'x1 and x2 produced different loss terms'
    losses = {k: (per_sample[0][k] + per_sample[1][k]) / 2 for k in keys}
    return losses, grads_of(model)


def compare_losses(losses_a, losses_b, atol, rtol):
    print('\n== loss terms: (a) B=2 loop vs (b) mean of two B=1 runs ==')
    ok = losses_a.keys() == losses_b.keys()
    if not ok:
        print(f'  key mismatch: only in (a) {sorted(losses_a.keys() - losses_b.keys())}, '
              f'only in (b) {sorted(losses_b.keys() - losses_a.keys())}')
    width = max(len(k) for k in losses_a)
    print(f'  {"term":<{width}}  {"(a)":>14}  {"(b)":>14}  {"max|a-b|":>10}')
    worst = 0.0
    for k in sorted(losses_a.keys() & losses_b.keys()):
        a = losses_a[k].double()
        b = losses_b[k].double()
        diff = (a - b).abs().max().item()
        worst = max(worst, diff)
        close = torch.allclose(a, b, atol=atol, rtol=rtol)
        ok &= close
        print(f'  {k:<{width}}  {a.mean().item():14.7e}  {b.mean().item():14.7e}  '
              f'{diff:10.3e}{"" if close else "  MISMATCH"}')
    print(f'  largest |a-b| over {len(losses_a)} terms: {worst:.3e}')
    return ok


def compare_grads(grads_a, grads_b, atol, rtol, title):
    print(f'\n== {title} ==')
    groups = defaultdict(lambda: dict(n=0, diff=0.0, scale=0.0, worst='', bad=0,
                                      only_a=0, only_b=0))
    no_grad = 0
    for name in grads_a:
        ga, gb = grads_a[name], grads_b[name]
        if ga is None and gb is None:
            no_grad += 1
            continue
        g = groups[name.split('.')[0]]
        g['n'] += 1
        if ga is None:
            g['only_b'] += 1
            ga = torch.zeros_like(gb)
        if gb is None:
            g['only_a'] += 1
            gb = torch.zeros_like(ga)
        diff = (ga - gb).abs().max().item()
        g['scale'] = max(g['scale'], gb.abs().max().item())
        if diff >= g['diff']:
            g['diff'], g['worst'] = diff, name
        if not torch.allclose(ga, gb, atol=atol, rtol=rtol):
            g['bad'] += 1

    ok = True
    width = max([len(k) for k in groups] + [5])
    print(f'  {"group":<{width}}  {"#params":>7}  {"max|g_b|":>10}  {"max|a-b|":>10}  worst parameter')
    for key in sorted(groups):
        g = groups[key]
        notes = []
        if g['bad']:
            notes.append(f'{g["bad"]} MISMATCH')
        if g['only_a'] or g['only_b']:
            notes.append(f'grad only in (a): {g["only_a"]}, only in (b): {g["only_b"]}')
        ok &= not g['bad']
        print(f'  {key:<{width}}  {g["n"]:>7}  {g["scale"]:10.3e}  {g["diff"]:10.3e}  '
              f'{g["worst"]}{"  [" + "; ".join(notes) + "]" if notes else ""}')
    print(f'  largest |a-b| over all groups: '
          f'{max((g["diff"] for g in groups.values()), default=0.0):.3e}'
          f'  ({no_grad} trainable params got no gradient in either run)')
    return ok


def compare_backbone(calls_a, calls_b1, calls_b2, slice_batch):
    print('\n== backbone outputs: (a) batched B=2 vs (b) B=1 ==')
    worst = 0.0
    for i, out_a in enumerate(calls_a):
        diffs = [_max_abs_diff(slice_batch(out_a, b), calls_b[i])
                 for b, calls_b in enumerate((calls_b1, calls_b2))]
        worst = max([worst] + diffs)
        print(f'  frame {i}: max|a-b| x1 {diffs[0]:.3e}, x2 {diffs[1]:.3e}')
    return worst


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_plugin(cfg, args.config)
    torch.cuda.set_device(args.gpu)

    seed_all(args.seed)
    dataset = build_dataset(cfg.data.train)
    x1, x2 = (dataset[i] for i in args.indices)
    print(f'x1 = dataset[{args.indices[0]}], x2 = dataset[{args.indices[1]}]')

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    else:
        print('no --checkpoint: comparing on randomly initialised weights')
    model.cuda(args.gpu)
    model.train()
    disable_randomness(model)
    disable_visualiser(model)
    probe = BackboneProbe(model)

    # (a) one B=2 step
    probe.unbatched = args.unbatched_backbone
    seed_all(args.seed)
    model.zero_grad(set_to_none=True)
    loss_a, losses_a = run_batch(model, [x1, x2], args.gpu)
    loss_a.backward()
    losses_a = {k: v.detach() for k, v in losses_a.items()}
    grads_a = grads_of(model)
    calls_a = probe.take()
    probe.unbatched = False

    # (b) two B=1 steps, averaged
    losses_b, grads_b = run_reference(model, x1, x2, args.gpu, args.seed)
    calls_b = probe.take()
    num_frame = len(calls_a)
    calls_b1, calls_b2 = calls_b[:num_frame], calls_b[num_frame:]

    feat_diff = compare_backbone(calls_a, calls_b1, calls_b2, type(model)._slice_batch)
    ok = compare_losses(losses_a, losses_b, args.atol, args.rtol)
    ok &= compare_grads(grads_a, grads_b, args.atol, args.rtol,
                        'gradients per top-level module: (a) vs (b)')

    if args.noise_floor:
        _, grads_b_again = run_reference(model, x1, x2, args.gpu, args.seed)
        probe.take()
        compare_grads(grads_b, grads_b_again, args.atol, args.rtol,
                      'noise floor: (b) vs (b) repeated (informational)')

    print(f'\nresult: {"PASS" if ok else "FAIL"} (atol={args.atol}, rtol={args.rtol}'
          f'{", unbatched backbone" if args.unbatched_backbone else ""})')
    if not ok and feat_diff > 0 and not args.unbatched_backbone:
        print('the batched backbone features already differ from B=1 by '
              f'{feat_diff:.3e}; rerun with --unbatched-backbone to check the '
              'per-clip loop on bit-identical features')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
