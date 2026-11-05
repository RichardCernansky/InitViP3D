"""Break detection false positives down by cause.

Predictions are matched to GT exactly as the nuScenes devkit does it (per
class, in descending score order, greedily to the nearest unmatched GT box,
BEV centre distance < --dist), after the devkit's own range / zero-point /
bike-rack filtering. Every unmatched prediction of --classes then goes into the
first bucket that fits:

  duplicate        a same-class GT within --dist exists, but a higher-scoring
                   box already took it
  wrong_class      a GT of another class within --dist
  gt_no_points     a GT within --dist that the devkit dropped for having no
                   LiDAR/radar points in the key frame
  gt_out_of_range  a GT within --dist that the devkit dropped as beyond its
                   class range (box near the range edge)
  gt_bike_rack     a GT within --dist dropped as being in a bike rack
  near_miss        a same-class GT within 2 x --dist: a real object, badly placed
  trail            behind a moving GT (> 1 m/s): along its reversed velocity,
                   up to --trail-len back and 1.5 m to the side -- where the
                   points of older sweeps of that object lie
  ghost            none of the above

Reads det/results_det.json, which tools/detection_eval.py writes, and saves
det/fp_breakdown.json next to it. GT is loaded once for all runs.

    python tools/fp_breakdown.py work_dirs/.../eval/epoch_4_max work_dirs/.../eval/epoch_12
    python tools/fp_breakdown.py "work_dirs/.../eval/*/"
"""
import argparse
import copy
import json
import os.path as osp
from collections import defaultdict
from glob import glob

import numpy as np

BUCKETS = ['duplicate', 'wrong_class', 'gt_no_points', 'gt_out_of_range',
           'gt_bike_rack', 'near_miss', 'trail', 'ghost']
GHOST_RANGES = [(0, 20), (20, 30), (30, 40), (40, np.inf)]


def frame_arrays(boxes):
    """EvalBoxes -> {sample_token: dict of numpy arrays}, for vectorised lookups."""
    out = {}
    for tok in boxes.sample_tokens:
        bs = boxes[tok]
        out[tok] = dict(
            xy=np.array([b.translation[:2] for b in bs], dtype=np.float64).reshape(-1, 2),
            vel=np.array([b.velocity for b in bs], dtype=np.float64).reshape(-1, 2),
            cls=np.array([b.detection_name for b in bs]),
            boxes=bs)
    return out


def match(gt_boxes, pred_boxes, cls, dist_th):
    """Devkit accumulate() matching for one class.

    -> (list of unmatched predictions, set of (sample_token, gt index) taken).
    gt index refers to gt_boxes[sample_token], the filtered GT list.
    """
    from nuscenes.eval.common.utils import center_distance

    preds = [b for b in pred_boxes.all if b.detection_name == cls]
    confs = [b.detection_score for b in preds]
    order = [i for (_, i) in sorted((v, i) for (i, v) in enumerate(confs))][::-1]
    taken, fps = set(), []
    for idx in order:
        p = preds[idx]
        best, best_gt = np.inf, None
        for g_idx, g in enumerate(gt_boxes[p.sample_token]):
            if g.detection_name == cls and (p.sample_token, g_idx) not in taken:
                dist = center_distance(g, p)
                if dist < best:
                    best, best_gt = dist, g_idx
        if best < dist_th:
            taken.add((p.sample_token, best_gt))
        else:
            fps.append(p)
    return fps, taken


def drop_reason(box, cfg):
    """Why filter_eval_boxes removed a GT box (same order of checks)."""
    if not box.ego_dist < cfg.class_range[box.detection_name]:
        return 'gt_out_of_range'
    if box.num_pts == 0:
        return 'gt_no_points'
    return 'gt_bike_rack'


def classify(fp, gt_f, dropped, cls, dist_th, trail_len):
    """-> bucket name for one false positive."""
    xy = np.array(fp.translation[:2])
    f = gt_f.get(fp.sample_token)
    if f is not None and len(f['xy']):
        d = np.linalg.norm(f['xy'] - xy, axis=1)
        same = f['cls'] == cls
        # any same-class GT this close is taken, or the greedy match would have used it
        if np.any(same & (d < dist_th)):
            return 'duplicate'
        if np.any(~same & (d < dist_th)):
            return 'wrong_class'

    drop = dropped.get(fp.sample_token, [])
    near_drop = [(np.linalg.norm(np.array(b.translation[:2]) - xy), r)
                 for b, r in drop]
    near_drop = [r for dd, r in near_drop if dd < dist_th]
    if near_drop:
        return near_drop[0]

    if f is not None and len(f['xy']):
        if np.any(same & (d < 2 * dist_th)):
            return 'near_miss'
        speed = np.linalg.norm(f['vel'], axis=1)
        moving = np.nan_to_num(speed) > 1.0
        if np.any(moving):
            back = -f['vel'][moving] / speed[moving, None]   # unit vector, behind the object
            rel = xy - f['xy'][moving]
            along = (rel * back).sum(1)
            side = np.abs(rel[:, 0] * back[:, 1] - rel[:, 1] * back[:, 0])
            if np.any((along > 0.5) & (along < trail_len) & (side < 1.5)):
                return 'trail'
    return 'ghost'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('eval_dirs', nargs='+', help='eval dirs holding det/results_det.json')
    p.add_argument('--dataroot', default='data/nuscenes')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--eval-set', default='val')
    p.add_argument('--classes', nargs='*', default=['car', 'pedestrian'])
    p.add_argument('--dist', type=float, default=2.0, help='match distance, m')
    p.add_argument('--trail-len', type=float, default=8.0,
                   help='how far behind a moving object counts as its trail, m '
                        '(10 sweeps = 0.5 s; 8 m covers ~16 m/s)')
    args = p.parse_args()

    dirs = [d.rstrip('/') for pat in args.eval_dirs for d in sorted(glob(pat))]
    todo = [d for d in dirs if osp.isfile(osp.join(d, 'det', 'results_det.json'))]
    for d in dirs:
        if d not in todo:
            print(f'skipping {d}: no det/results_det.json (run tools/detection_eval.py first)')
    if not todo:
        raise SystemExit('nothing to analyse')

    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import (add_center_dist, filter_eval_boxes,
                                              load_gt, load_prediction)
    from nuscenes.eval.detection.data_classes import DetectionBox

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    cfg = config_factory('detection_cvpr_2019')

    # GT does not depend on the run: load and filter it once
    print('loading GT ...', flush=True)
    gt_all = add_center_dist(nusc, load_gt(nusc, args.eval_set, DetectionBox))
    gt_kept = filter_eval_boxes(nusc, copy.deepcopy(gt_all), cfg.class_range)
    gt_f = frame_arrays(gt_kept)
    dropped = {}
    for tok in gt_all.sample_tokens:
        kept = {tuple(b.translation) for b in gt_kept[tok]}
        dropped[tok] = [(b, drop_reason(b, cfg)) for b in gt_all[tok]
                        if tuple(b.translation) not in kept]

    results = {}
    for d in todo:
        preds, _ = load_prediction(osp.join(d, 'det', 'results_det.json'),
                                   cfg.max_boxes_per_sample, DetectionBox)
        preds = filter_eval_boxes(nusc, add_center_dist(nusc, preds), cfg.class_range)

        res = {}
        for cls in args.classes:
            fps, taken = match(gt_kept, preds, cls, args.dist)
            n_gt = sum(1 for b in gt_kept.all if b.detection_name == cls)
            buckets = defaultdict(list)
            for fp in fps:
                buckets[classify(fp, gt_f, dropped, cls, args.dist, args.trail_len)].append(fp)
            ghosts = buckets['ghost']
            res[cls] = dict(
                gt=n_gt, tp=len(taken), fp=len(fps),
                buckets={b: dict(
                    n=len(buckets[b]),
                    share=len(buckets[b]) / len(fps) if fps else 0.0,
                    mean_score=float(np.mean([x.detection_score for x in buckets[b]]))
                    if buckets[b] else None,
                    share_score_ge_0_5=float(np.mean([x.detection_score >= 0.5 for x in buckets[b]]))
                    if buckets[b] else None)
                    for b in BUCKETS},
                ghost_by_ego_dist={f'{lo}-{hi}m': int(sum(lo <= g.ego_dist < hi for g in ghosts))
                                   for lo, hi in GHOST_RANGES})
        results[d] = res
        with open(osp.join(d, 'det', 'fp_breakdown.json'), 'w') as f:
            json.dump(dict(dist_th=args.dist, trail_len=args.trail_len, per_class=res), f, indent=2)
        print(f'done {d}', flush=True)

    for cls in args.classes:
        print(f'\n===== {cls}: false positives @ {args.dist} m, by cause '
              f'(count, % of FP, mean score) =====')
        print(f'  {"run":<44}{"TP":>7}{"FP":>7}' + ''.join(f'{b:>18}' for b in BUCKETS))
        for d, res in results.items():
            r = res[cls]
            cells = ''
            for b in BUCKETS:
                x = r['buckets'][b]
                s = f'{x["mean_score"]:.2f}' if x['mean_score'] is not None else '  - '
                cells += f'{x["n"]:>7d} {100 * x["share"]:4.1f}% {s}'.rjust(18)
            print(f'  {d[-42:]:<44}{r["tp"]:>7d}{r["fp"]:>7d}{cells}')
        print(f'  ghosts by distance from ego: ' + '; '.join(
            f'{d[-30:]}: ' + ' '.join(f'{k} {v}' for k, v in res[cls]['ghost_by_ego_dist'].items())
            for d, res in results.items()))
    print('\n  Buckets are exclusive and checked in column order; see the docstring.')


if __name__ == '__main__':
    main()
