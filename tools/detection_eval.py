"""Run the nuScenes DETECTION benchmark on a tracking results file.

tools/test.py only runs the tracking benchmark (AMOTA/MOTA), which scores
frames as a sequence and charges ID switches and fragmentation. The detection
benchmark scores every frame independently -- no track identity, no history --
and gives what tracking eval does not:

  * mAP averaged over four centre-distance thresholds (0.5/1/2/4 m), not just 2 m
  * the TP error breakdown: mATE, mASE, mAOE, mAVE, mAAE
  * NDS
  * precision-recall curves, rendered by the devkit itself

The only difference between the two result formats is field names, so this
reuses any results_nusc.json already on disk -- no re-inference.

    python tools/detection_eval.py work_dirs/.../eval/epoch_4_max
    python tools/detection_eval.py work_dirs/.../eval/*/          # several at once

Everything lands in a det/ subdirectory of each input: the converted
results_det.json, the metrics, and the rendered curves. Prints a one-line
summary per run.
"""
import argparse
import json
import os
import os.path as osp
from glob import glob

# tracking field -> detection field
RENAME = {'tracking_name': 'detection_name', 'tracking_score': 'detection_score'}
# carried by the tracking format, meaningless to the detection benchmark
DROP = ('tracking_id', 'pred_outputs', 'pred_probs')


def to_detection_format(src, dst):
    """Rewrite a tracking submission as a detection submission."""
    with open(src) as f:
        data = json.load(f)
    results = data.get('results', data)

    n = 0
    for boxes in results.values():
        for b in boxes:
            for old, new in RENAME.items():
                if old in b:
                    b[new] = b.pop(old)
            for k in DROP:
                b.pop(k, None)
            n += 1

    meta = data.get('meta', {})
    # the detection benchmark wants all five modality flags present
    for k in ('use_camera', 'use_lidar', 'use_radar', 'use_map', 'use_external'):
        meta.setdefault(k, False)

    with open(dst, 'w') as f:
        json.dump({'meta': meta, 'results': results}, f)
    return len(results), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('eval_dirs', nargs='+',
                   help='directories holding a results_nusc.json')
    p.add_argument('--dataroot', default='data/nuscenes')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--eval-set', default='val')
    p.add_argument('--no-curves', action='store_true',
                   help='skip rendering the PR / TP curves')
    p.add_argument('--classes', nargs='*', default=['car', 'pedestrian'],
                   help='classes to break out in the per-class table')
    args = p.parse_args()

    dirs = [d for pat in args.eval_dirs for d in sorted(glob(pat))]
    todo = [d for d in dirs if osp.isfile(osp.join(d, 'results_nusc.json'))]
    for d in dirs:
        if d not in todo:
            print(f'skipping {d}: no results_nusc.json')
    if not todo:
        raise SystemExit('nothing to evaluate')

    # importing nuScenes is slow, and loading the DB slower, so do it once
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    cfg = config_factory('detection_cvpr_2019')

    summary = []
    for d in todo:
        src = osp.join(d, 'results_nusc.json')
        out = osp.join(d, 'det')
        os.makedirs(out, exist_ok=True)
        dst = osp.join(out, 'results_det.json')
        nframes, nboxes = to_detection_format(src, dst)
        print(f'\n=== {d}\n    {nboxes} boxes over {nframes} frames -> {dst}')

        ev = DetectionEval(nusc, config=cfg, result_path=dst,
                           eval_set=args.eval_set, output_dir=out, verbose=False)
        m = ev.main(render_curves=not args.no_curves)
        summary.append((d, m))
        print(f'    mAP={m["mean_ap"]:.4f}  NDS={m["nd_score"]:.4f}  -> {out}')

    print('\n===== detection metrics =====')
    print(f'  {"run":<46}{"mAP":>8}{"NDS":>8}{"mATE":>8}{"mAOE":>8}')
    for d, m in summary:
        t = m['tp_errors']
        print(f'  {d[-44:]:<46}{m["mean_ap"]:>8.4f}{m["nd_score"]:>8.4f}'
              f'{t["trans_err"]:>8.3f}{t["orient_err"]:>8.3f}')

    # Overall mAP/NDS average over all 10 nuScenes classes, including the ones
    # this model never predicts (they score 0 and drag the mean down). Break out
    # the classes that actually carry the comparison.
    print(f'\n===== per class ({", ".join(args.classes)}) =====')
    print(f'  {"run":<40}{"class":<12}{"AP":>7}{"NDS*":>7}'
          f'{"ATE":>7}{"ASE":>7}{"AOE":>7}{"AVE":>7}{"AAE":>7}')
    for d, m in summary:
        for c in args.classes:
            if c not in m['mean_dist_aps']:
                print(f'  {d[-38:]:<40}{c:<12}  not in the result')
                continue
            ap = m['mean_dist_aps'][c]
            e = m['label_tp_errors'][c]
            errs = [e['trans_err'], e['scale_err'], e['orient_err'],
                    e['vel_err'], e['attr_err']]
            # NDS as nuScenes defines it, applied to one class: not an official
            # per-class metric, but the same formula on that class's numbers.
            nds = (5 * ap + sum(1 - min(1.0, x) for x in errs)) / 10
            print(f'  {d[-38:]:<40}{c:<12}{ap:>7.3f}{nds:>7.3f}'
                  + ''.join(f'{x:>7.3f}' for x in errs))
    print('  NDS* = (5*AP + sum(1 - min(1, err))) / 10 on this class alone;'
          ' NDS is officially defined over all 10 classes.')


if __name__ == '__main__':
    main()
