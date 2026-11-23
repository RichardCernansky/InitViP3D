"""EPA as a function of the output score threshold, without re-running inference.

The model writes every active track scoring above model.bbox_coder.score_threshold
(DETRTrack3DCoder default 0.2, unset in the configs) to results_nusc.json, and EPA
charges 0.5 for every box that matches no agent. That threshold only filters
finished boxes -- the tracker state does not depend on it -- so dropping the boxes
of an existing results file with tracking_score <= t gives the boxes a GPU run with
score_threshold=t would write.

One approximation: at inference the predictor builds a track's past from the frames
in which the track was output, so frames hidden by a higher threshold would leave
gaps in its history. Here every box keeps the trajectory predicted from the full
history: detection and matching are exact, trajectories a close approximation.
Confirm the value you pick with one normal eval:
    EXTRA="model.dedup_radius=0.5 model.bbox_coder.score_threshold=X"

Scoring reuses tools/prediction_eval.py unchanged, with the settings
env/eval_epochs.sbatch runs it with (all 7 classes, no camera-visibility filter), so
the 0.2 column reproduces each directory's prediction_metrics.json (printed next to
it as a check).

    python tools/epa_threshold_sweep.py 'work_dirs/.../eval/epoch_*_dedup/'
    python tools/epa_threshold_sweep.py DIR1 DIR2 --thresholds 0.2 0.3 0.4 --workers 4

Writes epa_threshold_sweep.json into every eval dir and prints one table, plus the
single threshold with the best mean EPA over all dirs -- pick one value for all runs
rather than the best per run, which would be tuning on the validation set.
"""
import argparse
import contextlib
import io
import json
import os.path as osp
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import prediction_eval as pe  # noqa: E402

pe.tqdm = lambda it, **kwargs: it     # no progress bars from parallel workers

CLASSES = ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']
_INFOS = None                         # prediction GT infos, loaded once per worker


def _init_worker(infos_path):
    global _INFOS
    with open(infos_path) as f:
        _INFOS = list(json.load(f).values())


def load_results(path):
    """results_nusc.json -> {sample_token: (scores, [PredAgent])}."""
    with open(path) as f:
        data = json.load(f)
    data = data.get('results', data)
    return {tok: (np.array([b['tracking_score'] for b in boxes], dtype=np.float64),
                  [pe.PredAgent.deserialize(b) for b in boxes])
            for tok, boxes in data.items()}


def epa_at(results, thr):
    """Score the boxes with tracking_score > thr, as prediction_eval.py would."""
    ev = pe.PredictionEval.__new__(pe.PredictionEval)   # skip its file loading
    ev.sample_token_2_pred_agents = {
        tok: [a for a, s in zip(agents, scores) if s > thr]
        for tok, (scores, agents) in results.items()}
    ev.prediction_infos = _INFOS
    ev.nusc = None
    ev.class_indices = set(range(len(CLASSES)))
    ev._eval_class_names = set(CLASSES)
    with contextlib.redirect_stdout(io.StringIO()):     # evaluate() prints every minFDE
        m = ev.evaluate()
    s = m.serialize()
    return dict(threshold=thr, EPA=float(s['EPA']), minADE=float(s['minADE']),
                minFDE=float(s['minFDE']), MR=float(s['MR']),
                boxes=sum(len(v) for v in ev.sample_token_2_pred_agents.values()),
                gt_agents=int(m.gt_agent_num.get_sum()),
                matched=int(m.matched.get_sum()),
                hits=int(m.matched_and_prediction_hit.get_sum()),
                false_pos=int(m.unmatched.get_sum()))


def sweep_dir(job):
    d, thresholds = job
    results = load_results(osp.join(d, 'results_nusc.json'))
    rows = [epa_at(results, t) for t in thresholds]
    ref = None
    if osp.isfile(osp.join(d, 'prediction_metrics.json')):
        with open(osp.join(d, 'prediction_metrics.json')) as f:
            ref = json.load(f).get('EPA')
    with open(osp.join(d, 'epa_threshold_sweep.json'), 'w') as f:
        json.dump(dict(reference_EPA=ref, rows=rows), f, indent=2)
    print(f'done {d}', flush=True)
    return d, ref, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('eval_dirs', nargs='+', help='eval dirs holding results_nusc.json (globs ok)')
    p.add_argument('--thresholds', nargs='+', type=float,
                   default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    p.add_argument('--prediction-infos', default='nuscenes_prediction_infos_val.json',
                   help='the GT file env/eval_epochs.sbatch passes to prediction_eval.py')
    p.add_argument('--workers', type=int, default=4,
                   help='dirs scored in parallel; each holds one results file (~2 GB)')
    args = p.parse_args()

    dirs = [d.rstrip('/') for pat in args.eval_dirs for d in sorted(glob(pat))]
    todo = [d for d in dirs if osp.isfile(osp.join(d, 'results_nusc.json'))]
    for d in dirs:
        if d not in todo:
            print(f'skipping {d}: no results_nusc.json')
    if not todo:
        raise SystemExit('nothing to sweep')
    thresholds = sorted(args.thresholds)

    with Pool(min(args.workers, len(todo)), initializer=_init_worker,
              initargs=(args.prediction_infos,)) as pool:
        out = pool.map(sweep_dir, [(d, thresholds) for d in todo])

    # one row per dir: EPA at every threshold, best marked with *
    print(f'\nEPA by output score threshold (model.bbox_coder.score_threshold; current 0.2)')
    print(f'{"eval dir":<58}{"ref":>7}' + ''.join(f'{t:>8.2f}' for t in thresholds))
    for d, ref, rows in out:
        best = max(range(len(rows)), key=lambda i: rows[i]['EPA'])
        cells = ''.join(f'{r["EPA"]:>7.3f}' + ('*' if i == best else ' ')
                        for i, r in enumerate(rows))
        ref_s = f'{ref:>7.3f}' if ref is not None else '      -'
        print(f'{d[-56:]:<58}{ref_s}{cells}')

    mean = [float(np.mean([rows[i]['EPA'] for _, _, rows in out])) for i in range(len(thresholds))]
    i_best = int(np.argmax(mean))
    print(f'{"mean over dirs":<58}{"":>7}' + ''.join(f'{m:>8.3f}' for m in mean))
    print(f'\nbest single threshold: {thresholds[i_best]:.2f} (mean EPA {mean[i_best]:.3f})')

    # what the best single threshold trades, per dir
    print(f'\nat {thresholds[0]:.2f} -> {thresholds[i_best]:.2f}: boxes, matched, false positives, MR')
    for d, _, rows in out:
        a, b = rows[0], rows[i_best]
        print(f'  {d[-56:]:<58} boxes {a["boxes"]:>6} -> {b["boxes"]:>6}   matched {a["matched"]:>6} -> '
              f'{b["matched"]:>6}   FP {a["false_pos"]:>6} -> {b["false_pos"]:>6}   MR {a["MR"]:.3f} -> {b["MR"]:.3f}')
    print('\n  ref = EPA in the dir\'s prediction_metrics.json; the 0.20 column should match it.')


if __name__ == '__main__':
    main()
