"""Plot the total training loss from one or more mmdet runs.

    python tools/plot_loss.py work_dirs/.../20261005_194522.log.json [more...]
    python tools/plot_loss.py work_dirs/perun/non-augmented/*/            # dir = newest log in it

Reads the .log.json mmdet writes next to the .log (one JSON object per logged
iteration), so there is no regex over the text log. Point it at the .log and it
uses the .log.json beside it; point it at a work dir and it picks the newest.

By default each run gets its own png, written into that run's work dir as
loss_<key>.png. Pass --out to get a single combined figure instead, which is
what you want when comparing runs.

    --key loss          which loss term to plot (default: the total, "loss")
    --smooth 20         moving average over N logged points (0 = off)
    --out cmp.png       one combined figure at this path, all runs overlaid
    --list-keys         print the available loss terms and exit
"""
import argparse
import json
import os.path as osp
from glob import glob

import matplotlib
matplotlib.use('Agg')          # no display on the cluster
import matplotlib.pyplot as plt


def resolve(path):
    """Accept a .log.json, a .log, or a work dir; return the .log.json."""
    if osp.isdir(path):
        cands = sorted(glob(osp.join(path, '*.log.json')))
        if not cands:
            return None
        return cands[-1]
    if path.endswith('.log'):
        return path + '.json' if osp.exists(path + '.json') else None
    return path if osp.exists(path) else None


def read(log_json, key):
    """-> (global step, value) for every train record that has `key`."""
    xs, ys, iters_per_epoch = [], [], 0
    with open(log_json) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('mode') != 'train' or key not in r:
                continue
            # mmdet logs iter as 1..N within the epoch; stitch epochs together
            iters_per_epoch = max(iters_per_epoch, r['iter'])
            xs.append((r['epoch'], r['iter']))
            ys.append(r[key])
    steps = [(e - 1) * iters_per_epoch + i for e, i in xs]
    return steps, ys, iters_per_epoch


def keys_in(log_json):
    with open(log_json) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('mode') == 'train':
                return sorted(k for k, v in r.items()
                              if isinstance(v, (int, float))
                              and k not in ('epoch', 'iter', 'memory'))
    return []


def smooth(ys, n):
    if n < 2:
        return ys
    out, acc = [], []
    for y in ys:
        acc.append(y)
        if len(acc) > n:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('logs', nargs='+', help='.log.json, .log, or work dir')
    p.add_argument('--key', default='loss')
    p.add_argument('--smooth', type=int, default=20)
    p.add_argument('--out', default=None)
    p.add_argument('--list-keys', action='store_true')
    args = p.parse_args()

    resolved = [(p_, resolve(p_)) for p_ in args.logs]
    for given, lj in resolved:
        if lj is None:
            print(f'skipping {given}: no .log.json found')

    pairs = [(g, lj) for g, lj in resolved if lj]
    if not pairs:
        raise SystemExit('nothing to plot')

    if args.list_keys:
        for _, lj in pairs:
            print(f'{lj}:')
            for k in keys_in(lj):
                print(f'    {k}')
        return

    def draw(ax, lj):
        """Plot one run onto `ax`; returns its label, or None if it had no data."""
        steps, ys, ipe = read(lj, args.key)
        if not ys:
            print(f'skipping {lj}: no train records with "{args.key}"')
            return None
        # name the curve after the run directory, not the timestamped file
        label = osp.basename(osp.dirname(osp.abspath(lj))) or osp.basename(lj)
        ax.plot(steps, smooth(ys, args.smooth), linewidth=1.4, label=label)
        print(f'{label:<34} {len(ys):>5} points  '
              f'{ipe} iters/epoch  first={ys[0]:.3f} last={ys[-1]:.3f} min={min(ys):.3f}')
        return label

    def finish(fig, ax, title, legend, out):
        ax.set_xlabel('iteration')
        ax.set_ylabel(args.key + (f'  (moving avg {args.smooth})' if args.smooth > 1 else ''))
        ax.set_title(title)
        ax.grid(alpha=.3)
        if legend:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f'  -> {out}')

    if args.out:
        # explicit path: one combined figure, so runs can be compared
        fig, ax = plt.subplots(figsize=(10, 5))
        drawn = [d for _, lj in pairs if (d := draw(ax, lj))]
        if not drawn:
            raise SystemExit('nothing plotted')
        finish(fig, ax, args.key, len(drawn) > 1, args.out)
    else:
        # default: one png per run, written into that run's work dir
        for _, lj in pairs:
            fig, ax = plt.subplots(figsize=(10, 5))
            label = draw(ax, lj)
            if label is None:
                plt.close(fig)
                continue
            out = osp.join(osp.dirname(osp.abspath(lj)), f'loss_{args.key}.png')
            finish(fig, ax, f'{label} — {args.key}', False, out)


if __name__ == '__main__':
    main()
