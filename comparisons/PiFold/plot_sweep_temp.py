"""Plot PiFold test perplexity and top-1 accuracy vs. mdCATH temperature (sweep_temp.sub).

Reads the final test summary row printed after ">>> testing <<<" in each
logs/sweep_temp_mdcath<T>_traj_<cluster>_<proc>.out -- per-protein
mean +/- std of top-1/5/10 accuracy and perplexity (lib/stats_utils.ResidueMetrics.summary_line).
When a temperature has several completed runs, the latest cluster wins. Runs cut off by the
--max_hours budget ("Stopping at epoch ...") are skipped unless --keep_time_limited is given.
EXTRA_RUNS adds runs submitted outside sweep_temp.sub.

    python plot_sweep_temp.py [--out sweep_temp_metrics.png]
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NAME_RE = re.compile(r'(?:sweep_temp|pifold)_mdcath(\d+)_traj_(\d+)_\d+\.out$')
# 450 K: train.sub run (early-stopped normally); note --frames_per_protein 4, not 5.
EXTRA_RUNS = ('pifold_mdcath450_traj_8246151_0.out',)
ROW_RE = re.compile(r'\$([\d.]+) \\pm ([\d.]+)\$')


def load_results(log_dir, keep_time_limited=False):
    results = {}  # temp -> (cluster, metrics)
    paths = glob.glob(os.path.join(log_dir, 'sweep_temp_mdcath*_traj_*.out'))
    paths += [os.path.join(log_dir, name) for name in EXTRA_RUNS]
    for path in paths:
        m = NAME_RE.search(os.path.basename(path))
        if not m:
            continue
        temp, cluster = int(m.group(1)), int(m.group(2))
        text = open(path).read()
        if '>>> testing' not in text and 'testing  <' not in text:
            continue  # job never reached the test pass
        if not keep_time_limited and 'Stopping at epoch' in text:
            print('skipping {} (stopped by --max_hours)'.format(os.path.basename(path)))
            continue
        test_part = text.split('testing  <', 1)[1]
        rows = [l for l in test_part.splitlines() if ROW_RE.search(l)]
        if not rows:
            continue
        vals = [(float(a), float(b)) for a, b in ROW_RE.findall(rows[-1])]
        metrics = dict(zip(('top1', 'top5', 'top10', 'perplexity'), vals))
        if temp not in results or cluster > results[temp][0]:
            results[temp] = (cluster, metrics)
    return {t: m for t, (_, m) in sorted(results.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', default=os.path.join(HERE, 'logs'))
    ap.add_argument('--out', default=os.path.join(HERE, 'sweep_temp_metrics.png'))
    ap.add_argument('--keep_time_limited', action='store_true',
                    help='also plot runs cut off by the --max_hours budget')
    args = ap.parse_args()

    results = load_results(args.log_dir, args.keep_time_limited)
    if not results:
        raise SystemExit('no completed sweep_temp runs found in ' + args.log_dir)
    temps = list(results)
    print('temp  top1 (mean ± std)    perplexity (mean ± std)')
    for t, m in results.items():
        print('{:4d}  {:.3f} ± {:.3f}        {:.3f} ± {:.3f}'.format(t, *m['top1'], *m['perplexity']))

    color, ink, muted, grid = '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': muted, 'axes.labelcolor': ink,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = (('perplexity', 'Test perplexity (lower is better)', 'Perplexity'),
              ('top1', 'Test top-1 accuracy (higher is better)', 'Top-1 accuracy'))
    for ax, (key, title, ylabel) in zip(axes, panels):
        means = [results[t][key][0] for t in temps]
        stds = [results[t][key][1] for t in temps]
        ax.errorbar(temps, means, yerr=stds, color=color, linewidth=2, marker='o',
                    markersize=7, capsize=4, elinewidth=1.5, markeredgecolor='white',
                    markeredgewidth=1.5)
        for t, mu in zip(temps, means):
            # Centered over the point, on a surface-colored box so the error bar
            # doesn't strike through it; never collides with a neighbor's label.
            ax.annotate('{:.3f}'.format(mu), (t, mu), textcoords='offset points',
                        xytext=(0, 9), ha='center', fontsize=9, color=muted,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))
        ax.set_title(title, color=ink, loc='left', fontsize=12)
        ax.set_xlabel('mdCATH temperature (K)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(temps)
        ax.grid(axis='y', color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.suptitle('PiFold temperature sweep (traj) — error bars: ±1 std across test proteins',
                 fontsize=10, color=muted, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
