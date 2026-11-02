#!/usr/bin/env python3
"""Render the three-seed nonlinear-assimilation summary for the README."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / 'results/nonlinear_assimilation/replication-001/summary.json'
OUT = ROOT / 'results/nonlinear_assimilation/replication-001'


def main():
    data = json.loads(SUMMARY.read_text())
    arms = [('direct', 'Direct supervised'), ('output', 'Teacher + output'),
            ('guided', 'Teacher + guided'), ('random', 'Teacher + random')]
    colors = ['#3b6ea8', '#e07a35', '#3c9b78', '#8a6db1']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axes[0]
    x = np.arange(5)
    for (arm, label), color in zip(arms, colors):
        vals = data[f'{arm}_mean_test_accuracy']
        ax.plot(x, [100 * vals[k] for k in 'ABCDE'], marker='o', linewidth=2.2,
                markersize=6, label=label, color=color)
    ax.set_xticks(x, ['A\nSelect', 'B\nFilter', 'C\nSort', 'D\nGroup', 'E\nJoin'])
    ax.set_ylim(-3, 105)
    ax.set_ylabel('Strict exact accuracy (%)')
    ax.set_title('Atomic skills')
    ax.grid(axis='y', alpha=.25)
    ax.legend(frameon=False, loc='lower left', fontsize=9)

    ax = axes[1]
    tasks = ['BA', 'BC', 'BD', 'BE', 'BDA', 'BCE']
    width = .19
    offsets = (np.arange(len(arms)) - (len(arms)-1)/2) * width
    for idx, ((arm, _), color) in enumerate(zip(arms, colors)):
        vals = data[f'{arm}_mean_composition_accuracy']
        ax.bar(np.arange(len(tasks)) + offsets[idx], [100 * vals[t] for t in tasks],
               width=width, color=color)
    ax.set_xticks(np.arange(len(tasks)), tasks)
    ax.set_ylim(0, 105)
    ax.set_title('Held-out compositions')
    ax.grid(axis='y', alpha=.25)
    ax.set_axisbelow(True)
    ax.text(.02, .97, '16 atomic / 8 composition examples per seed', transform=ax.transAxes,
            va='top', ha='left', fontsize=8, color='#4b5563')
    fig.suptitle('Matched nonlinear-assimilation replication · 3 training seeds',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.text(.5, -.015, 'Synthetic record-processing tasks. Same training cohort and token budget; seeds vary initialization/order. '
             'Means are descriptive, not confidence intervals.', ha='center', fontsize=9, color='#4b5563')
    fig.tight_layout()
    fig.savefig(OUT / 'overview.png', dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / 'overview.svg', bbox_inches='tight', facecolor='white')


if __name__ == '__main__':
    main()
