"""
Part 5: Aggregate figure generator

Runs all figure-generation scripts from Part1-4, then generates
a summary chart that combines the key metrics across all scales.

Run this AFTER Part1-4 have produced their JSON outputs.
"""

import os
import subprocess
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def run_script(path, name):
    print(f"\n{'=' * 60}")
    print(f"Running {name}...")
    print('=' * 60)
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[!] {name} failed:")
        print(result.stderr)
        return False
    return True


def generate_summary_chart():
    """Combined summary chart across all parts."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    # Load baseline metric
    with open(os.path.join(PROJECT_ROOT, 'Part1', 'baseline_metrics.json')) as f:
        p1 = json.load(f)
    with open(os.path.join(PROJECT_ROOT, 'Part4', 'multi_scale_optimization_report.json')) as f:
        p4 = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar chart comparing all key models
    models = ['Baseline\n(fp32)', 'Pruned\n50%', 'Dynamic\nRange',
              'Full\nINT8', 'Cloud\n(float16)', 'Edge\n(dynamic)', 'Tiny\n(INT8)']
    sizes = [1.24, 1.24, 0.334, 0.341, 0.627, 0.328, 0.018]
    accs = [82.99, 83.75, 83.20, 83.55, 83.35, 83.20, 60.95]
    colors = ['gray', 'green', 'steelblue', 'steelblue',
              'orange', 'orange', 'red']

    x = np.arange(len(models))
    width = 0.5

    ax2 = ax.twinx()
    bars = ax.bar(x, sizes, width, color=colors, alpha=0.7)
    ax2.plot(x, accs, 'ko-', linewidth=2, markersize=10, label='Accuracy')

    for i, (s, a) in enumerate(zip(sizes, accs)):
        ax.text(i, s + 0.05, f'{s:.3f}\nMB', ha='center',
                fontsize=9, fontweight='bold')
        ax2.text(i, a + 1, f'{a:.1f}%', ha='center',
                 fontsize=9, fontweight='bold', color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Model size (MB)')
    ax2.set_ylabel('Test accuracy (%)')
    ax2.set_ylim([50, 95])
    ax.set_title('Summary: All Optimized Models Across Deployment Scales')
    ax2.legend(loc='lower left')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p5_summary.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


if __name__ == '__main__':
    print("#" * 60)
    print("# Part 5: Generating ALL figures")
    print("#" * 60)

    scripts = [
        (os.path.join(PROJECT_ROOT, 'Part1', 'plot_baseline.py'), 'Part 1'),
        (os.path.join(PROJECT_ROOT, 'Part2', 'plot_cloud.py'), 'Part 2'),
        (os.path.join(PROJECT_ROOT, 'Part3', 'plot_edge.py'), 'Part 3'),
        (os.path.join(PROJECT_ROOT, 'Part4', 'plot_pipeline.py'), 'Part 4'),
    ]

    for path, name in scripts:
        if os.path.exists(path):
            run_script(path, name)
        else:
            print(f"[!] {name}: script not found at {path}")

    # Summary chart
    print(f"\n{'=' * 60}")
    print("Generating summary chart...")
    print('=' * 60)
    generate_summary_chart()

    print(f"\n{'#' * 60}")
    print(f"# All figures generated in:")
    for p in range(1, 6):
        print(f"#   Part{p}/figures/")
    print(f"#   Part5/figures/")
    print("#" * 60)