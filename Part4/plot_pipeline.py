"""
Part 4: Visualization script

Generates:
    1. Multi-scale comparison (accuracy, size, latency)
    2. Pareto frontier (accuracy vs size)
    3. Scaling efficiency chart
    4. Cascaded deployment diagram

Reads from: Part4/multi_scale_optimization_report.json
Writes to:  Part4/figures/*.png
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_report():
    with open(os.path.join(SCRIPT_DIR, 'multi_scale_optimization_report.json')) as f:
        return json.load(f)


# =====================================================================
# 1. Multi-scale comparison
# =====================================================================
def plot_multi_scale_comparison(report):
    results = report['optimization_results']
    targets = list(results.keys())

    accs = [results[t]['accuracy'] * 100 for t in targets]
    sizes_mb = [results[t]['model_size_mb'] for t in targets]
    latencies = [results[t]['estimated_latency_ms'] for t in targets]
    memories = [results[t]['memory_usage_mb'] for t in targets]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['steelblue', 'orange', 'green']

    # --- Accuracy ---
    bars = axes[0, 0].bar(targets, accs, color=colors, width=0.6)
    for bar, v in zip(bars, accs):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, v + 1,
                        f'{v:.2f}%', ha='center', fontweight='bold')
    axes[0, 0].set_ylabel('Test accuracy (%)')
    axes[0, 0].set_title('Accuracy Across Deployment Targets')
    axes[0, 0].set_ylim([0, 100])

    # --- Size (log) ---
    bars = axes[0, 1].bar(targets, sizes_mb, color=colors, width=0.6)
    for bar, v in zip(bars, sizes_mb):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, v * 1.15,
                        f'{v:.3f} MB', ha='center', fontweight='bold')
    axes[0, 1].set_ylabel('Model size (MB)')
    axes[0, 1].set_title('Model Size (log scale)')
    axes[0, 1].set_yscale('log')

    # --- Latency (log) ---
    bars = axes[1, 0].bar(targets, latencies, color=colors, width=0.6)
    for bar, v in zip(bars, latencies):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, v * 1.15,
                        f'{v:.3f} ms', ha='center', fontweight='bold')
    axes[1, 0].set_ylabel('Inference latency (ms)')
    axes[1, 0].set_title('Latency (log scale)')
    axes[1, 0].set_yscale('log')

    # --- Memory (log) ---
    bars = axes[1, 1].bar(targets, memories, color=colors, width=0.6)
    for bar, v in zip(bars, memories):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, v * 1.15,
                        f'{v:.3f} MB', ha='center', fontweight='bold')
    axes[1, 1].set_ylabel('Memory usage (MB)')
    axes[1, 1].set_title('Memory (log scale)')
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p4_multi_scale_comparison.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 2. Pareto frontier
# =====================================================================
def plot_pareto_frontier(report):
    results = report['optimization_results']

    fig, ax = plt.subplots(figsize=(10, 6))

    # All points
    for target, r in results.items():
        ax.scatter(r['model_size_mb'], r['accuracy'] * 100,
                   s=300, label=target, zorder=5,
                   edgecolors='black', linewidth=1.5)

    # Annotate
    for target, r in results.items():
        ax.annotate(target,
                    (r['model_size_mb'], r['accuracy'] * 100),
                    textcoords='offset points', xytext=(10, 8),
                    fontsize=11, fontweight='bold')

    # Pareto front (all 3 points are Pareto-optimal in our case)
    sizes = [r['model_size_mb'] for r in results.values()]
    accs = [r['accuracy'] * 100 for r in results.values()]
    sorted_pairs = sorted(zip(sizes, accs))
    pareto_sizes, pareto_accs = zip(*sorted_pairs)

    ax.plot(pareto_sizes, pareto_accs, 'gray', linestyle='--',
            linewidth=2, alpha=0.6, label='Pareto frontier')

    ax.set_xscale('log')
    ax.set_xlabel('Model size (MB, log scale)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Pareto Frontier: Accuracy vs Model Size')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Annotate compression
    baseline_size = 1.24
    cloud_size = results['cloud_server']['model_size_mb']
    tiny_size = results['microcontroller']['model_size_mb']
    ax.text(0.02, 0.98,
            f"Baseline: {baseline_size:.2f} MB\n"
            f"Cloud: {cloud_size:.3f} MB\n"
            f"Edge: {results['edge_device']['model_size_mb']:.3f} MB\n"
            f"Tiny: {tiny_size:.3f} MB\n"
            f"Total compression: {baseline_size/tiny_size:.1f}×",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p4_pareto_frontier.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 3. Scaling efficiency
# =====================================================================
def plot_scaling_efficiency(report):
    results = report['optimization_results']
    targets = list(results.keys())

    # Normalized metrics
    accs = [results[t]['accuracy'] for t in targets]
    sizes = [results[t]['model_size_mb'] for t in targets]
    latencies = [results[t]['estimated_latency_ms'] for t in targets]

    # Normalize to cloud=1.0
    acc_norm = [a / accs[0] for a in accs]
    size_norm = [s / sizes[0] for s in sizes]
    lat_norm = [l / latencies[0] for l in latencies]

    x = np.arange(len(targets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, acc_norm, width, label='Accuracy (norm.)', color='green')
    ax.bar(x, size_norm, width, label='Size (norm.)', color='blue')
    ax.bar(x + width, lat_norm, width, label='Latency (norm.)', color='orange')

    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel('Normalized to cloud (log scale)')
    ax.set_title('Scaling Efficiency Across Deployment Targets')
    ax.set_yscale('log')
    ax.legend()
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Annotate
    for i, (a, s, l) in enumerate(zip(acc_norm, size_norm, lat_norm)):
        ax.text(i - width, a * 1.1, f'{a:.2f}',
                ha='center', fontsize=9)
        ax.text(i, s * 1.1, f'{s:.2f}',
                ha='center', fontsize=9)
        ax.text(i + width, l * 1.1, f'{l:.2f}',
                ha='center', fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p4_scaling_efficiency.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 4. Deployment recommendations summary
# =====================================================================
def plot_deployment_recommendations(report):
    results = report['optimization_results']
    targets = list(results.keys())

    # Constraint satisfaction
    size_ok = [results[t]['meets_size_constraint'] for t in targets]
    lat_ok = [results[t]['meets_latency_constraint'] for t in targets]
    mem_ok = [results[t]['meets_memory_constraint'] for t in targets]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(targets))
    width = 0.25

    ax.bar(x - width, [1 if v else 0 for v in size_ok], width,
           label='Size constraint', color='steelblue')
    ax.bar(x, [1 if v else 0 for v in lat_ok], width,
           label='Latency constraint', color='coral')
    ax.bar(x + width, [1 if v else 0 for v in mem_ok], width,
           label='Memory constraint', color='mediumpurple')

    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['✗ Violates', '✓ Meets'])
    ax.set_title('Constraint Satisfaction by Deployment Target')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.3])

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p4_recommendations.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


if __name__ == '__main__':
    print("=" * 60)
    print("Part 4: Generating figures")
    print("=" * 60)
    report = load_report()
    plot_multi_scale_comparison(report)
    plot_pareto_frontier(report)
    plot_scaling_efficiency(report)
    plot_deployment_recommendations(report)
    print(f"\nAll figures saved to: {FIG_DIR}")