"""
Part 3: Visualization script

Generates:
    1. Pruning sparsity vs accuracy trade-off
    2. Quantization strategy comparison (size + accuracy)
    3. Architecture optimization results
    4. NAS candidates Pareto frontier
    5. TFLite model comparison
    6. Deployment viability by platform

Reads from: Part3/edge_optimization_results.json
Writes to:  Part3/figures/*.png
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


def load_results():
    with open(os.path.join(SCRIPT_DIR, 'edge_optimization_results.json')) as f:
        return json.load(f)


# =====================================================================
# 1. Pruning sparsity vs accuracy
# =====================================================================
def plot_pruning(results):
    pruning = results['pruning']
    sparsities = sorted([float(k.split('_')[1]) / 100
                         for k in pruning.keys()])
    accuracies = [pruning[f'sparsity_{int(s*100)}']['test_accuracy'] * 100
                  for s in sparsities]
    actual = [pruning[f'sparsity_{int(s*100)}']['actual_sparsity'] * 100
              for s in sparsities]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(actual, accuracies, 'b-o', linewidth=2.5, markersize=10,
            label='Pruned model')
    ax.axhline(y=82.99, color='red', linestyle='--', linewidth=2,
               label='Baseline (82.99%)')

    for x, y in zip(actual, accuracies):
        ax.annotate(f'{y:.2f}%', (x, y), textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=10,
                    fontweight='bold')

    ax.set_xlabel('Actual sparsity (%)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Magnitude Pruning: Sparsity vs Accuracy Trade-off')
    ax.legend(loc='lower left')
    ax.set_ylim([65, 90])

    # Highlight best
    best_idx = int(np.argmax(accuracies))
    ax.scatter([actual[best_idx]], [accuracies[best_idx]],
               s=300, facecolors='none', edgecolors='green',
               linewidth=2.5, zorder=5)
    ax.annotate('Best', (actual[best_idx], accuracies[best_idx]),
                textcoords='offset points', xytext=(15, 0),
                fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p3_pruning.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 2. Quantization comparison
# =====================================================================
def plot_quantization(results):
    quant = results['quantization']
    baseline_acc = results['baseline']['test_accuracy'] * 100

    strategies = ['Baseline\n(fp32)', 'Dynamic\nRange', 'Float16', 'Full\nINT8']
    sizes_kb = [1240, quant['dynamic_range']['size_kb'],
                quant['float16']['size_kb'], quant['int8']['size_kb']]
    accuracies = [baseline_acc, quant['dynamic_range']['accuracy'] * 100,
                  quant['float16']['accuracy'] * 100,
                  quant['int8']['accuracy'] * 100]
    colors = ['gray', 'steelblue', 'orange', 'green']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Size ---
    bars = axes[0].bar(strategies, sizes_kb, color=colors, width=0.6)
    for bar, v in zip(bars, sizes_kb):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 20,
                     f'{v:.1f} KB', ha='center', fontweight='bold')
    axes[0].set_ylabel('Model size (KB)')
    axes[0].set_title('Quantization: Model Size')
    axes[0].set_ylim([0, max(sizes_kb) * 1.15])

    # --- Accuracy ---
    bars = axes[1].bar(strategies, accuracies, color=colors, width=0.6)
    for bar, v in zip(bars, accuracies):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.3,
                     f'{v:.2f}%', ha='center', fontweight='bold')
    axes[1].axhline(y=baseline_acc, color='red', linestyle='--',
                    alpha=0.6, label='Baseline')
    axes[1].set_ylabel('Test accuracy (%)')
    axes[1].set_title('Quantization: Accuracy Preservation')
    axes[1].set_ylim([min(accuracies) - 3, max(accuracies) + 3])
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p3_quantization.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 3. Architecture optimization + NAS
# =====================================================================
def plot_architecture_and_nas(results):
    arch = results['architecture']
    nas = results['nas']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Architecture optimization ---
    labels = ['Baseline', 'Architecture\nOptimized']
    params = [results['baseline']['params'], arch['params']]
    accs = [results['baseline']['test_accuracy'] * 100,
            arch['test_accuracy'] * 100]

    x = np.arange(len(labels))
    width = 0.35
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    bars1 = ax1.bar(x - width/2, params, width, label='Params',
                    color='steelblue')
    bars2 = ax1_twin.bar(x + width/2, accs, width, label='Accuracy',
                         color='orange')

    ax1.set_ylabel('Parameters', color='steelblue')
    ax1_twin.set_ylabel('Accuracy (%)', color='orange')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title(f"Architecture Optimization\n"
                  f"({arch['compression_vs_baseline']:.2f}× compression)")

    for bar, v in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 5000,
                 f'{v:,}', ha='center', fontsize=9, fontweight='bold')
    for bar, v in zip(bars2, accs):
        ax1_twin.text(bar.get_x() + bar.get_width()/2, v + 1,
                      f'{v:.2f}%', ha='center', fontsize=9,
                      fontweight='bold', color='darkorange')

    # --- NAS Pareto ---
    cands = nas['candidates']
    params = [c['params'] for c in cands]
    accs = [c['test_accuracy'] * 100 for c in cands]
    ids = [f"C{c['id']}" for c in cands]

    ax2 = axes[1]
    scatter = ax2.scatter(params, accs, s=200, c=range(len(cands)),
                          cmap='viridis', edgecolors='black', linewidth=1.5)
    for i, (p, a, lbl) in enumerate(zip(params, accs, ids)):
        ax2.annotate(lbl, (p, a), textcoords='offset points',
                     xytext=(10, 5), fontsize=10, fontweight='bold')
    ax2.set_xlabel('Parameters')
    ax2.set_ylabel('Test accuracy (%)')
    ax2.set_title('NAS Candidates: Pareto Frontier')

    # Mark best
    best_id = nas['best']['id']
    best_cand = next(c for c in cands if c['id'] == best_id)
    ax2.scatter([best_cand['params']], [best_cand['test_accuracy'] * 100],
                s=500, facecolors='none', edgecolors='red',
                linewidth=3, zorder=5)
    ax2.annotate('Best by score', (best_cand['params'],
                                    best_cand['test_accuracy'] * 100),
                 textcoords='offset points', xytext=(15, -20),
                 fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p3_architecture_nas.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 4. TFLite conversions
# =====================================================================
def plot_tflite(results):
    tflite = results['tflite_conversions']

    names = list(tflite.keys())
    sizes = [tflite[n]['size_kb'] for n in names]
    latencies = [tflite[n]['latency_ms'] for n in names]
    accs = [tflite[n]['accuracy'] * 100 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Size ---
    axes[0].bar(names, sizes, color='steelblue', width=0.6)
    for i, v in enumerate(sizes):
        axes[0].text(i, v + 10, f'{v:.1f} KB', ha='center',
                     fontweight='bold')
    axes[0].set_ylabel('Size (KB)')
    axes[0].set_title('TFLite Model Sizes')
    axes[0].tick_params(axis='x', rotation=15)

    # --- Latency ---
    axes[1].bar(names, latencies, color='coral', width=0.6)
    for i, v in enumerate(latencies):
        axes[1].text(i, v + 0.01, f'{v:.2f} ms', ha='center',
                     fontweight='bold')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('TFLite Inference Latency')
    axes[1].tick_params(axis='x', rotation=15)

    # --- Accuracy ---
    axes[2].bar(names, accs, color='mediumpurple', width=0.6)
    for i, v in enumerate(accs):
        axes[2].text(i, v + 0.5, f'{v:.2f}%', ha='center',
                     fontweight='bold')
    axes[2].axhline(y=82.99, color='red', linestyle='--',
                    alpha=0.6, label='Keras baseline')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('TFLite Accuracy Preservation')
    axes[2].legend()
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p3_tflite.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 5. Deployment viability
# =====================================================================
def plot_deployment_viability(results):
    viability = results['deployment_viability']

    platforms = list(viability.keys())
    viable = [v['viable'] for v in viability.values()]
    sizes = [v['best_model_size_kb'] for v in viability.values()]

    # Hardware specs (manually defined from the code)
    ram = [4096, 4096, 0.5, 0.25]
    power = [5000, 10000, 500, 50]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Viability ---
    colors = ['green' if v else 'red' for v in viable]
    bars = axes[0].barh(platforms, [1 if v else 0 for v in viable],
                        color=colors, alpha=0.7)
    axes[0].set_xlim([0, 1.5])
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Not viable', 'Viable'])
    axes[0].set_title('Deployment Viability (best model: 335 KB)')
    for i, (v, p) in enumerate(zip(viable, platforms)):
        axes[0].text(1.05, i, '✓' if v else '✗',
                     fontsize=20, ha='center', va='center',
                     color='green' if v else 'red')

    # --- RAM vs model size ---
    model_size_mb = 0.335
    axes[1].barh(platforms, ram, color='lightblue', alpha=0.6,
                 label='Available RAM (MB)')
    axes[1].barh(platforms, [model_size_mb]*len(platforms),
                 color='red', alpha=0.8, label='Model size (MB)')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Memory (MB, log scale)')
    axes[1].set_title('Memory Constraint Analysis')
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p3_deployment_viability.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


if __name__ == '__main__':
    print("=" * 60)
    print("Part 3: Generating figures")
    print("=" * 60)
    results = load_results()
    plot_pruning(results)
    plot_quantization(results)
    plot_architecture_and_nas(results)
    plot_tflite(results)
    plot_deployment_viability(results)
    print(f"\nAll figures saved to: {FIG_DIR}")