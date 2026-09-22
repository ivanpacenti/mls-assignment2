"""
Part 2: Visualization script

Generates:
    1. Batch processing throughput comparison
    2. Mixed precision vs float32 comparison
    3. Knowledge distillation training curve
    4. Cloud optimization summary (radar chart)

Reads from: Part2/cloud_optimization_results.json
Writes to:  Part2/figures/*.png
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
    with open(os.path.join(SCRIPT_DIR, 'cloud_optimization_results.json')) as f:
        return json.load(f)


# =====================================================================
# 1. Batch processing throughput
# =====================================================================
def plot_batch_processing(results):
    bp = results['batch_processing']
    batch_sizes = [64, 128, 256, 512]
    throughput = [bp[str(b)]['throughput_samples_per_s'] for b in batch_sizes]
    accuracy = [bp[str(b)]['test_accuracy'] * 100 for b in batch_sizes]
    times = [bp[str(b)]['training_time_3ep_s'] for b in batch_sizes]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Throughput ---
    axes[0].plot(batch_sizes, throughput, 'b-o', linewidth=2, markersize=8)
    for x, y in zip(batch_sizes, throughput):
        axes[0].annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=9)
    axes[0].set_xlabel('Batch size')
    axes[0].set_ylabel('Throughput (samples/s)')
    axes[0].set_title('Batch Processing Throughput')
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(batch_sizes)
    axes[0].set_xticklabels([str(b) for b in batch_sizes])

    # --- Accuracy ---
    axes[1].plot(batch_sizes, accuracy, 'g-o', linewidth=2, markersize=8)
    for x, y in zip(batch_sizes, accuracy):
        axes[1].annotate(f'{y:.2f}%', (x, y), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=9)
    axes[1].set_xlabel('Batch size')
    axes[1].set_ylabel('Test accuracy (%)')
    axes[1].set_title('Accuracy vs Batch Size')
    axes[1].set_xscale('log', base=2)
    axes[1].set_xticks(batch_sizes)
    axes[1].set_xticklabels([str(b) for b in batch_sizes])

    # --- Training time ---
    axes[2].bar([str(b) for b in batch_sizes], times,
                color='coral', width=0.6)
    axes[2].set_xlabel('Batch size')
    axes[2].set_ylabel('Training time (s)')
    axes[2].set_title('Training Time for 3 Epochs')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p2_batch_processing.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 2. Mixed precision comparison
# =====================================================================
def plot_mixed_precision(results):
    mp = results['mixed_precision']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Speed comparison ---
    labels = ['float32', 'mixed_float16']
    times = [mp['fp32_training_time_3ep_s'], mp['training_time_3ep_s']]
    axes[0].bar(labels, times, color=['steelblue', 'orange'], width=0.5)
    for i, v in enumerate(times):
        axes[0].text(i, v + 0.2, f'{v:.2f}s', ha='center', fontweight='bold')
    axes[0].set_ylabel('Training time (s)')
    axes[0].set_title(f"Training Speed (speedup: {mp['speedup_vs_fp32']:.3f}×)")

    # --- Accuracy comparison ---
    accs = [mp['fp32_test_accuracy'] * 100, mp['test_accuracy'] * 100]
    axes[1].bar(labels, accs, color=['steelblue', 'orange'], width=0.5)
    for i, v in enumerate(accs):
        axes[1].text(i, v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold')
    axes[1].set_ylabel('Test accuracy (%)')
    axes[1].set_title(f"Accuracy (Δ = {mp['accuracy_delta']*100:+.2f}%)")
    axes[1].set_ylim([min(accs) - 2, max(accs) + 3])

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p2_mixed_precision.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 3. Knowledge distillation
# =====================================================================
def plot_knowledge_distillation(results):
    kd = results['knowledge_distillation']
    hist = kd['student_history']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    epochs = range(1, len(hist['loss']) + 1)

    # --- Loss ---
    axes[0].plot(epochs, hist['loss'], 'b-o', label='Train loss', linewidth=2)
    axes[0].plot(epochs, hist['val_loss'], 'r-o', label='Val loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Distillation Training Loss')
    axes[0].legend()

    # --- Accuracy ---
    axes[1].plot(epochs, [a*100 for a in hist['accuracy']],
                 'b-o', label='Student train', linewidth=2)
    axes[1].plot(epochs, [a*100 for a in hist['val_accuracy']],
                 'r-o', label='Student val', linewidth=2)
    axes[1].axhline(y=kd['teacher_test_accuracy']*100,
                    color='purple', linestyle='--', linewidth=2,
                    label=f"Teacher ({kd['teacher_test_accuracy']*100:.2f}%)")
    axes[1].axhline(y=kd['student_test_accuracy']*100,
                    color='green', linestyle=':', linewidth=2,
                    label=f"Student final ({kd['student_test_accuracy']*100:.2f}%)")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Knowledge Distillation Progress')
    axes[1].legend(loc='lower right')

    # Add text about compression
    fig.suptitle(f"KD: Teacher {kd['teacher_params']:,} params → "
                 f"Student {kd['student_params']:,} params "
                 f"({kd['compression_ratio']:.2f}× compression)",
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p2_knowledge_distillation.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 4. Overall cloud summary (radar)
# =====================================================================
def plot_cloud_summary(results):
    strategies = ['Mixed\nPrecision', 'Model\nParallelism',
                  'Batch\n512', 'KD\nStudent']
    # Normalized scores (0-100) for visualization
    scores = {
        'Speedup': [min(results['mixed_precision']['speedup_vs_fp32'] / 1.2, 1.0),
                    results['model_parallelism']['num_replicas'] / 4.0,
                    results['batch_processing']['512']['throughput_samples_per_s'] / 2500,
                    1.0],
        'Accuracy': [results['mixed_precision']['test_accuracy'],
                     results['model_parallelism']['test_accuracy'],
                     results['batch_processing']['512']['test_accuracy'],
                     results['knowledge_distillation']['student_test_accuracy']],
        'Memory':  [1.0, 1.0, 1.0, 1.0],  # not measured
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(strategies))
    width = 0.35

    ax.bar(x - width/2, scores['Speedup'], width, label='Speedup (norm.)',
           color='steelblue')
    ax.bar(x + width/2, scores['Accuracy'], width, label='Accuracy',
           color='orange')

    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel('Normalized score')
    ax.set_title('Cloud Optimization Strategies Comparison')
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p2_cloud_summary.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


if __name__ == '__main__':
    print("=" * 60)
    print("Part 2: Generating figures")
    print("=" * 60)
    results = load_results()
    plot_batch_processing(results)
    plot_mixed_precision(results)
    plot_knowledge_distillation(results)
    plot_cloud_summary(results)
    print(f"\nAll figures saved to: {FIG_DIR}")