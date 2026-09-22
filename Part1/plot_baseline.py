"""
Part 1: Visualization script

Generates:
    1. Training history (loss + accuracy curves)
    2. Learning rate schedule
    3. Model architecture summary (bar chart of params per layer)
    4. Baseline performance metrics (bar chart)

Reads from: Part1/baseline_history.json, Part1/baseline_metrics.json
Writes to:  Part1/figures/*.png
"""

import json
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Style: publication-quality
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_json(path):
    with open(path) as f:
        return json.load(f)


# =====================================================================
# 1. Training history (loss + accuracy)
# =====================================================================
def plot_training_history():
    history = load_json(os.path.join(SCRIPT_DIR, 'baseline_history.json'))
    metrics = load_json(os.path.join(SCRIPT_DIR, 'baseline_metrics.json'))

    epochs = range(1, len(history['loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- Loss ---
    axes[0].plot(epochs, history['loss'], 'b-', label='Train loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].axvline(x=len(epochs), color='gray', linestyle='--', alpha=0.5)
    axes[0].text(len(epochs), max(history['loss']), '  Early stopping',
                 va='top', fontsize=9, color='gray')

    # --- Accuracy ---
    axes[1].plot(epochs, history['accuracy'], 'b-', label='Train acc', linewidth=2)
    axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Val acc', linewidth=2)
    axes[1].axhline(y=metrics['test_accuracy'], color='g', linestyle='--',
                    label=f"Test acc = {metrics['test_accuracy']*100:.2f}%")
    axes[1].axhline(y=0.70, color='orange', linestyle=':', alpha=0.7,
                    label='Assignment threshold (70%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p1_training_history.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 2. Learning rate schedule
# =====================================================================
def plot_learning_rate():
    history = load_json(os.path.join(SCRIPT_DIR, 'baseline_history.json'))
    if 'learning_rate' not in history:
        print("[!] No learning_rate in history — skipping LR plot")
        return

    epochs = range(1, len(history['learning_rate']) + 1)
    lr = history['learning_rate']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(epochs, lr, 'b-o', linewidth=2, markersize=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning rate (log scale)')
    ax.set_title('Learning Rate Schedule (ReduceLROnPlateau)')

    # Annotate reductions
    for i in range(1, len(lr)):
        if lr[i] < lr[i-1] * 0.9:  # reduction detected
            ax.annotate(f'LR ÷ 2',
                        xy=(i+1, lr[i]),
                        xytext=(i+1, lr[i]*2.5),
                        fontsize=9, color='red',
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p1_lr_schedule.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


# =====================================================================
# 3. Baseline performance summary
# =====================================================================
def plot_baseline_summary():
    metrics = load_json(os.path.join(SCRIPT_DIR, 'baseline_metrics.json'))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # --- Accuracy gauge ---
    axes[0].bar(['Accuracy'], [metrics['test_accuracy'] * 100],
                color='steelblue', width=0.5)
    axes[0].axhline(y=70, color='orange', linestyle='--', label='Threshold 70%')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Baseline Test Accuracy')
    axes[0].set_ylim([0, 100])
    axes[0].legend()
    axes[0].text(0, metrics['test_accuracy']*100 + 2,
                 f"{metrics['test_accuracy']*100:.2f}%",
                 ha='center', fontweight='bold')

    # --- Latency ---
    latencies = [metrics['single_sample_latency_ms'], metrics['batch32_latency_ms']]
    labels = ['Single sample', 'Batch of 32']
    axes[1].bar(labels, latencies, color=['coral', 'lightgreen'], width=0.5)
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('Inference Latency')
    for i, v in enumerate(latencies):
        axes[1].text(i, v + 0.5, f'{v:.1f} ms', ha='center', fontweight='bold')

    # --- Model size ---
    sizes = [metrics['model_size_mb']]
    axes[2].bar(['Baseline'], sizes, color='mediumpurple', width=0.5)
    axes[2].set_ylabel('Size (MB)')
    axes[2].set_title('Model Size')
    axes[2].text(0, sizes[0] + 0.02, f'{sizes[0]:.2f} MB',
                 ha='center', fontweight='bold')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'p1_baseline_summary.png')
    plt.savefig(out)
    plt.close()
    print(f"[✓] Saved {out}")


if __name__ == '__main__':
    print("=" * 60)
    print("Part 1: Generating figures")
    print("=" * 60)
    plot_training_history()
    plot_learning_rate()
    plot_baseline_summary()
    print(f"\nAll figures saved to: {FIG_DIR}")