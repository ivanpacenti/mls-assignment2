"""
Part 1: Baseline Model Development

This module implements a baseline CNN for CIFAR-10 classification that will serve
as the starting point for optimization across different deployment targets (cloud,
edge, and microcontroller).

"""
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS', '0')
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import json


def create_baseline_model():
    """
    Create a moderately complex CNN for CIFAR-10 classification.
    This model is intentionally over-parameterized to demonstrate optimization potential.

    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    model = keras.Sequential([
        # ============================================================
        # Block 1: Conv2D(32) -> BN -> ReLU -> Conv2D(32) -> BN -> ReLU -> MaxPool
        # ============================================================
        # Input layer: 32x32x3 (CIFAR-10 images)
        keras.layers.Conv2D(
            32, (3, 3),
            padding='same',
            input_shape=(32, 32, 3),
            name='block1_conv1'
        ),
        keras.layers.BatchNormalization(name='block1_bn1'),
        keras.layers.ReLU(name='block1_relu1'),
        keras.layers.Conv2D(32, (3, 3), padding='same', name='block1_conv2'),
        keras.layers.BatchNormalization(name='block1_bn2'),
        keras.layers.ReLU(name='block1_relu2'),
        keras.layers.MaxPooling2D((2, 2), name='block1_pool'),  # 16x16x32

        # ============================================================
        # Block 2: Conv2D(64) -> BN -> ReLU -> Conv2D(64) -> BN -> ReLU -> MaxPool
        # ============================================================
        keras.layers.Conv2D(64, (3, 3), padding='same', name='block2_conv1'),
        keras.layers.BatchNormalization(name='block2_bn1'),
        keras.layers.ReLU(name='block2_relu1'),
        keras.layers.Conv2D(64, (3, 3), padding='same', name='block2_conv2'),
        keras.layers.BatchNormalization(name='block2_bn2'),
        keras.layers.ReLU(name='block2_relu2'),
        keras.layers.MaxPooling2D((2, 2), name='block2_pool'),  # 8x8x64

        # ============================================================
        # Block 3: Conv2D(128) -> BN -> ReLU -> Conv2D(128) -> BN -> ReLU -> MaxPool
        # ============================================================
        keras.layers.Conv2D(128, (3, 3), padding='same', name='block3_conv1'),
        keras.layers.BatchNormalization(name='block3_bn1'),
        keras.layers.ReLU(name='block3_relu1'),
        keras.layers.Conv2D(128, (3, 3), padding='same', name='block3_conv2'),
        keras.layers.BatchNormalization(name='block3_bn2'),
        keras.layers.ReLU(name='block3_relu2'),
        keras.layers.MaxPooling2D((2, 2), name='block3_pool'),  # 4x4x128

        # ============================================================
        # Classifier: GAP -> Dropout(0.5) -> Dense(256) -> Dropout(0.3) -> Dense(10)
        # ============================================================
        keras.layers.GlobalAveragePooling2D(name='gap'),
        keras.layers.Dropout(0.5, name='dropout1'),
        keras.layers.Dense(256, activation='relu', name='fc1'),
        keras.layers.Dropout(0.3, name='dropout2'),
        keras.layers.Dense(10, activation='softmax', name='predictions')
    ], name='baseline_cnn')

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def load_and_preprocess_data():
    """
    Load and preprocess CIFAR-10 dataset.

    Steps:
        1. Load CIFAR-10 from Keras datasets
        2. Normalize pixel values to [0, 1] range
        3. Apply data augmentation for training set
        4. Flatten labels for sparse_categorical_crossentropy

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    print("[Data] Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    print(f"[Data] Original train shape: {x_train.shape}")
    print(f"[Data] Original test shape:  {x_test.shape}")

    # ----------------------------------------------------------------
    # Normalize pixel values to [0, 1] range
    # ----------------------------------------------------------------
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # ----------------------------------------------------------------
    # Flatten labels: (N, 1) -> (N,) for sparse_categorical_crossentropy
    # ----------------------------------------------------------------
    y_train = y_train.flatten().astype('int32')
    y_test = y_test.flatten().astype('int32')

    # ----------------------------------------------------------------
    # Apply data augmentation for the training set
    # Using tf.keras preprocessing layers (on-the-fly augmentation)
    # ----------------------------------------------------------------
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomTranslation(0.1, 0.1),
    ], name='data_augmentation')

    # Apply augmentation to generate additional training samples
    # (I generate one augmented copy to double the training set)
    print("[Data] Applying data augmentation...")
    x_train_aug = data_augmentation(x_train, training=True).numpy()

    # Concatenate original + augmented samples
    x_train = np.concatenate([x_train, x_train_aug], axis=0)
    y_train = np.concatenate([y_train, y_train], axis=0)

    # Shuffle the combined training set
    shuffle_idx = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    print(f"[Data] Augmented train shape: {x_train.shape}")
    print(f"[Data] Test shape:            {x_test.shape}")
    print(f"[Data] Pixel value range:     [{x_train.min():.3f}, {x_train.max():.3f}]")

    return x_train, y_train, x_test, y_test


def train_baseline_model(model, x_train, y_train, x_test, y_test):
    """
    Train the baseline model with early stopping and learning rate scheduling.

    Callbacks:
        - EarlyStopping (patience=10) on val_loss
        - ReduceLROnPlateau (factor=0.5, patience=5) on val_loss
        - ModelCheckpoint saving best model on val_accuracy
    Maximum training: 50 epochs.

    Args:
        model: Compiled Keras model
        x_train, y_train: Training data
        x_test, y_test: Test data (used for validation)

    Returns:
        tuple: (model, training_history, training_metrics)
    """
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # ----------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------
    callbacks = [
        # Stop training when val_loss stops improving
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when val_loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Save the best model based on validation accuracy
        keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/baseline_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

    ]

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    print("\n[Training] Starting baseline training (max 50 epochs)...")
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=50,
        validation_split=0.1,   # 10% of training data for validation
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"\n[Training] Completed in {training_time:.2f} seconds "
          f"({len(history.history['loss'])} epochs)")

    # ----------------------------------------------------------------
    # Final evaluation on test set
    # ----------------------------------------------------------------
    print("[Evaluation] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # ----------------------------------------------------------------
    # Compute model metrics
    # ----------------------------------------------------------------
    total_params = model.count_params()
    trainable_params = int(sum(
        np.prod(w.shape) for w in model.trainable_weights
    ))
    non_trainable_params = int(sum(
        np.prod(w.shape) for w in model.non_trainable_weights
    ))

    # Estimate model size (float32 = 4 bytes per parameter)
    model_size_bytes = total_params * 4
    model_size_mb = model_size_bytes / (1024 * 1024)

    # ----------------------------------------------------------------
    # Inference time benchmark
    # ----------------------------------------------------------------
    print("[Benchmark] Measuring inference time...")

    # Warm-up
    for _ in range(5):
        _ = model.predict(x_test[:1], verbose=0)

    # Single-sample latency
    n_runs = 50
    t0 = time.time()
    for i in range(n_runs):
        _ = model.predict(x_test[i:i + 1], verbose=0)
    single_latency_ms = (time.time() - t0) / n_runs * 1000

    # Batch-32 latency
    t0 = time.time()
    for _ in range(10):
        _ = model.predict(x_test[:32], verbose=0)
    batch32_latency_ms = (time.time() - t0) / 10 * 1000

    # Throughput
    throughput = 32 / (batch32_latency_ms / 1000)

    # ----------------------------------------------------------------
    # Assemble metrics dictionary
    # ----------------------------------------------------------------
    metrics = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'training_time_seconds': float(training_time),
        'num_epochs_trained': len(history.history['loss']),
        'total_params': int(total_params),
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_mb': float(model_size_mb),
        'single_sample_latency_ms': float(single_latency_ms),
        'batch32_latency_ms': float(batch32_latency_ms),
        'throughput_samples_per_sec': float(throughput),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_val_loss': float(min(history.history['val_loss'])),
    }

    return model, history, metrics


def save_training_history(history, path='baseline_history.json'):
    """Save training history to JSON for later analysis/plotting."""
    history_dict = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    with open(path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"[Save] Training history saved to '{path}'")


if __name__ == "__main__":
    print("=" * 65)
    print("PART 1: BASELINE MODEL DEVELOPMENT")
    print("=" * 65)

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Step 1: Load and preprocess data
    # ------------------------------------------------------------------
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # ------------------------------------------------------------------
    # Step 2: Create and inspect baseline model
    # ------------------------------------------------------------------
    print("\n[Model] Creating baseline model...")
    model = create_baseline_model()
    model.summary()

    # ------------------------------------------------------------------
    # Step 3: Train baseline model
    # ------------------------------------------------------------------
    model, history, metrics = train_baseline_model(
        model, x_train, y_train, x_test, y_test
    )

    # ------------------------------------------------------------------
    # Step 4: Save baseline model and artifacts
    # ------------------------------------------------------------------
    model.save('baseline_model.keras')
    print("\n[Save] Baseline model saved to 'baseline_model.keras'")

    save_training_history(history, 'baseline_history.json')

    # Save metrics
    with open('baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("[Save] Baseline metrics saved to 'baseline_metrics.json'")

    # ------------------------------------------------------------------
    # Step 5: Print final results
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("BASELINE MODEL RESULTS")
    print("=" * 65)
    print(f"Total parameters:        {metrics['total_params']:,}")
    print(f"Trainable parameters:    {metrics['trainable_params']:,}")
    print(f"Non-trainable params:    {metrics['non_trainable_params']:,}")
    print(f"Model size:              {metrics['model_size_mb']:.2f} MB")
    print(f"Epochs trained:          {metrics['num_epochs_trained']}")
    print(f"Test accuracy:           {metrics['test_accuracy']:.4f}")
    print(f"Test loss:               {metrics['test_loss']:.4f}")
    print(f"Best val accuracy:       {metrics['best_val_accuracy']:.4f}")
    print(f"Training time:           {metrics['training_time_seconds']:.2f} s")
    print(f"Single-sample latency:   {metrics['single_sample_latency_ms']:.2f} ms")
    print(f"Batch-32 latency:        {metrics['batch32_latency_ms']:.2f} ms")
    print(f"Throughput:              {metrics['throughput_samples_per_sec']:.1f} samples/s")
    print("=" * 65)

    # Sanity check: baseline must exceed 70% accuracy
    if metrics['test_accuracy'] > 0.70:
        print(f"\n✓ Requirement met: test accuracy > 70%")
    else:
        print(f"\n✗ Warning: test accuracy ({metrics['test_accuracy']:.4f}) "
              f"is below 70% threshold. Consider training longer.")