"""
Part 3: Edge Deployment Optimization

Optimize the baseline model for edge deployment with severe resource constraints.

Strategies implemented:
    1. Magnitude-based pruning (75% target sparsity)
    2. Post-training quantization (dynamic range, float16, full INT8)
    3. Architecture optimization (depthwise separable convolutions)
    4. Simplified Neural Architecture Search (NAS)
    5. TensorFlow Lite conversion with metadata

Follows the exact class/method structure specified in the assignment.
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import time
import json

import os
os.environ['KERAS_HOME'] = os.path.expanduser('~/.keras')

# Note: vitis_quantize is Xilinx-specific (for FPGA DPU deployment).
# Import is kept for structural fidelity with the assignment template,
# but not used in the core benchmarks (no Xilinx hardware available).
try:
    from tensorflow_model_optimization.python.core.quantization.keras import vitis_quantize
    VITIS_AVAILABLE = True
except ImportError:
    VITIS_AVAILABLE = False
    print("[Info] vitis_quantize not available — Xilinx-specific optimizations disabled")


# =====================================================================
# Path handling
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

_CANDIDATE_PATHS = [
    os.path.join(PROJECT_ROOT, 'Part1', 'baseline_model.keras'),
    os.path.join(PROJECT_ROOT, 'baseline_model.keras'),
    'baseline_model.keras',
]
BASELINE_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)),
                     _CANDIDATE_PATHS[0])


def load_cifar10_cached():
    """
    Load CIFAR-10 from local pickle files, bypassing Keras download logic.
    Works regardless of Keras version (2 or 3) and cache layout.
    """
    import pickle
    import numpy as np

    # Search all likely cache paths
    candidate_dirs = [
        os.path.expanduser('~/.keras/datasets/cifar-10-batches-py'),
        os.path.expanduser('~/.keras/datasets/cifar-10-batches-py-target/cifar-10-batches-py'),
    ]

    cache_dir = next((d for d in candidate_dirs if os.path.exists(d)), None)

    if cache_dir is None:
        raise FileNotFoundError(
            "CIFAR-10 not found in any cache directory. "
            "Please download it manually and place it in ~/.keras/datasets/"
        )

    print(f"[Data] Loading CIFAR-10 from: {cache_dir}")

    def unpickle(file):
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='bytes')

    # Load all 5 training batches
    x_train_list, y_train_list = [], []
    for i in range(1, 6):
        batch = unpickle(os.path.join(cache_dir, f'data_batch_{i}'))
        x_train_list.append(batch[b'data'])
        y_train_list.extend(batch[b'labels'])

    x_train = np.concatenate(x_train_list).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train_list, dtype='int32').reshape(-1, 1)

    # Load test batch
    test_batch = unpickle(os.path.join(cache_dir, 'test_batch'))
    x_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_batch[b'labels'], dtype='int32').reshape(-1, 1)

    print(f"[Data] Loaded — train: {x_train.shape}, test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def load_keras_model_compat(path):
    """
    Load a Keras model saved with either Keras 2 or Keras 3.
    Handles the `batch_shape` -> `shape` rename in InputLayer config,
    which causes failures when loading Keras 2 models in Keras 3.
    """
    import zipfile
    import json
    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, 'patched.keras')

    with zipfile.ZipFile(path, 'r') as zin:
        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.namelist():
                data = zin.read(item)
                if item == 'config.json':
                    config = json.loads(data)
                    config = _fix_batch_shape(config)
                    data = json.dumps(config).encode('utf-8')
                zout.writestr(item, data)

    try:
        model = tf.keras.models.load_model(tmp_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return model


def _fix_batch_shape(obj):
    """Recursively replace 'batch_shape' with 'shape' in Keras config dicts."""
    if isinstance(obj, dict):
        if 'batch_shape' in obj and 'shape' not in obj:
            obj['shape'] = obj.pop('batch_shape')
        return {k: _fix_batch_shape(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_fix_batch_shape(item) for item in obj]
    return obj
class EdgeOptimizer:
    """
    Edge-side optimizer providing five orthogonal optimization strategies.

    The class mirrors the assignment template: same methods, same signatures.
    """

    def __init__(self, baseline_model_path):
        print(f"[EdgeOptimizer] Loading baseline from: {baseline_model_path}")
        self.baseline_model_path = baseline_model_path

        # Try the compatibility loader first (handles Keras 2 vs 3 mismatch)
        try:
            self.baseline_model = load_keras_model_compat(baseline_model_path)
            print("[EdgeOptimizer] Loaded via compatibility loader ✓")
        except Exception as e:
            print(f"[EdgeOptimizer] Compat loader failed: {e}")
            print("[EdgeOptimizer] Falling back to rebuild approach...")
            self.baseline_model = self._rebuild_baseline()
            try:
                self.baseline_model.load_weights(baseline_model_path)
                print("[EdgeOptimizer] Weights loaded via load_weights()")
            except Exception as e2:
                print(f"[EdgeOptimizer] Weight load also failed: {e2}")
                raise

        print(f"[EdgeOptimizer] Baseline loaded — "
              f"params: {self.baseline_model.count_params():,}")

    def _rebuild_baseline(self):
        """Rebuild the baseline architecture exactly as in Part 1."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),

            # Block 1
            tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 2
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 3
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Classifier
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax'),
        ], name='baseline_cnn')

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # =================================================================
    # 1. MAGNITUDE-BASED PRUNING
    # =================================================================
    def implement_pruning(self, target_sparsity=0.75, fine_tune_epochs=5):
        """
        Implement magnitude-based pruning for edge deployment.

        Strategy:
            - Wrap the model with `prune_low_magnitude`
            - Use PolynomialDecay schedule from 0% to target_sparsity
            - Fine-tune to recover accuracy
            - Strip pruning wrappers → smaller Keras model

        Args:
            target_sparsity: Target sparsity level (0.75 = 75% weights pruned)
            fine_tune_epochs: Number of fine-tuning epochs

        Returns:
            tf.keras.Model: Pruned model
        """
        print("\n" + "=" * 60)
        print(f"[Pruning] Magnitude-based pruning @ {target_sparsity*100:.0f}% sparsity")
        print("=" * 60)

        # Define pruning schedule: gradual sparsity increase over training
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=1000
        )

        # Wrap model with pruning — IMPORTANT: clone to avoid mutating self.baseline_model
        import copy
        baseline_copy = tf.keras.models.clone_model(self.baseline_model)
        baseline_copy.set_weights(self.baseline_model.get_weights())

        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            baseline_copy,  # ← usa la copia!
            pruning_schedule=pruning_schedule
        )

        model_for_pruning.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Load data
        (x_train, y_train), (x_test, y_test) = load_cifar10_cached()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten().astype('int32')
        y_test = y_test.flatten().astype('int32')

        # Callbacks: pruning step + checkpoint logging
        log_dir = os.path.join(SCRIPT_DIR, 'pruning_logs')
        os.makedirs(log_dir, exist_ok=True)

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
        ]

        # Fine-tune
        print(f"[Pruning] Fine-tuning for {fine_tune_epochs} epochs...")
        model_for_pruning.fit(
            x_train, y_train,
            batch_size=128,
            epochs=fine_tune_epochs,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )

        # Strip pruning wrappers so model is deployment-ready
        model_pruned = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        model_pruned.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Evaluate
        loss, accuracy = model_pruned.evaluate(x_test, y_test, verbose=0)

        # Compute actual sparsity
        total_weights = 0
        zero_weights = 0
        for w in model_pruned.weights:
            total_weights += int(tf.size(w).numpy())
            zero_weights += int(tf.math.count_nonzero(tf.equal(w, 0)).numpy())

        actual_sparsity = zero_weights / total_weights if total_weights > 0 else 0.0

        print(f"[Pruning] Test accuracy: {accuracy:.4f}")
        print(f"[Pruning] Actual sparsity: {actual_sparsity*100:.2f}%")
        print(f"[Pruning] Zero weights: {zero_weights:,} / {total_weights:,}")

        return model_pruned

    # =================================================================
    # 2. POST-TRAINING QUANTIZATION
    # =================================================================
    def implement_quantization(self):
        """
        Implement post-training quantization for edge deployment.

        Strategies:
            - Dynamic range quantization (weights → int8, activations → fp32 runtime)
            - Float16 quantization (weights → float16, activations → float16)
            - Full integer quantization (weights + activations → int8, requires calibration)

        Returns:
            dict: Quantized models with different strategies
        """
        print("\n" + "=" * 60)
        print("[Quantization] Post-training quantization (3 strategies)")
        print("=" * 60)

        quantized_models = {}

        # Load data for calibration and evaluation
        (x_train, y_train), (x_test, y_test) = load_cifar10_cached()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten().astype('int32')
        y_test = y_test.flatten().astype('int32')

        # Representative dataset for INT8 calibration
        def representative_dataset():
            for i in range(100):
                yield [x_train[i:i+1].astype('float32')]

        out_dir = os.path.join(SCRIPT_DIR, 'edge_optimized_models')
        os.makedirs(out_dir, exist_ok=True)

        # -------------------------------------------------------------
        # Strategy 1: Dynamic Range Quantization
        # -------------------------------------------------------------
        print("\n[Quantization] Strategy 1/3: Dynamic Range")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.baseline_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_dynamic = converter.convert()

            path = os.path.join(out_dir, 'model_dynamic_range.tflite')
            with open(path, 'wb') as f:
                f.write(tflite_dynamic)

            acc = self._evaluate_tflite_model(path, x_test[:2000], y_test[:2000], 'int8')
            size_kb = len(tflite_dynamic) / 1024

            quantized_models['dynamic_range'] = {
                'path': path,
                'size_kb': round(size_kb, 1),
                'accuracy': round(acc, 4),
                'bits': 8,
                'note': 'weights int8, activations float32 runtime'
            }
            print(f"[Quantization] Dynamic range: {size_kb:.1f} KB, acc={acc:.4f}")
        except Exception as e:
            print(f"[Quantization] Dynamic range failed: {e}")
            quantized_models['dynamic_range'] = {'error': str(e)}

        # -------------------------------------------------------------
        # Strategy 2: Float16 Quantization
        # -------------------------------------------------------------
        print("\n[Quantization] Strategy 2/3: Float16")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.baseline_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_float16 = converter.convert()

            path = os.path.join(out_dir, 'model_float16.tflite')
            with open(path, 'wb') as f:
                f.write(tflite_float16)

            acc = self._evaluate_tflite_model(path, x_test[:2000], y_test[:2000], 'float32')
            size_kb = len(tflite_float16) / 1024

            quantized_models['float16'] = {
                'path': path,
                'size_kb': round(size_kb, 1),
                'accuracy': round(acc, 4),
                'bits': 16,
                'note': 'weights float16, activations float16'
            }
            print(f"[Quantization] Float16: {size_kb:.1f} KB, acc={acc:.4f}")
        except Exception as e:
            print(f"[Quantization] Float16 failed: {e}")
            quantized_models['float16'] = {'error': str(e)}

        # -------------------------------------------------------------
        # Strategy 3: Full Integer Quantization (INT8)
        # -------------------------------------------------------------
        print("\n[Quantization] Strategy 3/3: Full INT8")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.baseline_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            tflite_int8 = converter.convert()

            path = os.path.join(out_dir, 'model_int8.tflite')
            with open(path, 'wb') as f:
                f.write(tflite_int8)

            acc = self._evaluate_tflite_model(path, x_test[:2000], y_test[:2000], 'int8')
            size_kb = len(tflite_int8) / 1024

            quantized_models['int8'] = {
                'path': path,
                'size_kb': round(size_kb, 1),
                'accuracy': round(acc, 4),
                'bits': 8,
                'note': 'weights + activations int8, requires calibration'
            }
            print(f"[Quantization] Full INT8: {size_kb:.1f} KB, acc={acc:.4f}")
        except Exception as e:
            print(f"[Quantization] Full INT8 failed: {e}")
            quantized_models['int8'] = {'error': str(e)}

        return quantized_models

    def _evaluate_tflite_model(self, model_path, x_test, y_test, input_type='float32'):
        """
        Evaluate a TFLite model on a subset of the test set.
        """
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Force labels to 1D scalars (FIX)
        y_test = np.asarray(y_test).flatten()

        correct = 0
        total = len(x_test)

        for i in range(total):
            x = x_test[i:i+1].astype(np.float32)

            if input_details[0]['dtype'] == np.int8:
                scale, zero_point = input_details[0]['quantization']
                x = (x / scale + zero_point).astype(np.int8)

            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()

            output = interpreter.get_tensor(output_details[0]['index'])

            # Force int comparison (FIX)
            pred = int(np.argmax(output))
            true = int(y_test[i])

            if pred == true:
                correct += 1

        return correct / total
        # =================================================================
    # 3. ARCHITECTURE OPTIMIZATION
    # =================================================================
    def implement_architecture_optimization(self):
        """
        Optimize model architecture for edge constraints.

        Strategies:
            - Replace Conv2D with Depthwise Separable Conv2D (fewer FLOPs)
            - Reduce model depth (fewer blocks)
            - Reduce model width (fewer filters per block)

        Returns:
            tf.keras.Model: Architecture-optimized model
        """
        print("\n" + "=" * 60)
        print("[Architecture] Depthwise separable + reduced width")
        print("=" * 60)

        # Lightweight architecture:
        #   - Depthwise Separable convs (≈8-9× fewer FLOPs than regular convs)
        #   - Fewer filters (24 / 48 / 96 instead of 32 / 64 / 128)
        #   - Same depth (3 blocks) to preserve receptive field
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3), name='input'),

            # Block 1 — 24 filters, depthwise separable
            tf.keras.layers.SeparableConv2D(24, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(24, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 2 — 48 filters
            tf.keras.layers.SeparableConv2D(48, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(48, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 3 — 96 filters
            tf.keras.layers.SeparableConv2D(96, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(96, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Classifier
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ], name='edge_cnn')

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        n_params = model.count_params()
        print(f"[Architecture] Params: {n_params:,} "
              f"(baseline: {self.baseline_model.count_params():,})")
        print(f"[Architecture] Compression: "
              f"{self.baseline_model.count_params() / n_params:.2f}×")

        return model

    # =================================================================
    # 4. SIMPLIFIED NEURAL ARCHITECTURE SEARCH
    # =================================================================
    def implement_neural_architecture_search(self, num_candidates=3, train_epochs=2):
        """
        Implement simplified NAS for finding optimal edge architecture.

        Search space:
            - Width: {16, 24, 32, 48} filters per block
            - Depth: 2 or 3 blocks
            - Kernel size: {3, 5}

        Objective: Pareto-optimal trade-off between accuracy and model size.

        Args:
            num_candidates: Number of architectures to sample
            train_epochs: Training epochs per candidate (short budget)

        Returns:
            tuple: (best_architecture, search_results)
        """
        print("\n" + "=" * 60)
        print(f"[NAS] Simplified Neural Architecture Search "
              f"({num_candidates} candidates)")
        print("=" * 60)

        # Load data
        (x_train, y_train), (x_test, y_test) = load_cifar10_cached()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten().astype('int32')
        y_test = y_test.flatten().astype('int32')

        # Use subset for speed
        x_train_small = x_train[:20000]
        y_train_small = y_train[:20000]
        x_test_small = x_test[:2000]
        y_test_small = y_test[:2000]

        # Search space
        width_options = [16, 24, 32, 48]
        kernel_options = [3, 5]

        # Sample random architectures (deterministic seed)
        rng = np.random.RandomState(42)
        candidates = []
        for i in range(num_candidates):
            w1 = int(rng.choice(width_options))
            w2 = int(w1 * 2)
            w3 = int(w1 * 4)
            k = int(rng.choice(kernel_options))
            candidates.append({
                'id': i,
                'widths': [w1, w2, w3],
                'kernel': k,
            })

        search_results = []
        for cand in candidates:
            print(f"\n[NAS] Candidate {cand['id']+1}/{num_candidates}: "
                  f"widths={cand['widths']}, kernel={cand['kernel']}")
            try:
                model = self._build_nas_candidate(
                    widths=cand['widths'],
                    kernel_size=cand['kernel']
                )
                n_params = model.count_params()

                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                start = time.time()
                model.fit(
                    x_train_small, y_train_small,
                    batch_size=128, epochs=train_epochs,
                    validation_split=0.1, verbose=0
                )
                train_time = time.time() - start

                _, acc = model.evaluate(x_test_small, y_test_small, verbose=0)

                search_results.append({
                    'id': cand['id'],
                    'widths': cand['widths'],
                    'kernel': cand['kernel'],
                    'params': int(n_params),
                    'test_accuracy': round(float(acc), 4),
                    'training_time_s': round(train_time, 2),
                })
                print(f"  → params={n_params:,}, acc={acc:.4f}, "
                      f"time={train_time:.1f}s")
            except Exception as e:
                print(f"  ✗ Candidate {cand['id']} failed: {e}")

        # Select best by accuracy/sqrt(params) heuristic
        if search_results:
            def score(r):
                return r['test_accuracy'] / np.sqrt(r['params'] / 1e5)
            best = max(search_results, key=score)

            print(f"\n[NAS] Best candidate: id={best['id']}, "
                  f"acc={best['test_accuracy']:.4f}, "
                  f"params={best['params']:,}")

            best_model = self._build_nas_candidate(
                widths=best['widths'], kernel_size=best['kernel']
            )
        else:
            best_model = None
            best = None

        return best_model, {
            'candidates': search_results,
            'best': best,
        }

    def _build_nas_candidate(self, widths, kernel_size):
        """Build a NAS candidate model with given widths and kernel size."""
        k = kernel_size
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),

            tf.keras.layers.SeparableConv2D(widths[0], (k, k), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.SeparableConv2D(widths[1], (k, k), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.SeparableConv2D(widths[2], (k, k), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ], name=f'nas_w{widths[0]}_k{k}')
        return model

    # =================================================================
    # 5. TFLITE CONVERSION & BENCHMARKING
    # =================================================================
    def create_tflite_models(self, models_dict):
        """
        Convert optimized models to TensorFlow Lite format.

        Args:
            models_dict: Dictionary of {name: keras.Model}

        Returns:
            dict: TensorFlow Lite models with metadata
        """
        print("\n" + "=" * 60)
        print("[TFLite] Converting optimized models to TFLite")
        print("=" * 60)

        (_, _), (x_test, y_test) = load_cifar10_cached()
        x_test = x_test.astype('float32') / 255.0
        y_test = y_test.flatten().astype('int32')

        out_dir = os.path.join(SCRIPT_DIR, 'edge_optimized_models')
        os.makedirs(out_dir, exist_ok=True)

        results = {}
        for name, model in models_dict.items():
            print(f"\n[TFLite] Converting: {name}")
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_bytes = converter.convert()

                path = os.path.join(out_dir, f'{name}.tflite')
                with open(path, 'wb') as f:
                    f.write(tflite_bytes)

                size_kb = len(tflite_bytes) / 1024

                # Benchmark latency
                latency_ms = self._measure_tflite_latency(path, x_test[:1])

                # Evaluate accuracy
                acc = self._evaluate_tflite_model(
                    path, x_test[:2000], y_test[:2000]
                )

                results[name] = {
                    'path': path,
                    'size_kb': round(size_kb, 1),
                    'latency_ms': round(latency_ms, 2),
                    'accuracy': round(float(acc), 4),
                }
                print(f"  → size={size_kb:.1f} KB, "
                      f"latency={latency_ms:.2f} ms, acc={acc:.4f}")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")
                results[name] = {'error': str(e)}

        return results

    def _measure_tflite_latency(self, model_path, sample_input, num_runs=50):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()

        x = sample_input.astype(np.float32)
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            x = (x / scale + zero_point).astype(np.int8)

        # Warm-up
        for _ in range(5):
            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()

        # Timed
        start = time.time()
        for _ in range(num_runs):
            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()
        return (time.time() - start) / num_runs * 1000

# =====================================================================
# BENCHMARK ENTRYPOINT
# =====================================================================
def benchmark_edge_optimizations():
    """
    Comprehensive benchmarking of edge optimization strategies.

    Returns:
        dict: Detailed performance analysis
    """
    print("\n" + "#" * 60)
    print("# EDGE OPTIMIZATION BENCHMARKING")
    print("#" * 60)

    if not os.path.exists(BASELINE_PATH):
        print(f"ERROR: baseline model not found at {BASELINE_PATH}")
        return {}

    optimizer = EdgeOptimizer(BASELINE_PATH)
    results = {}

    # Baseline reference metrics
    print("\n[Baseline] Measuring reference metrics...")
    (_, _), (x_test, y_test) = load_cifar10_cached()
    x_test = x_test.astype('float32') / 255.0
    y_test = y_test.flatten().astype('int32')

    baseline_loss, baseline_acc = optimizer.baseline_model.evaluate(
        x_test, y_test, verbose=0
    )
    baseline_params = optimizer.baseline_model.count_params()

    results['baseline'] = {
        'params': int(baseline_params),
        'test_accuracy': round(float(baseline_acc), 4),
        'model_size_mb': round(baseline_params * 4 / (1024 * 1024), 2),
    }
    print(f"[Baseline] Params: {baseline_params:,}, "
          f"acc: {baseline_acc:.4f}")

    # -----------------------------------------------------------------
    # 1. PRUNING at multiple sparsity levels
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Benchmark 1/4: Pruning at multiple sparsity levels")
    print("-" * 60)

    pruning_results = {}
    for sparsity in [0.5, 0.75, 0.9]:
        print(f"\n[Pruning] Target sparsity: {sparsity*100:.0f}%")
        try:
            model = optimizer.implement_pruning(
                target_sparsity=sparsity,
                fine_tune_epochs=3
            )
            _, acc = model.evaluate(x_test, y_test, verbose=0)

            # Compute actual sparsity
            total_w, zero_w = 0, 0
            for w in model.weights:
                total_w += int(tf.size(w).numpy())
                zero_w += int(tf.math.count_nonzero(tf.equal(w, 0)).numpy())

            actual_sparsity = zero_w / total_w if total_w > 0 else 0.0

            pruning_results[f'sparsity_{int(sparsity*100)}'] = {
                'target_sparsity': sparsity,
                'actual_sparsity': round(float(actual_sparsity), 4),
                'test_accuracy': round(float(acc), 4),
                'params': int(model.count_params()),
                'non_zero_params': int(model.count_params() * (1 - actual_sparsity)),
            }
            print(f"✓ Sparsity {sparsity*100:.0f}% — "
                  f"acc: {acc:.4f}, actual: {actual_sparsity*100:.1f}%")
        except Exception as e:
            print(f"✗ Pruning @ {sparsity} failed: {e}")
            import traceback; traceback.print_exc()
            pruning_results[f'sparsity_{int(sparsity*100)}'] = {'error': str(e)}

    results['pruning'] = pruning_results

    # -----------------------------------------------------------------
    # 2. QUANTIZATION
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Benchmark 2/4: Quantization (3 strategies)")
    print("-" * 60)


    try:
        print("\n[Sanity] Reloading baseline for quantization...")
        optimizer = EdgeOptimizer(BASELINE_PATH)
        quantization_results = optimizer.implement_quantization()
        results['quantization'] = quantization_results
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        import traceback; traceback.print_exc()
        results['quantization'] = {'error': str(e)}

    # -----------------------------------------------------------------
    # 3. ARCHITECTURE OPTIMIZATION
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Benchmark 3/4: Architecture Optimization")
    print("-" * 60)

    try:
        arch_model = optimizer.implement_architecture_optimization()

        (x_train_full, y_train_full), _ = load_cifar10_cached()
        x_train_full = x_train_full.astype('float32') / 255.0
        y_train_full = y_train_full.flatten().astype('int32')

        # Train a few epochs to give the architecture a fair chance
        print("[Architecture] Training for 30 epochs with LR scheduling...")
        arch_callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=7, restore_best_weights=True
            ),
        ]
        arch_model.fit(
            x_train_full, y_train_full,
            batch_size=128, epochs=30,
            validation_split=0.1,
            callbacks=arch_callbacks,
            verbose=1
        )
        _, arch_acc = arch_model.evaluate(x_test, y_test, verbose=0)

        results['architecture'] = {
            'params': int(arch_model.count_params()),
            'test_accuracy': round(float(arch_acc), 4),
            'compression_vs_baseline': round(
                baseline_params / arch_model.count_params(), 2
            ),
            'model_size_mb': round(
                arch_model.count_params() * 4 / (1024 * 1024), 2
            ),
        }
        print(f"✓ Architecture — params: {arch_model.count_params():,}, "
              f"acc: {arch_acc:.4f}")
    except Exception as e:
        print(f"✗ Architecture optimization failed: {e}")
        import traceback; traceback.print_exc()
        results['architecture'] = {'error': str(e)}

    # -----------------------------------------------------------------
    # 4. NEURAL ARCHITECTURE SEARCH
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Benchmark 4/4: Simplified NAS")
    print("-" * 60)

    try:
        best_model, nas_results = optimizer.implement_neural_architecture_search(
            num_candidates=3, train_epochs=15
        )
        results['nas'] = nas_results
    except Exception as e:
        print(f"✗ NAS failed: {e}")
        import traceback; traceback.print_exc()
        results['nas'] = {'error': str(e)}

    # -----------------------------------------------------------------
    # 5. TFLITE CONVERSION OF ALL MODELS
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Additional: TFLite conversion of optimized models")
    print("-" * 60)

    try:
        # Build a dictionary of models to convert
        models_to_convert = {
            'baseline_tflite': optimizer.baseline_model,
        }
        if isinstance(results.get('architecture'), dict) and \
           'error' not in results['architecture']:
            models_to_convert['architecture_opt'] = arch_model

        tflite_results = optimizer.create_tflite_models(models_to_convert)
        results['tflite_conversions'] = tflite_results
    except Exception as e:
        print(f"✗ TFLite conversion failed: {e}")
        results['tflite_conversions'] = {'error': str(e)}

    # -----------------------------------------------------------------
    # 6. EDGE DEPLOYMENT VIABILITY ANALYSIS
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Additional: Edge deployment viability analysis")
    print("-" * 60)

    viability = _analyze_deployment_viability(results)
    results['deployment_viability'] = viability

    # -----------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------
    out_path = os.path.join(SCRIPT_DIR, 'edge_optimization_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Save] Results → {out_path}")

    return results


def _analyze_deployment_viability(results):
    """
    Estimate viability of each optimized model on real edge hardware.

    Reference platforms (rough):
        - Raspberry Pi 4:     1.5 GHz ARM Cortex-A72, 4-8 GB RAM, 3-5 W
        - NVIDIA Jetson Nano: 1.43 GHz ARM A57, 4 GB RAM, 5-10 W
        - ESP32:              240 MHz Xtensa, 520 KB SRAM, ~0.5 W
        - Arduino Nano 33 BLE:64 MHz Cortex-M4, 256 KB RAM, ~0.05 W
    """
    platforms = {
        'raspberry_pi_4': {
            'ram_mb': 4096,
            'flash_mb': 32000,
            'power_mw': 5000,
            'supports_tflite': True,
        },
        'jetson_nano': {
            'ram_mb': 4096,
            'flash_mb': 16000,
            'power_mw': 10000,
            'supports_tflite': True,
        },
        'esp32': {
            'ram_mb': 0.5,
            'flash_mb': 4,
            'power_mw': 500,
            'supports_tflite': True,  # via TFLite Micro
        },
        'arduino_nano_33_ble': {
            'ram_mb': 0.25,
            'flash_mb': 1,
            'power_mw': 50,
            'supports_tflite': True,  # via TFLite Micro
        },
    }

    # Use best model size from quantization results
    best_size_kb = None
    if 'quantization' in results and isinstance(results['quantization'], dict):
        for key in ['int8', 'dynamic_range', 'float16']:
            if key in results['quantization'] and \
               'size_kb' in results['quantization'][key]:
                size = results['quantization'][key]['size_kb']
                if best_size_kb is None or size < best_size_kb:
                    best_size_kb = size

    if best_size_kb is None:
        best_size_kb = results.get('baseline', {}).get('model_size_mb', 1.24) * 1024

    viability = {}
    for name, p in platforms.items():
        fits_ram = best_size_kb / 1024 < p['ram_mb']
        fits_flash = best_size_kb / 1024 < p['flash_mb']
        viable = fits_ram and fits_flash
        viability[name] = {
            'fits_in_ram': bool(fits_ram),
            'fits_in_flash': bool(fits_flash),
            'viable': bool(viable),
            'best_model_size_kb': round(best_size_kb, 1),
            'power_budget_mw': p['power_mw'],
        }

    return viability


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    results = benchmark_edge_optimizations()

    print("\nEdge Optimization Results:")
    for optimization, metrics in results.items():
        print(f"{optimization}: {metrics}")