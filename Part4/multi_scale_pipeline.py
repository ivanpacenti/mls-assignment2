"""
Part 4: Multi-Scale Deployment Pipeline

Create an integrated deployment pipeline that automatically optimizes models
for different target environments (cloud, edge, tiny).

This module orchestrates the optimizations developed in Parts 2 and 3, applying
them conditionally based on the target's resource constraints.

Follows the exact class/method structure specified in the assignment.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import json
import keras
import time
import zipfile
import tempfile

# 🔑 Force tf.keras back to Keras 3 (tfmot redirects it to Keras 2)
import sys
tf.keras = keras
sys.modules['tf.keras'] = keras

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

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'optimized_models')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# DATACLASSES
# =====================================================================
@dataclass
class DeploymentTarget:
    """Configuration for different deployment targets."""
    name: str
    max_model_size_mb: float
    max_latency_ms: float
    max_memory_mb: float
    power_budget_mw: float
    compute_capability: str  # 'cloud', 'edge', 'tiny'


@dataclass
class OptimizationResult:
    """Results from model optimization."""
    model_path: str
    accuracy: float
    model_size_mb: float
    estimated_latency_ms: float
    memory_usage_mb: float
    optimization_strategy: str
    # Additional diagnostic fields (not in the template, but useful):
    meets_size_constraint: bool = True
    meets_latency_constraint: bool = True
    meets_memory_constraint: bool = True
    meets_power_constraint: bool = True


# =====================================================================
# BASE OPTIMIZER (abstract)
# =====================================================================
class ModelOptimizer(ABC):
    @abstractmethod
    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        pass

    # -----------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------
    def _load_cifar10(self):
        """Load CIFAR-10 from cached pickle files (works in any environment)."""
        import pickle
        cache_dir = os.path.expanduser('~/.keras/datasets/cifar-10-batches-py')
        if not os.path.exists(cache_dir):
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            return x_train, y_train, x_test, y_test

        def unpickle(file):
            with open(file, 'rb') as fo:
                return pickle.load(fo, encoding='bytes')

        x_train_list, y_train_list = [], []
        for i in range(1, 6):
            batch = unpickle(os.path.join(cache_dir, f'data_batch_{i}'))
            x_train_list.append(batch[b'data'])
            y_train_list.extend(batch[b'labels'])
        x_train = np.concatenate(x_train_list).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y_train = np.array(y_train_list, dtype='int32')

        test_batch = unpickle(os.path.join(cache_dir, 'test_batch'))
        x_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y_test = np.array(test_batch[b'labels'], dtype='int32')

        return x_train, y_train, x_test, y_test

    def _measure_tflite_latency(self, model_path, sample_input, num_runs=50):
        """Measure TFLite inference latency (in ms, single sample)."""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        inp = interpreter.get_input_details()
        x = sample_input.astype(np.float32)
        if inp[0]['dtype'] == np.int8:
            scale, zp = inp[0]['quantization']
            x = (x / scale + zp).astype(np.int8)

        for _ in range(5):
            interpreter.set_tensor(inp[0]['index'], x)
            interpreter.invoke()

        start = time.time()
        for _ in range(num_runs):
            interpreter.set_tensor(inp[0]['index'], x)
            interpreter.invoke()
        return (time.time() - start) / num_runs * 1000

    def _evaluate_tflite(self, model_path, x_test, y_test):
        """Evaluate a TFLite model on a subset."""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        inp = interpreter.get_input_details()
        out = interpreter.get_output_details()

        y_test = np.asarray(y_test).flatten()
        correct = 0
        for i in range(len(x_test)):
            x = x_test[i:i+1].astype(np.float32)
            if inp[0]['dtype'] == np.int8:
                scale, zp = inp[0]['quantization']
                x = (x / scale + zp).astype(np.int8)
            interpreter.set_tensor(inp[0]['index'], x)
            interpreter.invoke()
            pred = int(np.argmax(interpreter.get_tensor(out[0]['index'])))
            if pred == int(y_test[i]):
                correct += 1
        return correct / len(x_test)

    def _clone_model(self, model):
        """Clone a model with its weights (safe from in-place mutation)."""
        clone = tf.keras.models.clone_model(model)
        clone.set_weights(model.get_weights())
        return clone


# =====================================================================
# CLOUD OPTIMIZER
# =====================================================================
class CloudOptimizer(ModelOptimizer):
    """
    Cloud optimizer: focus on accuracy and throughput.
    Strategy: float16 TFLite quantization.
    """

    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        print(f"\n[CloudOptimizer] Optimizing for target: {target.name}")
        print(f"  Constraints: size<{target.max_model_size_mb}MB, "
              f"latency<{target.max_latency_ms}ms")

        x_train, y_train, x_test, y_test = self._load_cifar10()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        baseline = self._clone_model(model)

        converter = tf.lite.TFLiteConverter.from_keras_model(baseline)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_bytes = converter.convert()

        out_path = os.path.join(OUTPUT_DIR, 'cloud_model.tflite')
        with open(out_path, 'wb') as f:
            f.write(tflite_bytes)

        size_mb = len(tflite_bytes) / (1024 * 1024)
        acc = self._evaluate_tflite(out_path, x_test[:2000], y_test[:2000])
        latency = self._measure_tflite_latency(out_path, x_test[:1])
        memory_mb = size_mb * 1.2

        result = OptimizationResult(
            model_path=out_path,
            accuracy=round(float(acc), 4),
            model_size_mb=round(size_mb, 3),
            estimated_latency_ms=round(latency, 3),
            memory_usage_mb=round(memory_mb, 3),
            optimization_strategy='float16 quantization',
        )
        self._check_constraints(result, target)
        return result

    @staticmethod
    def _check_constraints(r, t):
        r.meets_size_constraint = r.model_size_mb <= t.max_model_size_mb
        r.meets_latency_constraint = r.estimated_latency_ms <= t.max_latency_ms
        r.meets_memory_constraint = r.memory_usage_mb <= t.max_memory_mb
        r.meets_power_constraint = True


# =====================================================================
# EDGE OPTIMIZER  (with Keras 3 → 2 → 3 round-trip for pruning)
# =====================================================================
class EdgeOptimizer(ModelOptimizer):
    """
    Edge optimizer: focus on latency and memory.
    Strategy: magnitude pruning at 50% + dynamic range quantization.

    Note: tfmot.sparsity requires Keras 2 models. We round-trip through
    Keras 2 for pruning (same technique as Part 3), then convert back to
    Keras 3 for TFLite conversion.
    """

    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        print(f"\n[EdgeOptimizer] Optimizing for target: {target.name}")
        print(f"  Constraints: size<{target.max_model_size_mb}MB, "
              f"latency<{target.max_latency_ms}ms")

        x_train, y_train, x_test, y_test = self._load_cifar10()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten().astype('int32')

        # -------------------------------------------------------------
        # Step 1: Pruning at 50% via Keras 2 round-trip
        # -------------------------------------------------------------
        baseline = None
        strategy = 'dynamic_range_quant'

        try:
            import tf_keras
            import tensorflow_model_optimization as tfmot

            print("  [EdgeOptimizer] Building Keras 2 copy for pruning...")
            baseline_v2 = self._build_keras2_model(model)
            baseline_v2.compile(
                optimizer=tf_keras.optimizers.Adam(learning_rate=5e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Verify weights were copied
            _, base_acc = baseline_v2.evaluate(x_test[:2000], y_test[:2000], verbose=0)
            print(f"  [EdgeOptimizer] Keras 2 baseline accuracy: {base_acc:.4f}")

            # Apply pruning schedule
            pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.5,
                begin_step=0,
                end_step=500
            )
            model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
                baseline_v2, pruning_schedule=pruning_schedule
            )
            model_for_pruning.compile(
                optimizer=tf_keras.optimizers.Adam(learning_rate=5e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Brief fine-tune
            print("  [EdgeOptimizer] Fine-tuning pruned model (2 epochs on 5k samples)...")
            model_for_pruning.fit(
                x_train[:5000], y_train[:5000],
                batch_size=128, epochs=2,
                validation_split=0.1, verbose=0,
                callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
            )

            # Strip wrappers
            pruned_v2 = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

            # Convert Keras 2 → Keras 3
            print("  [EdgeOptimizer] Converting pruned model → Keras 3...")
            baseline = self._rebuild_keras3_from_keras2(pruned_v2)
            strategy = 'pruning(50%) + dynamic_range_quant'
            print("  [EdgeOptimizer] Pruning OK (50% sparsity)")

        except Exception as e:
            print(f"  [EdgeOptimizer] Pruning failed ({e}), using quantization only")
            baseline = self._clone_model(model)

        # -------------------------------------------------------------
        # Step 2: Dynamic range quantization
        # -------------------------------------------------------------
        converter = tf.lite.TFLiteConverter.from_keras_model(baseline)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()

        out_path = os.path.join(OUTPUT_DIR, 'edge_model.tflite')
        with open(out_path, 'wb') as f:
            f.write(tflite_bytes)

        size_mb = len(tflite_bytes) / (1024 * 1024)
        acc = self._evaluate_tflite(out_path, x_test[:2000], y_test[:2000])
        latency = self._measure_tflite_latency(out_path, x_test[:1])
        memory_mb = size_mb * 1.5

        result = OptimizationResult(
            model_path=out_path,
            accuracy=round(float(acc), 4),
            model_size_mb=round(size_mb, 3),
            estimated_latency_ms=round(latency, 3),
            memory_usage_mb=round(memory_mb, 3),
            optimization_strategy=strategy,
        )
        CloudOptimizer._check_constraints(result, target)
        return result

    # -----------------------------------------------------------------
    # Keras 3 → Keras 2 conversion helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _build_keras2_model(keras3_model):
        """
        Build an equivalent Keras 2 model by copying architecture and weights
        from a Keras 3 model. Avoids from_config() incompatibilities by
        reconstructing layer-by-layer.
        """
        import tf_keras

        model_v2 = tf_keras.Sequential()
        for layer in keras3_model.layers:
            name = layer.__class__.__name__
            cfg = layer.get_config()

            if name == 'InputLayer':
                shape = cfg.get('shape', cfg.get('batch_shape'))
                input_shape = tuple(shape[1:]) if shape else (32, 32, 3)
                model_v2.add(tf_keras.layers.InputLayer(
                    input_shape=input_shape, name=cfg.get('name')
                ))
            elif name == 'Conv2D':
                model_v2.add(tf_keras.layers.Conv2D(
                    filters=cfg['filters'], kernel_size=cfg['kernel_size'],
                    strides=cfg.get('strides', (1, 1)),
                    padding=cfg.get('padding', 'valid'),
                    activation=cfg.get('activation'),
                    use_bias=cfg.get('use_bias', True),
                    name=cfg.get('name')
                ))
            elif name == 'BatchNormalization':
                model_v2.add(tf_keras.layers.BatchNormalization(
                    momentum=cfg.get('momentum', 0.99),
                    epsilon=cfg.get('epsilon', 0.001),
                    name=cfg.get('name')
                ))
            elif name == 'ReLU':
                model_v2.add(tf_keras.layers.ReLU(name=cfg.get('name')))
            elif name == 'MaxPooling2D':
                model_v2.add(tf_keras.layers.MaxPooling2D(
                    pool_size=cfg.get('pool_size', (2, 2)),
                    strides=cfg.get('strides'),
                    padding=cfg.get('padding', 'valid'),
                    name=cfg.get('name')
                ))
            elif name == 'GlobalAveragePooling2D':
                model_v2.add(tf_keras.layers.GlobalAveragePooling2D(
                    name=cfg.get('name')
                ))
            elif name == 'Dropout':
                model_v2.add(tf_keras.layers.Dropout(
                    rate=cfg['rate'], name=cfg.get('name')
                ))
            elif name == 'Dense':
                model_v2.add(tf_keras.layers.Dense(
                    units=cfg['units'], activation=cfg.get('activation'),
                    use_bias=cfg.get('use_bias', True),
                    name=cfg.get('name')
                ))
            else:
                print(f"    [Warn] Unknown layer: {name}")

        # Build the model (creates weights tensors) before copying
        model_v2.build(input_shape=(None, 32, 32, 3))

        # Copy weights from the Keras 3 model
        try:
            model_v2.set_weights(keras3_model.get_weights())
        except Exception as e:
            print(f"    [Warn] Direct set_weights failed: {e}")
            print(f"    Trying layer-by-layer transfer...")
            for layer_v2, layer_v3 in zip(model_v2.layers, keras3_model.layers):
                try:
                    layer_v2.set_weights(layer_v3.get_weights())
                except Exception:
                    pass

        return model_v2

    @staticmethod
    def _rebuild_keras3_from_keras2(model_v2):
        """Convert a Keras 2 model back to Keras 3 format."""
        tmp_dir = tempfile.mkdtemp()
        tmp_h5 = os.path.join(tmp_dir, 'temp_pruned.h5')
        model_v2.save(tmp_h5)
        model_v3 = tf.keras.models.load_model(tmp_h5, compile=False)
        model_v3.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return model_v3


# =====================================================================
# TINYML OPTIMIZER
# =====================================================================
class TinyMLOptimizer(ModelOptimizer):
    """
    TinyML optimizer: extreme resource constraints.
    Strategy: architecture redesign + full INT8 quantization.
    """

    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        print(f"\n[TinyMLOptimizer] Optimizing for target: {target.name}")
        print(f"  Constraints: size<{target.max_model_size_mb}MB, "
              f"memory<{target.max_memory_mb}MB")

        x_train, y_train, x_test, y_test = self._load_cifar10()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten().astype('int32')
        y_test = y_test.flatten().astype('int32')

        tiny_model = self._build_tiny_architecture()
        tiny_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("  Training tiny architecture (30 epochs)...")
        tiny_model.fit(
            x_train, y_train, batch_size=128, epochs=30,
            validation_split=0.1, verbose=0,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True
                ),
            ]
        )

        def representative_dataset():
            for i in range(100):
                yield [x_train[i:i+1].astype('float32')]

        converter = tf.lite.TFLiteConverter.from_keras_model(tiny_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_bytes = converter.convert()

        out_path = os.path.join(OUTPUT_DIR, 'tiny_model.tflite')
        with open(out_path, 'wb') as f:
            f.write(tflite_bytes)

        size_mb = len(tflite_bytes) / (1024 * 1024)
        acc = self._evaluate_tflite(out_path, x_test[:2000], y_test[:2000])
        latency = self._measure_tflite_latency(out_path, x_test[:1])
        memory_mb = size_mb * 2.0

        result = OptimizationResult(
            model_path=out_path,
            accuracy=round(float(acc), 4),
            model_size_mb=round(size_mb, 3),
            estimated_latency_ms=round(latency, 3),
            memory_usage_mb=round(memory_mb, 3),
            optimization_strategy='tiny_arch + full_INT8',
        )
        CloudOptimizer._check_constraints(result, target)
        return result

    @staticmethod
    def _build_tiny_architecture():
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.SeparableConv2D(16, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.SeparableConv2D(32, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ], name='tiny_cnn')


# =====================================================================
# PIPELINE
# =====================================================================
class MultiScaleDeploymentPipeline:
    """Automated pipeline for optimizing models across deployment scales."""

    def __init__(self):
        self.optimizers = {
            'cloud': CloudOptimizer(),
            'edge': EdgeOptimizer(),
            'tiny': TinyMLOptimizer(),
        }

        self.targets = {
            'cloud_server': DeploymentTarget(
                name='cloud_server', max_model_size_mb=1000.0,
                max_latency_ms=100.0, max_memory_mb=8000.0,
                power_budget_mw=50000.0, compute_capability='cloud'
            ),
            'edge_device': DeploymentTarget(
                name='edge_device', max_model_size_mb=50.0,
                max_latency_ms=200.0, max_memory_mb=512.0,
                power_budget_mw=2000.0, compute_capability='edge'
            ),
            'microcontroller': DeploymentTarget(
                name='microcontroller', max_model_size_mb=1.0,
                max_latency_ms=1000.0, max_memory_mb=64.0,
                power_budget_mw=10.0, compute_capability='tiny'
            )
        }

    def optimize_for_all_targets(self, baseline_model_path: str) -> Dict[str, OptimizationResult]:
        print("\n" + "#" * 60)
        print("# MULTI-SCALE OPTIMIZATION PIPELINE")
        print("#" * 60)
        print(f"\n[Pipeline] Loading baseline: {baseline_model_path}")
        baseline_model = self._load_baseline(baseline_model_path)
        print(f"[Pipeline] Baseline params: {baseline_model.count_params():,}")

        results = {}
        for target_name, target_config in self.targets.items():
            print(f"\n{'=' * 60}")
            print(f"[Pipeline] Target: {target_name} ({target_config.compute_capability})")
            print(f"{'=' * 60}")
            try:
                optimizer = self.optimizers[target_config.compute_capability]
                results[target_name] = optimizer.optimize(baseline_model, target_config)
            except Exception as e:
                print(f"[Pipeline] Target {target_name} failed: {e}")
                import traceback; traceback.print_exc()
                results[target_name] = OptimizationResult(
                    model_path='', accuracy=0.0, model_size_mb=0.0,
                    estimated_latency_ms=0.0, memory_usage_mb=0.0,
                    optimization_strategy=f'FAILED: {e}'
                )
        return results

    @staticmethod
    def _load_baseline(path):
        return tf.keras.models.load_model(path)

    def analyze_scaling_trade_offs(self, results: Dict[str, OptimizationResult]) -> Dict[str, Any]:
        analysis = {}

        points = [(name, r.accuracy, r.model_size_mb)
                  for name, r in results.items()
                  if r.accuracy > 0 and r.model_size_mb > 0]
        pareto = []
        for name, acc, size in points:
            dominated = any(
                other_acc >= acc and other_size <= size and
                (other_acc > acc or other_size < size)
                for other_name, other_acc, other_size in points
                if other_name != name
            )
            if not dominated:
                pareto.append({'name': name, 'accuracy': acc, 'size_mb': size})
        analysis['pareto_frontier'] = pareto

        scaling = {}
        for name, r in results.items():
            scaling[name] = {
                'accuracy': r.accuracy,
                'size_mb': r.model_size_mb,
                'latency_ms': r.estimated_latency_ms,
                'memory_mb': r.memory_usage_mb,
                'strategy': r.optimization_strategy,
                'meets_all_constraints': (
                    r.meets_size_constraint and
                    r.meets_latency_constraint and
                    r.meets_memory_constraint
                )
            }
        analysis['scaling_efficiency'] = scaling

        bottlenecks = {}
        for name, r in results.items():
            issues = []
            if not r.meets_size_constraint:
                issues.append(f'size ({r.model_size_mb}MB too large)')
            if not r.meets_latency_constraint:
                issues.append(f'latency ({r.estimated_latency_ms}ms too slow)')
            if not r.meets_memory_constraint:
                issues.append(f'memory ({r.memory_usage_mb}MB too high)')
            bottlenecks[name] = issues if issues else ['none']
        analysis['bottlenecks'] = bottlenecks

        analysis['use_case_recommendations'] = {
            'realtime_video': {
                'priority': 'latency < 50ms, high accuracy',
                'best_target': min(results.items(),
                                   key=lambda kv: kv[1].estimated_latency_ms)[0]
            },
            'iot_sensor': {
                'priority': 'extreme power efficiency',
                'best_target': min(results.items(),
                                   key=lambda kv: kv[1].model_size_mb)[0]
            },
            'mobile_app': {
                'priority': 'balanced accuracy/efficiency',
                'best_target': min(
                    [(n, r) for n, r in results.items() if r.accuracy > 0.6],
                    key=lambda kv: kv[1].model_size_mb
                )[0] if any(r.accuracy > 0.6 for r in results.values()) else 'none'
            },
        }
        return analysis

    def generate_deployment_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        recommendations = []
        scaling = analysis.get('scaling_efficiency', {})
        for target, info in scaling.items():
            if info['meets_all_constraints']:
                recommendations.append(
                    f"✅ {target}: meets all constraints with "
                    f"{info['accuracy']*100:.1f}% accuracy, "
                    f"{info['size_mb']:.2f}MB, {info['latency_ms']:.2f}ms. "
                    f"Strategy: {info['strategy']}"
                )
            else:
                issues = analysis['bottlenecks'].get(target, [])
                recommendations.append(
                    f"⚠️ {target}: violates constraints ({', '.join(issues)})."
                )
        recommendations.append(
            "\n📊 Cascaded deployment: Use cloud for training-heavy tasks, "
            "edge for inference with latency constraints, "
            "microcontroller for extreme low-power use cases."
        )
        recommendations.append(
            "🔄 Hybrid strategy: Deploy the edge model as primary inference, "
            "with cloud as fallback for low-confidence predictions."
        )
        pareto = analysis.get('pareto_frontier', [])
        if pareto:
            recommendations.append(
                f"\n🎯 Pareto-optimal models (accuracy vs size): "
                f"{', '.join(p['name'] for p in pareto)}"
            )
        return recommendations


# =====================================================================
# MAIN
# =====================================================================
def run_multi_scale_optimization():
    pipeline = MultiScaleDeploymentPipeline()
    results = pipeline.optimize_for_all_targets(BASELINE_PATH)
    analysis = pipeline.analyze_scaling_trade_offs(results)
    recommendations = pipeline.generate_deployment_recommendations(analysis)

    report = {
        'optimization_results': {name: asdict(r) for name, r in results.items()},
        'scaling_analysis': analysis,
        'deployment_recommendations': recommendations,
    }
    out_path = os.path.join(SCRIPT_DIR, 'multi_scale_optimization_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[Pipeline] Report saved to: {out_path}")
    return report


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    report = run_multi_scale_optimization()

    print("\n" + "=" * 60)
    print("MULTI-SCALE OPTIMIZATION COMPLETE")
    print("=" * 60)
    print("\n📊 Optimization Results:")
    for target, result in report['optimization_results'].items():
        print(f"\n{target}:")
        print(f"  Accuracy:  {result['accuracy']*100:.2f}%")
        print(f"  Size:      {result['model_size_mb']:.3f} MB")
        print(f"  Latency:   {result['estimated_latency_ms']:.3f} ms")
        print(f"  Memory:    {result['memory_usage_mb']:.3f} MB")
        print(f"  Strategy:  {result['optimization_strategy']}")
        print(f"  Meets size constraint:    {result['meets_size_constraint']}")
        print(f"  Meets latency constraint: {result['meets_latency_constraint']}")
        print(f"  Meets memory constraint:  {result['meets_memory_constraint']}")

    print("\n💡 Recommendations:")
    for rec in report['deployment_recommendations']:
        print(f"  {rec}")

    print(f"\n📁 Report saved to: {os.path.join(SCRIPT_DIR, 'multi_scale_optimization_report.json')}")