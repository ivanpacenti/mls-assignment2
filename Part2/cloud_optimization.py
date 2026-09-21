"""
Part 2: Cloud-Scale Optimization

Optimize the baseline model for cloud deployment where computational resources
are abundant but efficiency still matters for cost and throughput.

Strategies implemented:
    1. Mixed precision training/inference
    2. Model parallelism (MirroredStrategy / MultiWorkerMirroredStrategy / ParameterServer)
    3. Batch processing optimization (gradient accumulation + tf.data pipeline)
    4. Knowledge distillation (teacher -> student)

Follows the exact class/method structure specified in the assignment.
"""

import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np
import time
import json
import os

# =====================================================================
# Path handling: allow running from Part2/ while loading baseline from Part1/
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Try common locations for the baseline model
_CANDIDATE_PATHS = [
    os.path.join(PROJECT_ROOT, 'Part1', 'baseline_model.keras'),
    os.path.join(PROJECT_ROOT, 'baseline_model.keras'),
    'baseline_model.keras',
]
BASELINE_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)),
                     _CANDIDATE_PATHS[0])


class CloudOptimizer:
    """
    Cloud-side optimizer providing four orthogonal optimization strategies.

    The class mirrors the assignment template: same methods, same signatures.
    """

    def __init__(self, baseline_model_path):
        print(f"[CloudOptimizer] Loading baseline from: {baseline_model_path}")
        self.baseline_model = tf.keras.models.load_model(baseline_model_path)
        self.baseline_model_path = baseline_model_path
        print(f"[CloudOptimizer] Baseline loaded — "
              f"params: {self.baseline_model.count_params():,}")

    # =================================================================
    # 1. MIXED PRECISION
    # =================================================================
    def implement_mixed_precision(self):
        """
        Implement mixed precision training/inference for cloud deployment.

        - Enables the `mixed_float16` global policy (float16 compute, float32 master weights)
        - Rebuilds the model so all layers inherit the new policy
        - Keeps the final softmax layer in float32 for numerical stability
        - Loss scaling is handled automatically by Keras when mixed_float16 is active

        Returns:
            tf.keras.Model: Model optimized with mixed precision
        """
        print("\n" + "=" * 60)
        print("[Mixed Precision] Enabling mixed_float16 policy")
        print("=" * 60)

        # Enable global mixed precision policy
        mixed_precision.set_global_policy('mixed_float16')
        print(f"[Mixed Precision] Global policy → {mixed_precision.global_policy()}")

        # Rebuild model from config so layers pick up the new dtype policy
        model_config = self.baseline_model.get_config()
        mp_model = tf.keras.Sequential.from_config(model_config)

        # Keep the output layer in float32 for numerical stability
        try:
            mp_model.layers[-1].dtype_policy = mixed_precision.Policy('float32')
        except Exception:
            pass  # Some Keras versions don't expose dtype_policy on layers

        # Copy weights from baseline
        mp_model.set_weights(self.baseline_model.get_weights())

        # Compile with Adam; loss scaling is automatic under mixed_float16
        mp_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"[Mixed Precision] Model policy: {mp_model.dtype_policy}")
        print(f"[Mixed Precision] Output dtype: {mp_model.layers[-1].dtype}")

        return mp_model

    # =================================================================
    # 2. MODEL PARALLELISM
    # =================================================================
    def implement_model_parallelism(self, strategy='mirrored'):
        """
        Implement distributed training strategy for multi-GPU cloud deployment.

        Args:
            strategy: 'mirrored', 'multi_worker_mirrored', or 'parameter_server'

        Returns:
            tuple: (distributed_model, training_strategy)
        """
        print("\n" + "=" * 60)
        print(f"[Model Parallelism] Setting up '{strategy}' strategy")
        print("=" * 60)

        # Detect available devices
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')
        print(f"[Model Parallelism] GPUs: {len(gpus)} — CPUs: {len(cpus)}")

        # Build the strategy object
        if strategy == 'mirrored':
            if len(gpus) >= 1:
                strategy_obj = tf.distribute.MirroredStrategy()
            else:
                # Fallback: simulate on CPU (single replica)
                print("[Model Parallelism] WARNING: no GPU found — "
                      "simulating MirroredStrategy on CPU")
                strategy_obj = tf.distribute.MirroredStrategy(devices=['CPU:0'])
        elif strategy == 'multi_worker_mirrored':
            strategy_obj = tf.distribute.MultiWorkerMirroredStrategy()
        elif strategy == 'parameter_server':
            cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
            strategy_obj = tf.distribute.experimental.ParameterServerStrategy(
                cluster_resolver
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        n_replicas = strategy_obj.num_replicas_in_sync
        print(f"[Model Parallelism] Replicas: {n_replicas}")

        # Build & compile inside the strategy scope
        with strategy_obj.scope():
            model_config = self.baseline_model.get_config()
            dist_model = tf.keras.Sequential.from_config(model_config)

            # Scale learning rate linearly with the number of replicas
            base_lr = 1e-3
            scaled_lr = base_lr * max(1, n_replicas)

            dist_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=scaled_lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

        print(f"[Model Parallelism] LR scaled: {base_lr} → {scaled_lr}")

        return dist_model, strategy_obj

    # =================================================================
    # 3. BATCH PROCESSING OPTIMIZATION
    # =================================================================
    def optimize_batch_processing(self, target_batch_size=256):
        """
        Optimize for large batch processing typical in cloud environments.

        Args:
            target_batch_size: Target effective batch size for cloud deployment

        Returns:
            dict: Optimized training configuration
        """
        print("\n" + "=" * 60)
        print(f"[Batch Processing] Target batch size: {target_batch_size}")
        print("=" * 60)

        # Hardware-friendly batch size (per step); if target is bigger,
        # we accumulate gradients over multiple micro-batches.
        hardware_batch = min(target_batch_size, 256)
        accumulation_steps = max(1, target_batch_size // hardware_batch)

        config = {
            'target_batch_size': target_batch_size,
            'hardware_batch_size': hardware_batch,
            'gradient_accumulation_steps': accumulation_steps,
            'prefetch_buffer_size': tf.data.AUTOTUNE,
            'num_parallel_calls': tf.data.AUTOTUNE,
            'use_caching': True,
            'shuffle_buffer_size': 10000,
            'steps_per_execution': 10,  # speeds up small models
        }

        print(f"[Batch Processing] Hardware batch: {hardware_batch}")
        print(f"[Batch Processing] Accumulation steps: {accumulation_steps}")
        print(f"[Batch Processing] Effective batch: "
              f"{hardware_batch * accumulation_steps}")

        return config

    def _build_optimized_data_pipeline(self, x_train, y_train, batch_size=256):
        """
        Helper: build a high-throughput tf.data pipeline.
        Used by benchmark_cloud_optimizations() to test batch sizes.
        """
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds = ds.shuffle(buffer_size=10000)
        ds = ds.batch(batch_size, drop_remainder=True)
        if len(x_train) <= 100000:
            ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # =================================================================
    # 4. KNOWLEDGE DISTILLATION
    # =================================================================
    def implement_knowledge_distillation(self):
        """
        Create a larger teacher model and distill knowledge to student model.

        Returns:
            tuple: (teacher_model, student_model, distillation_training_function)
        """
        print("\n" + "=" * 60)
        print("[Knowledge Distillation] Building teacher and student")
        print("=" * 60)

        # -------------------------------------------------------------
        # Teacher: roughly 2× the parameters of the baseline.
        # Baseline uses 32/64/128 conv filters → teacher uses 64/128/256.
        # -------------------------------------------------------------
        teacher_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3), name='input'),

            # Block 1 (64 filters instead of 32)
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 2 (128 filters instead of 64)
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 3 (256 filters instead of 128)
            tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Classifier
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax')
        ], name='teacher_cnn')

        teacher_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # -------------------------------------------------------------
        # Student: same architecture as the baseline.
        # -------------------------------------------------------------
        student_model = tf.keras.models.clone_model(self.baseline_model)
        student_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"[KD] Teacher params: {teacher_model.count_params():,}")
        print(f"[KD] Student params: {student_model.count_params():,}")
        print(f"[KD] Compression ratio: "
              f"{teacher_model.count_params() / student_model.count_params():.2f}×")

        # -------------------------------------------------------------
        # Custom distillation training function
        # -------------------------------------------------------------
        def distillation_training_function(
            teacher, student, x_train, y_train,
            x_test, y_test, epochs=10, batch_size=128,
            temperature=4.0, alpha=0.3
        ):
            """
            Combined loss:
                L = alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T))
                  + (1 - alpha) * CE(y_true, student)

            Args:
                temperature: softens the teacher distribution (T > 1)
                alpha: weight on the distillation loss vs hard-label loss
            """
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
            ce = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            )

            n_samples = len(x_train)
            steps_per_epoch = n_samples // batch_size

            history = {'loss': [], 'accuracy': [],
                       'val_loss': [], 'val_accuracy': []}

            for epoch in range(epochs):
                t0 = time.time()
                epoch_loss, epoch_acc = 0.0, 0.0

                # Shuffle each epoch
                idx = np.random.permutation(n_samples)
                x_shuf, y_shuf = x_train[idx], y_train[idx]

                for step in range(steps_per_epoch):
                    s = step * batch_size
                    e = s + batch_size
                    x_batch = x_shuf[s:e]
                    y_batch = y_shuf[s:e]

                    # Teacher soft targets (no gradient)
                    teacher_probs = teacher(x_batch, training=False)

                    with tf.GradientTape() as tape:
                        student_logits = student(x_batch, training=True)

                        soft_teacher = tf.nn.softmax(teacher_probs / temperature)
                        soft_student = tf.nn.softmax(student_logits / temperature)

                        d_loss = tf.reduce_mean(kl(soft_teacher, soft_student))
                        d_loss = d_loss * (temperature ** 2)  # standard scaling

                        s_loss = tf.reduce_mean(ce(y_batch, student_logits))

                        total_loss = alpha * d_loss + (1 - alpha) * s_loss

                    grads = tape.gradient(total_loss, student.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, student.trainable_variables)
                    )

                    epoch_loss += float(total_loss.numpy())

                    preds = tf.argmax(student_logits, axis=1, output_type=tf.int32)
                    epoch_acc += float(tf.reduce_mean(
                        tf.cast(tf.equal(preds, y_batch), tf.float32)
                    ).numpy())

                val_loss, val_acc = student.evaluate(x_test, y_test, verbose=0)
                avg_loss = epoch_loss / steps_per_epoch
                avg_acc = epoch_acc / steps_per_epoch

                history['loss'].append(avg_loss)
                history['accuracy'].append(avg_acc)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)

                print(f"  Epoch {epoch+1}/{epochs} — "
                      f"loss: {avg_loss:.4f} — acc: {avg_acc:.4f} — "
                      f"val_loss: {val_loss:.4f} — val_acc: {val_acc:.4f} — "
                      f"{time.time()-t0:.1f}s")

            return student, history

        return teacher_model, student_model, distillation_training_function


# =====================================================================
# BENCHMARK ENTRYPOINT
# =====================================================================
def benchmark_cloud_optimizations():
    """
    Benchmark different cloud optimization strategies.

    Returns:
        dict: Performance metrics for each optimization
    """
    print("\n" + "#" * 60)
    print("# CLOUD OPTIMIZATION BENCHMARKING")
    print("#" * 60)

    if not os.path.exists(BASELINE_PATH):
        print(f"ERROR: baseline model not found at {BASELINE_PATH}")
        print("Please run Part1/baseline.py first.")
        return {}

    optimizer = CloudOptimizer(BASELINE_PATH)
    results = {}

    # Load CIFAR-10 (already cached by Part 1)
    print("\n[Data] Loading CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = y_train.flatten().astype('int32')
    y_test = y_test.flatten().astype('int32')

    # Subset for fast benchmarking
    SUBSET = 10000
    x_train_small = x_train[:SUBSET]
    y_train_small = y_train[:SUBSET]
    print(f"[Data] Using {SUBSET} samples for benchmarking")

    # -----------------------------------------------------------------
    # 1. Mixed precision
    #    → measure training time, memory usage, accuracy
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Benchmark 1/4: Mixed Precision")
    print("-" * 60)
    try:
        mp_model = optimizer.implement_mixed_precision()

        start = time.time()
        mp_model.fit(x_train_small, y_train_small,
                     epochs=3, batch_size=128, verbose=1)
        mp_time = time.time() - start
        _, mp_acc = mp_model.evaluate(x_test, y_test, verbose=0)

        # Float32 reference for comparison
        print("[Mixed Precision] Float32 reference run...")
        mixed_precision.set_global_policy('float32')
        fp32 = tf.keras.models.clone_model(optimizer.baseline_model)
        fp32.set_weights(optimizer.baseline_model.get_weights())
        fp32.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        start = time.time()
        fp32.fit(x_train_small, y_train_small,
                 epochs=3, batch_size=128, verbose=0)
        fp32_time = time.time() - start
        _, fp32_acc = fp32.evaluate(x_test, y_test, verbose=0)

        results['mixed_precision'] = {
            'params': int(mp_model.count_params()),
            'training_time_3ep_s': round(mp_time, 2),
            'fp32_training_time_3ep_s': round(fp32_time, 2),
            'speedup_vs_fp32': round(fp32_time / mp_time, 3),
            'test_accuracy': round(float(mp_acc), 4),
            'fp32_test_accuracy': round(float(fp32_acc), 4),
            'accuracy_delta': round(float(mp_acc - fp32_acc), 4),
            'note': 'Apple Silicon does not accelerate float16 on GPU; '
                    'speedup limited on this hardware.'
        }
        print(f"✓ Mixed precision — acc: {mp_acc:.4f}, time: {mp_time:.2f}s")
    except Exception as e:
        print(f"✗ Mixed precision failed: {e}")
        import traceback; traceback.print_exc()
        results['mixed_precision'] = {'error': str(e)}

    # Reset global policy
    mixed_precision.set_global_policy('float32')

    # -----------------------------------------------------------------
    # 2. Model parallelism
    #    → measure scaling efficiency (real if multi-GPU, simulated otherwise)
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Benchmark 2/4: Model Parallelism (MirroredStrategy)")
    print("-" * 60)
    try:
        dist_model, strategy = optimizer.implement_model_parallelism('mirrored')

        start = time.time()
        dist_model.fit(x_train_small, y_train_small,
                       epochs=3, batch_size=128, verbose=1)
        dist_time = time.time() - start
        _, dist_acc = dist_model.evaluate(x_test, y_test, verbose=0)

        n_rep = int(strategy.num_replicas_in_sync)
        results['model_parallelism'] = {
            'strategy': 'mirrored',
            'num_replicas': n_rep,
            'training_time_3ep_s': round(dist_time, 2),
            'test_accuracy': round(float(dist_acc), 4),
            'scaling_efficiency': (
                'N/A — single device' if n_rep <= 1
                else f'{n_rep}× replicas, efficiency depends on cluster'
            ),
            'note': 'On this machine, MirroredStrategy falls back to CPU '
                    'with 1 replica; scaling requires multi-GPU cluster.'
        }
        print(f"✓ Model parallelism — replicas: {n_rep}, acc: {dist_acc:.4f}")
    except Exception as e:
        print(f"✗ Model parallelism failed: {e}")
        import traceback; traceback.print_exc()
        results['model_parallelism'] = {'error': str(e)}

    # -----------------------------------------------------------------
    # 3. Batch processing
    #    → measure throughput at different batch sizes
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Benchmark 3/4: Batch Processing")
    print("-" * 60)

    batch_results = {}
    for bs in [64, 128, 256, 512]:
        print(f"\n[Batch {bs}] Training 3 epochs...")
        try:
            m = tf.keras.models.clone_model(optimizer.baseline_model)
            m.set_weights(optimizer.baseline_model.get_weights())
            m.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

            # Use optimized tf.data pipeline
            ds = optimizer._build_optimized_data_pipeline(
                x_train_small, y_train_small, batch_size=bs
            )

            start = time.time()
            m.fit(ds, epochs=3, verbose=0)
            train_time = time.time() - start
            throughput = (len(x_train_small) * 3) / train_time

            _, acc = m.evaluate(x_test, y_test, verbose=0)

            batch_results[str(bs)] = {
                'training_time_3ep_s': round(train_time, 2),
                'throughput_samples_per_s': round(throughput, 1),
                'test_accuracy': round(float(acc), 4),
            }
            print(f"  → {train_time:.2f}s — "
                  f"{throughput:.1f} samples/s — acc: {acc:.4f}")
        except Exception as e:
            print(f"  ✗ Batch {bs} failed: {e}")
            batch_results[str(bs)] = {'error': str(e)}

    # Configuration for gradient accumulation
    batch_results['config'] = optimizer.optimize_batch_processing(
        target_batch_size=1024
    )
    results['batch_processing'] = batch_results

    # -----------------------------------------------------------------
    # 4. Knowledge distillation
    #    → measure student performance vs teacher
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Benchmark 4/4: Knowledge Distillation")
    print("-" * 60)
    try:
        teacher, student, distill_fn = optimizer.implement_knowledge_distillation()

        # Short teacher training
        print("\n[KD] Teacher training (3 epochs)...")
        teacher.fit(x_train_small, y_train_small,
                    epochs=3, batch_size=128,
                    validation_split=0.1, verbose=1)
        _, teacher_acc = teacher.evaluate(x_test, y_test, verbose=0)

        # Distill to student
        print("\n[KD] Distillation to student...")
        distilled_student, kd_history = distill_fn(
            teacher, student,
            x_train_small, y_train_small,
            x_test, y_test,
            epochs=3, batch_size=128,
            temperature=4.0, alpha=0.3
        )
        _, student_acc = distilled_student.evaluate(x_test, y_test, verbose=0)

        # Save distilled student
        out_dir = os.path.join(SCRIPT_DIR, 'cloud_optimized_models')
        os.makedirs(out_dir, exist_ok=True)
        distilled_student.save(os.path.join(out_dir, 'distilled_student.keras'))

        results['knowledge_distillation'] = {
            'teacher_params': int(teacher.count_params()),
            'student_params': int(distilled_student.count_params()),
            'compression_ratio': round(
                teacher.count_params() / distilled_student.count_params(), 2
            ),
            'teacher_test_accuracy': round(float(teacher_acc), 4),
            'student_test_accuracy': round(float(student_acc), 4),
            'student_history': {
                k: [round(float(v), 4) for v in vals]
                for k, vals in kd_history.items()
            },
        }
        print(f"✓ KD — teacher: {teacher_acc:.4f}, student: {student_acc:.4f}")
    except Exception as e:
        print(f"✗ Knowledge distillation failed: {e}")
        import traceback; traceback.print_exc()
        results['knowledge_distillation'] = {'error': str(e)}

    # -----------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------
    out_path = os.path.join(SCRIPT_DIR, 'cloud_optimization_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Save] Results → {out_path}")

    return results


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    results = benchmark_cloud_optimizations()

    print("\nCloud Optimization Results:")
    for optimization, metrics in results.items():
        print(f"{optimization}: {metrics}")