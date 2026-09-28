"""
Edge deployment inference example.

Loads the edge-optimized TFLite model (pruning + dynamic range quantization)
and runs a prediction on a CIFAR-10 test sample.

Model: edge_model.tflite
Size: 0.328 MB
Accuracy: 82.35%
Strategy: pruning(50%) + dynamic_range_quant
"""

import os
import numpy as np
import pickle
import tensorflow as tf


def load_cifar10_sample(idx=42):
    cache_dir = os.path.expanduser('~/.keras/datasets/cifar-10-batches-py')
    with open(os.path.join(cache_dir, 'test_batch'), 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    x = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(batch[b'labels'])
    return x[idx:idx+1].astype('float32') / 255.0, y[idx]


def run_inference(model_path, sample):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    x = sample
    if inp[0]['dtype'] == np.int8:
        scale, zp = inp[0]['quantization']
        x = (x / scale + zp).astype(np.int8)

    interpreter.set_tensor(inp[0]['index'], x)
    interpreter.invoke()
    pred = interpreter.get_tensor(out[0]['index'])

    if pred.dtype == np.int8:
        scale, zp = out[0]['quantization']
        pred = (pred.astype(np.float32) - zp) * scale

    return pred


if __name__ == '__main__':
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    MODEL_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'optimized_models', 'edge_model.tflite'
    )

    print("=" * 60)
    print("EDGE DEPLOYMENT INFERENCE")
    print("=" * 60)
    print(f"Model:    {MODEL_PATH}")
    print(f"Size:     {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")
    print(f"Strategy: pruning(50%) + dynamic_range_quant")
    print()

    sample, true_label = load_cifar10_sample(idx=42)
    pred = run_inference(MODEL_PATH, sample)
    pred_class = int(np.argmax(pred))
    confidence = float(np.max(tf.nn.softmax(pred[0])))

    print(f"True label:       {CLASS_NAMES[true_label]}")
    print(f"Predicted label:  {CLASS_NAMES[pred_class]}")
    print(f"Confidence:       {confidence * 100:.2f}%")
    print(f"Correct:          {'✅' if pred_class == true_label else '❌'}")