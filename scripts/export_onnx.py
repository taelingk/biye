#!/usr/bin/env python3
"""Export trained Keras model to ONNX with built-in standardization.

The exported ONNX is self-contained: standardize layers are embedded,
so raw (125,3) signal + (10,) clinical features go in, CO+VO2 come out.

Usage:
    python scripts/export_onnx.py --checkpoint outputs/checkpoints/best_model.keras
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cardiofit.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--output", type=str, default="outputs/onnx/cardiofit_multimodal.onnx"
    )
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    setup_logging()

    # --- 1. Load model ---
    model = tf.keras.models.load_model(args.checkpoint, compile=False)
    logger.info(f"Loaded model from {args.checkpoint}")
    logger.info(f"Inputs: {[i.name for i in model.inputs]}")
    logger.info(f"Outputs: {[o.name for o in model.outputs]}")

    # --- 2. Convert to ONNX ---
    spec = (
        tf.TensorSpec(model.inputs[0].shape, tf.float32, name="signal_input"),
        tf.TensorSpec(model.inputs[1].shape, tf.float32, name="clinical_input"),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=args.opset,
        output_path=str(output_path),
    )
    logger.info(f"ONNX exported to {output_path}")

    # --- 3. Verify with ONNX Runtime ---
    session = ort.InferenceSession(str(output_path))

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    logger.info(f"ONNX inputs: {input_names}")
    logger.info(f"ONNX outputs: {output_names}")

    # Test inference
    sig_test = np.random.randn(1, 125, 3).astype(np.float32)
    cli_test = np.random.randn(1, 10).astype(np.float32)

    onnx_pred = session.run(
        output_names, {"signal_input": sig_test, "clinical_input": cli_test}
    )
    keras_pred = model.predict([sig_test, cli_test], verbose=0)

    for i, (onnx_out, keras_out) in enumerate(zip(onnx_pred, keras_pred)):
        diff = np.abs(onnx_out - keras_out).max()
        name = output_names[i]
        logger.info(f"  {name}: max diff Keras vs ONNX = {diff:.6f}")
        assert diff < 1e-4, f"ONNX verification failed for {name}: max diff = {diff}"

    logger.info(
        "ONNX verification PASSED — self-contained with built-in standardization"
    )


if __name__ == "__main__":
    main()
