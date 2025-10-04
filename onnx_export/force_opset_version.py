"""
Force ONNX models to use a specific opset version by directly modifying metadata.

This is useful when the models use operators that are compatible with the target opset,
but the automatic version converter fails.

Usage:
    python force_opset_version.py --onnx_dir onnx_models --target_opset 24
"""

import argparse
import os
import onnx


def force_opset_version(input_path, output_path, target_opset):
    """
    Force an ONNX model to use a specific opset version.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save modified model
        target_opset: Target opset version
    """
    print(f"Forcing {os.path.basename(input_path)} to opset {target_opset}...")

    # Load model
    model = onnx.load(input_path)

    # Check current opset
    current_opset = model.opset_import[0].version
    print(f"  Current opset: {current_opset}")

    if current_opset == target_opset:
        print(f"  Already at target opset {target_opset}, skipping")
        if input_path != output_path:
            onnx.save(model, output_path)
        return

    # Modify opset version directly
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset.version = target_opset

    print(f"  Updated opset to {target_opset}")

    # Validate the modified model
    try:
        onnx.checker.check_model(model)
        print(f"  ✅ Model is valid")
    except Exception as e:
        print(f"  ⚠️  Model validation warning: {e}")
        print(f"  Proceeding anyway...")

    # Save modified model
    onnx.save(model, output_path)
    print(f"  Saved to {output_path}")


def force_all_models(onnx_dir, target_opset, in_place=True):
    """
    Force all ONNX models in a directory to use target opset.

    Args:
        onnx_dir: Directory containing ONNX models
        target_opset: Target opset version
        in_place: If True, overwrite original files
    """
    model_files = [
        "vision_encoder.onnx",
        "modality_projector.onnx",
        "language_decoder_prefill.onnx",
        "language_decoder_decode.onnx"
    ]

    for model_file in model_files:
        input_path = os.path.join(onnx_dir, model_file)

        if not os.path.exists(input_path):
            print(f"⚠️  {model_file} not found, skipping")
            continue

        if in_place:
            output_path = input_path
        else:
            base_name = model_file.replace('.onnx', '')
            output_path = os.path.join(onnx_dir, f"{base_name}_opset{target_opset}.onnx")

        force_opset_version(input_path, output_path, target_opset)

    print(f"\n✅ All models updated to opset {target_opset}!")


def main():
    parser = argparse.ArgumentParser(
        description='Force ONNX models to use a specific opset version'
    )
    parser.add_argument(
        '--onnx_dir',
        type=str,
        default='onnx_models',
        help='Directory containing ONNX models'
    )
    parser.add_argument(
        '--target_opset',
        type=int,
        default=24,
        help='Target ONNX opset version'
    )
    parser.add_argument(
        '--in_place',
        action='store_true',
        default=True,
        help='Overwrite original files (default: True)'
    )
    parser.add_argument(
        '--no_in_place',
        action='store_false',
        dest='in_place',
        help='Create new files instead of overwriting'
    )

    args = parser.parse_args()

    force_all_models(args.onnx_dir, args.target_opset, args.in_place)


if __name__ == '__main__':
    main()
