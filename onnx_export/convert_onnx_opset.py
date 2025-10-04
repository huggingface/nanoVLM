"""
Convert ONNX models to a different opset version.

Usage:
    python convert_onnx_opset.py --onnx_dir onnx_models --target_opset 24
"""

import argparse
import os
import onnx
from onnx import version_converter


def convert_model_opset(input_path, output_path, target_opset):
    """
    Convert an ONNX model to a different opset version.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save converted model
        target_opset: Target opset version
    """
    print(f"Converting {os.path.basename(input_path)} to opset {target_opset}...")

    # Load model
    model = onnx.load(input_path)

    # Check current opset
    current_opset = model.opset_import[0].version
    print(f"  Current opset: {current_opset}")

    if current_opset == target_opset:
        print(f"  Already at target opset {target_opset}, skipping conversion")
        if input_path != output_path:
            onnx.save(model, output_path)
        return

    # Convert to target opset
    try:
        converted_model = version_converter.convert_version(model, target_opset)
        print(f"  Converted to opset {target_opset}")

        # Save converted model
        onnx.save(converted_model, output_path)
        print(f"  Saved to {output_path}")

    except Exception as e:
        print(f"  ⚠️  Conversion failed: {e}")
        print(f"  Saving original model unchanged")
        if input_path != output_path:
            onnx.save(model, output_path)


def convert_all_models(onnx_dir, target_opset, in_place=True):
    """
    Convert all ONNX models in a directory to target opset.

    Args:
        onnx_dir: Directory containing ONNX models
        target_opset: Target opset version
        in_place: If True, overwrite original files. If False, create new files with _opsetXX suffix
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
            # Create output filename with opset suffix
            base_name = model_file.replace('.onnx', '')
            output_path = os.path.join(onnx_dir, f"{base_name}_opset{target_opset}.onnx")

        convert_model_opset(input_path, output_path, target_opset)

    print(f"\n✅ Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX models to different opset version')
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
        help='Create new files with _opsetXX suffix instead of overwriting'
    )

    args = parser.parse_args()

    convert_all_models(args.onnx_dir, args.target_opset, args.in_place)


if __name__ == '__main__':
    main()
