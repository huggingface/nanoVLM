"""
Fix Attention operator outputs for ONNX Runtime compatibility.

PyTorch's ONNX exporter creates Attention nodes with KV cache outputs even when
they're not used. ONNX Runtime 1.23.0 requires that if present_key/present_value
outputs exist, then past_key/past_value inputs must also exist. This script
removes the unused KV cache outputs.

Usage:
    python fix_attention_outputs.py --onnx_dir onnx_models
"""

import argparse
import os
import onnx
from onnx import helper


def fix_attention_node_outputs(model_path, output_path):
    """
    Remove unused KV cache outputs from Attention nodes.

    Args:
        model_path: Path to input ONNX model
        output_path: Path to save fixed model
    """
    print(f"Fixing {os.path.basename(model_path)}...")

    model = onnx.load(model_path)
    graph = model.graph

    # Find all Attention nodes
    attention_nodes = [n for n in graph.node if n.op_type == 'Attention']
    print(f"  Found {len(attention_nodes)} Attention nodes")

    if not attention_nodes:
        print("  No Attention nodes to fix")
        if model_path != output_path:
            onnx.save(model, output_path)
        return

    # Check if any Attention outputs are actually used
    # Build a set of all tensor names that are used as inputs
    used_tensors = set()
    for node in graph.node:
        for input_name in node.input:
            if input_name:  # Skip empty strings
                used_tensors.add(input_name)

    # Also check graph outputs
    for output in graph.output:
        used_tensors.add(output.name)

    nodes_modified = 0
    outputs_removed = 0

    for node in attention_nodes:
        if len(node.output) <= 1:
            continue  # Already has only 1 output

        # Keep only the first output (the attention result)
        # Remove KV cache outputs if they're not used
        original_outputs = list(node.output)
        used_outputs = [original_outputs[0]]  # Always keep first output

        # Check if additional outputs are used
        for i, output_name in enumerate(original_outputs[1:], start=1):
            if output_name in used_tensors:
                used_outputs.append(output_name)
                print(f"    Warning: Output {output_name} is used, keeping it")

        if len(used_outputs) < len(original_outputs):
            # Modify the node to have fewer outputs
            del node.output[:]
            node.output.extend(used_outputs)
            nodes_modified += 1
            outputs_removed += len(original_outputs) - len(used_outputs)

    print(f"  Modified {nodes_modified} nodes, removed {outputs_removed} unused outputs")

    # Save the modified model
    onnx.save(model, output_path)
    print(f"  Saved to {output_path}")


def fix_all_models(onnx_dir, in_place=True):
    """
    Fix all ONNX models in a directory.

    Args:
        onnx_dir: Directory containing ONNX models
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
            output_path = os.path.join(onnx_dir, f"{base_name}_fixed.onnx")

        fix_attention_node_outputs(input_path, output_path)

    print(f"\n✅ All models fixed!")


def main():
    parser = argparse.ArgumentParser(
        description='Fix Attention operator outputs for ONNX Runtime compatibility'
    )
    parser.add_argument(
        '--onnx_dir',
        type=str,
        default='onnx_models',
        help='Directory containing ONNX models'
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

    fix_all_models(args.onnx_dir, args.in_place)


if __name__ == '__main__':
    main()
