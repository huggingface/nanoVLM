"""
Export nanoVLM to ExecuTorch format.

ExecuTorch is designed for on-device inference with pure PyTorch models.
We export the model with multiple signatures (entry points) for different phases.

Usage:
    pip install executorch
    python export_executorch.py --checkpoint lusxvr/nanoVLM --output_dir executorch_models
"""

import argparse
import os
import torch
from torch.export import Dim
from models.vision_language_model import VisionLanguageModel


class ExecuTorchVLMWrapper(torch.nn.Module):
    """Wrapper for nanoVLM with multiple signatures for ExecuTorch export."""

    def __init__(self, vlm_model):
        super().__init__()
        self.vision_encoder = vlm_model.vision_encoder
        self.modality_projector = vlm_model.MP
        self.decoder = vlm_model.decoder
        self.cfg = vlm_model.cfg

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to vision features.

        Args:
            images: [batch_size, 3, img_size, img_size]

        Returns:
            Vision features: [batch_size, num_patches, vit_hidden_dim]
        """
        return self.vision_encoder(images)

    def project_features(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to language embedding space.

        Args:
            vision_features: [batch_size, num_patches, vit_hidden_dim]

        Returns:
            Projected embeddings: [batch_size, mp_image_token_length, lm_hidden_dim]
        """
        return self.modality_projector(vision_features)

    def prefill(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """
        Prefill phase: process full sequence.

        Args:
            embeddings: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]

        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim]
            kv_cache: List of dicts with 'key' and 'value' tensors
        """
        return self.decoder(
            embeddings,
            attention_mask=attention_mask,
            return_kv_cache=True
        )

    def decode(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        start_pos: torch.Tensor,
        kv_cache: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """
        Decode phase: process single token with KV cache.

        Args:
            embeddings: [batch_size, 1, hidden_dim]
            attention_mask: [batch_size, total_seq_len]
            start_pos: [batch_size] position index
            kv_cache: List of dicts with 'key' and 'value' tensors

        Returns:
            hidden_states: [batch_size, 1, hidden_dim]
            updated_kv_cache: List of dicts with 'key' and 'value' tensors
        """
        return self.decoder(
            embeddings,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            start_pos=start_pos,
            return_kv_cache=True
        )


def export_to_executorch(checkpoint_path: str, output_dir: str, quantize: bool = False):
    """
    Export nanoVLM to ExecuTorch format.

    Args:
        checkpoint_path: Path to model checkpoint or HF repo
        output_dir: Directory to save ExecuTorch model
        quantize: Whether to apply int8 quantization (reduces model size ~4x)
    """
    print(f"Loading model from {checkpoint_path}...")
    vlm_model = VisionLanguageModel.from_pretrained(checkpoint_path)
    vlm_model.eval()

    cfg = vlm_model.cfg
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings BEFORE quantization (for text generation)
    embeddings_path = os.path.join(output_dir, "embeddings.pt")
    torch.save({
        'token_embedding': vlm_model.decoder.token_embedding.weight.clone(),
        'lm_head': vlm_model.decoder.head.weight.clone()
    }, embeddings_path)

    # Apply manual quantization if requested
    if quantize:
        print("Applying int8 weight-only quantization...")
        try:
            from torchao.quantization import quantize_, int8_weight_only

            # Apply weight-only int8 quantization (export-compatible)
            quantize_(vlm_model.vision_encoder, int8_weight_only())
            quantize_(vlm_model.MP, int8_weight_only())
            quantize_(vlm_model.decoder, int8_weight_only())
            print("✅ Model quantized (int8 weight-only)")
        except ImportError:
            print("⚠️  torchao not installed, skipping quantization")
            print("   Install with: pip install torchao")
            quantize = False

    # Create wrapper with multiple entry points
    wrapper = ExecuTorchVLMWrapper(vlm_model)
    wrapper.eval()

    print("\nExporting with torch.export...")

    # Create example inputs for each signature
    batch_size = 1

    # Vision encoding example
    print("\n1. Vision encoding signature...")
    img_size = cfg.vit_img_size
    example_image = torch.randn(batch_size, 3, img_size, img_size)

    # Create module wrappers for each function
    class VisionEncoderModule(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, images):
            return self.encoder(images)

    vision_module = VisionEncoderModule(wrapper.vision_encoder)
    vision_program = torch.export.export(
        vision_module,
        (example_image,),
        strict=False
    )
    print(f"   ✅ Vision encoding exported")

    # Modality projection example
    print("\n2. Modality projection signature...")
    num_patches = (cfg.vit_img_size // cfg.vit_patch_size) ** 2
    example_vision_features = torch.randn(batch_size, num_patches, cfg.vit_hidden_dim)

    class ProjectionModule(torch.nn.Module):
        def __init__(self, projector):
            super().__init__()
            self.projector = projector

        def forward(self, features):
            return self.projector(features)

    projection_module = ProjectionModule(wrapper.modality_projector)
    projection_program = torch.export.export(
        projection_module,
        (example_vision_features,),
        strict=False
    )
    print(f"   ✅ Modality projection exported")

    # Prefill example
    print("\n3. Language decoder prefill signature...")
    seq_len = 128
    example_embeddings = torch.randn(batch_size, seq_len, cfg.lm_hidden_dim)
    example_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    example_position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    class PrefillModule(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, embeddings, attention_mask, position_ids):
            return self.decoder(
                embeddings,
                attention_mask=attention_mask,
                kv_cache=None,
                position_ids=position_ids
            )

    prefill_module = PrefillModule(wrapper.decoder)

    # Define dynamic shapes for variable sequence length
    seq_dim = Dim("seq_len", min=1, max=cfg.lm_max_position_embeddings)
    prefill_dynamic_shapes = {
        "embeddings": {1: seq_dim},        # [batch, seq_len, hidden]
        "attention_mask": {1: seq_dim},    # [batch, seq_len]
        "position_ids": {1: seq_dim}       # [batch, seq_len]
    }

    prefill_program = torch.export.export(
        prefill_module,
        (example_embeddings, example_attention_mask, example_position_ids),
        dynamic_shapes=prefill_dynamic_shapes,
        strict=False
    )
    print(f"   ✅ Prefill exported (with dynamic sequence length)")

    # Decode example (with KV cache)
    print("\n4. Language decoder decode signature...")
    decode_embeddings = torch.randn(batch_size, 1, cfg.lm_hidden_dim)
    decode_attention_mask = torch.ones(batch_size, seq_len + 1, dtype=torch.long)
    decode_position_ids = torch.tensor([[seq_len]], dtype=torch.long)

    # Create example KV cache
    n_kv_heads = cfg.lm_n_kv_heads
    head_dim = cfg.lm_hidden_dim // cfg.lm_n_heads
    example_kv_cache = []
    for _ in range(cfg.lm_n_blocks):
        example_kv_cache.append({
            'key': torch.randn(batch_size, n_kv_heads, seq_len, head_dim),
            'value': torch.randn(batch_size, n_kv_heads, seq_len, head_dim)
        })

    class DecodeModule(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, embeddings, attention_mask, position_ids, kv_cache):
            return self.decoder(
                embeddings,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                position_ids=position_ids
            )

    decode_module = DecodeModule(wrapper.decoder)

    # Define dynamic shapes for decode (attention mask and KV cache grow)
    kv_seq_dim = Dim("kv_seq_len", min=1, max=cfg.lm_max_position_embeddings)
    decode_dynamic_shapes = {
        "embeddings": None,  # Always [1, 1, hidden]
        "attention_mask": {1: kv_seq_dim + 1},  # [batch, kv_seq_len + 1]
        "position_ids": None,  # Always [1, 1]
        "kv_cache": [{
            "key": {2: kv_seq_dim},    # [batch, n_kv_heads, kv_seq_len, head_dim]
            "value": {2: kv_seq_dim}
        } for _ in range(cfg.lm_n_blocks)]
    }

    decode_program = torch.export.export(
        decode_module,
        (decode_embeddings, decode_attention_mask, decode_position_ids, example_kv_cache),
        dynamic_shapes=decode_dynamic_shapes,
        strict=False
    )
    print(f"   ✅ Decode exported (with dynamic KV cache length)")

    # Save .pt2 files (torch.export format) for testing/inference
    print("\nSaving torch.export .pt2 files...")
    torch.export.save(vision_program, os.path.join(output_dir, "vision_encoder.pt2"))
    torch.export.save(projection_program, os.path.join(output_dir, "modality_projector.pt2"))
    torch.export.save(prefill_program, os.path.join(output_dir, "language_decoder_prefill.pt2"))
    torch.export.save(decode_program, os.path.join(output_dir, "language_decoder_decode.pt2"))
    print("   ✅ Saved .pt2 files")

    print("\nConverting to ExecuTorch format...")

    try:
        from executorch.exir import to_edge
        from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager

        # Convert each program to edge dialect
        print("   Converting to edge dialect...")
        vision_edge = to_edge(vision_program)
        projection_edge = to_edge(projection_program)
        prefill_edge = to_edge(prefill_program)
        decode_edge = to_edge(decode_program)

        print("   ✅ Converted to edge dialect")

        # Convert to ExecuTorch
        vision_et = vision_edge.to_executorch()
        projection_et = projection_edge.to_executorch()
        prefill_et = prefill_edge.to_executorch()
        decode_et = decode_edge.to_executorch()

        print("   ✅ Converted to ExecuTorch format")

        # Save models
        vision_path = os.path.join(output_dir, "vision_encoder.pte")
        projection_path = os.path.join(output_dir, "modality_projector.pte")
        prefill_path = os.path.join(output_dir, "language_decoder_prefill.pte")
        decode_path = os.path.join(output_dir, "language_decoder_decode.pte")

        with open(vision_path, "wb") as f:
            f.write(vision_et.buffer)
        with open(projection_path, "wb") as f:
            f.write(projection_et.buffer)
        with open(prefill_path, "wb") as f:
            f.write(prefill_et.buffer)
        with open(decode_path, "wb") as f:
            f.write(decode_et.buffer)

        print(f"\n✅ ExecuTorch models saved to {output_dir}:")
        print(f"   - vision_encoder.pte")
        print(f"   - modality_projector.pte")
        print(f"   - language_decoder_prefill.pte")
        print(f"   - language_decoder_decode.pte")

    except ImportError as e:
        print(f"\n⚠️  ExecuTorch not installed: {e}")
        print("   Exported programs saved as .pt2 files instead:")
    except Exception as e:
        print(f"\n⚠️  ExecuTorch conversion failed: {type(e).__name__}")
        print(f"   Error: {str(e)[:200]}")
        print("   This is likely due to SDPA (scaled_dot_product_attention) decomposition issues.")
        print("   Exported programs saved as .pt2 files instead:")

        # Save as ExportedProgram files
        vision_path = os.path.join(output_dir, "vision_encoder.pt2")
        projection_path = os.path.join(output_dir, "modality_projector.pt2")
        prefill_path = os.path.join(output_dir, "language_decoder_prefill.pt2")
        decode_path = os.path.join(output_dir, "language_decoder_decode.pt2")

        torch.export.save(vision_program, vision_path)
        torch.export.save(projection_program, projection_path)
        torch.export.save(prefill_program, prefill_path)
        torch.export.save(decode_program, decode_path)

        print(f"   - vision_encoder.pt2")
        print(f"   - modality_projector.pt2")
        print(f"   - language_decoder_prefill.pt2")
        print(f"   - language_decoder_decode.pt2")

        print("\nTo convert to ExecuTorch format, install: pip install executorch")

    # Save config
    import json
    from dataclasses import asdict
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"   - config.json")
    print(f"   - embeddings.pt (saved before quantization)")


def main():
    parser = argparse.ArgumentParser(description='Export nanoVLM to ExecuTorch')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='lusxvr/nanoVLM',
        help='Model checkpoint path or HuggingFace repo'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='executorch_models',
        help='Output directory for ExecuTorch models'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Apply int8 quantization to reduce model size'
    )

    args = parser.parse_args()

    export_to_executorch(args.checkpoint, args.output_dir, quantize=args.quantize)


if __name__ == '__main__':
    main()
