"""
Export nanoVLM model components to ONNX format.

This script exports the nanoVLM model into separate ONNX models:
1. Vision Encoder (ViT)
2. Modality Projector
3. Language Decoder (Prefill)
4. Language Decoder (Decode with KV Cache)

Usage:
    python export_onnx.py --checkpoint lusxvr/nanoVLM-450M --output_dir onnx_models
"""

import argparse
import os
import torch
import torch.nn as nn
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig


class VisionEncoderWrapper(nn.Module):
    """Wrapper for vision encoder to ensure clean ONNX export."""
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, images):
        """
        Args:
            images: [batch_size, 3, height, width]
        Returns:
            vision_features: [batch_size, num_patches, vit_hidden_dim]
        """
        return self.vision_encoder(images)


class ModalityProjectorWrapper(nn.Module):
    """Wrapper for modality projector to ensure clean ONNX export."""
    def __init__(self, modality_projector):
        super().__init__()
        self.modality_projector = modality_projector

    def forward(self, vision_features):
        """
        Args:
            vision_features: [batch_size, num_patches, vit_hidden_dim]
        Returns:
            projected_features: [batch_size, mp_image_token_length, lm_hidden_dim]
        """
        return self.modality_projector(vision_features)


class LanguageDecoderPrefillWrapper(nn.Module):
    """Wrapper for language decoder prefill phase with ONNX-compatible attention."""
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.n_blocks = len(decoder.blocks)
        self.n_kv_heads = decoder.cfg.lm_n_kv_heads
        self.head_dim = decoder.cfg.lm_hidden_dim // decoder.cfg.lm_n_heads

        # Patch attention blocks to use ONNX-compatible attention
        # PyTorch ONNX exporter doesn't support is_causal=True with attn_mask
        # So we need to combine causal and padding masks
        self._patch_attention_for_onnx()

    def _patch_attention_for_onnx(self):
        """
        Monkey-patch F.scaled_dot_product_attention to combine causal and padding masks.
        PyTorch ONNX exporter doesn't support is_causal=True with attn_mask.
        """
        import torch.nn.functional as F
        original_sdpa = F.scaled_dot_product_attention

        def onnx_compatible_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            """
            Wrapper that combines causal mask into attn_mask for ONNX export compatibility.
            """
            if is_causal and attn_mask is not None:
                # Need to combine causal mask with attention mask
                # because ONNX exporter doesn't support both at once
                batch, n_heads, seq_len, _ = query.shape

                # Create causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=query.device, dtype=query.dtype) * torch.finfo(query.dtype).min,
                    diagonal=1
                ).view(1, 1, seq_len, seq_len)

                # Combine with existing mask
                combined_mask = attn_mask + causal_mask

                # Call SDPA with combined mask and is_causal=False
                return original_sdpa(query, key, value, attn_mask=combined_mask, dropout_p=dropout_p, is_causal=False, **kwargs)
            else:
                # Normal call
                return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

        # Replace F.scaled_dot_product_attention with our wrapper
        F.scaled_dot_product_attention = onnx_compatible_sdpa

    def forward(self, embeddings, attention_mask):
        """
        Prefill phase: process full sequence and return outputs + KV cache.

        Args:
            embeddings: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] - 1 for valid tokens, 0 for padding
        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim]
            A flattened tuple of KV cache tensors (k0, v0, k1, v1, ..., kN, vN)
        """
        hidden_states, kv_cache_list = self.decoder(
            embeddings,
            attention_mask=attention_mask,
            kv_cache=None,
            start_pos=0
        )

        # Flatten KV cache for ONNX export
        kv_outputs = []
        for block_cache in kv_cache_list:
            kv_outputs.append(block_cache['key'])
            kv_outputs.append(block_cache['value'])

        return (hidden_states,) + tuple(kv_outputs)


class LanguageDecoderDecodeWrapper(nn.Module):
    """Wrapper for language decoder decode phase with KV cache."""
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.n_blocks = len(decoder.blocks)
        self.n_kv_heads = decoder.cfg.lm_n_kv_heads
        self.head_dim = decoder.cfg.lm_hidden_dim // decoder.cfg.lm_n_heads

        # Patch SDPA for ONNX compatibility
        self._patch_attention_for_onnx()

    def _patch_attention_for_onnx(self):
        """
        Monkey-patch F.scaled_dot_product_attention to combine causal and padding masks.
        PyTorch ONNX exporter doesn't support is_causal=True with attn_mask.
        """
        import torch.nn.functional as F
        original_sdpa = F.scaled_dot_product_attention

        def onnx_compatible_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            """
            Wrapper that combines causal mask into attn_mask for ONNX export compatibility.
            """
            if is_causal and attn_mask is not None:
                batch, n_heads, seq_len, _ = query.shape

                # Create causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=query.device, dtype=query.dtype) * torch.finfo(query.dtype).min,
                    diagonal=1
                ).view(1, 1, seq_len, seq_len)

                # Combine with existing mask
                combined_mask = attn_mask + causal_mask

                # Call SDPA with combined mask and is_causal=False
                return original_sdpa(query, key, value, attn_mask=combined_mask, dropout_p=dropout_p, is_causal=False, **kwargs)
            else:
                # Normal call
                return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

        # Replace F.scaled_dot_product_attention with our wrapper
        F.scaled_dot_product_attention = onnx_compatible_sdpa

    def forward(self, embeddings, attention_mask, start_pos, *kv_cache_flat):
        """
        Decode phase: process single token with KV cache.

        Args:
            embeddings: [batch_size, 1, hidden_dim] - single token embedding
            attention_mask: [batch_size, total_seq_len] - attention mask for all tokens so far
            start_pos: [1] - scalar tensor indicating position of current token
            *kv_cache_flat: Flattened KV cache (k0, v0, k1, v1, ..., kN, vN)
                Each k/v has shape [batch_size, n_kv_heads, past_seq_len, head_dim]
        Returns:
            hidden_states: [batch_size, 1, hidden_dim]
            Updated KV cache (k0, v0, k1, v1, ..., kN, vN)
        """
        # Reconstruct KV cache from flattened inputs
        kv_cache_list = []
        for i in range(0, len(kv_cache_flat), 2):
            kv_cache_list.append({
                'key': kv_cache_flat[i],
                'value': kv_cache_flat[i + 1]
            })

        # Extract scalar start position
        start_pos_int = start_pos.item() if isinstance(start_pos, torch.Tensor) else start_pos

        hidden_states, updated_kv_cache = self.decoder(
            embeddings,
            attention_mask=attention_mask,
            kv_cache=kv_cache_list,
            start_pos=start_pos_int
        )

        # Flatten updated KV cache for output
        kv_outputs = []
        for block_cache in updated_kv_cache:
            kv_outputs.append(block_cache['key'])
            kv_outputs.append(block_cache['value'])

        return (hidden_states,) + tuple(kv_outputs)


def export_vision_encoder(vlm_model, output_dir, opset_version=17):
    """Export vision encoder to ONNX using modern dynamo_export."""
    print("Exporting vision encoder...")

    vision_encoder_wrapper = VisionEncoderWrapper(vlm_model.vision_encoder)
    vision_encoder_wrapper.eval()

    # Get config for dummy input
    cfg = vlm_model.cfg
    batch_size = 1

    # Create dummy input: [batch_size, 3, img_size, img_size]
    dummy_image = torch.randn(batch_size, 3, cfg.vit_img_size, cfg.vit_img_size)

    output_path = os.path.join(output_dir, "vision_encoder.onnx")

    # Use modern ONNX export (dynamo-based)
    torch.onnx.export(
        vision_encoder_wrapper,
        (dummy_image,),
        output_path,
        dynamo=True,
        opset_version=opset_version
    )

    print(f"Vision encoder exported to {output_path}")
    return output_path


def export_modality_projector(vlm_model, output_dir, opset_version=17):
    """Export modality projector to ONNX using modern dynamo_export."""
    print("Exporting modality projector...")

    mp_wrapper = ModalityProjectorWrapper(vlm_model.MP)
    mp_wrapper.eval()

    cfg = vlm_model.cfg
    batch_size = 1
    num_patches = (cfg.vit_img_size // cfg.vit_patch_size) ** 2

    # Create dummy input: [batch_size, num_patches, vit_hidden_dim]
    dummy_vision_features = torch.randn(batch_size, num_patches, cfg.vit_hidden_dim)

    output_path = os.path.join(output_dir, "modality_projector.onnx")

    # Use modern ONNX export (dynamo-based)
    torch.onnx.export(
        mp_wrapper,
        (dummy_vision_features,),
        output_path,
        dynamo=True,
        opset_version=opset_version
    )

    print(f"Modality projector exported to {output_path}")
    return output_path


def export_language_decoder_prefill(vlm_model, output_dir, opset_version=17):
    """Export language decoder prefill phase to ONNX using modern dynamo_export."""
    print("Exporting language decoder (prefill)...")

    decoder_wrapper = LanguageDecoderPrefillWrapper(vlm_model.decoder)
    decoder_wrapper.eval()

    cfg = vlm_model.cfg
    batch_size = 1
    seq_len = 128  # Example sequence length

    # Create dummy inputs
    dummy_embeddings = torch.randn(batch_size, seq_len, cfg.lm_hidden_dim)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    output_path = os.path.join(output_dir, "language_decoder_prefill.onnx")

    # Use modern ONNX export (dynamo-based)
    torch.onnx.export(
        decoder_wrapper,
        (dummy_embeddings, dummy_attention_mask),
        output_path,
        dynamo=True,
        opset_version=opset_version
    )

    print(f"Language decoder (prefill) exported to {output_path}")
    return output_path


def export_language_decoder_decode(vlm_model, output_dir, opset_version=17):
    """Export language decoder decode phase to ONNX using modern dynamo_export."""
    print("Exporting language decoder (decode)...")

    decoder_wrapper = LanguageDecoderDecodeWrapper(vlm_model.decoder)
    decoder_wrapper.eval()

    cfg = vlm_model.cfg
    batch_size = 1
    past_seq_len = 128  # Example past sequence length
    n_kv_heads = cfg.lm_n_kv_heads
    head_dim = cfg.lm_hidden_dim // cfg.lm_n_heads
    n_blocks = cfg.lm_n_blocks

    # Create dummy inputs
    dummy_embeddings = torch.randn(batch_size, 1, cfg.lm_hidden_dim)  # Single token
    dummy_attention_mask = torch.ones(batch_size, past_seq_len + 1, dtype=torch.long)
    dummy_start_pos = torch.tensor([past_seq_len], dtype=torch.long)

    # Create dummy KV cache
    dummy_kv_cache = []
    for _ in range(n_blocks):
        dummy_kv_cache.append(torch.randn(batch_size, n_kv_heads, past_seq_len, head_dim))  # key
        dummy_kv_cache.append(torch.randn(batch_size, n_kv_heads, past_seq_len, head_dim))  # value

    output_path = os.path.join(output_dir, "language_decoder_decode.onnx")

    # Use modern ONNX export (dynamo-based)
    torch.onnx.export(
        decoder_wrapper,
        (dummy_embeddings, dummy_attention_mask, dummy_start_pos, *dummy_kv_cache),
        output_path,
        dynamo=True,
        opset_version=opset_version
    )

    print(f"Language decoder (decode) exported to {output_path}")
    return output_path


def export_all(checkpoint_path, output_dir, opset_version=17):
    """Export all nanoVLM components to ONNX."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    print(f"Loading model from {checkpoint_path}...")
    vlm_model = VisionLanguageModel.from_pretrained(checkpoint_path)
    vlm_model.eval()

    # Export each component
    with torch.no_grad():
        vision_encoder_path = export_vision_encoder(vlm_model, output_dir, opset_version)
        mp_path = export_modality_projector(vlm_model, output_dir, opset_version)
        prefill_path = export_language_decoder_prefill(vlm_model, output_dir, opset_version)
        decode_path = export_language_decoder_decode(vlm_model, output_dir, opset_version)

    # Save config for inference
    import json
    from dataclasses import asdict
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(asdict(vlm_model.cfg), f, indent=2)
    print(f"Config saved to {config_path}")

    print("\nExport complete! ONNX models saved to:", output_dir)
    print("  - vision_encoder.onnx")
    print("  - modality_projector.onnx")
    print("  - language_decoder_prefill.onnx")
    print("  - language_decoder_decode.onnx")
    print("  - config.json")

    # Force models to use opset 23 for better ONNX Runtime compatibility
    # The dynamo exporter produces opset 18, but ONNX Runtime 1.23+ works better with opset 23
    print("\nUpdating models to opset 23 for ONNX Runtime compatibility...")
    import onnx
    for model_path in [vision_encoder_path, mp_path, prefill_path, decode_path]:
        model = onnx.load(model_path)
        for opset in model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                opset.version = 23
        onnx.save(model, model_path)
    print("✅ Models updated to opset 23")

    # Fix Attention operator outputs for ONNX Runtime compatibility
    # PyTorch's exporter creates Attention nodes with unused KV cache outputs
    # which ONNX Runtime 1.23.0 doesn't handle correctly
    print("\nFixing Attention operator outputs...")
    for model_path in [vision_encoder_path, mp_path, prefill_path, decode_path]:
        model = onnx.load(model_path)
        graph = model.graph

        # Find all Attention nodes and remove unused outputs
        attention_nodes = [n for n in graph.node if n.op_type == 'Attention']
        if attention_nodes:
            # Build set of used tensors
            used_tensors = set()
            for node in graph.node:
                for input_name in node.input:
                    if input_name:
                        used_tensors.add(input_name)
            for output in graph.output:
                used_tensors.add(output.name)

            # Remove unused outputs from Attention nodes
            for node in attention_nodes:
                if len(node.output) > 1:
                    original_outputs = list(node.output)
                    used_outputs = [original_outputs[0]]  # Keep attention output
                    # Keep additional outputs only if they're used
                    for output_name in original_outputs[1:]:
                        if output_name in used_tensors:
                            used_outputs.append(output_name)
                    if len(used_outputs) < len(original_outputs):
                        del node.output[:]
                        node.output.extend(used_outputs)

            onnx.save(model, model_path)
    print("✅ Attention operators fixed")

    return {
        'vision_encoder': vision_encoder_path,
        'modality_projector': mp_path,
        'language_decoder_prefill': prefill_path,
        'language_decoder_decode': decode_path,
        'config': config_path,
    }


def main():
    parser = argparse.ArgumentParser(description='Export nanoVLM to ONNX')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='lusxvr/nanoVLM-450M',
        help='Path to model checkpoint or HuggingFace model ID'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='onnx_models',
        help='Directory to save ONNX models'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=24,
        help='ONNX opset version (24 or higher recommended)'
    )

    args = parser.parse_args()

    export_all(args.checkpoint, args.output_dir, args.opset_version)


if __name__ == '__main__':
    main()
