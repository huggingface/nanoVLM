"""
Test that ExecuTorch exported models produce the same outputs as the original PyTorch model.
"""
import torch
import os
import numpy as np
from models.vision_language_model import VisionLanguageModel


def compare_tensors(t1, t2, name, rtol=1e-3, atol=1e-5):
    """Compare two tensors and return True if they're close."""
    if t1.shape != t2.shape:
        print(f"   ❌ {name}: Shape mismatch! {t1.shape} vs {t2.shape}")
        return False

    max_diff = torch.max(torch.abs(t1 - t2)).item()
    mean_diff = torch.mean(torch.abs(t1 - t2)).item()

    is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)

    if is_close:
        print(f"   ✅ {name}: MATCH (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
    else:
        print(f"   ❌ {name}: MISMATCH (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
        print(f"      Original range: [{t1.min():.3f}, {t1.max():.3f}]")
        print(f"      Exported range: [{t2.min():.3f}, {t2.max():.3f}]")

    return is_close


def test_accuracy(checkpoint, exported_dir, quantized=False):
    """Test that exported models match original PyTorch model."""

    print(f"\n{'='*70}")
    print(f"Testing {'QUANTIZED' if quantized else 'UNQUANTIZED'} export accuracy")
    print(f"{'='*70}")

    # Load original PyTorch model
    print("\nLoading original PyTorch model...")
    original_model = VisionLanguageModel.from_pretrained(checkpoint)
    original_model.eval()
    print("✅ Original model loaded")

    # Load exported models
    print("\nLoading exported models...")
    vision_prog = torch.export.load(os.path.join(exported_dir, "vision_encoder.pt2"))
    vision_exported = vision_prog.module()

    proj_prog = torch.export.load(os.path.join(exported_dir, "modality_projector.pt2"))
    proj_exported = proj_prog.module()

    prefill_prog = torch.export.load(os.path.join(exported_dir, "language_decoder_prefill.pt2"))
    prefill_exported = prefill_prog.module()

    decode_prog = torch.export.load(os.path.join(exported_dir, "language_decoder_decode.pt2"))
    decode_exported = decode_prog.module()

    print("✅ Exported models loaded")

    cfg = original_model.cfg
    all_match = True

    # Test 1: Vision Encoder
    print(f"\n{'─'*70}")
    print("TEST 1: Vision Encoder")
    print(f"{'─'*70}")

    test_image = torch.randn(1, 3, cfg.vit_img_size, cfg.vit_img_size)

    with torch.no_grad():
        vision_orig = original_model.vision_encoder(test_image)
        vision_exp = vision_exported(test_image)

    match = compare_tensors(vision_orig, vision_exp, "Vision features", rtol=1e-2 if quantized else 1e-3)
    all_match = all_match and match

    # Test 2: Modality Projector
    print(f"\n{'─'*70}")
    print("TEST 2: Modality Projector")
    print(f"{'─'*70}")

    with torch.no_grad():
        proj_orig = original_model.MP(vision_orig)
        proj_exp = proj_exported(vision_exp)

    match = compare_tensors(proj_orig, proj_exp, "Projected embeddings", rtol=1e-2 if quantized else 1e-3)
    all_match = all_match and match

    # Test 3: Language Decoder Prefill
    print(f"\n{'─'*70}")
    print("TEST 3: Language Decoder - Prefill")
    print(f"{'─'*70}")

    # Create test embeddings (seq_len=128 to match export)
    seq_len = 128
    test_embeddings = torch.randn(1, seq_len, cfg.lm_hidden_dim)
    test_mask = torch.ones(1, seq_len, dtype=torch.long)
    test_pos = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        # Original model
        hidden_orig, kv_cache_orig = original_model.decoder(
            test_embeddings,
            attention_mask=test_mask,
            kv_cache=None,
            position_ids=test_pos
        )

        # Exported model
        hidden_exp, kv_cache_exp = prefill_exported(
            test_embeddings,
            test_mask,
            test_pos
        )

    match = compare_tensors(hidden_orig, hidden_exp, "Hidden states", rtol=1e-2 if quantized else 1e-3)
    all_match = all_match and match

    # Compare KV cache
    print(f"\n   Comparing KV cache ({len(kv_cache_orig)} blocks):")
    kv_matches = 0
    for i in range(min(3, len(kv_cache_orig))):  # Check first 3 blocks
        k_match = compare_tensors(
            kv_cache_orig[i]['key'],
            kv_cache_exp[i]['key'],
            f"Block {i} keys",
            rtol=1e-2 if quantized else 1e-3
        )
        v_match = compare_tensors(
            kv_cache_orig[i]['value'],
            kv_cache_exp[i]['value'],
            f"Block {i} values",
            rtol=1e-2 if quantized else 1e-3
        )
        if k_match and v_match:
            kv_matches += 1

    if kv_matches == 3:
        print(f"   ✅ KV cache matches (checked 3/{len(kv_cache_orig)} blocks)")
    else:
        print(f"   ⚠️  KV cache partial match ({kv_matches}/3 blocks)")
        all_match = False

    # Test 4: Language Decoder Decode
    print(f"\n{'─'*70}")
    print("TEST 4: Language Decoder - Decode")
    print(f"{'─'*70}")

    next_emb = torch.randn(1, 1, cfg.lm_hidden_dim)
    decode_mask = torch.ones(1, seq_len + 1, dtype=torch.long)
    decode_pos = torch.tensor([[seq_len]], dtype=torch.long)

    with torch.no_grad():
        # Original model
        hidden_decode_orig, kv_cache_decode_orig = original_model.decoder(
            next_emb,
            attention_mask=decode_mask,
            kv_cache=kv_cache_orig,
            position_ids=decode_pos
        )

        # Exported model
        hidden_decode_exp, kv_cache_decode_exp = decode_exported(
            next_emb,
            decode_mask,
            decode_pos,
            kv_cache_exp
        )

    match = compare_tensors(
        hidden_decode_orig,
        hidden_decode_exp,
        "Decode hidden states",
        rtol=1e-2 if quantized else 1e-3
    )
    all_match = all_match and match

    # Final result
    print(f"\n{'='*70}")
    if all_match:
        print("✅ ALL TESTS PASSED - Exported models match original!")
    else:
        print("⚠️  SOME TESTS FAILED - Check differences above")
        if quantized:
            print("   Note: Small differences are expected with quantization")
    print(f"{'='*70}\n")

    return all_match


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test ExecuTorch export accuracy')
    parser.add_argument('--checkpoint', default='lusxvr/nanoVLM', help='Original checkpoint')
    parser.add_argument('--unquantized_dir', default='executorch_models', help='Unquantized models')
    parser.add_argument('--quantized_dir', default='executorch_models_quantized', help='Quantized models')
    parser.add_argument('--skip_unquantized', action='store_true', help='Skip unquantized test')
    parser.add_argument('--skip_quantized', action='store_true', help='Skip quantized test')

    args = parser.parse_args()

    results = []

    if not args.skip_unquantized and os.path.exists(args.unquantized_dir):
        result = test_accuracy(args.checkpoint, args.unquantized_dir, quantized=False)
        results.append(('Unquantized', result))

    if not args.skip_quantized and os.path.exists(args.quantized_dir):
        result = test_accuracy(args.checkpoint, args.quantized_dir, quantized=True)
        results.append(('Quantized', result))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✅ PASS" if passed else "⚠️  FAIL"
        print(f"{name:20s}: {status}")
    print("="*70)


if __name__ == '__main__':
    main()
