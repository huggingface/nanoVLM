"""
Test ExecuTorch exported models by generating a description of a cat image.
"""
import argparse
import torch
from PIL import Image
from transformers import AutoTokenizer
import json
import os


def load_exported_models(model_dir):
    """Load all exported .pt2 models."""
    print(f"Loading models from {model_dir}...")

    vision_prog = torch.export.load(os.path.join(model_dir, "vision_encoder.pt2"))
    projection_prog = torch.export.load(os.path.join(model_dir, "modality_projector.pt2"))
    prefill_prog = torch.export.load(os.path.join(model_dir, "language_decoder_prefill.pt2"))
    decode_prog = torch.export.load(os.path.join(model_dir, "language_decoder_decode.pt2"))

    vision_module = vision_prog.module()
    projection_module = projection_prog.module()
    prefill_module = prefill_prog.module()
    decode_module = decode_prog.module()

    # Load config
    with open(os.path.join(model_dir, "config.json"), 'r') as f:
        config = json.load(f)

    # Load embeddings
    embeddings_path = os.path.join(model_dir, "embeddings.pt")
    if os.path.exists(embeddings_path):
        embeddings = torch.load(embeddings_path, map_location='cpu')
    else:
        embeddings = None

    print("✅ Models loaded successfully")
    return vision_module, projection_module, prefill_module, decode_module, config, embeddings


def preprocess_image(image_path, config):
    """Preprocess image using the actual image processor (handles splitting)."""
    from data.processors import get_image_processor

    image = Image.open(image_path).convert('RGB')

    # Get the image processor from config
    resize_to_max_side_len = config.get('resize_to_max_side_len', False)
    image_processor = get_image_processor(
        config['max_img_size'],
        config['vit_img_size'],
        resize_to_max_side_len
    )

    # Process image (returns list of image tensors + split ratio)
    processed_images, splitted_ratio = image_processor(image)

    return processed_images, splitted_ratio


def generate_description(
    vision_module,
    projection_module,
    prefill_module,
    decode_module,
    config,
    embeddings,
    image_path,
    prompt="Describe this image in detail.",
    max_new_tokens=50
):
    """Generate image description using exported models."""

    print(f"\nProcessing image: {image_path}")
    print(f"Prompt: {prompt}")

    # 1. Preprocess image (splits into grid if needed)
    processed_images, splitted_ratio = preprocess_image(image_path, config)
    print(f"Number of images: {len(processed_images)} (grid: {splitted_ratio})")

    # 2. Vision encoding for all images
    print("Running vision encoder on all images...")
    all_image_embeddings = []
    for img_tensor in processed_images:
        img_batch = img_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            vision_features = vision_module(img_batch)
            image_emb = projection_module(vision_features)
        all_image_embeddings.append(image_emb)

    # Concatenate all image embeddings [num_images, 64, hidden_dim]
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    print(f"All image embeddings shape: {all_image_embeddings.shape}")

    # 4. Prepare text with image tokens (global + 64 regular)
    from data.processors import get_tokenizer, get_image_string

    tokenizer = get_tokenizer(config['lm_tokenizer'], config['vlm_extra_tokens'], config['lm_chat_template'])

    # Create image string with correct grid ratio
    image_string = get_image_string(tokenizer, [splitted_ratio], config['mp_image_token_length'])

    # Format with chat template
    messages = [{'role': 'user', 'content': image_string + prompt}]
    formatted_prompt = tokenizer.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)[0]

    # Tokenize
    tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    print(f"Input tokens: {len(tokens)}")

    # 5. Get token embeddings and replace image token
    if embeddings is None:
        print("⚠️  No embeddings.pt found, cannot continue")
        return None

    token_embedding_weight = embeddings['token_embedding']
    lm_head_weight = embeddings['lm_head']

    # Find image token IDs
    image_token = config['vlm_extra_tokens']['image_token']
    global_image_token = config['vlm_extra_tokens']['global_image_token']
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    global_image_token_id = tokenizer.convert_tokens_to_ids(global_image_token)

    # Create embeddings
    text_embeddings = torch.nn.functional.embedding(input_ids, token_embedding_weight)

    # Flatten all_image_embeddings to [batch, total_image_tokens, hidden_dim]
    # Shape: [num_images, 64, hidden_dim] -> [1, num_images * 64, hidden_dim]
    image_embeddings_flat = all_image_embeddings.reshape(1, -1, all_image_embeddings.shape[-1])
    print(f"Flattened image embeddings shape: {image_embeddings_flat.shape}")

    # Build final embedding sequence, replacing image tokens
    combined_embeddings = []
    image_emb_idx = 0
    for i in range(input_ids.shape[1]):
        token_id = input_ids[0, i].item()
        if token_id == image_token_id or token_id == global_image_token_id:
            # Replace this image token with the corresponding image embedding
            if image_emb_idx < image_embeddings_flat.shape[1]:
                combined_embeddings.append(image_embeddings_flat[0, image_emb_idx:image_emb_idx+1])
                image_emb_idx += 1
            else:
                print(f"⚠️  Warning: More image tokens than embeddings!")
                combined_embeddings.append(text_embeddings[0, i:i+1])
        else:
            # Keep text embedding
            combined_embeddings.append(text_embeddings[0, i:i+1])

    combined_embeddings = torch.cat(combined_embeddings, dim=0).unsqueeze(0)
    seq_len = combined_embeddings.shape[1]
    print(f"Combined embeddings shape: {combined_embeddings.shape}")

    # Create attention mask and position IDs (no padding needed with dynamic shapes)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)

    # 7. Prefill phase
    print("Running prefill...")
    with torch.no_grad():
        outputs = prefill_module(
            combined_embeddings,
            attention_mask,
            position_ids
        )

    # Check output structure
    if isinstance(outputs, tuple) and len(outputs) == 2:
        hidden_states, kv_cache = outputs
        print(f"Prefill outputs: hidden_states {hidden_states.shape}, kv_cache (list of {len(kv_cache)} blocks)")
    else:
        print(f"⚠️  Unexpected output format: {type(outputs)}")
        hidden_states = outputs if not isinstance(outputs, tuple) else outputs[0]
        kv_cache = None

    # Get logits for last token
    last_hidden = hidden_states[:, -1:, :]
    logits = torch.matmul(last_hidden, lm_head_weight.T)
    next_token_id = torch.argmax(logits, dim=-1)

    generated_ids = [next_token_id.item()]
    print(f"Generated tokens: ", end='', flush=True)

    # Check if we can continue with decode
    if kv_cache is None:
        print("\n⚠️  KV cache not available, cannot decode")
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 8. Decode phase
    for step in range(max_new_tokens - 1):
        # Get embedding for next token
        # next_token_id is [1, 1], so embedding will be [1, 1, hidden_dim]
        next_embedding = torch.nn.functional.embedding(
            next_token_id,
            token_embedding_weight
        )

        # Update attention mask and position IDs
        current_pos = seq_len + step
        current_seq_len = current_pos + 1
        decode_attention_mask = torch.ones(1, current_seq_len, dtype=torch.long)
        decode_position_ids = torch.tensor([[current_pos]], dtype=torch.long)

        # Run decode
        with torch.no_grad():
            decode_outputs = decode_module(
                next_embedding,
                decode_attention_mask,
                decode_position_ids,
                kv_cache
            )

        # Update KV cache
        if isinstance(decode_outputs, tuple) and len(decode_outputs) == 2:
            hidden_states, kv_cache = decode_outputs
        else:
            print(f"\n⚠️  Unexpected decode output format")
            break

        # Get next token
        last_hidden = hidden_states[:, -1:, :]
        logits = torch.matmul(last_hidden, lm_head_weight.T)
        next_token_id = torch.argmax(logits, dim=-1)

        token_id = next_token_id.item()
        generated_ids.append(token_id)

        # Print token
        token_str = tokenizer.decode([token_id])
        print(token_str, end='', flush=True)

        # Check for EOS
        if token_id == tokenizer.eos_token_id:
            break

    print()  # Newline

    # Decode all generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Test ExecuTorch export with cat image')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='executorch_models',
        help='Directory containing exported models'
    )
    parser.add_argument(
        '--image',
        type=str,
        default='assets/demo.png',
        help='Path to image file'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='Describe this image in detail.',
        help='Text prompt'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help='Maximum tokens to generate'
    )

    args = parser.parse_args()

    # Load models
    vision_module, projection_module, prefill_module, decode_module, config, embeddings = load_exported_models(args.model_dir)

    # Generate description
    result = generate_description(
        vision_module,
        projection_module,
        prefill_module,
        decode_module,
        config,
        embeddings,
        args.image,
        args.prompt,
        args.max_new_tokens
    )

    if result:
        print(f"\n{'='*60}")
        print(f"Generated description:")
        print(f"{'='*60}")
        print(result)
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
