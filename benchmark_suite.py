import torch
import time
import argparse
from PIL import Image

from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
from data.processors import get_tokenizer, get_image_processor

# Ensure reproducibility
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def benchmark_vlm(
    vit_model_type: str,
    lm_model_type: str,
    lm_tokenizer_path: str,
    mp_pixel_shuffle_factor: int,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    num_runs: int,
    warmup_runs: int,
    device: torch.device,
):
    """
    Benchmarks a VLM configuration and prints detailed timing information.
    """

    print(f"\n--- Benchmarking Configuration ---")
    print(f"ViT Model: {vit_model_type}")
    print(f"LLM Model: {lm_model_type}")
    print(f"LLM Tokenizer: {lm_tokenizer_path}")
    print(f"Pixel Shuffle Factor: {mp_pixel_shuffle_factor}")
    print(f"Image: {image_path}")
    print(f"Prompt: '{prompt}'")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Device: {device}")
    print(f"Warmup Runs: {warmup_runs}, Benchmark Runs: {num_runs}")
    print("-----------------------------------\n")

    if device.type == 'cuda':
        torch.cuda.synchronize() # Synchronize before resetting stats
        torch.cuda.reset_peak_memory_stats(device) # Reset peak for this entire inference run      

    # 1. Configuration and Model Loading
    cfg = VLMConfig(
        vit_model_type=vit_model_type,
        lm_model_type=lm_model_type,
        lm_tokenizer=lm_tokenizer_path,
        mp_pixel_shuffle_factor=mp_pixel_shuffle_factor,
        vlm_load_backbone_weights=True # Always load from hub for this script
    )
    
    model = VisionLanguageModel(cfg, load_backbone=True).to(device).eval()
    tokenizer = get_tokenizer(cfg.lm_tokenizer)
    image_processor = get_image_processor(cfg.vit_img_size)

    initial_vram_model_mb = 0
    if device.type == 'cuda':
        torch.cuda.synchronize() # Ensure model is fully loaded
        initial_vram_model_bytes = torch.cuda.memory_allocated(device)
        initial_vram_model_mb = initial_vram_model_bytes / (1024 ** 2)
        print(f"Initial VRAM allocated for model: {initial_vram_model_mb:.2f} MB")
    else:
        print("VRAM measurement for model loading skipped (not on CUDA device).")

    # 2. Prepare Inputs
    template = f"Question: {prompt} Answer:"
    encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
    input_ids = encoded_batch['input_ids'].to(device)
    attention_mask = encoded_batch['attention_mask'].to(device)

    pil_image = Image.open(image_path)
    image_tensor = image_processor(pil_image).unsqueeze(0).to(device)

    # --- Warmup Runs ---
    for _ in range(warmup_runs):
        # Vision Encoding part
        image_embd_warmup = model.vision_encoder(image_tensor)
        image_embd_warmup = model.MP(image_embd_warmup)

        # LLM part (simplified for warmup)
        image_embd_warmup = model.vision_encoder(image_tensor)
        image_embd_warmup = model.MP(image_embd_warmup)
        token_embd_warmup = model.decoder.token_embedding(input_ids)
        combined_embd_warmup = torch.cat((image_embd_warmup, token_embd_warmup), dim=1)
        
        current_attention_mask_warmup = None
        if attention_mask is not None:
            img_seq_len_warmup = image_embd_warmup.size(1)
            image_attention_mask_warmup = torch.ones((image_embd_warmup.size(0), img_seq_len_warmup), device=attention_mask.device, dtype=attention_mask.dtype)
            current_attention_mask_warmup = torch.cat((image_attention_mask_warmup, attention_mask), dim=1)

        outputs_warmup = combined_embd_warmup
        for _ in range(max_new_tokens):
            model_out_warmup = model.decoder(outputs_warmup, current_attention_mask_warmup)
            last_token_logits_warmup = model_out_warmup[:, -1, :]
            if not model.decoder.lm_use_tokens:
                last_token_logits_warmup = model.decoder.head(last_token_logits_warmup)
            probs_warmup = torch.softmax(last_token_logits_warmup, dim=-1)
            next_token_warmup = torch.multinomial(probs_warmup, num_samples=1)
            next_embd_warmup = model.decoder.token_embedding(next_token_warmup)
            outputs_warmup = torch.cat((outputs_warmup, next_embd_warmup), dim=1)
            if current_attention_mask_warmup is not None:
                current_attention_mask_warmup = torch.cat((current_attention_mask_warmup, torch.ones((image_embd_warmup.size(0), 1), device=attention_mask.device)), dim=1)
    if torch.cuda.is_available():
            torch.cuda.synchronize()


    # --- Benchmark Runs ---
    vision_encoding_times = []
    time_to_first_token_times = []
    llm_processing_times = [] # Time for all subsequent tokens
    generated_tokens_counts = []
    peak_vram_inference_mb_list = []

    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize() # Synchronize before resetting stats
            torch.cuda.reset_peak_memory_stats(device) # Reset peak for this entire inference run
        
        # 3. Vision Encoding
        start_vision_encoding = time.perf_counter()
        image_embd = model.vision_encoder(image_tensor)
        image_embd = model.MP(image_embd)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_vision_encoding = time.perf_counter()
        vision_encoding_time = end_vision_encoding - start_vision_encoding
        vision_encoding_times.append(vision_encoding_time)

        # 4. LLM Processing (Token by Token for detailed timing)
        token_embd = model.decoder.token_embedding(input_ids)
        combined_embd = torch.cat((image_embd, token_embd), dim=1)

        batch_size = image_embd.size(0)
        img_seq_len = image_embd.size(1)
        
        current_attention_mask = None
        if attention_mask is not None:
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            current_attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)
        
        outputs = combined_embd
        # generated_tokens_for_run = torch.zeros((batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_first_token_llm = time.perf_counter() # Includes first token LLM step

        # First token
        model_out = model.decoder(outputs, current_attention_mask)
        last_token_logits = model_out[:, -1, :]
        if not model.decoder.lm_use_tokens:
            last_token_logits = model.decoder.head(last_token_logits)
        
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        # generated_tokens_for_run[:, 0] = next_token.squeeze(-1)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_first_token_llm = time.perf_counter()
        
        time_to_first_token = (end_vision_encoding - start_vision_encoding) + (end_first_token_llm - start_first_token_llm)
        time_to_first_token_times.append(time_to_first_token)

        # Subsequent tokens
        start_llm_subsequent = time.perf_counter()
        next_embd = model.decoder.token_embedding(next_token)
        outputs = torch.cat((outputs, next_embd), dim=1)
        if current_attention_mask is not None:
            current_attention_mask = torch.cat((current_attention_mask, torch.ones((batch_size, 1), device=current_attention_mask.device)), dim=1)

        for i in range(1, max_new_tokens):
            model_out = model.decoder(outputs, current_attention_mask)
            last_token_logits = model_out[:, -1, :]
            if not model.decoder.lm_use_tokens:
                last_token_logits = model.decoder.head(last_token_logits)

            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # generated_tokens_for_run[:, i] = next_token.squeeze(-1)
            
            next_embd = model.decoder.token_embedding(next_token)
            outputs = torch.cat((outputs, next_embd), dim=1)

            if current_attention_mask is not None:
                current_attention_mask = torch.cat((current_attention_mask, torch.ones((batch_size, 1), device=current_attention_mask.device)), dim=1)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_llm_subsequent = time.perf_counter()
        
        llm_processing_time_run = (end_first_token_llm - start_first_token_llm) + (end_llm_subsequent - start_llm_subsequent)
        llm_processing_times.append(llm_processing_time_run)
        generated_tokens_counts.append(max_new_tokens)

        if device.type == 'cuda':
            torch.cuda.synchronize() # Ensure all ops are done before measuring peak VRAM
            current_peak_vram_inference_bytes = torch.cuda.max_memory_allocated(device)
            peak_vram_inference_mb_list.append(current_peak_vram_inference_bytes / (1024 ** 2))
        else:
            peak_vram_inference_mb_list.append(0) # Append 0 if not on CUDA


    # 5. Calculate and Print Averages
    avg_vision_encoding_time = sum(vision_encoding_times) / num_runs
    avg_time_to_first_token = sum(time_to_first_token_times) / num_runs
    avg_llm_processing_time = sum(llm_processing_times) / num_runs
    
    # Tokens per second for tokens *after* the first one
    if max_new_tokens > 1:
        avg_subsequent_tokens_time = avg_llm_processing_time - (avg_time_to_first_token - avg_vision_encoding_time) # Subtract first token LLM time
        avg_tokens_per_sec_after_first = (max_new_tokens - 1) / avg_subsequent_tokens_time if avg_subsequent_tokens_time > 0 else float('inf')
    else:
        avg_tokens_per_sec_after_first = float('nan') # Not applicable if only one token is generated

    avg_peak_vram_inference_mb = 0
    if device.type == 'cuda' and peak_vram_inference_mb_list:
        avg_peak_vram_inference_mb = sum(peak_vram_inference_mb_list) / len(peak_vram_inference_mb_list)

    print(f"--- Results (averaged over {num_runs} runs) ---")
    if device.type == 'cuda':
        print(f"Initial VRAM for Model: {initial_vram_model_mb:.2f} MB") # Printed once, but good to have in summary
        print(f"Average Peak VRAM during Inference: {avg_peak_vram_inference_mb:.2f} MB")
    print(f"Average Vision Encoding Time: {avg_vision_encoding_time:.4f} seconds")
    print(f"Average Time to First Token: {avg_time_to_first_token:.4f} seconds")
    print(f"Average LLM Processing Time (for {max_new_tokens} tokens): {avg_llm_processing_time:.4f} seconds")
    if max_new_tokens > 1:
        print(f"Average Tokens/Second (after first): {avg_tokens_per_sec_after_first:.2f}")
    print("--------------------------------------\n")

    # Cleanup
    del model
    del tokenizer
    del image_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "avg_vision_encoding_time": avg_vision_encoding_time,
        "avg_time_to_first_token": avg_time_to_first_token,
        "avg_llm_processing_time": avg_llm_processing_time,
        "avg_tokens_per_sec_after_first": avg_tokens_per_sec_after_first if max_new_tokens > 1 else None,
        "initial_vram_model_mb": initial_vram_model_mb if device.type == 'cuda' else 0,
        "avg_peak_vram_inference_mb": avg_peak_vram_inference_mb if device.type == 'cuda' else 0,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark VLM inference speed with detailed timings.")
    
    # Model Config Args
    parser.add_argument("--vit_model_types", type=str, nargs='+', default=["google/siglip-base-patch16-224"], help="List of ViT model identifiers from Hugging Face Hub.")
    parser.add_argument("--lm_model_type", type=str, default="HuggingFaceTB/SmolLM2-135M", help="LLM model identifier from Hugging Face Hub.")
    parser.add_argument("--lm_tokenizer", type=str, default="HuggingFaceTB/cosmo2-tokenizer", help="LLM tokenizer identifier from Hugging Face Hub.")
    parser.add_argument("--mp_pixel_shuffle_factors", type=int, nargs='+', default=[2], help="List of pixel shuffle factors for the modality projector.")

    # Input Args
    parser.add_argument("--image_path", type=str, default="assets/image.png", help="Path to the input image.")
    parser.add_argument("--prompt", type=str, default="What is in this image?", help="Prompt for the VLM.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Number of new tokens to generate.")

    # Benchmark Args
    parser.add_argument("--num_runs", type=int, default=10, help="Number of times to run the benchmark.")
    parser.add_argument("--warmup_runs", type=int, default=3, help="Number of warmup runs before benchmarking.")
    
    args = parser.parse_args()

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {current_device}")

    # Default single benchmark run (can be removed if only suite is needed)
    # benchmark_vlm(
    #     vit_model_type=args.vit_model_types[0], # Default to first if not running suite
    #     lm_model_type=args.lm_model_type,
    #     lm_tokenizer_path=args.lm_tokenizer,
    #     mp_pixel_shuffle_factor=args.mp_pixel_shuffle_factors[0], # Default to first
    #     image_path=args.image_path,
    #     prompt=args.prompt,
    #     max_new_tokens=args.max_new_tokens,
    #     num_runs=args.num_runs,
    #     warmup_runs=args.warmup_runs,
    #     device=current_device,
    # )

    # Example of how to run multiple configurations:
    print("\n\n--- Running a suite of benchmarks ---")
    
    import itertools

    vit_model_types_to_test = args.vit_model_types
    mp_pixel_shuffle_factors_to_test = args.mp_pixel_shuffle_factors
    # You can also define these lists directly in the code if you prefer not to use args for them:
    vit_model_types_to_test = ["google/siglip-base-patch16-512", "google/siglip-base-patch16-256"]
    mp_pixel_shuffle_factors_to_test = [1]#, 2, 4]
    lm_model_types_to_test = [ "HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M"]

    # Common settings from args
    common_tokenizer = args.lm_tokenizer
    
    # Generate all combinations
    all_combinations = list(itertools.product(vit_model_types_to_test, mp_pixel_shuffle_factors_to_test, lm_model_types_to_test))

    for vit_model_type_combo, mp_pixel_shuffle_factor_combo, lm_model_type_combo in all_combinations:
        print(f"\n--- Preparing to benchmark combination ---")
        print(f"ViT: {vit_model_type_combo}, Shuffle Factor: {mp_pixel_shuffle_factor_combo}, LLM: {lm_model_type_combo}")
        benchmark_vlm(
            vit_model_type=vit_model_type_combo,
            lm_model_type=lm_model_type_combo,
            lm_tokenizer_path=common_tokenizer,
            mp_pixel_shuffle_factor=mp_pixel_shuffle_factor_combo,
            image_path=args.image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
            device=current_device,
        ) 