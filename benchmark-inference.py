import torch; torch.manual_seed(0);
from PIL import Image
import time

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M").to(device)
    model.eval()
    
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    text = "What is this?"
    template = f"Question: {text} Answer:"
    encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded_batch['input_ids'].to(device)

    image_path = 'assets/image.png'
    pil_image = Image.open(image_path)
    image_tensor = image_processor(pil_image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    max_new_tokens_to_generate = 100
    num_benchmark_runs = 10

    all_timings = []

    print(f"Running benchmark for {num_benchmark_runs} iterations...")
    for i in range(num_benchmark_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        _generated_output, current_timings = model.generate(
            tokens, 
            image_tensor, 
            max_new_tokens=max_new_tokens_to_generate
        )
        
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        current_timings['benchmark_run_total_time'] = end_time - start_time
        all_timings.append(current_timings)
        print(f"Run {i+1}/{num_benchmark_runs} completed.")

    avg_total_gen_time = sum(t['total_generation_time'] for t in all_timings) / num_benchmark_runs
    avg_benchmark_run_total_time = sum(t['benchmark_run_total_time'] for t in all_timings) / num_benchmark_runs
    avg_ttft = sum(t['ttft'] for t in all_timings if t['ttft'] is not None) / len([t for t in all_timings if t['ttft'] is not None])
    avg_vision_time = sum(t['vision_encoder_time'] for t in all_timings) / num_benchmark_runs
    avg_lm_time = sum(t['language_model_time'] for t in all_timings) / num_benchmark_runs
    
   
    total_actual_tokens_generated = sum(t.get('num_generated_tokens', 0) for t in all_timings)
    total_model_internal_generation_time = sum(t.get('total_generation_time', 0) for t in all_timings)

    tps = total_actual_tokens_generated / total_model_internal_generation_time if total_model_internal_generation_time > 0 else 0
    
    print("\n--- Inference Performance Metrics ---")
    print(f"Number of benchmark runs: {num_benchmark_runs}")
    print(f"Requested max new tokens per run: {max_new_tokens_to_generate}")

   
    if num_benchmark_runs > 0 and all('num_generated_tokens' in t for t in all_timings):
        avg_actual_tokens_per_run = total_actual_tokens_generated / num_benchmark_runs
        
        if abs(avg_actual_tokens_per_run - max_new_tokens_to_generate) > 1e-5 or \
           avg_actual_tokens_per_run == max_new_tokens_to_generate: # Show if different or same for clarity
             print(f"Average actual tokens generated per run (from model timings): {avg_actual_tokens_per_run:.2f}")

    print(f"Average total inference time (model internal): {avg_total_gen_time:.4f} seconds")
    print(f"Average total benchmark run time (end-to-end): {avg_benchmark_run_total_time:.4f} seconds")
    print(f"Tokens Per Second (TPS, based on model internal time and actual tokens): {tps:.2f}")
    print(f"Time to First Token (TTFT) / Prefill Time: {avg_ttft:.4f} seconds")
    print("Fine-Grained VLM Speed Analysis (average per run):")
    print(f"  Time in Vision Encoder: {avg_vision_time:.4f} seconds")
    print(f"  Time in Language Model (generation loop): {avg_lm_time:.4f} seconds")
