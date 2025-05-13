import torch; torch.manual_seed(0);
from PIL import Image
import time

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from utils.metrics import TPSTracker, format_metrics

from torch.utils import benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_tokens(tokens, image):
    gen = model.generate(tokens, image, max_new_tokens=100)
    return gen

def generate_tokens_with_metrics(tokens, image, model, max_new_tokens=100):
    """Generate tokens with performance metrics"""
    return model.generate(tokens, image, max_new_tokens=max_new_tokens, 
                         return_tps_metrics=True, tps_tracker=None)

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
    image = Image.open(image_path)
    image = image_processor(image)
    image = image.unsqueeze(0).to(device)

    # First: Standard PyTorch benchmarking for overall performance
    time_benchmark = benchmark.Timer(
        stmt="generate_tokens(tokens, image)",
        setup='from __main__ import generate_tokens',
        globals={"tokens": tokens, "image": image},
        num_threads=torch.get_num_threads(),
    )

    print("=== PyTorch Benchmark Results (Overall Performance) ===")
    print(time_benchmark.timeit(10))
    
    print("\n=== Detailed Token Generation Performance ===")
    print("Running detailed TPS benchmark...")
    token_lengths = [20, 50, 100, 200]
    
    results = []
    for max_tokens in token_lengths:
        print(f"\nBenchmarking with {max_tokens} tokens...")
        _ = model.generate(tokens, image, max_new_tokens=1)
        _, metrics = generate_tokens_with_metrics(tokens, image, model, max_tokens)
        
        results.append({
            'tokens': max_tokens,
            'time': metrics['total_time'],
            'tps': metrics['overall_tps'],
            'ttft': metrics['time_to_first_token']
        })
        
        print(format_metrics(metrics))
    
    print("\n=== Summary of Results ===")
    print(f"{'Tokens':<10} {'Time (s)':<12} {'TPS':<10} {'TTFT (s)':<10}")
    print("-" * 42)
    for result in results:
        print(f"{result['tokens']:<10} {result['time']:<12.3f} {result['tps']:<10.2f} {result['ttft']:<10.3f}")
