#!/usr/bin/env python
"""
Tokens Per Second (TPS) Benchmark Tool for nanoVLM

This script measures the token generation performance of the model with different 
configurations, providing detailed metrics including TPS, layer performance, and memory usage.
"""

import argparse
import json
import os
import time
import torch
from PIL import Image

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from utils.metrics import TPSTracker, format_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TPS performance")
    parser.add_argument("--model", type=str, default="lusxvr/nanoVLM-222M", 
                        help="Model name or path")
    parser.add_argument("--prompt", type=str, default="What is this?", 
                        help="Text prompt to generate from")
    parser.add_argument("--image", type=str, default="assets/image.png", 
                        help="Path to input image")
    parser.add_argument("--max-tokens", type=str, default="20,50,100,200", 
                        help="Comma-separated list of max token counts to benchmark")
    parser.add_argument("--iterations", type=int, default=3, 
                        help="Number of iterations for each token count")
    parser.add_argument("--output", type=str, default="tps_benchmark_results.json", 
                        help="Output file to save results")
    parser.add_argument("--detailed", action="store_true", 
                        help="Show detailed metrics")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to run on (cuda, mps, cpu)")
                        
    parser.add_argument("--log-file", type=str, default=None,
                        help="Optional file to log metrics to (if not provided, logs to logs/ directory automatically)")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable automatic logging of metrics")
    parser.add_argument("--log-level", type=str, choices=["basic", "detailed"], default="detailed",
                        help="Detail level for metrics log (default: detailed)")
    
    return parser.parse_args()

def setup_device(device_arg):
    if device_arg:
        return device_arg
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def run_benchmark(model, tokens, image, token_count, iterations, detailed=False, 
                 warm_up_iterations=2, discard_warm_up_tokens=True):
    results = []
    
    if warm_up_iterations > 0:
        print(f"Running {warm_up_iterations} warm-up iterations...")
        for _ in range(warm_up_iterations):
            _ = model.generate(tokens, image, max_new_tokens=min(20, token_count))
    
    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations} with {token_count} tokens...")
        
        tracker = TPSTracker(model)
        
        gen, metrics = model.generate(
            tokens, 
            image, 
            max_new_tokens=token_count,
            return_tps_metrics=True,
            tps_tracker=tracker,
            discard_warm_up_tokens=discard_warm_up_tokens
        )
        
        results.append(metrics)
        
        if detailed:
            print(format_metrics(metrics, show_detailed=True))
        else:
            print(f"  TPS: {metrics['overall_tps']:.2f} tokens/s")
            print(f"  Input TPS: {metrics['input_tps']:.2f} tokens/s")
            print(f"  Output TPS: {metrics['output_tps']:.2f} tokens/s")
            print(f"  Time: {metrics['total_time']:.2f}s")
            print(f"  TTFT: {metrics['time_to_first_token']:.2f}s")
            print(f"  TPS stability: ±{metrics['tps_relative_stability']:.1f}% of mean")
    
    avg_metrics = {
        'token_count': token_count,
        'iterations': iterations,
        'avg_tps': sum(r['overall_tps'] for r in results) / iterations,
        'avg_time': sum(r['total_time'] for r in results) / iterations,
        'avg_ttft': sum(r['time_to_first_token'] for r in results) / iterations,
        'peak_tps': max(r['peak_tps'] for r in results),
        'min_time': min(r['total_time'] for r in results),
        'tps_stability': sum(r['tps_stability'] for r in results) / iterations,
        'detailed_results': results if detailed else None
    }
    
    return avg_metrics


def main():
    args = parse_args()
    device = setup_device(args.device)
    token_counts = [int(x) for x in args.max_tokens.split(',')]
    output_prefix = args.output.rsplit('.', 1)[0]
    
    print(f"Running TPS benchmark on device: {device}")
    print(f"Model: {args.model}")
    print(f"Token counts: {token_counts}")
    print(f"Iterations per token count: {args.iterations}")
    
    model = VisionLanguageModel.from_pretrained(args.model).to(device)
    model.eval()
    
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)
    
    template = f"Question: {args.prompt} Answer:"
    encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded_batch['input_ids'].to(device)
    
    image = Image.open(args.image)
    image = image_processor(image)
    image = image.unsqueeze(0).to(device)
    
    print("Running warmup...")
    _ = model.generate(tokens, image, max_new_tokens=5)
    
    all_results = []
    
    for token_count in token_counts:
        result = run_benchmark(
            model, tokens, image, token_count, 
            args.iterations, args.detailed
        )
        all_results.append(result)
        
        print(f"\nResults for {token_count} tokens:")
        print(f"  Average TPS: {result['avg_tps']:.2f} tokens/s")
        print(f"  Average time: {result['avg_time']:.2f}s")
        print(f"  Average TTFT: {result['avg_ttft']:.2f}s")
        print(f"  Peak TPS: {result['peak_tps']:.2f} tokens/s")
        print()
    
    with open(args.output, 'w') as f:
        json.dump({
            'model': args.model,
            'device': device,
            'prompt': args.prompt,
            'iterations': args.iterations,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'results': all_results
        }, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    
    print("\n=== Summary of Results ===")
    print(f"{'Tokens':<10} {'Avg Time (s)':<15} {'Avg TPS':<12} {'TTFT (s)':<10}")
    print("-" * 55)
    for result in all_results:
        print(f"{result['token_count']:<10} {result['avg_time']:<15.3f} {result['avg_tps']:<12.2f} {result['avg_ttft']:<10.3f}")
        
    if not args.no_log:
        try:
            from utils.logging_utils import log_metrics_to_file
            consolidated_log_path = None
            
            for i, result in enumerate(all_results):
                enriched_result = result.copy()
                enriched_result['model'] = args.model
                enriched_result['prompt'] = args.prompt
                enriched_result['device'] = device
                enriched_result['benchmark_type'] = 'tps'
                
                should_append = i > 0 and consolidated_log_path is not None
                
                log_path = log_metrics_to_file(
                    enriched_result, 
                    filepath=args.log_file if i == 0 else consolidated_log_path,
                    level=args.log_level,
                    append=should_append
                )
                
                if i == 0:
                    consolidated_log_path = log_path
            
            if consolidated_log_path:
                print(f"Metrics logged to: {consolidated_log_path}")
                
                try:
                    with open(consolidated_log_path, 'r') as f:
                        log_lines = f.readlines()
                        preview_lines = min(10, len(log_lines))
                        print("\nLog preview:")
                        print("".join(log_lines[:preview_lines]))
                        if len(log_lines) > preview_lines:
                            print(f"... (see {consolidated_log_path} for full log)")
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: Could not log results: {e}")
            

if __name__ == "__main__":
    main()
