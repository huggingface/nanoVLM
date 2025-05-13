#!/usr/bin/env python
"""
Text generation script for nanoVLM.

This script generates text from a given prompt and image using the nanoVLM model.
It can optionally display performance metrics for the generation process.
"""

import argparse
import torch
import time
from PIL import Image

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from utils.metrics import TPSTracker, format_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from nanoVLM model")
    parser.add_argument("--prompt", type=str, default="What is this?", 
                      help="Text prompt to generate from")
    parser.add_argument("--image", type=str, default="assets/image.png", 
                      help="Path to input image")
    parser.add_argument("--num-generations", type=int, default=5,
                      help="Number of generations to run")
    parser.add_argument("--max-tokens", type=int, default=100,
                      help="Maximum tokens to generate")
    parser.add_argument("--metrics", action="store_true",
                      help="Display detailed performance metrics")
    parser.add_argument("--detailed-metrics", action="store_true",
                      help="Show detailed layer breakdown in metrics")
    parser.add_argument("--warmup", action="store_true",
                      help="Run warmup iterations before generation")
    parser.add_argument("--model", type=str, default="lusxvr/nanoVLM-222M",
                      help="Model name or path")
    
    parser.add_argument("--log-file", type=str, default=None,
                      help="Optional file to log metrics to (if not provided, logs to logs/ directory automatically)")
    parser.add_argument("--no-log", action="store_true",
                      help="Disable automatic logging of metrics")
    parser.add_argument("--log-level", type=str, choices=["basic", "detailed"], default="detailed",
                      help="Detail level for metrics log (default: detailed)")
    
                      
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(0)
    
    all_runs_log_path = None

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    model = VisionLanguageModel.from_pretrained(args.model).to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    template = f"Question: {args.prompt} Answer:"
    encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded_batch['input_ids'].to(device)

    try:
        image = Image.open(args.image)
        image = image_processor(image)
        image = image.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("\nInput: ")
    print(f'{args.prompt}')
    print("\nOutput:")

    if args.warmup:
        print("Running warm-up...")
        _ = model.generate(tokens, image, max_new_tokens=5)

    for i in range(args.num_generations):
        if args.metrics:
            tracker = TPSTracker(model)
            
            gen, metrics = model.generate(
                tokens, 
                image, 
                max_new_tokens=args.max_tokens,
                return_tps_metrics=True,
                tps_tracker=tracker,
                warm_up_iterations=2 if args.warmup else 0,
                discard_warm_up_tokens=True
            )
            
            print(f"\nGeneration {i+1}: {tokenizer.batch_decode(gen, skip_special_tokens=True)[0]}")
            print(format_metrics(metrics, show_detailed=args.detailed_metrics))
            print("\n" + "-"*50)
            
            if not args.no_log:
                try:
                    from utils.logging_utils import log_metrics_to_file
                    enriched_metrics = metrics.copy()
                    enriched_metrics['model'] = args.model
                    enriched_metrics['prompt'] = args.prompt
                    enriched_metrics['max_tokens'] = args.max_tokens
                    enriched_metrics['generation_number'] = i + 1
                    
                    should_append = i > 0 and all_runs_log_path is not None
                    
                    log_path = log_metrics_to_file(
                        enriched_metrics, 
                        filepath=args.log_file if i == 0 else all_runs_log_path,
                        level=args.log_level,
                        append=should_append
                    )
                    
                    if i == 0:
                        all_runs_log_path = log_path
                        print(f"Metrics logged to: {log_path}")
                        
                        try:
                            with open(log_path, 'r') as f:
                                log_lines = f.readlines()
                                preview_lines = min(8, len(log_lines))
                                print("\nLog preview:")
                                print("".join(log_lines[:preview_lines]))
                                if len(log_lines) > preview_lines:
                                    print(f"... (see {log_path} for full log)")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Warning: Could not log metrics to file: {e}")
                    
        else:
            start_time = time.time()
            gen = model.generate(
                tokens, 
                image, 
                max_new_tokens=args.max_tokens
            )
            elapsed = time.time() - start_time
            
            print(f"\nGeneration {i+1}: {tokenizer.batch_decode(gen, skip_special_tokens=True)[0]}")
            print(f"Generation time: {elapsed:.2f}s")
            print("-"*50)

if __name__ == "__main__":
    main()
