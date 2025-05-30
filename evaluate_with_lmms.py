#!/usr/bin/env python3
"""
Example script to evaluate a trained nanoVLM model using lmms-eval.

Usage:
    python evaluate_with_lmms.py --model_path lusxvr/nanoVLM-222M --tasks mmstar,mme,gqa
"""

import argparse
import torch
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from evaluation import run_lmms_evaluation, print_evaluation_results, get_available_tasks


def main():
    parser = argparse.ArgumentParser(description="Evaluate nanoVLM with lmms-eval")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="lusxvr/nanoVLM-222M",
        help="Path to the model (local or HuggingFace Hub)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmstar,mme",
        help="Comma-separated list of tasks to evaluate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for debugging)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="List all available tasks and exit"
    )
    
    args = parser.parse_args()
    
    # List available tasks if requested
    if args.list_tasks:
        print("Available lmms-eval tasks:")
        tasks = get_available_tasks()
        for task in tasks:
            print(f"  - {task}")
        return
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = VisionLanguageModel.from_pretrained(args.model_path)
    model = model.to(args.device)
    model.eval()

    # Get tokenizer and image processor
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)
    
    # Parse tasks
    tasks = [t.strip() for t in args.tasks.split(",")]
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Evaluating on tasks: {tasks}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    if args.limit:
        print(f"Limiting to {args.limit} examples per task")
    
    # Run evaluation
    results = run_lmms_evaluation(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        tasks=tasks,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        output_path=args.output_path,
        log_samples=False,
        verbosity="DEBUG", # Set to DEBUG to check the evaluation process's errors and to let it raise errors
    )
    
    # Print results
    print_evaluation_results(results)
    
    if args.output_path:
        print(f"\nDetailed results saved to: {args.output_path}")


if __name__ == "__main__":
    main()