"""
Integration of lmms-eval for intermediate evaluation during training.
"""

import os
import json
import torch
import argparse
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    from lmms_eval import evaluator
    from lmms_eval.api.instance import Instance
    from lmms_eval.loggers import EvaluationTracker
    from lmms_eval.tasks import TaskManager
    LMMS_EVAL_AVAILABLE = True
except ImportError:
    LMMS_EVAL_AVAILABLE = False
    logger.warning("lmms-eval not installed. Install with: pip install lmms-eval")

from data.processors import get_tokenizer, get_image_processor
from eval.lmms_eval_wrapper import NanoVLMWrapper
from eval.utils import print_evaluation_results
from models.vision_language_model import VisionLanguageModel


def run_lmms_evaluation(
    model: VisionLanguageModel,
    tokenizer,
    image_processor,
    tasks: List[str],
    device: str = "cuda",
    batch_size: int = 32,
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    log_samples: bool = False,
    verbosity: str = "INFO",
) -> Dict[str, Any]:
    """
    Run lmms-eval evaluation on specified tasks.
    
    Args:
        model: The nanoVLM model to evaluate
        tokenizer: The tokenizer for the model
        image_processor: The image processor for the model
        tasks: List of task names to evaluate (e.g., ["mmstar", "mme", "mathvista"])
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        num_fewshot: Number of few-shot examples (if applicable)
        limit: Limit number of examples per task (for debugging)
        output_path: Path to save evaluation results
        log_samples: Whether to log individual samples
        verbosity: Logging verbosity level
        
    Returns:
        Dictionary containing evaluation results
    """
    if not LMMS_EVAL_AVAILABLE:
        logger.error("lmms-eval is not installed. Cannot run evaluation.")
        return {}
    
    # Wrap the model for lmms-eval compatibility
    wrapped_model = NanoVLMWrapper(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        batch_size=batch_size
    )
    
    # Initialize task manager
    task_manager = TaskManager(verbosity=verbosity)
    
    # Create model args string
    model_args = f"device={device}"

    # Create a simple args object with the necessary attributes
    class SimpleArgs:
        def __init__(self):
            self.process_with_media = True  # Set to True for VLM tasks
            self.output_path = '.' #Otherwise e.g. Ocrbench will throw an error
    
    cli_args = SimpleArgs()
    
    # Run evaluation
    try:
        # Debug: check distributed_executor_backend
        logger.info(f"Using distributed_executor_backend: accelerate")
        
        results = evaluator.simple_evaluate(
            model=wrapped_model,
            model_args=model_args,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit,
            log_samples=log_samples,
            task_manager=task_manager,
            verbosity=verbosity,
            cli_args=cli_args,
            distributed_executor_backend="accelerate",
        )

        # Save results if output path is provided
        if output_path and "results" in results:
            if os.path.dirname(output_path): # This line is key
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results["results"], f, indent=2)
            logger.info(f"Evaluation results saved to {output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during lmms evaluation: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return {}


def get_available_tasks() -> List[str]:
    """Get list of available evaluation tasks from lmms-eval."""
    if not LMMS_EVAL_AVAILABLE:
        return []
    
    try:
        from lmms_eval.tasks import TaskManager
        task_manager = TaskManager()
        return sorted(task_manager.all_tasks)
    except Exception as e:
        logger.error(f"Error getting available tasks: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Evaluate nanoVLM with lmms-eval")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="lusxvr/nanoVLM-450M",
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
        default=32,
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
    if args.output_path:
        print(f"\nDetailed results saved to: {args.output_path}")
    print_evaluation_results(results)


if __name__ == "__main__":
    main()