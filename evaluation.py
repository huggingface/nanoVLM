"""
Integration of lmms-eval for intermediate evaluation during training.
"""

import os
import torch
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

from models.lmms_eval_wrapper import NanoVLMWrapper
from models.vision_language_model import VisionLanguageModel


def run_lmms_evaluation(
    model: VisionLanguageModel,
    tokenizer,
    image_processor,
    tasks: List[str],
    device: str = "cuda",
    batch_size: int = 8,
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
        if output_path and results:
            import json
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
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


def print_evaluation_results(results: Dict[str, Any]):
    """Pretty print evaluation results in table format."""
    if not results:
        return
    
    try:
        # Try to use pytablewriter if available
        from pytablewriter import MarkdownTableWriter
        use_table_writer = True
    except ImportError:
        use_table_writer = False
        logger.debug("pytablewriter not available, using simple table format")
    
    print("\n" + "="*80)
    print("LMMS-Eval Results")
    print("="*80)
    
    # Prepare table data
    if "results" in results:
        if use_table_writer:
            _print_results_with_tablewriter(results)
        else:
            _print_results_simple_table(results)
    
    # Print group results if available
    if "groups" in results:
        print("\nGroup Results:")
        if use_table_writer:
            _print_groups_with_tablewriter(results)
        else:
            _print_groups_simple_table(results)
                    
    print("="*80 + "\n")


def _print_results_with_tablewriter(results: Dict[str, Any]):
    """Print results using pytablewriter."""
    from pytablewriter import MarkdownTableWriter
    
    headers = ["Task", "Version", "Metric", "Value", "Stderr"]
    values = []
    
    for task_name, task_results in results["results"].items():
        version = results.get("versions", {}).get(task_name, "N/A")
        
        # Skip non-metric entries
        if isinstance(task_results, dict):
            for metric_key, value in sorted(task_results.items()):
                if metric_key in ["alias", "samples"] or "_stderr" in metric_key:
                    continue
                
                # Parse metric name and filter
                if "," in metric_key:
                    metric, filter_name = metric_key.split(",", 1)
                else:
                    metric = metric_key
                    filter_name = ""
                
                # Format value
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                # Get stderr if available
                stderr_key = f"{metric}_stderr,{filter_name}" if filter_name else f"{metric}_stderr"
                stderr_value = task_results.get(stderr_key, "")
                if isinstance(stderr_value, float):
                    stderr_str = f"± {stderr_value:.4f}"
                else:
                    stderr_str = ""
                
                # Display metric with filter if present
                metric_display = f"{metric} ({filter_name})" if filter_name else metric
                
                values.append([task_name, version, metric_display, value_str, stderr_str])
    
    if values:
        writer = MarkdownTableWriter()
        writer.headers = headers
        writer.value_matrix = values
        print(writer.dumps())


def _print_results_simple_table(results: Dict[str, Any]):
    """Print results using simple table format."""
    # Column widths
    col_widths = {
        "task": 20,
        "version": 10,
        "metric": 20,
        "value": 12,
        "stderr": 12
    }
    
    # Print header
    header = f"{'Task':<{col_widths['task']}} | {'Version':<{col_widths['version']}} | " \
             f"{'Metric':<{col_widths['metric']}} | {'Value':<{col_widths['value']}} | " \
             f"{'Stderr':<{col_widths['stderr']}}"
    print(header)
    print("-" * len(header))
    
    # Print data
    for task_name, task_results in results["results"].items():
        version = results.get("versions", {}).get(task_name, "N/A")
        
        if isinstance(task_results, dict):
            for metric_key, value in sorted(task_results.items()):
                if metric_key in ["alias", "samples"] or "_stderr" in metric_key:
                    continue
                
                # Parse metric name and filter
                if "," in metric_key:
                    metric, filter_name = metric_key.split(",", 1)
                else:
                    metric = metric_key
                    filter_name = ""
                
                # Format value
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                # Get stderr if available
                stderr_key = f"{metric}_stderr,{filter_name}" if filter_name else f"{metric}_stderr"
                stderr_value = task_results.get(stderr_key, "")
                if isinstance(stderr_value, float):
                    stderr_str = f"± {stderr_value:.4f}"
                else:
                    stderr_str = ""
                
                # Display metric with filter if present
                metric_display = f"{metric} ({filter_name})" if filter_name else metric
                
                # Truncate strings if too long
                task_display = task_name[:col_widths["task"]]
                version_display = str(version)[:col_widths["version"]]
                metric_display = metric_display[:col_widths["metric"]]
                
                print(f"{task_display:<{col_widths['task']}} | "
                      f"{version_display:<{col_widths['version']}} | "
                      f"{metric_display:<{col_widths['metric']}} | "
                      f"{value_str:<{col_widths['value']}} | "
                      f"{stderr_str:<{col_widths['stderr']}}")


def _print_groups_with_tablewriter(results: Dict[str, Any]):
    """Print group results using pytablewriter."""
    from pytablewriter import MarkdownTableWriter
    
    headers = ["Group", "Metric", "Value", "Stderr"]
    values = []
    
    for group_name, group_results in results["groups"].items():
        if isinstance(group_results, dict):
            for metric_key, value in sorted(group_results.items()):
                if metric_key in ["alias", "samples"] or "_stderr" in metric_key:
                    continue
                
                # Parse metric name and filter
                if "," in metric_key:
                    metric, filter_name = metric_key.split(",", 1)
                else:
                    metric = metric_key
                    filter_name = ""
                
                # Format value
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                # Get stderr if available
                stderr_key = f"{metric}_stderr,{filter_name}" if filter_name else f"{metric}_stderr"
                stderr_value = group_results.get(stderr_key, "")
                if isinstance(stderr_value, float):
                    stderr_str = f"± {stderr_value:.4f}"
                else:
                    stderr_str = ""
                
                # Display metric with filter if present
                metric_display = f"{metric} ({filter_name})" if filter_name else metric
                
                values.append([group_name, metric_display, value_str, stderr_str])
    
    if values:
        writer = MarkdownTableWriter()
        writer.headers = headers
        writer.value_matrix = values
        print(writer.dumps())


def _print_groups_simple_table(results: Dict[str, Any]):
    """Print group results using simple table format."""
    # Column widths
    col_widths = {
        "group": 25,
        "metric": 20,
        "value": 12,
        "stderr": 12
    }
    
    # Print header
    header = f"{'Group':<{col_widths['group']}} | {'Metric':<{col_widths['metric']}} | " \
             f"{'Value':<{col_widths['value']}} | {'Stderr':<{col_widths['stderr']}}"
    print(header)
    print("-" * len(header))
    
    # Print data
    for group_name, group_results in results["groups"].items():
        if isinstance(group_results, dict):
            for metric_key, value in sorted(group_results.items()):
                if metric_key in ["alias", "samples"] or "_stderr" in metric_key:
                    continue
                
                # Parse metric name and filter
                if "," in metric_key:
                    metric, filter_name = metric_key.split(",", 1)
                else:
                    metric = metric_key
                    filter_name = ""
                
                # Format value
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                # Get stderr if available
                stderr_key = f"{metric}_stderr,{filter_name}" if filter_name else f"{metric}_stderr"
                stderr_value = group_results.get(stderr_key, "")
                if isinstance(stderr_value, float):
                    stderr_str = f"± {stderr_value:.4f}"
                else:
                    stderr_str = ""
                
                # Display metric with filter if present
                metric_display = f"{metric} ({filter_name})" if filter_name else metric
                
                # Truncate strings if too long
                group_display = group_name[:col_widths["group"]]
                metric_display = metric_display[:col_widths["metric"]]
                
                print(f"{group_display:<{col_widths['group']}} | "
                      f"{metric_display:<{col_widths['metric']}} | "
                      f"{value_str:<{col_widths['value']}} | "
                      f"{stderr_str:<{col_widths['stderr']}}")


# Convenience function for common VLM benchmarks
def evaluate_vlm_benchmarks(
    model: VisionLanguageModel,
    tokenizer,
    image_processor,
    benchmarks: Optional[List[str]] = None,
    device: str = "cuda",
    batch_size: int = 8,
    output_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on common VLM benchmarks.
    
    Args:
        model: The nanoVLM model to evaluate
        tokenizer: The tokenizer
        image_processor: The image processor
        benchmarks: List of benchmarks to run. If None, runs a default set.
        device: Device to run on
        batch_size: Batch size for evaluation
        output_dir: Directory to save results
        limit: Limit examples per task (for debugging)
        
    Returns:
        Dictionary with all benchmark results
    """
    # Default set of lightweight benchmarks suitable for intermediate evaluation
    if benchmarks is None:
        benchmarks = [
            "mmstar",      # Multi-modal reasoning
            "mme",         # Multi-modal understanding
            "ai2d",        # Multi-modal K12 science questions
            "chartqa",     # Chart understanding
            "docvqa",      # Document VQA
        ]
    
    all_results = {}
    
    for benchmark in benchmarks:
        logger.info(f"Running evaluation on {benchmark}...")
        
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, f"{benchmark}_results.json")
        
        results = run_lmms_evaluation(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            tasks=[benchmark],
            device=device,
            batch_size=batch_size,
            limit=limit,
            output_path=output_path,
            log_samples=False,
            verbosity="DEBUG",
        )
        
        if results:
            all_results[benchmark] = results
            print_evaluation_results(results)
    
    return all_results