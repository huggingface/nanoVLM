import argparse
import os
import json
import torch
import torch.distributed as dist
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from models.vision_language_model import VisionLanguageModel

from torch.nn.parallel import DistributedDataParallel

def init_dist():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def destroy_dist():
    dist.destroy_process_group()

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_master():
    return dist.get_rank() == 0 if is_dist() else True

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all

def wrap_model(model):
    return DistributedDataParallel(model, device_ids=[dist.get_rank()])

def run_evaluation(checkpoint_path, global_step, tasks, limit, batch_size):
    from evaluation import cli_evaluate
    model = VisionLanguageModel.from_pretrained(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Running lmms-eval...")
    eval_args = argparse.Namespace(
        model=model,
        tasks=tasks,
        limit=limit,
        batch_size=batch_size,
        process_with_media=True,
        device=device,
    )
    
    eval_results = cli_evaluate(eval_args)

    dist.barrier()  # Ensure all processes finish before proceeding

    if is_master():
        output_data = {
            'global_step': global_step,
            'results': {}
        }

        if eval_results and "results" in eval_results[0]:
            print("Processing evaluation results.")
            for task_name, task_results in eval_results[0]["results"].items():
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        key = f"{task_name}_{metric_name.split(',')[0]}"
                        output_data['results'][key] = metric_value
        else:
            print("No evaluation results to process.")

        return output_data


def discover_checkpoints(checkpoints_dir: str) -> Dict[str, List[int]]:
    """
    Discover all checkpoint steps in a directory.
    
    Args:
        checkpoints_dir: Path to checkpoints directory
        
    Returns:
        Dict mapping run_name to list of step numbers
    """
    checkpoints_path = Path(checkpoints_dir)
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    run_steps = {}
    run_name = checkpoints_path.name
    
    # Find all step_* subdirectories
    step_dirs = [d for d in checkpoints_path.iterdir() if d.is_dir() and d.name.startswith('step_')]
    steps = []
    
    for step_dir in step_dirs:
        try:
            step_num = int(step_dir.name.split('_')[1])
            steps.append(step_num)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse step number from {step_dir.name}")
            continue
    
    if steps:
        run_steps[run_name] = sorted(steps)
    
    return run_steps


def get_existing_eval_results(eval_results_dir: str, run_name: str) -> Dict[int, Dict[str, Set[str]]]:
    """
    Get existing evaluation results for a run.
    
    Args:
        eval_results_dir: Path to eval_results directory
        run_name: Name of the training run
        
    Returns:
        Dict mapping step numbers to dict of tasks and their metrics
    """
    eval_path = Path(eval_results_dir) / run_name
    existing_results = {}
    
    if not eval_path.exists():
        return existing_results
    
    # Find all step_*.json files
    result_files = eval_path.glob('step_*.json')
    
    for result_file in result_files:
        try:
            step_num = int(result_file.stem.split('_')[1])
            
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            if 'results' in data:
                # Extract task names from metric keys
                tasks_metrics = {}
                for key in data['results'].keys():
                    # Keys are typically like "mmmu_val_mmmu_acc", "textvqa_val_exact_match"
                    task_name = key.split('_')[0]  # First part is usually the task
                    if task_name not in tasks_metrics:
                        tasks_metrics[task_name] = set()
                    tasks_metrics[task_name].add(key)
                
                existing_results[step_num] = tasks_metrics
                
        except (ValueError, IndexError, json.JSONError) as e:
            print(f"Warning: Could not parse {result_file}: {e}")
            continue
    
    return existing_results


def identify_missing_evaluations(
    run_steps: Dict[str, List[int]], 
    existing_results: Dict[int, Dict[str, Set[str]]], 
    tasks: List[str],
    specific_steps: Optional[List[int]] = None
) -> List[Tuple[int, str]]:
    """
    Identify which evaluations are missing.
    
    Args:
        run_steps: Dict of run_name to step numbers
        existing_results: Existing evaluation results
        tasks: List of task names to evaluate
        specific_steps: Optional list of specific steps to evaluate
        
    Returns:
        List of (step_number, missing_tasks_string) tuples
    """
    missing_evaluations = []
    tasks_list = tasks.split(",")
    
    for _, steps in run_steps.items():
        for step in steps:
            # Skip if specific_steps provided and this step not in it
            if specific_steps is not None and step not in specific_steps:
                continue
                
            missing_tasks = []
            
            if step not in existing_results:
                # No results exist for this step at all
                missing_tasks = tasks.copy()
            else:
                # Check which tasks are missing
                existing_tasks = set(existing_results[step].keys())
                for task in tasks_list:
                    if task not in existing_tasks:
                        missing_tasks.append(task)

            if missing_tasks:
                missing_evaluations.append((step, ",".join(missing_tasks)))

    return missing_evaluations


def save_evaluation_results(
    eval_results_dir: str, 
    run_name: str, 
    step: int, 
    new_results: Dict
) -> None:
    """
    Save evaluation results to JSON file, merging with existing if present.
    
    Args:
        eval_results_dir: Path to eval_results directory
        run_name: Name of the training run
        step: Step number
        new_results: New evaluation results to save
    """
    eval_path = Path(eval_results_dir) / run_name
    eval_path.mkdir(parents=True, exist_ok=True)
    
    result_file = eval_path / f"step_{step}.json"
    
    # Load existing results if they exist
    if result_file.exists():
        with open(result_file, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {
            'global_step': step,
            'results': {}
        }
    
    # Merge new results with existing
    existing_data['results'].update(new_results['results'])
    existing_data['global_step'] = step
    
    # Save updated results
    with open(result_file, 'w') as f:
        json.dump(existing_data, f, indent=4)


def orchestrate_evaluations(
    checkpoints_dir: str,
    tasks: str,
    eval_results_dir: str = "eval_results",
    specific_steps: Optional[List[int]] = None,
    limit: Optional[int] = None,
    batch_size: int = 128
) -> None:
    """
    Main orchestration function for running evaluations.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory for a specific run
        tasks: List of evaluation tasks to run
        eval_results_dir: Base directory for evaluation results
        specific_steps: Optional list of specific steps to evaluate
        limit: Optional limit for number of examples per task
        batch_size: Batch size for evaluation
    """
    if is_master():
        print(f"Starting evaluation orchestration for: {checkpoints_dir}")
        print(f"Tasks to evaluate: {tasks}")
        if specific_steps:
            print(f"Specific steps: {specific_steps}")
        
        # 1. Discover available checkpoints
        print("\n1. Discovering checkpoints...")
        run_steps = discover_checkpoints(checkpoints_dir)
        
        if not run_steps:
            print("No checkpoint steps found!")
            missing_evaluations = []
        else:
            run_name = list(run_steps.keys())[0]
            steps = run_steps[run_name]
            print(f"Found {len(steps)} checkpoint steps for {run_name}: {steps}")
            
            # 2. Check existing evaluation results
            print("\n2. Checking existing evaluation results...")
            existing_results = get_existing_eval_results(eval_results_dir, run_name)
            print(f"Found existing results for {len(existing_results)} steps")
            
            # 3. Identify missing evaluations
            print("\n3. Identifying missing evaluations...")
            missing_evaluations = identify_missing_evaluations(
                run_steps, existing_results, tasks, specific_steps
            )
            
            if not missing_evaluations:
                print("No missing evaluations found! All requested evaluations are complete.")
            else:
                print(f"Found {len(missing_evaluations)} missing evaluations:")
                for step, missing_tasks in missing_evaluations:
                    print(f"  Step {step}: {missing_tasks}")
    else:
        missing_evaluations = None

    # Broadcast missing_evaluations from master to all processes
    if is_dist():
        object_list = [missing_evaluations]
        dist.broadcast_object_list(object_list, src=0)
        missing_evaluations = object_list[0]
    
    if not missing_evaluations:
        return

    # 4. Run missing evaluations
    print(f"\n4. Running missing evaluations...")
    for i, (step, missing_tasks) in enumerate(missing_evaluations, 1):
        print(f"\nRunning evaluation {i}/{len(missing_evaluations)}: Step {step}, Tasks: {missing_tasks}")
        
        checkpoint_path = os.path.join(checkpoints_dir, f"step_{step}")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint path does not exist: {checkpoint_path}")
            continue
        
        try:
            # Run evaluation for missing tasks
            results = run_evaluation(checkpoint_path, step, missing_tasks, limit, batch_size)
            
            # Save results
            if is_master():
                save_evaluation_results(eval_results_dir, run_name, step, results)
            print(f"✓ Completed evaluation for step {step}")
            dist.barrier()

        except Exception as e:
            print(f"✗ Failed evaluation for step {step}: {e}")
            continue

    if is_master():
        print(f"\n✓ Evaluation orchestration complete!")


def main():
    parser = argparse.ArgumentParser(description="Orchestrate checkpoint evaluations")
    parser.add_argument("--checkpoints_dir", required=True, help="Path to checkpoints directory")
    parser.add_argument("--eval_tasks", type=str, required=True, help="List of evaluation tasks")
    parser.add_argument("--eval_results_dir", default="eval_results", help="Directory for evaluation results")
    parser.add_argument("--steps", nargs="*", type=int, help="Specific steps to evaluate")
    parser.add_argument("--limit", type=int, help="Limit number of examples per task")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    
    args = parser.parse_args()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
    
    orchestrate_evaluations(
        checkpoints_dir=args.checkpoints_dir,
        tasks=args.eval_tasks,
        eval_results_dir=args.eval_results_dir,
        specific_steps=args.steps,
        limit=args.limit,
        batch_size=args.batch_size
    )

    if is_dist():
        destroy_dist()

if __name__ == "__main__":
    main()