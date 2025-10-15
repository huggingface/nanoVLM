import argparse
import os
import json
import torch
from models.vision_language_model import VisionLanguageModel
import models.config as config

def main():
    parser = argparse.ArgumentParser(description="Run lmms-eval on a model checkpoint.")
    parser.add_argument('--checkpoint_path', type=str, help="Path to the model checkpoint directory.")
    parser.add_argument('--global_step', type=int, help="Global step at which the checkpoint was saved.")
    parser.add_argument('--run_name', type=str, help="The name of the training run.")

    # These arguments are based on TrainConfig, passed from the eval.slurm script
    parser.add_argument('--tasks', type=str, default='mmstar,mmmu,ocrbench,textvqa', help='Tasks for lmms-eval, comma-separated.')
    parser.add_argument('--limit', type=int, default=None, help='Limit for lmms-eval.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for lmms-eval.')
    
    args = parser.parse_args()

    from evaluation import cli_evaluate
    model = VisionLanguageModel.from_pretrained(args.checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Running lmms-eval...")
    eval_args = argparse.Namespace(
        model=model,
        tasks=args.tasks,
        limit=args.limit,
        batch_size=args.batch_size,
        process_with_media=True,
        device=device,
    )
    
    eval_results = cli_evaluate(eval_args)

    output_data = {
        'global_step': args.global_step,
        'results': {}
    }

    if eval_results is not None and "results" in eval_results[0]:
        print("Processing evaluation results.")
        for task_name, task_results in eval_results[0]["results"].items():
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    key = f"{task_name}_{metric_name.split(',')[0]}"
                    output_data['results'][key] = metric_value
    else:
        print("No evaluation results to process.")

    output_dir = os.path.join('eval_results', args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    sanitized_tasks = args.tasks.replace("/", "_")
    output_path = os.path.join(output_dir, f'step_{args.global_step}_{sanitized_tasks}.json')
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Evaluation results for step {args.global_step} saved to {output_path}")

if __name__ == "__main__":
    main() 