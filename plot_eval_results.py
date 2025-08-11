#!/usr/bin/env python3
import json
import os
import sys
import glob
import matplotlib.pyplot as plt

def load_eval_results(eval_folder):
    """Load all JSON files from the evaluation folder and extract results."""
    json_files = glob.glob(os.path.join(eval_folder, "step_*.json"))
    
    if not json_files:
        print(f"No JSON files found in {eval_folder}")
        return None
    
    results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            step = data.get('global_step', 0)
            metrics = data.get('results', {})
            
            result = {'step': step}
            result.update(metrics)
            results.append(result)
    
    # Sort by step
    results.sort(key=lambda x: x['step'])
    return results

def get_legend_name(eval_folder):
    """Extract legend name from folder path - last two numbers after underscore."""
    folder_name = os.path.basename(eval_folder)
    return folder_name.split('_')[-1]

def plot_results(all_results, eval_folders):
    """Plot the evaluation results for multiple folders."""
    if not all_results:
        return
    
    # Extract all metric names from all results
    metric_names = set()
    for results in all_results:
        for result in results:
            metric_names.update(k for k in result.keys() if k != 'step')
    
    metric_names = sorted(list(metric_names))
    
    # Create subplots
    n_metrics = len(metric_names)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    _, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Define colors for different runs
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, metric in enumerate(metric_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot each run
        for j, (results, eval_folder) in enumerate(zip(all_results, eval_folders)):
            # Extract values for this metric
            values = []
            metric_steps = []
            for result in results:
                if metric in result:
                    values.append(result[metric])
                    metric_steps.append(result['step'])
            
            if values:
                legend_name = get_legend_name(eval_folder)
                color = colors[j % len(colors)]
                ax.plot(metric_steps, values, marker='o', markersize=2, 
                       color=color, label=legend_name)
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        if len(eval_folders) > 1:
            ax.legend()
    
    # Hide unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Create assets folder if it doesn't exist
    assets_folder = '/fsx/luis_wiedmann/nanoVLM/assets'
    os.makedirs(assets_folder, exist_ok=True)
    
    # Save the plot to assets folder
    if len(eval_folders) == 1:
        folder_name = os.path.basename(eval_folders[0])
        output_file = os.path.join(assets_folder, f'{folder_name}_evaluation_plots.png')
    else:
        output_file = os.path.join(assets_folder, 'comparison_evaluation_plots.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_eval_results.py <eval_folder1> [eval_folder2] [eval_folder3] ...")
        sys.exit(1)
    
    eval_folders = sys.argv[1:]
    
    # Check if all folders exist
    for eval_folder in eval_folders:
        if not os.path.exists(eval_folder):
            print(f"Error: Folder {eval_folder} does not exist")
            sys.exit(1)
    
    # Load results from all folders
    all_results = []
    for eval_folder in eval_folders:
        print(f"Loading evaluation results from: {eval_folder}")
        results = load_eval_results(eval_folder)
        if results:
            print(f"Found {len(results)} evaluation steps")
            all_results.append(results)
        else:
            print(f"No evaluation results found in {eval_folder}")
            all_results.append([])
    
    if any(all_results):
        plot_results(all_results, eval_folders)
    else:
        print("No evaluation results found in any folder")

if __name__ == "__main__":
    main()