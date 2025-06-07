from loguru import logger
from typing import Dict, Any

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