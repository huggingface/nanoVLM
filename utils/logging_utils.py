#!/usr/bin/env python
"""
Utilities for logging TPS metrics to files.

This module provides functions for logging TPS metrics to CSV or JSONL files,
without disrupting the main control flow of the application.
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Tuple

def flatten_metrics(metrics: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
    flattened = {}
    
    for key, value in metrics.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        
        if isinstance(value, dict):
            if key == 'layer_metrics' and 'detailed' in value:
                if 'grouped' in value:
                    for group_key, group_val in value['grouped'].items():
                        group_flat_key = f"{new_key}.grouped.{group_key}"
                        if isinstance(group_val, dict):
                            for k, v in group_val.items():
                                flattened[f"{group_flat_key}.{k}"] = v
                        else:
                            flattened[group_flat_key] = group_val
                if 'total_time' in value:
                    flattened[f"{new_key}.total_time"] = value['total_time']
            else:
                flattened.update(flatten_metrics(value, new_key))
        elif isinstance(value, list):
            if key not in ['per_second_tps', 'per_token_tps', 'token_timestamps']:
                flattened[new_key] = str(value)
        elif key == 'timestamp' and isinstance(value, (int, float)):
            flattened[new_key] = datetime.fromtimestamp(value).isoformat()
        else:
            flattened[new_key] = value
            
    return flattened

def get_log_filename(metrics: Dict[str, Any]) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_name = "unknown_model"
    if 'model' in metrics:
        model_parts = metrics['model'].split('/')
        model_name = model_parts[-1]
    
    token_info = ""
    if 'token_count' in metrics:
        token_info = f"_{metrics['token_count']}tokens"
    elif 'max_tokens' in metrics:
        token_info = f"_{metrics['max_tokens']}tokens"
    elif 'total_tokens' in metrics:
        token_info = f"_{metrics['total_tokens']}tokens"
    
    benchmark_type = ""
    if 'benchmark_type' in metrics:
        benchmark_type = f"_{metrics['benchmark_type']}"
    
    return f"logs/tps_metrics_{model_name}{token_info}{benchmark_type}_{timestamp}.log"


def format_summary_log(metrics: Dict[str, Any], level: str = 'basic') -> str:
    timestamp_str = metrics.get('timestamp', datetime.now().isoformat())
    if isinstance(timestamp_str, str):
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            formatted_time = timestamp_str
    else:
        formatted_time = str(timestamp_str)
    
    model_name = "unknown model"
    if 'model' in metrics:
        model_parts = metrics['model'].split('/')
        model_name = model_parts[-1]
    
    device = metrics.get('device', 'unknown')
    if not device and 'hostname' in metrics:
        device = metrics.get('hostname', 'unknown')
    
    summary = []
    summary.append("=" * 50)
    summary.append(f"GENERATION SUMMARY")
    summary.append("=" * 50)
    summary.append(f"Model: {model_name}")
    summary.append(f"Date: {formatted_time}")
    summary.append(f"Device: {device}")
    summary.append("")
    
    summary.append("-" * 20 + " PERFORMANCE METRICS " + "-" * 20)
    
    total_tokens = metrics.get('total_tokens', 0)
    input_tokens = metrics.get('input_tokens', 0)
    output_tokens = metrics.get('output_tokens', 0)
    
    if total_tokens:
        summary.append(f"Total tokens generated: {total_tokens}")
        if input_tokens:
            summary.append(f"  - Input tokens: {input_tokens}")
        if output_tokens:
            summary.append(f"  - Output tokens: {output_tokens}")
    
    total_time = metrics.get('total_time', 0)
    if total_time:
        summary.append(f"Total generation time: {total_time:.2f}s")
    
    overall_tps = metrics.get('overall_tps', 0)
    if overall_tps:
        summary.append(f"Overall tokens per second: {overall_tps:.2f} TPS")
        
        input_tps = metrics.get('input_tps', 0)
        output_tps = metrics.get('output_tps', 0)
        if input_tps:
            summary.append(f"  - Input TPS: {input_tps:.2f} tokens/s")
        if output_tps:
            summary.append(f"  - Output TPS: {output_tps:.2f} tokens/s")
    
    ttft = metrics.get('time_to_first_token', 0)
    if ttft:
        summary.append(f"Time to first token (TTFT): {ttft:.2f}s")
    
    peak_tps = metrics.get('peak_tps', 0)
    if peak_tps:
        summary.append(f"Peak TPS: {peak_tps:.2f} tokens/s")
    
    tps_stability = metrics.get('tps_stability', 0)
    tps_relative_stability = metrics.get('tps_relative_stability', 0)
    if tps_stability or tps_relative_stability:
        summary.append("TPS stability:")
        if tps_stability:
            summary.append(f"  - Absolute: ±{tps_stability:.2f} TPS")
        if tps_relative_stability:
            summary.append(f"  - Relative: ±{tps_relative_stability:.1f}% of mean TPS")
    
    memory = metrics.get('memory', {})
    if memory:
        summary.append("\nMemory usage:")
        peak_memory = memory.get('peak_memory_mb', 0)
        if peak_memory:
            summary.append(f"- Peak: {peak_memory:.2f} MB")
        memory_increase = memory.get('memory_increase_mb', 0)
        if memory_increase:
            summary.append(f"- Increase: {memory_increase:.2f} MB")
        memory_per_token = memory.get('memory_per_token_mb', 0)
        if memory_per_token:
            summary.append(f"- Per token: {memory_per_token:.2f} MB")
    
    layer_metrics = metrics.get('layer_metrics', {})
    grouped = layer_metrics.get('grouped', {})
    
    if grouped:
        summary.append("\nLayer performance:")
        for layer_name, layer_data in grouped.items():
            if isinstance(layer_data, dict):
                time_value = layer_data.get('total_time', 0)
                percentage = layer_data.get('percentage', 0)
                summary.append(f"- {layer_name}: {percentage:.1f}% of time ({time_value:.2f}s)")
    
    if level == 'detailed' and 'layer_metrics' in metrics and 'detailed' in layer_metrics:
        detailed = layer_metrics['detailed']
        if detailed:
            sorted_layers = sorted(
                [(k, v) for k, v in detailed.items() if isinstance(v, dict) and 'total_time' in v],
                key=lambda x: x[1]['total_time'] if 'total_time' in x[1] else 0,
                reverse=True
            )
            
            top_layers = sorted_layers[:10]
            if top_layers:
                summary.append("\nDetailed layer breakdown (top 10):")
                for layer_name, layer_data in top_layers:
                    if isinstance(layer_data, dict):
                        time_value = layer_data.get('total_time', 0)
                        percentage = layer_data.get('percentage', 0)
                        calls = layer_data.get('calls', 0)
                        summary.append(f"- {layer_name}: {percentage:.1f}% ({time_value:.2f}s, {calls} calls)")
    
    per_second_tps = metrics.get('per_second_tps', [])
    if per_second_tps and len(per_second_tps) > 0:
        summary.append("\nTPS per second:")
        for i, tps in enumerate(per_second_tps):
            summary.append(f"- {i}-{i+1}s: {tps:.1f}")
    
    return "\n".join(summary)

def log_to_file(metrics: Dict[str, Any], filepath: Optional[str] = None, level: str = 'detailed', append: bool = False) -> str:
    summary_text = format_summary_log(metrics, level)
    
    if filepath is None:
        filepath = get_log_filename(metrics)
    
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    mode = 'a' if append else 'w'
    with open(filepath, mode, encoding='utf-8') as f:
        f.write(summary_text)
        if append:
            f.write("\n\n" + "="*80 + "\n\n")  
    
    return filepath

def log_metrics_to_file(metrics: Dict[str, Any], filepath: Optional[str] = None, level: str = 'detailed', append: bool = False) -> str:
    enriched_metrics = metrics.copy()
    enriched_metrics['timestamp'] = datetime.now().isoformat()
    enriched_metrics['hostname'] = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    
    return log_to_file(enriched_metrics, filepath, level, append)
