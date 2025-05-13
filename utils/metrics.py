import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any

class LayerPerformanceTracker:
    def __init__(self, model):
        self.model = model
        self.layer_times = {}
        self.handles = []
        self.active = False
        
    def _register_hooks(self):
        def create_hooks(name):
            layer_time = {'start': None, 'total': 0.0, 'calls': 0}
            self.layer_times[name] = layer_time
            
            def pre_hook(module, input):
                if self.active:
                    layer_time['start'] = time.perf_counter()
                return None
                
            def post_hook(module, input, output):
                if self.active and layer_time['start'] is not None:
                    end_time = time.perf_counter()
                    duration = end_time - layer_time['start']
                    layer_time['total'] += duration
                    layer_time['calls'] += 1
                    layer_time['start'] = None
                return None
                
            return pre_hook, post_hook
        
        for name, module in self.model.named_modules():
            if (isinstance(module, (nn.MultiheadAttention, nn.Linear)) or 
                'block' in name or 'attn' in name or 'mlp' in name):
                pre_hook, post_hook = create_hooks(name)
                self.handles.append(module.register_forward_pre_hook(pre_hook))
                self.handles.append(module.register_forward_hook(post_hook))
                
    def start_tracking(self):
        if not self.handles:
            self._register_hooks()
        self.active = True
        
    def stop_tracking(self):
        self.active = False
        
    def reset(self):
        for layer_time in self.layer_times.values():
            layer_time['total'] = 0.0
            layer_time['calls'] = 0
            
    def get_layer_metrics(self):
        metrics = {}
        total_time = sum(data['total'] for data in self.layer_times.values())
        
        grouped_metrics = {
            'attention': 0.0,
            'ffn': 0.0,
            'vision_encoder': 0.0,
            'other': 0.0
        }
        
        for name, data in self.layer_times.items():
            metrics[name] = {
                'total_time': data['total'],
                'avg_time': data['total'] / data['calls'] if data['calls'] > 0 else 0,
                'percentage': (data['total'] / total_time * 100) if total_time > 0 else 0,
                'calls': data['calls']
            }
            
            if 'attn' in name or 'attention' in name:
                grouped_metrics['attention'] += data['total']
            elif 'mlp' in name or 'ffn' in name:
                grouped_metrics['ffn'] += data['total']
            elif 'vision' in name or 'vit' in name:
                grouped_metrics['vision_encoder'] += data['total']
            else:
                grouped_metrics['other'] += data['total']
        
        for key in grouped_metrics:
            grouped_metrics[key] = {
                'total_time': grouped_metrics[key],
                'percentage': (grouped_metrics[key] / total_time * 100) if total_time >
                 0 else 0
            }
            
        return {
            'detailed': metrics,
            'grouped': grouped_metrics,
            'total_time': total_time
        }
            
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class MemoryTracker:
    def __init__(self):
        self.memory_stats = []
        self.peak_memory = 0
        self.start_memory = 0
        self.use_psutil = False
        self.has_memory_tracking = True
        
        try:
            import psutil
            self.psutil = psutil
            self.use_psutil = not torch.cuda.is_available()
        except ImportError:
            self.use_psutil = False
            if not torch.cuda.is_available():
                self.has_memory_tracking = False
        
    def start(self):
        self.memory_stats = []
        self.peak_memory = 0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            self.start_memory = torch.cuda.memory_allocated()
        elif self.use_psutil:
            self.start_memory = self.psutil.Process().memory_info().rss
        else:
            self.start_memory = 0
        
    def log_memory(self):
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated()
            self.memory_stats.append(current)
            self.peak_memory = max(self.peak_memory, current)
        elif self.use_psutil:
            current = self.psutil.Process().memory_info().rss
            self.memory_stats.append(current)
            self.peak_memory = max(self.peak_memory, current)
        elif self.has_memory_tracking:
            self.memory_stats.append(0)
            
    def get_metrics(self):
        if not self.has_memory_tracking or not self.memory_stats:
            return {
                'peak_memory_mb': -1,
                'memory_increase_mb': -1,
                'memory_per_token_mb': -1,
                'note': "Memory tracking not available on this device"
            }
            
        peak_memory_mb = self.peak_memory / (1024 * 1024)
        memory_increase_mb = (self.peak_memory - self.start_memory) / (1024 * 1024)
        
        return {
            'peak_memory_mb': peak_memory_mb,
            'memory_increase_mb': memory_increase_mb,
            'memory_per_token_mb': memory_increase_mb / len(self.memory_stats) if self.memory_stats else 0
        }


class TPSTracker:
    def __init__(self, model=None):
        self.start_time = None
        self.end_time = None
        self.token_timestamps = []
        self.token_counts = []
        self.step_times = []
        
        self.first_token_time = None
        
        self.input_token_timestamps = []
        self.input_token_counts = []
        self.output_token_timestamps = []
        self.output_token_counts = []
        
        self.layer_tracker = None
        if model is not None:
            self.layer_tracker = LayerPerformanceTracker(model)
            
        self.memory_tracker = MemoryTracker()
        
    def start(self):
        self.start_time = time.perf_counter()
        if self.layer_tracker:
            self.layer_tracker.reset()
            self.layer_tracker.start_tracking()
        self.memory_tracker.start()
        return self
        
    def log_token(self, count=1, is_input=False):
        current_time = time.perf_counter()
        self.token_timestamps.append(current_time)
        self.token_counts.append(count)
        
        if self.first_token_time is None and not is_input:
            self.first_token_time = current_time
            
        if is_input:
            self.input_token_timestamps.append(current_time)
            self.input_token_counts.append(count)
        else:
            self.output_token_timestamps.append(current_time)
            self.output_token_counts.append(count)
            
        self.memory_tracker.log_memory()
    
    def log_first_token(self):
        self.first_token_time = time.perf_counter()
        
    def log_step(self):
        self.step_times.append(time.perf_counter())
        
    def stop(self):
        self.end_time = time.perf_counter()
        if self.layer_tracker:
            self.layer_tracker.stop_tracking()
        
    def get_metrics(self, discard_first_n_tokens=0):
        if not self.token_timestamps or self.start_time is None or self.end_time is None:
            return {'error': 'No tokens were tracked or tracking was not started/stopped properly'}
            
        token_timestamps = self.token_timestamps[discard_first_n_tokens:]
        token_counts = self.token_counts[discard_first_n_tokens:]
        
        if not token_timestamps:
            return {'error': 'No tokens left after discarding warm-up tokens'}
            
        total_tokens = sum(token_counts)
        total_time = self.end_time - self.start_time
        overall_tps = total_tokens / total_time if total_time > 0 else 0
        
        per_token_tps = []
        for i in range(1, len(token_timestamps)):
            elapsed = token_timestamps[i] - token_timestamps[i-1]
            if elapsed > 0:
                per_token_tps.append(token_counts[i] / elapsed)
            else:
                per_token_tps.append(0)
        
        per_second_tps = []
        tokens_in_current_second = 0
        current_second = 1
        
        for i, timestamp in enumerate(token_timestamps):
            elapsed = timestamp - self.start_time
            second_bucket = int(elapsed)
            
            while current_second < second_bucket:
                per_second_tps.append(tokens_in_current_second)
                tokens_in_current_second = 0
                current_second += 1
                
            tokens_in_current_second += token_counts[i]
            
        if tokens_in_current_second > 0:
            per_second_tps.append(tokens_in_current_second)
            
        step_tps = []
        prev_time = self.start_time
        tokens_in_step = 0
        
        for i, timestamp in enumerate(self.token_timestamps):
            if i > 0 and i < len(self.step_times) and timestamp > self.step_times[i-1]:
                step_duration = self.step_times[i-1] - prev_time
                step_tps.append(tokens_in_step / step_duration if step_duration > 0 else 0)
                prev_time = self.step_times[i-1]
                tokens_in_step = 0
                
            tokens_in_step += self.token_counts[i]
        
        if self.first_token_time is not None:
            ttft = self.first_token_time - self.start_time
        else:
            timestamps = self.output_token_timestamps if self.output_token_timestamps else token_timestamps
            ttft = timestamps[0] - self.start_time if timestamps else 0
        
        peak_tps = max(per_second_tps) if per_second_tps else 0
        
        tps_std = np.std(per_second_tps) if len(per_second_tps) > 1 else 0
        tps_relative_std = (tps_std / overall_tps * 100) if overall_tps > 0 else 0
            
        input_tokens = sum(self.input_token_counts)
        output_tokens = sum(self.output_token_counts)
        
        input_tps = 0
        output_tps = 0
        
        if self.input_token_timestamps:
            input_time = self.input_token_timestamps[-1] - self.start_time
            input_tps = input_tokens / input_time if input_time > 0 else 0
            
        if self.output_token_timestamps:
            start = self.input_token_timestamps[-1] if self.input_token_timestamps else self.start_time
            output_time = self.end_time - start
            output_tps = output_tokens / output_time if output_time > 0 else 0
        
        metrics = {
            'total_tokens': total_tokens,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_time': total_time,
            'overall_tps': overall_tps,
            'input_tps': input_tps,
            'output_tps': output_tps,
            'per_token_tps': per_token_tps,
            'per_second_tps': per_second_tps,
            'step_tps': step_tps,
            'time_to_first_token': ttft,
            'peak_tps': peak_tps,
            'tps_stability': tps_std,
            'tps_relative_stability': tps_relative_std,
            'timestamps': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'first_token_time': self.first_token_time,
                'token_timestamps': token_timestamps
            },
            'memory': self.memory_tracker.get_metrics()
        }
        
        if self.layer_tracker:
            metrics['layer_metrics'] = self.layer_tracker.get_layer_metrics()
            
        return metrics
        
    def cleanup(self):
        if self.layer_tracker:
            self.layer_tracker.remove_hooks()


def measure_generation(model_func, *args, track_layers=True, **kwargs):
    model = kwargs.get('model', None)
    if model is None and len(args) > 0:
        model = args[0]
        
    tracker = TPSTracker(model if track_layers else None)
    tracker.start()
    
    result = model_func(*args, **kwargs)
    
    if hasattr(result, 'shape'):
        num_tokens = result.shape[1] if len(result.shape) > 1 else result.shape[0]
    elif isinstance(result, list):
        num_tokens = len(result)
    else:
        num_tokens = 1
        
    tracker.log_token(num_tokens)
    tracker.stop()
    
    metrics = tracker.get_metrics()
    tracker.cleanup()
    
    return result, metrics


def format_metrics(metrics, show_detailed=False):
    if 'error' in metrics:
        return f"Error: {metrics['error']}"
        
    output = []
    output.append("=== Token Generation Metrics ===")
    output.append(f"Total tokens generated: {metrics['total_tokens']}")
    
    if metrics.get('input_tokens', 0) > 0 or metrics.get('output_tokens', 0) > 0:
        output.append(f"  - Input tokens: {metrics['input_tokens']}")
        output.append(f"  - Output tokens: {metrics['output_tokens']}")
    
    output.append(f"Total generation time: {metrics['total_time']:.2f}s")
    output.append(f"Overall tokens per second: {metrics['overall_tps']:.2f} TPS")
    
    if metrics.get('input_tps', 0) > 0 or metrics.get('output_tps', 0) > 0:
        output.append(f"  - Input TPS: {metrics['input_tps']:.2f} tokens/s")
        output.append(f"  - Output TPS: {metrics['output_tps']:.2f} tokens/s")
    
    output.append(f"Time to first token (TTFT): {metrics['time_to_first_token']:.2f}s")
    output.append(f"Peak TPS: {metrics['peak_tps']:.2f} tokens/s")
    
    output.append(f"TPS stability:")
    output.append(f"  - Absolute: ±{metrics['tps_stability']:.2f} TPS")
    if 'tps_relative_stability' in metrics:
        output.append(f"  - Relative: ±{metrics['tps_relative_stability']:.1f}% of mean TPS")
    
    if 'memory' in metrics:
        mem = metrics['memory']
        output.append("\nMemory usage:")
        if 'note' in mem or mem.get('peak_memory_mb', 0) < 0:
            output.append(f"- {mem.get('note', 'Memory tracking not available')}")
        else:
            output.append(f"- Peak: {mem['peak_memory_mb']:.2f} MB")
            output.append(f"- Increase: {mem['memory_increase_mb']:.2f} MB")
            output.append(f"- Per token: {mem['memory_per_token_mb']:.2f} MB")
    
    if 'layer_metrics' in metrics:
        layer_metrics = metrics['layer_metrics']
        output.append("\nLayer performance:")
        for name, data in layer_metrics['grouped'].items():
            output.append(f"- {name}: {data['percentage']:.1f}% of time ({data['total_time']:.2f}s)")
            
        if show_detailed and 'detailed' in layer_metrics:
            output.append("\nDetailed layer breakdown:")
            sorted_layers = sorted(
                layer_metrics['detailed'].items(),
                key=lambda x: x[1]['total_time'],
                reverse=True
            )
            for name, data in sorted_layers[:10]:
                output.append(f"- {name}: {data['percentage']:.1f}% ({data['total_time']:.2f}s, {data['calls']} calls)")
    
    if 'per_second_tps' in metrics and metrics['per_second_tps']:
        output.append("\nTPS per second:")
        per_second = metrics['per_second_tps']
        chunks = [per_second[i:i+10] for i in range(0, len(per_second), 10)]
        for i, chunk in enumerate(chunks):
            start_second = i * 10
            output.append(f"- {start_second}-{start_second+len(chunk)-1}s: {', '.join(f'{tps:.1f}' for tps in chunk)}")
    
    return "\n".join(output)
