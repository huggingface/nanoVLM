# Enhanced Tokens Per Second (TPS) Tracking in nanoVLM

This document explains the enhanced TPS tracking functionality implemented in the nanoVLM project, providing comprehensive metrics for model performance evaluation and comparison.

## Overview

The implementation measures and reports an extensive set of metrics:

- **Tokens Per Second (TPS)**: 
  - Overall TPS: Total speed of token generation
  - Input TPS: Processing speed of input prompt tokens
  - Output TPS: Generation speed of output tokens
  
- **Time Measurements**:
  - Time to First Token (TTFT): Actual latency to generate the first token
  - Total Generation Time: Complete time for the entire generation
  - Per-token Generation Time: Time spent generating each individual token
  
- **Performance Analysis**:
  - Layer Performance: Detailed breakdown of time spent in different model components
  - TPS Stability: Standard deviation and relative variance of generation speed
  - Peak TPS: Maximum tokens per second achieved during generation
  
- **Resource Monitoring**:
  - Memory Usage: Peak and per-token memory consumption
  - Device Utilization: Metrics specific to the processing device (CUDA/MPS/CPU)

## Advanced Features

The enhanced TPS tracking implementation includes:

- **Warmup Handling**: Support for model warm-up iterations to exclude initial overhead
- **Separate Input/Output Tracking**: Distinct measurements for input processing vs. output generation
- **Per-Token Profiling**: Individual timing data for each generated token
- **Layer-Level Analysis**: PyTorch hooks for non-intrusive introspection of model internals
- **Stability Metrics**: Both absolute and relative measures of TPS consistency
- **Visualization Suite**: Comprehensive plotting capabilities for performance analysis
- **Seamless Integration**: Direct support in model generation methods

## Usage Examples

### Direct Integration with Model Generation

```python
from models.vision_language_model import VisionLanguageModel

# Load model
model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M").to(device)

# Generate with built-in metrics
gen, metrics = model.generate(
    tokens, 
    image, 
    max_new_tokens=100,
    return_tps_metrics=True,  # Enable metrics collection
    warm_up_iterations=2,     # Run 2 warm-up iterations before measuring
    discard_warm_up_tokens=True  # Exclude first token from metrics calculation
)

# Print formatted metrics
print(format_metrics(metrics, show_detailed=True))
```

### Using the generate.py Script

The `generate.py` script now provides command-line options to enable or disable metrics:

```bash
# Basic usage without metrics (default)
python generate.py

# Same as above but explicitly stated
python generate.py --prompt "What is this?" --image assets/image.png

# Enable performance metrics
python generate.py --metrics

# Enable performance metrics with detailed layer breakdown
python generate.py --metrics --detailed-metrics

# Custom generation parameters with metrics
python generate.py --prompt "Describe this scene" --image custom_image.jpg --max-tokens 200 --metrics --warmup
```

Command-line options:
- `--prompt`: Text prompt to generate from (default: "What is this?")
- `--image`: Path to input image (default: "assets/image.png")
- `--num-generations`: Number of generations to run (default: 5)
- `--max-tokens`: Maximum tokens to generate (default: 100)
- `--metrics`: Enable detailed performance metrics display
- `--detailed-metrics`: Show detailed layer breakdown in metrics
- `--warmup`: Run warmup iterations before generation
- `--model`: Model name or path (default: "lusxvr/nanoVLM-222M")

### Manual Tracking for Custom Scenarios

```python
from utils.metrics import TPSTracker, format_metrics

# Create a tracker with layer profiling
tracker = TPSTracker(model)

# Start tracking
tracker.start()

# Track input tokens processing
tracker.log_token(input_tokens, is_input=True)

# For each generated token
for i in range(max_tokens):
    # Generate token
    # ...
    
    # For the first token, explicitly log it
    if i == 0:
        tracker.log_first_token()
    
    # Log token generation
    tracker.log_token(1, is_input=False)
    tracker.log_step()

# Finalize tracking
tracker.stop()

# Get detailed metrics with optional warm-up token discarding
metrics = tracker.get_metrics(discard_first_n_tokens=1)

# Clean up hooks
tracker.cleanup()
```

### Using the Benchmark Tool

For comprehensive performance analysis, use the dedicated benchmark tool:

```bash
# Basic usage
python benchmark-tps.py

# Comprehensive benchmark with visualizations
python benchmark-tps.py \
  --model lusxvr/nanoVLM-222M \
  --prompt "Describe this image in detail" \
  --image assets/image.png \
  --max-tokens 20,50,100,200,500 \
  --iterations 5 \
  --plot \
  --detailed \
  --device cuda
```

Command-line options:
- `--model`: Model name or path (default: "lusxvr/nanoVLM-222M")
- `--prompt`: Text prompt to generate from (default: "What is this?")
- `--image`: Path to input image (default: "assets/image.png")
- `--max-tokens`: Comma-separated list of token counts to benchmark (default: "20,50,100,200")
- `--iterations`: Number of iterations per token count (default: 3)
- `--output`: Output file for results (default: "tps_benchmark_results.json")
- `--plot`: Generate visualization plots
- `--detailed`: Show detailed metrics
- `--device`: Device to run on (cuda, mps, cpu)

### Enhanced Output Example

```
=== Token Generation Metrics ===
Total tokens generated: 100
  - Input tokens: 12
  - Output tokens: 88
Total generation time: 4.25s
Overall tokens per second: 23.53 TPS
  - Input TPS: 48.35 tokens/s
  - Output TPS: 21.87 tokens/s
Time to first token (TTFT): 0.12s
Peak TPS: 35.21 tokens/s

TPS stability:
  - Absolute: ±3.45 TPS
  - Relative: ±14.7% of mean TPS

Memory usage:
- Peak: 1542.34 MB
- Increase: 124.56 MB
- Per token: 1.25 MB

Layer performance:
- attention: 32.5% of time (1.38s)
- ffn: 15.2% of time (0.65s)
- vision_encoder: 8.7% of time (0.37s)
- other: 43.6% of time (1.85s)

Detailed layer breakdown:
- blocks.3.attn: 9.1% (0.39s, 88 calls)
- blocks.2.attn: 8.8% (0.37s, 88 calls)
- blocks.4.mlp: 7.2% (0.31s, 88 calls)
- vision_encoder.blocks.1: 4.5% (0.19s, 1 calls)
- blocks.0.norm1: 2.1% (0.09s, 88 calls)
- ...

TPS per second:
- 0-0s: 8.3
- 1-1s: 22.1
- 2-2s: 25.4
- 3-3s: 29.7
- 4-4s: 25.9
```


## Implementation Architecture

The implementation follows a modular architecture:

1. **Core Trackers**:
   - **TPSTracker**: Main coordinator for all metrics
   - **LayerPerformanceTracker**: Uses PyTorch hooks for non-intrusive layer profiling
   - **MemoryTracker**: Device-specific memory monitoring

2. **Integration Points**:
   - **Model Method Enhancement**: Direct integration with generate() methods
   - **Benchmark Tools**: Dedicated benchmarking utilities
   - **Utility Functions**: For formatting and visualization

3. **Visualization Components**:
   - **Matplotlib Integration**: For comprehensive performance charts
   - **Data Export**: JSON output for further analysis

## Consolidated Metrics Logging

The TPS tracking system now uses a single consolidated log file format that combines human-readable metrics in a structured, easy-to-analyze layout.

```bash
# Automatically generate structured logs (detailed by default)
python generate.py --metrics

# Control the level of detail
python generate.py --metrics --log-level basic
python generate.py --metrics --log-level detailed

# Specify a custom log filename if needed
python generate.py --metrics --log-file custom_name.log

# Disable logging if needed
python generate.py --metrics --no-log

# Full benchmark with automatic logging
python benchmark-tps.py --detailed
```

### Automatic File Naming & Organization

Log files are automatically named based on model, token count, and timestamp:

```
logs/tps_metrics_nanoVLM-222M_100tokens_20250513_014520.log
logs/tps_metrics_nanoVLM-222M_benchmark_20250513_015300.log
```

### Multi-run Consolidation

For multiple generations or benchmark runs, all metrics are consolidated into a single log file with clear separators between runs. This makes it easier to analyze and compare results without managing multiple files:

- **Multiple generations of the same model**: All runs appended to the same file
- **Benchmark across different token counts**: All configurations in one file
- **Clear separators**: Each run's results clearly divided for easy analysis

### Comprehensive Structured Format

The log format has been carefully designed to be both human-readable and information-rich:

```
==================================================
GENERATION SUMMARY
==================================================
Model: nanoVLM-222M
Date: 2025-05-13 01:51:00
Device: mps

-------------------- PERFORMANCE METRICS --------------------
Total tokens generated: 20
  - Input tokens: 8
  - Output tokens: 20
Total generation time: 1.96s
Overall tokens per second: 10.22 TPS
  - Input TPS: 81.00 tokens/s
  - Output TPS: 10.77 tokens/s
Time to first token (TTFT): 0.27s
Peak TPS: 20.00 tokens/s
TPS stability:
  - Absolute: ±0.00 TPS
  - Relative: ±0.0% of mean TPS

Memory usage:
- Peak: 0.00 MB
- Increase: 0.00 MB
- Per token: 0.00 MB

Layer performance:
- attention: 27.7% of time (0.87s)
- ffn: 12.0% of time (0.38s)
- vision_encoder: 2.3% (0.07s)
- other: 58.0% (1.82s)

Detailed layer breakdown (top 10):
- decoder.blocks.0: 25.5% (0.80s, 20 calls)
- decoder.blocks.0.norm1: 7.0% (0.22s, 20 calls)
- decoder.blocks.0.mlp: 4.8% (0.15s, 20 calls)
- vision_encoder.blocks.0: 1.4% (0.04s, 1 calls)
- decoder.blocks.0.mlp.gate_proj: 1.2% (0.04s, 20 calls)
...

TPS per second:
- 0-0s: 20.0

================================================================================

GENERATION SUMMARY
...
```

### Examples

**Single Generation Log**:
```
logs/tps_metrics_nanoVLM-222M_20tokens_20250513_015230.log
```

Contains multiple generation runs with the same configuration, making it easy to compare stability across runs.

**Benchmark Log**:
```
logs/tps_metrics_nanoVLM-222M_benchmark_20250513_015300.log
```

Contains results for multiple token counts (20, 50, 100), allowing for easy analysis of how performance scales with sequence length.

The consolidated approach keeps logs organized, easy to read, and provides all the information needed for thorough performance analysis while eliminating the duplication of unnecessary raw data.


## Best Practices for Performance Testing

When using the TPS tracking tools, consider these best practices:

1. **Always use warm-up iterations** to stabilize the model before measurement
2. **Run multiple iterations** for statistically significant results
3. **Test with varying sequence lengths** to understand scaling behavior
4. **Compare input vs. output TPS** to identify bottlenecks
5. **Analyze layer breakdown** to target optimization efforts
6. **Track TTFT separately** from overall TPS for user experience optimization
7. **Consider TPS stability** as an important metric alongside raw speed
8. **Log results to files** for long-term tracking and comparison
9. **Use heatmaps** to identify layer-specific bottlenecks

## Future Directions

Potential enhancements for future development:

1. **Advanced KV-Cache Analysis**: Detailed metrics on cache hit rates and efficiency
2. **Multi-Modal Breakdown**: Separate performance tracking for vision vs. language components
3. **Context Length Scaling**: How performance varies with increasing context windows
4. **Batch Size Optimization**: Find optimal batch sizes for different hardware
5. **Hardware Utilization Correlation**: Connect TPS with GPU/CPU utilization metrics
6. **Fine-grained Memory Tracking**: Per-layer memory consumption analysis
7. **Dynamic Visualization Dashboard**: Real-time performance monitoring during generation
8. **PyTorch Profiler Integration**: More detailed performance insights via torch.profiler
