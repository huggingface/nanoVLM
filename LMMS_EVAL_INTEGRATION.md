# PR: Add lmms-eval Integration for Enhanced Model Evaluation

## Summary

This PR integrates [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) into nanoVLM, providing comprehensive multi-modal evaluation capabilities beyond the default MMStar benchmark. Users can now evaluate their models on 50+ vision-language benchmarks both during training and as standalone evaluations.

## Key Features

### 🎯 Standalone Evaluation Script
- New `evaluate_with_lmms.py` script for evaluating trained models
- Support for multiple benchmarks in a single run
- Configurable batch size and example limits
- Built-in task discovery with `--list_tasks`

### 🔧 Seamless Training Integration
- Automatic evaluation during training at specified intervals
- Results logged to wandb alongside existing metrics
- Configurable through `TrainConfig` parameters
- Optional feature - can be disabled if not needed

### 📊 Enhanced Evaluation Capabilities
- Support for 50+ vision-language benchmarks including:
  - MME, MMStar, GQA, VQAv2
  - AI2D, ChartQA, DocVQA, TextVQA
  - And many more specialized benchmarks
- Pretty-printed results with table formatting
- Detailed JSON output for further analysis

## Usage Examples

### Quick Start

```bash
# Install lmms-eval (optional dependency)
pip install lmms-eval

# Evaluate a trained model on multiple benchmarks
python evaluate_with_lmms.py --model_path lusxvr/nanoVLM-222M --tasks mmstar,mme,gqa

# List all available evaluation tasks
python evaluate_with_lmms.py --list_tasks

# Evaluate with custom settings
python evaluate_with_lmms.py \
    --model_path ./checkpoints/my_model \
    --tasks mme,vqav2 \
    --batch_size 16 \
    --limit 100  # Useful for quick debugging
```

### Training Integration

Enable lmms-eval during training by modifying your `TrainConfig`:

```python
from models.config import TrainConfig

train_cfg = TrainConfig(
    # ... other configs ...
    
    # Enable lmms-eval integration
    use_lmms_eval=True,
    
    # Select evaluation tasks
    lmms_eval_tasks=("mme", "gqa", "vqav2"),
    
    # Configure evaluation settings
    lmms_eval_batch_size=8,
    lmms_eval_limit=None,  # Set to small number for debugging
    save_lmms_results=True  # Save detailed results to checkpoint dir
)
```

### Programmatic Usage

```python
from evaluation import run_lmms_evaluation, get_available_tasks

# List available tasks
tasks = get_available_tasks()
print(f"Available tasks: {tasks}")

# Run evaluation
results = run_lmms_evaluation(
    model=model,
    tasks=["mmstar", "mme"],
    batch_size=8
)

# Results format:
# {
#     "mmstar": {
#         "accuracy": 0.75,
#         "accuracy_stderr": 0.02
#     },
#     "mme": {
#         "perception_score": 1245.5,
#         "cognition_score": 287.5
#     }
# }
```

## Implementation Details

### New Files
- `evaluate_with_lmms.py` - Standalone evaluation script
- `evaluation.py` - Core evaluation functions
- `models/lmms_eval_wrapper.py` - Model wrapper for lmms-eval compatibility
- `tests/test_lmms_eval_integration.py` - Unit tests

### Modified Files
- `train.py` - Added lmms-eval integration during training
- `models/config.py` - Added configuration parameters
- `README.md` - Updated documentation
- `requirements.txt` - Added lmms-eval as optional dependency

### Architecture
The integration follows a clean separation of concerns:
1. **Wrapper Layer** (`NanoVLMWrapper`) - Adapts nanoVLM to lmms-eval's API
2. **Evaluation Module** - High-level functions for running evaluations
3. **Training Integration** - Optional hooks in the training loop
4. **CLI Interface** - User-friendly command-line tool

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_lmms_eval` | `False` | Enable lmms-eval during training |
| `lmms_eval_tasks` | `("mmstar",)` | Tasks to evaluate |
| `lmms_eval_batch_size` | `4` | Batch size for evaluation |
| `lmms_eval_limit` | `None` | Limit examples per task |
| `save_lmms_results` | `True` | Save detailed results |

## Performance Considerations

- Evaluation can be time-consuming for large benchmarks
- Use `--limit` flag for quick debugging
- Batch size affects memory usage and speed
- Some benchmarks require significant disk space for datasets

## Backward Compatibility

- lmms-eval is an **optional dependency**
- Existing training workflows remain unchanged
- Default behavior preserved when `use_lmms_eval=False`
- Graceful fallback if lmms-eval not installed

## Testing

Run the integration tests:
```bash
python -m pytest tests/test_lmms_eval_integration.py -v
```

## Future Enhancements

- [ ] Distributed evaluation support for faster benchmarking
- [ ] Custom task creation guide
- [ ] Benchmark result visualization tools
- [ ] Automatic model card generation with eval results

## Examples of Evaluation Output

```
Evaluation Results:
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task    ┃ Metric    ┃ Value           ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ mmstar  │ accuracy  │ 0.3542 ± 0.0138 │
├─────────┼───────────┼─────────────────┤
│ mme     │ score     │ 1532.86         │
├─────────┼───────────┼─────────────────┤
│ gqa     │ accuracy  │ 0.5834 ± 0.0156 │
├─────────┼───────────┼─────────────────┤
│ vqav2   │ accuracy  │ 0.6721 ± 0.0149 │
└─────────┴───────────┴─────────────────┘
```

## Acknowledgments

This integration builds upon the excellent work of the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) project, which provides a unified framework for evaluating large multi-modal models.

---

**Note**: This PR maintains nanoVLM's philosophy of simplicity and educational value while significantly expanding its evaluation capabilities. The integration is designed to be optional and non-intrusive to existing workflows.