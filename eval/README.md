# lmms-eval Integration for Enhanced Model Evaluation

### Quick Start

```bash
# Evaluate a trained model on multiple benchmarks
python evaluation.py --model_path lusxvr/nanoVLM-222M --tasks mmstar,mme,gqa

# List all available evaluation tasks
python evaluation.py --list_tasks

# Evaluate with custom settings
python evaluation.py \
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

## Acknowledgments

This integration builds upon the excellent work of the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) project, which provides a unified framework for evaluating large multi-modal models.
