# Dual-Teacher Knowledge Distillation for Mathematical Reasoning

Master's thesis implementation of dual-teacher knowledge distillation using Gemma-2-9B and Qwen2.5-7B as teachers to improve a Gemma-3-1B student model on GSM8K mathematical reasoning.

## Quick Start

```bash
# Run the quick start script
python quick_start.py

# Or run experiments directly
python train_dual_teacher.py --experiment main
python train_dual_teacher.py --ablation
```

## Key Files

- `train_dual_teacher.py` - Main training script with ablation studies
- `dual_teacher_trainer.py` - Custom trainer implementation
- `data_utils.py` - Data loading and evaluation utilities
- `quick_start.py` - Setup and testing script
- `analysis_tools.py` - Result analysis and visualization
- `save_results.py` - Result saving utilities

## Workflow

1. **Setup**: Run `python quick_start.py` for initial setup and testing
2. **Train**: Run experiments with `train_dual_teacher.py`
3. **Analyze**: Use `analysis_tools.py` to visualize results
4. **Results**: Check `./results/` for saved models and metrics

## Requirements

- GPU with 40GB+ memory (for full models)
- PyTorch 2.0+, Transformers 4.35+, Datasets 2.14+

## Experiments

- Main dual-teacher distillation
- Temperature ablation (2.0, 4.0, 6.0)
- Alpha ablation (0.5, 0.7, 0.9)
- Single teacher baselines