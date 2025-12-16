# SR-GRPO: Soft Reward Group Relative Policy Optimization

A variant of GRPO (Group Relative Policy Optimization) that uses softmax-weighted advantages for smoother and more stable training.

## Overview

SR-GRPO modifies the standard GRPO algorithm by changing how advantages are computed within each group of completions. Instead of using simple mean-normalized advantages, SR-GRPO applies softmax weighting based on the rewards.

### Key Differences from Standard GRPO

| Aspect | Standard GRPO | SR-GRPO |
|--------|--------------|---------|
| Advantage | `(reward - mean) / std` | Softmax-weighted sum |
| Weighting | Uniform | Higher weight on better completions |
| Temperature | N/A | Configurable via `tau` parameter |



No additional installation required! This module inherits from the `trl` library's `GRPOTrainer`. Just ensure you have:

```bash
pip install trl transformers accelerate
# Optional: for fast inference
pip install unsloth
```

## Files

```
grpo/
├── module/
    ├── sr_grpo_trainer.py    # SR-GRPO Trainer implementation
    ├── train_sr_grpo.py      # Example training script
    ├── eval_gsm8k.py
└── README.md             # This file
```

## Configuration

### SRGRPOConfig Parameters

All parameters from `GRPOConfig` are supported, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | float | 0.5 | Temperature for softmax weighting |

### Inherited Parameters (Common)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_generations` | int | 8 | Number of completions per prompt |
| `beta` | float | 0.1 | KL penalty coefficient |
| `max_prompt_length` | int | 256 | Maximum prompt length |
| `max_completion_length` | int | 512 | Maximum completion length |
| `use_vllm` | bool | False | Use vLLM for fast generation |

## Logged Metrics

SR-GRPO logs additional metrics compared to standard GRPO:

- `soft_advantage_mean`: Mean of computed soft advantages
- `soft_advantage_std`: Standard deviation of soft advantages
- `reward`: Mean reward across all completions
- `reward_std`: Standard deviation of rewards
- `kl`: KL divergence from reference model
- `completion_length`: Average completion length

## Example: GSM8K Training

See `train_sr_grpo.py` for a complete example training on the GSM8K math dataset.

```bash
python train_sr_grpo.py
```

## Tips for Choosing τ (Temperature)

- **τ = 0.1 - 0.3**: Aggressive weighting, focuses heavily on best samples. Good when you have clear quality differences.
- **τ = 0.5**: Balanced weighting (default). Works well for most cases.
- **τ = 1.0+**: Smooth weighting, closer to standard GRPO. Good when reward signals are noisy.
