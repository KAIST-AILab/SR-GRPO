"""
SR-GRPO (Soft Reward GRPO) Trainer

This module implements a variant of GRPO (Group Relative Policy Optimization) that uses
softmax-weighted advantages instead of simple mean-normalized advantages.

The key difference from standard GRPO:
- Standard GRPO: advantages = (rewards - mean) / std
- SR-GRPO: advantages are computed using softmax weights based on normalized rewards

This approach provides smoother gradient updates by giving more weight to higher-reward
completions within each group.
"""

import torch
from typing import Any, Optional, Union
from trl import GRPOTrainer, GRPOConfig


class SRGRPOConfig(GRPOConfig):
    """
    Configuration class for SR-GRPO Trainer.
    
    Inherits all parameters from GRPOConfig and adds:
    
    Args:
        tau (`float`, *optional*, defaults to `0.5`):
            Temperature parameter for softmax weighting. Lower values make the 
            weighting sharper (more weight on high-reward samples), higher values
            make it smoother (more uniform weighting).
    """
    
    def __init__(
        self,
        *args,
        tau: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tau = tau


class SRGRPOTrainer(GRPOTrainer):
    """
    Trainer for Soft Reward GRPO (SR-GRPO).
    
    This trainer extends the standard GRPO trainer by modifying how advantages
    are computed. Instead of using simple mean-normalized advantages, SR-GRPO
    uses softmax-weighted advantages based on the rewards within each group.
    
    The advantage computation:
    1. Normalize rewards within each group: z = (r - mean) / std
    2. Compute softmax weights: w = softmax(z / tau)
    3. Compute weighted advantage: soft_adv = sum(w * r)
    4. Broadcast to all samples in the group
    
    This approach gives more weight to high-reward completions, potentially
    leading to more stable and effective training.
    
    Example:
    
    ```python
    from sr_grpo_trainer import SRGRPOTrainer, SRGRPOConfig
    from datasets import load_dataset
    
    dataset = load_dataset("trl-lib/tldr", split="train")
    
    def reward_func(completions, **kwargs):
        return [float(len(set(completion))) for completion in completions]
    
    config = SRGRPOConfig(
        output_dir="sr_grpo_output",
        tau=0.5,  # Temperature for softmax weighting
        num_generations=8,
        per_device_train_batch_size=8,
    )
    
    trainer = SRGRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
    )
    
    trainer.train()
    ```
    
    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Same as GRPOTrainer.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions. Same as GRPOTrainer.
        args ([`SRGRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset: Dataset to use for training.
        eval_dataset: Dataset to use for evaluation.
        processing_class: Processing class (tokenizer).
        reward_processing_classes: Processing classes for reward models.
        callbacks: List of callbacks.
        optimizers: Tuple of optimizer and scheduler.
        peft_config: PEFT configuration.
    """
    
    _tag_names = ["trl", "grpo", "sr-grpo"]
    
    def __init__(
        self,
        model,
        reward_funcs,
        args: Optional[SRGRPOConfig] = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
    ):
        # Set default config if not provided
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = SRGRPOConfig(f"{model_name}-SR-GRPO")
        
        # Store tau before calling parent __init__
        self.tau = getattr(args, 'tau', 0.5)
        
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the SR-GRPO loss with softmax-weighted advantages.
        
        The key modification from standard GRPO is in how advantages are computed:
        - Standard GRPO: advantages = (rewards - mean) / std
        - SR-GRPO: advantages = softmax-weighted sum of rewards
        
        Args:
            model: The model being trained.
            inputs: Dictionary containing prompt_ids, completion_ids, masks, etc.
            return_outputs: If True, also return the model outputs (not supported).
            num_items_in_batch: Number of items in batch (optional).
            
        Returns:
            The computed loss tensor.
        """
        if return_outputs:
            raise ValueError("The SRGRPOTrainer does not support returning outputs")
        
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        
        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps) 
            - (ref_per_token_logps - per_token_logps) 
            - 1
        )
        
        # ============ SR-GRPO: Compute softmax-weighted advantages ============
        K = num_items_in_batch or self.args.num_generations
        rewards = inputs["rewards"]  # (B,)
        
        assert rewards.numel() % K == 0, "batch must be multiple of K (num_generations)"
        
        # Reshape rewards into groups
        groups = rewards.view(-1, K)  # (N, K) where N = batch_size / K
        
        # Normalize within each group
        group_mean = groups.mean(dim=1, keepdim=True)
        group_std = groups.std(dim=1, keepdim=True)
        z = (groups - group_mean) / (group_std + 1e-5)
        
        # Compute softmax weights with temperature tau
        w = torch.nn.functional.softmax(z / self.tau, dim=1)  # (N, K)
        
        # Compute weighted advantage for each group
        soft_adv_group = (w * groups).sum(dim=1, keepdim=True)  # (N, 1)
        
        # Broadcast advantage to all samples in the group
        advantages = soft_adv_group.repeat(1, K).view(-1)  # (B,)
        # ======================================================================
        
        # Compute the loss
        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        # Log SR-GRPO specific metrics
        self._metrics["soft_advantage_mean"].append(advantages.mean().item())
        self._metrics["soft_advantage_std"].append(advantages.std().item())
        
        return loss


# For backward compatibility, also export with alternate names
SoftRewardGRPOTrainer = SRGRPOTrainer
SoftRewardGRPOConfig = SRGRPOConfig
