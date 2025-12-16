"""
Example script for training with SR-GRPO (Soft Reward GRPO).

This script demonstrates how to use the SRGRPOTrainer for fine-tuning
language models on the GSM8K math dataset.
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import re
from datasets import load_dataset, Dataset

# Import our custom SR-GRPO trainer
from grpo.SR_GRPO.sr_grpo_trainer import SRGRPOTrainer, SRGRPOConfig


# ============ Model Configuration ============
max_seq_length = 1024
lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.5,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# ============ Dataset Preparation ============
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data


dataset = get_gsm8k_questions()


# ============ Reward Functions ============
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward for correct answers."""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", 
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward for numeric answers."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for strict XML format compliance."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for soft XML format compliance."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """Partial reward for XML structure elements."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001

    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward based on XML element counting."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# ============ Training Configuration ============
training_args = SRGRPOConfig(
    # SR-GRPO specific parameter
    tau=0.5,  # Temperature for softmax weighting (lower = sharper, higher = smoother)
    
    # vLLM settings
    use_vllm=True,
    
    # Optimizer settings
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    
    # Training settings
    logging_steps=10,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    num_generations=8,
    max_prompt_length=256,
    max_completion_length=512,
    num_train_epochs=1,
    max_grad_norm=0.1,
    
    # Output settings
    output_dir="sr_grpo_output",
    # report_to="wandb",  # Uncomment to enable W&B logging
)


# ============ Initialize and Train ============
trainer = SRGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Save the trained LoRA adapter
model.save_lora("sr_grpo_saved_lora")

print("Training completed! LoRA saved to 'sr_grpo_saved_lora'")
