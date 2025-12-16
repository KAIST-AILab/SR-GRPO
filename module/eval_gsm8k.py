"""
GSM8K Accuracy Evaluation Script for SR-GRPO trained model.

This script evaluates the trained model on the GSM8K test set and calculates accuracy.
"""

from unsloth import FastLanguageModel
from vllm import SamplingParams
from datasets import load_dataset
from tqdm import tqdm
import re
import json
from datetime import datetime

# ============ Configuration ============
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "sr_grpo_saved_lora"  # Path to your trained LoRA
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 32  # Adjust based on GPU memory

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
    """Extract the answer from XML format."""
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    return ""


def extract_hash_answer(text: str) -> str | None:
    """Extract the answer from GSM8K format (after ####)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove commas from numbers (e.g., "1,000" -> "1000")
    answer = answer.replace(",", "")
    # Remove dollar signs and other common symbols
    answer = answer.replace("$", "").replace("%", "")
    # Strip whitespace
    answer = answer.strip()
    return answer


def load_model(lora_path: str = None):
    """Load the model, optionally with a LoRA adapter."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
    )
    
    lora_request = None

    
    return model, tokenizer, lora_request


def prepare_prompts(tokenizer, questions: list[str]) -> list[str]:
    """Prepare prompts for batch generation."""
    prompts = []
    for question in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(text)
    return prompts


def batch_generate(model, prompts: list[str], lora_request=None) -> list[str]:
    """Generate responses for a batch of prompts."""
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for evaluation
        top_p=1.0,
        max_tokens=512,
    )
    
    outputs = model.fast_generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    
    return [output.outputs[0].text for output in outputs]


def evaluate_gsm8k(model, tokenizer, lora_request=None, num_samples: int = None):
    """Evaluate model on GSM8K test set."""
    # Load GSM8K test set
    print("Loading GSM8K test set...")
    dataset = load_dataset('openai/gsm8k', 'main')['test']
    
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    questions = [item['question'] for item in dataset]
    gold_answers = [extract_hash_answer(item['answer']) for item in dataset]
    
    # Generate responses in batches
    all_responses = []
    for i in tqdm(range(0, len(questions), BATCH_SIZE), desc="Generating"):
        batch_questions = questions[i:i + BATCH_SIZE]
        batch_prompts = prepare_prompts(tokenizer, batch_questions)
        batch_responses = batch_generate(model, batch_prompts, lora_request)
        all_responses.extend(batch_responses)
    
    # Calculate accuracy
    correct = 0
    results = []
    
    for i, (question, response, gold) in enumerate(zip(questions, all_responses, gold_answers)):
        pred = extract_xml_answer(response)
        pred_normalized = normalize_answer(pred)
        gold_normalized = normalize_answer(gold) if gold else ""
        
        is_correct = pred_normalized == gold_normalized
        if is_correct:
            correct += 1
        
        results.append({
            "id": i,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": pred,
            "full_response": response,
            "correct": is_correct
        })
    
    accuracy = correct / len(dataset) * 100
    
    return accuracy, correct, len(dataset), results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SR-GRPO model on GSM8K")
    parser.add_argument("--lora", type=str, default=LORA_PATH, help="Path to LoRA adapter")
    parser.add_argument("--no-lora", action="store_true", help="Evaluate base model without LoRA")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--save-results", type=str, default=None, help="Path to save detailed results (JSON)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for generation")
    args = parser.parse_args()
    
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    
    print("=" * 60)
    print("GSM8K Accuracy Evaluation")
    print("=" * 60)
    
    # Load model
    lora_path = None if args.no_lora else args.lora
    print(f"\nModel: {MODEL_NAME}")
    print(f"LoRA: {lora_path if lora_path else 'None (base model)'}")
    
    model, tokenizer, lora_request = load_model(lora_path)
    
    # Evaluate
    accuracy, correct, total, results = evaluate_gsm8k(
        model, tokenizer, lora_request, 
        num_samples=args.num_samples
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Save detailed results if requested
    if args.save_results:
        output = {
            "model": MODEL_NAME,
            "lora": lora_path,
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
        with open(args.save_results, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.save_results}")
    
    # Show some examples
    print("\n" + "=" * 60)
    print("Sample Results (first 5)")
    print("=" * 60)
    
    for r in results[:5]:
        status = "✓" if r["correct"] else "✗"
        print(f"\n[{status}] Question {r['id'] + 1}:")
        print(f"  Q: {r['question'][:100]}...")
        print(f"  Gold: {r['gold_answer']}")
        print(f"  Pred: {r['predicted_answer']}")
    
    # Show some incorrect examples
    incorrect = [r for r in results if not r["correct"]]
    if incorrect:
        print("\n" + "=" * 60)
        print(f"Sample Incorrect Results (showing 3 of {len(incorrect)})")
        print("=" * 60)
        
        for r in incorrect[:3]:
            print(f"\n[✗] Question {r['id'] + 1}:")
            print(f"  Q: {r['question'][:100]}...")
            print(f"  Gold: {r['gold_answer']}")
            print(f"  Pred: {r['predicted_answer']}")


if __name__ == "__main__":
    main()
