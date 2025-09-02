#!/usr/bin/env python3
"""
Clean test file for evaluating accuracy on s1K dataset variant.
The LLM takes reasoning trajectory with prompt as input and outputs answer in <answer> tags.
Accuracy is computed by matching extracted numbers.
"""

import torch
import re
import argparse
import json
from vllm import LLM, SamplingParams, TokensPrompt
from datasets import load_dataset
from tqdm import tqdm

def extract_answer_from_text(text):
    """Extract answer from text within \boxed{} tags."""
    # Look for \boxed{...} format
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, text, re.DOTALL)
    
    if matches:
        answer_text = matches[-1].strip()
        return answer_text
    
    # Fallback to <answer> tags if boxed format not found
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        answer_text = matches[-1].strip()
        return answer_text
    
    return None


def extract_number_from_text(text):
    """Extract the final numerical answer from text."""
    if not text:
        return None
    
    # Remove any HTML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Look for numbers (including decimals and negatives)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    if numbers:
        try:
            # Return the last number found
            return float(numbers[-1])
        except ValueError:
            return None
    
    return None


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer with appropriate settings using vLLM."""
    print(f"Loading model: {model_name}")
    
    # Load model using vLLM with optimized settings for batch processing
    model = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=16384,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        enable_prefix_caching=True,  # Enable prefix caching for better performance
    )
    
    # Get tokenizer from the vLLM model
    tokenizer = model.get_tokenizer()
    
    return model, tokenizer


def generate_batch_response(model, tokenizer, prompts):
    """Generate responses for multiple prompts in a single batch using vLLM."""
    # Convert all prompts to TokensPrompt format
    prompt_tokens = [TokensPrompt(prompt_token_ids=prompt.tolist()[0]) for prompt in prompts]
    
    # Create sampling parameters for deterministic generation
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=1.0,
        max_tokens=4096,
    )
    
    print(f"Processing {len(prompts)} prompts in batch...")
    
    # Generate responses for all prompts in a single batch
    outputs = model.generate(prompt_tokens, sampling_params, use_tqdm=True)
    
    # Extract text from all outputs
    responses = []
    for i, output in enumerate(outputs):
        if output.outputs and len(output.outputs) > 0:
            response_text = output.outputs[0].text
            responses.append(response_text)
        else:
            print(f"Warning: No output for prompt {i}")
            responses.append("")
    
    print(f"Successfully generated {len(responses)} responses")
    return responses

    


def evaluate_dataset(model, tokenizer, dataset, max_samples=None):
    """Evaluate model on dataset and return results using batched processing."""
    results = []
    correct_count = 0
    total_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    print(f"Evaluating on {total_samples} samples...")
    
    # Prepare all prompts in batch
    print("Preparing prompts for batch processing...")
    all_prompts = []
    all_samples = []
    
    for i in tqdm(range(total_samples), desc="Preparing prompts"):
        sample = dataset[i]
        
        # Create prompt with reasoning trajectory using chat template
        question = sample["question"]
        reasoning = sample["deepseek_thinking_trajectory"]
        # Remove any text after "Final Answer" (case-insensitive) from reasoning
        import re
        if '**Final Answer**' in reasoning:
            reasoning = re.split(r"\*\*Final Answer\*\*", reasoning, flags=re.IGNORECASE)[0].strip()
        elif 'Final Answer' in reasoning:
            reasoning = re.split(r"Final Answer", reasoning, flags=re.IGNORECASE)[0].strip()
        elif r'\boxed' in reasoning:
            reasoning = re.split(r'\\boxed{', reasoning, flags=re.IGNORECASE)[0].strip()
        
        # Use the tokenizer's chat template with reasoning as assistant response
        messages = [
            {
                "role": "user", 
                "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {question}"
            },
            {
                "role": "assistant",
                "content": f"<think>\n{reasoning}\n"
            }
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            return_tensors="pt",
            continue_final_message=True,
            # add_generation_prompt=True
        )
        # add </think> to the end of the prompt
        prompt = torch.cat([prompt, tokenizer.encode("</think><answer>", add_special_tokens=False, return_tensors="pt")], dim=1)
        
        all_prompts.append(prompt)
        all_samples.append(sample)
    
    # Process all prompts in a single batch
    print("Generating responses in batch...")
    try:
        all_responses = generate_batch_response(model, tokenizer, all_prompts)
        
        # Process results
        for i, (sample, response) in enumerate(zip(all_samples, all_responses)):
            question = sample["question"]
            
            # Extract answers
            predicted_answer = extract_answer_from_text(response)
            predicted_number = extract_number_from_text(predicted_answer)
            ground_truth_number = extract_number_from_text(sample["deepseek_attempt"])
            
            # Check if correct
            is_correct = False
            if predicted_number is not None and ground_truth_number is not None:
                is_correct = abs(predicted_number - ground_truth_number) < 1e-6
            
            if is_correct:
                correct_count += 1
            
            results.append({
                "sample_idx": i,
                "question": question[:100] + "..." if len(question) > 100 else question,
                "predicted_answer": predicted_answer,
                "predicted_number": predicted_number,
                "ground_truth_number": ground_truth_number,
                "is_correct": is_correct,
                "response": response[:200] + "..." if len(response) > 200 else response
            })
            
    except Exception as e:
        import traceback
        print(f"Error in batch processing: {e}")
        traceback.print_exc()
        # Fallback to individual processing if batch fails
        print("Falling back to individual processing...")
        for i in range(total_samples):
            sample = all_samples[i]
            question = sample["question"]
            results.append({
                "sample_idx": i,
                "question": question[:100] + "..." if len(question) > 100 else question,
                "predicted_answer": None,
                "predicted_number": None,
                "ground_truth_number": extract_number_from_text(sample["deepseek_attempt"]),
                "is_correct": False,
                "error": str(e)
            })
    
    accuracy = correct_count / total_samples if total_samples > 0 else 0
    return results, accuracy, correct_count, total_samples


def main():
    parser = argparse.ArgumentParser(description="Test s1K dataset accuracy")
    parser.add_argument("--model_name", type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                       help="Model name or path")
    parser.add_argument("--dataset_name", type=str, 
                       default="simplescaling/s1K-1.1",
                       help="Dataset name")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of samples to test")
    parser.add_argument("--output_file", type=str, default="test_results.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu) - vLLM handles placement automatically")
    
    args = parser.parse_args()
    
    # Load model and tokenizer (vLLM handles device placement automatically)
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    
    # Evaluate
    results, accuracy, correct_count, total_samples = evaluate_dataset(
        model, tokenizer, dataset, args.max_samples
    )
    
    # Print results
    print(f"\nResults:")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Save results
    output_data = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "total_samples": total_samples,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "results": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
