#!/usr/bin/env python3
"""
Real GSM8K evaluation - replace simulation with actual model inference
"""

import torch
import re
from data_utils import GSM8KDataset

def evaluate_gsm8k_real(model, tokenizer, test_dataset, device, max_new_tokens=64, num_samples=50):
    """Real GSM8K evaluation with actual model generation"""
    model.eval()
    correct = 0
    total = 0
    
    print(f"Running real GSM8K evaluation on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset.data))):
            question = test_dataset.data[i]['question']
            true_answer = test_dataset.extract_answer(test_dataset.data[i]['answer'])
            
            # Create prompt
            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=400).to(device)
            
            try:
                # Generate answer
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode generated part only
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                # Extract numerical answer
                pred_answer = extract_numerical_answer(generated_text)
                
                if pred_answer == true_answer:
                    correct += 1
                    
                total += 1
                
                if i % 10 == 0:
                    print(f"Evaluated {i+1}/{num_samples}, Accuracy: {correct/total:.3f}")
                    if i < 3:  # Show first few examples
                        print(f"  Q: {question[:60]}...")
                        print(f"  Generated: {generated_text[:80]}...")
                        print(f"  Pred: {pred_answer}, True: {true_answer}, Correct: {pred_answer == true_answer}")
                        
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                total += 1
                continue
    
    accuracy = correct / total if total > 0 else 0
    print(f"Final accuracy: {accuracy:.3f} ({correct}/{total})")
    return accuracy

def extract_numerical_answer(text: str) -> str:
    """Extract numerical answer from generated text"""
    text = text.strip().lower()
    
    # Patterns to find numerical answers (including LaTeX)
    patterns = [
        r'\\boxed\{([0-9,]+)\}',  # LaTeX boxed answers
        r'(?:the answer is|answer is|answer:|final answer:)\s*([0-9,]+)',
        r'####\s*([0-9,]+)',
        r'\$([0-9,]+)',
        r'([0-9,]+)\s*(?:dollars?|cents?)',
        r'therefore[^0-9]*([0-9,]+)',
        r'so[^0-9]*([0-9,]+)(?:\s*$|\s*\.)',
        r'([0-9,]+)(?:\s*$|\s*\.)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(',', '')
    
    # Fallback: find the last number
    numbers = re.findall(r'([0-9,]+)', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return "0"

def test_real_evaluation():
    """Test real evaluation on a small sample"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("Testing real evaluation...")
    
    # Load student model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create test dataset
    test_dataset = GSM8KDataset(tokenizer, split='test', max_length=256)
    
    # Run evaluation on 10 samples
    accuracy = evaluate_gsm8k_real(model, tokenizer, test_dataset, model.device, num_samples=10)
    
    print(f"Real evaluation test completed: {accuracy:.3f}")
    return accuracy

if __name__ == "__main__":
    test_real_evaluation()