import torch
import re
from datasets import load_dataset

def extract_numerical_answer(text):
    """Extract numerical answer from generated text"""
    # Remove commas from numbers
    text = text.replace(',', '')
    
    # Priority patterns
    patterns = [
        r'\$\\boxed\{(\d+)\}\$',           # $\boxed{18}$
        r'\\boxed\{(\d+)\}',                # \boxed{18}
        r'####\s*(\d+)',                    # #### 18 (GSM8K format)
        r'(?:answer is|answer:)\s*\$?\s*(\d+)',  # answer is 18
        r'=\s*\$?\s*(\d+)\s*$',            # = 18 at end
        r'(?:total|result).*?(\d+)',       # total is 18
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Last resort: find last number in text
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        return numbers[-1]
    
    return "0"

def evaluate_gsm8k_detailed(model, tokenizer, test_dataset, num_samples=100, show_examples=5):
    """
    Comprehensive evaluation with detailed output
    
    Args:
        model: The trained student model
        tokenizer: Student tokenizer
        test_dataset: GSM8K test dataset
        num_samples: Number of samples to evaluate on
        show_examples: Number of examples to display in detail
    """
    model.eval()
    correct = 0
    results = []
    
    print("\n" + "="*80)
    print(f"EVALUATING DUAL-TEACHER DISTILLED MODEL ON {num_samples} SAMPLES")
    print("="*80 + "\n")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            item = test_dataset.data[i]
            question = item['question']
            full_answer = item['answer']
            correct_answer = extract_numerical_answer(full_answer)
            
            # Create prompt (same format as training)
            prompt = f"Question: {question}\n\nAnswer: "
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=400,
                truncation=True
            ).to(model.device)
            
            # Generate answer
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3
            )
            
            # Decode
            input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = full_text[len(input_text):].strip()
            
            # Extract model's numerical answer
            model_answer = extract_numerical_answer(generated_part)
            
            # Check correctness
            is_correct = (model_answer == correct_answer)
            if is_correct:
                correct += 1
            
            # Store result
            results.append({
                'question': question,
                'correct_answer': correct_answer,
                'model_answer': model_answer,
                'generated_text': generated_part,
                'is_correct': is_correct
            })
            
            # Display detailed examples
            if i < show_examples:
                print(f"{'─'*80}")
                print(f"EXAMPLE {i+1}")
                print(f"{'─'*80}")
                print(f"Question: {question}")
                print(f"\nGenerated Answer:")
                print(f"{generated_part[:300]}{'...' if len(generated_part) > 300 else ''}")
                print(f"\nExtracted Model Answer: {model_answer}")
                print(f"Correct Answer: {correct_answer}")
                print(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
                print()
            
            # Progress updates
            if (i + 1) % 25 == 0:
                current_accuracy = correct / (i + 1)
                print(f"Progress: {i+1}/{num_samples} | Accuracy: {current_accuracy:.1%} ({correct}/{i+1})")
    
    # Final results
    final_accuracy = correct / min(num_samples, len(test_dataset))
    
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    print(f"Total Samples: {min(num_samples, len(test_dataset))}")
    print(f"Correct: {correct}")
    print(f"Wrong: {min(num_samples, len(test_dataset)) - correct}")
    print(f"Accuracy: {final_accuracy:.2%} ({final_accuracy:.4f})")
    print("="*80 + "\n")
    
    return final_accuracy, results
    
def analyze_improvements(baseline_accuracy, final_accuracy, model_name="Dual-Teacher Distilled Model"):
    """Analyze and display improvement statistics"""
    
    absolute_improvement = final_accuracy - baseline_accuracy
    relative_improvement = (absolute_improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
    
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"{'─'*80}")
    print(f"Baseline Accuracy:     {baseline_accuracy:.2%}")
    print(f"Final Accuracy:        {final_accuracy:.2%}")
    print(f"{'─'*80}")
    print(f"Absolute Improvement:  {absolute_improvement:+.2%} ({absolute_improvement:+.4f})")
    print(f"Relative Improvement:  {relative_improvement:+.1f}%")
    print(f"{'─'*80}")
    
    # Interpretation
    if absolute_improvement > 0.05:
        print("🎉 EXCELLENT: Significant improvement achieved!")
    elif absolute_improvement > 0.02:
        print("✓ GOOD: Meaningful improvement observed")
    elif absolute_improvement > 0:
        print("✓ MODEST: Small improvement gained")
    else:
        print("⚠️  WARNING: No improvement or decline detected")
    
    print("="*80 + "\n")
    
    return {
        'baseline': float(baseline_accuracy),
        'final': float(final_accuracy),
        'absolute_improvement': float(absolute_improvement),
        'relative_improvement': float(relative_improvement)
    }

def error_analysis(results, show_errors=10):
    """Analyze incorrect predictions"""
    
    incorrect = [r for r in results if not r['is_correct']]
    
    print("\n" + "="*80)
    print(f"ERROR ANALYSIS ({len(incorrect)} errors)")
    print("="*80 + "\n")
    
    if len(incorrect) == 0:
        print("✓ Perfect score - no errors to analyze!\n")
        return
    
    print(f"Showing first {min(show_errors, len(incorrect))} errors:\n")
    
    for i, error in enumerate(incorrect[:show_errors]):
        print(f"{'─'*80}")
        print(f"ERROR {i+1}")
        print(f"{'─'*80}")
        print(f"Question: {error['question'][:150]}...")
        print(f"\nModel Answer: {error['model_answer']}")
        print(f"Correct Answer: {error['correct_answer']}")
        print(f"\nGenerated: {error['generated_text'][:200]}...")
        print()

def run_complete_evaluation(student_model, student_tokenizer, baseline_accuracy=0.42):
    """
    Run complete evaluation pipeline
    """
    
    print("\n" + "="*80)
    print("LOADING TEST DATASET")
    print("="*80)
    
    # Load test dataset
    class GSM8KDataset:
        def __init__(self, tokenizer, split='train', max_length=512):
            self.tokenizer = tokenizer
            self.max_length = max_length
            dataset = load_dataset("gsm8k", "main")
            self.data = dataset[split]
            self.tokenizer.padding_side = 'right'
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        def __len__(self):
            return len(self.data)
    
    test_dataset = GSM8KDataset(student_tokenizer, split='test')
    print(f"✓ Loaded {len(test_dataset)} test samples\n")
    
    # Run evaluation
    final_accuracy, results = evaluate_gsm8k_detailed(
        student_model, 
        student_tokenizer, 
        test_dataset, 
        num_samples=100,
        show_examples=5
    )
    
    # Analyze improvements
    improvement_stats = analyze_improvements(baseline_accuracy, final_accuracy)
    
    # Error analysis
    error_analysis(results, show_errors=5)
    
    # Save results
    from datetime import datetime
    import json
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'evaluation_results_{timestamp}.json'
    
    save_data = {
        'timestamp': timestamp,
        'statistics': improvement_stats,
        'num_samples': len(results),
        'correct': sum(1 for r in results if r['is_correct']),
        'accuracy': float(final_accuracy),
        'sample_results': [
            {
                'question': r['question'],
                'correct_answer': r['correct_answer'],
                'model_answer': r['model_answer'],
                'is_correct': r['is_correct']
            }
            for r in results[:20]  # Save first 20 examples
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}\n")
    
    return final_accuracy, results, improvement_stats


if __name__ == "__main__":
    # Assuming student_model and student_tokenizer are already loaded from training
    # If running separately, uncomment below:
    
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # import os
    # 
    # os.environ['HF_TOKEN'] = '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'
    # 
    # student_tokenizer = AutoTokenizer.from_pretrained(
    #     "google/gemma-2-2b-it",
    #     token=os.environ['HF_TOKEN']
    # )
    # student_model = AutoModelForCausalLM.from_pretrained(
    #     "./dual_teacher_fp16/checkpoint-XXX",  # Your checkpoint path
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )
    
    # Run complete evaluation
    final_accuracy, results, stats = run_complete_evaluation(
        student_model, 
        student_tokenizer, 
        baseline_accuracy=0.42
    )
    
    print("\n🎯 EVALUATION COMPLETE!")
