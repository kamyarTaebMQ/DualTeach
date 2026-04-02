"""
DualTeach Evaluation Script
============================
Evaluates a trained student model on GSM8K test set.

Features:
  - GSM8K exact-match accuracy
  - Inference speed measurement (tokens/sec, latency)
  - McNemar's test for statistical significance
  - Multi-seed evaluation for mean ± std reporting

Usage:
    # Single model evaluation
    python evaluate.py --checkpoint checkpoints/student/epoch_3

    # Compare multiple checkpoints
    python evaluate.py --compare \
        --checkpoints checkpoints/student/epoch_1 checkpoints/student/epoch_2 checkpoints/student/epoch_3

    # Speed measurement only
    python evaluate.py --checkpoint checkpoints/student/epoch_3 --speed_only

    # Multi-seed training + evaluation (requires train.py to be runnable)
    python evaluate.py --checkpoint checkpoints/student/epoch_3 --multi_seed
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

# ── GPU selection ─────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from scipy.stats import chi2_contingency
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  Answer extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_answer(text: str) -> str | None:
    """
    Extract the final numeric answer from model output or GSM8K reference.

    Tries the '####' pattern first (standard GSM8K format), then falls back
    to the last number in the text.
    """
    # GSM8K canonical format: "#### 42"
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last number
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None


# ══════════════════════════════════════════════════════════════════════════════
#  Model utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str):
    """Load tokenizer + model from a HuggingFace checkpoint directory onto GPU 0."""
    print(f"Loading model from {checkpoint_path} → GPU 0 ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.bfloat16,
        device_map={"": 0},   # all layers → cuda:0
    )
    model.eval()
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
#  GSM8K accuracy evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_gsm8k(
    model,
    tokenizer,
    test_dataset,
    max_new_tokens: int = 256,
    return_predictions: bool = False,
):
    """
    Evaluate model on GSM8K test set using exact-match accuracy.

    Args:
        model:            HuggingFace causal LM (eval mode)
        tokenizer:        matching tokenizer
        test_dataset:     list of {"question": ..., "answer": ...} dicts
        max_new_tokens:   maximum tokens to generate per sample
        return_predictions: if True, also return per-sample bool list

    Returns:
        accuracy (float), [predictions] (list[bool]) if return_predictions=True
    """
    device     = next(model.parameters()).device
    correct    = 0
    total      = len(test_dataset)
    predictions = []

    with torch.no_grad():
        for sample in tqdm(test_dataset, desc="Evaluating"):
            prompt  = f"Problem: {sample['question']}\nAnswer:"
            inputs  = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            pred = extract_answer(generated)
            gold = extract_answer(sample["answer"])

            is_correct = bool(pred and gold and pred.strip() == gold.strip())
            predictions.append(is_correct)
            if is_correct:
                correct += 1

    accuracy = correct / total
    print(f"\nGSM8K Accuracy: {accuracy:.4f}  ({correct}/{total})")

    if return_predictions:
        return accuracy, predictions
    return accuracy


# ══════════════════════════════════════════════════════════════════════════════
#  Inference speed measurement
# ══════════════════════════════════════════════════════════════════════════════

def measure_inference_speed(
    model,
    tokenizer,
    num_runs: int = 50,
    max_new_tokens: int = 64,
):
    """
    Measure inference speed (latency + tokens/sec).

    Args:
        model:          HuggingFace causal LM (eval mode)
        tokenizer:      matching tokenizer
        num_runs:       number of timed runs after warmup
        max_new_tokens: tokens generated per run

    Returns:
        dict with mean_ms, std_ms, tokens_per_sec
    """
    device = next(model.parameters()).device

    test_input = (
        "Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast "
        "every morning and bakes muffins for her friends every day with 4. "
        "She sells the remainder at the farmers' market daily for $2 per fresh "
        "duck egg. How much in dollars does she make every day at the farmers' market?\n"
        "Answer:"
    )
    inputs = tokenizer(test_input, return_tensors="pt").to(device)

    latencies = []
    model.eval()
    num_output_tokens = None

    with torch.no_grad():
        # Warmup
        for _ in range(5):
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Timed runs
        for _ in tqdm(range(num_runs), desc="Speed test"):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
            num_output_tokens = out.shape[1] - inputs["input_ids"].shape[1]

    mean_ms      = float(np.mean(latencies))
    std_ms       = float(np.std(latencies))
    tokens_per_s = num_output_tokens / (mean_ms / 1000.0)

    print(f"\n── Inference Speed ────────────────────────────────")
    print(f"  Mean latency:  {mean_ms:.1f} ms")
    print(f"  Std latency:   {std_ms:.1f} ms")
    print(f"  Tokens/sec:    {tokens_per_s:.1f}")
    print(f"  Output tokens: {num_output_tokens}")

    return {
        "mean_ms":       round(mean_ms, 1),
        "std_ms":        round(std_ms, 1),
        "tokens_per_sec": round(tokens_per_s, 1),
        "num_output_tokens": num_output_tokens,
        "num_runs":       num_runs,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Statistical significance — McNemar's test
# ══════════════════════════════════════════════════════════════════════════════

def mcnemar_test(predictions_a: list[bool], predictions_b: list[bool]) -> dict:
    """
    McNemar's test comparing two models on the same test set.

    Args:
        predictions_a: per-sample correct/incorrect (bool) for model A
        predictions_b: per-sample correct/incorrect (bool) for model B

    Returns:
        dict with chi2, p_value, significant (p < 0.01), b, c counts
    """
    assert len(predictions_a) == len(predictions_b), "Prediction lists must have equal length."

    # b: A correct, B wrong; c: A wrong, B correct
    b = sum(1 for a, bb in zip(predictions_a, predictions_b) if a and not bb)
    c = sum(1 for a, bb in zip(predictions_a, predictions_b) if not a and bb)

    # Continuity-corrected McNemar statistic
    if b + c == 0:
        chi2_stat = 0.0
        p_value   = 1.0
    else:
        chi2_stat = (abs(b - c) - 1.0) ** 2 / (b + c)
        from scipy.stats import chi2
        p_value = 1.0 - chi2.cdf(chi2_stat, df=1)

    significant = p_value < 0.01

    print(f"\nMcNemar's Test:")
    print(f"  b (A✓ B✗): {b},  c (A✗ B✓): {c}")
    print(f"  χ² = {chi2_stat:.4f},  p = {p_value:.6f}")
    print(f"  Significant at p<0.01: {significant}")

    return {
        "chi2":        round(chi2_stat, 4),
        "p_value":     round(p_value, 6),
        "significant": significant,
        "b":           b,
        "c":           c,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-seed evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_multiple_seeds(
    checkpoint_dir: str,
    test_dataset,
    seeds: list[int] = None,
    max_new_tokens: int = 256,
):
    """
    Evaluate the same checkpoint multiple times with different random seeds
    to report mean ± std accuracy.

    Note: For true multi-seed results, train.py should be run once per seed
    and checkpoints stored separately. This function evaluates a *single*
    checkpoint with different generation seeds (stochastic if do_sample=True,
    but since we use greedy decoding the numbers will be identical). It is
    kept here as a scaffold for the full multi-seed workflow.

    For proper ablations, call train.py with --seed and separate output dirs,
    then call evaluate.py on each and aggregate with this function.
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    model, tokenizer = load_model(checkpoint_dir)
    accuracies = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n[Seed {seed}] Evaluating...")
        acc = evaluate_gsm8k(model, tokenizer, test_dataset,
                             max_new_tokens=max_new_tokens)
        accuracies.append(acc)

    mean_acc = float(np.mean(accuracies))
    std_acc  = float(np.std(accuracies))

    print(f"\nMulti-seed results ({seeds}):")
    print(f"  Accuracies: {[f'{a:.4f}' for a in accuracies]}")
    print(f"  Mean: {mean_acc:.4f}  Std: {std_acc:.4f}")
    print(f"  → {mean_acc*100:.1f}% ± {std_acc*100:.1f}%  (target: 52.2% ± 0.5%)")

    return {"seeds": seeds, "accuracies": accuracies, "mean": mean_acc, "std": std_acc}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DualTeach Evaluation Script")

    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to HuggingFace checkpoint directory.")
    p.add_argument("--compare", action="store_true",
                   help="Compare multiple checkpoints (list via --checkpoints).")
    p.add_argument("--checkpoints", nargs="+", default=[],
                   help="Checkpoint paths when using --compare.")
    p.add_argument("--speed_only", action="store_true",
                   help="Only run inference speed test, skip accuracy eval.")
    p.add_argument("--multi_seed", action="store_true",
                   help="Evaluate with 5 seeds and report mean ± std.")
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max generation tokens (default: 256).")
    p.add_argument("--speed_runs", type=int, default=50,
                   help="Number of speed measurement runs (default: 50).")
    p.add_argument("--output", type=str, default="results/accuracy_results.json",
                   help="JSON file to save evaluation results.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Evaluate on first N samples only (quick sanity check).")

    return p.parse_args()


def main():
    args = parse_args()

    print("Loading GSM8K test set...")
    dataset     = load_dataset("openai/gsm8k", "main")
    test_data   = list(dataset["test"])
    if args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"  {len(test_data)} samples loaded (subset of {len(list(dataset['test']))} total).")
    else:
        print(f"  {len(test_data)} test samples loaded.")

    results = {}

    # ── Single checkpoint mode ─────────────────────────────────────────────
    if args.checkpoint and not args.compare:
        model, tokenizer = load_model(args.checkpoint)

        if args.speed_only:
            speed = measure_inference_speed(model, tokenizer,
                                            num_runs=args.speed_runs)
            results["speed"] = speed
        elif args.multi_seed:
            ms = run_multiple_seeds(args.checkpoint, test_data,
                                    max_new_tokens=args.max_new_tokens)
            results["multi_seed"] = ms
        else:
            acc, preds = evaluate_gsm8k(
                model, tokenizer, test_data,
                max_new_tokens=args.max_new_tokens,
                return_predictions=True,
            )
            speed = measure_inference_speed(model, tokenizer,
                                            num_runs=args.speed_runs)
            results[args.checkpoint] = {
                "accuracy": acc,
                "speed":    speed,
            }

            # Save per-sample predictions for later McNemar tests
            pred_path = Path("results") / "predictions_dualteach.json"
            with open(pred_path, "w") as f:
                json.dump(preds, f)
            print(f"Predictions saved → {pred_path}")

    # ── Comparison mode ────────────────────────────────────────────────────
    elif args.compare and args.checkpoints:
        all_preds  = {}
        all_accs   = {}

        for ckpt in args.checkpoints:
            model, tokenizer = load_model(ckpt)
            acc, preds = evaluate_gsm8k(
                model, tokenizer, test_data,
                max_new_tokens=args.max_new_tokens,
                return_predictions=True,
            )
            all_preds[ckpt] = preds
            all_accs[ckpt]  = acc
            del model  # free VRAM

        # McNemar tests vs first checkpoint (assumed baseline)
        ckpt_names = args.checkpoints
        baseline   = ckpt_names[0]
        sig_results = {}
        for ckpt in ckpt_names[1:]:
            print(f"\nMcNemar: {ckpt} vs {baseline}")
            sig = mcnemar_test(all_preds[ckpt], all_preds[baseline])
            sig_results[f"{ckpt}_vs_{baseline}"] = sig

        results = {
            "accuracies":   all_accs,
            "significance": sig_results,
        }

    else:
        print("ERROR: Provide --checkpoint or --compare --checkpoints.")
        return

    # ── Save results ───────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
