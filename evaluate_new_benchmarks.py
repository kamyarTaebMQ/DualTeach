"""
evaluate_new_benchmarks.py
===========================
Zero-shot evaluation of DualTeach and baseline checkpoints on MATH and AQuA-RAT.

Consistent with GSM8K evaluation style (evaluate.py):
  - 0-shot prompting
  - bfloat16
  - padding_side='right'
  - greedy decoding

Resumable: per-sample results saved to results/progress/<benchmark>_<model>.jsonl
so interrupted runs continue from where they left off.

Usage:
    # Evaluate one model on one benchmark (recommended for parallel runs)
    python evaluate_new_benchmarks.py --model dualteach --benchmark math
    python evaluate_new_benchmarks.py --model baseline  --benchmark aqua

    # Evaluate all models on all benchmarks (sequentially)
    python evaluate_new_benchmarks.py

    # After parallel runs finish, collect results into final JSON
    python evaluate_new_benchmarks.py --collect_only
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINTS = {
    "dualteach":         "checkpoints/student/epoch_5",
    "dualteach_aqua":    "checkpoints/aqua_dualteach/epoch_3",
    "dualteach_aqua_a03":"checkpoints/aqua_dualteach_a03/epoch_3",
    "aqua_single_t1":    "checkpoints/aqua_single_t1/epoch_3",
    "aqua_single_t2":    "checkpoints/aqua_single_t2/epoch_3",
    "math_dualteach":    "checkpoints/math_dualteach/epoch_3",
    "math_single_t1":    "checkpoints/math_single_t1/epoch_3",
    "math_single_t2":    "checkpoints/math_single_t2/epoch_3",
    "single_t1":         "checkpoints/single_t1/epoch_3",
    "single_t2":         "checkpoints/single_t2/epoch_3",
    "baseline":          "checkpoints/baseline/epoch_5",
    "base":              "google/gemma-3-1b-it",
}

MODEL_NAMES = {
    "dualteach":         "DualTeach (GSM8K only)",
    "dualteach_aqua":    "DualTeach (AQuA-trained, α=0.7)",
    "dualteach_aqua_a03":"DualTeach (AQuA-trained, α=0.3)",
    "aqua_single_t1":    "Single-T1 (AQuA-trained, 12B teacher)",
    "aqua_single_t2":    "Single-T2 (AQuA-trained, 4B teacher)",
    "math_dualteach":    "DualTeach (MATH-trained)",
    "math_single_t1":    "Single-T1 (MATH-trained, 12B teacher)",
    "math_single_t2":    "Single-T2 (MATH-trained, 4B teacher)",
    "single_t1":         "Single-T1 (GSM8K only, 12B teacher)",
    "single_t2":         "Single-T2 (GSM8K only, 4B teacher)",
    "baseline":          "Baseline fine-tune (GSM8K only)",
    "base":              "Gemma-3-1B-IT (no fine-tuning)",
}

# MATH needs more tokens for multi-step LaTeX solutions
MAX_NEW_TOKENS = {
    "math": 512,
    "aqua": 128,
}

RESULTS_DIR  = Path("results")
PROGRESS_DIR = RESULTS_DIR / "progress"
FINAL_JSON   = RESULTS_DIR / "new_benchmarks.json"


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def load_math_dataset():
    """
    Load MATH benchmark (EleutherAI/hendrycks_math, test split, ~5000 problems).

    The dataset is split by subject; we load all 7 configs and concatenate.
    Each sample fields used:
      problem  (str): the math problem text
      solution (str): full solution; ground truth is inside \\boxed{} at the end
      level    (str): difficulty "Level 1" – "Level 5"
      type     (str): subject category
    """
    MATH_CONFIGS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    print("Loading MATH dataset (EleutherAI/hendrycks_math, test split) ...")
    all_samples = []
    for cfg in MATH_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
        for s in ds:
            all_samples.append({
                "id":       len(all_samples),
                "problem":  s["problem"],
                "solution": s["solution"],
                "level":    s["level"],
                "type":     cfg,
            })
    print(f"  {len(all_samples)} problems loaded across {len(MATH_CONFIGS)} subjects.")
    return all_samples


def load_aqua_dataset():
    """
    Load AQuA-RAT dataset (aqua_rat, test split, ~254 samples).

    Each sample fields used:
      question (str):       the problem text
      options  (list[str]): e.g. ["A)12", "B)24", "C)36", "D)48", "E)60"]
      correct  (str):       answer letter, e.g. "B"

    Assumption: options are already prefixed with letters (A), B), …).
    If they are not, make_aqua_prompt() handles that case.
    """
    print("Loading AQuA-RAT dataset (aqua_rat, test split) ...")
    ds = load_dataset("aqua_rat", "raw", split="test")
    samples = [
        {"id": i, "question": s["question"], "options": s["options"],
         "correct": s["correct"].strip().upper()}
        for i, s in enumerate(ds)
    ]
    print(f"  {len(samples)} problems loaded.")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
#  Prompt construction
# ══════════════════════════════════════════════════════════════════════════════

def make_math_prompt(problem: str) -> str:
    """
    3-shot MATH prompt.
    Examples cover algebra, number theory, and geometry to show the model
    that the final answer must always be inside \\boxed{}.
    """
    return (
        "Solve the following math problem step by step. "
        "Always put your final answer inside \\boxed{}.\n\n"
        "Problem: What is the value of $x$ if $2x + 3 = 11$?\n"
        "Solution: Subtract 3 from both sides: $2x = 8$. "
        "Divide by 2: $x = 4$. "
        "The answer is $\\boxed{4}$.\n\n"
        "Problem: Find the sum of all positive divisors of 12.\n"
        "Solution: The divisors of 12 are 1, 2, 3, 4, 6, 12. "
        "Their sum is $1+2+3+4+6+12 = 28$. "
        "The answer is $\\boxed{28}$.\n\n"
        "Problem: A right triangle has legs of length 3 and 4. "
        "What is the length of the hypotenuse?\n"
        "Solution: By the Pythagorean theorem, $c^2 = 3^2 + 4^2 = 9 + 16 = 25$, "
        "so $c = 5$. "
        "The answer is $\\boxed{5}$.\n\n"
        f"Problem: {problem}\n"
        "Solution:"
    )


def make_aqua_prompt(question: str, options: list) -> str:
    """
    0-shot AQuA-RAT prompt.
    Options come as e.g. ["A)12", "B)24", …]; format them one per line.
    If options are bare values without letter prefixes, we add them.
    Instructs the model to output a single letter A-E.
    """
    letters = "ABCDE"
    lines = []
    for i, opt in enumerate(options):
        opt = opt.strip()
        # Already prefixed: "A)..." or "A) ..."
        if opt and opt[0].upper() in letters and len(opt) > 1 and opt[1] in ")":
            lines.append(opt)
        else:
            lines.append(f"{letters[i]}) {opt}")

    return (
        "Solve the following multiple-choice math problem. "
        "Think step by step, then give your final answer as a single letter (A, B, C, D, or E).\n\n"
        f"Question: {question}\n"
        "Options:\n" + "\n".join(lines) + "\n"
        "Answer:"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Answer extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_boxed(text: str) -> str | None:
    r"""
    Extract content of the last \boxed{...} in text, handling nested braces.

    Why "last": models sometimes show intermediate boxed steps, but the
    final answer is the last \boxed{} in the output.

    Example: "\boxed{\frac{3}{4}}"  →  "\frac{3}{4}"
    Example: "\boxed{42}"           →  "42"
    Returns None if no \boxed{ found.
    """
    # Find all \boxed{ positions; take the last one
    positions = [m.start() for m in re.finditer(r'\\boxed\s*\{', text)]
    if not positions:
        return None

    idx = positions[-1]
    start = text.find("{", idx) + 1
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1

    if depth != 0:
        return None  # unclosed brace — malformed output

    return text[start:pos - 1].strip()


def extract_math_answer(text: str) -> str | None:
    r"""
    Extract the final answer from a MATH model output using multiple strategies,
    in priority order:

    1. \boxed{...}        — standard MATH format (base model, MATH-trained models)
    2. #### <answer>      — GSM8K format (GSM8K-trained models output this)
    3. "The answer is X"  — explicit statement
    4. Last number in text — numeric fallback for simple numeric answers

    Takes the FIRST #### occurrence (models trained on GSM8K repeat it many times).
    """
    text = text.strip()

    # 1. \boxed{} — standard MATH format
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed

    # 2. #### answer — GSM8K format; take the FIRST occurrence
    m = re.search(r'####\s*(.+)', text)
    if m:
        return m.group(1).strip()

    # 3. "The answer is X" / "answer: X"
    m = re.search(
        r'(?:the\s+)?answer(?:\s+is)?[:\s]+([^\n\.]+)',
        text, re.IGNORECASE
    )
    if m:
        val = m.group(1).strip().rstrip('.')
        if val:
            return val

    # 4. Last number on the last non-empty line (numeric fallback)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        nums = re.findall(r'-?\d+(?:[./]\d+)?', lines[-1])
        if nums:
            return nums[-1]

    return None


def normalize_math(answer: str | None) -> str | None:
    """
    Light normalization before comparing MATH answers:
      - strip whitespace
      - remove trailing ".0" (e.g. "3.0" → "3")
      - lowercase (for symbolic answers like "\\frac{1}{2}")
      - remove spaces inside fractions for robustness

    We deliberately stay light here — heavy normalization (sympy equivalence)
    is outside scope; exact string match after normalization is standard
    in distillation ablation papers.
    """
    if answer is None:
        return None
    s = answer.strip()
    # Integer-valued float: "3.0" → "3"
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    # Remove spaces within LaTeX commands
    s = re.sub(r"\s+", "", s)
    return s.lower()


def extract_aqua_answer(text: str) -> str | None:
    """
    Extract final letter answer (A-E) from model output for AQuA-RAT.

    Priority order:
    1. "The answer is X" / "answer: X" pattern (most explicit)
    2. Last standalone letter A-E in the text
    3. None if not found

    Why last letter: models often say "...so the answer is B" at the end.
    """
    text = text.strip()

    # Pattern 1: explicit answer statement
    m = re.search(
        r'(?:the\s+)?answer(?:\s+is)?[:\s]+\(?([A-Ea-e])\)?',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    # Pattern 2: "Therefore, B" or "Thus B"
    m = re.search(r'(?:therefore|thus|so)[,\s]+\(?([A-Ea-e])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 3: last standalone letter A-E (word boundary)
    matches = re.findall(r'\b([A-Ea-e])\b', text)
    if matches:
        return matches[-1].upper()

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str):
    """
    Load full fine-tuned model (no LoRA / PEFT — just standard from_pretrained).

    Notes:
    - bfloat16 for H200 efficiency
    - device_map='auto' spreads across GPUs if needed (1B fits on one)
    - padding_side='right' per project requirement; we switch to 'left'
      temporarily during batch generation (causal LMs need left-padding
      so all sequences are right-aligned for generation)
    """
    print(f"Loading '{checkpoint_path}' ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram = round(torch.cuda.memory_allocated() / 1e9, 2)
        print(f"  Loaded. VRAM: {vram} GB")

    return tokenizer, model


# ══════════════════════════════════════════════════════════════════════════════
#  Batch inference
# ══════════════════════════════════════════════════════════════════════════════

def generate_batch(tokenizer, model, prompts: list, max_new_tokens: int,
                   batch_size: int = 8) -> list:
    """
    Generate text for a list of prompts using batched inference.
    Returns list of generated strings (prompt NOT included).

    Temporarily sets padding_side='left' for generation — HuggingFace
    requires left-padding for batched causal generation so all sequences
    end at the same position for the model to continue from.
    """
    tokenizer.padding_side = "left"
    outputs_text = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy — same as GSM8K eval
                pad_token_id=tokenizer.eos_token_id,
            )

        # Slice off input tokens; decode only the new generation
        input_len = inputs["input_ids"].shape[1]
        for out in out_ids:
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            outputs_text.append(decoded.strip())

    tokenizer.padding_side = "right"   # restore
    return outputs_text


# ══════════════════════════════════════════════════════════════════════════════
#  Progress file helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_progress(progress_file: Path) -> dict:
    """Load already-completed sample records keyed by sample id."""
    completed = {}
    if progress_file.exists():
        with open(progress_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    completed[rec["id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


# ══════════════════════════════════════════════════════════════════════════════
#  Per-benchmark evaluation
# ══════════════════════════════════════════════════════════════════════════════

def eval_math(model_key: str, tokenizer, model, samples: list,
              batch_size: int = 8) -> dict:
    """Evaluate on MATH dataset. Saves per-sample progress for resumability."""
    progress_file = PROGRESS_DIR / f"math_{model_key}.jsonl"
    completed = load_progress(progress_file)
    if completed:
        print(f"  Resuming: {len(completed)}/{len(samples)} already done.")

    remaining = [s for s in samples if s["id"] not in completed]
    prompts   = [make_math_prompt(s["problem"]) for s in remaining]

    pbar = tqdm(total=len(remaining), desc=f"MATH [{model_key}]")
    with open(progress_file, "a") as f:
        for i in range(0, len(remaining), batch_size):
            batch_samples = remaining[i: i + batch_size]
            batch_prompts = prompts[i: i + batch_size]
            outputs = generate_batch(tokenizer, model, batch_prompts,
                                     MAX_NEW_TOKENS["math"], batch_size)

            for sample, output in zip(batch_samples, outputs):
                pred_raw  = extract_math_answer(output)
                pred_norm = normalize_math(pred_raw)
                gold_raw  = extract_boxed(sample["solution"])
                gold_norm = normalize_math(gold_raw)
                correct   = bool(pred_norm and gold_norm and pred_norm == gold_norm)

                rec = {
                    "id":        sample["id"],
                    "level":     sample["level"],
                    "type":      sample["type"],
                    "correct":   correct,
                    "pred_raw":  pred_raw,
                    "gold_raw":  gold_raw,
                }
                completed[sample["id"]] = rec
                f.write(json.dumps(rec) + "\n")
                f.flush()

            pbar.update(len(batch_samples))
    pbar.close()

    all_recs  = list(completed.values())
    n_correct = sum(1 for r in all_recs if r["correct"])
    accuracy  = n_correct / len(all_recs) if all_recs else 0.0
    print(f"  MATH accuracy: {accuracy:.4f}  ({n_correct}/{len(all_recs)})")
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": len(all_recs)}


def eval_aqua(model_key: str, tokenizer, model, samples: list,
              batch_size: int = 8) -> dict:
    """Evaluate on AQuA-RAT dataset. Saves per-sample progress for resumability."""
    progress_file = PROGRESS_DIR / f"aqua_{model_key}.jsonl"
    completed = load_progress(progress_file)
    if completed:
        print(f"  Resuming: {len(completed)}/{len(samples)} already done.")

    remaining = [s for s in samples if s["id"] not in completed]
    prompts   = [make_aqua_prompt(s["question"], s["options"]) for s in remaining]

    pbar = tqdm(total=len(remaining), desc=f"AQuA [{model_key}]")
    with open(progress_file, "a") as f:
        for i in range(0, len(remaining), batch_size):
            batch_samples = remaining[i: i + batch_size]
            batch_prompts = prompts[i: i + batch_size]
            outputs = generate_batch(tokenizer, model, batch_prompts,
                                     MAX_NEW_TOKENS["aqua"], batch_size)

            for sample, output in zip(batch_samples, outputs):
                pred    = extract_aqua_answer(output)
                correct = (pred == sample["correct"]) if pred else False

                rec = {
                    "id":      sample["id"],
                    "correct": correct,
                    "pred":    pred,
                    "gold":    sample["correct"],
                }
                completed[sample["id"]] = rec
                f.write(json.dumps(rec) + "\n")
                f.flush()

            pbar.update(len(batch_samples))
    pbar.close()

    all_recs  = list(completed.values())
    n_correct = sum(1 for r in all_recs if r["correct"])
    accuracy  = n_correct / len(all_recs) if all_recs else 0.0
    print(f"  AQuA accuracy: {accuracy:.4f}  ({n_correct}/{len(all_recs)})")
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": len(all_recs)}


# ══════════════════════════════════════════════════════════════════════════════
#  Collect results from progress files
# ══════════════════════════════════════════════════════════════════════════════

def collect_results():
    """
    Read all progress JSONL files and compile into results/new_benchmarks.json.
    Run this after parallel jobs finish to merge results.
    """
    results = {}
    for model_key in CHECKPOINTS:
        results[model_key] = {"model_name": MODEL_NAMES[model_key]}
        for benchmark in ["math", "aqua"]:
            pf = PROGRESS_DIR / f"{benchmark}_{model_key}.jsonl"
            if not pf.exists():
                continue
            recs = []
            with open(pf) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            recs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            if not recs:
                continue
            n_correct = sum(1 for r in recs if r["correct"])
            acc = n_correct / len(recs)
            results[model_key][benchmark] = {
                "accuracy":  round(acc, 4),
                "n_correct": n_correct,
                "n_total":   len(recs),
            }
            print(f"  {model_key:12s}  {benchmark:5s}  {n_correct}/{len(recs)} = {acc:.4f}")

    FINAL_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {FINAL_JSON}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation on MATH and AQuA-RAT"
    )
    parser.add_argument(
        "--model",
        choices=list(CHECKPOINTS.keys()) + ["all"],
        default="all",
        help="Model to evaluate (default: all, sequentially).",
    )
    parser.add_argument(
        "--benchmark",
        choices=["math", "aqua", "all"],
        default="all",
        help="Benchmark to run (default: all).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Generation batch size. Try 16 on H200 for speed (default: 8).",
    )
    parser.add_argument(
        "--collect_only",
        action="store_true",
        help="Skip evaluation; collect progress files into final JSON only.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    if args.collect_only:
        collect_results()
        return

    models_to_run     = list(CHECKPOINTS.keys()) if args.model == "all" else [args.model]
    benchmarks_to_run = ["math", "aqua"] if args.benchmark == "all" else [args.benchmark]

    # Load datasets upfront (only what's needed)
    math_samples = load_math_dataset() if "math" in benchmarks_to_run else None
    aqua_samples = load_aqua_dataset() if "aqua" in benchmarks_to_run else None

    all_results = {}

    for model_key in models_to_run:
        ckpt = CHECKPOINTS[model_key]
        print(f"\n{'='*60}")
        print(f"Model : {MODEL_NAMES[model_key]}")
        print(f"Ckpt  : {ckpt}")
        print(f"{'='*60}")

        tokenizer, model = load_model(ckpt)
        all_results[model_key] = {"model_name": MODEL_NAMES[model_key]}

        if "math" in benchmarks_to_run:
            print("\n--- MATH ---")
            r = eval_math(model_key, tokenizer, model, math_samples, args.batch_size)
            all_results[model_key]["math"] = r

        if "aqua" in benchmarks_to_run:
            print("\n--- AQuA-RAT ---")
            r = eval_aqua(model_key, tokenizer, model, aqua_samples, args.batch_size)
            all_results[model_key]["aqua"] = r

        # Free GPU memory before loading next model
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  GPU memory released.")

    # Merge with any existing results (from parallel jobs) and save
    existing = {}
    if FINAL_JSON.exists():
        try:
            existing = json.loads(FINAL_JSON.read_text())
        except json.JSONDecodeError:
            pass
    for k, v in all_results.items():
        existing.setdefault(k, {}).update(v)
    FINAL_JSON.write_text(json.dumps(existing, indent=2))
    print(f"\nResults saved → {FINAL_JSON}")


if __name__ == "__main__":
    main()
